# app/ui/actions/target_media_actions.py
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from app.helpers.thumbnail_manager import (
    DiscoveredMediaItem,
    MediaScannerWorker,
    ThumbnailManager,
    get_file_type,
    iter_media_files,
)
from app.ui.core.event_bus import (
    AppEventBus,
    TargetFilesAddedEvent,
    TargetFolderChangedEvent,
    TargetMediaFilterChangedEvent,
    TargetMediaSelectedEvent,
    get_event_bus,
)
from app.ui.widgets.card_manager import CardLayoutMode, CardManager, CardSizePreset
from app.ui.widgets.panels.target_media_panel import TargetMediaPanel
from app.ui.widgets.sortable_widgets import (
    DRAG_DROP_FOLDER_KEY,
    DRAG_DROP_FOLDER_LABEL,
    FolderHeaderListWidgetItem,
    MediaMetadata,
    MediaType,
    SortableListWidget,
    SortableListWidgetItem,
)

try:
    from send2trash import send2trash
except ImportError:
    send2trash = None  # type: ignore[assignment]


class TargetMediaActions(QtCore.QObject):
    """Controller owning all signal connections, sorting, filtering, Drag & Drop,
    multi-folder retention, right-click context menu, and thumbnail loading for TargetMediaPanel.
    """

    DEFAULT_MAX_DIMENSION: int = 4096

    def __init__(
        self,
        panel: TargetMediaPanel,
        event_bus: Optional[AppEventBus] = None,
        thumbnail_manager: Optional[ThumbnailManager] = None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.panel = panel
        self.bus = event_bus or get_event_bus()
        self.thumbnail_manager = thumbnail_manager or ThumbnailManager()
        self._scanner_worker: Optional[MediaScannerWorker] = None
        self._current_folder_path: Optional[Path] = None
        self._loaded_folder_paths: list[Path] = []
        self._dropped_file_paths: list[Path] = []
        self._scan_is_additive: bool = False
        self._warning_icon: QtGui.QIcon = self._create_warning_triangle_icon()

        # Enable independent collapsible folder groups & custom right-click menu inside the cards list
        self.panel.targetVideosList.setGroupByFolder(True)
        self.panel.targetVideosList.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )

        # Attach reusable CardManager to handle grid geometry, delegate rendering, and size combobox
        self.card_manager = CardManager(
            list_widget=self.panel.targetVideosList,
            size_combobox=self.panel.targetVideosViewSizeComboBox,
            layout_mode=CardLayoutMode.WIDESCREEN_16_9,
            default_preset=CardSizePreset.MEDIUM,
            show_text=True,
            parent=self,
        )

        self._initialize_controls()
        self._connect_signals()

    # =========================================================================
    # INITIALIZATION & SIGNAL WIRING
    # =========================================================================

    @staticmethod
    def _create_warning_triangle_icon() -> QtGui.QIcon:
        """Build a crisp custom '/!\\' warning triangle icon for destructive menu actions."""
        pix = QtGui.QPixmap(18, 18)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        tri = QtGui.QPainterPath()
        tri.moveTo(9.0, 2.0)
        tri.lineTo(16.5, 15.5)
        tri.lineTo(1.5, 15.5)
        tri.closeSubpath()

        painter.setBrush(QtGui.QColor("#e5a50a"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#b58105"), 1.2))
        painter.drawPath(tri)

        font = painter.font()
        font.setPixelSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#1a1a1a"))
        painter.drawText(
            QtCore.QRect(0, 4, 18, 14),
            int(Qt.AlignmentFlag.AlignCenter),
            "!",
        )
        painter.end()
        return QtGui.QIcon(pix)

    def _initialize_controls(self) -> None:
        """Configure combobox items, slider bounds, and default media filters."""
        p = self.panel

        if p.targetVideosSortComboBox.count() == 0:
            SortableListWidget.populate_sort_combobox(p.targetVideosSortComboBox)

        for slider, spinbox in (
            (p.targetVideosWidthSlider, p.targetVideosWidthSpinBox),
            (p.targetVideosHeightSlider, p.targetVideosHeightSpinBox),
        ):
            slider.setRange(0, self.DEFAULT_MAX_DIMENSION)
            slider.setSingleStep(16)
            slider.setPageStep(128)
            spinbox.setRange(0, self.DEFAULT_MAX_DIMENSION)
            spinbox.setSingleStep(16)
            spinbox.setSuffix(" px")

        self._sync_allowed_media_types()

    def _connect_signals(self) -> None:
        """Wire all TargetMediaPanel widgets to actions, SortableListWidget, and EventBus."""
        p = self.panel
        lst = p.targetVideosList

        p.targetVideosFilterMenuButton.toggled.connect(
            p.targetVideosFilterContainer.setVisible
        )

        p.buttonTargetVideosPath.clicked.connect(self.select_target_folder_dialog)
        p.targetVideosScanSubfoldersCheckBox.toggled.connect(self._on_scan_subfolders_toggled)
        p.targetVideosKeepFoldersCheckBox.toggled.connect(
            lambda _: self.bus.workspaceStateDirty.emit()
        )
        p.targetVideosSearchBox.textChanged.connect(lst.setSearchText)
        p.targetVideosSearchResetButton.clicked.connect(p.targetVideosSearchBox.clear)

        p.targetVideosSortComboBox.currentIndexChanged.connect(self._on_sort_mode_changed)
        p.sortTargetVideosDirectionButton.toggled.connect(lst.setSortDescending)

        p.targetVideosFilterImagesCheckBox.toggled.connect(self._on_media_type_filter_changed)
        p.targetVideosFilterVideosCheckBox.toggled.connect(self._on_media_type_filter_changed)
        p.targetVideosFilterWebcamsCheckBox.toggled.connect(self._on_webcam_filter_toggled)

        p.targetVideosWidthSlider.valueChanged.connect(p.targetVideosWidthSpinBox.setValue)
        p.targetVideosWidthSpinBox.valueChanged.connect(p.targetVideosWidthSlider.setValue)
        p.targetVideosHeightSlider.valueChanged.connect(p.targetVideosHeightSpinBox.setValue)
        p.targetVideosHeightSpinBox.valueChanged.connect(p.targetVideosHeightSlider.setValue)

        p.targetVideosWidthSpinBox.valueChanged.connect(self._on_min_resolution_changed)
        p.targetVideosHeightSpinBox.valueChanged.connect(self._on_min_resolution_changed)

        # External Drag & Drop signals from both the list widget and the dock panel
        lst.pathsDropped.connect(self.handle_dropped_paths)
        p.pathsDropped.connect(self.handle_dropped_paths)

        # Right-click context menu on the list widget
        lst.customContextMenuRequested.connect(self._on_context_menu_requested)

        lst.currentItemChanged.connect(self._on_current_item_changed)
        lst.visibleCountChanged.connect(self._on_visible_count_changed)

        self.card_manager.sizePresetChanged.connect(lambda _: self.bus.workspaceStateDirty.emit())
        self.bus.applicationClosing.connect(self.stop_active_scan)

    # =========================================================================
    # FOLDER, FILE & DRAG-AND-DROP LOADING
    # =========================================================================

    @property
    def current_folder_path(self) -> Optional[Path]:
        return self._current_folder_path

    @property
    def loaded_folder_paths(self) -> tuple[Path, ...]:
        return tuple(self._loaded_folder_paths)

    @QtCore.Slot()
    def stop_active_scan(self) -> None:
        if self._scanner_worker is not None:
            if self._scanner_worker.isRunning():
                self._scanner_worker.request_stop()
                self._scanner_worker.wait(2000)
            self._scanner_worker.deleteLater()
            self._scanner_worker = None
        self.panel.targetVideosList.finalize_folder_loading()

    @QtCore.Slot(bool)
    def _on_scan_subfolders_toggled(self, _checked: bool) -> None:
        if self._loaded_folder_paths:
            self._rescan_all_loaded_sources()
        elif self._current_folder_path is not None and self._current_folder_path.is_dir():
            self.set_target_folder(self._current_folder_path)
        else:
            self.bus.workspaceStateDirty.emit()

    def _rescan_all_loaded_sources(self) -> None:
        """Rescan all tracked folders (and preserve Drag & Drop items) when Subfolders is toggled."""
        folders_to_scan = [p for p in self._loaded_folder_paths if p.is_dir()]
        saved_drops = [p for p in self._dropped_file_paths if p.is_file()]

        self.stop_active_scan()
        self.panel.targetVideosList.clear()
        self._dropped_file_paths = saved_drops

        if saved_drops:
            drop_items: list[SortableListWidgetItem] = []
            for drop_file in saved_drops:
                thumb_path, metadata = self.thumbnail_manager.ensure_thumbnail_and_metadata(
                    drop_file
                )
                pixmap = QtGui.QPixmap(thumb_path) if thumb_path else QtGui.QPixmap()
                item = self._build_card_item(
                    file_path=drop_file,
                    raw_pixmap=pixmap,
                    metadata=metadata,
                    folder_key=DRAG_DROP_FOLDER_KEY,
                )
                drop_items.append(item)
            self.panel.targetVideosList.add_streamed_items_batch(drop_items)

        if folders_to_scan:
            include_subfolders = self.panel.targetVideosScanSubfoldersCheckBox.isChecked()
            self._scan_is_additive = False
            self._start_scanner_worker(
                paths=folders_to_scan,
                is_folder=True,
                include_images=True,
                include_videos=True,
                include_subfolders=include_subfolders,
            )
        else:
            self.panel.targetVideosList.apply_sort()
            self.panel.targetVideosList.apply_filters()

        self.bus.workspaceStateDirty.emit()

    @QtCore.Slot()
    def select_target_folder_dialog(self) -> None:
        start_dir = (
            str(self._current_folder_path)
            if (self._current_folder_path and self._current_folder_path.is_dir())
            else str(Path.home())
        )
        selected_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self.panel, "Select Target Videos/Images Folder", start_dir
        )
        if selected_dir:
            self.set_target_folder(Path(selected_dir))

    @QtCore.Slot()
    def select_target_files_dialog(self) -> None:
        start_dir = (
            str(self._current_folder_path)
            if (self._current_folder_path and self._current_folder_path.is_dir())
            else str(Path.home())
        )
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self.panel,
            "Select Target Video/Image Files",
            start_dir,
            "Media Files (*.mp4 *.avi *.mkv *.mov *.webm *.wmv *.flv *.m4v *.3gp *.gif "
            "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff);;All Files (*)",
        )
        if files:
            paths = tuple(Path(os.path.abspath(f)) for f in files)
            keep_existing = self.panel.targetVideosKeepFoldersCheckBox.isChecked()
            self.load_target_files(paths, clear_existing=not keep_existing)

    @QtCore.Slot(list)
    def handle_dropped_paths(self, dropped_paths: Sequence[Path]) -> None:
        """Handle files or folders dropped onto the Target Media panel by placing them
        into the virtual 'Drag & Drop' folder group.
        """
        include_subfolders = self.panel.targetVideosScanSubfoldersCheckBox.isChecked()
        candidate_files: list[Path] = []

        for raw_path in dropped_paths:
            abs_str = os.path.abspath(str(raw_path))
            if os.path.isfile(abs_str):
                if get_file_type(abs_str) is not None:
                    candidate_files.append(Path(abs_str))
            elif os.path.isdir(abs_str):
                candidate_files.extend(
                    iter_media_files(
                        abs_str,
                        include_images=True,
                        include_videos=True,
                        include_subfolders=include_subfolders,
                    )
                )

        if not candidate_files:
            return

        lst = self.panel.targetVideosList
        new_files: list[Path] = []
        for f in candidate_files:
            if not lst.has_media_item(DRAG_DROP_FOLDER_KEY, f):
                self._dropped_file_paths.append(f)
                new_files.append(f)

        if not new_files:
            return

        self.stop_active_scan()
        self._scan_is_additive = True
        self._start_scanner_worker(
            paths=new_files,
            is_folder=False,
            include_images=True,
            include_videos=True,
            include_subfolders=False,
            virtual_folder_key=DRAG_DROP_FOLDER_KEY,
        )

        self.bus.targetFilesAdded.emit(TargetFilesAddedEvent(file_paths=tuple(new_files)))
        self.bus.workspaceStateDirty.emit()

    def set_target_folder(self, folder_path: Path, *, keep_existing: Optional[bool] = None) -> None:
        resolved = Path(os.path.abspath(str(folder_path)))
        if not resolved.is_dir():
            return

        keep = (
            self.panel.targetVideosKeepFoldersCheckBox.isChecked()
            if keep_existing is None
            else keep_existing
        )
        self._current_folder_path = resolved
        self.stop_active_scan()

        if keep:
            if resolved not in self._loaded_folder_paths:
                self._loaded_folder_paths.append(resolved)
            self._scan_is_additive = True
        else:
            self.clear_media_items()
            self._loaded_folder_paths = [resolved]
            self.panel.targetVideosList.setRootFolderPath(resolved)
            self._scan_is_additive = False

        tooltip_lines = [str(p) for p in self._loaded_folder_paths]
        self.panel.buttonTargetVideosPath.setToolTip("\n".join(tooltip_lines))

        include_subfolders = self.panel.targetVideosScanSubfoldersCheckBox.isChecked()
        self._start_scanner_worker(
            paths=[resolved],
            is_folder=True,
            include_images=True,
            include_videos=True,
            include_subfolders=include_subfolders,
        )

        self.bus.targetFolderChanged.emit(
            TargetFolderChangedEvent(
                folder_path=resolved,
                include_images=self.panel.targetVideosFilterImagesCheckBox.isChecked(),
                include_videos=self.panel.targetVideosFilterVideosCheckBox.isChecked(),
            )
        )
        self.bus.workspaceStateDirty.emit()

    def load_target_files(
        self,
        file_paths: Sequence[Path],
        *,
        clear_existing: bool = True,
        virtual_folder_key: Optional[str] = None,
    ) -> None:
        resolved_paths = tuple(
            Path(os.path.abspath(str(p))) for p in file_paths if os.path.isfile(str(p))
        )
        if not resolved_paths:
            return
        self.stop_active_scan()
        if clear_existing:
            self.clear_media_items()
            parent_dir = resolved_paths[0].parent
            self._current_folder_path = parent_dir
            self.panel.buttonTargetVideosPath.setToolTip(str(parent_dir))
            self.panel.targetVideosList.setRootFolderPath(parent_dir)

        self._scan_is_additive = not clear_existing
        self._start_scanner_worker(
            paths=resolved_paths,
            is_folder=False,
            include_images=True,
            include_videos=True,
            include_subfolders=False,
            virtual_folder_key=virtual_folder_key,
        )

        self.bus.targetFilesAdded.emit(TargetFilesAddedEvent(file_paths=resolved_paths))
        self.bus.workspaceStateDirty.emit()

    def _start_scanner_worker(
        self,
        paths: Sequence[Path],
        *,
        is_folder: bool,
        include_images: bool,
        include_videos: bool,
        include_subfolders: bool = False,
        virtual_folder_key: Optional[str] = None,
    ) -> None:
        self._scanner_worker = MediaScannerWorker(
            paths=paths,
            thumbnail_manager=self.thumbnail_manager,
            is_folder=is_folder,
            include_images=include_images,
            include_videos=include_videos,
            include_subfolders=include_subfolders,
            virtual_folder_key=virtual_folder_key,
            parent=self,
        )
        self._scanner_worker.foldersDiscovered.connect(self._on_folders_discovered)
        self._scanner_worker.itemsBatchDiscovered.connect(self._on_media_items_batch_discovered)
        self._scanner_worker.scanFinished.connect(self._on_scan_finished)
        self._scanner_worker.start()

    @QtCore.Slot(object)
    def _on_folders_discovered(self, folder_totals: dict[str, int]) -> None:
        self.panel.targetVideosList.initialize_folder_totals(
            folder_totals, additive=self._scan_is_additive
        )

    @QtCore.Slot(object)
    def _on_media_items_batch_discovered(
        self, packets: list[DiscoveredMediaItem]
    ) -> None:
        """Process a time-sliced batch of discovered items in a single coalesced UI update."""
        lst = self.panel.targetVideosList
        new_items: list[SortableListWidgetItem] = []

        for packet in packets:
            target_folder_key = (
                packet.folder_key
                if packet.folder_key is not None
                else os.path.dirname(str(packet.file_path))
            )
            if lst.has_media_item(target_folder_key, packet.file_path):
                continue

            pixmap = (
                QtGui.QPixmap(packet.thumbnail_path)
                if packet.thumbnail_path
                else QtGui.QPixmap()
            )
            item = self._build_card_item(
                file_path=packet.file_path,
                raw_pixmap=pixmap,
                metadata=packet.metadata,
                folder_key=packet.folder_key,
            )
            new_items.append(item)

        if new_items:
            lst.add_streamed_items_batch(new_items)

    @QtCore.Slot(int)
    def _on_scan_finished(self, total_loaded: int) -> None:
        lst = self.panel.targetVideosList
        lst.finalize_folder_loading()
        lst.apply_sort()
        lst.apply_filters()
        self.refresh_dimension_slider_bounds()
        self.bus.statusMessagePublished.emit(f"Loaded {total_loaded} target media item(s).", 3000)

    # =========================================================================
    # RIGHT-CLICK CONTEXT MENU ACTIONS
    # =========================================================================

    @QtCore.Slot(QtCore.QPoint)
    def _on_context_menu_requested(self, pos: QtCore.QPoint) -> None:
        lst = self.panel.targetVideosList
        clicked_item = lst.itemAt(pos)
        menu = QtWidgets.QMenu(lst)

        if isinstance(clicked_item, SortableListWidgetItem):
            if not clicked_item.isSelected():
                lst.clearSelection()
                clicked_item.setSelected(True)
                lst.setCurrentItem(clicked_item)

            selected_cards = [
                it
                for it in lst.selectedItems()
                if isinstance(it, SortableListWidgetItem)
            ]
            if not selected_cards:
                selected_cards = [clicked_item]

            fileinfo = clicked_item.fileInfo()
            clicked_path = (
                Path(os.path.abspath(fileinfo.filePath()))
                if (fileinfo and fileinfo.filePath())
                else None
            )

            act_open = menu.addAction("Open File Location")
            act_open.setEnabled(clicked_path is not None and clicked_path.exists())
            act_open.triggered.connect(
                lambda: self._open_in_file_manager(clicked_path, select_file=True)
            )

            remove_label = (
                f"Remove from list ({len(selected_cards)})"
                if len(selected_cards) > 1
                else "Remove from list"
            )
            act_remove = menu.addAction(remove_label)
            act_remove.triggered.connect(
                lambda: self._remove_media_cards_from_list(selected_cards)
            )

            del_label = (
                f"Delete {len(selected_cards)} files to recycle bin"
                if len(selected_cards) > 1
                else "Delete file to recycle bin"
            )
            act_delete = menu.addAction(self._warning_icon, del_label)
            act_delete.setEnabled(clicked_path is not None)
            act_delete.triggered.connect(
                lambda: self._delete_media_cards_to_recycle_bin(selected_cards)
            )
            menu.addSeparator()

        elif isinstance(clicked_item, FolderHeaderListWidgetItem):
            folder_key = clicked_item.folderPath()
            is_virtual_drop = folder_key == DRAG_DROP_FOLDER_KEY
            folder_path = Path(folder_key) if (folder_key and not is_virtual_drop) else None

            act_open_folder = menu.addAction("Open Folder Location")
            act_open_folder.setEnabled(folder_path is not None and folder_path.is_dir())
            act_open_folder.triggered.connect(
                lambda: self._open_in_file_manager(folder_path, select_file=False)
            )

            act_remove_folder = menu.addAction("Remove folder from list")
            act_remove_folder.triggered.connect(
                lambda: self._remove_folder_group_from_list(folder_key)
            )

            act_delete_folder = menu.addAction(
                self._warning_icon, "Delete folder to recycle bin"
            )
            act_delete_folder.triggered.connect(
                lambda: self._delete_folder_group_to_recycle_bin(folder_key)
            )
            menu.addSeparator()

        has_any_media = len(lst.sortable_items()) > 0

        act_clean_all = menu.addAction("Clean all lists")
        act_clean_all.setEnabled(has_any_media)
        act_clean_all.triggered.connect(self._clean_all_lists)

        act_delete_all = menu.addAction(
            self._warning_icon, "Delete All to recycle bin"
        )
        act_delete_all.setEnabled(has_any_media)
        act_delete_all.triggered.connect(self._delete_all_to_recycle_bin)

        menu.exec(lst.viewport().mapToGlobal(pos))

    def _open_in_file_manager(
        self, target_path: Optional[Path], *, select_file: bool = False
    ) -> None:
        if target_path is None:
            return
        win_path = os.path.normpath(os.path.abspath(str(target_path)))
        if not os.path.exists(win_path):
            return
        try:
            if sys.platform.startswith("win"):
                if select_file and os.path.isfile(win_path):
                    # Explorer requires quotes ONLY around the path after /select, and native backslashes
                    subprocess.Popen(f'explorer.exe /select,"{win_path}"')
                else:
                    folder = win_path if os.path.isdir(win_path) else os.path.dirname(win_path)
                    os.startfile(folder)  # type: ignore[attr-defined]
            else:
                folder = win_path if os.path.isdir(win_path) else os.path.dirname(win_path)
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))
        except Exception as exc:
            self.bus.statusMessagePublished.emit(
                f"Could not open location: {exc}", 4000
            )

    def _remove_media_cards_from_list(
        self, cards: Sequence[SortableListWidgetItem]
    ) -> None:
        if not cards:
            return
        lst = self.panel.targetVideosList
        current_selected = lst.currentItem()
        removing_current = current_selected in cards

        for card in cards:
            if card.folderPath() == DRAG_DROP_FOLDER_KEY:
                info = card.fileInfo()
                if info and info.filePath():
                    p = Path(os.path.abspath(info.filePath()))
                    if p in self._dropped_file_paths:
                        self._dropped_file_paths.remove(p)

        lst.remove_media_items(cards)
        self._sync_loaded_folder_paths_after_removal()
        self.refresh_dimension_slider_bounds()

        if removing_current:
            self.bus.targetMediaCleared.emit()
        self.bus.workspaceStateDirty.emit()
        self.bus.statusMessagePublished.emit(
            f"Removed {len(cards)} media item(s) from list.", 3000
        )

    def _remove_folder_group_from_list(self, folder_key: str) -> None:
        lst = self.panel.targetVideosList
        current_selected = lst.currentItem()
        removing_current = (
            isinstance(current_selected, SortableListWidgetItem)
            and current_selected.folderPath() == folder_key
        )

        removed_items = lst.remove_folder_group(folder_key)
        if folder_key == DRAG_DROP_FOLDER_KEY:
            self._dropped_file_paths.clear()
        else:
            self._sync_loaded_folder_paths_after_removal()

        self.refresh_dimension_slider_bounds()
        if removing_current:
            self.bus.targetMediaCleared.emit()
        self.bus.workspaceStateDirty.emit()
        self.bus.statusMessagePublished.emit(
            f"Removed folder group ({len(removed_items)} item(s)) from list.", 3000
        )

    def _delete_media_cards_to_recycle_bin(
        self, cards: Sequence[SortableListWidgetItem]
    ) -> None:
        if not cards:
            return
        paths_to_delete: list[Path] = []
        for card in cards:
            info = card.fileInfo()
            if info and info.filePath():
                p = Path(os.path.abspath(info.filePath()))
                if p.exists():
                    paths_to_delete.append(p)

        if not paths_to_delete:
            self._remove_media_cards_from_list(cards)
            return

        count = len(paths_to_delete)
        title = "Delete to Recycle Bin"
        msg = (
            f"Move '{paths_to_delete[0].name}' to the Recycle Bin?"
            if count == 1
            else f"Move {count} selected media files to the Recycle Bin?"
        )
        reply = QtWidgets.QMessageBox.warning(
            self.panel,
            title,
            msg,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        deleted_cards: list[SortableListWidgetItem] = []
        errors: list[str] = []
        for card in cards:
            info = card.fileInfo()
            if not (info and info.filePath()):
                continue
            p = Path(os.path.abspath(info.filePath()))
            if self._send_path_to_recycle_bin(p, errors):
                deleted_cards.append(card)

        if deleted_cards:
            self._remove_media_cards_from_list(deleted_cards)
            self.bus.statusMessagePublished.emit(
                f"Moved {len(deleted_cards)} file(s) to the Recycle Bin.", 4000
            )
        if errors:
            QtWidgets.QMessageBox.critical(
                self.panel,
                "Recycle Bin Error",
                "Some files could not be moved to the Recycle Bin:\n"
                + "\n".join(errors[:5]),
            )

    def _delete_folder_group_to_recycle_bin(self, folder_key: str) -> None:
        lst = self.panel.targetVideosList
        folder_cards = lst.items_in_folder(folder_key)
        if not folder_cards:
            return

        is_virtual_drop = folder_key == DRAG_DROP_FOLDER_KEY
        display_name = DRAG_DROP_FOLDER_LABEL if is_virtual_drop else folder_key

        if is_virtual_drop:
            msg = (
                f"Move all {len(folder_cards)} media file(s) in '{DRAG_DROP_FOLDER_LABEL}' "
                "to the Recycle Bin?"
            )
        else:
            msg = (
                f"Move the folder '{display_name}' ({len(folder_cards)} media file(s)) "
                "to the Recycle Bin?"
            )

        reply = QtWidgets.QMessageBox.warning(
            self.panel,
            "Delete Folder to Recycle Bin",
            msg,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        errors: list[str] = []
        if not is_virtual_drop and os.path.isdir(folder_key):
            folder_path = Path(os.path.abspath(folder_key))
            if self._send_path_to_recycle_bin(folder_path, errors):
                self._remove_folder_group_from_list(folder_key)
                self.bus.statusMessagePublished.emit(
                    f"Moved folder '{folder_path.name}' to the Recycle Bin.", 4000
                )
        else:
            deleted_cards: list[SortableListWidgetItem] = []
            for card in folder_cards:
                info = card.fileInfo()
                if info and info.filePath():
                    p = Path(os.path.abspath(info.filePath()))
                    if self._send_path_to_recycle_bin(p, errors):
                        deleted_cards.append(card)
            if deleted_cards:
                self._remove_media_cards_from_list(deleted_cards)
                self.bus.statusMessagePublished.emit(
                    f"Moved {len(deleted_cards)} file(s) to the Recycle Bin.", 4000
                )

        if errors:
            QtWidgets.QMessageBox.critical(
                self.panel,
                "Recycle Bin Error",
                "Could not move all items to the Recycle Bin:\n" + "\n".join(errors[:5]),
            )

    def _clean_all_lists(self) -> None:
        self.stop_active_scan()
        self.clear_media_items()
        self.refresh_dimension_slider_bounds()
        self.bus.targetMediaCleared.emit()
        self.bus.workspaceStateDirty.emit()
        self.bus.statusMessagePublished.emit("Cleared all target media lists.", 3000)

    def _delete_all_to_recycle_bin(self) -> None:
        lst = self.panel.targetVideosList
        all_cards = lst.sortable_items()
        if not all_cards:
            return

        reply = QtWidgets.QMessageBox.warning(
            self.panel,
            "Delete All to Recycle Bin",
            f"Are you sure you want to move ALL {len(all_cards)} loaded media file(s) "
            "across all folders to the Recycle Bin?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self.stop_active_scan()
        errors: list[str] = []
        deleted_cards: list[SortableListWidgetItem] = []
        seen_paths: set[str] = set()

        for card in all_cards:
            info = card.fileInfo()
            if not (info and info.filePath()):
                continue
            p_str = os.path.abspath(info.filePath())
            norm_str = os.path.normcase(p_str)
            if norm_str in seen_paths:
                deleted_cards.append(card)
                continue
            if self._send_path_to_recycle_bin(Path(p_str), errors):
                seen_paths.add(norm_str)
                deleted_cards.append(card)

        if deleted_cards:
            self._remove_media_cards_from_list(deleted_cards)
            self.bus.statusMessagePublished.emit(
                f"Moved {len(seen_paths)} media file(s) to the Recycle Bin.", 4000
            )
        if errors:
            QtWidgets.QMessageBox.critical(
                self.panel,
                "Recycle Bin Error",
                "Some files could not be moved to the Recycle Bin:\n"
                + "\n".join(errors[:5]),
            )

    @staticmethod
    def _send_path_to_recycle_bin(target: Path, errors: list[str]) -> bool:
        if not target.exists():
            return True
        try:
            if send2trash is not None:
                send2trash(str(target))
                return True
            if QtCore.QFile.moveToTrash(str(target)):
                return True
            errors.append(f"{target.name}: Recycle Bin unavailable")
            return False
        except Exception as exc:
            errors.append(f"{target.name}: {exc}")
            return False

    def _sync_loaded_folder_paths_after_removal(self) -> None:
        """Keep `self._loaded_folder_paths` synchronized with remaining folders in the list."""
        remaining_folder_strs = {
            item.folderPath()
            for item in self.panel.targetVideosList.sortable_items()
            if item.folderPath() and item.folderPath() != DRAG_DROP_FOLDER_KEY
        }
        kept: list[Path] = []
        for root_dir in self._loaded_folder_paths:
            has_any = False
            for rem_str in remaining_folder_strs:
                rem_path = Path(rem_str)
                try:
                    if rem_path == root_dir or rem_path.is_relative_to(root_dir):
                        has_any = True
                        break
                except ValueError:
                    pass
            if has_any:
                kept.append(root_dir)
        self._loaded_folder_paths = kept
        tooltip_lines = [str(p) for p in self._loaded_folder_paths]
        self.panel.buttonTargetVideosPath.setToolTip(
            "\n".join(tooltip_lines) if tooltip_lines else "Select Target Videos/Images Folder"
        )

    # =========================================================================
    # SORTING & FILTERING SLOTS
    # =========================================================================

    @QtCore.Slot(int)
    def _on_sort_mode_changed(self, index: int) -> None:
        mode = self.panel.targetVideosSortComboBox.itemData(index)
        self.panel.targetVideosList.setSortingMode(mode)
        self.bus.workspaceStateDirty.emit()

    @QtCore.Slot()
    def _on_media_type_filter_changed(self) -> None:
        self._sync_allowed_media_types()
        self.bus.workspaceStateDirty.emit()

    @QtCore.Slot(bool)
    def _on_webcam_filter_toggled(self, checked: bool) -> None:
        self._sync_allowed_media_types()
        self.bus.targetWebcamsToggled.emit(checked)
        self.bus.workspaceStateDirty.emit()

    def _sync_allowed_media_types(self) -> None:
        allowed: set[str] = set()
        if self.panel.targetVideosFilterImagesCheckBox.isChecked():
            allowed.add(MediaType.IMAGE.value)
        if self.panel.targetVideosFilterVideosCheckBox.isChecked():
            allowed.add(MediaType.VIDEO.value)
        if self.panel.targetVideosFilterWebcamsCheckBox.isChecked():
            allowed.add(MediaType.WEBCAM.value)
        self.panel.targetVideosList.setAllowedFileTypes(allowed)

    @QtCore.Slot()
    def _on_min_resolution_changed(self) -> None:
        min_w = self.panel.targetVideosWidthSpinBox.value()
        min_h = self.panel.targetVideosHeightSpinBox.value()
        self.panel.targetVideosList.setMinResolution(min_w, min_h)

    @QtCore.Slot(QtWidgets.QListWidgetItem, QtWidgets.QListWidgetItem)
    def _on_current_item_changed(
        self,
        current: Optional[QtWidgets.QListWidgetItem],
        _previous: Optional[QtWidgets.QListWidgetItem],
    ) -> None:
        if isinstance(current, FolderHeaderListWidgetItem):
            return
        if not isinstance(current, SortableListWidgetItem):
            self.bus.targetMediaCleared.emit()
            return

        fileinfo = current.fileInfo()
        file_path = Path(fileinfo.filePath()) if fileinfo and fileinfo.filePath() else None
        try:
            media_type = MediaType(current.fileType())
        except ValueError:
            media_type = MediaType.UNKNOWN

        payload = current.payload()
        webcam_idx = payload if (media_type == MediaType.WEBCAM and isinstance(payload, int)) else None
        media_id = str(payload) if payload is not None else (str(file_path) if file_path else current.text())

        self.bus.targetMediaSelected.emit(
            TargetMediaSelectedEvent(
                media_id=media_id,
                media_type=media_type,
                file_path=file_path,
                webcam_index=webcam_idx,
                metadata=current.mediaMetadata(),
            )
        )

    @QtCore.Slot(int)
    def _on_visible_count_changed(self, visible_count: int) -> None:
        lst = self.panel.targetVideosList
        self.bus.targetMediaFilterChanged.emit(
            TargetMediaFilterChangedEvent(
                visible_count=visible_count,
                total_count=len(lst.sortable_items()),
                search_text=self.panel.targetVideosSearchBox.text().strip(),
                min_dimension=lst.minImageDimension(),
            )
        )

    # =========================================================================
    # ITEM POPULATION HELPERS
    # =========================================================================

    def clear_media_items(self) -> None:
        self._loaded_folder_paths.clear()
        self._dropped_file_paths.clear()
        self.panel.targetVideosList.clear()

    @staticmethod
    def _build_card_item(
        file_path: str | Path,
        *,
        raw_pixmap: QtGui.QPixmap,
        metadata: Optional[MediaMetadata] = None,
        media_id: Optional[str] = None,
        folder_key: Optional[str] = None,
    ) -> SortableListWidgetItem:
        abs_str = os.path.abspath(str(file_path))
        item = SortableListWidgetItem.from_file_path(
            file_path=abs_str,
            icon=QtGui.QIcon(),
            metadata=metadata,
            payload=media_id or abs_str,
            folder_key=folder_key,
        )
        CardManager.prepare_list_item(
            item=item,
            raw_pixmap=raw_pixmap,
            file_path=abs_str,
            layout_mode=CardLayoutMode.WIDESCREEN_16_9,
            crop_to_fill=False,
        )
        return item

    def add_media_item(
        self,
        file_path: str | Path,
        *,
        raw_pixmap: QtGui.QPixmap,
        metadata: Optional[MediaMetadata] = None,
        media_id: Optional[str] = None,
        folder_key: Optional[str] = None,
        defer_sort: bool = False,
    ) -> SortableListWidgetItem:
        abs_str = os.path.abspath(str(file_path))
        target_folder_key = (
            folder_key if folder_key is not None else os.path.dirname(abs_str)
        )
        lst = self.panel.targetVideosList

        item = self._build_card_item(
            file_path=abs_str,
            raw_pixmap=raw_pixmap,
            metadata=metadata,
            media_id=media_id,
            folder_key=target_folder_key,
        )

        if not lst.has_media_item(target_folder_key, abs_str):
            lst.add_streamed_items_batch([item])

        if not defer_sort:
            lst.apply_sort()
            lst.apply_filters()
            self.refresh_dimension_slider_bounds()
        return item

    def refresh_dimension_slider_bounds(self) -> None:
        max_dim = self.panel.targetVideosList.max_image_dimension()
        max_w = max(max_dim.width, self.panel.targetVideosWidthSpinBox.value(), 256)
        max_h = max(max_dim.height, self.panel.targetVideosHeightSpinBox.value(), 256)

        self.panel.targetVideosWidthSlider.setMaximum(max_w)
        self.panel.targetVideosWidthSpinBox.setMaximum(max_w)
        self.panel.targetVideosHeightSlider.setMaximum(max_h)
        self.panel.targetVideosHeightSpinBox.setMaximum(max_h)