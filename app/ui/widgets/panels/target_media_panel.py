# app/ui/widgets/panels/target_media_panel.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from app.ui.core.asset_manager import AssetManager
from app.ui.widgets.panels.base_panel import BaseDockPanel, DrawerZone
from app.ui.widgets.sortable_widgets import SortableListWidget


class TargetMediaPanel(BaseDockPanel):
    """Modular dock panel for selecting, searching, and filtering target videos/images."""

    panel_id = "input_Target_DockWidget"
    panel_title = "Target Videos/Images"
    default_zone = DrawerZone.LEFT

    DEFAULT_MAX_DIMENSION: int = 4096

    pathsDropped = QtCore.Signal(list)  # list[Path] emitted when files/folders are dropped on the panel

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        self.setAcceptDrops(True)

    # --- Panel-Wide Drag & Drop Support ---

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            dropped_paths: list[Path] = []
            for url in mime.urls():
                if url.isLocalFile():
                    local_path = Path(url.toLocalFile())
                    if local_path.exists():
                        dropped_paths.append(local_path.resolve())
            if dropped_paths:
                event.acceptProposedAction()
                self.pathsDropped.emit(dropped_paths)
                return
        event.ignore()

    def _build_ui(self) -> None:
        self.container.setObjectName("dockWidgetContents")
        self.vboxLayout = QtWidgets.QVBoxLayout(self.container)
        self.vboxLayout.setContentsMargins(4, 4, 4, 4)

        # ---------------------------------------------------------
        # 1. Folder Selection + Subfolders + Keep Folders + Thumbnail Size Row
        # ---------------------------------------------------------
        folder_row = QtWidgets.QHBoxLayout()
        folder_row.setSpacing(6)

        self.buttonTargetVideosPath = QtWidgets.QPushButton(
            "Select Folder", self.container
        )
        self.buttonTargetVideosPath.setObjectName("buttonTargetVideosPath")
        self.buttonTargetVideosPath.setIconSize(QtCore.QSize(18, 18))
        self.buttonTargetVideosPath.setFlat(True)

        self.targetVideosScanSubfoldersCheckBox = QtWidgets.QCheckBox(
            "Subfolders", self.container
        )
        self.targetVideosScanSubfoldersCheckBox.setObjectName(
            "targetVideosScanSubfoldersCheckBox"
        )
        self.targetVideosScanSubfoldersCheckBox.setToolTip(
            "Scan subfolders recursively for target videos and images"
        )
        self.targetVideosScanSubfoldersCheckBox.setChecked(False)

        self.targetVideosKeepFoldersCheckBox = QtWidgets.QCheckBox(
            "Keep Folders", self.container
        )
        self.targetVideosKeepFoldersCheckBox.setObjectName(
            "targetVideosKeepFoldersCheckBox"
        )
        self.targetVideosKeepFoldersCheckBox.setToolTip(
            "Keep already loaded folders and Drag & Drop items when selecting a new folder"
        )
        self.targetVideosKeepFoldersCheckBox.setChecked(False)

        # Thumbnail Card Size Selector (Managed by CardManager)
        self.targetVideosViewSizeComboBox = QtWidgets.QComboBox(self.container)
        self.targetVideosViewSizeComboBox.setObjectName("targetVideosViewSizeComboBox")
        self.targetVideosViewSizeComboBox.setToolTip("Thumbnail Card Size")
        self.targetVideosViewSizeComboBox.addItems(["Small", "Medium", "Large"])
        self.targetVideosViewSizeComboBox.setCurrentIndex(1)
        self.targetVideosViewSizeComboBox.setStyleSheet(
            "QComboBox#targetVideosViewSizeComboBox { min-width: 56px; max-width: 56px; padding-right: 14px; }"
        )
        self.targetVideosViewSizeComboBox.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )

        folder_row.addWidget(self.buttonTargetVideosPath, 1)
        folder_row.addWidget(self.targetVideosScanSubfoldersCheckBox)
        folder_row.addWidget(self.targetVideosKeepFoldersCheckBox)
        folder_row.addWidget(self.targetVideosViewSizeComboBox)
        self.vboxLayout.addLayout(folder_row)

        # ---------------------------------------------------------
        # 2. Search + Filter Toggle Row
        # ---------------------------------------------------------
        search_row = QtWidgets.QHBoxLayout()
        self.targetVideosSearchBox = QtWidgets.QLineEdit(self.container)
        self.targetVideosSearchBox.setObjectName("targetVideosSearchBox")
        self.targetVideosSearchBox.setPlaceholderText("Search Videos/Images")

        self.targetVideosSearchResetButton = QtWidgets.QPushButton(self.container)
        self.targetVideosSearchResetButton.setObjectName("targetVideosSearchResetButton")
        self.targetVideosSearchResetButton.setFixedSize(QtCore.QSize(28, 28))
        self.targetVideosSearchResetButton.setToolTip("Clear search text")
        self.targetVideosSearchResetButton.setIcon(AssetManager.get_icon("glyphs/times_on.svg"))
        self.targetVideosSearchResetButton.setIconSize(QtCore.QSize(26, 26))
        self.targetVideosSearchResetButton.setFlat(True)

        self.targetVideosFilterMenuButton = QtWidgets.QPushButton(self.container)
        self.targetVideosFilterMenuButton.setObjectName("targetVideosFilterMenuButton")
        self.targetVideosFilterMenuButton.setFixedSize(QtCore.QSize(28, 28))
        self.targetVideosFilterMenuButton.setToolTip("Show/Hide filter and sorting options")
        self.targetVideosFilterMenuButton.setIcon(
            AssetManager.get_stateful_icon("glyphs/filter_bold.svg", on_file="glyphs/filter_on.svg")
        )
        self.targetVideosFilterMenuButton.setIconSize(QtCore.QSize(26, 26))
        self.targetVideosFilterMenuButton.setCheckable(True)
        self.targetVideosFilterMenuButton.setFlat(True)

        search_row.addWidget(self.targetVideosSearchBox, 1)
        search_row.addWidget(self.targetVideosSearchResetButton)
        search_row.addWidget(self.targetVideosFilterMenuButton)
        self.vboxLayout.addLayout(search_row)

        # ---------------------------------------------------------
        # 3. Collapsible Filter & Sort Container
        # ---------------------------------------------------------
        self.targetVideosFilterContainer = QtWidgets.QWidget(self.container)
        self.targetVideosFilterContainer.setObjectName("targetVideosFilterContainer")
        filter_vbox = QtWidgets.QVBoxLayout(self.targetVideosFilterContainer)
        filter_vbox.setContentsMargins(0, 0, 0, 0)
        filter_vbox.setSpacing(4)

        sort_row = QtWidgets.QHBoxLayout()
        self.targetVideosSortComboBox = QtWidgets.QComboBox(self.targetVideosFilterContainer)
        self.targetVideosSortComboBox.setObjectName("targetVideosSortComboBox")
        self.targetVideosSortComboBox.setMinimumHeight(28)
        self.targetVideosSortComboBox.setToolTip("Sort target media by...")
        SortableListWidget.populate_sort_combobox(self.targetVideosSortComboBox)

        self.sortTargetVideosDirectionButton = QtWidgets.QPushButton(self.targetVideosFilterContainer)
        self.sortTargetVideosDirectionButton.setObjectName("sortTargetVideosDirectionButton")
        self.sortTargetVideosDirectionButton.setFixedSize(QtCore.QSize(28, 28))
        self.sortTargetVideosDirectionButton.setToolTip("Sort ascending / descending")
        self.sortTargetVideosDirectionButton.setIcon(
            AssetManager.get_stateful_icon("glyphs/sort-amount-down_bold.svg", on_file="glyphs/sort-amount-up_bold.svg")
        )
        self.sortTargetVideosDirectionButton.setIconSize(QtCore.QSize(24, 24))
        self.sortTargetVideosDirectionButton.setCheckable(True)
        self.sortTargetVideosDirectionButton.setFlat(True)

        self.targetVideosFilterImagesCheckBox = QtWidgets.QCheckBox(self.targetVideosFilterContainer)
        self.targetVideosFilterImagesCheckBox.setObjectName("targetVideosFilterImagesCheckBox")
        self.targetVideosFilterImagesCheckBox.setToolTip("Include Images")
        self.targetVideosFilterImagesCheckBox.setChecked(True)
        self.targetVideosFilterImagesCheckBox.setIcon(
            AssetManager.get_stateful_icon("glyphs/images_bold.svg", on_file="glyphs/images_on.svg")
        )
        self.targetVideosFilterImagesCheckBox.setIconSize(QtCore.QSize(22, 22))

        self.targetVideosFilterVideosCheckBox = QtWidgets.QCheckBox(self.targetVideosFilterContainer)
        self.targetVideosFilterVideosCheckBox.setObjectName("targetVideosFilterVideosCheckBox")
        self.targetVideosFilterVideosCheckBox.setToolTip("Include Videos")
        self.targetVideosFilterVideosCheckBox.setChecked(True)
        self.targetVideosFilterVideosCheckBox.setIcon(
            AssetManager.get_stateful_icon("glyphs/film_duo.svg", on_file="glyphs/film_on.svg")
        )
        self.targetVideosFilterVideosCheckBox.setIconSize(QtCore.QSize(22, 22))

        self.targetVideosFilterWebcamsCheckBox = QtWidgets.QCheckBox(self.targetVideosFilterContainer)
        self.targetVideosFilterWebcamsCheckBox.setObjectName("targetVideosFilterWebcamsCheckBox")
        self.targetVideosFilterWebcamsCheckBox.setToolTip("Include Webcams")
        self.targetVideosFilterWebcamsCheckBox.setChecked(False)
        self.targetVideosFilterWebcamsCheckBox.setIcon(
            AssetManager.get_stateful_icon("glyphs/camcorder_bold.svg", on_file="glyphs/camcorder_on.svg")
        )
        self.targetVideosFilterWebcamsCheckBox.setIconSize(QtCore.QSize(22, 22))

        sort_row.addWidget(self.targetVideosSortComboBox, 1)
        sort_row.addWidget(self.sortTargetVideosDirectionButton)
        sort_row.addWidget(self.targetVideosFilterImagesCheckBox)
        sort_row.addWidget(self.targetVideosFilterVideosCheckBox)
        sort_row.addWidget(self.targetVideosFilterWebcamsCheckBox)
        filter_vbox.addLayout(sort_row)

        width_row = QtWidgets.QHBoxLayout()
        self.targetVideosWidthLabel = QtWidgets.QLabel("Min Width", self.targetVideosFilterContainer)
        self.targetVideosWidthLabel.setMinimumWidth(62)
        self.targetVideosWidthSlider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self.targetVideosFilterContainer)
        self.targetVideosWidthSlider.setObjectName("targetVideosWidthSlider")
        self.targetVideosWidthSlider.setRange(0, self.DEFAULT_MAX_DIMENSION)
        self.targetVideosWidthSlider.setSingleStep(16)
        self.targetVideosWidthSlider.setPageStep(128)

        self.targetVideosWidthSpinBox = QtWidgets.QSpinBox(self.targetVideosFilterContainer)
        self.targetVideosWidthSpinBox.setObjectName("targetVideosWidthSpinBox")
        self.targetVideosWidthSpinBox.setFixedWidth(68)
        self.targetVideosWidthSpinBox.setRange(0, self.DEFAULT_MAX_DIMENSION)
        self.targetVideosWidthSpinBox.setSingleStep(16)

        width_row.addWidget(self.targetVideosWidthLabel)
        width_row.addWidget(self.targetVideosWidthSlider)
        width_row.addWidget(self.targetVideosWidthSpinBox)
        filter_vbox.addLayout(width_row)

        height_row = QtWidgets.QHBoxLayout()
        self.targetVideosHeightLabel = QtWidgets.QLabel("Min Height", self.targetVideosFilterContainer)
        self.targetVideosHeightLabel.setMinimumWidth(62)
        self.targetVideosHeightSlider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self.targetVideosFilterContainer)
        self.targetVideosHeightSlider.setObjectName("targetVideosHeightSlider")
        self.targetVideosHeightSlider.setRange(0, self.DEFAULT_MAX_DIMENSION)
        self.targetVideosHeightSlider.setSingleStep(16)
        self.targetVideosHeightSlider.setPageStep(128)

        self.targetVideosHeightSpinBox = QtWidgets.QSpinBox(self.targetVideosFilterContainer)
        self.targetVideosHeightSpinBox.setObjectName("targetVideosHeightSpinBox")
        self.targetVideosHeightSpinBox.setFixedWidth(68)
        self.targetVideosHeightSpinBox.setRange(0, self.DEFAULT_MAX_DIMENSION)
        self.targetVideosHeightSpinBox.setSingleStep(16)

        height_row.addWidget(self.targetVideosHeightLabel)
        height_row.addWidget(self.targetVideosHeightSlider)
        height_row.addWidget(self.targetVideosHeightSpinBox)
        filter_vbox.addLayout(height_row)

        self.vboxLayout.addWidget(self.targetVideosFilterContainer)

        # ---------------------------------------------------------
        # 4. Sortable Target Media Grid View (Grouped by Folder)
        # ---------------------------------------------------------
        self.targetVideosList = SortableListWidget(self.container)
        self.targetVideosList.setObjectName("targetVideosList")
        self.targetVideosList.setGroupByFolder(True)
        self.vboxLayout.addWidget(self.targetVideosList)

        # Default visual state: filter container collapsed
        self.targetVideosFilterContainer.setVisible(False)

        # Export all widgets
        for name in (
            "buttonTargetVideosPath",
            "targetVideosScanSubfoldersCheckBox",
            "targetVideosKeepFoldersCheckBox",
            "targetVideosSearchBox",
            "targetVideosSearchResetButton",
            "targetVideosViewSizeComboBox",
            "targetVideosFilterMenuButton",
            "targetVideosFilterContainer",
            "targetVideosSortComboBox",
            "sortTargetVideosDirectionButton",
            "targetVideosFilterImagesCheckBox",
            "targetVideosFilterVideosCheckBox",
            "targetVideosFilterWebcamsCheckBox",
            "targetVideosWidthLabel",
            "targetVideosWidthSlider",
            "targetVideosWidthSpinBox",
            "targetVideosHeightLabel",
            "targetVideosHeightSlider",
            "targetVideosHeightSpinBox",
            "targetVideosList",
        ):
            self.export_widget(name, getattr(self, name))