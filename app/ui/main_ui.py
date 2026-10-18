# app/ui/main_ui.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from app.helpers.app_metadata import get_app_display_metadata
from app.helpers.thumbnail_manager import ThumbnailManager
from app.ui.actions.target_media_actions import TargetMediaActions
from app.ui.core.event_bus import AppEventBus, get_event_bus
from app.ui.core.modular_window import Ui_MainWindow
from app.ui.widgets.panels import (
    ControlOptionsPanel,
    DrawerZone,
    FacesEmbeddingsPanel,
    InputFacesPanel,
    MenuBarPanel,
    TargetMediaPanel,
    TimelinePanel,
)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """Modular UI shell assembling the central viewport and 5-zone drawer modules."""

    placeholder_update_signal = QtCore.Signal(QtWidgets.QListWidget, bool)
    gpu_memory_update_signal = QtCore.Signal(int, int)
    model_loading_signal = QtCore.Signal()
    model_loaded_signal = QtCore.Signal()
    display_messagebox_signal = QtCore.Signal(str, str, QtWidgets.QWidget)

    def __init__(
        self,
        gpu_id: int = 0,
        event_bus: Optional[AppEventBus] = None,
        thumbnail_manager: Optional[ThumbnailManager] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.gpu_id: int = gpu_id
        self.project_root_path: Path = Path(__file__).resolve().parent.parent.parent

        # 0. Initialize shared core services (Event Bus & Thumbnail Manager)
        self.event_bus: AppEventBus = event_bus or get_event_bus()
        self.thumbnail_manager: ThumbnailManager = (
            thumbnail_manager
            or ThumbnailManager(thumbnail_dir=self.project_root_path / ".thumbnails")
        )

        # 1. Initialize pure canvas & central viewport
        self.init_canvas(self)

        # 2. Populate drawers with default modular panels
        self._assemble_default_modules()

        # 3. Initialize static/dynamic overlay manager
        self.init_overlay_controller(self, trigger_ratio=0.10)

        # 4. Instantiate Action Controllers & wire shell actions
        self._assemble_action_controllers()
        self._wire_shell_actions()

        # 5. Set window title with version/branch metadata
        self._base_window_title: str = self.windowTitle()
        self._apply_runtime_window_title()

    def _assemble_default_modules(self) -> None:
        """Instantiates and registers the default VisoMaster widget modules."""
        # Top Menu Bar (Isolated on its own layer above all dock areas)
        self.menu_bar_panel = MenuBarPanel(self)
        self.register_menu_bar(self, self.menu_bar_panel)

        # Left Drawer: Target Videos/Images + Input Faces
        self.target_media_panel = TargetMediaPanel(self)
        self.input_faces_panel = InputFacesPanel(self)
        self.register_panel(
            self, self.target_media_panel, DrawerZone.LEFT, Qt.Orientation.Vertical
        )
        self.register_panel(
            self, self.input_faces_panel, DrawerZone.LEFT, Qt.Orientation.Vertical
        )

        # Right Drawer: Control Options & Parameter Tabs
        self.control_options_panel = ControlOptionsPanel(self)
        self.register_panel(self, self.control_options_panel, DrawerZone.RIGHT)

        # Bottom Drawer: Timeline & Playback stacked above Faces & Embeddings
        self.timeline_panel = TimelinePanel(self)
        self.faces_embeddings_panel = FacesEmbeddingsPanel(self)
        self.register_panel(
            self, self.timeline_panel, DrawerZone.BOTTOM, Qt.Orientation.Vertical
        )
        self.register_panel(
            self,
            self.faces_embeddings_panel,
            DrawerZone.BOTTOM,
            Qt.Orientation.Vertical,
        )
        # Legacy alias compatibility
        self.bottomControlsDockWidget = self.timeline_panel

        # Populate View menu with all registered dock toggles + Theme menu
        self.menu_bar_panel.finalize_view_menu(
            panels=self._all_panels,
            set_all_visible_cb=self.set_all_panels_visible,
            main_window=self,
        )

    def _assemble_action_controllers(self) -> None:
        """Instantiates per-panel action controllers with shared Event Bus & ThumbnailManager."""
        self.target_media_actions = TargetMediaActions(
            panel=self.target_media_panel,
            event_bus=self.event_bus,
            thumbnail_manager=self.thumbnail_manager,
            parent=self,
        )
        # When InputFacesActions is created next, it shares the exact same instances:
        # self.input_faces_actions = InputFacesActions(
        #     panel=self.input_faces_panel,
        #     event_bus=self.event_bus,
        #     thumbnail_manager=self.thumbnail_manager,
        #     parent=self,
        # )

    def _wire_shell_actions(self) -> None:
        """Connects shell-level menu and overlay mode toggles."""
        self.actionTheatre_Overlay_Mode.toggled.connect(
            self.theatreModeButton.setChecked
        )
        self.theatreModeButton.toggled.connect(
            self.actionTheatre_Overlay_Mode.setChecked
        )
        self.theatreModeButton.toggled.connect(
            self.overlayController.set_overlay_mode
        )
        self.theatreModeButton.setChecked(False)

        # Top Menu Bar -> Target Media file/folder dialogs
        self.actionOpen_Videos_Folder.triggered.connect(
            self.target_media_actions.select_target_folder_dialog
        )
        self.actionOpen_Video_Files.triggered.connect(
            self.target_media_actions.select_target_files_dialog
        )

        self.actionQuit.triggered.connect(self.close)
        self.actionQuit_WithoutSaving.triggered.connect(self.close)

    def _apply_runtime_window_title(self) -> None:
        """Applies the project version and git metadata to the window title."""
        try:
            self.app_display_metadata = get_app_display_metadata(
                self.project_root_path, self._base_window_title
            )
            self.setWindowTitle(self.app_display_metadata.window_title)
        except Exception:
            self.setWindowTitle(self._base_window_title)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Notify all action controllers/workers to stop cleanly before closing."""
        self.event_bus.applicationClosing.emit()
        event.accept()