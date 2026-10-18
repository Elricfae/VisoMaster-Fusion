# app/ui/widgets/panels/menu_bar_panel.py
from __future__ import annotations

from typing import Callable, Dict, List, Optional
from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.core.theme_manager import ThemeManager


class MenuBarPanel(QtWidgets.QMenuBar):
    """Dedicated top menu bar that sits strictly above all dock areas.

    Cannot be docked into by widget modules, while supporting auto-hide
    reparenting into the top menu overlay drawer during Theatre/Overlay Mode.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("menubar")
        self._exported_widgets: Dict[str, QtCore.QObject] = {}
        self._build_menus()
        self._build_vram_monitor()

    def export_widget(self, name: str, widget: QtCore.QObject) -> None:
        setattr(self, name, widget)
        self._exported_widgets[name] = widget

    def bind_to_host(self, host: QtWidgets.QMainWindow) -> None:
        host.menubar = self
        for attr_name, widget in self._exported_widgets.items():
            setattr(host, attr_name, widget)

    def _build_vram_monitor(self) -> None:
        self.vramContainer = QtWidgets.QWidget(self)
        # Force the container background to be transparent
        self.vramContainer.setObjectName("vramContainer")
        self.vramContainer.setStyleSheet("#vramContainer { background-color: transparent; }")
        
        layout = QtWidgets.QHBoxLayout(self.vramContainer)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)

        self.vramProgressBar = QtWidgets.QProgressBar(self.vramContainer)
        self.vramProgressBar.setObjectName("vramProgressBar")
        self.vramProgressBar.setRange(0, 100)
        self.vramProgressBar.setValue(0)
        self.vramProgressBar.setFormat("VRAM: 0 MB / 0 MB")
        self.vramProgressBar.setFixedWidth(200) # Prevents it from stretching across the menu

        self.clearMemoryButton = QtWidgets.QPushButton(
            "Clear GPU Memory", self.vramContainer
        )
        self.clearMemoryButton.setObjectName("clearMemoryButton")
        self.clearMemoryButton.setFlat(True)

        layout.addWidget(self.vramProgressBar)
        layout.addWidget(self.clearMemoryButton)

        # Pin the container to the top right corner of the QMenuBar
        self.setCornerWidget(self.vramContainer, QtCore.Qt.Corner.TopRightCorner)

        self.export_widget("vramProgressBar", self.vramProgressBar)
        self.export_widget("clearMemoryButton", self.clearMemoryButton)

    def _build_menus(self) -> None:
        self.menuFile = QtWidgets.QMenu("File", self)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu("Edit", self)
        self.menuEdit.setObjectName("menuEdit")
        self.menuView = QtWidgets.QMenu("View", self)
        self.menuView.setObjectName("menuView")
        self.menuHelp = QtWidgets.QMenu("Help", self)
        self.menuHelp.setObjectName("menuHelp")

        for name in ("menuFile", "menuEdit", "menuView", "menuHelp"):
            self.export_widget(name, getattr(self, name))

        # File Menu Actions
        file_actions = [
            ("actionLoad_SavedWorkspace", "Load Saved Workspace"),
            ("actionSave_CurrentWorkspace", "Save Current Workspace"),
            ("actionReset_to_LastWorkspace", "Reset to Last Workspace"),
            ("actionSave_LastWorkspace", "Save Last Workspace"),
            ("actionOpen_Videos_Folder", "Load Target Videos/Images Folder"),
            ("actionOpen_Video_Files", "Load Target Video/Image Files"),
            ("actionLoad_Source_Images_Folder", "Load Source Images Folder"),
            ("actionLoad_Source_Image_Files", "Load Source Image Files"),
            ("actionLoad_Embeddings", "Load Embeddings"),
            ("actionSave_Embeddings", "Save Embeddings"),
            ("actionSave_Embeddings_As", "Save Embeddings As"),
            ("actionQuit_WithoutSaving", "Quit without Saving"),
            ("actionQuit", "Save and Quit"),
        ]
        for attr_name, label in file_actions:
            act = QtGui.QAction(label, self)
            self.menuFile.addAction(act)
            self.export_widget(attr_name, act)

        # View Menu Base Actions
        self.actionView_Fullscreen_F11 = QtGui.QAction("View Fullscreen (F11)", self)
        self.actionTheatre_Overlay_Mode = QtGui.QAction(
            "Auto-Hide Overlay Mode (T)", self
        )
        self.actionTheatre_Overlay_Mode.setCheckable(True)
        self.export_widget("actionView_Fullscreen_F11", self.actionView_Fullscreen_F11)
        self.export_widget(
            "actionTheatre_Overlay_Mode", self.actionTheatre_Overlay_Mode
        )

        self.menuView.addAction(self.actionView_Fullscreen_F11)
        self.menuView.addAction(self.actionTheatre_Overlay_Mode)
        self.menuView.addSeparator()

        # Help Menu Actions
        self.actionView_Help_Shortcuts = QtGui.QAction("View Shortcuts", self)
        self.actionView_Help_Presets = QtGui.QAction("View Presets", self)
        self.export_widget("actionView_Help_Shortcuts", self.actionView_Help_Shortcuts)
        self.export_widget("actionView_Help_Presets", self.actionView_Help_Presets)
        self.menuHelp.addAction(self.actionView_Help_Shortcuts)
        self.menuHelp.addAction(self.actionView_Help_Presets)

        self.addAction(self.menuFile.menuAction())
        self.addAction(self.menuEdit.menuAction())
        self.addAction(self.menuView.menuAction())
        self.addAction(self.menuHelp.menuAction())

    def finalize_view_menu(
        self,
        panels: List[QtWidgets.QDockWidget],
        set_all_visible_cb: Callable[[bool], None],
        main_window: QtWidgets.QMainWindow,
    ) -> None:
        """Populates dock visibility toggles and theme selector after panels are registered."""
        for panel in panels:
            self.menuView.addAction(panel.toggleViewAction())

        self.menuView.addSeparator()

        self.actionShow_All_Panels = QtGui.QAction("Show All Panels", main_window)
        self.actionShow_All_Panels.triggered.connect(lambda: set_all_visible_cb(True))
        self.actionHide_All_Panels = QtGui.QAction("Hide All Panels", main_window)
        self.actionHide_All_Panels.triggered.connect(lambda: set_all_visible_cb(False))
        self.export_widget("actionShow_All_Panels", self.actionShow_All_Panels)
        self.export_widget("actionHide_All_Panels", self.actionHide_All_Panels)

        self.menuView.addAction(self.actionShow_All_Panels)
        self.menuView.addAction(self.actionHide_All_Panels)
        self.menuView.addSeparator()

        self.menuTheme = ThemeManager.instance().populate_theme_menu(
            self.menuView, main_window
        )
        self.export_widget("menuTheme", self.menuTheme)