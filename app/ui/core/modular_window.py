# app/ui/core/modular_window.py
from __future__ import annotations

from typing import Dict, List, Optional
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from app.ui.core.asset_manager import AssetManager
from app.ui.core.overlay_manager import ViewportOverlayController
from app.ui.widgets.panels.base_panel import BaseDockPanel, DrawerZone
from app.ui.widgets.panels.menu_bar_panel import MenuBarPanel
from app.ui.widgets.panels.viewport_panel import ViewportPanel


class Ui_MainWindow:
    """Pure modular canvas handler and drawer registry for VisoMaster.

    Manages the central viewport, dedicated top menu bar, 4 widget drawer zones
    (Top, Bottom, Left, Right), and the ViewportOverlayController.
    """

    def init_canvas(self, MainWindow: QtWidgets.QMainWindow) -> None:
        AssetManager.initialize()

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1736, 989)
        MainWindow.setWindowTitle("VisoMaster - Fusion Edition")
        MainWindow.setWindowIcon(AssetManager.get_icon("visomaster_small.png"))

        MainWindow.setDockOptions(
            QtWidgets.QMainWindow.DockOption.AnimatedDocks
            | QtWidgets.QMainWindow.DockOption.AllowNestedDocks
            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
            | QtWidgets.QMainWindow.DockOption.GroupedDragging
        )

        # Corner priorities: Left & Right span vertically from top to bottom;
        # Top and Bottom sit cleanly between Left & Right.
        MainWindow.setCorner(
            Qt.Corner.TopLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea
        )
        MainWindow.setCorner(
            Qt.Corner.TopRightCorner, Qt.DockWidgetArea.RightDockWidgetArea
        )
        MainWindow.setCorner(
            Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea
        )
        MainWindow.setCorner(
            Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea
        )

        self._menu_bar: Optional[MenuBarPanel] = None
        self._drawer_registry: Dict[DrawerZone, List[BaseDockPanel]] = {
            zone: [] for zone in DrawerZone
        }
        self._all_panels: List[BaseDockPanel] = []

        # Central Viewport Canvas
        self.viewport_panel = ViewportPanel(MainWindow)
        MainWindow.setCentralWidget(self.viewport_panel)
        self.viewport_panel.bind_to_host(MainWindow)

    def register_menu_bar(
        self, MainWindow: QtWidgets.QMainWindow, menu_bar: MenuBarPanel
    ) -> MenuBarPanel:
        """Installs the top MenuBarPanel in MainWindow's dedicated menuBar slot."""
        self._menu_bar = menu_bar
        MainWindow.setMenuBar(menu_bar)
        menu_bar.bind_to_host(MainWindow)
        return menu_bar

    def register_panel(
        self,
        MainWindow: QtWidgets.QMainWindow,
        panel: BaseDockPanel,
        zone: Optional[DrawerZone] = None,
        split_orientation: Qt.Orientation = Qt.Orientation.Vertical,
        tabify: bool = False,
    ) -> BaseDockPanel:
        """Registers a modular panel into a target drawer zone and binds its exports."""
        target_zone = zone or panel.default_zone
        panel.current_zone = target_zone
        existing_in_zone = self._drawer_registry[target_zone]

        MainWindow.addDockWidget(target_zone.qt_dock_area, panel)

        if existing_in_zone:
            prev_panel = existing_in_zone[-1]
            if tabify:
                MainWindow.tabifyDockWidget(prev_panel, panel)
            else:
                MainWindow.splitDockWidget(prev_panel, panel, split_orientation)

        existing_in_zone.append(panel)
        self._all_panels.append(panel)
        setattr(MainWindow, panel.objectName(), panel)
        panel.bind_to_host(MainWindow)
        return panel

    def set_all_panels_visible(self, visible: bool) -> None:
        """Shows or hides all registered workspace panels."""
        for panel in self._all_panels:
            panel.setVisible(visible)

    def init_overlay_controller(
        self,
        MainWindow: QtWidgets.QMainWindow,
        trigger_ratio: float = 0.10,
    ) -> ViewportOverlayController:
        """Initializes the dynamic auto-hide drawer controller."""
        self.overlayController = ViewportOverlayController(
            main_window=MainWindow,
            host_widget=self.viewport_panel,
            menu_bar=self._menu_bar,
            panels=self._all_panels,
            trigger_ratio=trigger_ratio,
        )
        return self.overlayController