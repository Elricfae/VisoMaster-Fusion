# app/ui/widgets/panels/base_panel.py
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QRect, Qt


class DrawerZone(str, Enum):
    """Target drawer zones managed by the Modular Window Canvas."""

    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"

    @property
    def qt_dock_area(self) -> Qt.DockWidgetArea:
        mapping = {
            DrawerZone.TOP: Qt.DockWidgetArea.TopDockWidgetArea,
            DrawerZone.BOTTOM: Qt.DockWidgetArea.BottomDockWidgetArea,
            DrawerZone.LEFT: Qt.DockWidgetArea.LeftDockWidgetArea,
            DrawerZone.RIGHT: Qt.DockWidgetArea.RightDockWidgetArea,
        }
        return mapping[self]


class PanelDockState(str, Enum):
    """Explicit runtime docking state for a modular panel."""

    DOCKED = "docked"
    FLOATING = "floating"
    HIDDEN = "hidden"


class BaseDockPanel(QtWidgets.QDockWidget):
    """Abstract base class for all modular dock panels in VisoMaster."""

    panel_id: str = "base_panel"
    panel_title: str = "Panel"
    default_zone: DrawerZone = DrawerZone.LEFT

    dock_state_changed = QtCore.Signal(object, str)  # (self, PanelDockState)

    def __init__(
        self,
        title: Optional[str] = None,
        object_name: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(title or self.panel_title, parent)
        self.setObjectName(object_name or self.panel_id)
        self._exported_widgets: Dict[str, QtCore.QObject] = {}

        # Track current drawer zone and floating window geometry
        self.current_zone: DrawerZone = self.default_zone
        self._saved_floating_rect: Optional[QRect] = None

        # Only allow docking in the 4 widget zones (Top, Bottom, Left, Right)
        self.setAllowedAreas(
            Qt.DockWidgetArea.TopDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self.container = QtWidgets.QWidget(self)
        self.container.setObjectName(f"{self.objectName()}_container")
        self.setWidget(self.container)

        self.topLevelChanged.connect(self._on_top_level_changed)
        self._build_ui()

    def _build_ui(self) -> None:
        """Override in subclasses to populate `self.container`."""
        raise NotImplementedError

    @property
    def dock_state(self) -> PanelDockState:
        """Returns whether the module is currently DOCKED, FLOATING, or HIDDEN."""
        if not self.toggleViewAction().isChecked():
            return PanelDockState.HIDDEN
        if self.isFloating():
            return PanelDockState.FLOATING
        return PanelDockState.DOCKED

    def _on_top_level_changed(self, is_floating: bool) -> None:
        if is_floating:
            self._saved_floating_rect = self.geometry()
            self.dock_state_changed.emit(self, PanelDockState.FLOATING.value)
        else:
            self.dock_state_changed.emit(self, PanelDockState.DOCKED.value)

    def snapshot_floating_geometry(self) -> None:
        """Saves current floating window geometry if undocked."""
        if self.isFloating():
            self._saved_floating_rect = self.geometry()

    def restore_floating_geometry(self) -> None:
        """Restores saved floating window geometry so undocked panels never jump."""
        if self._saved_floating_rect is not None and not self._saved_floating_rect.isEmpty():
            self.setGeometry(self._saved_floating_rect)

    def export_widget(self, name: str, widget: QtCore.QObject) -> None:
        """Registers an internal widget so it can be bound to the host MainWindow."""
        setattr(self, name, widget)
        self._exported_widgets[name] = widget

    def bind_to_host(self, host: QtWidgets.QMainWindow) -> None:
        """Exposes exported child widgets on `host` for controller compatibility."""
        for attr_name, widget in self._exported_widgets.items():
            setattr(host, attr_name, widget)