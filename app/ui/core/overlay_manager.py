# app/ui/core/overlay_manager.py
from __future__ import annotations

from typing import Dict, List, Optional
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import (
    QByteArray,
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
)

from app.ui.widgets.panels.base_panel import BaseDockPanel


class EdgeDrawer(QtWidgets.QFrame):
    """Resizable, translucent overlay drawer with its own internal QMainWindow dock area."""

    size_changed = QtCore.Signal()
    GRIP_WIDTH = 6

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        edge: str,
        drawer_size: int = 360,
        min_size: int = 36,
        max_size: int = 900,
        use_dock_host: bool = True,
    ) -> None:
        super().__init__(parent)
        self.edge = edge  # "top_menu", "top", "bottom", "left", "right"
        self.drawer_size = drawer_size
        self.min_size = min_size
        self.max_size = max_size
        self.is_revealed = False

        self._resizing = False
        self._drag_start_pos = QPoint()
        self._drag_start_size = drawer_size

        self.setObjectName(f"EdgeDrawer_{edge}")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMouseTracking(True)

        border_rule = {
            "top_menu": "border-bottom: 1px solid rgba(100, 150, 255, 90);",
            "top": "border-bottom: 2px solid rgba(100, 150, 255, 90);",
            "bottom": "border-top: 2px solid rgba(100, 150, 255, 90);",
            "left": "border-right: 2px solid rgba(100, 150, 255, 90);",
            "right": "border-left: 2px solid rgba(100, 150, 255, 90);",
        }[edge]
        self.setStyleSheet(
            f"""
            QFrame#EdgeDrawer_{edge} {{
                background-color: rgba(20, 22, 26, 235);
                {border_rule}
            }}
            """
        )

        self.outer_layout = QtWidgets.QVBoxLayout(self)
        pad_l = self.GRIP_WIDTH if edge == "right" else 2
        pad_r = self.GRIP_WIDTH if edge == "left" else 2
        pad_t = self.GRIP_WIDTH if edge == "bottom" else 2
        pad_b = self.GRIP_WIDTH if edge == "top" else 2
        self.outer_layout.setContentsMargins(pad_l, pad_t, pad_r, pad_b)
        self.outer_layout.setSpacing(0)

        self.dock_host: Optional[QtWidgets.QMainWindow] = None
        if use_dock_host:
            self.dock_host = QtWidgets.QMainWindow(self)
            self.dock_host.setWindowFlags(Qt.WindowType.Widget)
            self.dock_host.setDockOptions(
                QtWidgets.QMainWindow.DockOption.AnimatedDocks
                | QtWidgets.QMainWindow.DockOption.AllowNestedDocks
                | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
                | QtWidgets.QMainWindow.DockOption.GroupedDragging
            )
            self.dock_host.setStyleSheet("QMainWindow { background: transparent; }")
            self.outer_layout.addWidget(self.dock_host)

        self.anim = QPropertyAnimation(self, b"geometry", self)
        self.anim.setDuration(200)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    @property
    def require_dock_host(self) -> QtWidgets.QMainWindow:
        """Returns the internal QMainWindow dock host, raising RuntimeError if unconfigured."""
        if self.dock_host is None:
            raise RuntimeError(
                f"EdgeDrawer '{self.edge}' does not have an internal QMainWindow dock host configured."
            )
        return self.dock_host

    def _is_on_resize_grip(self, pos: QPoint) -> bool:
        if self.edge == "top_menu":
            return False
        if self.edge == "top":
            return pos.y() >= self.height() - self.GRIP_WIDTH
        if self.edge == "bottom":
            return pos.y() <= self.GRIP_WIDTH
        if self.edge == "left":
            return pos.x() >= self.width() - self.GRIP_WIDTH
        if self.edge == "right":
            return pos.x() <= self.GRIP_WIDTH
        return False

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        pos = event.position().toPoint()
        if self._resizing:
            delta = event.globalPosition().toPoint() - self._drag_start_pos
            if self.edge == "top":
                new_size = self._drag_start_size + delta.y()
            elif self.edge == "bottom":
                new_size = self._drag_start_size - delta.y()
            elif self.edge == "left":
                new_size = self._drag_start_size + delta.x()
            else:
                new_size = self._drag_start_size - delta.x()

            self.drawer_size = max(self.min_size, min(self.max_size, new_size))
            self.size_changed.emit()
            event.accept()
            return

        if self._is_on_resize_grip(pos):
            cursor = (
                Qt.CursorShape.SizeVerCursor
                if self.edge in ("top", "bottom")
                else Qt.CursorShape.SizeHorCursor
            )
            self.setCursor(cursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._is_on_resize_grip(event.position().toPoint())
        ):
            self._resizing = True
            self._drag_start_pos = event.globalPosition().toPoint()
            self._drag_start_size = self.drawer_size
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._resizing:
            self._resizing = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.unsetCursor()
        super().leaveEvent(event)

    def compute_rects(
        self,
        parent_rect: QRect,
        left_inset: int = 0,
        right_inset: int = 0,
        top_offset: int = 0,
    ) -> tuple[QRect, QRect]:
        w, h = parent_rect.width(), parent_rect.height()
        is_vertical = self.edge in ("top_menu", "top", "bottom")
        s = min(
            self.drawer_size,
            max(self.min_size, (h if is_vertical else w) - 60),
        )
        avail_w = max(100, w - left_inset - right_inset)

        if self.edge == "top_menu":
            hidden = QRect(0, -s, w, s)
            shown = QRect(0, 0, w, s)
        elif self.edge == "top":
            hidden = QRect(left_inset, -s, avail_w, s)
            shown = QRect(left_inset, top_offset, avail_w, s)
        elif self.edge == "bottom":
            hidden = QRect(left_inset, h, avail_w, s)
            shown = QRect(left_inset, h - s, avail_w, s)
        elif self.edge == "left":
            hidden = QRect(-s, top_offset, s, h - top_offset)
            shown = QRect(0, top_offset, s, h - top_offset)
        else:  # right
            hidden = QRect(w, top_offset, s, h - top_offset)
            shown = QRect(w - s, top_offset, s, h - top_offset)
        return hidden, shown

    def sync_to_parent(
        self,
        parent_rect: QRect,
        left_inset: int = 0,
        right_inset: int = 0,
        top_offset: int = 0,
    ) -> None:
        self.anim.stop()
        hidden, shown = self.compute_rects(
            parent_rect, left_inset, right_inset, top_offset
        )
        self.setGeometry(shown if self.is_revealed else hidden)

    def slide_in(
        self,
        parent_rect: QRect,
        left_inset: int = 0,
        right_inset: int = 0,
        top_offset: int = 0,
    ) -> None:
        _, shown = self.compute_rects(parent_rect, left_inset, right_inset, top_offset)
        if self.is_revealed and self.anim.endValue() == shown:
            return
        self.is_revealed = True
        self.show()
        self.raise_()
        self.unsetCursor()
        self.anim.stop()
        self.anim.setStartValue(self.geometry())
        self.anim.setEndValue(shown)
        self.anim.start()

    def slide_out(
        self,
        parent_rect: QRect,
        left_inset: int = 0,
        right_inset: int = 0,
        top_offset: int = 0,
    ) -> None:
        if not self.is_revealed:
            return
        self.is_revealed = False
        hidden, _ = self.compute_rects(
            parent_rect, left_inset, right_inset, top_offset
        )
        self.unsetCursor()
        self.anim.stop()
        self.anim.setStartValue(self.geometry())
        self.anim.setEndValue(hidden)
        self.anim.start()


class ViewportOverlayController(QtCore.QObject):
    """Orchestrates the dedicated Menu Bar drawer + 4-sided widget drawers (Top, Bottom, Left, Right).

    Preserves floating (undocked) panels across mode switches and restores exact
    QMainWindow dock state so undocked/moved panels never corrupt the layout.
    """

    def __init__(
        self,
        main_window: QtWidgets.QMainWindow,
        host_widget: QtWidgets.QWidget,
        menu_bar: Optional[QtWidgets.QMenuBar] = None,
        panels: Optional[List[BaseDockPanel]] = None,
        trigger_ratio: float = 0.10,
    ) -> None:
        super().__init__(main_window)
        self.main_window = main_window
        self.host = host_widget
        self.menu_bar = menu_bar
        self.panels: List[BaseDockPanel] = panels or []
        self.trigger_ratio = trigger_ratio
        self.overlay_enabled = False

        # Snapshot of QMainWindow dock layout before entering Overlay Mode
        self._saved_static_state: Optional[QByteArray] = None

        # Active docks currently inside each overlay drawer
        self._active_drawer_docks: Dict[str, List[BaseDockPanel]] = {
            "top": [],
            "bottom": [],
            "left": [],
            "right": [],
        }

        # Top Menu drawer uses a direct widget layout (no dock_host) so modules can never dock into it
        self.top_menu_drawer = EdgeDrawer(
            self.host,
            "top_menu",
            drawer_size=40,
            min_size=30,
            max_size=44,
            use_dock_host=False,
        )
        self.top_drawer = EdgeDrawer(
            self.host, "top", drawer_size=140, min_size=60, max_size=450
        )
        self.bottom_drawer = EdgeDrawer(
            self.host, "bottom", drawer_size=330, min_size=100, max_size=600
        )
        self.left_drawer = EdgeDrawer(
            self.host, "left", drawer_size=360, min_size=240, max_size=750
        )
        self.right_drawer = EdgeDrawer(
            self.host, "right", drawer_size=400, min_size=260, max_size=800
        )

        self.drawers: Dict[str, EdgeDrawer] = {
            "top_menu": self.top_menu_drawer,
            "top": self.top_drawer,
            "bottom": self.bottom_drawer,
            "left": self.left_drawer,
            "right": self.right_drawer,
        }

        for drawer in self.drawers.values():
            drawer.size_changed.connect(self._sync_all)
            drawer.hide()

        self.host.installEventFilter(self)

        self._hover_timer = QTimer(self)
        self._hover_timer.setInterval(40)
        self._hover_timer.timeout.connect(self._check_cursor_proximity)
        self._hover_timer.start()

    @staticmethod
    def _attach_docks_stacked(
        target_window: Optional[QtWidgets.QMainWindow],
        area: Qt.DockWidgetArea,
        docks: List[BaseDockPanel],
        orientation: Qt.Orientation,
        visibility_map: Dict[BaseDockPanel, bool],
    ) -> None:
        """Adds panels to a QMainWindow dock area and stacks them sequentially."""
        if target_window is None or not docks:
            return

        first_dock = docks[0]
        first_dock.setFloating(False)
        target_window.addDockWidget(area, first_dock)
        first_dock.setVisible(visibility_map.get(first_dock, True))

        prev_dock = first_dock
        for dock in docks[1:]:
            dock.setFloating(False)
            target_window.addDockWidget(area, dock)
            target_window.splitDockWidget(prev_dock, dock, orientation)
            dock.setVisible(visibility_map.get(dock, True))
            prev_dock = dock

    def _resolve_panel_edge(self, panel: BaseDockPanel) -> str:
        """Determines which drawer edge a docked panel currently belongs to."""
        area = self.main_window.dockWidgetArea(panel)
        if area == Qt.DockWidgetArea.TopDockWidgetArea:
            return "top"
        if area == Qt.DockWidgetArea.BottomDockWidgetArea:
            return "bottom"
        if area == Qt.DockWidgetArea.RightDockWidgetArea:
            return "right"
        if area == Qt.DockWidgetArea.LeftDockWidgetArea:
            return "left"
        return panel.current_zone.value

    def set_overlay_mode(self, enabled: bool) -> None:
        if self.overlay_enabled == enabled:
            return
        self.overlay_enabled = enabled

        # 1. Temporarily disable AnimatedDocks so QMainWindowLayout collapses
        # dock areas synchronously instead of holding gap placeholders for animation.
        orig_dock_opts = self.main_window.dockOptions()
        self.main_window.setDockOptions(
            orig_dock_opts & ~QtWidgets.QMainWindow.DockOption.AnimatedDocks
        )

        # 2. Snapshot explicit visibility and floating state of every panel
        visibility_map: Dict[BaseDockPanel, bool] = {}
        floating_panels: List[BaseDockPanel] = []
        docked_panels: List[BaseDockPanel] = []

        for panel in self.panels:
            is_checked = panel.toggleViewAction().isChecked()
            visibility_map[panel] = is_checked
            if panel.isFloating():
                panel.snapshot_floating_geometry()
                floating_panels.append(panel)
            else:
                docked_panels.append(panel)

        docked_features = (
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        # Lock panels in place during Overlay Mode: disable both Movable and Floatable
        overlay_features = (
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        if enabled:
            # Save the exact static QMainWindow dock layout before modifying anything
            self._saved_static_state = self.main_window.saveState()

            # Group ONLY docked (non-floating) panels by their current dock area
            for key in self._active_drawer_docks:
                self._active_drawer_docks[key].clear()

            for panel in docked_panels:
                edge = self._resolve_panel_edge(panel)
                self._active_drawer_docks[edge].append(panel)

            # Trick: Calling restoreState() right before removing docks flushes any
            # lingering manual-drag placeholder items in QDockAreaLayout (which is why
            # switching static -> overlay a 2nd time worked previously!).
            self.main_window.restoreState(self._saved_static_state)

            # Remove ALL panels (both docked and floating) from MainWindow's dock layout
            for panel in self.panels:
                orig_area = self.main_window.dockWidgetArea(panel)
                panel.setProperty("_saved_dock_area", orig_area)
                panel.hide()
                self.main_window.removeDockWidget(panel)

            # Move the dedicated Menu Bar into the top_menu_drawer safely
            if self.menu_bar is not None:
                self.menu_bar.setParent(self.top_menu_drawer)
                dummy_mb = QtWidgets.QMenuBar(self.main_window)
                dummy_mb.setFixedHeight(0)
                dummy_mb.hide()
                self.main_window.setMenuBar(dummy_mb)
                self.top_menu_drawer.outer_layout.addWidget(self.menu_bar)
                self.menu_bar.show()

            for drawer in self.drawers.values():
                drawer.is_revealed = False
                drawer.show()

            # Attach docked panels into their respective EdgeDrawer QMainWindow hosts
            for panel in docked_panels:
                panel.setFeatures(overlay_features)

            self._attach_docks_stacked(
                self.top_drawer.require_dock_host,
                Qt.DockWidgetArea.TopDockWidgetArea,
                self._active_drawer_docks["top"],
                Qt.Orientation.Horizontal,
                visibility_map,
            )
            self._attach_docks_stacked(
                self.left_drawer.require_dock_host,
                Qt.DockWidgetArea.LeftDockWidgetArea,
                self._active_drawer_docks["left"],
                Qt.Orientation.Vertical,
                visibility_map,
            )
            self._attach_docks_stacked(
                self.right_drawer.require_dock_host,
                Qt.DockWidgetArea.RightDockWidgetArea,
                self._active_drawer_docks["right"],
                Qt.Orientation.Vertical,
                visibility_map,
            )
            self._attach_docks_stacked(
                self.bottom_drawer.require_dock_host,
                Qt.DockWidgetArea.BottomDockWidgetArea,
                self._active_drawer_docks["bottom"],
                Qt.Orientation.Vertical,
                visibility_map,
            )

            # Keep floating panels as standalone top-level Tool windows
            # and prevent them from snapping into any dock area
            for panel in floating_panels:
                panel.setFeatures(overlay_features)
                panel.setAllowedAreas(Qt.DockWidgetArea.NoDockWidgetArea)
                panel.setParent(
                    self.main_window,
                    Qt.WindowType.Tool,
                )
                panel.restore_floating_geometry()
                panel.setVisible(visibility_map.get(panel, True))
                panel.raise_()

            # Force QMainWindow layout to recalculate and expand centralWidget immediately
            if self.main_window.layout() is not None:
                self.main_window.layout().invalidate()
                self.main_window.layout().activate()
            self.main_window.centralWidget().updateGeometry()
            QtWidgets.QApplication.sendPostedEvents(
                self.main_window, QtCore.QEvent.Type.LayoutRequest
            )

            self._sync_all()
            QTimer.singleShot(0, self._sync_all)

        else:
            # 1. Remove docked panels from the EdgeDrawer dock_hosts first
            for edge, edge_panels in self._active_drawer_docks.items():
                drawer_host = self.drawers[edge].require_dock_host
                for panel in edge_panels:
                    panel.hide()
                    drawer_host.removeDockWidget(panel)
                    panel.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
                    panel.setFeatures(docked_features)

            for drawer in self.drawers.values():
                drawer.anim.stop()
                drawer.is_revealed = False
                drawer.hide()

            # 2. Restore Menu Bar to MainWindow's dedicated menu bar slot safely
            if self.menu_bar is not None:
                self.top_menu_drawer.outer_layout.removeWidget(self.menu_bar)
                self.menu_bar.setParent(self.main_window)
                self.main_window.setMenuBar(self.menu_bar)
                self.menu_bar.show()

            # 3. Restore static layout corner priorities
            self.main_window.setCorner(
                Qt.Corner.TopLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea
            )
            self.main_window.setCorner(
                Qt.Corner.TopRightCorner, Qt.DockWidgetArea.RightDockWidgetArea
            )
            self.main_window.setCorner(
                Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea
            )
            self.main_window.setCorner(
                Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea
            )

            # 4. Re-attach panels back to MainWindow
            area_map = {
                "top": (Qt.DockWidgetArea.TopDockWidgetArea, Qt.Orientation.Horizontal),
                "left": (Qt.DockWidgetArea.LeftDockWidgetArea, Qt.Orientation.Vertical),
                "right": (Qt.DockWidgetArea.RightDockWidgetArea, Qt.Orientation.Vertical),
                "bottom": (Qt.DockWidgetArea.BottomDockWidgetArea, Qt.Orientation.Vertical),
            }
            for edge, (area, orient) in area_map.items():
                edge_panels = self._active_drawer_docks[edge]
                self._attach_docks_stacked(
                    self.main_window, area, edge_panels, orient, visibility_map
                )

            for panel in floating_panels:
                saved_area = panel.property("_saved_dock_area")
                if not saved_area or saved_area == Qt.DockWidgetArea.NoDockWidgetArea:
                    saved_area = panel.current_zone.qt_dock_area
                panel.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
                panel.setFeatures(docked_features)
                self.main_window.addDockWidget(saved_area, panel)
                panel.setFloating(True)

            # 5. Restore exact QMainWindow static state (splitter sizes, tabs, dock positions)
            if self._saved_static_state is not None:
                self.main_window.restoreState(self._saved_static_state)

            # 6. Re-apply current visibility & floating geometry for any undocked panels
            for panel in self.panels:
                panel.setFeatures(docked_features)
                if panel in floating_panels:
                    panel.setFloating(True)
                    panel.restore_floating_geometry()
                panel.setVisible(visibility_map.get(panel, True))

        # Restore original DockOptions (including AnimatedDocks)
        self.main_window.setDockOptions(orig_dock_opts)

    def _get_insets(self) -> tuple[int, int, int]:
        l_in = self.left_drawer.drawer_size if self.left_drawer.is_revealed else 0
        r_in = self.right_drawer.drawer_size if self.right_drawer.is_revealed else 0
        t_off = (
            self.top_menu_drawer.drawer_size
            if self.top_menu_drawer.is_revealed
            else 0
        )
        return l_in, r_in, t_off

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.host and event.type() == QtCore.QEvent.Type.Resize:
            if self.overlay_enabled:
                self._sync_all()
        return super().eventFilter(watched, event)

    def _sync_all(self) -> None:
        if not self.overlay_enabled:
            return
        rect = self.host.rect()
        l_in, r_in, t_off = self._get_insets()
        self.top_menu_drawer.sync_to_parent(rect)
        self.top_drawer.sync_to_parent(rect, l_in, r_in, t_off)
        self.bottom_drawer.sync_to_parent(rect, l_in, r_in)
        self.left_drawer.sync_to_parent(rect, top_offset=t_off)
        self.right_drawer.sync_to_parent(rect, top_offset=t_off)

    def _is_busy(self, drawer: EdgeDrawer) -> bool:
        if drawer._resizing:
            return True
        if QtGui.QGuiApplication.mouseButtons() != Qt.MouseButton.NoButton:
            return drawer.is_revealed
        focus_widget = QtWidgets.QApplication.focusWidget()
        if isinstance(focus_widget, QtWidgets.QLineEdit) and drawer.isAncestorOf(
            focus_widget
        ):
            return True
        return False

    def _get_active_drawer_under_cursor(
        self, local_pos: QPoint
    ) -> Optional[EdgeDrawer]:
        for drawer in self.drawers.values():
            if drawer.is_revealed and drawer.geometry().contains(local_pos):
                return drawer
        return None

    def _check_cursor_proximity(self) -> None:
        if not self.overlay_enabled or not self.host.isVisible():
            return

        window = self.host.window()
        if not window.isActiveWindow():
            return

        if QtWidgets.QApplication.activePopupWidget() is not None:
            return

        global_pos = QtGui.QCursor.pos()
        local_pos: QPoint = self.host.mapFromGlobal(global_pos)
        rect: QRect = self.host.rect()
        l_in, r_in, t_off = self._get_insets()

        if not rect.contains(local_pos):
            for drawer in self.drawers.values():
                if not self._is_busy(drawer):
                    drawer.slide_out(rect, l_in, r_in, t_off)
            return

        active_drawer = self._get_active_drawer_under_cursor(local_pos)

        # 1. Link the menu bar and the top drawer as a single unified hover zone
        is_top_zone_active = active_drawer in (self.top_menu_drawer, self.top_drawer)

        for drawer in self.drawers.values():
            if drawer.is_revealed and drawer is not active_drawer:
                # Keep both top drawers locked open if the cursor is in either of them
                if is_top_zone_active and drawer in (self.top_menu_drawer, self.top_drawer):
                    continue

                if not drawer.geometry().adjusted(-10, -10, 10, 10).contains(
                    local_pos
                ) and not self._is_busy(drawer):
                    drawer.slide_out(rect, l_in, r_in, t_off)

        if active_drawer is not None:
            return

        margin_x = int(rect.width() * self.trigger_ratio)
        margin_y = int(rect.height() * self.trigger_ratio)

        # Top Edge Trigger: Reveals Top Menu Bar AND Top Widget Drawer (if populated)
        if local_pos.y() <= margin_y:
            if self.menu_bar is not None:
                self.top_menu_drawer.slide_in(rect)
            if any(not d.isHidden() for d in self._active_drawer_docks["top"]):
                menu_h = (
                    self.top_menu_drawer.drawer_size
                    if self.menu_bar is not None
                    else 0
                )
                self.top_drawer.slide_in(rect, l_in, r_in, top_offset=menu_h)
            return

        # Bottom Edge Trigger
        if local_pos.y() >= rect.height() - margin_y:
            if any(not d.isHidden() for d in self._active_drawer_docks["bottom"]):
                self.bottom_drawer.slide_in(rect, l_in, r_in)
            return

        # Left Edge Trigger
        if local_pos.x() <= margin_x:
            if any(not d.isHidden() for d in self._active_drawer_docks["left"]):
                self.left_drawer.slide_in(rect, top_offset=t_off)
            return

        # Right Edge Trigger
        if local_pos.x() >= rect.width() - margin_x:
            if any(not d.isHidden() for d in self._active_drawer_docks["right"]):
                self.right_drawer.slide_in(rect, top_offset=t_off)
            return