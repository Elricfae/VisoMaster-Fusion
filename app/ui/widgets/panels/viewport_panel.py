# app/ui/widgets/panels/viewport_panel.py
from __future__ import annotations

from typing import Optional
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt


class ViewportPanel(QtWidgets.QWidget):
    """Central video/image canvas viewport that holds main window focus."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("centralwidget")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)

        self.mediaLayout = QtWidgets.QWidget(self)
        self.mediaLayout.setObjectName("mediaLayout")
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        size_policy.setHorizontalStretch(2)
        size_policy.setVerticalStretch(2)
        self.mediaLayout.setSizePolicy(size_policy)

        self.verticalLayout_MediaControls = QtWidgets.QVBoxLayout(self.mediaLayout)
        self.verticalLayout_MediaControls.setObjectName("verticalLayout_MediaControls")
        self.verticalLayout_MediaControls.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_MediaControls.setSpacing(0)

        self.graphicsViewFrame = QtWidgets.QGraphicsView(self.mediaLayout)
        self.graphicsViewFrame.setObjectName("graphicsViewFrame")
        self.graphicsViewFrame.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.graphicsViewFrame.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.verticalLayout_MediaControls.addWidget(self.graphicsViewFrame)

        self.verticalLayout.addWidget(self.mediaLayout, stretch=1)

        self.scene = QtWidgets.QGraphicsScene(self)
        self.graphicsViewFrame.setScene(self.scene)

    def bind_to_host(self, host: QtWidgets.QMainWindow) -> None:
        host.centralwidget = self
        host.mediaLayout = self.mediaLayout
        host.verticalLayout_MediaControls = self.verticalLayout_MediaControls
        host.graphicsViewFrame = self.graphicsViewFrame
        host.scene = self.scene