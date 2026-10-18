# app/ui/widgets/panels/input_faces_panel.py
from __future__ import annotations

from typing import Optional
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from app.ui.widgets.panels.base_panel import BaseDockPanel, DrawerZone


class InputFacesPanel(BaseDockPanel):
    """Modular dock panel for selecting and searching source/input face images."""

    panel_id = "input_Faces_DockWidget"
    panel_title = "Input Faces"
    default_zone = DrawerZone.LEFT

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)

    def _build_ui(self) -> None:
        self.container.setObjectName("dockWidgetContents_Faces")
        vbox = QtWidgets.QVBoxLayout(self.container)
        vbox.setContentsMargins(4, 4, 4, 4)

        folder_row = QtWidgets.QHBoxLayout()
        self.buttonInputFacesPath = QtWidgets.QPushButton(
            "Select Folder", self.container
        )
        self.buttonInputFacesPath.setObjectName("buttonInputFacesPath")
        self.buttonInputFacesPath.setIconSize(QtCore.QSize(18, 18))
        self.buttonInputFacesPath.setFlat(True)

        self.inputFacesPathLineEdit = QtWidgets.QLineEdit(self.container)
        self.inputFacesPathLineEdit.setObjectName("inputFacesPathLineEdit")
        self.inputFacesPathLineEdit.setPlaceholderText("Select Face Images Folder")
        self.inputFacesPathLineEdit.setReadOnly(True)

        folder_row.addWidget(self.buttonInputFacesPath)
        folder_row.addWidget(self.inputFacesPathLineEdit)
        vbox.addLayout(folder_row)

        self.inputFacesSearchBox = QtWidgets.QLineEdit(self.container)
        self.inputFacesSearchBox.setObjectName("inputFacesSearchBox")
        self.inputFacesSearchBox.setPlaceholderText("Search Faces")
        self.inputFacesSearchBox.setClearButtonEnabled(True)
        vbox.addWidget(self.inputFacesSearchBox)

        self.inputFacesList = QtWidgets.QListWidget(self.container)
        self.inputFacesList.setObjectName("inputFacesList")
        self.inputFacesList.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.inputFacesList.setAutoScroll(False)
        self.inputFacesList.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.inputFacesList.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.inputFacesList.setMovement(QtWidgets.QListView.Movement.Static)
        self.inputFacesList.setLayoutMode(QtWidgets.QListView.LayoutMode.Batched)
        vbox.addWidget(self.inputFacesList)

        for name in (
            "buttonInputFacesPath",
            "inputFacesPathLineEdit",
            "inputFacesSearchBox",
            "inputFacesList",
        ):
            self.export_widget(name, getattr(self, name))