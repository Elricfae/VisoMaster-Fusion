# app/ui/widgets/panels/faces_embeddings_panel.py
from __future__ import annotations

from typing import Optional
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from app.ui.core.asset_manager import AssetManager
from app.ui.widgets.panels.base_panel import BaseDockPanel, DrawerZone


class FacesEmbeddingsPanel(BaseDockPanel):
    """Modular dock panel for detected target faces, embeddings management, and batch actions."""

    panel_id = "facesDockWidget"
    panel_title = "Detected Faces & Embeddings"
    default_zone = DrawerZone.BOTTOM

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)

    def _build_ui(self) -> None:
        self.container.setObjectName("facesDockContainer")
        faces_vbox = QtWidgets.QVBoxLayout(self.container)
        faces_vbox.setContentsMargins(8, 4, 8, 4)
        faces_vbox.setSpacing(4)

        self.facesPanelGroupBox = QtWidgets.QGroupBox(self.container)
        self.facesPanelGroupBox.setObjectName("facesPanelGroupBox")
        self.facesPanelGroupBox.setMinimumHeight(150)

        grid = QtWidgets.QGridLayout(self.facesPanelGroupBox)
        grid.setObjectName("gridLayout_2")

        # Column 0: Face action buttons
        self.facesButtonsWidget = QtWidgets.QWidget(self.facesPanelGroupBox)
        self.facesButtonsWidget.setObjectName("facesButtonsWidget")
        btn_layout = QtWidgets.QVBoxLayout(self.facesButtonsWidget)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(0)

        self.findTargetFacesButton = QtWidgets.QPushButton(
            "Find Faces", self.facesButtonsWidget
        )
        self.findTargetFacesButton.setObjectName("findTargetFacesButton")
        self.findTargetFacesButton.setMaximumWidth(100)
        self.findTargetFacesButton.setFlat(True)

        self.clearTargetFacesButton = QtWidgets.QPushButton(
            "Clear Faces", self.facesButtonsWidget
        )
        self.clearTargetFacesButton.setObjectName("clearTargetFacesButton")
        self.clearTargetFacesButton.setFlat(True)

        self.swapfacesButton = QtWidgets.QPushButton(
            "Swap Faces", self.facesButtonsWidget
        )
        self.swapfacesButton.setObjectName("swapfacesButton")
        self.swapfacesButton.setCheckable(True)
        self.swapfacesButton.setFlat(True)

        self.editFacesButton = QtWidgets.QPushButton(
            "Edit Faces", self.facesButtonsWidget
        )
        self.editFacesButton.setObjectName("editFacesButton")
        self.editFacesButton.setCheckable(True)
        self.editFacesButton.setFlat(True)

        for btn in (
            self.findTargetFacesButton,
            self.clearTargetFacesButton,
            self.swapfacesButton,
            self.editFacesButton,
        ):
            btn_layout.addWidget(btn)

        grid.addWidget(self.facesButtonsWidget, 1, 0, 1, 1)

        # Column 1: Target Faces List
        self.targetFacesList = QtWidgets.QListWidget(self.facesPanelGroupBox)
        self.targetFacesList.setObjectName("targetFacesList")
        self.targetFacesList.setAutoScroll(False)
        self.targetFacesList.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.targetFacesList.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.targetFacesList.setMovement(QtWidgets.QListView.Movement.Static)
        self.targetFacesList.setLayoutMode(QtWidgets.QListView.LayoutMode.Batched)
        grid.addWidget(self.targetFacesList, 1, 1, 1, 1)

        # Column 2: Embeddings search, file I/O, and list
        embed_header_row = QtWidgets.QHBoxLayout()
        self.saveEmbeddingAsButton = QtWidgets.QPushButton(self.facesPanelGroupBox)
        self.saveEmbeddingAsButton.setObjectName("saveEmbeddingAsButton")
        self.saveEmbeddingAsButton.setToolTip("Save Embedding As...")
        self.saveEmbeddingAsButton.setIcon(AssetManager.get_icon("save_file_as.png"))
        self.saveEmbeddingAsButton.setFlat(True)

        self.saveEmbeddingButton = QtWidgets.QPushButton(self.facesPanelGroupBox)
        self.saveEmbeddingButton.setObjectName("saveEmbeddingButton")
        self.saveEmbeddingButton.setToolTip("Save Embedding")
        self.saveEmbeddingButton.setIcon(AssetManager.get_icon("save_file.png"))
        self.saveEmbeddingButton.setFlat(True)

        self.openEmbeddingButton = QtWidgets.QPushButton(self.facesPanelGroupBox)
        self.openEmbeddingButton.setObjectName("openEmbeddingButton")
        self.openEmbeddingButton.setToolTip("Open Embedding File")
        self.openEmbeddingButton.setIcon(AssetManager.get_icon("open_file.png"))
        self.openEmbeddingButton.setFlat(True)

        self.openEditorButton = QtWidgets.QPushButton(
            " Embedding Editor", self.facesPanelGroupBox
        )
        self.openEditorButton.setObjectName("openEditorButton")
        self.openEditorButton.setFlat(True)

        self.inputEmbeddingsSearchBox = QtWidgets.QLineEdit(self.facesPanelGroupBox)
        self.inputEmbeddingsSearchBox.setObjectName("inputEmbeddingsSearchBox")
        self.inputEmbeddingsSearchBox.setPlaceholderText("Search Embeddings")
        self.inputEmbeddingsSearchBox.setClearButtonEnabled(True)

        embed_header_row.addWidget(self.inputEmbeddingsSearchBox)
        embed_header_row.addWidget(self.openEditorButton)
        embed_header_row.addWidget(self.openEmbeddingButton)
        embed_header_row.addWidget(self.saveEmbeddingButton)
        embed_header_row.addWidget(self.saveEmbeddingAsButton)
        grid.addLayout(embed_header_row, 0, 2, 1, 1)

        self.inputEmbeddingsList = QtWidgets.QListWidget(self.facesPanelGroupBox)
        self.inputEmbeddingsList.setObjectName("inputEmbeddingsList")
        self.inputEmbeddingsList.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.inputEmbeddingsList.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.inputEmbeddingsList.setAutoScroll(False)
        self.inputEmbeddingsList.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.inputEmbeddingsList.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.inputEmbeddingsList.setMovement(QtWidgets.QListView.Movement.Static)
        self.inputEmbeddingsList.setLayoutMode(QtWidgets.QListView.LayoutMode.Batched)
        grid.addWidget(self.inputEmbeddingsList, 1, 2, 1, 1)

        # Column 3: Save Image / Batch Process buttons
        self.verticalWidget = QtWidgets.QWidget(self.facesPanelGroupBox)
        self.verticalWidget.setObjectName("verticalWidget")
        batch_layout = QtWidgets.QVBoxLayout(self.verticalWidget)
        batch_layout.setContentsMargins(0, 0, 0, 0)

        self.saveImageButton = QtWidgets.QPushButton("Save Image", self.verticalWidget)
        self.saveImageButton.setObjectName("saveImageButton")
        self.saveImageButton.setFlat(True)

        self.batchImageButton = QtWidgets.QPushButton(
            "Batch Process", self.verticalWidget
        )
        self.batchImageButton.setObjectName("batchImageButton")
        self.batchImageButton.setFlat(True)

        self.batchallImageButton = QtWidgets.QPushButton(
            "Batch Process AllImages", self.verticalWidget
        )
        self.batchallImageButton.setObjectName("batchallImageButton")
        self.batchallImageButton.setFlat(True)

        batch_layout.addWidget(self.saveImageButton)
        batch_layout.addWidget(self.batchImageButton)
        batch_layout.addWidget(self.batchallImageButton)
        grid.addWidget(self.verticalWidget, 1, 3, 1, 1)

        faces_vbox.addWidget(self.facesPanelGroupBox, stretch=1)

        for name in (
            "facesPanelGroupBox",
            "findTargetFacesButton",
            "clearTargetFacesButton",
            "swapfacesButton",
            "editFacesButton",
            "targetFacesList",
            "saveEmbeddingAsButton",
            "saveEmbeddingButton",
            "openEmbeddingButton",
            "openEditorButton",
            "inputEmbeddingsSearchBox",
            "inputEmbeddingsList",
            "saveImageButton",
            "batchImageButton",
            "batchallImageButton",
        ):
            self.export_widget(name, getattr(self, name))