# app/ui/widgets/panels/control_options_panel.py
from __future__ import annotations

from typing import Optional
from PySide6 import QtWidgets

from app.ui.widgets.panels.base_panel import BaseDockPanel, DrawerZone


class ControlOptionsPanel(BaseDockPanel):
    """Modular right-side inspector dock with VRAM monitor and parameter tabs."""

    panel_id = "controlOptionsDockWidget"
    panel_title = "Control Options"
    default_zone = DrawerZone.RIGHT

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)

    # app/ui/widgets/panels/control_options_panel.py

    def _build_ui(self) -> None:
        self.container.setObjectName("dockWidgetContents_2")
        grid = QtWidgets.QGridLayout(self.container)
        grid.setContentsMargins(2, 0, 2, 0)
        grid.setVerticalSpacing(0)

        # Parameter Tabs
        self.tabWidget = QtWidgets.QTabWidget(self.container)
        self.tabWidget.setObjectName("tabWidget")
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        size_policy.setHorizontalStretch(1)
        self.tabWidget.setSizePolicy(size_policy)

        self.swapWidgetsLayout = self.add_scrollable_tab("face_swap_tab", "Face Swap")
        self.faceEditorWidgetsLayout = self.add_scrollable_tab(
            "face_editor_tab", "Face Editor"
        )
        self.commonWidgetsLayout = self.add_scrollable_tab("common_tab", "Restorers")
        self.denoiserWidgetsLayout = self.add_scrollable_tab(
            "denoiser_tab", "Denoiser"
        )
        self._build_settings_tab()
        self._build_presets_tab()

        # Changed from row 1 to 0 since the VRAM bar is gone
        grid.addWidget(self.tabWidget, 0, 0, 1, 1)

        for name in (
            "tabWidget",
            "swapWidgetsLayout",
            "faceEditorWidgetsLayout",
            "commonWidgetsLayout",
            "denoiserWidgetsLayout",
            "settings_tab",
            "settingsWidgetsLayout",
            "outputFolderLineEdit",
            "outputFolderButton",
            "outputOpenButton",
            "presets_tab",
            "presetsList",
            "applyPresetButton",
            "savePresetButton",
            "controlPresetButton",
        ):
            self.export_widget(name, getattr(self, name))

    def add_scrollable_tab(
        self, tab_obj_name: str, tab_title: str
    ) -> QtWidgets.QVBoxLayout:
        """Creates and registers a scrollable parameter tab, returning its inner layout."""
        tab = QtWidgets.QWidget()
        tab.setObjectName(tab_obj_name)
        outer_layout = QtWidgets.QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 8, 0, 8)
        outer_layout.setSpacing(0)

        scroll_area = QtWidgets.QScrollArea(tab)
        scroll_area.setWidgetResizable(True)
        scroll_contents = QtWidgets.QWidget()
        content_vbox = QtWidgets.QVBoxLayout(scroll_contents)
        content_vbox.setContentsMargins(0, 0, 0, 0)
        content_vbox.setSpacing(0)

        scroll_area.setWidget(scroll_contents)
        outer_layout.addWidget(scroll_area)

        self.tabWidget.addTab(tab, tab_title)
        return content_vbox

    def _build_settings_tab(self) -> None:
        self.settings_tab = QtWidgets.QWidget()
        self.settings_tab.setObjectName("settings_tab")
        outer_layout = QtWidgets.QVBoxLayout(self.settings_tab)
        outer_layout.setContentsMargins(0, 8, 0, 8)
        outer_layout.setSpacing(0)

        scroll_area = QtWidgets.QScrollArea(self.settings_tab)
        scroll_area.setWidgetResizable(True)
        scroll_contents = QtWidgets.QWidget()
        self.settingsWidgetsLayout = QtWidgets.QVBoxLayout(scroll_contents)
        self.settingsWidgetsLayout.setContentsMargins(0, 0, 0, 0)
        self.settingsWidgetsLayout.setSpacing(0)

        output_row = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel("Output Directory", scroll_contents)
        self.outputFolderLineEdit = QtWidgets.QLineEdit(scroll_contents)
        self.outputFolderLineEdit.setObjectName("outputFolderLineEdit")
        self.outputFolderLineEdit.setReadOnly(True)

        self.outputFolderButton = QtWidgets.QPushButton(
            "Browse Folder", scroll_contents
        )
        self.outputFolderButton.setObjectName("outputFolderButton")
        self.outputFolderButton.setFlat(True)

        self.outputOpenButton = QtWidgets.QPushButton("Open Folder", scroll_contents)
        self.outputOpenButton.setObjectName("outputOpenButton")
        self.outputOpenButton.setFlat(True)

        output_row.addWidget(self.outputFolderLineEdit)
        output_row.addWidget(self.outputFolderButton)
        output_row.addWidget(self.outputOpenButton)

        self.settingsWidgetsLayout.addWidget(self.label)
        self.settingsWidgetsLayout.addLayout(output_row)

        scroll_area.setWidget(scroll_contents)
        outer_layout.addWidget(scroll_area)
        self.tabWidget.addTab(self.settings_tab, "Settings")

    def _build_presets_tab(self) -> None:
        self.presets_tab = QtWidgets.QWidget()
        self.presets_tab.setObjectName("presets_tab")
        presets_layout = QtWidgets.QVBoxLayout(self.presets_tab)

        self.presetsList = QtWidgets.QListWidget(self.presets_tab)
        self.presetsList.setObjectName("presetsList")
        presets_layout.addWidget(self.presetsList)

        btn_row = QtWidgets.QHBoxLayout()
        self.applyPresetButton = QtWidgets.QPushButton("Apply", self.presets_tab)
        self.applyPresetButton.setObjectName("applyPresetButton")
        self.applyPresetButton.setFlat(True)

        self.savePresetButton = QtWidgets.QPushButton("Save Current", self.presets_tab)
        self.savePresetButton.setObjectName("savePresetButton")
        self.savePresetButton.setFlat(True)

        self.controlPresetButton = QtWidgets.QPushButton(
            "Apply Settings", self.presets_tab
        )
        self.controlPresetButton.setObjectName("controlPresetButton")
        self.controlPresetButton.setCheckable(True)
        self.controlPresetButton.setFlat(True)

        btn_row.addWidget(self.applyPresetButton)
        btn_row.addWidget(self.savePresetButton)
        btn_row.addWidget(self.controlPresetButton)
        presets_layout.addLayout(btn_row)

        self.tabWidget.addTab(self.presets_tab, "Presets")