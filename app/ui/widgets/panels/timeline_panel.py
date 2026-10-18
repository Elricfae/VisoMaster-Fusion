# app/ui/widgets/panels/timeline_panel.py
from __future__ import annotations

from typing import Optional
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from app.ui.core.asset_manager import AssetManager
from app.ui.widgets.panels.base_panel import BaseDockPanel, DrawerZone


class TimelinePanel(BaseDockPanel):
    """Modular timeline seekbar, zoom slider, and playback transport controls."""

    panel_id = "timelineDockWidget"
    panel_title = "Timeline & Playback"
    default_zone = DrawerZone.BOTTOM

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

    def _build_ui(self) -> None:
        self.container.setObjectName("timelineControlsContainer")
        timeline_vbox = QtWidgets.QVBoxLayout(self.container)
        timeline_vbox.setContentsMargins(8, 4, 8, 4)
        timeline_vbox.setSpacing(4)

        self._build_timeline_row()
        self._build_transport_controls_row()

        timeline_vbox.addLayout(self.horizontalLayoutMediaSlider)
        timeline_vbox.addLayout(self.horizontalLayoutMediaButtons)

        for name in (
            "timelineScrollArea",
            "videoSeekSlider",
            "videoSeekLineEdit",
            "zoomLabel",
            "timelineZoomSlider",
            "faceCompareCheckBox",
            "faceMaskCheckBox",
            "liveSoundButton",
            "frameRewindButton",
            "buttonMediaRecord",
            "buttonMediaPlay",
            "frameAdvanceButton",
            "addMarkerButton",
            "removeMarkerButton",
            "previousMarkerButton",
            "nextMarkerButton",
            "viewFullScreenButton",
            "theatreModeButton",
        ):
            self.export_widget(name, getattr(self, name))

    def _build_timeline_row(self) -> None:
        self.horizontalLayoutMediaSlider = QtWidgets.QHBoxLayout()
        self.horizontalLayoutMediaSlider.setObjectName("horizontalLayoutMediaSlider")

        self.timelineScrollArea = QtWidgets.QScrollArea(self.container)
        self.timelineScrollArea.setObjectName("timelineScrollArea")
        self.timelineScrollArea.setMaximumHeight(42)
        self.timelineScrollArea.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.timelineScrollArea.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.timelineScrollArea.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.timelineScrollArea.setWidgetResizable(True)

        scroll_contents = QtWidgets.QWidget()
        scroll_contents.setObjectName("scrollAreaWidgetContents")
        vbox_tl = QtWidgets.QVBoxLayout(scroll_contents)
        vbox_tl.setContentsMargins(0, 0, 0, 0)
        vbox_tl.setSpacing(0)

        self.videoSeekSlider = QtWidgets.QSlider(scroll_contents)
        self.videoSeekSlider.setObjectName("videoSeekSlider")
        self.videoSeekSlider.setOrientation(Qt.Orientation.Horizontal)
        vbox_tl.addWidget(self.videoSeekSlider)

        self.timelineScrollArea.setWidget(scroll_contents)
        self.horizontalLayoutMediaSlider.addWidget(self.timelineScrollArea)

        self.videoSeekLineEdit = QtWidgets.QLineEdit(self.container)
        self.videoSeekLineEdit.setObjectName("videoSeekLineEdit")
        self.videoSeekLineEdit.setMaximumSize(QtCore.QSize(70, 16777215))
        self.videoSeekLineEdit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.videoSeekLineEdit.setClearButtonEnabled(False)
        self.horizontalLayoutMediaSlider.addWidget(self.videoSeekLineEdit)

        self.zoomLabel = QtWidgets.QLabel("Zoom:", self.container)
        self.zoomLabel.setObjectName("zoomLabel")
        self.horizontalLayoutMediaSlider.addWidget(self.zoomLabel)

        self.timelineZoomSlider = QtWidgets.QSlider(self.container)
        self.timelineZoomSlider.setObjectName("timelineZoomSlider")
        self.timelineZoomSlider.setMaximumSize(QtCore.QSize(100, 16777215))
        self.timelineZoomSlider.setMinimum(1)
        self.timelineZoomSlider.setMaximum(30)
        self.timelineZoomSlider.setPageStep(1)
        self.timelineZoomSlider.setOrientation(Qt.Orientation.Horizontal)
        self.horizontalLayoutMediaSlider.addWidget(self.timelineZoomSlider)

    def _build_transport_controls_row(self) -> None:
        self.horizontalLayoutMediaButtons = QtWidgets.QHBoxLayout()
        self.horizontalLayoutMediaButtons.setObjectName("horizontalLayoutMediaButtons")

        self.faceCompareCheckBox = QtWidgets.QCheckBox("Face Compare", self.container)
        self.faceCompareCheckBox.setObjectName("faceCompareCheckBox")
        self.faceCompareCheckBox.hide()

        self.faceMaskCheckBox = QtWidgets.QCheckBox("Face Mask", self.container)
        self.faceMaskCheckBox.setObjectName("faceMaskCheckBox")
        self.faceMaskCheckBox.hide()

        def _make_flat_btn(
            name: str,
            tooltip: str,
            icon_file: Optional[str] = None,
            checkable: bool = False,
        ) -> QtWidgets.QPushButton:
            btn = QtWidgets.QPushButton(self.container)
            btn.setObjectName(name)
            btn.setToolTip(tooltip)
            btn.setFlat(True)
            btn.setCheckable(checkable)
            if icon_file:
                btn.setIcon(AssetManager.get_icon(icon_file))
            return btn

        self.liveSoundButton = _make_flat_btn(
            "liveSoundButton", "Toggle Audio (Real-time)", "audio_off.png", True
        )
        self.frameRewindButton = _make_flat_btn(
            "frameRewindButton", "Rewind Frames", "tl_left_off.png"
        )
        self.frameRewindButton.setMaximumWidth(100)

        self.buttonMediaRecord = _make_flat_btn(
            "buttonMediaRecord", "Record Video", "rec_off.png", True
        )
        self.buttonMediaPlay = _make_flat_btn(
            "buttonMediaPlay", "Play / Pause", "play_off.png", True
        )
        self.frameAdvanceButton = _make_flat_btn(
            "frameAdvanceButton", "Advance Frames", "tl_right_off.png"
        )
        self.frameAdvanceButton.setMaximumWidth(100)

        self.addMarkerButton = _make_flat_btn(
            "addMarkerButton", "Add Marker", "add_marker_off.png"
        )
        self.removeMarkerButton = _make_flat_btn(
            "removeMarkerButton", "Remove Marker", "remove_marker_off.png"
        )
        self.previousMarkerButton = _make_flat_btn(
            "previousMarkerButton",
            "Move to Previous Marker",
            "previous_marker_off.png",
        )
        self.nextMarkerButton = _make_flat_btn(
            "nextMarkerButton", "Move to Next Marker", "next_marker_off.png"
        )
        self.viewFullScreenButton = _make_flat_btn(
            "viewFullScreenButton", "View Fullscreen (F11)", "fullscreen_v2.png"
        )
        self.theatreModeButton = _make_flat_btn(
            "theatreModeButton",
            "Toggle Auto-Hide Overlay Mode vs Dockable Pinned Mode (T)",
            "theatre_mode.png",
            True,
        )
        self.theatreModeButton.setIconSize(QtCore.QSize(24, 24))

        for widget in (
            self.liveSoundButton,
            self.frameRewindButton,
            self.buttonMediaRecord,
            self.buttonMediaPlay,
            self.frameAdvanceButton,
            self.addMarkerButton,
            self.removeMarkerButton,
            self.previousMarkerButton,
            self.nextMarkerButton,
            self.viewFullScreenButton,
            self.theatreModeButton,
        ):
            self.horizontalLayoutMediaButtons.addWidget(widget)