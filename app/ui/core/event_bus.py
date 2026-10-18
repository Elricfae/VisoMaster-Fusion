# app/ui/core/event_bus.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6 import QtCore

from app.ui.widgets.sortable_widgets import (
    MediaMetadata,
    MediaType,
    SortableImageDimension,
)


@dataclass(frozen=True)
class TargetMediaSelectedEvent:
    """Emitted when the user selects a target image, video, or webcam."""

    media_id: str
    media_type: MediaType
    file_path: Optional[Path] = None
    webcam_index: Optional[int] = None
    metadata: Optional[MediaMetadata] = None


@dataclass(frozen=True)
class TargetFolderChangedEvent:
    """Emitted when a new target media directory is chosen."""

    folder_path: Path
    include_images: bool
    include_videos: bool


@dataclass(frozen=True)
class TargetFilesAddedEvent:
    """Emitted when individual target media files are selected via file dialog."""

    file_paths: tuple[Path, ...]


@dataclass(frozen=True)
class TargetMediaFilterChangedEvent:
    """Emitted when target list filter or sort state changes."""

    visible_count: int
    total_count: int
    search_text: str
    min_dimension: Optional[SortableImageDimension]


class AppEventBus(QtCore.QObject):
    """Centralized, decoupled signal hub for cross-panel and UI-to-Engine events."""

    # --- Target Media Events ---
    targetFolderChanged = QtCore.Signal(object)         # TargetFolderChangedEvent
    targetFilesAdded = QtCore.Signal(object)            # TargetFilesAddedEvent
    targetMediaSelected = QtCore.Signal(object)         # TargetMediaSelectedEvent
    targetMediaCleared = QtCore.Signal()
    targetMediaFilterChanged = QtCore.Signal(object)    # TargetMediaFilterChangedEvent
    targetWebcamsToggled = QtCore.Signal(bool)

    # --- Input Faces & Embeddings Events ---
    inputFacesFolderChanged = QtCore.Signal(Path)
    inputFacesSelectionChanged = QtCore.Signal(object)
    embeddingSelectionChanged = QtCore.Signal(object)

    # --- Playback / Timeline Events ---
    playbackToggled = QtCore.Signal(bool)
    recordToggled = QtCore.Signal(bool)
    frameSeekRequested = QtCore.Signal(int)

    # --- System & Lifecycle Events ---
    statusMessagePublished = QtCore.Signal(str, int)
    workspaceStateDirty = QtCore.Signal()
    applicationClosing = QtCore.Signal()


_global_event_bus: Optional[AppEventBus] = None


def get_event_bus() -> AppEventBus:
    """Return the shared application-wide AppEventBus singleton."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = AppEventBus()
    return _global_event_bus