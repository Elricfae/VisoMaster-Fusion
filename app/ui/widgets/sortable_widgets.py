# app/ui/widgets/sortable_widgets.py
from __future__ import annotations

import enum
import os
import re
from dataclasses import dataclass, field
from functools import total_ordering
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt


# =============================================================================
# 1. ENUMS, CONSTANTS & ROLES
# =============================================================================

DRAG_DROP_FOLDER_KEY: str = "__drag_and_drop__"
DRAG_DROP_FOLDER_LABEL: str = "Drag & Drop"


class MediaType(enum.StrEnum):
    """Standardized media/item categories for filtering and display."""

    IMAGE = "image"
    VIDEO = "video"
    WEBCAM = "webcam"
    EMBEDDING = "embedding"
    AUDIO = "audio"
    UNKNOWN = "unknown"


_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".jp2"}
)
_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".gif"}
)
_EMBEDDING_EXTENSIONS: frozenset[str] = frozenset({".json", ".npy", ".npz", ".kv", ".pt"})


def infer_media_type(file_path: str | Path) -> str:
    """Infer standard media type string from a file path extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return MediaType.IMAGE.value
    if suffix in _VIDEO_EXTENSIONS:
        return MediaType.VIDEO.value
    if suffix in _EMBEDDING_EXTENSIONS:
        return MediaType.EMBEDDING.value
    return MediaType.UNKNOWN.value


class SortableDataRole(enum.IntEnum):
    """Custom Qt ItemDataRoles starting at 0x180 (above Qt.ItemDataRole.UserRole)."""

    FileInfoRole = 0x180             # QtCore.QFileInfo
    ImageDimensionRole = 0x181       # SortableImageDimension
    PayloadRole = 0x182              # Generic domain payload / ID
    FileTypeRole = 0x183             # str (MediaType value: "image", "video", "webcam", etc.)
    MediaMetadataRole = 0x184        # MediaMetadata
    CustomSortKeyRole = 0x185        # Any comparable value for custom widget sorting
    IsFolderHeaderRole = 0x186       # bool: True if item is a FolderHeaderListWidgetItem
    FolderPathRole = 0x187           # str: Normalized absolute folder path or virtual folder key
    FolderDepthRole = 0x188          # int: Kept for role compatibility (always 0 for flat groups)
    FolderExpandedRole = 0x189       # bool: True if expanded, False if collapsed
    FolderVisibleCountRole = 0x18A   # int: Number of visible media cards in folder
    FolderTotalCountRole = 0x18B     # int: Total number of media cards in folder
    FolderProgressRole = 0x18C       # float: Loading progress in [0.0, 1.0) while scanning, or -1.0 when complete


class SortingMode(enum.IntEnum):
    """Supported sorting strategies for SortableListWidget."""

    FileBirthDate = 0x140
    FileName = 0x141
    FileSize = 0x143
    ImageDimension = 0x144
    ImagePixels = 0x145
    VideoLength = 0x146
    CustomKey = 0x147

    def to_id(self) -> str:
        return self.name

    @property
    def label(self) -> str:
        return _SORTING_MODE_LABELS.get(self, self.name)


_SORTING_MODE_LABELS: dict[SortingMode, str] = {
    SortingMode.FileName: "Sort by Name",
    SortingMode.FileBirthDate: "Sort by File Date",
    SortingMode.FileSize: "Sort by File Size",
    SortingMode.ImageDimension: "Sort by Dimensions",
    SortingMode.ImagePixels: "Sort by Total Pixels",
    SortingMode.VideoLength: "Sort by Frame Length",
    SortingMode.CustomKey: "Sort by Custom Order",
}

# Default tuple list for media panels (kept for backward compatibility)
initialize_combobox_items: list[tuple[str, SortingMode]] = [
    (SortingMode.FileName.label, SortingMode.FileName),
    (SortingMode.FileBirthDate.label, SortingMode.FileBirthDate),
    (SortingMode.FileSize.label, SortingMode.FileSize),
    (SortingMode.ImageDimension.label, SortingMode.ImageDimension),
    (SortingMode.ImagePixels.label, SortingMode.ImagePixels),
    (SortingMode.VideoLength.label, SortingMode.VideoLength),
]

_FILE_MODES: frozenset[SortingMode] = frozenset(
    {SortingMode.FileBirthDate, SortingMode.FileName, SortingMode.FileSize}
)


# =============================================================================
# 2. VALUE OBJECTS & METADATA CONTRACTS
# =============================================================================


@dataclass(frozen=True)
class MediaMetadata:
    """Immutable media stream metadata attached to a SortableListWidgetItem."""

    width: int = 0
    height: int = 0
    total_frames: int = 0
    frame_rate: float = 0.0
    bitrate_kbits: float = 0.0
    duration_seconds: float = 0.0
    codec: str = ""


@total_ordering
@dataclass(frozen=True)
class SortableImageDimension:
    """Immutable 2D resolution value object supporting comparison and filtering."""

    width: int
    height: int

    def size(self) -> QtCore.QSize:
        return QtCore.QSize(self.width, self.height)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def sort_key(self) -> tuple[int, int]:
        return self.width, self.height

    def contains(self, minimum: "SortableImageDimension") -> bool:
        """True when this dimension is at least as large as minimum on both axes."""
        return self.width >= minimum.width and self.height >= minimum.height

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SortableImageDimension):
            return NotImplemented
        return self.sort_key == other.sort_key

    def __hash__(self) -> int:
        return hash(self.sort_key)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SortableImageDimension):
            return NotImplemented
        return self.sort_key < other.sort_key

    @classmethod
    def from_string(cls, value: str) -> "SortableImageDimension":
        match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", value)
        if match is None:
            raise ValueError(f"Not a WIDTHxHEIGHT dimension: {value!r}")
        return cls(int(match.group(1)), int(match.group(2)))

    @classmethod
    def from_metadata(
        cls, metadata: Optional[MediaMetadata]
    ) -> Optional["SortableImageDimension"]:
        if metadata is None or not (metadata.width and metadata.height):
            return None
        return cls(int(metadata.width), int(metadata.height))


@dataclass(frozen=True)
class FilterCriteria:
    """Immutable snapshot of active filter rules for a SortableListWidget."""

    search_text: str = ""
    allowed_file_types: Optional[frozenset[str]] = None
    min_dimension: Optional[SortableImageDimension] = None
    custom_predicate: Optional[Callable[["SortableListWidgetItem"], bool]] = field(
        default=None, compare=False
    )


# =============================================================================
# 3. FOLDER HEADER & SORTABLE LIST WIDGET ITEMS
# =============================================================================


class FolderHeaderListWidgetItem(QtWidgets.QListWidgetItem):
    """Full-width independent collapsible folder group header inside SortableListWidget."""

    def __init__(
        self,
        folder_path: str | Path,
        display_text: Optional[str] = None,
        expanded: bool = True,
        parent: Optional[QtWidgets.QListWidget] = None,
    ) -> None:
        raw_str = str(folder_path)
        if raw_str == DRAG_DROP_FOLDER_KEY:
            norm_path = DRAG_DROP_FOLDER_KEY
            default_label = DRAG_DROP_FOLDER_LABEL
        elif raw_str:
            norm_path = os.path.abspath(raw_str)
            default_label = norm_path
        else:
            norm_path = ""
            default_label = "Devices / Live Sources"

        label = display_text if display_text is not None else default_label
        super().__init__(label, parent)
        self.setData(SortableDataRole.IsFolderHeaderRole, True)
        self.setData(SortableDataRole.FolderPathRole, norm_path)
        self.setData(SortableDataRole.FolderDepthRole, 0)
        self.setData(SortableDataRole.FolderExpandedRole, bool(expanded))
        self.setData(SortableDataRole.FolderVisibleCountRole, 0)
        self.setData(SortableDataRole.FolderTotalCountRole, 0)
        self.setData(SortableDataRole.FolderProgressRole, -1.0)
        self.setToolTip(label)
        # Headers are clickable to expand/collapse, but not selectable as a media item
        self.setFlags(Qt.ItemFlag.ItemIsEnabled)

    def folderPath(self) -> str:
        return str(self.data(SortableDataRole.FolderPathRole) or "")

    def isExpanded(self) -> bool:
        val = self.data(SortableDataRole.FolderExpandedRole)
        return True if val is None else bool(val)

    def setExpanded(self, expanded: bool) -> None:
        self.setData(SortableDataRole.FolderExpandedRole, bool(expanded))

    def visibleCount(self) -> int:
        return int(self.data(SortableDataRole.FolderVisibleCountRole) or 0)

    def totalCount(self) -> int:
        return int(self.data(SortableDataRole.FolderTotalCountRole) or 0)

    def setCounts(self, visible: int, total: int) -> None:
        self.setData(SortableDataRole.FolderVisibleCountRole, max(0, int(visible)))
        self.setData(SortableDataRole.FolderTotalCountRole, max(0, int(total)))

    def progress(self) -> float:
        val = self.data(SortableDataRole.FolderProgressRole)
        return -1.0 if val is None else float(val)

    def setProgress(self, progress: float) -> None:
        self.setData(SortableDataRole.FolderProgressRole, float(progress))


class SortableListWidgetItem(QtWidgets.QListWidgetItem):
    """Native QListWidgetItem with typed metadata roles and multi-mode comparison."""

    def __init__(
        self,
        text: str = "",
        icon: Optional[QtGui.QIcon] = None,
        parent: Optional[QtWidgets.QListWidget] = None,
    ) -> None:
        if icon is not None:
            super().__init__(icon, text, parent)
        else:
            super().__init__(text, parent)

    @classmethod
    def from_file_path(
        cls,
        file_path: str | Path,
        *,
        icon: Optional[QtGui.QIcon] = None,
        file_type: Optional[str] = None,
        metadata: Optional[MediaMetadata] = None,
        payload: Any = None,
        folder_key: Optional[str] = None,
        parent: Optional[QtWidgets.QListWidget] = None,
    ) -> "SortableListWidgetItem":
        """Convenience factory that populates QFileInfo, dimensions, folder group, and tooltip."""
        abs_str = os.path.abspath(str(file_path))
        info = QtCore.QFileInfo(abs_str)
        item = cls(text=info.fileName(), icon=icon, parent=parent)
        item.setFileInfo(info)
        item.setData(
            SortableDataRole.FolderPathRole,
            folder_key if folder_key is not None else os.path.dirname(abs_str),
        )
        resolved_type = file_type or infer_media_type(abs_str)
        item.setFileType(resolved_type)

        if metadata is not None:
            item.setMediaMetadata(metadata)
            dim = SortableImageDimension.from_metadata(metadata)
            if dim is not None:
                item.setImageDimension(dim)

        if payload is not None:
            item.setPayload(payload)

        item.refresh_tooltip()
        return item

    # --- Role Getters & Setters ---

    def folderPath(self) -> str:
        stored = self.data(SortableDataRole.FolderPathRole)
        if stored:
            return str(stored)
        info = self.fileInfo()
        if info is not None and info.filePath():
            return os.path.dirname(os.path.abspath(info.filePath()))
        return ""

    def setFolderPath(self, folder_path: str | Path) -> None:
        self.setData(SortableDataRole.FolderPathRole, str(folder_path))

    def fileInfo(self) -> Optional[QtCore.QFileInfo]:
        fileinfo = self.data(SortableDataRole.FileInfoRole)
        return fileinfo if isinstance(fileinfo, QtCore.QFileInfo) else None

    def setFileInfo(self, info: QtCore.QFileInfo | str | Path) -> None:
        qinfo = info if isinstance(info, QtCore.QFileInfo) else QtCore.QFileInfo(str(info))
        self.setData(SortableDataRole.FileInfoRole, qinfo)

    def fileType(self) -> str:
        stored = self.data(SortableDataRole.FileTypeRole)
        if stored:
            return str(stored)
        fileinfo = self.fileInfo()
        if fileinfo is None:
            return ""
        return infer_media_type(fileinfo.filePath())

    def setFileType(self, file_type: str | MediaType) -> None:
        value = file_type.value if isinstance(file_type, MediaType) else str(file_type)
        self.setData(SortableDataRole.FileTypeRole, value)

    def imageDimension(self) -> Optional[SortableImageDimension]:
        imgdim = self.data(SortableDataRole.ImageDimensionRole)
        return imgdim if isinstance(imgdim, SortableImageDimension) else None

    def setImageDimension(self, imgdim: Optional[SortableImageDimension]) -> None:
        self.setData(SortableDataRole.ImageDimensionRole, imgdim)

    def mediaMetadata(self) -> Optional[MediaMetadata]:
        metadata = self.data(SortableDataRole.MediaMetadataRole)
        return metadata if isinstance(metadata, MediaMetadata) else None

    def setMediaMetadata(self, metadata: Optional[MediaMetadata]) -> None:
        self.setData(SortableDataRole.MediaMetadataRole, metadata)
        if metadata is not None and self.imageDimension() is None:
            dim = SortableImageDimension.from_metadata(metadata)
            if dim is not None:
                self.setImageDimension(dim)

    def payload(self) -> Any:
        """Generic domain object, ID, or path attached to this item."""
        return self.data(SortableDataRole.PayloadRole)

    def setPayload(self, value: Any) -> None:
        self.setData(SortableDataRole.PayloadRole, value)

    def customSortKey(self) -> Any:
        return self.data(SortableDataRole.CustomSortKeyRole)

    def setCustomSortKey(self, key: Any) -> None:
        self.setData(SortableDataRole.CustomSortKeyRole, key)

    # --- Filtering Evaluation ---

    def matches_filter(self, criteria: FilterCriteria) -> bool:
        """Return True if this item satisfies all active filter criteria."""
        # 1. File / Media Type filter
        if criteria.allowed_file_types is not None:
            item_type = self.fileType()
            if item_type and item_type not in criteria.allowed_file_types:
                return False

        # 2. Search query filter (checks item text and filename)
        if criteria.search_text:
            needle = criteria.search_text.casefold()
            haystack = self.text().casefold()
            fileinfo = self.fileInfo()
            if fileinfo is not None:
                haystack = f"{haystack} {fileinfo.fileName().casefold()}"
            if needle not in haystack:
                return False

        # 3. Minimum Image Dimension filter (webcams without dimensions bypass resolution check)
        if criteria.min_dimension is not None and self.fileType() != MediaType.WEBCAM.value:
            imgdim = self.imageDimension()
            if imgdim is not None and not imgdim.contains(criteria.min_dimension):
                return False

        # 4. Custom predicate
        if criteria.custom_predicate is not None:
            if not criteria.custom_predicate(self):
                return False

        return True

    # --- Sorting Comparators ---

    def _sort_key(self, sort_mode: SortingMode) -> Any:
        """Value to order this item by, or None when it cannot be determined."""
        if sort_mode == SortingMode.CustomKey:
            return self.customSortKey()

        if sort_mode in _FILE_MODES:
            fileinfo = self.fileInfo()
            if fileinfo is None:
                return None
            if sort_mode == SortingMode.FileBirthDate:
                return fileinfo.birthTime()
            if sort_mode == SortingMode.FileName:
                return fileinfo.baseName().casefold()
            return fileinfo.size()

        if sort_mode == SortingMode.VideoLength:
            metadata = self.mediaMetadata()
            return None if metadata is None else metadata.total_frames

        imgdim = self.imageDimension()
        if imgdim is None:
            return None
        if sort_mode == SortingMode.ImageDimension:
            return imgdim.sort_key
        if sort_mode == SortingMode.ImagePixels:
            return imgdim.pixels
        return None

    def compare_with(
        self,
        other: "SortableListWidgetItem",
        sort_mode: SortingMode,
        descending: bool = False,
    ) -> bool:
        """Folder-independent comparator returning True if self should precede other."""
        left = self._sort_key(sort_mode)
        right = other._sort_key(sort_mode)

        # Items lacking the sorted attribute always sink below items that have it
        if left is None and right is None:
            res = self._tie_break_key() < other._tie_break_key()
            return not res if descending else res
        if left is None:
            return False
        if right is None:
            return True

        if left != right:
            res = bool(left < right)
            return not res if descending else res

        res = self._tie_break_key() < other._tie_break_key()
        return not res if descending else res

    def __lt__(self, other: object) -> bool:
        list_widget = self.listWidget()
        if not isinstance(list_widget, SortableListWidget):
            return super().__lt__(other)  # type: ignore[arg-type]
        if not isinstance(other, SortableListWidgetItem):
            return False

        left = self._sort_key(list_widget.sortingMode())
        right = other._sort_key(list_widget.sortingMode())

        if left is None or right is None:
            return left is not None

        if left != right:
            return bool(left < right)

        return self._tie_break_key() < other._tie_break_key()

    def _tie_break_key(self) -> str:
        fileinfo = self.fileInfo()
        if fileinfo is not None:
            return fileinfo.fileName().casefold()
        return self.text().casefold()

    # --- Tooltip Formatting ---

    def to_tooltip(self) -> str:
        tipslist: list[str] = []
        fileinfo = self.fileInfo()
        if fileinfo is not None and fileinfo.fileName():
            tipslist.append(f"FileName: {fileinfo.fileName()}")
            tipslist.append(f"FileType: {self.fileType()}")
            birth = fileinfo.birthTime()
            if birth.isValid():
                tipslist.append(
                    f"FileDate: {birth.toLocalTime().toString('yyyy-MM-dd HH:mm:ss')}"
                )
            tipslist.append(
                f"FileSize: {QtCore.QLocale.system().formattedDataSize(fileinfo.size())}"
            )
            tipslist.append(f"Path: {os.path.dirname(fileinfo.filePath())}")

        image_dim = self.imageDimension()
        if image_dim is not None:
            tipslist.append(f"Dimensions: {image_dim}")

        metadata = self.mediaMetadata()
        if metadata is not None and self.fileType() == MediaType.VIDEO.value:
            tipslist.append(f"Length: {metadata.total_frames} frames")
            if metadata.frame_rate:
                tipslist.append(f"Frame Rate: {metadata.frame_rate:.3f}")
            if metadata.bitrate_kbits:
                tipslist.append(f"Bitrate: {metadata.bitrate_kbits:.0f} kBit/s")

        return "\n".join(tipslist)

    def refresh_tooltip(self) -> None:
        """Update the native Qt item tooltip from stored metadata."""
        self.setToolTip(self.to_tooltip())


# =============================================================================
# 4. SORTABLE & FILTERABLE LIST WIDGET (WITH FOLDER GROUPS, PROGRESS & DRAG-DROP)
# =============================================================================


class SortableListWidget(QtWidgets.QListWidget):
    """Globally reusable QListWidget with built-in multi-mode sorting, global filtering,
    external Drag & Drop support, and independent collapsible folder groups with live loading progress.
    """

    sortingChanged = QtCore.Signal(SortingMode, Qt.SortOrder)
    filterChanged = QtCore.Signal(FilterCriteria)
    visibleCountChanged = QtCore.Signal(int)
    folderExpansionChanged = QtCore.Signal(str, bool)
    pathsDropped = QtCore.Signal(list)  # list[Path] emitted when external files/folders are dropped

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._sorting_mode: SortingMode = SortingMode.FileName
        self._sort_order: Qt.SortOrder = Qt.SortOrder.AscendingOrder
        self._search_text: str = ""
        self._allowed_file_types: Optional[frozenset[str]] = None
        self._min_image_dimension: Optional[SortableImageDimension] = None
        self._custom_predicate: Optional[Callable[[SortableListWidgetItem], bool]] = None

        # Independent Collapsible Folder Group & Loading Progress state
        self._group_by_folder: bool = False
        self._root_folder_path: Optional[Path] = None
        self._folder_headers: dict[str, FolderHeaderListWidgetItem] = {}
        self._folder_expanded: dict[str, bool] = {}
        self._folder_expected_totals: dict[str, int] = {}
        self._folder_loaded_counts: dict[str, int] = {}
        self._folder_visible_counts: dict[str, int] = {}
        # Fast O(1) deduplication set storing (folder_key, normcase_file_path)
        self._known_item_keys: set[tuple[str, str]] = set()

        # Enable external file/folder Drag & Drop
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)

        self.itemClicked.connect(self._on_item_clicked)

    # --- External Drag & Drop Event Handlers ---

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            dropped_paths: list[Path] = []
            for url in mime.urls():
                if url.isLocalFile():
                    local_str = os.path.abspath(url.toLocalFile())
                    if os.path.exists(local_str):
                        dropped_paths.append(Path(local_str))
            if dropped_paths:
                event.acceptProposedAction()
                self.pathsDropped.emit(dropped_paths)
                return
        event.ignore()

    # --- ComboBox Helper ---

    @staticmethod
    def populate_sort_combobox(
        combobox: QtWidgets.QComboBox,
        items: Optional[Iterable[tuple[str, SortingMode]]] = None,
    ) -> None:
        """Populate any QComboBox with (label, SortingMode) pairs."""
        combobox.blockSignals(True)
        combobox.clear()
        for label, mode in (items if items is not None else initialize_combobox_items):
            combobox.addItem(label, mode)
        combobox.blockSignals(False)

    # --- O(1) Deduplication Helpers ---

    @staticmethod
    def _make_dedup_key(folder_key: str, file_path: str | Path) -> tuple[str, str]:
        return (folder_key, os.path.normcase(os.path.abspath(str(file_path))))

    def has_media_item(self, folder_key: str, file_path: str | Path) -> bool:
        """O(1) check whether a file is already present in `folder_key`."""
        return self._make_dedup_key(folder_key, file_path) in self._known_item_keys

    def _register_item_key(self, item: SortableListWidgetItem) -> None:
        info = item.fileInfo()
        if info and info.filePath():
            self._known_item_keys.add(self._make_dedup_key(item.folderPath(), info.filePath()))

    def _unregister_item_key(self, item: SortableListWidgetItem) -> None:
        info = item.fileInfo()
        if info and info.filePath():
            self._known_item_keys.discard(self._make_dedup_key(item.folderPath(), info.filePath()))

    # --- Folder Grouping & Progress API ---

    def groupByFolder(self) -> bool:
        return self._group_by_folder

    def setGroupByFolder(self, enabled: bool) -> None:
        if self._group_by_folder != enabled:
            self._group_by_folder = enabled
            if enabled:
                self.setUniformItemSizes(False)
                self.setGridSize(QtCore.QSize())
            self.apply_sort()
            self.apply_filters()

    def rootFolderPath(self) -> Optional[Path]:
        return self._root_folder_path

    def setRootFolderPath(self, root_path: Optional[str | Path]) -> None:
        self._root_folder_path = Path(os.path.abspath(str(root_path))) if root_path else None

    def clear(self) -> None:
        self._folder_headers.clear()
        self._folder_expanded.clear()
        self._folder_expected_totals.clear()
        self._folder_loaded_counts.clear()
        self._folder_visible_counts.clear()
        self._known_item_keys.clear()
        super().clear()

    @staticmethod
    def _format_folder_label(folder_str: str) -> str:
        if folder_str == DRAG_DROP_FOLDER_KEY:
            return DRAG_DROP_FOLDER_LABEL
        if not folder_str:
            return "Devices / Live Sources"
        return str(Path(folder_str))

    def _folder_group_key(self, f_str: str) -> tuple[int, tuple[str, ...]]:
        if f_str == DRAG_DROP_FOLDER_KEY:
            return (-1, ("",))
        if not f_str:
            return (1, ("",))
        p = Path(f_str)
        if self._root_folder_path is not None:
            try:
                rel = p.relative_to(self._root_folder_path)
                return (0, tuple(part.casefold() for part in rel.parts))
            except ValueError:
                pass
        return (0, tuple(part.casefold() for part in p.parts))

    def initialize_folder_totals(
        self, folder_totals: dict[str, int], *, additive: bool = False
    ) -> None:
        """Pre-create all discovered folder headers with `shown / total` and progress bar."""
        if not self._group_by_folder:
            return

        self.setUpdatesEnabled(False)
        try:
            for f_str in sorted(folder_totals.keys(), key=self._folder_group_key):
                added_total = max(0, int(folder_totals[f_str]))
                if added_total <= 0:
                    continue
                already_loaded = self._folder_loaded_counts.get(f_str, 0)
                total = (already_loaded + added_total) if additive else added_total
                self._folder_expected_totals[f_str] = total
                self._folder_loaded_counts.setdefault(f_str, 0)
                self._folder_visible_counts.setdefault(f_str, 0)
                header = self.ensure_folder_header(f_str)
                loaded = self._folder_loaded_counts[f_str]
                vis = self._folder_visible_counts[f_str]
                header.setCounts(vis, total)
                header.setProgress(loaded / total if loaded < total else -1.0)
                header.setHidden(False)
        finally:
            self.setUpdatesEnabled(True)

        self.scheduleDelayedItemsLayout()
        self.viewport().update()

    def ensure_folder_header(self, folder_str: str) -> FolderHeaderListWidgetItem:
        """Ensure an independent folder group header exists for `folder_str`."""
        header = self._folder_headers.get(folder_str)
        label = self._format_folder_label(folder_str)
        if header is None:
            expanded = self._folder_expanded.get(folder_str, True)
            header = FolderHeaderListWidgetItem(
                folder_path=folder_str,
                display_text=label,
                expanded=expanded,
            )
            self._folder_headers[folder_str] = header
            self._folder_expanded[folder_str] = expanded
            super().addItem(header)
        else:
            header.setText(label)
            header.setToolTip(label)
        return header

    def addItem(self, item: QtWidgets.QListWidgetItem | str) -> None:  # type: ignore[override]
        if isinstance(item, SortableListWidgetItem):
            self._register_item_key(item)
            if self._group_by_folder:
                f_str = item.folderPath()
                self.ensure_folder_header(f_str)
        super().addItem(item)

    def add_streamed_items_batch(self, items: Sequence[SortableListWidgetItem]) -> None:
        """Insert a batch of streamed items inside a single `setUpdatesEnabled(False)` block
        and update modified folder headers only once per batch to avoid O(N^2) Qt relayouts.
        """
        if not items:
            return

        criteria = self.filterCriteria()
        touched_folders: set[str] = set()

        self.setUpdatesEnabled(False)
        self.blockSignals(True)
        try:
            for item in items:
                self._register_item_key(item)
                is_match = item.matches_filter(criteria)

                if not self._group_by_folder:
                    super().addItem(item)
                    item.setHidden(not is_match)
                    continue

                f_str = item.folderPath()
                self.ensure_folder_header(f_str)
                super().addItem(item)

                is_expanded = self._folder_expanded.get(f_str, True)
                item.setHidden(not (is_match and is_expanded))

                self._folder_loaded_counts[f_str] = self._folder_loaded_counts.get(f_str, 0) + 1
                if is_match:
                    self._folder_visible_counts[f_str] = self._folder_visible_counts.get(f_str, 0) + 1
                touched_folders.add(f_str)

            # Update folder header roles only once per folder in the batch
            for f_str in touched_folders:
                header = self._folder_headers.get(f_str)
                if header is None:
                    continue
                loaded = self._folder_loaded_counts.get(f_str, 0)
                vis = self._folder_visible_counts.get(f_str, 0)
                expected_total = max(loaded, self._folder_expected_totals.get(f_str, loaded))
                header.setCounts(vis, expected_total)
                header.setProgress(loaded / expected_total if loaded < expected_total else -1.0)
                header.setHidden(False)
        finally:
            self.blockSignals(False)
            self.setUpdatesEnabled(True)

        self.scheduleDelayedItemsLayout()
        self.viewport().update()

    def register_streamed_item(self, item: SortableListWidgetItem) -> None:
        """Single-item fallback for incremental registration."""
        criteria = self.filterCriteria()
        is_match = item.matches_filter(criteria)

        if not self._group_by_folder:
            item.setHidden(not is_match)
            return

        f_str = item.folderPath()
        header = self.ensure_folder_header(f_str)
        is_expanded = self._folder_expanded.get(f_str, True)
        item.setHidden(not (is_match and is_expanded))

        loaded = self._folder_loaded_counts.get(f_str, 0) + 1
        self._folder_loaded_counts[f_str] = loaded

        vis = self._folder_visible_counts.get(f_str, 0) + (1 if is_match else 0)
        self._folder_visible_counts[f_str] = vis

        expected_total = max(loaded, self._folder_expected_totals.get(f_str, loaded))
        header.setCounts(vis, expected_total)
        header.setProgress(loaded / expected_total if loaded < expected_total else -1.0)
        header.setHidden(False)

    def finalize_folder_loading(self) -> None:
        """Mark all folder loading progress bars as completed."""
        self._folder_expected_totals.clear()
        self.setUpdatesEnabled(False)
        try:
            for header in self._folder_headers.values():
                if header.progress() >= 0.0:
                    header.setProgress(-1.0)
        finally:
            self.setUpdatesEnabled(True)
        self.viewport().update()

    def isFolderExpanded(self, folder_str: str) -> bool:
        return self._folder_expanded.get(folder_str, True)

    @QtCore.Slot(QtWidgets.QListWidgetItem)
    def _on_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        if isinstance(item, FolderHeaderListWidgetItem):
            f_str = item.folderPath()
            new_state = not item.isExpanded()
            item.setExpanded(new_state)
            self._folder_expanded[f_str] = new_state
            self.setUpdatesEnabled(False)
            try:
                self._update_folder_group_visibility()
            finally:
                self.setUpdatesEnabled(True)
            self.scheduleDelayedItemsLayout()
            self.viewport().update()
            self.folderExpansionChanged.emit(f_str, new_state)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._group_by_folder and self._folder_headers:
            self.scheduleDelayedItemsLayout()

    # --- Item & Folder Removal API ---

    def items_in_folder(self, folder_str: str) -> list[SortableListWidgetItem]:
        """Return all media items belonging to a specific folder group."""
        result: list[SortableListWidgetItem] = []
        for row in range(self.count()):
            item = self.item(row)
            if isinstance(item, SortableListWidgetItem) and item.folderPath() == folder_str:
                result.append(item)
        return result

    def remove_media_items(self, items: Sequence[SortableListWidgetItem]) -> None:
        """Remove specific media items from the list and clean up any emptied folder headers."""
        if not items:
            return
        self.setUpdatesEnabled(False)
        try:
            for item in items:
                self._unregister_item_key(item)
                row = self.row(item)
                if row >= 0:
                    self.takeItem(row)
            self._cleanup_empty_folder_headers()
            visible_count = self._recalculate_visibility_and_counts()
        finally:
            self.setUpdatesEnabled(True)

        self.scheduleDelayedItemsLayout()
        self.viewport().update()
        self.visibleCountChanged.emit(visible_count)

    def remove_folder_group(self, folder_str: str) -> list[SortableListWidgetItem]:
        """Remove an entire folder group (its header + all its media items) and return the removed media items."""
        removed_items: list[SortableListWidgetItem] = []
        self.setUpdatesEnabled(False)
        try:
            for row in range(self.count() - 1, -1, -1):
                it = self.item(row)
                if isinstance(it, SortableListWidgetItem) and it.folderPath() == folder_str:
                    self._unregister_item_key(it)
                    taken = self.takeItem(row)
                    if isinstance(taken, SortableListWidgetItem):
                        removed_items.append(taken)
                elif isinstance(it, FolderHeaderListWidgetItem) and it.folderPath() == folder_str:
                    self.takeItem(row)

            self._folder_headers.pop(folder_str, None)
            self._folder_expanded.pop(folder_str, None)
            self._folder_expected_totals.pop(folder_str, None)
            self._folder_loaded_counts.pop(folder_str, None)
            self._folder_visible_counts.pop(folder_str, None)

            visible_count = self._recalculate_visibility_and_counts()
        finally:
            self.setUpdatesEnabled(True)

        self.scheduleDelayedItemsLayout()
        self.viewport().update()
        self.visibleCountChanged.emit(visible_count)
        return removed_items

    def _cleanup_empty_folder_headers(self) -> None:
        """Remove folder headers that no longer have any media items or expected pre-scan items."""
        active_folders: set[str] = set(self._folder_expected_totals.keys())
        for row in range(self.count()):
            it = self.item(row)
            if isinstance(it, SortableListWidgetItem):
                active_folders.add(it.folderPath())

        for f_str in list(self._folder_headers.keys()):
            if f_str not in active_folders:
                hdr = self._folder_headers.pop(f_str)
                hdr_row = self.row(hdr)
                if hdr_row >= 0:
                    self.takeItem(hdr_row)
                self._folder_expanded.pop(f_str, None)
                self._folder_loaded_counts.pop(f_str, None)
                self._folder_visible_counts.pop(f_str, None)

    def _recalculate_visibility_and_counts(self) -> int:
        if self._group_by_folder:
            return self._update_folder_group_visibility()
        criteria = self.filterCriteria()
        visible_count = 0
        for row in range(self.count()):
            item = self.item(row)
            if isinstance(item, SortableListWidgetItem):
                is_match = item.matches_filter(criteria)
            else:
                is_match = (
                    not criteria.search_text
                    or criteria.search_text.casefold() in item.text().casefold()
                )
            item.setHidden(not is_match)
            if is_match:
                visible_count += 1
        return visible_count

    # --- Sorting API ---

    def sortingMode(self) -> SortingMode:
        return self._sorting_mode

    def setSortingMode(self, sort_mode: SortingMode | int | None) -> None:
        if isinstance(sort_mode, int) and not isinstance(sort_mode, SortingMode):
            try:
                sort_mode = SortingMode(sort_mode)
            except ValueError:
                return
        if isinstance(sort_mode, SortingMode) and sort_mode != self._sorting_mode:
            self._sorting_mode = sort_mode
            self.apply_sort()
            self.sortingChanged.emit(self._sorting_mode, self._sort_order)

    def sortOrder(self) -> Qt.SortOrder:
        return self._sort_order

    def setSortOrder(self, order: Qt.SortOrder) -> None:
        if order != self._sort_order:
            self._sort_order = order
            self.apply_sort()
            self.sortingChanged.emit(self._sorting_mode, self._sort_order)

    def setSortDescending(self, descending: bool) -> None:
        """Slot-friendly helper for a checkable direction toggle button."""
        order = (
            Qt.SortOrder.DescendingOrder
            if descending
            else Qt.SortOrder.AscendingOrder
        )
        self.setSortOrder(order)

    def apply_sort(self) -> None:
        """Execute sort. When folder grouping is active, sorts cards independently inside each folder."""
        if not self._group_by_folder:
            self.sortItems(self._sort_order)
            return

        from functools import cmp_to_key

        current_selected = self.currentItem()
        media_by_folder: dict[str, list[SortableListWidgetItem]] = {}

        self.setUpdatesEnabled(False)
        self.blockSignals(True)
        try:
            # 1. Detach all items without deleting them
            while self.count() > 0:
                it = self.takeItem(0)
                if isinstance(it, FolderHeaderListWidgetItem):
                    self._folder_headers[it.folderPath()] = it
                elif isinstance(it, SortableListWidgetItem):
                    f_str = it.folderPath()
                    media_by_folder.setdefault(f_str, []).append(it)

            # Keep headers for folders that have media items OR are actively expected from pre-scan
            active_folders = set(media_by_folder.keys()) | set(self._folder_expected_totals.keys())
            self._folder_headers = {
                f_str: hdr
                for f_str, hdr in self._folder_headers.items()
                if f_str in active_folders
            }
            for f_str in active_folders:
                self.ensure_folder_header(f_str)

            sorted_folders = sorted(self._folder_headers.keys(), key=self._folder_group_key)
            descending = self._sort_order == Qt.SortOrder.DescendingOrder

            def _cmp_cards(a: SortableListWidgetItem, b: SortableListWidgetItem) -> int:
                if a.compare_with(b, self._sorting_mode, descending=descending):
                    return -1
                if b.compare_with(a, self._sorting_mode, descending=descending):
                    return 1
                return 0

            # 2. Re-insert each folder group header followed by its independently sorted cards
            for f_str in sorted_folders:
                header = self._folder_headers[f_str]
                label = self._format_folder_label(f_str)
                header.setText(label)
                header.setToolTip(label)
                super().addItem(header)

                cards = media_by_folder.get(f_str, [])
                cards.sort(key=cmp_to_key(_cmp_cards))
                for card in cards:
                    super().addItem(card)

            self._update_folder_group_visibility()
            if isinstance(current_selected, SortableListWidgetItem) and not current_selected.isHidden():
                self.setCurrentItem(current_selected)
        finally:
            self.blockSignals(False)
            self.setUpdatesEnabled(True)

        self.scheduleDelayedItemsLayout()
        self.viewport().update()

    # --- Filtering API ---

    def filterCriteria(self) -> FilterCriteria:
        return FilterCriteria(
            search_text=self._search_text,
            allowed_file_types=self._allowed_file_types,
            min_dimension=self._min_image_dimension,
            custom_predicate=self._custom_predicate,
        )

    def setSearchText(self, text: str) -> None:
        cleaned = text.strip()
        if cleaned != self._search_text:
            self._search_text = cleaned
            self.apply_filters()

    def setAllowedFileTypes(self, file_types: Optional[Iterable[str]]) -> None:
        normalized = frozenset(file_types) if file_types is not None else None
        if normalized != self._allowed_file_types:
            self._allowed_file_types = normalized
            self.apply_filters()

    def minImageDimension(self) -> Optional[SortableImageDimension]:
        return self._min_image_dimension

    def setMinImageDimension(
        self, imgdim: Optional[SortableImageDimension]
    ) -> None:
        if imgdim != self._min_image_dimension:
            self._min_image_dimension = imgdim
            self.apply_filters()

    def setMinResolution(self, min_width: int, min_height: int) -> None:
        """Convenience setter for width/height sliders or spinboxes."""
        if min_width <= 0 and min_height <= 0:
            self.setMinImageDimension(None)
        else:
            self.setMinImageDimension(
                SortableImageDimension(max(0, min_width), max(0, min_height))
            )

    def setCustomFilterPredicate(
        self, predicate: Optional[Callable[[SortableListWidgetItem], bool]]
    ) -> None:
        self._custom_predicate = predicate
        self.apply_filters()

    def _update_folder_group_visibility(self) -> int:
        """Evaluate global filter criteria across all folder groups, update `shown / total`
        counts and loading progress on every folder header, and hide/show cards based on
        their folder's expanded state.
        """
        criteria = self.filterCriteria()
        direct_totals: dict[str, int] = {f: 0 for f in self._folder_headers}
        direct_matches: dict[str, int] = {f: 0 for f in self._folder_headers}
        total_matching_media = 0

        # Pass 1: Evaluate filters on all media items and tally per-folder counts
        for row in range(self.count()):
            item = self.item(row)
            if isinstance(item, SortableListWidgetItem):
                f_str = item.folderPath()
                direct_totals[f_str] = direct_totals.get(f_str, 0) + 1
                is_match = item.matches_filter(criteria)
                if is_match:
                    direct_matches[f_str] = direct_matches.get(f_str, 0) + 1
                    total_matching_media += 1
                is_expanded = self._folder_expanded.get(f_str, True)
                item.setHidden(not (is_match and is_expanded))

        # Pass 2: Update each folder header's `shown / total` badge and loading progress
        for f_str, header in self._folder_headers.items():
            vis_c = direct_matches.get(f_str, 0)
            loaded_c = direct_totals.get(f_str, 0)
            expected_c = self._folder_expected_totals.get(f_str, 0)
            tot_c = max(loaded_c, expected_c)

            self._folder_loaded_counts[f_str] = loaded_c
            self._folder_visible_counts[f_str] = vis_c

            header.setCounts(vis_c, tot_c)
            if expected_c > 0 and loaded_c < expected_c:
                header.setProgress(loaded_c / expected_c)
            else:
                header.setProgress(-1.0)
            header.setHidden(tot_c == 0)

        return total_matching_media

    def apply_filters(self) -> int:
        """Evaluate all items against current FilterCriteria and update visibility."""
        criteria = self.filterCriteria()

        self.setUpdatesEnabled(False)
        try:
            visible_count = self._recalculate_visibility_and_counts()
        finally:
            self.setUpdatesEnabled(True)

        self.scheduleDelayedItemsLayout()
        self.viewport().update()
        self.filterChanged.emit(criteria)
        self.visibleCountChanged.emit(visible_count)
        return visible_count

    # --- Item Iteration Helpers ---

    def sortable_items(self) -> list[SortableListWidgetItem]:
        """Return all SortableListWidgetItems currently in the list (excluding folder headers)."""
        items: list[SortableListWidgetItem] = []
        for row in range(self.count()):
            item = self.item(row)
            if isinstance(item, SortableListWidgetItem):
                items.append(item)
        return items

    def max_image_dimension(self) -> SortableImageDimension:
        """Compute maximum width and height across all items (useful for slider bounds)."""
        max_w = 0
        max_h = 0
        for item in self.sortable_items():
            dim = item.imageDimension()
            if dim is not None:
                max_w = max(max_w, dim.width)
                max_h = max(max_h, dim.height)
        return SortableImageDimension(max_w, max_h)