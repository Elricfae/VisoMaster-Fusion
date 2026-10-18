# app/helpers/thumbnail_manager.py
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, PngImagePlugin
from PySide6 import QtCore

from app.ui.widgets.sortable_widgets import MediaMetadata, MediaType

IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".jpe",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
    ".jp2",
    ".bmp",
    ".exr",
    ".hdr",
    ".ras",
    ".pnm",
    ".ppm",
    ".pgm",
    ".pbm",
    ".pfm",
)

VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".3gp",
    ".gif",
)

# High-resolution thumbnail bounds (16:9 widescreen master with high-DPI headroom)
DEFAULT_THUMB_MAX_WIDTH: int = 320
DEFAULT_THUMB_MAX_HEIGHT: int = 180


# =============================================================================
# 1. FILE DISCOVERY & MEDIA PROBING HELPERS
# =============================================================================


def iter_media_files(
    directory: str | Path,
    *,
    include_images: bool = True,
    include_videos: bool = True,
    include_subfolders: bool = False,
) -> Iterator[Path]:
    """Yield matching image and/or video files from a directory."""
    root_str = os.path.abspath(str(directory))
    if not os.path.isdir(root_str):
        return

    exts: list[str] = []
    if include_images:
        exts.extend(IMAGE_EXTENSIONS)
    if include_videos:
        exts.extend(VIDEO_EXTENSIONS)
    if not exts:
        return
    allowed_exts = tuple(exts)

    if include_subfolders:
        for dirpath, dirnames, filenames in os.walk(root_str):
            dirnames.sort(key=str.casefold)
            for fname in sorted(filenames, key=str.casefold):
                if fname.lower().endswith(allowed_exts):
                    yield Path(os.path.join(dirpath, fname))
    else:
        try:
            entries = sorted(os.scandir(root_str), key=lambda e: e.name.casefold())
        except OSError:
            return
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith(allowed_exts):
                yield Path(entry.path)


def get_file_type(file_path: str | Path) -> Optional[str]:
    """Return 'image', 'video', or None based on file extension."""
    lower = str(file_path).lower()
    if lower.endswith(IMAGE_EXTENSIONS):
        return MediaType.IMAGE.value
    if lower.endswith(VIDEO_EXTENSIONS):
        return MediaType.VIDEO.value
    return None


def read_image_bgr(image_path: str | Path) -> Optional[np.ndarray]:
    """Unicode-safe OpenCV image reader returning a BGR numpy array."""
    try:
        img_array = np.fromfile(str(image_path), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def get_scaled_resolution(
    media_width: int,
    media_height: int,
    max_width: int = DEFAULT_THUMB_MAX_WIDTH,
    max_height: int = DEFAULT_THUMB_MAX_HEIGHT,
) -> Tuple[int, int]:
    """Calculate aspect-ratio-preserving dimensions within (max_width, max_height)."""
    if media_width <= 0 or media_height <= 0:
        return 0, 0
    if media_width > max_width or media_height > max_height:
        scale = min(max_width / media_width, max_height / media_height)
        return max(1, int(round(media_width * scale))), max(1, int(round(media_height * scale)))
    return max(1, int(media_width)), max(1, int(media_height))


def get_video_rotation(media_path: str | Path) -> int:
    """Return 0, 90, 180, or 270. Skips ffprobe when OpenCV supports auto-rotation."""
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        return 0
    if shutil.which("ffprobe") is None:
        return 0

    try:
        process = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-select_streams",
                "v:0",
                str(media_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            check=False,
        )
        if process.returncode != 0:
            return 0
        data = json.loads(process.stdout)

        def _find_rotation(obj: object) -> object:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.lower() == "rotation":
                        return v
                    res = _find_rotation(v)
                    if res is not None:
                        return res
            elif isinstance(obj, list):
                for item in obj:
                    res = _find_rotation(item)
                    if res is not None:
                        return res
            return None

        raw = _find_rotation(data)
        if raw is not None:
            angle = int(float(str(raw))) % 360
            if 85 <= angle <= 95:
                return 90
            if 175 <= angle <= 185:
                return 180
            if 265 <= angle <= 275:
                return 270
    except Exception:
        pass
    return 0


def probe_media_metadata(
    media_file_path: str | Path, file_type: Optional[str] = None
) -> Optional[MediaMetadata]:
    """Probe dimensions, frame count, FPS, and bitrate without full decoding."""
    resolved_type = file_type or get_file_type(media_file_path)
    if resolved_type == MediaType.IMAGE.value:
        try:
            with Image.open(media_file_path) as img:
                width, height = img.size
            return MediaMetadata(width=int(width), height=int(height), total_frames=1)
        except Exception:
            return None

    if resolved_type != MediaType.VIDEO.value:
        return None

    cap: Optional[cv2.VideoCapture] = None
    try:
        cap = cv2.VideoCapture(str(media_file_path))
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = float(cap.get(cv2.CAP_PROP_FPS))
        bitrate_kbits = float(cap.get(cv2.CAP_PROP_BITRATE))
    except Exception:
        return None
    finally:
        if cap is not None:
            cap.release()

    if get_video_rotation(media_file_path) in (90, 270):
        width, height = height, width

    return MediaMetadata(
        width=max(0, width),
        height=max(0, height),
        total_frames=max(0, total_frames),
        frame_rate=max(0.0, frame_rate),
        bitrate_kbits=max(0.0, bitrate_kbits),
    )


def extract_representative_video_frame(
    video_path: str | Path,
    metadata: Optional[MediaMetadata] = None,
) -> Optional[np.ndarray]:
    """Extract a frame from the middle of a video (50% duration/frame count)
    to avoid black fade-in frames at index 0. Falls back to frame 0 if seeking fails.
    """
    video_str = str(video_path)
    cap = cv2.VideoCapture(video_str)
    frame: Optional[np.ndarray] = None

    try:
        if not cap.isOpened():
            return None

        total_frames = (
            metadata.total_frames
            if (metadata is not None and metadata.total_frames > 0)
            else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )

        target_frame = max(0, total_frames // 2) if total_frames > 1 else 0
        if target_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_frame))
            ret, decoded = cap.read()
            if ret and decoded is not None:
                frame = decoded

        if frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
            ret, decoded = cap.read()
            if ret and decoded is not None:
                frame = decoded
    except Exception:
        return None
    finally:
        cap.release()

    if frame is not None:
        rotation = get_video_rotation(video_str)
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame


# =============================================================================
# 2. THUMBNAIL MANAGER (Disk + Embedded EXIF/PNG Metadata Cache)
# =============================================================================


class ThumbnailManager:
    """Thread-safe disk thumbnail & embedded MediaMetadata cache manager."""

    _METADATA_KEY = "VisoMasterMediaMetadata"
    _METADATA_SCHEMA_VERSION = 3  # Bumped to v3 for 320x180 high-res thumbnails
    _JPEG_USER_COMMENT_TAG = 0x9286
    _JPEG_USER_COMMENT_PREFIX = b"ASCII\x00\x00\x00"

    def __init__(
        self,
        thumbnail_dir: str | Path = ".thumbnails",
        max_width: int = DEFAULT_THUMB_MAX_WIDTH,
        max_height: int = DEFAULT_THUMB_MAX_HEIGHT,
    ) -> None:
        self.thumbnail_dir = os.path.abspath(str(thumbnail_dir))
        self.max_width = max_width
        self.max_height = max_height
        self._lock = threading.Lock()
        os.makedirs(self.thumbnail_dir, exist_ok=True)

    def _get_file_hash(self, file_path: str) -> str:
        name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        hash_input = f"{name}_{file_size}"
        return hashlib.md5(hash_input.encode("utf-8")).hexdigest()

    def get_thumbnail_path(self, file_path: str | Path) -> Tuple[str, str]:
        file_hash = self._get_file_hash(str(file_path))
        png_path = os.path.join(self.thumbnail_dir, f"{file_hash}.png")
        jpg_path = os.path.join(self.thumbnail_dir, f"{file_hash}.jpg")
        return png_path, jpg_path

    def find_existing_thumbnail(self, file_path: str | Path) -> Optional[str]:
        try:
            png_path, jpg_path = self.get_thumbnail_path(str(file_path))
        except OSError:
            return None
        with self._lock:
            if os.path.exists(png_path):
                return png_path
            if os.path.exists(jpg_path):
                return jpg_path
        return None

    def _build_metadata_payload(
        self, file_path: str, metadata: Optional[MediaMetadata]
    ) -> Optional[str]:
        if metadata is None:
            return None
        try:
            stat = os.stat(file_path)
        except OSError:
            return None

        payload = {
            "version": self._METADATA_SCHEMA_VERSION,
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "media": {
                "width": int(metadata.width),
                "height": int(metadata.height),
                "total_frames": int(metadata.total_frames),
                "frame_rate": float(metadata.frame_rate),
                "bitrate_kbits": float(metadata.bitrate_kbits),
            },
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)

    def _metadata_from_payload(
        self, payload: dict, file_path: str
    ) -> Optional[MediaMetadata]:
        try:
            stat = os.stat(file_path)
        except OSError:
            return None

        if (
            int(payload.get("version", -1)) != self._METADATA_SCHEMA_VERSION
            or int(payload.get("source_size", -1)) != int(stat.st_size)
            or int(payload.get("source_mtime_ns", -1)) != int(stat.st_mtime_ns)
        ):
            return None

        media = payload.get("media", {})
        try:
            return MediaMetadata(
                width=max(0, int(media.get("width", 0))),
                height=max(0, int(media.get("height", 0))),
                total_frames=max(0, int(media.get("total_frames", 0))),
                frame_rate=max(0.0, float(media.get("frame_rate", 0.0))),
                bitrate_kbits=max(0.0, float(media.get("bitrate_kbits", 0.0))),
            )
        except (TypeError, ValueError):
            return None

    def get_thumbnail_metadata(
        self, thumbnail_path: str | Path, file_path: str | Path
    ) -> Optional[MediaMetadata]:
        """Read cached MediaMetadata embedded in an existing thumbnail."""
        try:
            with self._lock:
                with Image.open(thumbnail_path) as image:
                    if image.format == "PNG":
                        raw_payload = image.info.get(self._METADATA_KEY)
                    elif image.format == "JPEG":
                        raw_payload = image.getexif().get(self._JPEG_USER_COMMENT_TAG)
                    else:
                        raw_payload = None
        except Exception:
            return None

        if isinstance(raw_payload, bytes):
            if raw_payload.startswith(self._JPEG_USER_COMMENT_PREFIX):
                raw_payload = raw_payload[len(self._JPEG_USER_COMMENT_PREFIX) :]
            try:
                raw_payload = raw_payload.decode("ascii")
            except UnicodeDecodeError:
                return None
        if not isinstance(raw_payload, str):
            return None

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return None
        return self._metadata_from_payload(payload, str(file_path))

    def create_thumbnail(
        self,
        frame: np.ndarray,
        file_path: str | Path,
        metadata: Optional[MediaMetadata] = None,
    ) -> Optional[str]:
        """Save a crisp 320x180 max thumbnail with embedded metadata and return its path."""
        file_str = str(file_path)
        png_path, jpg_path = self.get_thumbnail_path(file_str)

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        height, width = frame.shape[:2]
        scaled_w, scaled_h = get_scaled_resolution(
            media_width=width,
            media_height=height,
            max_width=self.max_width,
            max_height=self.max_height,
        )
        resized_frame = cv2.resize(
            frame, (max(1, scaled_w), max(1, scaled_h)), interpolation=cv2.INTER_LANCZOS4
        )
        metadata_payload = self._build_metadata_payload(file_str, metadata)

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        try:
            pnginfo = PngImagePlugin.PngInfo()
            if metadata_payload:
                pnginfo.add_text(self._METADATA_KEY, metadata_payload)
            with self._lock:
                image.save(png_path, format="PNG", optimize=True, pnginfo=pnginfo)
                # If PNG is larger than 80KB, fallback to quality JPEG
                if os.path.getsize(png_path) > 80 * 1024:
                    os.remove(png_path)
                    raise ValueError("PNG > 80KB; fallback to JPEG")
                if os.path.exists(jpg_path):
                    os.remove(jpg_path)
            return png_path
        except Exception:
            exif_bytes = b""
            if metadata_payload:
                exif = Image.Exif()
                exif[self._JPEG_USER_COMMENT_TAG] = (
                    self._JPEG_USER_COMMENT_PREFIX + metadata_payload.encode("ascii")
                )
                exif_bytes = exif.tobytes()
            with self._lock:
                save_kwargs: dict = {
                    "format": "JPEG",
                    "quality": 95,
                    "optimize": True,
                    "progressive": True,
                }
                if exif_bytes:
                    save_kwargs["exif"] = exif_bytes
                image.save(jpg_path, **save_kwargs)
                if os.path.exists(png_path):
                    os.remove(png_path)
            return jpg_path

    def ensure_thumbnail_and_metadata(
        self, file_path: str | Path
    ) -> Tuple[Optional[str], Optional[MediaMetadata]]:
        """Fast-path: return cached thumbnail + embedded metadata, or generate both."""
        file_str = str(file_path)
        file_type = get_file_type(file_str)
        if not file_type:
            return None, None

        existing_thumb = self.find_existing_thumbnail(file_str)
        if existing_thumb:
            cached_meta = self.get_thumbnail_metadata(existing_thumb, file_str)
            if cached_meta is not None:
                return existing_thumb, cached_meta

        # Cache miss or stale metadata: probe media & extract representative frame
        metadata = probe_media_metadata(file_str, file_type)
        frame: Optional[np.ndarray] = None

        if file_type == MediaType.IMAGE.value:
            frame = read_image_bgr(file_str)
        elif file_type == MediaType.VIDEO.value:
            frame = extract_representative_video_frame(file_str, metadata=metadata)

        if frame is not None:
            thumb_path = self.create_thumbnail(frame, file_str, metadata=metadata)
            return thumb_path, metadata

        return existing_thumb, metadata


# =============================================================================
# 3. BACKGROUND QT MEDIA SCANNER WORKER
# =============================================================================


@dataclass(frozen=True)
class DiscoveredMediaItem:
    """Immutable packet emitted from background scanner to UI thread."""

    file_path: Path
    media_type: str
    thumbnail_path: Optional[str]
    metadata: Optional[MediaMetadata]
    folder_key: Optional[str] = None


class MediaScannerWorker(QtCore.QThread):
    """Background worker that pre-counts media per folder, then streams thumbnails + metadata in batches."""

    foldersDiscovered = QtCore.Signal(object)      # dict[str, int] mapping folder_key -> total media count
    itemDiscovered = QtCore.Signal(object)         # Single DiscoveredMediaItem (kept for compatibility)
    itemsBatchDiscovered = QtCore.Signal(object)   # list[DiscoveredMediaItem] time-sliced batch
    scanFinished = QtCore.Signal(int)              # total discovered count

    BATCH_INTERVAL_SEC: float = 0.035  # ~28 FPS batch flush to keep UI 100% responsive
    MAX_BATCH_SIZE: int = 24

    def __init__(
        self,
        paths: Sequence[Path],
        thumbnail_manager: ThumbnailManager,
        *,
        is_folder: bool = True,
        include_images: bool = True,
        include_videos: bool = True,
        include_subfolders: bool = False,
        virtual_folder_key: Optional[str] = None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._paths = tuple(paths)
        self._thumbnail_manager = thumbnail_manager
        self._is_folder = is_folder
        self._include_images = include_images
        self._include_videos = include_videos
        self._include_subfolders = include_subfolders
        self._virtual_folder_key = virtual_folder_key
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        self._stop_requested.set()

    def _iter_all_candidates(self) -> Iterator[Path]:
        if self._is_folder:
            for folder_path in self._paths:
                yield from iter_media_files(
                    folder_path,
                    include_images=self._include_images,
                    include_videos=self._include_videos,
                    include_subfolders=self._include_subfolders,
                )
        else:
            yield from self._paths

    def run(self) -> None:
        # 1. Fast pre-scan using os.path.abspath (no slow Windows kernel Path.resolve syscalls)
        discovered_files: list[tuple[Path, str, str]] = []
        folder_totals: dict[str, int] = {}
        seen_keys: set[tuple[str, str]] = set()

        for file_path in self._iter_all_candidates():
            if self._stop_requested.is_set():
                return
            file_str = os.path.abspath(str(file_path))
            mtype = get_file_type(file_str)
            if not mtype:
                continue
            folder_str = (
                self._virtual_folder_key
                if self._virtual_folder_key is not None
                else os.path.dirname(file_str)
            )
            dedup_key = (folder_str, os.path.normcase(file_str))
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            discovered_files.append((Path(file_str), mtype, folder_str))
            folder_totals[folder_str] = folder_totals.get(folder_str, 0) + 1

        if self._stop_requested.is_set():
            return

        # Emit pre-counted folder totals immediately so headers render 0 / total + progress bar
        if folder_totals:
            self.foldersDiscovered.emit(folder_totals)

        # 2. Generate/load thumbnails and emit in time-sliced batches so the UI thread never blocks
        count = 0
        batch: list[DiscoveredMediaItem] = []
        last_flush = time.perf_counter()

        for file_path, mtype, folder_str in discovered_files:
            if self._stop_requested.is_set():
                break

            thumb_path, metadata = (
                self._thumbnail_manager.ensure_thumbnail_and_metadata(file_path)
            )
            if self._stop_requested.is_set():
                break

            batch.append(
                DiscoveredMediaItem(
                    file_path=file_path,
                    media_type=mtype,
                    thumbnail_path=thumb_path,
                    metadata=metadata,
                    folder_key=folder_str,
                )
            )
            count += 1

            now = time.perf_counter()
            if len(batch) >= self.MAX_BATCH_SIZE or (now - last_flush) >= self.BATCH_INTERVAL_SEC:
                self.itemsBatchDiscovered.emit(batch)
                batch = []
                last_flush = now

        if batch and not self._stop_requested.is_set():
            self.itemsBatchDiscovered.emit(batch)

        self.scanFinished.emit(count)