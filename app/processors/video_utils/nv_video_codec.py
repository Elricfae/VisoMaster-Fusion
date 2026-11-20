"""Hardware-accelerated VRAM video decoding and encoding via NVIDIA PyNvVideoCodec.

PyNvVideoCodec is strictly optional: if it is not installed in the user's Python
environment (or on non-NVIDIA / macOS / CPU systems), all functions and classes
in this module degrade cleanly without raising import errors and transparently
fall back to OpenCV (``cv2.VideoCapture``) and FFmpeg.

Provides:
- ``NvVideoCapture``: A thread-safe, ``cv2.VideoCapture``-compatible GPU decoder
  backed by ``PyNvVideoCodec.SimpleDecoder`` (NVDEC). Supports both zero-copy
  VRAM reads (``read_gpu`` returning ``[3, H, W]`` RGB ``uint8`` CUDA tensors)
  and standard CPU BGR ``read()`` compatibility for UI/legacy callers.
- ``open_video_capture``: Opens ``NvVideoCapture`` when supported on the current
  system/GPU/codec and transparently falls back to ``cv2.VideoCapture``.
- ``NvVideoEncoder``: Hardware NVENC encoder backed by
  ``PyNvVideoCodec.CreateEncoder`` + ``PyNvVideoCodec.FFmpegMuxer`` that encodes
  directly from CUDA ``[3, H, W]`` RGB ``uint8`` tensors without copying raw
  pixels back across PCIe.
"""

from __future__ import annotations

import fractions
import json
import os
import shutil
import subprocess
import threading
from typing import Any, Mapping, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_PYNV_CHECK_LOCK = threading.Lock()
_PYNV_AVAILABLE: Optional[bool] = None
_PYNV_MODULE: Any = None
_PYNV_MISSING_LOGGED: bool = False

# Cache ffprobe rotation metadata per (path, mtime_ns, size) to avoid repeated subprocesses.
_ROTATION_CACHE_LOCK = threading.Lock()
_ROTATION_CACHE: dict[Tuple[str, int, int], int] = {}
_ROTATION_CACHE_MAX = 256


def reset_pynv_availability_cache() -> None:
    """Clears cached PyNvVideoCodec probe state (primarily for unit tests)."""
    global _PYNV_AVAILABLE, _PYNV_MODULE, _PYNV_MISSING_LOGGED
    with _PYNV_CHECK_LOCK:
        _PYNV_AVAILABLE = None
        _PYNV_MODULE = None
        _PYNV_MISSING_LOGGED = False


def is_pynv_available() -> bool:
    """Returns True when CUDA + optional PyNvVideoCodec are available and enabled."""
    global _PYNV_AVAILABLE, _PYNV_MODULE, _PYNV_MISSING_LOGGED

    if os.environ.get("VISOMASTER_DISABLE_PYNV", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return False

    with _PYNV_CHECK_LOCK:
        if _PYNV_AVAILABLE is not None:
            return _PYNV_AVAILABLE

        if not torch.cuda.is_available():
            _PYNV_AVAILABLE = False
            return False

        try:
            import PyNvVideoCodec as nvc  # type: ignore[import-not-found]

            has_api = (
                hasattr(nvc, "SimpleDecoder")
                and hasattr(nvc, "CreateEncoder")
                and hasattr(nvc, "FFmpegMuxer")
                and hasattr(nvc, "OutputColorType")
            )
            _PYNV_MODULE = nvc if has_api else None
            _PYNV_AVAILABLE = bool(has_api)
        except Exception:
            _PYNV_MODULE = None
            _PYNV_AVAILABLE = False
            if not _PYNV_MISSING_LOGGED:
                _PYNV_MISSING_LOGGED = True
                print(
                    "[INFO] Optional package 'PyNvVideoCodec' is not installed; "
                    "using OpenCV/FFmpeg video decode and encode fallback."
                )

        return _PYNV_AVAILABLE


def _get_nvc() -> Any:
    if not is_pynv_available():
        raise RuntimeError("PyNvVideoCodec is not available in this environment.")
    return _PYNV_MODULE


def probe_video_rotation_ffprobe(media_path: str) -> int:
    """Probes video display rotation metadata (0, 90, 180, or 270 degrees).

    Unlike ``misc_helpers.get_video_rotation`` (which short-circuits to 0 when
    OpenCV supports ``CAP_PROP_ORIENTATION_AUTO``), NVDEC always decodes stored
    pixel orientation, so we query the container rotation tag once per file.
    """
    if not media_path or not os.path.isfile(media_path):
        return 0

    try:
        st = os.stat(media_path)
        cache_key = (os.path.abspath(media_path), st.st_mtime_ns, st.st_size)
    except OSError:
        cache_key = None

    if cache_key is not None:
        with _ROTATION_CACHE_LOCK:
            cached = _ROTATION_CACHE.get(cache_key)
            if cached is not None:
                return cached

    rotation = 0
    if shutil.which("ffprobe") is not None:
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-select_streams",
                "v:0",
                str(media_path),
            ]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
                check=False,
            )
            if res.returncode == 0 and res.stdout:
                data = json.loads(res.stdout)

                def _find_rot(obj: Any) -> Any:
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if k.lower() == "rotation":
                                return v
                            found = _find_rot(v)
                            if found is not None:
                                return found
                    elif isinstance(obj, list):
                        for item in obj:
                            found = _find_rot(item)
                            if found is not None:
                                return found
                    return None

                raw_rot = _find_rot(data)
                if raw_rot is not None:
                    angle = int(round(float(raw_rot))) % 360
                    if angle in (90, 180, 270):
                        rotation = angle
        except Exception:
            rotation = 0

    if cache_key is not None:
        with _ROTATION_CACHE_LOCK:
            if len(_ROTATION_CACHE) >= _ROTATION_CACHE_MAX:
                _ROTATION_CACHE.clear()
            _ROTATION_CACHE[cache_key] = rotation

    return rotation


def apply_gpu_rotation(tensor_chw: torch.Tensor, rotation: int) -> torch.Tensor:
    """Applies clockwise video rotation (90, 180, 270) to a ``[C, H, W]`` tensor."""
    angle = int(rotation) % 360
    if angle == 0:
        return tensor_chw
    # torch.rot90 rotates counter-clockwise for positive k; clockwise 90 deg = k=-1 (or 3).
    if angle == 90:
        return torch.rot90(tensor_chw, k=3, dims=(1, 2)).contiguous()
    if angle == 180:
        return torch.rot90(tensor_chw, k=2, dims=(1, 2)).contiguous()
    if angle == 270:
        return torch.rot90(tensor_chw, k=1, dims=(1, 2)).contiguous()
    return tensor_chw


def resize_gpu_chw(
    tensor_chw: torch.Tensor,
    target_height: int,
    target_width: Optional[int] = None,
) -> torch.Tensor:
    """Resizes a ``[C, H, W]`` uint8 CUDA tensor directly in VRAM.

    Matches ``misc_helpers.read_frame`` sizing rules (even-aligned width when
    computing from ``target_height``).
    """
    orig_h, orig_w = int(tensor_chw.shape[1]), int(tensor_chw.shape[2])
    if orig_h <= 0 or orig_w <= 0 or target_height <= 0:
        return tensor_chw

    if target_width is None:
        if target_height == orig_h:
            return tensor_chw
        aspect_ratio = orig_w / float(orig_h)
        out_w = int(target_height * aspect_ratio)
        if out_w % 2 != 0:
            out_w += 1
        out_w = max(2, out_w)
        out_h = int(target_height)
    else:
        out_h = int(target_height)
        out_w = int(target_width)
        if out_h == orig_h and out_w == orig_w:
            return tensor_chw

    is_downscale = out_h < orig_h or out_w < orig_w
    t_float = tensor_chw.unsqueeze(0).float()
    if is_downscale:
        resized = F.interpolate(
            t_float,
            size=(out_h, out_w),
            mode="area",
        )
    else:
        resized = F.interpolate(
            t_float,
            size=(out_h, out_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    return resized.squeeze(0).clamp_(0, 255).to(torch.uint8).contiguous()


class NvVideoCapture:
    """``cv2.VideoCapture``-compatible GPU video decoder backed by PyNvVideoCodec.

    Decodes directly into GPU VRAM in planar RGB (``[3, H, W]`` ``uint8``) via
    ``PyNvVideoCodec.SimpleDecoder``.
    """

    is_gpu_backed: bool = True

    def __init__(self, media_path: str, gpu_id: int = 0) -> None:
        if not media_path or not os.path.isfile(media_path):
            raise FileNotFoundError(f"Video file not found: {media_path}")

        nvc = _get_nvc()
        torch.cuda.init()

        self.media_path = str(media_path)
        self.gpu_id = int(gpu_id)
        self._lock = threading.RLock()
        self._closed = False
        self._orientation_auto = True

        # Dedicated CUDA stream for decode + dlpack clone + optional GPU rotate/resize
        self._stream = torch.cuda.Stream(device=self.gpu_id)

        # Open NVDEC SimpleDecoder directly into VRAM as planar RGB (CHW uint8)
        self._decoder = nvc.SimpleDecoder(
            self.media_path,
            gpu_id=self.gpu_id,
            cuda_stream=self._stream.cuda_stream,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGBP,
        )

        meta = self._decoder.get_stream_metadata()
        self._stored_width = int(getattr(meta, "width", 0) or 0)
        self._stored_height = int(getattr(meta, "height", 0) or 0)
        self._num_frames = int(getattr(meta, "num_frames", 0) or 0)
        self._fps = float(getattr(meta, "average_fps", 0.0) or 0.0)
        self._bitrate_bps = float(getattr(meta, "bitrate", 0.0) or 0.0)
        self._codec_name = str(getattr(meta, "codec_name", "") or "")

        if self._stored_width <= 0 or self._stored_height <= 0 or self._num_frames <= 0:
            self.release()
            raise RuntimeError(
                f"Invalid stream metadata from PyNvVideoCodec "
                f"({self._stored_width}x{self._stored_height}, frames={self._num_frames})"
            )

        # Probe container rotation so CAP_PROP_ORIENTATION_AUTO matches OpenCV behavior.
        self._container_rotation = probe_video_rotation_ffprobe(self.media_path)

        self._pos: int = 0
        self._needs_seek: bool = False

        # Sanity-verify that NVDEC actually decodes frame 0 cleanly (catches unsupported
        # pixel formats/profiles such as 4:2:2/12-bit/ProRes immediately on open).
        try:
            with torch.cuda.stream(self._stream):
                test_batch = self._decoder.get_batch_frames(1)
                if not test_batch:
                    raise RuntimeError("PyNvVideoCodec returned empty batch on frame 0")
                test_tensor = torch.from_dlpack(test_batch[0])
                if test_tensor.ndim != 3 or int(test_tensor.shape[0]) != 3:
                    raise RuntimeError(
                        f"Unexpected decoded tensor shape: {tuple(test_tensor.shape)}"
                    )
                del test_tensor, test_batch
            self._stream.synchronize()
            self._decoder.seek_to_index(0)
            self._pos = 0
            self._needs_seek = False
        except Exception:
            self.release()
            raise

    @property
    def codec_name(self) -> str:
        return self._codec_name

    def isOpened(self) -> bool:
        with self._lock:
            return not self._closed and self._decoder is not None

    def get(self, prop_id: int) -> float:
        with self._lock:
            if self._closed:
                return 0.0
            rot_swaps = self._orientation_auto and self._container_rotation in (90, 270)
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self._stored_height if rot_swaps else self._stored_width)
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self._stored_width if rot_swaps else self._stored_height)
            if prop_id == cv2.CAP_PROP_FPS:
                return float(self._fps)
            if prop_id == cv2.CAP_PROP_FRAME_COUNT:
                return float(self._num_frames)
            if prop_id == cv2.CAP_PROP_POS_FRAMES:
                return float(self._pos)
            if prop_id == cv2.CAP_PROP_BITRATE:
                return float(self._bitrate_bps / 1000.0) if self._bitrate_bps > 0 else 0.0
            if (
                hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO")
                and prop_id == cv2.CAP_PROP_ORIENTATION_AUTO
            ):
                return 1.0 if self._orientation_auto else 0.0
            return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        with self._lock:
            if self._closed or self._decoder is None:
                return False
            if (
                hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO")
                and prop_id == cv2.CAP_PROP_ORIENTATION_AUTO
            ):
                self._orientation_auto = bool(int(value))
                return True
            if prop_id == cv2.CAP_PROP_POS_FRAMES:
                target = int(round(value))
                if target < 0:
                    target = 0
                if target >= self._num_frames:
                    self._pos = target
                    self._needs_seek = False
                    return True
                try:
                    self._decoder.seek_to_index(target)
                    self._pos = target
                    self._needs_seek = False
                    return True
                except Exception:
                    self._pos = target
                    self._needs_seek = True
                    return False
            return False

    def read_gpu(
        self,
        rotation: int = 0,
        target_height: Optional[int] = None,
    ) -> Tuple[bool, Optional[torch.Tensor]]:
        """Decodes the next frame directly into VRAM as ``[3, H, W]`` RGB ``uint8``.

        Applies container auto-rotation (when enabled), optional explicit
        ``rotation``, and optional GPU resize to ``target_height`` completely in
        VRAM, then synchronizes the decoder stream so the returned tensor is safe
        to consume on any downstream CUDA stream or worker thread.
        """
        with self._lock:
            if self._closed or self._decoder is None:
                return False, None
            if self._pos < 0 or self._pos >= self._num_frames:
                return False, None

            try:
                if self._needs_seek:
                    self._decoder.seek_to_index(self._pos)
                    self._needs_seek = False

                with torch.cuda.stream(self._stream):
                    batch = self._decoder.get_batch_frames(1)
                    if not batch:
                        return False, None

                    decoded_frame = batch[0]
                    # Clone out of NVDEC's internal surface pool so downstream queues
                    # never alias or exhaust decoder surfaces.
                    tensor_chw = torch.from_dlpack(decoded_frame).clone()
                    del decoded_frame, batch

                    # Combine container display rotation (when auto-orientation is on)
                    # with any explicit caller rotation.
                    total_rot = (
                        (self._container_rotation if self._orientation_auto else 0)
                        + int(rotation or 0)
                    ) % 360
                    if total_rot != 0:
                        tensor_chw = apply_gpu_rotation(tensor_chw, total_rot)

                    if target_height is not None and target_height > 0:
                        tensor_chw = resize_gpu_chw(tensor_chw, int(target_height))

                # Ensure all decode/clone/rotate/resize ops finish before handing
                # the tensor across threads/CUDA streams.
                self._stream.synchronize()
                self._pos += 1
                return True, tensor_chw
            except Exception as e:
                print(
                    f"[WARN] NvVideoCapture.read_gpu failed at frame {self._pos}: {e}"
                )
                return False, None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """``cv2.VideoCapture.read()`` compatibility returning CPU BGR uint8."""
        ok, tensor_chw = self.read_gpu(rotation=0, target_height=None)
        if not ok or tensor_chw is None:
            return False, None
        # Convert RGB [C, H, W] -> BGR [H, W, C] on GPU before downloading once
        with torch.cuda.stream(self._stream):
            bgr_hwc = tensor_chw[[2, 1, 0]].permute(1, 2, 0).contiguous()
            frame_bgr = bgr_hwc.cpu().numpy()
        self._stream.synchronize()
        del tensor_chw, bgr_hwc
        return True, frame_bgr

    def release(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            dec = self._decoder
            self._decoder = None
            if dec is not None:
                try:
                    self._stream.synchronize()
                except Exception:
                    pass
                del dec

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def open_video_capture(
    media_path: str,
    prefer_gpu: bool = True,
    gpu_id: int = 0,
) -> Union[NvVideoCapture, cv2.VideoCapture]:
    """Opens a video file using NVDEC (``NvVideoCapture``) with OpenCV fallback.

    Falls back automatically to ``cv2.VideoCapture`` when:
    - ``prefer_gpu`` is False
    - CUDA or optional ``PyNvVideoCodec`` is unavailable (macOS, CPU, AMD/Intel GPUs, un-updated venv)
    - The file uses an unsupported codec/pixel format/container (e.g., ProRes, FFV1)
    """
    if prefer_gpu and media_path and is_pynv_available():
        try:
            cap = NvVideoCapture(media_path, gpu_id=gpu_id)
            print(
                f"[INFO] Opened video with PyNvVideoCodec NVDEC (VRAM): "
                f"{os.path.basename(media_path)} "
                f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                f"@ {cap.get(cv2.CAP_PROP_FPS):.3f}fps, codec={cap.codec_name})"
            )
            return cap
        except Exception as e:
            print(
                f"[INFO] PyNvVideoCodec unavailable for '{os.path.basename(str(media_path))}' "
                f"({e}); falling back to OpenCV VideoCapture."
            )

    cap_cv = cv2.VideoCapture(media_path)
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap_cv.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap_cv


def _fps_to_rational_and_timebase(fps: float) -> Tuple[int, int, int, int]:
    """Converts float FPS to ``(fps_num, fps_den, timebase_den, pts_inc)``.

    Ensures ``pts_inc`` is an exact integer tick count per frame with a high
    timebase (>= 90 kHz) so fractional frame rates (23.976, 29.97, 59.94) have
    zero drift.
    """
    safe_fps = float(fps) if fps and fps > 0 else 30.0
    # Recognize standard NTSC rates cleanly
    for standard_num, standard_den in (
        (24000, 1001),
        (30000, 1001),
        (60000, 1001),
        (120000, 1001),
    ):
        if abs(safe_fps - (standard_num / standard_den)) < 0.005:
            fps_num, fps_den = standard_num, standard_den
            break
    else:
        frac = fractions.Fraction(safe_fps).limit_denominator(100000)
        fps_num, fps_den = frac.numerator, frac.denominator

    tb_den = fps_num
    pts_inc = fps_den
    if tb_den < 90000:
        scale = (90000 + tb_den - 1) // tb_den
        tb_den *= scale
        pts_inc *= scale

    return int(fps_num), int(fps_den), int(tb_den), int(pts_inc)


class NvVideoEncoder:
    """Hardware NVENC video encoder consuming CUDA tensors directly from VRAM.

    Encodes ``[3, H, W]`` RGB ``uint8`` CUDA tensors via
    ``PyNvVideoCodec.CreateEncoder`` (``ABGR`` device input -> HEVC/H.264/AV1)
    and streams packets directly to disk via ``PyNvVideoCodec.FFmpegMuxer``.

    For segment mode (``is_segment=True``), writes a temporary video-only MP4
    and runs a fast stream-copy (``-c:v copy``) FFmpeg audio mux on close.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._encoder: Any = None
        self._muxer: Any = None
        self._stream: Optional[torch.cuda.Stream] = None
        self._rgba_buf: Optional[torch.Tensor] = None
        self._active: bool = False

        self._output_filename: str = ""
        self._mux_target_path: str = ""
        self._is_segment: bool = False
        self._media_path: Optional[str] = None
        self._start_time_sec: float = 0.0
        self._end_time_sec: float = 0.0

        self._out_width: int = 0
        self._out_height: int = 0
        self._codec: str = "hevc"
        self._pts_inc: int = 3000
        self._num_b_frames: int = 0
        self._out_packet_idx: int = 0
        self.frames_written: int = 0

    def is_running(self) -> bool:
        with self._lock:
            return self._active and self._encoder is not None and self._muxer is not None

    def start(
        self,
        output_filename: str,
        out_width: int,
        out_height: int,
        fps: float,
        control: Mapping[str, Any],
        ffquality: int,
        is_segment: bool = False,
        media_path: Optional[str] = None,
        start_time_sec: float = 0.0,
        end_time_sec: float = 0.0,
        gpu_id: int = 0,
    ) -> bool:
        """Initializes NVENC + FFmpegMuxer. Returns False to trigger FFmpeg fallback."""
        if not bool(control.get("PyNvVideoCodecEncodeToggle", True)):
            return False

        if not is_pynv_available():
            return False

        # HDR encoding relies on CPU libx265 with BT.2020 / SMPTE2084 mastering metadata.
        if control.get("HDREncodeToggle"):
            return False

        if (
            out_width <= 0
            or out_height <= 0
            or (out_width % 2 != 0)
            or (out_height % 2 != 0)
        ):
            return False

        with self._lock:
            self.close()

            try:
                nvc = _get_nvc()
                torch.cuda.init()

                codec = (
                    str(control.get("NvEncCodecSelection", "hevc") or "hevc")
                    .strip()
                    .lower()
                )
                if codec not in {"hevc", "h264", "av1"}:
                    codec = "hevc"

                tuning = (
                    str(
                        control.get("NvEncTuningSelection", "high_quality")
                        or "high_quality"
                    )
                    .strip()
                    .lower()
                )
                if tuning not in {
                    "high_quality",
                    "low_latency",
                    "ultra_low_latency",
                    "lossless",
                }:
                    tuning = "high_quality"

                preset_raw = str(
                    control.get("FFPresetsSDRSelection", "p5") or "p5"
                ).upper()
                if preset_raw not in {"P1", "P2", "P3", "P4", "P5", "P6", "P7"}:
                    preset_raw = "P5"

                spatial_aq = "1" if int(control.get("FFSpatialAQToggle", 0)) else "0"
                temporal_aq = "1" if int(control.get("FFTemporalAQToggle", 0)) else "0"
                cq_clamped = max(0, min(51, int(ffquality)))

                if codec == "av1" and (tuning == "lossless" or cq_clamped == 0):
                    print(
                        "[INFO] NVENC AV1 does not support lossless/QP 0; "
                        "using high_quality with constqp=1."
                    )
                    tuning = "high_quality"
                    cq_clamped = 1
                    av1_near_lossless = True
                else:
                    av1_near_lossless = False

                self._stream = torch.cuda.Stream(device=gpu_id)

                enc_kwargs: dict[str, Any] = {
                    "cudastream": self._stream.cuda_stream,
                    "codec": codec,
                    "preset": preset_raw,
                    "tuning_info": tuning,
                    "fps": f"{float(fps):.6f}",
                }
                if av1_near_lossless:
                    enc_kwargs["rc"] = "constqp"
                    enc_kwargs["constqp"] = "1"
                elif tuning == "lossless" or cq_clamped == 0:
                    enc_kwargs["rc"] = "constqp"
                    enc_kwargs["constqp"] = "0"
                else:
                    enc_kwargs["rc"] = "vbr"
                    enc_kwargs["cq"] = str(cq_clamped)
                    enc_kwargs["spatialaq"] = spatial_aq
                    enc_kwargs["temporalaq"] = temporal_aq

                encoder = nvc.CreateEncoder(
                    int(out_width),
                    int(out_height),
                    "ABGR",
                    False,
                    **enc_kwargs,
                )

                num_b_frames = (
                    int(encoder.GetNumBFrames())
                    if hasattr(encoder, "GetNumBFrames")
                    else 0
                )
                seq_params = (
                    encoder.GetSequenceParams()
                    if hasattr(encoder, "GetSequenceParams")
                    else b""
                )
                seq_bytes = bytes(seq_params) if seq_params else b""

                fps_num, fps_den, tb_den, pts_inc = _fps_to_rational_and_timebase(fps)

                if is_segment and media_path:
                    mux_target = f"{output_filename}.nvenc_video_only.mp4"
                else:
                    mux_target = output_filename

                if os.path.exists(mux_target):
                    try:
                        os.remove(mux_target)
                    except OSError:
                        pass

                muxer = nvc.FFmpegMuxer(
                    mux_target,
                    nvc.MEDIA_FORMAT.MP4,
                    codec,
                    int(out_width),
                    int(out_height),
                    fps_num,
                    fps_den,
                    1,
                    tb_den,
                    seq_bytes,
                )

                # Preallocate reusable VRAM RGBA (little-endian ABGR) staging buffer.
                with torch.cuda.stream(self._stream):
                    rgba_buf = torch.empty(
                        (int(out_height), int(out_width), 4),
                        dtype=torch.uint8,
                        device=f"cuda:{gpu_id}",
                    )
                    rgba_buf[:, :, 3].fill_(255)
                self._stream.synchronize()

                self._encoder = encoder
                self._muxer = muxer
                self._rgba_buf = rgba_buf
                self._output_filename = output_filename
                self._mux_target_path = mux_target
                self._is_segment = bool(is_segment)
                self._media_path = media_path
                self._start_time_sec = float(start_time_sec)
                self._end_time_sec = float(end_time_sec)
                self._out_width = int(out_width)
                self._out_height = int(out_height)
                self._codec = codec
                self._pts_inc = pts_inc
                self._num_b_frames = num_b_frames
                self._out_packet_idx = 0
                self.frames_written = 0
                self._active = True

                print(
                    f"[INFO] Started PyNvVideoCodec NVENC VRAM encoder: "
                    f"{out_width}x{out_height} @ {fps:.3f}fps "
                    f"({codec.upper()} {preset_raw}, tuning={tuning}, cq={cq_clamped}, "
                    f"spatial_aq={spatial_aq}, temporal_aq={temporal_aq})"
                )
                return True
            except Exception as e:
                print(
                    f"[INFO] PyNvVideoCodec NVENC initialization failed ({e}); "
                    "falling back to FFmpeg subprocess encoder."
                )
                self._cleanup_handles()
                return False

    def _mux_packets(self, packets: list[dict[str, Any]]) -> None:
        if not packets or self._muxer is None:
            return
        for pkt in packets:
            data = pkt.get("data")
            if not data:
                continue
            ts = int(pkt.get("timestamp", self._out_packet_idx))
            pic_type = int(pkt.get("picture_type", 0))
            # NV_ENC_PIC_TYPE_I = 2, NV_ENC_PIC_TYPE_IDR = 3
            is_key = 1 if pic_type in (2, 3) else 0
            pts = ts * self._pts_inc
            dts = (self._out_packet_idx - self._num_b_frames) * self._pts_inc
            self._muxer.Mux(
                data,
                pts,
                dts,
                self._pts_inc,
                0,
                is_key,
                self._out_packet_idx,
            )
            self._out_packet_idx += 1

    def write_frame_gpu(self, frame_chw_rgb: torch.Tensor) -> bool:
        """Encodes a ``[3, H, W]`` RGB ``uint8`` CUDA tensor directly from VRAM."""
        with self._lock:
            if not self.is_running() or self._rgba_buf is None or self._stream is None:
                return False
            try:
                src = frame_chw_rgb
                if src.device != self._rgba_buf.device:
                    src = src.to(self._rgba_buf.device, non_blocking=True)
                if src.dtype != torch.uint8:
                    src = src.clamp(0, 255).to(torch.uint8)

                # Ensure our encoder stream waits for whichever CUDA stream produced `src`
                # so we never read stale/uninitialized VRAM across streams.
                caller_stream = torch.cuda.current_stream(self._rgba_buf.device)
                if caller_stream != self._stream:
                    self._stream.wait_stream(caller_stream)

                with torch.cuda.stream(self._stream):
                    if src.is_cuda:
                        src.record_stream(self._stream)
                    if (
                        int(src.shape[1]) != self._out_height
                        or int(src.shape[2]) != self._out_width
                    ):
                        src = resize_gpu_chw(
                            src,
                            target_height=self._out_height,
                            target_width=self._out_width,
                        )
                    # Pack planar RGB [3, H, W] into interleaved RGBA [H, W, 4]
                    # (matches NVENC ABGR little-endian byte layout: R, G, B, A).
                    self._rgba_buf[:, :, :3].copy_(src.permute(1, 2, 0))
                    # Ensure copy into _rgba_buf completes before NVENC consumes it
                    # (PyNvEncoder may dispatch on its internal CUDA stream).
                    self._stream.synchronize()
                    packets = self._encoder.Encode(self._rgba_buf)

                self._mux_packets(packets)
                self.frames_written += 1
                return True
            except Exception as e:
                print(f"[WARN] NvVideoEncoder.write_frame_gpu failed: {e}")
                return False

    def write_frame_cpu_bgr(self, frame_bgr: np.ndarray) -> bool:
        """Uploads a CPU BGR ``[H, W, 3]`` uint8 array to VRAM and encodes via NVENC."""
        with self._lock:
            if not self.is_running() or self._rgba_buf is None or self._stream is None:
                return False
            try:
                bgr_contig = np.ascontiguousarray(frame_bgr)
                with torch.cuda.stream(self._stream):
                    gpu_bgr = torch.from_numpy(bgr_contig).to(
                        self._rgba_buf.device, non_blocking=True
                    )
                    gpu_rgb_chw = gpu_bgr[..., [2, 1, 0]].permute(2, 0, 1).contiguous()
                return self.write_frame_gpu(gpu_rgb_chw)
            except Exception as e:
                print(f"[WARN] NvVideoEncoder.write_frame_cpu_bgr failed: {e}")
                return False

    def _mux_segment_audio(self) -> None:
        """Stream-copies NVENC video + original segment audio into final segment MP4."""
        if not self._mux_target_path or not os.path.exists(self._mux_target_path):
            return

        if not self._media_path or not os.path.exists(self._media_path):
            os.replace(self._mux_target_path, self._output_filename)
            return

        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            self._mux_target_path,
            "-ss",
            str(self._start_time_sec),
            "-to",
            str(self._end_time_sec),
            "-i",
            self._media_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
        ]
        if self._codec == "hevc":
            args.extend(["-tag:v", "hvc1"])
        args.extend(
            [
                "-c:a",
                "aac",
                "-shortest",
                self._output_filename,
            ]
        )
        try:
            subprocess.run(args, check=True, capture_output=True, text=True, timeout=120)
            try:
                os.remove(self._mux_target_path)
            except OSError:
                pass
        except Exception as e:
            print(
                f"[WARN] NVENC segment audio mux failed ({e}); keeping video-only segment."
            )
            try:
                os.replace(self._mux_target_path, self._output_filename)
            except OSError:
                pass

    def _cleanup_handles(self) -> None:
        self._active = False
        if self._stream is not None:
            try:
                self._stream.synchronize()
            except Exception:
                pass
        self._rgba_buf = None
        self._muxer = None
        self._encoder = None
        self._stream = None

    def close(self) -> None:
        """Flushes NVENC B-frames, finalizes MP4 container, and releases GPU handles."""
        with self._lock:
            if not self._active and self._encoder is None and self._muxer is None:
                return

            try:
                if self._encoder is not None:
                    if self._stream is not None:
                        with torch.cuda.stream(self._stream):
                            end_packets = self._encoder.EndEncode()
                        self._stream.synchronize()
                    else:
                        end_packets = self._encoder.EndEncode()
                    self._mux_packets(end_packets)
            except Exception as e:
                print(f"[WARN] Error flushing NvVideoEncoder: {e}")

            try:
                if self._muxer is not None:
                    self._muxer.Finalize()
            except Exception as e:
                print(f"[WARN] Error finalizing NvVideoEncoder muxer: {e}")

            is_segment = self._is_segment
            self._cleanup_handles()

            # Segment mode: mux audio into final segment path after muxer file handle is closed.
            if is_segment and self._mux_target_path != self._output_filename:
                self._mux_segment_audio()
