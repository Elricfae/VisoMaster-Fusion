"""Tests for optional PyNvVideoCodec VRAM decode/encode pipeline and OpenCV/FFmpeg fallbacks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

import app.helpers.miscellaneous as misc_helpers
from app.helpers.vr_utils import EquirectangularConverter, PerspectiveConverter
from app.processors.video_utils.nv_video_codec import (
    NvVideoCapture,
    NvVideoEncoder,
    apply_gpu_rotation,
    is_pynv_available,
    open_video_capture,
    reset_pynv_availability_cache,
    resize_gpu_chw,
)
from app.processors.video_utils.sequential_detector import SequentialDetector
from app.processors.video_utils.video_encoding import FFmpegEncoder
from app.processors.workers.frame_worker import FrameWorker


def _requires_pynv():
    if not is_pynv_available():
        pytest.skip("PyNvVideoCodec + CUDA not available in this environment")


def _base_sdr_control(**overrides):
    ctrl = {
        "PyNvVideoCodecEncodeToggle": True,
        "HDREncodeToggle": False,
        "FrameEnhancerEnableToggle": False,
        "FrameEnhancerDownToggle": False,
        "FFMpegOptionsToggle": True,
        "FFAutoMatchSourceQualityToggle": False,
        "NvEncCodecSelection": "hevc",
        "NvEncTuningSelection": "high_quality",
        "FFPresetsSDRSelection": "p4",
        "FFQualitySlider": 18,
        "FFSpatialAQToggle": True,
        "FFTemporalAQToggle": True,
    }
    ctrl.update(overrides)
    return ctrl


# ---------------------------------------------------------------------------
# Pure tensor helpers (run on CPU or CUDA)
# ---------------------------------------------------------------------------


def test_apply_gpu_rotation_angles():
    t = torch.zeros((3, 2, 4), dtype=torch.uint8)
    t[:, 0, 0] = torch.tensor([10, 20, 30], dtype=torch.uint8)
    t[:, 0, 3] = torch.tensor([40, 50, 60], dtype=torch.uint8)

    assert torch.equal(apply_gpu_rotation(t, 0), t)

    r90 = apply_gpu_rotation(t, 90)
    assert r90.shape == (3, 4, 2)
    assert r90[:, 0, 1].tolist() == [10, 20, 30]

    r180 = apply_gpu_rotation(t, 180)
    assert r180.shape == (3, 2, 4)
    assert r180[:, 1, 3].tolist() == [10, 20, 30]

    r270 = apply_gpu_rotation(t, 270)
    assert r270.shape == (3, 4, 2)
    assert r270[:, 3, 0].tolist() == [10, 20, 30]


def test_resize_gpu_chw_even_width_alignment():
    t = torch.full((3, 720, 1280), 128, dtype=torch.uint8)
    out = resize_gpu_chw(t, target_height=360)
    assert out.shape == (3, 360, 640)
    assert out.dtype == torch.uint8

    t_odd = torch.full((3, 300, 301), 64, dtype=torch.uint8)
    out_odd = resize_gpu_chw(t_odd, target_height=150)
    assert out_odd.shape[1] == 150
    assert out_odd.shape[2] % 2 == 0


# ---------------------------------------------------------------------------
# Missing PyNvVideoCodec module (un-updated venv) graceful fallback test
# ---------------------------------------------------------------------------


def test_missing_pynv_module_does_not_crash_and_falls_back(tmp_path, monkeypatch):
    """Simulates a user venv without PyNvVideoCodec installed: zero crashes, clean OpenCV/FFmpeg fallback."""
    test_mp4 = str(tmp_path / "no_pynv_src.mp4")
    writer = cv2.VideoWriter(
        test_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 180)
    )
    for i in range(6):
        writer.write(np.full((180, 320, 3), (i * 40) % 256, dtype=np.uint8))
    writer.release()

    # Setting sys.modules["PyNvVideoCodec"] = None causes Python import to raise ModuleNotFoundError
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", None)
    reset_pynv_availability_cache()
    try:
        assert is_pynv_available() is False

        # Decoding gracefully falls back to OpenCV even with prefer_gpu=True
        cap = open_video_capture(test_mp4, prefer_gpu=True)
        try:
            assert isinstance(cap, cv2.VideoCapture)
            assert getattr(cap, "is_gpu_backed", False) is False
            assert cap.isOpened() is True
            ok, frame = misc_helpers.read_frame_gpu(cap)
            assert ok is True and isinstance(frame, np.ndarray)
            assert frame.shape == (180, 320, 3)
        finally:
            misc_helpers.release_capture(cap)

        # Encoding gracefully declines NVENC so FFmpegEncoder uses FFmpeg subprocess
        nv_enc = NvVideoEncoder()
        assert (
            nv_enc.start(
                output_filename=str(tmp_path / "no_pynv_out.mp4"),
                out_width=320,
                out_height=180,
                fps=25.0,
                control=_base_sdr_control(PyNvVideoCodecEncodeToggle=True),
                ffquality=20,
            )
            is False
        )
    finally:
        monkeypatch.undo()
        reset_pynv_availability_cache()


# ---------------------------------------------------------------------------
# End-to-end GPU NVENC -> MP4 -> GPU NVDEC round-trip
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_nv_vram_encode_and_decode_roundtrip(tmp_path):
    _requires_pynv()

    out_mp4 = str(tmp_path / "vram_roundtrip.mp4")
    width, height = 640, 360
    num_frames = 30
    fps = 29.97002997

    encoder = FFmpegEncoder()
    assert encoder.start_process(
        output_filename=out_mp4,
        frame_width=width,
        frame_height=height,
        fps=fps,
        control=_base_sdr_control(),
        is_segment=False,
    )
    assert encoder.is_gpu_encoder is True
    assert encoder.is_running() is True

    for i in range(num_frames):
        frame_gpu = torch.zeros((3, height, width), dtype=torch.uint8, device="cuda")
        frame_gpu[0].fill_((i * 8) % 256)
        frame_gpu[1].fill_(120)
        frame_gpu[2].fill_(255 - ((i * 8) % 256))
        assert encoder.write_frame(frame_gpu) is True

    encoder.close_process()
    assert encoder.is_running() is False
    assert os.path.isfile(out_mp4)
    assert os.path.getsize(out_mp4) > 0

    if shutil.which("ffprobe"):
        probe = json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_streams",
                    "-count_frames",
                    out_mp4,
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        stream = probe["streams"][0]
        assert stream["codec_name"] == "hevc"
        assert int(stream["nb_read_frames"]) == num_frames
        assert stream["avg_frame_rate"] == "30000/1001"

    cap = open_video_capture(out_mp4)
    try:
        assert isinstance(cap, NvVideoCapture)
        assert cap.isOpened() is True
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == width
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == height
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == num_frames
        assert abs(cap.get(cv2.CAP_PROP_FPS) - fps) < 0.05

        decoded_count = 0
        for i in range(num_frames):
            ok, tensor_gpu = misc_helpers.read_frame_gpu(cap)
            assert ok is True
            assert isinstance(tensor_gpu, torch.Tensor)
            assert tensor_gpu.is_cuda is True
            assert tensor_gpu.shape == (3, height, width)
            assert tensor_gpu.dtype == torch.uint8

            px = tensor_gpu[:, height // 2, width // 2].tolist()
            expected = [(i * 8) % 256, 120, 255 - ((i * 8) % 256)]
            for c in range(3):
                assert abs(px[c] - expected[c]) <= 6, (
                    f"Frame {i} channel {c} mismatch: got {px}, expected {expected}"
                )
            decoded_count += 1

        assert decoded_count == num_frames
        eof_ok, eof_frame = misc_helpers.read_frame_gpu(cap)
        assert eof_ok is False
        assert eof_frame is None

        assert misc_helpers.seek_frame(cap, 15) is True
        ok15, t15 = misc_helpers.read_frame_gpu(cap)
        assert ok15 is True and isinstance(t15, torch.Tensor)
        px15 = t15[:, height // 2, width // 2].tolist()
        exp15 = [(15 * 8) % 256, 120, 255 - ((15 * 8) % 256)]
        for c in range(3):
            assert abs(px15[c] - exp15[c]) <= 6

        assert misc_helpers.seek_frame(cap, 0) is True
        ok_r, t_resized = misc_helpers.read_frame_gpu(cap, preview_target_height=180)
        assert ok_r is True and isinstance(t_resized, torch.Tensor)
        assert t_resized.shape == (3, 180, 320)

        assert misc_helpers.seek_frame(cap, 0) is True
        ok_cpu, frame_bgr = misc_helpers.read_frame(cap, preview_target_height=180)
        assert ok_cpu is True and isinstance(frame_bgr, np.ndarray)
        assert frame_bgr.shape == (180, 320, 3)
        bgr_px = frame_bgr[90, 160].tolist()
        assert abs(bgr_px[0] - 255) <= 6
        assert abs(bgr_px[1] - 120) <= 6
        assert abs(bgr_px[2] - 0) <= 6
    finally:
        misc_helpers.release_capture(cap)
        assert cap.isOpened() is False


@pytest.mark.gpu
def test_nv_encoder_downscale_and_segment_mode(tmp_path):
    _requires_pynv()

    src_mp4 = str(tmp_path / "src.mp4")
    seg_mp4 = str(tmp_path / "segment_000.mp4")

    enc_src = FFmpegEncoder()
    assert enc_src.start_process(
        output_filename=src_mp4,
        frame_width=640,
        frame_height=360,
        fps=30.0,
        control=_base_sdr_control(),
    )
    for _ in range(10):
        assert enc_src.write_frame(
            torch.full((3, 360, 640), 100, dtype=torch.uint8, device="cuda")
        )
    enc_src.close_process()

    enc_seg = FFmpegEncoder()
    assert enc_seg.start_process(
        output_filename=seg_mp4,
        frame_width=2560,
        frame_height=1440,
        fps=30.0,
        control=_base_sdr_control(FrameEnhancerDownToggle=True),
        is_segment=True,
        media_path=src_mp4,
        start_time_sec=0.0,
        end_time_sec=0.33,
    )
    assert enc_seg.is_gpu_encoder is True
    for _ in range(8):
        assert enc_seg.write_frame(
            torch.full((3, 1440, 2560), 180, dtype=torch.uint8, device="cuda")
        )
    enc_seg.close_process()

    assert os.path.isfile(seg_mp4)
    cap = open_video_capture(seg_mp4)
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1920
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 1080
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    finally:
        misc_helpers.release_capture(cap)


# ---------------------------------------------------------------------------
# Automatic Fallback Tests (OpenCV & FFmpeg subprocess)
# ---------------------------------------------------------------------------


def test_disable_env_triggers_clean_opencv_and_ffmpeg_fallback(tmp_path, monkeypatch):
    test_mp4 = str(tmp_path / "fallback_src.mp4")
    writer = cv2.VideoWriter(
        test_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 180)
    )
    for i in range(6):
        img = np.full((180, 320, 3), (i * 30) % 256, dtype=np.uint8)
        writer.write(img)
    writer.release()

    monkeypatch.setenv("VISOMASTER_DISABLE_PYNV", "1")
    reset_pynv_availability_cache()
    try:
        assert is_pynv_available() is False

        cap = open_video_capture(test_mp4)
        try:
            assert isinstance(cap, cv2.VideoCapture)
            assert getattr(cap, "is_gpu_backed", False) is False
            assert cap.isOpened() is True
            ok, frame = misc_helpers.read_frame_gpu(cap)
            assert ok is True
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (180, 320, 3)
        finally:
            misc_helpers.release_capture(cap)

        nv_enc = NvVideoEncoder()
        assert (
            nv_enc.start(
                output_filename=str(tmp_path / "should_decline.mp4"),
                out_width=320,
                out_height=180,
                fps=25.0,
                control=_base_sdr_control(),
                ffquality=20,
            )
            is False
        )
    finally:
        monkeypatch.delenv("VISOMASTER_DISABLE_PYNV", raising=False)
        reset_pynv_availability_cache()


def test_hdr_encode_declines_nv_encoder_for_libx265_fallback(tmp_path):
    nv_enc = NvVideoEncoder()
    assert (
        nv_enc.start(
            output_filename=str(tmp_path / "hdr.mp4"),
            out_width=640,
            out_height=360,
            fps=30.0,
            control=_base_sdr_control(HDREncodeToggle=True),
            ffquality=18,
        )
        is False
    )


def test_pynv_encode_toggle_disabled_declines_nv_encoder(tmp_path):
    nv_enc = NvVideoEncoder()
    assert (
        nv_enc.start(
            output_filename=str(tmp_path / "disabled_nvenc.mp4"),
            out_width=640,
            out_height=360,
            fps=30.0,
            control=_base_sdr_control(PyNvVideoCodecEncodeToggle=False),
            ffquality=18,
        )
        is False
    )


@pytest.mark.gpu
def test_unsupported_codec_falls_back_to_opencv(tmp_path):
    _requires_pynv()
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required to synthesize FFV1 test clip")

    ffv1_avi = str(tmp_path / "unsupported_ffv1.avi")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:d=0.2:r=25",
            "-c:v",
            "ffv1",
            ffv1_avi,
        ],
        check=True,
    )

    cap = open_video_capture(ffv1_avi)
    try:
        assert isinstance(cap, cv2.VideoCapture)
        assert cap.isOpened() is True
        ok, frame = misc_helpers.read_frame_gpu(cap)
        assert ok is True
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (180, 320, 3)
    finally:
        misc_helpers.release_capture(cap)


# ---------------------------------------------------------------------------
# Codecs, Tuning Modes, and UI Toggle Integration
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    "codec,tuning",
    [
        ("h264", "high_quality"),
        ("hevc", "low_latency"),
        ("av1", "high_quality"),
        ("av1", "lossless"),
    ],
)
def test_nv_encoder_codecs_and_tuning_modes(tmp_path, codec, tuning):
    _requires_pynv()

    out_mp4 = str(tmp_path / f"test_{codec}_{tuning}.mp4")
    enc = FFmpegEncoder()
    assert enc.start_process(
        output_filename=out_mp4,
        frame_width=640,
        frame_height=360,
        fps=30.0,
        control=_base_sdr_control(
            PyNvVideoCodecEncodeToggle=True,
            NvEncCodecSelection=codec,
            NvEncTuningSelection=tuning,
        ),
    )
    assert enc.is_gpu_encoder is True
    for i in range(8):
        frame_gpu = torch.full(
            (3, 360, 640), (i * 25) % 256, dtype=torch.uint8, device="cuda"
        )
        assert enc.write_frame(frame_gpu) is True
    enc.close_process()

    assert os.path.isfile(out_mp4)
    cap = open_video_capture(out_mp4)
    try:
        assert isinstance(cap, NvVideoCapture)
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    finally:
        misc_helpers.release_capture(cap)


def test_handle_pynv_decode_toggle_switches_capture_live():
    from app.ui.widgets.actions.control_actions import handle_pynv_decode_toggle

    reopen_calls = []

    class DummyVP:
        file_type = "video"
        media_path = "clip.mp4"
        processing = False
        is_processing_segments = False
        current_frame_number = 42
        media_capture = SimpleNamespace(is_gpu_backed=False)
        _cached_raw_frame_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        _cached_raw_frame_number = 42

        def _reopen_video_capture(self, frame_idx: int) -> bool:
            reopen_calls.append(frame_idx)
            return True

    vp = DummyVP()
    mw = SimpleNamespace(
        control={"PyNvVideoCodecDecodeToggle": True}, video_processor=vp
    )

    handle_pynv_decode_toggle(mw, False)
    assert mw.control["PyNvVideoCodecDecodeToggle"] is False
    assert reopen_calls == [42]
    assert vp._cached_raw_frame_bgr is None
    assert vp._cached_raw_frame_number is None


# ---------------------------------------------------------------------------
# Pipeline integration: SequentialDetector, FrameWorker, VR converters
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_sequential_detector_gpu_downscale_without_cpu_frame():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    captured_inputs = []

    def fake_run_detect(img_tensor, *args, **kwargs):
        captured_inputs.append(tuple(img_tensor.shape))
        bboxes = np.array([[100.0, 50.0, 200.0, 150.0]], dtype=np.float32)
        kpss_5 = np.array(
            [
                [
                    [120.0, 80.0],
                    [180.0, 80.0],
                    [150.0, 100.0],
                    [130.0, 130.0],
                    [170.0, 130.0],
                ]
            ],
            dtype=np.float32,
        )
        return bboxes, kpss_5, np.empty((0, 5, 2), dtype=np.float32)

    fn_worker = SimpleNamespace(
        run_detect=fake_run_detect,
        run_recognize_direct=lambda *a, **k: (np.ones(512, dtype=np.float32), None),
        run_detect_landmark=lambda *a, **k: (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            None,
        ),
        reset_face_tracker=lambda: None,
    )
    main_window = SimpleNamespace(
        models_processor=SimpleNamespace(device=torch.device("cuda")),
        function_worker=fn_worker,
        target_faces={},
        default_parameters=SimpleNamespace(data={}),
    )

    det = SequentialDetector(main_window)
    frame_4k_gpu = torch.zeros((3, 2160, 3840), dtype=torch.uint8, device="cuda")

    bboxes, kpss_5, _, _ = det.run(
        frame_rgb=None,
        local_control_for_worker={
            "DetectorModelSelection": "RetinaFace",
            "LandmarkDetectToggle": False,
            "ShowAllDetectedFacesBBoxToggle": True,
            "ForceSwapToggle": True,
        },
        frame_tensor=frame_4k_gpu,
        frame_number=0,
    )

    assert captured_inputs == [(3, 288, 512)]
    assert bboxes.shape == (1, 4)
    assert abs(float(bboxes[0, 0]) - 750.0) < 1.0
    assert abs(float(bboxes[0, 1]) - 375.0) < 1.0


@pytest.mark.gpu
def test_frame_worker_zero_copy_gpu_tensor_inout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    main_window = SimpleNamespace(
        models_processor=SimpleNamespace(
            device=torch.device("cuda"), device_type="cuda"
        ),
        video_processor=SimpleNamespace(
            file_type="video",
            recording=True,
            is_processing_segments=False,
            encoder=SimpleNamespace(is_gpu_encoder=True),
        ),
        function_worker=SimpleNamespace(
            set_frame_edits_transforms=lambda *a, **k: None,
        ),
        swapfacesButton=SimpleNamespace(isChecked=lambda: True),
        editFacesButton=SimpleNamespace(isChecked=lambda: False),
        faceCompareCheckBox=SimpleNamespace(isChecked=lambda: False),
        faceMaskCheckBox=SimpleNamespace(isChecked=lambda: False),
        target_faces={},
        default_parameters=SimpleNamespace(data={}),
    )

    worker = FrameWorker(main_window=main_window, is_single_frame=False)
    worker.standard_processor.process_standard_frame = MagicMock(
        side_effect=lambda t, ctrl, ev: t
    )

    gpu_in = torch.full((3, 720, 1280), 42, dtype=torch.uint8, device="cuda")
    worker.frame = gpu_in
    worker.frame_number = 0
    control = {
        "VR180ModeEnableToggle": False,
        "FrameEnhancerEnableToggle": False,
        "get_cropped_face_kpsTypeSelection": "BILINEAR",
        "original_face_128_384TypeSelection": "BILINEAR",
        "original_face_512TypeSelection": "BILINEAR",
        "UntransformTypeSelection": "BILINEAR",
        "ScalebackFrameTypeSelection": "BILINEAR",
        "expression_faceeditor_t256TypeSelection": "BILINEAR",
        "expression_faceeditor_backTypeSelection": "BILINEAR",
        "block_shiftTypeSelection": "BILINEAR",
        "AntialiasTypeSelection": False,
    }

    out_gpu = worker.process_frame(control, threading.Event(), keep_output_on_gpu=True)
    assert isinstance(out_gpu, torch.Tensor)
    assert out_gpu.is_cuda is True
    assert out_gpu.shape == (3, 720, 1280)

    out_cpu = worker.process_frame(control, threading.Event(), keep_output_on_gpu=False)
    assert isinstance(out_cpu, np.ndarray)
    assert out_cpu.shape == (720, 1280, 3)
    assert out_cpu.dtype == np.uint8


@pytest.mark.gpu
def test_vr_converters_accept_gpu_tensor_directly():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda")
    eq_gpu = torch.zeros((3, 360, 720), dtype=torch.uint8, device=device)

    e2p = EquirectangularConverter(eq_gpu, device=device)
    assert e2p.height == 360 and e2p.width == 720
    assert e2p.equirect_tensor_cxhxw_rgb_uint8.is_cuda is True

    p2e = PerspectiveConverter(eq_gpu, device=device)
    assert p2e.orig_height == 360 and p2e.orig_width == 720 and p2e.orig_channels == 3
