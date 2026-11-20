"""Unit tests for Phase 1 + Phase 2 ROI-only crop/paste and CPU-resident 4K frame processing."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import kornia.geometry.transform as kgm
import numpy as np
import pytest
import torch
from skimage import transform as trans

from app.helpers.miscellaneous import ParametersDict
from app.processors.utils import faceutil
from app.processors.video_utils.sequential_detector import SequentialDetector
from app.processors.workers.frame_worker import FrameWorker
from app.processors.workers.frame_worker_standard import StandardProcessor


def _make_similarity_matrix(
    cx: float,
    cy: float,
    scale: float = 1.0,
    angle_deg: float = 0.0,
    dst_size: int = 512,
) -> np.ndarray:
    t = (
        trans.SimilarityTransform(translation=(-cx, -cy))
        + trans.SimilarityTransform(rotation=np.deg2rad(angle_deg), scale=scale)
        + trans.SimilarityTransform(translation=(dst_size / 2.0, dst_size / 2.0))
    )
    return t.params[:2, :].astype(np.float32)


@pytest.mark.parametrize(
    ("cx", "cy", "scale", "angle_deg", "mode", "padding_mode"),
    [
        (960.0, 540.0, 1.3, 0.0, "bilinear", "zeros"),
        (960.0, 540.0, 1.4, 27.5, "bicubic", "zeros"),
        (960.0, 540.0, 1.2, -18.0, "bilinear", "border"),
        # Face partially crossing the top-left border
        (45.0, 60.0, 1.5, 15.0, "bilinear", "zeros"),
        # Face partially crossing the bottom-right border
        (1890.0, 1040.0, 1.25, -22.0, "bilinear", "border"),
    ],
)
def test_forward_warp_roi_matches_full_frame_warp(
    cx, cy, scale, angle_deg, mode, padding_mode
):
    rng = np.random.default_rng(42)
    h, w = 1080, 1920
    img = torch.from_numpy(rng.integers(0, 256, size=(3, h, w), dtype=np.uint8))

    M = _make_similarity_matrix(cx, cy, scale=scale, angle_deg=angle_deg, dst_size=512)
    M_t = torch.from_numpy(M).float().unsqueeze(0)

    full_out = kgm.warp_affine(
        img.unsqueeze(0).float(),
        M_t,
        dsize=(512, 512),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    ).squeeze(0)

    roi_info = faceutil.compute_forward_warp_roi(M, h, w, 512, 512, pad=4)
    assert roi_info is not None
    x1, y1, x2, y2, M_roi = roi_info
    assert (x2 - x1) < w and (y2 - y1) < h

    roi_out = kgm.warp_affine(
        img[:, y1:y2, x1:x2].unsqueeze(0).float(),
        torch.from_numpy(M_roi).float().unsqueeze(0),
        dsize=(512, 512),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    ).squeeze(0)

    diff = (full_out - roi_out).abs()
    # Inputs are random [0..255] uint8: float32 normalized grid precision (~1e-4 px * 255) is ~0.07
    assert float(diff.max().item()) < 0.15, f"max_diff={diff.max().item()}"
    assert float(diff.mean().item()) < 0.01, f"mean_diff={diff.mean().item()}"
    assert int((full_out.round().int() - roi_out.round().int()).abs().max().item()) <= 1


@pytest.mark.parametrize(
    ("cx", "cy", "scale", "angle_deg"),
    [
        (960.0, 540.0, 1.4, 0.0),
        (960.0, 540.0, 1.1, 31.0),
        (40.0, 50.0, 1.5, -14.0),
        (1880.0, 1050.0, 1.2, 20.0),
    ],
)
def test_inverse_paste_roi_matches_full_frame_composite(cx, cy, scale, angle_deg):
    rng = np.random.default_rng(7)
    h, w = 1080, 1920
    bg = torch.from_numpy(rng.integers(0, 256, size=(3, h, w), dtype=np.uint8))
    swap = torch.from_numpy(rng.integers(0, 256, size=(3, 512, 512), dtype=np.uint8))
    # Smooth interior mask (0 on edges, 1 in center)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 512), torch.linspace(-1.0, 1.0, 512), indexing="ij"
    )
    mask = ((1.0 - (xx**2 + yy**2).clamp(0.0, 1.0)) * 1.2).clamp(0.0, 1.0).unsqueeze(0)

    M_fwd = _make_similarity_matrix(
        cx, cy, scale=scale, angle_deg=angle_deg, dst_size=512
    )
    M_inv = faceutil.invertAffineTransform(M_fwd).astype(np.float32)

    # Full-frame reference
    M_inv_t = torch.from_numpy(M_inv).float().unsqueeze(0)
    swap_full = kgm.warp_affine(
        (swap.float() * mask).unsqueeze(0),
        M_inv_t,
        dsize=(h, w),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    mask_full = kgm.warp_affine(
        mask.unsqueeze(0),
        M_inv_t,
        dsize=(h, w),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    ref = (swap_full + bg.float() * (1.0 - mask_full)).clamp(0, 255).to(torch.uint8)

    # Localized ROI version
    out = bg.clone()
    paste_roi = faceutil.compute_inverse_paste_roi(M_inv, 512, 512, h, w, pad=2)
    assert paste_roi is not None
    x1, y1, x2, y2, M_local = paste_roi
    assert (x2 - x1) < w and (y2 - y1) < h

    M_local_t = torch.from_numpy(M_local).float().unsqueeze(0)
    swap_roi = kgm.warp_affine(
        (swap.float() * mask).unsqueeze(0),
        M_local_t,
        dsize=(y2 - y1, x2 - x1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    mask_roi = kgm.warp_affine(
        mask.unsqueeze(0),
        M_local_t,
        dsize=(y2 - y1, x2 - x1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    out[:, y1:y2, x1:x2] = (
        (swap_roi + out[:, y1:y2, x1:x2].float() * (1.0 - mask_roi))
        .clamp(0, 255)
        .to(torch.uint8)
    )

    # Must match within 1 LSB uint8 everywhere and 0 outside ROI
    assert int((ref.int() - out.int()).abs().max().item()) <= 1
    outside_mask = torch.ones((h, w), dtype=torch.bool)
    outside_mask[y1:y2, x1:x2] = False
    assert torch.equal(out[:, outside_mask], bg[:, outside_mask])


def test_sequential_detector_uses_downscaled_detect_and_roi_crops_on_4k():
    seen_shapes: dict[str, tuple[int, ...]] = {}
    seen_rec_kps: list[np.ndarray] = []
    seen_lm_bbox: list[np.ndarray] = []

    def fake_run_detect(img_t, *_args, **_kwargs):
        seen_shapes["detect"] = tuple(img_t.shape)
        # Return one face in the middle of the downscaled image
        dh, dw = img_t.shape[1], img_t.shape[2]
        bx1, by1, bx2, by2 = dw * 0.45, dh * 0.4, dw * 0.55, dh * 0.6
        bbox = np.array([[bx1, by1, bx2, by2]], dtype=np.float32)
        cx, cy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
        kps5 = np.array(
            [
                [
                    [cx - 8, cy - 6],
                    [cx + 8, cy - 6],
                    [cx, cy],
                    [cx - 6, cy + 8],
                    [cx + 6, cy + 8],
                ]
            ],
            dtype=np.float32,
        )
        return bbox, kps5, None

    def fake_run_recognize(img_t, kps5, *_args, **_kwargs):
        seen_shapes["recognize"] = tuple(img_t.shape)
        seen_rec_kps.append(np.asarray(kps5, dtype=np.float32))
        return np.ones(512, dtype=np.float32), None

    def fake_run_landmark(img_t, bbox, kps5, **_kwargs):
        seen_shapes["landmark"] = tuple(img_t.shape)
        seen_lm_bbox.append(np.asarray(bbox, dtype=np.float32))
        # Return 68 landmarks centered at local bbox center
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        pts = np.tile(np.array([[cx, cy]], dtype=np.float32), (68, 1))
        return np.empty((0, 2), dtype=np.float32), pts, None

    fw = SimpleNamespace(
        run_detect=fake_run_detect,
        run_recognize_direct=fake_run_recognize,
        run_detect_landmark=fake_run_landmark,
        findCosineDistance=lambda _a, _b: 0.0,
    )
    mw = SimpleNamespace(
        target_faces={"t1": SimpleNamespace()},
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
        models_processor=SimpleNamespace(device="cpu"),
        function_worker=fw,
    )
    det = SequentialDetector(mw)
    frame_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)
    control = {
        "DetectorModelSelection": "RetinaFace",
        "MaxFacesToDetectSlider": 5,
        "DetectorScoreSlider": 50,
        "LandmarkDetectToggle": True,
        "LandmarkDetectModelSelection": "68",
        "LandmarkDetectScoreSlider": 50,
        "DetectFromPointsToggle": False,
        "AutoRotationToggle": False,
        "LandmarkMeanEyesToggle": False,
        "KPSSmoothingEnableToggle": False,
    }

    with patch(
        "app.processors.video_utils.sequential_detector.find_best_target_match",
        return_value=(object(), None, 0.9),
    ):
        bboxes, kpss_5, kpss, _ = det.run(frame_4k, control, {})

    # Detector received <= 512px max side, NOT 2160x3840
    assert max(seen_shapes["detect"][1:]) <= 512
    # Recognition and landmark received small face ROIs, NOT 2160x3840
    assert seen_shapes["recognize"][1] < 2160 and seen_shapes["recognize"][2] < 3840
    assert seen_shapes["landmark"][1] < 2160 and seen_shapes["landmark"][2] < 3840
    # Returned bboxes & landmarks are accurately mapped back to 4K coordinates (~1920, ~1080)
    assert np.isclose(0.5 * (bboxes[0, 0] + bboxes[0, 2]), 1920.0, atol=20.0)
    assert np.isclose(0.5 * (bboxes[0, 1] + bboxes[0, 3]), 1080.0, atol=20.0)
    assert np.isclose(kpss[0, 0, 0], 1920.0, atol=20.0)
    assert np.isclose(kpss[0, 0, 1], 1080.0, atol=20.0)


def test_standard_processor_roi_cpu_only_uploads_face_roi_and_preserves_background():
    frame_h, frame_w = 1080, 1920
    rng = np.random.default_rng(123)
    frame_rgb = rng.integers(10, 240, size=(frame_h, frame_w, 3), dtype=np.uint8)

    kps_5 = np.array(
        [
            [920.0, 500.0],
            [1000.0, 500.0],
            [960.0, 545.0],
            [930.0, 590.0],
            [990.0, 590.0],
        ],
        dtype=np.float32,
    )
    bbox = np.array([880.0, 460.0, 1040.0, 630.0], dtype=np.float32)

    captured_swap_shapes: list[tuple[int, ...]] = []

    def fake_swap_core(roi_gpu, kps_5_local, *_args, **_kwargs):
        captured_swap_shapes.append(tuple(roi_gpu.shape))
        # Paint a distinct constant RGB marker inside the ROI center
        ch, rh, rw = roi_gpu.shape
        assert ch == 3
        roi_out = roi_gpu.clone()
        roi_out[:, rh // 4 : (3 * rh) // 4, rw // 4 : (3 * rw) // 4] = torch.tensor(
            [250, 10, 20], dtype=torch.uint8
        ).view(3, 1, 1)
        return roi_out, None, None

    default_params = {
        "SwapModelSelection": "Inswapper128",
        "DFMModelSelection": "",
        "FaceAdjEnableToggle": False,
        "FaceScaleAmountSlider": 0,
        "FaceMakeupEnableToggle": False,
        "HairMakeupEnableToggle": False,
        "EyeBrowsMakeupEnableToggle": False,
        "LipsMakeupEnableToggle": False,
        "SimilarityThresholdSlider": 50,
    }
    target_face = SimpleNamespace(
        face_id=1,
        assigned_kv_map=None,
        assigned_input_faces={},
        assigned_input_embedding={
            "Inswapper128ArcFace": np.ones(512, dtype=np.float32)
        },
        get_embedding=lambda _m: np.ones(512, dtype=np.float32),
    )

    worker = FrameWorker.__new__(FrameWorker)
    worker.is_single_frame = False
    worker.lock = threading.Lock()
    worker._MIN_FACE_PIXELS = 20
    worker.precomputed_bboxes = np.array([bbox], dtype=np.float32)
    worker.precomputed_kpss_5 = np.array([kps_5], dtype=np.float32)
    worker.precomputed_kpss = None
    worker.precomputed_kpss_203 = None
    worker.parameters = {"1": ParametersDict(default_params, default_params)}
    worker.models_processor = SimpleNamespace(
        device="cpu", device_type="cpu", model_lock=threading.RLock()
    )
    worker.main_window = SimpleNamespace(
        swapfacesButton=SimpleNamespace(isChecked=lambda: True),
        editFacesButton=SimpleNamespace(isChecked=lambda: False),
        target_faces={"1": target_face},
        default_parameters=SimpleNamespace(data=default_params),
    )
    worker.function_worker = SimpleNamespace(
        run_recognize_direct=lambda img_t, _kps, _s, _m: (
            np.ones(512, dtype=np.float32),
            None,
        ),
        get_arcface_model=lambda _m: "Inswapper128ArcFace",
    )
    worker._find_best_target_match = lambda *_a, **_kw: (
        target_face,
        ParametersDict(default_params, default_params),
        0.95,
    )
    worker._apply_auto_mouth = lambda p, *_a, **_kw: p
    worker.swap_core = fake_swap_core

    control = {
        "RecognitionModelSelection": "Inswapper128ArcFace",
        "SwapOnlyBestMatchEnableToggle": False,
        "ForceSwapToggle": True,
    }

    out_bgr = StandardProcessor(worker).process_standard_frame_roi_cpu(
        frame_rgb, control, threading.Event()
    )

    assert out_bgr.shape == (frame_h, frame_w, 3)
    assert out_bgr.flags["C_CONTIGUOUS"]
    # swap_core received a tight face ROI, NOT the full 1080x1920 frame!
    assert len(captured_swap_shapes) == 1
    assert captured_swap_shapes[0][1] < frame_h
    assert captured_swap_shapes[0][2] < frame_w

    # Far corners outside face ROI are bit-for-bit identical to original BGR
    orig_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    assert np.array_equal(out_bgr[:100, :100], orig_bgr[:100, :100])
    assert np.array_equal(out_bgr[-100:, -100:], orig_bgr[-100:, -100:])
    # Center of face has the swapped BGR marker [20, 10, 250]
    assert np.array_equal(out_bgr[540, 960], np.array([20, 10, 250], dtype=np.uint8))


def test_frame_worker_process_frame_routes_to_cpu_roi_or_full_gpu_fallback():
    def _make_worker():
        w = FrameWorker.__new__(FrameWorker)
        w.frame_number = 0
        w.last_processed_frame_number = -1
        w._last_scaling_control = None
        w.set_scaling_transforms = MagicMock()
        w.is_view_face_mask = False
        w.is_view_face_compare = False
        w.models_processor = SimpleNamespace(device="cpu", device_type="cuda")
        w.standard_processor = SimpleNamespace(
            process_standard_frame_roi_cpu=MagicMock(
                return_value=np.zeros((720, 1280, 3), dtype=np.uint8)
            ),
            process_standard_frame=MagicMock(
                return_value=torch.zeros((3, 720, 1280), dtype=torch.uint8)
            ),
        )
        w.vr_processor = SimpleNamespace(
            process_frame_vr180=MagicMock(
                return_value=torch.zeros((3, 720, 1280), dtype=torch.uint8)
            )
        )
        w.function_worker = SimpleNamespace(
            enhance_core=MagicMock(
                return_value=torch.zeros((3, 720, 1280), dtype=torch.uint8)
            )
        )
        w.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        return w

    base_control = {
        "VR180ModeEnableToggle": False,
        "FrameEnhancerEnableToggle": False,
        "FaceDetailerEnableToggle": False,
        "ManualRotationEnableToggle": False,
        "LandmarkDetectToggle": True,
        "YawNetEnableToggle": False,
        "ShowAllDetectedFacesBBoxToggle": False,
        "ShowByteTrackBBoxToggle": False,
        "ShowLandmarksEnableToggle": False,
        "ShowYawNetRingToggle": False,
    }

    # Standard HD/4K CUDA frame -> uses CPU-resident ROI fast path
    w1 = _make_worker()
    out1 = w1.process_frame(dict(base_control), threading.Event())
    assert out1.shape == (720, 1280, 3)
    w1.standard_processor.process_standard_frame_roi_cpu.assert_called_once()
    w1.standard_processor.process_standard_frame.assert_not_called()

    # Any full-frame feature (e.g. FrameEnhancer, FaceDetailer, YawNet, Overlay) -> falls back to full GPU path
    for flag in (
        "FrameEnhancerEnableToggle",
        "FaceDetailerEnableToggle",
        "ManualRotationEnableToggle",
        "YawNetEnableToggle",
        "ShowLandmarksEnableToggle",
    ):
        w_fb = _make_worker()
        ctl = dict(base_control)
        ctl[flag] = True
        w_fb.process_frame(ctl, threading.Event())
        w_fb.standard_processor.process_standard_frame_roi_cpu.assert_not_called()
        w_fb.standard_processor.process_standard_frame.assert_called_once()
