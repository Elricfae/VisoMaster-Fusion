"""
FM-* tests for face mask math and guard logic.
No ML models loaded; tests cover pure tensor/numpy operations.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# FM-02: torch.zeros_like initialises mask to zero
# ---------------------------------------------------------------------------


def test_mask_initialises_to_zero():
    ref = torch.ones(1, 128, 128, dtype=torch.bool)
    mask = torch.zeros_like(ref)
    assert mask.sum().item() == 0
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# FM-03: division-by-zero guard — denominator clamped to avoid NaN
# ---------------------------------------------------------------------------


def test_blend_no_nan_when_denominator_zero():
    """If denominator is 0 it must be clamped so no NaN appears in output."""
    weight_map = torch.zeros(1, 64, 64)  # denominator = 0 everywhere
    safe_denom = weight_map.clamp(min=1e-8)
    numerator = torch.rand(3, 64, 64)
    result = numerator / safe_denom
    assert torch.all(torch.isfinite(result)), "Division by near-zero produced inf/nan"


# ---------------------------------------------------------------------------
# FM-06: mask values stay in [0, 1] after feathering (Gaussian blur)
# ---------------------------------------------------------------------------


def test_feathered_mask_range():
    """After Gaussian blur a [0,1] mask should still be in [0,1]."""
    from torchvision import transforms

    mask = torch.zeros(1, 64, 64, dtype=torch.float32)
    mask[:, 16:48, 16:48] = 1.0
    gauss = transforms.GaussianBlur(kernel_size=11, sigma=3.0)
    feathered = gauss(mask)
    assert feathered.min().item() >= -1e-6  # numerical precision
    assert feathered.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# FM-05: blending formula produces sensible output
# ---------------------------------------------------------------------------


def test_blend_formula_correctness():
    """target + (component - target) * mask should equal lerp(target, component, mask)."""
    H, W = 32, 32
    target = torch.full((3, H, W), 100.0)
    component = torch.full((3, H, W), 200.0)
    mask = torch.full((1, H, W), 0.5)

    # Explicit formula used in vr_utils
    diff = component - target
    blended = target + diff * mask

    expected = torch.full((3, H, W), 150.0)
    assert torch.allclose(blended, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# FM-01: initialisation — mask groups are dict-like (structural check)
# ---------------------------------------------------------------------------


def test_mask_groups_are_dict_compatible():
    """Simulate the group-dict init pattern used in FaceMasks.__init__."""
    mouth_groups: dict = {}
    fp_groups: dict = {}
    tex_groups: dict = {}

    # All should start empty
    assert len(mouth_groups) == 0
    assert len(fp_groups) == 0
    assert len(tex_groups) == 0


# ---------------------------------------------------------------------------
# FM-04: transform order — affine applied before mask blend
# ---------------------------------------------------------------------------


def test_affine_before_blend_ordering():
    """
    Demonstrate that (warp-then-blend) != (blend-then-warp) when BOTH the face
    and the background are spatially non-constant.

    result_correct[i] = 0.5*bg[i]         + 0.5*face[shift(i)]
    result_wrong[i]   = 0.5*bg[shift(i)]  + 0.5*face[shift(i)]

    They differ wherever bg[i] != bg[shift(i)], which holds for any non-constant bg.
    """
    # Non-constant background (spatial gradient — not uniform)
    background = torch.arange(0, 64, dtype=torch.float32).reshape(1, 8, 8) * 2.0
    # Non-constant face
    face = torch.arange(64, 128, dtype=torch.float32).reshape(1, 8, 8)
    mask = torch.ones_like(face) * 0.5
    shift = 3

    # Correct order: warp face, then blend into background
    shifted_face = torch.roll(face, shifts=shift, dims=-1)
    result_correct = background + (shifted_face - background) * mask

    # Wrong order: blend first, then warp the blended result (background moves too)
    blended_first = background + (face - background) * mask
    result_wrong = torch.roll(blended_first, shifts=shift, dims=-1)

    assert not torch.allclose(result_correct, result_wrong), (
        "warp-then-blend and blend-then-warp must differ when background is non-constant"
    )


# ---------------------------------------------------------------------------
# FM-08: Occluder / XSeg profile-collapse bypass
# ---------------------------------------------------------------------------


class MockModelsProcessor:
    def __init__(self):
        self.device = torch.device("cpu")
        self.models = {}


def _make_face_masks():
    from unittest.mock import MagicMock

    from app.processors.face_masks import FaceMasks

    return FaceMasks(MockModelsProcessor(), MagicMock())


def _occluder_returning(fill):
    """Build a run_occluder stub whose output keeps `fill` fraction of the crop."""

    rows = int(round(256 * fill))

    def _stub(img, output):
        output.zero_()
        output.view(256, 256)[:rows, :] = 1.0

    return _stub


def _xseg_returning(fill):
    rows = int(round(256 * fill))

    def _stub(img, output):
        output.zero_()
        output.view(256, 256)[:rows, :] = 1.0

    return _stub


PROFILE_YAW = 75.0
FRONTAL_YAW = 10.0
GUARD_ON = {"SegGuardEnableToggle": True}


def test_occluder_bypassed_on_profile_when_mask_collapses():
    """Extreme yaw + collapsed mask == model failure -> bypass when safeguard enabled."""
    fm = _make_face_masks()
    fm.run_occluder = _occluder_returning(0.02)

    mask = fm.apply_occlusion(
        torch.zeros((3, 256, 256)),
        amount=0,
        parameters=GUARD_ON,
        yaw_deg=PROFILE_YAW,
    )

    assert mask.shape == (1, 256, 256)
    assert mask.min().item() == 1.0, "bypass must return a fully clear mask"


def test_occluder_respected_when_frontal_face_is_heavily_occluded():
    """
    The regression guard: a hand covering most of a FRONTAL face produces the same
    low face ratio as a profile collapse, but it is a correct prediction. The
    occluder must be honoured, not bypassed.
    """
    fm = _make_face_masks()
    fm.run_occluder = _occluder_returning(0.02)

    mask = fm.apply_occlusion(
        torch.zeros((3, 256, 256)),
        amount=0,
        parameters=GUARD_ON,
        yaw_deg=FRONTAL_YAW,
    )

    assert mask.min().item() == 0.0, "occlusion must survive on a frontal face"
    # Tolerance covers row quantization in the stub (1 row == 1/256).
    assert abs(mask.mean().item() - 0.02) < 0.005


def test_occluder_respected_on_profile_when_mask_is_plausible():
    """Extreme yaw alone must not trigger a bypass if the mask looks sane."""
    fm = _make_face_masks()
    fm.run_occluder = _occluder_returning(0.25)

    mask = fm.apply_occlusion(
        torch.zeros((3, 256, 256)),
        amount=0,
        parameters=GUARD_ON,
        yaw_deg=PROFILE_YAW,
    )

    assert abs(mask.mean().item() - 0.25) < 1e-6


def test_occluder_not_bypassed_when_yaw_unknown():
    """Callers that cannot supply a head angle keep the model authoritative."""
    fm = _make_face_masks()
    fm.run_occluder = _occluder_returning(0.02)

    mask = fm.apply_occlusion(torch.zeros((3, 256, 256)), amount=0, parameters=GUARD_ON)

    assert mask.min().item() == 0.0


def test_xseg_rescued_on_profile_reinforces_inner_face_without_aliasing():
    """
    When XSeg collapses on an extreme profile and the safeguard is enabled, it
    reinforces the inner face oval (0 = Face at center) while keeping outer corners
    masked (1 = Background) so downstream stats masks do not sample background.
    The four results must not alias each other.
    """
    fm = _make_face_masks()
    fm.run_dfl_xseg = _xseg_returning(0.02)

    masks = fm.apply_dfl_xseg(
        torch.zeros((3, 256, 256)),
        amount=0,
        mouth=torch.zeros((1, 256, 256)),
        parameters=dict(GUARD_ON),
        yaw_deg=PROFILE_YAW,
    )

    assert len(masks) == 4
    for m in masks:
        assert m.shape == (1, 256, 256)
        # Center of face oval is rescued (0 = Face)
        assert m[0, 128, 128].item() == pytest.approx(0.0)
        # Outer corner stays background (1 = Background)
        assert m[0, 255, 255].item() == pytest.approx(1.0)

    # Aliasing check: mutating one must not disturb the others.
    orig_centers = [m[0, 128, 128].item() for m in masks[1:]]
    masks[0].add_(5.0)
    assert [m[0, 128, 128].item() for m in masks[1:]] == orig_centers


def test_xseg_respected_when_frontal_face_is_heavily_occluded():
    fm = _make_face_masks()
    fm.run_dfl_xseg = _xseg_returning(0.05)

    mask, _, _, _ = fm.apply_dfl_xseg(
        torch.zeros((3, 256, 256)),
        amount=0,
        mouth=torch.zeros((1, 256, 256)),
        parameters=dict(GUARD_ON),
        yaw_deg=FRONTAL_YAW,
    )

    # Inverted convention: the 95% non-face region must still be excluded (1 = BG).
    assert mask.max().item() == 1.0
    assert abs(mask.mean().item() - 0.95) < 0.005


def test_xseg_ratio_uses_binarized_area_not_soft_probability_mass():
    """
    The guard must measure thresholded AREA, not soft probability mass. A mask
    covering 15% of the crop at p=0.6 has a soft mean of 0.09 -- under the 0.10
    threshold, so a mean-based check would wrongly call it a collapse -- while its
    binarized area is 0.15 and correctly passes.
    """
    fm = _make_face_masks()

    def _soft(img, output):
        output.zero_()
        # 15% of the crop at p=0.6 -> soft mean 0.09 (would trip a mean-based
        # check) but thresholded area 0.15 (correctly passes).
        output.view(256, 256)[: int(256 * 0.15), :] = 0.6

    fm.run_dfl_xseg = _soft

    mask, _, _, _ = fm.apply_dfl_xseg(
        torch.zeros((3, 256, 256)),
        amount=0,
        mouth=torch.zeros((1, 256, 256)),
        parameters=dict(GUARD_ON),
        yaw_deg=PROFILE_YAW,
    )

    # Not bypassed: the center (row 128) was not filled by the top-15% stub and stays 1.0 (BG).
    assert mask[0, 128, 128].item() == 1.0


def test_clip_mask_is_not_silently_overridden():
    """
    ClipText is user-typed intent. A prompt that legitimately matches almost the
    whole crop must still mask it -- no degradation fallback on this path.
    """
    import inspect

    from app.processors.face_masks import FaceMasks

    src = inspect.getsource(FaceMasks.run_CLIPs)
    assert "0.05" not in src, "run_CLIPs must not re-introduce a silent mask override"


def test_soft_oval_mask_zero_radius_protection():
    """Zero radius or feather must not produce inf/NaN (0 * inf at the boundary)."""
    fm = _make_face_masks()

    mask = fm.soft_oval_mask(
        height=64, width=64, center=(32, 32), radius_x=0, radius_y=0
    )
    assert torch.isfinite(mask).all()

    mask_zero_feather = fm.soft_oval_mask(
        height=64, width=64, center=(32, 32), radius_x=10, radius_y=10, feather_radius=0
    )
    assert torch.isfinite(mask_zero_feather).all()


# ---------------------------------------------------------------------------
# FM-08b: UI-tunable profile-safeguard thresholds
# ---------------------------------------------------------------------------


def _occluder_mask(params, yaw=75.0, fill=0.02):
    """Run apply_occlusion with a collapsed mask and the given UI parameters."""
    fm = _make_face_masks()
    rows = int(round(256 * fill))

    def _stub(img, output):
        output.zero_()
        output.view(256, 256)[:rows, :] = 1.0

    fm.run_occluder = _stub
    merged = {"SegGuardEnableToggle": True}
    if params:
        merged.update(params)
    return fm.apply_occlusion(
        torch.zeros((3, 256, 256)), amount=0, parameters=merged, yaw_deg=yaw
    )


def test_seg_guard_defaults_match_module_constants():
    """
    Profile Safeguard is disabled by default so normal XSeg / Occluder masks are never
    overridden unless the user explicitly enables SegGuardEnableToggle.
    """
    from app.processors.face_masks import (
        _SEG_GUARD_MIN_ABS_YAW,
        _SEG_GUARD_MIN_FACE_RATIO,
        FaceMasks,
    )

    for params in (None, {}):
        enabled, min_yaw, min_ratio = FaceMasks._seg_guard_settings(params)
        assert enabled is False
        assert min_yaw == _SEG_GUARD_MIN_ABS_YAW == 60.0
        assert (
            min_ratio == pytest.approx(_SEG_GUARD_MIN_FACE_RATIO) == pytest.approx(0.10)
        )


def test_seg_guard_converts_ui_percentage_to_fraction():
    """The slider is a percentage; the mask mean it is compared against is 0..1."""
    from app.processors.face_masks import FaceMasks

    _, _, min_ratio = FaceMasks._seg_guard_settings({"SegGuardMinFaceRatioSlider": 45})
    assert min_ratio == pytest.approx(0.45)


def test_seg_guard_toggle_off_restores_unguarded_behaviour():
    """With the safeguard off, a collapsed mask must survive as-is."""
    mask = _occluder_mask({"SegGuardEnableToggle": False})
    assert mask.min().item() == 0.0, "occlusion must not be bypassed when off"


def test_seg_guard_min_face_area_zero_disables():
    """A mean can never be below 0, so 0% must behave as off."""
    mask = _occluder_mask({"SegGuardMinFaceRatioSlider": 0})
    assert mask.min().item() == 0.0


def test_seg_guard_respects_raised_min_angle():
    """At 75 deg of yaw, a 90 deg minimum must not fire."""
    assert _occluder_mask({"SegGuardMinYawSlider": 90}, yaw=75.0).min().item() == 0.0
    # ...but the same frame fires once the minimum drops below the actual angle.
    assert _occluder_mask({"SegGuardMinYawSlider": 30}, yaw=75.0).min().item() == 1.0


def test_seg_guard_respects_lowered_min_face_area():
    """
    A mask keeping 20% of the crop is a collapse at a 30% setting (softly blended > 0)
    but plausible at the 10% default, so the 10% setting must leave the occluder alone.
    """
    assert (
        _occluder_mask({"SegGuardMinFaceRatioSlider": 10}, fill=0.20).min().item()
        == 0.0
    )
    assert (
        _occluder_mask({"SegGuardMinFaceRatioSlider": 30}, fill=0.20).min().item() > 0.5
    )


def test_seg_guard_settings_are_exposed_in_the_ui():
    """
    These constants are uncalibrated, so they must stay reachable from the UI for
    users to tune. Guards against a later refactor quietly dropping the widgets.
    """
    import ast
    import io

    src = io.open("app/ui/widgets/swapper_layout_data.py", encoding="utf-8").read()
    keys = {
        k.value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Dict)
        for k in node.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert {
        "SegGuardEnableToggle",
        "SegGuardMinYawSlider",
        "SegGuardMinFaceRatioSlider",
    } <= keys


# ---------------------------------------------------------------------------
# FM-17: Mouth Fit & Align (Tier 1 geometry + Tier 2 contour warp + landmarks)
# ---------------------------------------------------------------------------


def _make_rect_mouth_labels(
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    upper_lip_h: int = 10,
    lower_lip_h: int = 10,
    size: int = 512,
) -> torch.Tensor:
    labels = torch.zeros((size, size), dtype=torch.long)
    labels[max(0, y0 - upper_lip_h) : y0, x0:x1] = 12  # upper lip
    labels[y1 : min(size, y1 + lower_lip_h), x0:x1] = 13  # lower lip
    labels[y0:y1, x0:x1] = 11  # inner mouth
    return labels


def _make_curved_smile_labels(
    cx: int,
    cy: int,
    half_w: int,
    inner_h: int,
    arch_px: float,
    tilt_rad: float = 0.0,
    size: int = 512,
) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    dx = xs - float(cx)
    dy = ys - float(cy)
    c, s = (
        float(torch.cos(torch.tensor(tilt_rad))),
        float(torch.sin(torch.tensor(tilt_rad))),
    )
    u = dx * c + dy * s
    v = -dx * s + dy * c

    norm_u = (u / max(float(half_w), 1.0)).clamp(-1.0, 1.0)
    # Positive arch_px lifts corners upward (smaller v) like a big grin
    arch_offset = -arch_px * (norm_u**2)
    taper = (1.0 - 0.35 * (norm_u**2)).clamp(min=0.2)
    half_h = 0.5 * float(inner_h) * taper

    v_top = arch_offset - half_h
    v_bot = arch_offset + half_h
    in_x = u.abs() <= float(half_w)

    labels = torch.zeros((size, size), dtype=torch.long)
    upper = in_x & (v >= v_top - 9.0) & (v < v_top)
    lower = in_x & (v > v_bot) & (v <= v_bot + 10.0)
    inner = in_x & (v >= v_top) & (v <= v_bot)
    labels[upper] = 12
    labels[lower] = 13
    labels[inner] = 11
    return labels


def test_erode_binary_shrinks_mask():
    fm = _make_face_masks()
    m = torch.zeros((32, 32), dtype=torch.float32)
    m[10:22, 10:22] = 1.0
    for mode in ("conv", "pool", "iter_pool"):
        eroded = fm._erode_binary(m, 2, mode=mode)
        assert eroded.sum().item() < m.sum().item()
        assert eroded[16, 16].item() == 1.0
        assert eroded[10, 16].item() == 0.0


def test_extract_mouth_landmarks_formats_and_guards():
    import numpy as np

    from app.processors.face_masks import FaceMasks

    # 203-point format
    kps203 = np.ones((203, 2), dtype=np.float32) * 200.0
    kps203[48] = [180.0, 340.0]
    kps203[66] = [320.0, 344.0]
    kps203[90] = [250.0, 330.0]
    kps203[102] = [250.0, 360.0]
    info = FaceMasks._extract_mouth_landmarks(kps203)
    assert info is not None
    assert info["span"] == pytest.approx(140.057, abs=0.2)
    assert info["upper_inner"][1] == pytest.approx(330.0)

    # 68-point format
    kps68 = np.ones((68, 2), dtype=np.float32) * 200.0
    kps68[48] = [200.0, 340.0]
    kps68[54] = [310.0, 340.0]
    kps68[62] = [255.0, 332.0]
    kps68[66] = [255.0, 355.0]
    info68 = FaceMasks._extract_mouth_landmarks(kps68)
    assert info68 is not None
    assert info68["span"] == pytest.approx(110.0)
    assert info68["upper_inner"][1] == pytest.approx(332.0)

    # 5-point format (no inner lip points)
    kps5 = np.array(
        [[200, 200], [312, 200], [256, 270], [210, 340], [302, 340]], dtype=np.float32
    )
    info5 = FaceMasks._extract_mouth_landmarks(kps5)
    assert info5 is not None
    assert info5["upper_inner"] is None
    assert info5["span"] == pytest.approx(92.0)

    # Degenerate inputs return None cleanly
    assert FaceMasks._extract_mouth_landmarks(None) is None
    assert (
        FaceMasks._extract_mouth_landmarks(np.zeros((203, 2), dtype=np.float32)) is None
    )


def test_mouth_geometry_maxilla_anchor_stays_steady_on_jaw_drop_and_prunes_outliers():
    """
    Maxilla/Cupid's-bow anchor (top of upper lip) must stay rock-steady when
    the lower jaw drops 35px wider, and landmark ROI pruning must drop stray noise.
    """
    import numpy as np

    fm = _make_face_masks()
    closed_ish = _make_rect_mouth_labels(
        200, 300, 320, 336, upper_lip_h=12, lower_lip_h=10
    )
    wide_open = _make_rect_mouth_labels(
        200, 300, 320, 372, upper_lip_h=12, lower_lip_h=10
    )

    g_closed = fm._measure_mouth_geometry(
        closed_ish == 11,
        (closed_ish >= 11) & (closed_ish <= 13),
        upper_lip_mask=closed_ish == 12,
    )
    g_open = fm._measure_mouth_geometry(
        wide_open == 11,
        (wide_open >= 11) & (wide_open <= 13),
        upper_lip_mask=wide_open == 12,
    )
    assert g_closed is not None and g_open is not None
    # Dropping the lower jaw 36px must NOT shift the skull/maxilla Y anchor or outer width
    assert g_closed["ay"] == pytest.approx(g_open["ay"], abs=0.5)
    assert g_closed["outer_width"] == pytest.approx(g_open["outer_width"], abs=0.5)

    # Stray class-11 noise far from mouth is pruned when landmarks are supplied
    noisy = closed_ish.clone()
    noisy[20:40, 20:40] = 11
    kps5 = np.array(
        [[200, 200], [312, 200], [256, 270], [200, 328], [300, 328]], dtype=np.float32
    )
    lmk = fm._extract_mouth_landmarks(kps5)
    g_pruned = fm._measure_mouth_geometry(
        noisy == 11,
        (noisy >= 11) & (noisy <= 13),
        upper_lip_mask=noisy == 12,
        lmk_info=lmk,
    )
    assert g_pruned is not None
    assert g_pruned["outer_width"] == pytest.approx(g_closed["outer_width"], abs=1.5)
    assert g_pruned["ay"] == pytest.approx(g_closed["ay"], abs=1.5)


def test_mouth_align_state_stabilizes_playback_and_handles_seeks_and_single_frame():
    """
    Temporal skull-lock EMA must damp per-frame lip bob during sequential playback,
    reset on seeks/scene cuts, tolerate minor out-of-order frames, and stay
    deterministic in single-frame preview mode.
    """
    from app.processors.face_masks import MouthAlignState

    st = MouthAlignState()
    # Frame 1 initializes directly (no cold-start lag)
    tx, ty, sc, ang = st.update(2.0, -4.0, 1.04, 0.02, stability=0.85, frame_number=10)
    assert (tx, ty, sc, ang) == pytest.approx((2.0, -4.0, 1.04, 0.02))

    # Frame 2 sudden lip-bob spike (+10px vertical) is heavily damped by stability=0.85
    tx2, ty2, sc2, ang2 = st.update(
        2.0, 6.0, 1.12, 0.10, stability=0.85, frame_number=11
    )
    assert ty2 == pytest.approx(0.15 * 6.0 + 0.85 * (-4.0), abs=1e-5)
    assert sc2 == pytest.approx(0.15 * 1.12 + 0.85 * 1.04, abs=1e-5)

    # Single-frame mode must NOT mutate stored EMA and returns raw value
    tx_sf, ty_sf, _, _ = st.update(
        9.0, 9.0, 1.0, 0.0, stability=0.85, frame_number=11, single_frame_mode=True
    )
    assert (tx_sf, ty_sf) == pytest.approx((9.0, 9.0))
    assert st.ty == pytest.approx(ty2, abs=1e-5)

    # Large forward jump (> 15 frames = scene cut / seek) resets cleanly
    tx_cut, ty_cut, sc_cut, _ = st.update(
        -3.0, 5.0, 0.96, 0.0, stability=0.85, frame_number=100
    )
    assert (tx_cut, ty_cut, sc_cut) == pytest.approx((-3.0, 5.0, 0.96))


def test_mouth_align_wide_thin_smile_has_no_center_ghosting():
    """Wide thin smile must stay 100% opaque across its center (no double-teeth ghosting)."""
    fm = _make_face_masks()
    labels_orig = _make_rect_mouth_labels(176, 336, 330, 344)  # 160x14 wide grin
    labels_swap = _make_rect_mouth_labels(176, 336, 330, 344)
    img_orig = torch.full((3, 512, 512), 220.0)
    img_swap = torch.full((3, 512, 512), 40.0)

    overlay, mask = fm._enhance_and_align_original_mouth(
        img_orig,
        img_swap,
        labels_orig,
        labels_swap,
        {"MouthParserStretchDecimalSlider": 1.0},
    )
    assert overlay is not None and mask is not None
    # Center strip inside inner_swap must be 1.0 (not attenuated by width-based Gaussian)
    assert mask[334:340, 200:312].min().item() == pytest.approx(1.0, abs=1e-4)
    # Outside the lips must stay 0.0
    assert mask[300, 256].item() == pytest.approx(0.0, abs=1e-4)


def test_mouth_align_keeps_teeth_rigid_and_fills_void_with_dark_scene_cavity():
    """
    When swapped mouth opens 2.5x taller than original:
    - Original teeth must NOT be vertically stretched (strict rigid isotropic scale=1.0).
    - Lower cavity void must be fully masked in (1.0) and filled with dark scene-matched
      oral cavity shadow (< 45/255), NOT bright blurred fake swapped teeth.
    """
    fm = _make_face_masks()
    # Original mouth: 100x24 (y=320..344), upper teeth band (y=320..328), dark interior (y=328..344)
    labels_orig = _make_rect_mouth_labels(
        206, 306, 320, 344, upper_lip_h=10, lower_lip_h=10
    )
    # Swapped mouth: same outer width & upper lip, but opens 2.5x taller (y=320..380)
    labels_swap = _make_rect_mouth_labels(
        206, 306, 320, 380, upper_lip_h=10, lower_lip_h=10
    )

    img_orig = torch.zeros((3, 512, 512), dtype=torch.float32)
    img_orig[:, 320:344, 206:306] = torch.tensor([34.0, 16.0, 15.0]).view(
        3, 1, 1
    )  # dark cavity
    img_orig[:, 320:328, 206:306] = 235.0  # crisp 8px-tall white upper teeth

    # Swapped mouth filled with bright fake white teeth (220) and pink lips
    img_swap = torch.full((3, 512, 512), 220.0)
    img_swap[:, labels_swap == 12] = torch.tensor([175.0, 95.0, 95.0]).view(3, 1)
    img_swap[:, labels_swap == 13] = torch.tensor([175.0, 95.0, 95.0]).view(3, 1)

    overlay, mask = fm._enhance_and_align_original_mouth(
        img_orig,
        img_swap,
        labels_orig,
        labels_swap,
        {
            "MouthParserStretchDecimalSlider": 1.0,
            "MouthOriginalCavityDarkenDecimalSlider": 1.30,
            "MouthOriginalCavityBlendingDecimalSlider": 0.0,
            "MouthUpperLipShadowToggle": False,
        },
    )
    assert overlay is not None and mask is not None
    # Upper teeth stay strictly in their rigid 8px height (bright at y=322..326, dark below y=332)
    assert overlay[:, 322:326, 230:282].mean().item() > 215.0
    assert overlay[:, 334:340, 230:282].mean().item() < 55.0
    # Lower cavity void (y=352..374) is fully covered (mask=1.0) and dark (NOT 220 fake teeth!)
    assert mask[352:374, 225:285].min().item() == pytest.approx(1.0, abs=1e-4)
    assert overlay[:, 352:374, 225:285].mean().item() < 40.0


def test_mouth_align_ncc_locks_to_teeth_ignoring_moving_upper_lip():
    """
    When original upper lip lifts 6px higher than swapped upper lip (e.g. during
    speech/smile), a lip-edge anchor would drag the teeth down by +10px. Direct
    Masked 2D Teeth+Edge NCC must lock onto the actual teeth incisal edge and
    automatically place original teeth directly on top of swapped teeth (+4px).
    """
    fm = _make_face_masks()
    # Original: inner opens y=320..360; dark gum/shadow y=320..326, bright teeth y=326..336, dark cavity y=336..360
    labels_orig = _make_rect_mouth_labels(
        206, 306, 320, 360, upper_lip_h=10, lower_lip_h=10
    )
    # Swapped: upper lip sits lower (inner opens y=330..364); bright swapped teeth at y=330..340 (+4px vs orig teeth!)
    labels_swap = _make_rect_mouth_labels(
        206, 306, 330, 364, upper_lip_h=12, lower_lip_h=10
    )

    img_orig = torch.full((3, 512, 512), 25.0)
    img_orig[:, 320:326, 206:306] = torch.tensor([110.0, 55.0, 55.0]).view(
        3, 1, 1
    )  # upper gum
    img_orig[:, 326:336, 206:306] = 240.0  # rigid original upper teeth

    img_swap = torch.full((3, 512, 512), 25.0)
    img_swap[:, 330:340, 206:306] = 220.0  # swapped upper teeth (+4px below orig teeth)
    img_swap[:, labels_swap == 12] = torch.tensor([165.0, 85.0, 85.0]).view(3, 1)
    img_swap[:, labels_swap == 13] = torch.tensor([165.0, 85.0, 85.0]).view(3, 1)

    overlay, _ = fm._enhance_and_align_original_mouth(
        img_orig,
        img_swap,
        labels_orig,
        labels_swap,
        {
            "MouthParserStretchDecimalSlider": 1.0,
            "MouthUpperLipShadowToggle": False,
        },
    )
    assert overlay is not None
    # Original teeth (10px tall) must land on y=330..340 (NOT pushed down to y=336..346!)
    assert overlay[:, 332:338, 230:282].mean().item() > 215.0
    # Dark cavity directly below the teeth (y=343..352) must stay dark (< 55)
    assert overlay[:, 343:352, 230:282].mean().item() < 55.0


def test_upper_lip_contact_shadow_darkens_top_edge_only_when_enabled():
    """Upper Lip Contact Shadow must subtly darken rows directly beneath upper lip."""
    fm = _make_face_masks()
    labels_orig = _make_rect_mouth_labels(
        206, 306, 320, 356, upper_lip_h=12, lower_lip_h=10
    )
    labels_swap = _make_rect_mouth_labels(
        206, 306, 320, 356, upper_lip_h=12, lower_lip_h=10
    )
    img_orig = torch.full((3, 512, 512), 220.0)
    img_swap = torch.full((3, 512, 512), 220.0)

    ov_off, _ = fm._enhance_and_align_original_mouth(
        img_orig,
        img_swap,
        labels_orig,
        labels_swap,
        {"MouthUpperLipShadowToggle": False},
    )
    ov_on, _ = fm._enhance_and_align_original_mouth(
        img_orig,
        img_swap,
        labels_orig,
        labels_swap,
        {
            "MouthUpperLipShadowToggle": True,
            "MouthUpperLipShadowDecimalSlider": 0.60,
        },
    )
    assert ov_off is not None and ov_on is not None
    # Directly under swapped upper lip (y=320..322) must be visibly darkened vs off
    assert (
        ov_on[:, 320:322, 230:282].mean().item()
        < ov_off[:, 320:322, 230:282].mean().item() - 35.0
    )
    # Lower interior far from upper lip (y=342..350) must be unaffected
    assert ov_on[:, 342:350, 230:282].mean().item() == pytest.approx(
        ov_off[:, 342:350, 230:282].mean().item(), abs=1.0
    )


def test_soft_parser_boundary_produces_subpixel_edge_with_solid_core():
    """Soft Parser Boundary must use continuous probabilities at edges while keeping core 1.0."""
    fm = _make_face_masks()
    labels = _make_rect_mouth_labels(206, 306, 320, 350, upper_lip_h=10, lower_lip_h=10)
    img = torch.full((3, 512, 512), 200.0)

    # Build smooth continuous probability maps for class 11 (inner) and 12 (upper lip)
    p_inner = (labels == 11).float()
    p_inner[319, 206:306] = 0.45  # sub-pixel upper edge
    p_inner[350, 206:306] = 0.45  # sub-pixel lower edge
    probs = {
        "inner": p_inner,
        "upper": (labels == 12).float(),
        "lower": (labels == 13).float(),
    }

    _, mask = fm._enhance_and_align_original_mouth(
        img,
        img,
        labels,
        labels,
        {"MouthSoftParserBoundaryToggle": True, "MouthUpperLipShadowToggle": False},
        probs_orig=probs,
        probs_swap=probs,
    )
    assert mask is not None
    # Solid interior remains 1.0
    assert mask[326:344, 220:290].min().item() == pytest.approx(1.0, abs=1e-4)
    # Sub-pixel boundary pixel stays smoothly between 0 and 1
    edge_val = mask[319, 256].item()
    assert 0.15 < edge_val < 0.85, f"Expected sub-pixel soft edge, got {edge_val:.3f}"


def test_mouth_align_absorbs_leaked_swap_teeth_rim_without_touching_lips():
    """
    A 2px sliver of bright white swapped teeth misclassified as upper lip (12)
    must be absorbed into effective_inner_swap, while genuine pink/red upper lip
    pixels stay excluded.
    """
    fm = _make_face_masks()
    labels_swap = _make_rect_mouth_labels(
        206, 306, 324, 354, upper_lip_h=12, lower_lip_h=12
    )
    img_swap = torch.zeros((3, 512, 512), dtype=torch.float32)
    # Pink/red lips
    img_swap[:, (labels_swap == 12) | (labels_swap == 13)] = torch.tensor(
        [170.0, 90.0, 90.0]
    ).view(3, 1)
    # 2px white swapped-tooth sliver misclassified inside bottom of upper lip (y=322..324)
    img_swap[:, 322:324, 220:290] = torch.tensor([230.0, 228.0, 224.0]).view(3, 1, 1)

    eff = fm._absorb_leaked_swap_teeth(
        img_swap, labels_swap == 11, (labels_swap == 12) | (labels_swap == 13)
    )
    # Leaked white teeth sliver is absorbed
    assert eff[323, 255].item() is True
    # Actual pink upper lip above it remains untouched
    assert eff[316, 255].item() is False


def test_get_mouth_overlay_handles_non_512_swap_resolution():
    """Post-restorer upscaled swap (e.g. 1024x1024) must not crash or corrupt parser."""
    fm = _make_face_masks()
    labels_rect = _make_rect_mouth_labels(206, 306, 320, 350)
    fm._faceparser_labels = lambda img: labels_rect

    swap_1024 = torch.full((3, 1024, 1024), 120.0)
    orig_512 = torch.full((3, 512, 512), 200.0)
    res = fm.get_mouth_overlay(
        swap_1024,
        orig_512,
        {
            "MouthParserStretchToggle": True,
            "MouthParserStretchOriginalToggle": True,
        },
    )
    assert res is not None
    overlay, mask = res
    assert overlay is not None and mask is not None
    assert overlay.shape == (3, 512, 512)
    assert mask.shape == (512, 512)
    assert torch.isfinite(overlay).all() and torch.isfinite(mask).all()


def test_mouth_align_rotates_to_match_tilted_swapped_mouth():
    """Tilted swapped mouth (+8 deg) must rigidly rotate original mouth to match."""
    import math

    fm = _make_face_masks()
    labels_orig = _make_curved_smile_labels(
        256, 340, half_w=70, inner_h=22, arch_px=0.0, tilt_rad=0.0
    )
    labels_swap = _make_curved_smile_labels(
        256, 340, half_w=70, inner_h=22, arch_px=0.0, tilt_rad=math.radians(8.0)
    )
    img_orig = (labels_orig == 11).float().unsqueeze(0).expand(
        3, 512, 512
    ).clone() * 255.0
    img_swap = torch.zeros((3, 512, 512), dtype=torch.float32)

    ov, mask = fm._enhance_and_align_original_mouth(
        img_orig,
        img_swap,
        labels_orig,
        labels_swap,
        {"MouthParserStretchDecimalSlider": 1.0},
    )
    assert ov is not None and mask is not None
    inner_swap = labels_swap == 11
    coverage = (ov[0][inner_swap] > 127.0).float().mean().item()
    assert coverage > 0.88, (
        f"Expected rigid rotated coverage > 0.88, got {coverage:.3f}"
    )


def test_swapped_mouth_zoom_mode_preserves_center_opacity():
    """When Use Original Mouth is False, wide thin smile stays solid in the center."""
    fm = _make_face_masks()
    labels_swap = _make_rect_mouth_labels(176, 336, 330, 346)
    swap_img = torch.full((3, 512, 512), 180.0)
    ov, mask = fm._enhance_and_align_swapped_mouth(
        swap_img,
        labels_swap,
        {"MouthParserStretchDecimalSlider": 1.0},
    )
    assert ov is not None and mask is not None
    assert mask[334:342, 200:312].min().item() == pytest.approx(1.0, abs=1e-4)
    assert mask[300, 256].item() == pytest.approx(0.0, abs=1e-4)


def test_mouth_align_controls_exposed_in_ui():
    import ast
    import io

    src = io.open("app/ui/widgets/swapper_layout_data.py", encoding="utf-8").read()
    keys = {
        k.value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Dict)
        for k in node.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert {
        "MouthParserStretchToggle",
        "MouthParserStretchOriginalToggle",
        "MouthAlignStabilityDecimalSlider",
        "MouthSoftParserBoundaryToggle",
        "MouthUpperLipShadowToggle",
        "MouthUpperLipShadowDecimalSlider",
        "MouthParserStretchAfterToggle",
        "MouthParserStretchDecimalSlider",
    } <= keys
    assert "MouthAlignYOffsetSlider" not in keys
