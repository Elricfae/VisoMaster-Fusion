"""Unit tests for TensorRT hybrid context memory sharing toggle behavior."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from app.processors.models_data import trt_context_memory_sharing_models
from app.processors.models_processor import ModelsProcessor


def _make_processor(
    *,
    provider_name: str = "TensorRT",
    control: dict | None = None,
) -> ModelsProcessor:
    mp = ModelsProcessor.__new__(ModelsProcessor)
    mp.provider_name = provider_name
    mp.providers = [
        ("TensorrtExecutionProvider", {"device_id": 0}),
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]
    mp.trt_ep_options = {"device_id": 0}
    mp.model_lock = threading.RLock()
    mp.models = {}
    mp.models_path = {}
    mp.models_pending_build = set()
    mp.trt_shared_context_sessions = set()
    mp.force_unload_in_progress = False
    mp.deferred_unloads = {}
    mp._models_loading = set()
    mp.main_window = cast(
        Any,
        SimpleNamespace(
            control={} if control is None else dict(control),
            video_processor=SimpleNamespace(processing=False, current_frame_number=0),
        ),
    )
    return mp


def test_is_trt_context_memory_sharing_enabled_reads_control_with_true_default():
    mp = _make_processor(control={})
    assert mp.is_trt_context_memory_sharing_enabled() is True

    mp.main_window.control["TrtContextMemorySharingToggle"] = False
    assert mp.is_trt_context_memory_sharing_enabled() is False

    mp.main_window.control["TrtContextMemorySharingToggle"] = True
    assert mp.is_trt_context_memory_sharing_enabled() is True

    mp.main_window = SimpleNamespace(control=None)
    assert mp.is_trt_context_memory_sharing_enabled() is True


def test_unload_mismatched_heavy_trt_models_when_disabling_and_enabling():
    heavy_model = trt_context_memory_sharing_models[0]
    light_model = "RetinaFace"

    heavy_session = object()
    light_session = object()

    mp = _make_processor(
        provider_name="TensorRT",
        control={"KeepModelsAliveToggle": True, "TrtContextMemorySharingToggle": True},
    )
    mp.models[heavy_model] = heavy_session
    mp.models[light_model] = light_session
    mp.trt_shared_context_sessions.add(id(heavy_session))

    assert mp.has_mismatched_trt_shared_context_models(True) is False
    assert mp.has_mismatched_trt_shared_context_models(False) is True

    # Disabling sharing must force-unload the shared heavy model even with KeepModelsAliveToggle=True,
    # while leaving lightweight models untouched.
    unloaded = mp.unload_mismatched_trt_shared_context_models(False)
    assert unloaded == [heavy_model]
    assert mp.models[heavy_model] is None
    assert id(heavy_session) not in mp.trt_shared_context_sessions
    assert mp.models[light_model] is light_session
    assert mp.force_unload_in_progress is False

    # Re-load heavy model without shared context, then re-enable sharing -> mismatched again.
    private_heavy_session = object()
    mp.models[heavy_model] = private_heavy_session
    assert mp.has_mismatched_trt_shared_context_models(False) is False
    assert mp.has_mismatched_trt_shared_context_models(True) is True

    unloaded_on_enable = mp.unload_mismatched_trt_shared_context_models(True)
    assert unloaded_on_enable == [heavy_model]
    assert mp.models[heavy_model] is None


def test_unload_mismatched_is_noop_on_non_tensorrt_provider():
    heavy_model = trt_context_memory_sharing_models[0]
    heavy_session = object()

    mp = _make_processor(provider_name="CUDA")
    mp.models[heavy_model] = heavy_session

    assert mp.has_mismatched_trt_shared_context_models(True) is False
    assert mp.has_mismatched_trt_shared_context_models(False) is False
    assert mp.unload_mismatched_trt_shared_context_models(False) == []
    assert mp.models[heavy_model] is heavy_session


def test_load_model_respects_trt_context_memory_sharing_toggle():
    heavy_model = "FaceParser"
    assert heavy_model in trt_context_memory_sharing_models

    captured_providers: list = []

    def fake_inference_session(path, sess_options=None, providers=None):
        captured_providers.append(providers)
        return object()

    for toggle_value in (True, False):
        captured_providers.clear()
        mp = _make_processor(
            provider_name="TensorRT",
            control={"TrtContextMemorySharingToggle": toggle_value},
        )
        mp.models[heavy_model] = None
        mp.models_path[heavy_model] = "dummy_face_parser.onnx"
        mp.main_window.model_loading_signal = MagicMock()
        mp.main_window.model_loaded_signal = MagicMock()
        mp._ensure_trt_ready_onnx = lambda _name, p: p
        mp._check_tensorrt_cache_state = lambda _name, _p: "EXPLICIT"

        with (
            patch(
                "app.processors.models_processor.onnxruntime.InferenceSession",
                side_effect=fake_inference_session,
            ),
            patch(
                "app.processors.models_processor.onnxruntime.SessionOptions",
                return_value=SimpleNamespace(log_severity_level=3),
            ),
            patch(
                "app.processors.models_processor.QtCore.QCoreApplication.instance",
                return_value=None,
            ),
        ):
            session = mp.load_model(heavy_model)

        assert session is not None
        trt_opts = dict(captured_providers[0][0][1])
        assert trt_opts["trt_context_memory_sharing_enable"] is toggle_value
        assert (id(session) in mp.trt_shared_context_sessions) is toggle_value


def test_control_action_commits_value_and_reloads_only_when_needed():
    from app.ui.widgets.actions import control_actions

    heavy_model = trt_context_memory_sharing_models[0]
    heavy_session = object()

    mp = _make_processor(
        provider_name="TensorRT",
        control={"TrtContextMemorySharingToggle": True},
    )
    mp.models[heavy_model] = heavy_session
    mp.trt_shared_context_sessions.add(id(heavy_session))

    stop_processing = MagicMock()
    main_window = SimpleNamespace(
        control={"TrtContextMemorySharingToggle": True},
        models_processor=mp,
        video_processor=SimpleNamespace(stop_processing=stop_processing),
    )

    with patch(
        "app.ui.widgets.actions.control_actions.common_widget_actions.update_gpu_memory_progressbar"
    ) as mock_vram:
        control_actions.handle_trt_context_memory_sharing_toggle(main_window, False)

    assert main_window.control["TrtContextMemorySharingToggle"] is False
    stop_processing.assert_called_once()
    assert mp.models[heavy_model] is None
    mock_vram.assert_called_once_with(main_window)

    # Calling again with no mismatched models should not stop processing.
    stop_processing.reset_mock()
    with patch(
        "app.ui.widgets.actions.control_actions.common_widget_actions.update_gpu_memory_progressbar"
    ) as mock_vram:
        control_actions.handle_trt_context_memory_sharing_toggle(main_window, False)

    stop_processing.assert_not_called()
    mock_vram.assert_not_called()
