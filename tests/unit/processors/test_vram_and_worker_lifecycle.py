"""Unit tests for VRAM lifecycle, ORT arena-shrink isolation, and worker cleanup fixes."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from app.processors.face_reaging import FaceReaging
from app.processors.models_processor import ModelsProcessor
from app.processors.workers.function_worker import FunctionWorker


def _make_function_worker(provider_name: str = "TensorRT") -> FunctionWorker:
    fw = FunctionWorker.__new__(FunctionWorker)
    fw.mp = cast(
        ModelsProcessor,
        SimpleNamespace(
            provider_name=provider_name,
            device_type="cuda",
            gpu_id=0,
            models={},
            trt_shared_context_sessions=set(),
            syncvec=SimpleNamespace(cpu=lambda: None),
        ),
    )
    fw._session_locks = {}
    fw._session_locks_mutex = threading.Lock()
    fw._trt_shared_context_lock = threading.Lock()
    fw._ort_gate_condition = threading.Condition()
    fw._ort_gate_readers = 0
    fw._ort_gate_writer_active = False
    fw._ort_gate_writers_waiting = 0
    fw._ort_shrink_generation = 0
    fw._ort_shrunk_generation = {}
    return fw


def test_request_ort_arena_shrink_only_applies_to_cuda_provider():
    for trt_provider in ("TensorRT", "TensorRT-Engine"):
        fw = _make_function_worker(provider_name=trt_provider)
        fw.request_ort_arena_shrink()
        assert fw._ort_shrink_generation == 0
        assert fw._ort_run_options(object()) is None

    fw_cuda = _make_function_worker(provider_name="CUDA")
    fw_cuda.request_ort_arena_shrink()
    assert fw_cuda._ort_shrink_generation == 1

    s1, s2 = object(), object()
    ro1 = fw_cuda._ort_run_options(s1)
    ro2 = fw_cuda._ort_run_options(s2)
    assert ro1 is not None
    assert ro2 is not None
    # Each session gets its own RunOptions instance rather than sharing one mutable object.
    assert ro1 is not ro2
    # Subsequent calls in the same generation return None.
    assert fw_cuda._ort_run_options(s1) is None


def test_cuda_arena_shrink_runs_under_exclusive_ort_gate():
    fw = _make_function_worker(provider_name="CUDA")
    fw.request_ort_arena_shrink()

    gate_modes: list[bool] = []
    orig_gate = fw._ort_inference_gate

    def spy_gate(exclusive: bool):
        gate_modes.append(exclusive)
        return orig_gate(exclusive)

    fw._ort_inference_gate = spy_gate  # type: ignore[method-assign]

    session = MagicMock()
    io_binding = MagicMock()

    with patch(
        "app.processors.workers.function_worker.platform_support.blocking_stream_sync"
    ):
        # First run triggers shrink -> must be exclusive
        fw.run_ort_with_iobinding(session, io_binding)
        # Second run has no shrink -> shared reader gate
        fw.run_ort_with_iobinding(session, io_binding)

    assert gate_modes == [True, False]


def test_unload_model_releases_function_worker_session_tracking():
    fw = _make_function_worker(provider_name="TensorRT")
    session = object()
    sid = id(session)
    fw._get_session_lock(session)
    fw._ort_shrunk_generation[sid] = 1
    assert sid in fw._session_locks
    assert sid in fw._ort_shrunk_generation

    mp = ModelsProcessor.__new__(ModelsProcessor)
    mp.model_lock = threading.RLock()
    mp.models = {"FaceParser": session}
    mp.trt_shared_context_sessions = {sid}
    mp.force_unload_in_progress = True
    mp.deferred_unloads = {}
    mp.main_window = SimpleNamespace(
        control={"KeepModelsAliveToggle": False},
        video_processor=SimpleNamespace(processing=False),
        function_worker=fw,
    )

    with (
        patch(
            "app.processors.models_processor.torch.cuda.is_available",
            return_value=False,
        ),
    ):
        mp.unload_model("FaceParser", force_immediate=True)

    assert mp.models["FaceParser"] is None
    assert sid not in mp.trt_shared_context_sessions
    assert sid not in fw._session_locks
    assert sid not in fw._ort_shrunk_generation


def test_face_reaging_routes_through_function_worker_run_ort_session():
    fw = _make_function_worker(provider_name="TensorRT")
    session = MagicMock()
    inp_meta = MagicMock()
    inp_meta.name = "input"
    out_meta = MagicMock()
    out_meta.name = "output"
    session.get_inputs.return_value = [inp_meta]
    session.get_outputs.return_value = [out_meta]
    session.run.side_effect = lambda _out, feeds: [
        np.zeros((1, 3, 64, 64), dtype=np.float32)
    ]

    fw.mp.trt_shared_context_sessions.add(id(session))
    fw.run_ort_session = MagicMock(wraps=fw.run_ort_session)  # type: ignore[method-assign]

    mp = SimpleNamespace(device=torch.device("cpu"), load_model=lambda _name: session)
    reaging = FaceReaging(mp, fw)  # type: ignore[arg-type]

    face = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    out = reaging.apply_reaging(face, source_age=30, target_age=50)

    assert out.shape == (3, 64, 64)
    fw.run_ort_session.assert_called_once()


def test_worker_pool_manager_reuses_persistent_cuda_streams():
    from app.processors.video_utils.worker_pool_manager import WorkerPoolManager

    mgr = WorkerPoolManager.__new__(WorkerPoolManager)
    mgr.worker_threads = []
    mgr.worker_streams = []
    mgr.frame_queue = SimpleNamespace(
        mutex=threading.Lock(),
        queue=[],
        put=lambda *a, **kw: None,
    )
    mgr.main_window = SimpleNamespace(
        control={"nStreamSlider": 3},
        function_worker=SimpleNamespace(request_ort_arena_shrink=MagicMock()),
        models_processor=SimpleNamespace(device_type="cuda"),
    )

    created_streams: list[MagicMock] = []

    def make_stream():
        s = MagicMock(name=f"stream-{len(created_streams)}")
        created_streams.append(s)
        return s

    fake_workers: list[MagicMock] = []

    def make_worker(**kwargs):
        w = MagicMock()
        w.is_alive.return_value = False
        w.stop_event = threading.Event()
        w.worker_stream = kwargs.get("worker_stream")
        fake_workers.append(w)
        return w

    with (
        patch(
            "app.processors.video_utils.worker_pool_manager.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "app.processors.video_utils.worker_pool_manager.torch.cuda.is_initialized",
            return_value=True,
        ),
        patch(
            "app.processors.video_utils.worker_pool_manager.torch.cuda.Stream",
            side_effect=make_stream,
        ),
        patch(
            "app.processors.video_utils.worker_pool_manager.torch.cuda.synchronize"
        ) as mock_sync,
        patch(
            "app.processors.video_utils.worker_pool_manager.torch.cuda.empty_cache"
        ) as mock_empty,
        patch(
            "app.processors.video_utils.worker_pool_manager.FrameWorker",
            side_effect=make_worker,
        ),
    ):
        mgr.start_persistent_pool(num_threads=3)
        first_streams = list(mgr.worker_streams)
        assert len(first_streams) == 3
        assert len(created_streams) == 3

        mgr.join_and_clear_threads(clear_module_caches=False)
        # Streams remain allocated & reused across playbacks; synced before empty_cache.
        assert mgr.worker_streams == first_streams
        mock_sync.assert_called_once()
        mock_empty.assert_called_once()

        # Starting playback again with same settings allocates 0 new streams.
        mgr.start_persistent_pool(num_threads=3)
        assert mgr.worker_streams == first_streams
        assert len(created_streams) == 3


def test_media_pipeline_release_pinned_ring_buffers_clears_buffers():
    from app.processors.video_utils.media_pipeline import MediaPipeline

    mp_pipe = MediaPipeline.__new__(MediaPipeline)
    mp_pipe._video_pinned_ring_buffer = [np.zeros((4, 4, 3), dtype=np.uint8)]
    mp_pipe._video_pinned_ring_shape = (4, 4, 3)
    mp_pipe._video_pinned_ring_idx = 1
    mp_pipe._webcam_pinned_ring_buffer = [np.zeros((4, 4, 3), dtype=np.uint8)]
    mp_pipe._webcam_pinned_ring_shape = (4, 4, 3)
    mp_pipe._webcam_pinned_ring_idx = 1

    mp_pipe.release_pinned_ring_buffers()

    assert mp_pipe._video_pinned_ring_buffer == []
    assert mp_pipe._video_pinned_ring_shape is None
    assert mp_pipe._video_pinned_ring_idx == 0
    assert mp_pipe._webcam_pinned_ring_buffer == []
    assert mp_pipe._webcam_pinned_ring_shape is None
    assert mp_pipe._webcam_pinned_ring_idx == 0
