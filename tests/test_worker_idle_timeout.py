"""The caption-worker timeout must measure silence, not total duration.

One `build_blocks` request carries every block a worker owns for a video, so a
long video legitimately keeps a request open for longer than the limit. Before
this behaviour existed, a 9-block video crashed the run at the deadline while
the worker was still writing segments.
"""

import queue
import threading
import time

import pytest

from surrogate_rollout.captioning.history_aware_baseline import (
    HistoryAwareBaselineCaptionViewBuilder,
)


class _StubProcess:
    def __init__(self, alive=True):
        self.pid = 4242
        self.exitcode = None
        self._alive = alive

    def is_alive(self):
        return self._alive


def _builder(limit):
    builder = HistoryAwareBaselineCaptionViewBuilder(
        router=None,
        segment_captioner=object(),
        parallel_gpus=("4",),
        worker_result_timeout_seconds=limit,
        worker_health_poll_seconds=0.01,
    )
    builder._pool_processes = {"4": _StubProcess()}
    builder._pool_result_queue = queue.Queue()
    return builder


def _start_router(builder):
    thread = threading.Thread(target=builder._route_pool_results, daemon=True)
    thread.start()
    return thread


def _stop_router(builder, thread):
    builder._pool_result_queue.put({"request_id": "__parent_stop__"})
    thread.join(timeout=5)


def test_request_outliving_the_limit_survives_on_heartbeats():
    builder = _builder(0.3)
    thread = _start_router(builder)
    request_id = builder._register_pool_request("blocks")

    def worker():
        # Three blocks, each shorter than the limit, together well over it.
        for block_index in range(3):
            time.sleep(0.2)
            builder._pool_result_queue.put({
                "request_id": request_id, "gpu": "4",
                "event": "progress", "stage": "block_start",
                "block_index": block_index})
        time.sleep(0.2)
        builder._pool_result_queue.put({
            "request_id": request_id, "gpu": "4", "status": "ok"})

    threading.Thread(target=worker, daemon=True).start()
    started = time.monotonic()
    result = builder._wait_pool_result(request_id)
    assert result["status"] == "ok"
    assert time.monotonic() - started > 0.3
    _stop_router(builder, thread)


def test_silent_request_still_times_out():
    builder = _builder(0.2)
    thread = _start_router(builder)
    request_id = builder._register_pool_request("blocks")
    with pytest.raises(RuntimeError, match="timed out"):
        builder._wait_pool_result(request_id)
    _stop_router(builder, thread)


def test_heartbeat_never_satisfies_the_waiter():
    builder = _builder(5.0)
    thread = _start_router(builder)
    request_id = builder._register_pool_request("blocks")
    builder._pool_result_queue.put({
        "request_id": request_id, "gpu": "4",
        "event": "progress", "stage": "block_start", "block_index": 0})
    builder._pool_result_queue.put({
        "request_id": request_id, "gpu": "4", "status": "ok",
        "block_index": 0})
    result = builder._wait_pool_result(request_id)
    assert result["status"] == "ok"
    assert "event" not in result
    _stop_router(builder, thread)


def test_dead_worker_is_still_detected_before_the_limit():
    builder = _builder(30.0)
    thread = _start_router(builder)
    request_id = builder._register_pool_request("blocks")
    builder._pool_processes["4"]._alive = False
    with pytest.raises(RuntimeError, match="died while waiting"):
        builder._wait_pool_result(request_id)
    _stop_router(builder, thread)


def test_progress_state_is_released_with_the_request():
    builder = _builder(5.0)
    thread = _start_router(builder)
    request_id = builder._register_pool_request("blocks")
    builder._pool_result_queue.put({
        "request_id": request_id, "gpu": "4", "status": "ok"})
    builder._wait_pool_result(request_id)
    assert request_id not in builder._pool_progress
    assert request_id not in builder._pool_pending
    _stop_router(builder, thread)
