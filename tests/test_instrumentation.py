"""Wrapper-level tests (no DVD import, no GPU): hit capture, query restoration,
signature preservation, and behavior transparency."""

import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

import pytest

from surrogate_rollout.instrumentation import (
    RunRecorder,
    _wrap_frame_inspect,
    _wrap_retrieval_tool,
)
from surrogate_rollout.evaluation.dvd_qa import (
    _constrain_clip_search_schema,
    dvd_qa_execution_identity,
    serialized_dvd_qa_execution,
)


class FakeDB:
    def __init__(self, results):
        self._results = results

    def query(self, emb, top_k=16):
        return self._results[:top_k]


HITS = [{"time_start_secs": float(i * 10), "time_end_secs": float((i + 1) * 10)}
        for i in range(5)]


def clip_search_tool(
    database: Annotated[FakeDB, "db"],
    event_description: Annotated[str, "desc"],
    top_k: Annotated[int, "k"] = 16,
) -> str:
    """Original docstring."""
    results = database.query(None, top_k=top_k)
    return f"{len(results)} captions"


def test_hits_captured_and_result_unchanged():
    rec = RunRecorder()
    wrapped = _wrap_retrieval_tool(clip_search_tool, rec)
    db = FakeDB(HITS)
    out = wrapped(database=db, event_description="x", top_k=3)
    assert out == "3 captions"                      # behavior transparent
    assert len(rec.tool_events) == 1
    ev = rec.tool_events[0]
    assert ev["tool"] == "clip_search_tool"
    assert ev["n_hits"] == 3
    assert ev["hits"][0]["time_start_secs"] == 0.0
    assert ev["error"] is None


def test_clip_search_top_k_is_forced_and_requested_value_is_preserved():
    rec = RunRecorder()
    wrapped = _wrap_retrieval_tool(
        clip_search_tool, rec, forced_top_k=16)
    db = FakeDB(HITS * 4)
    out = wrapped(database=db, event_description="x", top_k=5)
    assert out == "16 captions"
    event = rec.tool_events[0]
    assert event["args"]["top_k"] == 16
    assert event["requested_args"]["top_k"] == 5
    assert event["argument_override"] == {
        "field": "top_k", "executed_value": 16,
        "policy_version": "fixed_clip_search_top_k_v1",
    }


def test_clip_search_schema_and_execution_identity_are_fixed_to_16():
    class Agent:
        function_schemas = [{
            "function": {
                "name": "clip_search_tool",
                "parameters": {"properties": {"top_k": {
                    "type": "integer", "description": "model selected",
                }}},
            },
        }]

    agent = Agent()
    _constrain_clip_search_schema(agent, 16)
    field = agent.function_schemas[0]["function"]["parameters"][
        "properties"]["top_k"]
    assert field["enum"] == [16]
    assert field["default"] == 16
    identity = dvd_qa_execution_identity(10)
    assert identity["clip_search_top_k"] == 16
    assert identity["clip_search_policy_version"] == \
        "fixed_clip_search_top_k_v1"
    assert identity["embedding_preload_policy_version"] == \
        "dvd_bge_parent_preload_v1"
    assert identity["qa_concurrency_policy_version"] == \
        "serialized_dvd_qa_execution_v1"


def test_dvd_qa_global_state_guard_serializes_parallel_callers():
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def enter(_index):
        nonlocal active, maximum_active
        with serialized_dvd_qa_execution():
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.005)
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=8) as pool:
        tuple(pool.map(enter, range(16)))

    assert maximum_active == 1


def test_query_method_restored_after_call():
    rec = RunRecorder()
    wrapped = _wrap_retrieval_tool(clip_search_tool, rec)
    db = FakeDB(HITS)
    original_query = db.query
    wrapped(database=db, event_description="x")
    assert db.query == original_query


def test_query_restored_even_on_tool_error():
    def failing_tool(database, event_description):
        database.query(None)
        raise RuntimeError("boom")

    rec = RunRecorder()
    wrapped = _wrap_retrieval_tool(failing_tool, rec)
    db = FakeDB(HITS)
    original_query = db.query
    with pytest.raises(RuntimeError):
        wrapped(database=db, event_description="x")
    assert db.query == original_query
    assert rec.tool_events[0]["error"] == "boom"
    assert rec.tool_events[0]["n_hits"] == len(HITS)


def test_schema_relevant_metadata_preserved():
    rec = RunRecorder()
    wrapped = _wrap_retrieval_tool(clip_search_tool, rec)
    assert wrapped.__name__ == "clip_search_tool"
    assert wrapped.__doc__ == "Original docstring."
    # as_json_schema uses signature + annotations; both must survive wrapping
    assert inspect.signature(wrapped) == inspect.signature(clip_search_tool)
    assert wrapped.__annotations__ == clip_search_tool.__annotations__


def test_frame_inspect_args_recorded():
    def frame_inspect_tool(database, question, time_ranges_hhmmss):
        return "answer"

    rec = RunRecorder()
    wrapped = _wrap_frame_inspect(frame_inspect_tool, rec)
    out = wrapped(database=object(), question="q?",
                  time_ranges_hhmmss=[("00:00:10", "00:00:20")])
    assert out == "answer"
    ev = rec.tool_events[0]
    assert ev["args"]["time_ranges_hhmmss"] == [("00:00:10", "00:00:20")]


def test_token_usage_summary_null_when_unexposed():
    rec = RunRecorder()
    rec.llm_calls.append({"route": "text", "model_name": "m", "usage": None})
    rec.llm_calls.append({"route": "text", "model_name": "m", "usage": None})
    summary = rec.token_usage_summary()
    assert summary["text"]["calls"] == 2
    assert summary["text"]["prompt_tokens"] is None       # never estimated
    assert summary["text"]["usage_source"] == "unavailable"
