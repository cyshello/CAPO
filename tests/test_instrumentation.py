"""Wrapper-level tests (no DVD import, no GPU): hit capture, query restoration,
signature preservation, and behavior transparency."""

import inspect
from typing import Annotated

import pytest

from surrogate_rollout.instrumentation import (
    RunRecorder,
    _wrap_frame_inspect,
    _wrap_retrieval_tool,
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
