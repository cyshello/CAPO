"""Non-invasive instrumentation of a DVD run.

Two recorders, both installed by monkeypatching *names* (the same pattern
dvd_backend.install_backend already uses) — no vendored DVD file is edited and
agent behavior is unchanged:

1. Tool sidecar: wraps clip_search_tool / global_browse_tool /
   frame_inspect_tool in the dvd_core namespace. For the two retrieval tools
   the wrapper shims `database.query` for the duration of the call to capture
   the exact hits (segment time ranges) the vector DB returned. Events land in
   a per-run list and are flushed to `tool_events.jsonl`.

2. LLM call recorder: wraps dvd_backend.make_router so every routed model call
   (vision / tool-calling / plain text) is logged with route, model, message
   sizes, latency, and token usage. None of the current backends exposes token
   counts through DVD's return shape, so token fields are recorded as None
   (never estimated) with `usage_source: "unavailable"`.

Install returns a RunRecorder whose .events / .llm_calls are plain lists;
uninstall() restores every patched name. Not thread-safe across concurrent
DVD runs in one process (DVD runs are sequential in this harness).
"""

from __future__ import annotations

import functools
import json
import time
from typing import Any


class RunRecorder:
    def __init__(self) -> None:
        self.tool_events: list[dict] = []
        self.llm_calls: list[dict] = []
        self._uninstallers: list = []

    # ------------------------------------------------------------------ #
    def dump(self, tool_events_path: str, llm_calls_path: str) -> None:
        with open(tool_events_path, "w") as f:
            for e in self.tool_events:
                f.write(json.dumps(e, default=str) + "\n")
        with open(llm_calls_path, "w") as f:
            for e in self.llm_calls:
                f.write(json.dumps(e, default=str) + "\n")

    def token_usage_summary(self) -> dict[str, Any]:
        """Per-route call counts; token totals None when never exposed."""
        summary: dict[str, Any] = {}
        for call in self.llm_calls:
            r = summary.setdefault(
                call["route"],
                {"calls": 0, "prompt_tokens": None, "completion_tokens": None,
                 "usage_source": "unavailable"},
            )
            r["calls"] += 1
            usage = call.get("usage")
            if usage:  # a backend that exposes usage would fill this
                r["prompt_tokens"] = (r["prompt_tokens"] or 0) + usage.get("prompt_tokens", 0)
                r["completion_tokens"] = (r["completion_tokens"] or 0) + usage.get("completion_tokens", 0)
                r["usage_source"] = "backend"
        return summary

    def uninstall(self) -> None:
        for undo in reversed(self._uninstallers):
            undo()
        self._uninstallers.clear()


def _hit_record(hit: dict) -> dict:
    return {
        "time_start_secs": hit.get("time_start_secs"),
        "time_end_secs": hit.get("time_end_secs"),
        "distance": hit.get("__metrics__"),
    }


def _wrap_retrieval_tool(tool, recorder: RunRecorder):
    @functools.wraps(tool)
    def wrapped(database, **kwargs):
        hits: list[dict] = []
        orig_query = database.query

        def recording_query(*qargs, **qkwargs):
            results = orig_query(*qargs, **qkwargs)
            hits.extend(results)
            return results

        database.query = recording_query
        t0 = time.time()
        error = None
        try:
            result = tool(database=database, **kwargs)
        except Exception as e:
            error = str(e)
            raise
        finally:
            database.query = orig_query
            recorder.tool_events.append({
                "tool": tool.__name__,
                "args": {k: v for k, v in kwargs.items()},
                "hits": [_hit_record(h) for h in hits],
                "n_hits": len(hits),
                "latency_seconds": time.time() - t0,
                "error": error,
            })
        return result

    return wrapped


def _wrap_frame_inspect(tool, recorder: RunRecorder):
    @functools.wraps(tool)
    def wrapped(database, question, time_ranges_hhmmss):
        t0 = time.time()
        error = None
        try:
            return tool(database=database, question=question,
                        time_ranges_hhmmss=time_ranges_hhmmss)
        except Exception as e:
            error = str(e)
            raise
        finally:
            recorder.tool_events.append({
                "tool": tool.__name__,
                "args": {"question": question,
                         "time_ranges_hhmmss": time_ranges_hhmmss},
                "hits": [],
                "n_hits": 0,
                "latency_seconds": time.time() - t0,
                "error": error,
            })

    return wrapped


def _wrap_router_factory(make_router, recorder: RunRecorder):
    @functools.wraps(make_router)
    def factory(*fargs, **fkwargs):
        router = make_router(*fargs, **fkwargs)

        @functools.wraps(router)
        def recording_router(messages, endpoints, model_name, api_key=None,
                             tools=(), image_paths=(), max_tokens=4096,
                             temperature=0.0, tool_choice="auto",
                             return_json=False):
            route = ("vision" if image_paths else
                     "tool_calling" if tools else "text")
            t0 = time.time()
            error = None
            resp = None
            try:
                resp = router(messages, endpoints, model_name, api_key=api_key,
                              tools=tools, image_paths=image_paths,
                              max_tokens=max_tokens, temperature=temperature,
                              tool_choice=tool_choice, return_json=return_json)
                return resp
            except Exception as e:
                error = str(e)
                raise
            finally:
                recorder.llm_calls.append({
                    "route": route,
                    "model_name": model_name,
                    "n_messages": len(messages or []),
                    "n_images": len(image_paths or ()),
                    "prompt_chars": sum(len(str(m.get("content") or ""))
                                        for m in (messages or [])),
                    "response_chars": len(str((resp or {}).get("content") or "")),
                    "has_tool_calls": bool((resp or {}).get("tool_calls")),
                    "usage": None,  # DVD's return shape drops usage; never estimate
                    "latency_seconds": time.time() - t0,
                    "error": error,
                })

        return recording_router

    return factory


def install(recorder: RunRecorder | None = None) -> RunRecorder:
    """Patch dvd_core tool names and dvd_backend.make_router. Call BEFORE
    run_dvd (agent binds tools at construction). Idempotent per recorder."""
    import dvd.dvd_core as dvd_core
    import dvd_backend

    rec = recorder or RunRecorder()

    orig_clip = dvd_core.clip_search_tool
    orig_browse = dvd_core.global_browse_tool
    orig_inspect = dvd_core.frame_inspect_tool
    orig_factory = dvd_backend.make_router

    dvd_core.clip_search_tool = _wrap_retrieval_tool(orig_clip, rec)
    dvd_core.global_browse_tool = _wrap_retrieval_tool(orig_browse, rec)
    dvd_core.frame_inspect_tool = _wrap_frame_inspect(orig_inspect, rec)
    dvd_backend.make_router = _wrap_router_factory(orig_factory, rec)

    def undo():
        dvd_core.clip_search_tool = orig_clip
        dvd_core.global_browse_tool = orig_browse
        dvd_core.frame_inspect_tool = orig_inspect
        dvd_backend.make_router = orig_factory

    rec._uninstallers.append(undo)
    return rec
