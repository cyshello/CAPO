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
import re
import time
from typing import Any

from surrogate_rollout import config


class RunRecorder:
    def __init__(self) -> None:
        self.tool_events: list[dict] = []
        self.llm_calls: list[dict] = []
        self._uninstallers: list = []
        self.frame_inspect_argument_failures = 0

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


def _wrap_retrieval_tool(
    tool, recorder: RunRecorder, *, forced_top_k: int | None = None,
):
    @functools.wraps(tool)
    def wrapped(database, **kwargs):
        requested_kwargs = dict(kwargs)
        if forced_top_k is not None:
            kwargs["top_k"] = forced_top_k
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
            event = {
                "tool": tool.__name__,
                "args": {k: v for k, v in kwargs.items()},
                "hits": [_hit_record(h) for h in hits],
                "n_hits": len(hits),
                "latency_seconds": time.time() - t0,
                "error": error,
            }
            if forced_top_k is not None:
                event["requested_args"] = requested_kwargs
                event["argument_override"] = {
                    "field": "top_k", "executed_value": forced_top_k,
                    "policy_version": config.DVD_CLIP_SEARCH_POLICY_VERSION,
                }
            recorder.tool_events.append(event)
        return result

    return wrapped


_HHMMSS_ARGUMENT = re.compile(
    r"^[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?$")


class FrameInspectArgumentValidationError(ValueError):
    pass


def _frame_inspect_argument_error(value: Any) -> str | None:
    if not isinstance(value, list) or not value:
        return "time_ranges_hhmmss must be a non-empty array"
    for index, pair in enumerate(value):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return f"time_ranges_hhmmss[{index}] must contain exactly two values"
        for endpoint_index, endpoint in enumerate(pair):
            if not isinstance(endpoint, str) or not _HHMMSS_ARGUMENT.fullmatch(
                    endpoint):
                return (
                    f"time_ranges_hhmmss[{index}][{endpoint_index}] must be "
                    "an HH:MM:SS string")
            _hours, minutes, seconds = endpoint.split(".", 1)[0].split(":")
            if int(minutes) >= 60 or int(seconds) >= 60:
                return (
                    f"time_ranges_hhmmss[{index}][{endpoint_index}] has an "
                    "invalid minute or second")
    return None


def _wrap_frame_inspect(tool, recorder: RunRecorder):
    @functools.wraps(tool)
    def wrapped(database, question, time_ranges_hhmmss):
        t0 = time.time()
        error = None
        validation_error = _frame_inspect_argument_error(time_ranges_hhmmss)
        if validation_error is not None:
            recorder.frame_inspect_argument_failures += 1
            retry_index = recorder.frame_inspect_argument_failures
            retry_limit = config.DVD_FRAME_INSPECT_CORRECTIVE_RETRY_LIMIT
            event = {
                "tool": tool.__name__,
                "args": {"question": question,
                         "time_ranges_hhmmss": time_ranges_hhmmss},
                "hits": [], "n_hits": 0,
                "latency_seconds": time.time() - t0,
                "error": validation_error,
                "status": "argument_validation_error",
                "execution_performed": False,
                "corrective_retry_index": retry_index,
                "corrective_retry_limit": retry_limit,
                "policy_version":
                    config.DVD_FRAME_INSPECT_TOOL_CONTRACT_VERSION,
            }
            recorder.tool_events.append(event)
            if retry_index > retry_limit:
                raise FrameInspectArgumentValidationError(
                    "frame_inspect_tool argument validation failed after the "
                    f"single corrective retry: {validation_error}")
            return (
                "Error: invalid frame_inspect_tool arguments. "
                f"{validation_error}. Correct the arguments and call "
                "frame_inspect_tool again using only HH:MM:SS strings. "
                "Exactly one corrective retry is allowed.")
        try:
            result = tool(database=database, question=question,
                          time_ranges_hhmmss=time_ranges_hhmmss)
            return result
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
                "status": "completed" if error is None else "execution_error",
                "execution_performed": True,
                "corrective_retry_index":
                    recorder.frame_inspect_argument_failures,
                "corrective_retry_limit":
                    config.DVD_FRAME_INSPECT_CORRECTIVE_RETRY_LIMIT,
                "policy_version":
                    config.DVD_FRAME_INSPECT_TOOL_CONTRACT_VERSION,
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


def install(
    recorder: RunRecorder | None = None, *, clip_search_top_k: int | None = None,
) -> RunRecorder:
    """Patch dvd_core tool names and dvd_backend.make_router. Call BEFORE
    run_dvd (agent binds tools at construction). Idempotent per recorder."""
    import dvd.dvd_core as dvd_core
    import dvd_backend

    rec = recorder or RunRecorder()

    orig_clip = dvd_core.clip_search_tool
    orig_browse = dvd_core.global_browse_tool
    orig_inspect = dvd_core.frame_inspect_tool
    orig_factory = dvd_backend.make_router

    if clip_search_top_k is not None and clip_search_top_k < 1:
        raise ValueError("clip_search_top_k must be positive")
    dvd_core.clip_search_tool = _wrap_retrieval_tool(
        orig_clip, rec, forced_top_k=clip_search_top_k)
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
