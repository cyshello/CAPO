"""Full-recaption intervention runner that captions segments concurrently.

The incumbent caption loop (``fresh_prompt_delta_evidence.py:1470-1507``) issues
one ``segment_captioner.caption`` call at a time. Under full recaption that loop
is the whole cost of a run: every episode recaptions the entire video, so a
351-segment video is 351 blocking calls at ~10s each.

Nothing in that loop is sequential by nature. Unlike the baseline captioner,
which grows the history as it goes and therefore has to be partitioned at frozen
history-block boundaries, the intervention reads its history from the baseline's
``frozen_histories_path`` (``fresh_prompt_delta_evidence.py:1457``) and passes
``histories[segment_id]`` straight through. Every iteration depends only on
frozen baseline state, so the segments are independent.

The persistent GPU pool is already built for concurrency: ``caption_segment``
releases ``_pool_lock`` before it enqueues and waits, routes replies by
request id through per-request queues, and gives each worker its own cache
manifest fragment (``history_aware_baseline.py:1691-1723``). Only the caller was
serial.

Zero-touch strategy (nothing in ``surrogate_rollout`` is edited), in three
passes over the parent ``run()``:

1. **Probe.** Run the parent with a captioner that records its kwargs and
   returns a placeholder, and a mixed-view builder that raises ``_ProbeComplete``
   the moment the caption loop ends. This yields the exact kwargs of every
   caption call -- prompt composition, contract validation and the intervention
   identity hash all computed by the parent, not restated here -- and touches no
   file, because the loop only reads and hashes before ``mixed.build``.
2. **Fan out.** Execute those recorded calls on the real captioner through a
   thread pool sized to the GPU pool.
3. **Replay.** Run the parent again with a captioner that returns the memoized
   ``HistoryAwareCaptionResult`` for each call, so the episode is assembled by
   the parent's own code from the parent's own results, in the parent's order.

Fidelity: captions are decoded greedily (``config.CAPTION_DECODING`` is
``temperature 0.0``/``top_p 1.0``), and each pool worker still serves one
request at a time, so a segment's caption does not depend on which GPU took it
or on what ran beside it. Replay returns the real result objects rather than
re-reading the cache, so ``model_call_count``/``caption_seconds`` stay the
counts the sequential run would have recorded.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from full_recaption_opt.full_recaption_runner import (
    FullRecaptionInterventionRunner,
)


#: Overrides the GPU-pool-derived degree of concurrency when set.
MAX_IN_FLIGHT_ENVIRONMENT_VARIABLE = "FULL_RECAPTION_MAX_IN_FLIGHT"


class _ProbeComplete(Exception):
    """Ends the probe pass once the caption loop has recorded every call."""


def _call_key(kwargs: dict) -> tuple[str, str]:
    """Identify one caption call by segment and intervention identity.

    The identity hash covers the baseline, the delta, the frozen history and the
    composed prompt, so a pair that matches here is a call the parent would have
    issued with byte-identical arguments.
    """
    return (str(kwargs["segment_id"]),
            str(kwargs["intervention_identity_hash"]))


class _CaptionerFacade:
    """Delegate every captioner attribute except ``caption``.

    ``configuration_identity`` in particular has to keep reading through to the
    real captioner: the parent folds it into the episode identity hash, so a
    stand-in that reported its own would change the cache key of every caption
    and the identity of every episode.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _RecordingCaptioner(_CaptionerFacade):
    """Record the caption kwargs the parent computes, without captioning."""

    def __init__(self, inner: Any, calls: list) -> None:
        super().__init__(inner)
        self._calls = calls

    def caption(self, **kwargs: Any):
        self._calls.append(dict(kwargs))
        # The loop reads only ``parsed["clip_description"]`` before the probe
        # ends at ``mixed.build``; the value is discarded with the probe pass.
        return _ProbeCaptionResult()


class _ProbeCaptionResult:
    parsed = {"clip_description": "probe"}


class _MemoizingCaptioner(_CaptionerFacade):
    """Return the already-computed result for each call the parent repeats."""

    def __init__(self, inner: Any, results: dict, misses: list) -> None:
        super().__init__(inner)
        self._results = results
        self._misses = misses

    def caption(self, **kwargs: Any):
        key = _call_key(kwargs)
        if key in self._results:
            return self._results.pop(key)
        # Not reachable while the probe and the replay agree, and harmless if
        # they ever stop agreeing: the fan-out already wrote this caption to the
        # cache, so the real captioner returns it as a hit. Recorded so the
        # divergence is visible instead of silently costing a model call.
        self._misses.append(key)
        return self._inner.caption(**kwargs)


class _StopBeforeMixedView:
    """Ends the probe pass at the first statement after the caption loop."""

    def build(self, **_kwargs: Any):
        raise _ProbeComplete


def _resolve_max_in_flight(captioner: Any) -> int:
    """Match concurrency to the GPU pool, since each worker serves one call."""
    override = os.environ.get(MAX_IN_FLIGHT_ENVIRONMENT_VARIABLE)
    if override:
        value = int(override)
        if value < 1:
            raise ValueError(
                f"{MAX_IN_FLIGHT_ENVIRONMENT_VARIABLE} must be at least 1")
        return value
    dispatcher = getattr(captioner, "remote_dispatcher", None)
    gpus = tuple(getattr(dispatcher, "parallel_gpus", ()) or ())
    # No pool means the captioner runs in this process: stay sequential.
    return max(1, len(gpus))


def _caption_in_parallel(
    captioner: Any, calls: list, max_in_flight: int,
) -> dict:
    results: dict = {}
    if not calls:
        return results
    workers = max(1, min(int(max_in_flight), len(calls)))
    if workers == 1:
        for kwargs in calls:
            results[_call_key(kwargs)] = captioner.caption(**kwargs)
        return results
    lock = threading.Lock()

    def _one(kwargs: dict) -> None:
        result = captioner.caption(**kwargs)
        with lock:
            results[_call_key(kwargs)] = result

    with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="recaption") as pool:
        futures = [pool.submit(_one, kwargs) for kwargs in calls]
        for future in futures:
            # Surface the first failure exactly as the sequential loop would.
            future.result()
    return results


class ParallelFullRecaptionInterventionRunner(FullRecaptionInterventionRunner):
    """Full recaption with the segment captions fanned out across the pool."""

    def run(
        self, *, baseline_video_manifest_path: str,
        parent_meta_prompt_id: str, plan: Any, output_directory: str,
    ):
        arguments = {
            "baseline_video_manifest_path": baseline_video_manifest_path,
            "parent_meta_prompt_id": parent_meta_prompt_id,
            "plan": plan,
            "output_directory": output_directory,
        }
        captioner = self.segment_captioner
        mixed = self.mixed
        calls: list = []
        self.segment_captioner = _RecordingCaptioner(captioner, calls)
        self.mixed = _StopBeforeMixedView()
        try:
            # A saved episode returns before the caption loop, and the probe
            # ends at the first statement after it. Nothing else reaches here.
            return super().run(**arguments)
        except _ProbeComplete:
            pass
        finally:
            self.segment_captioner = captioner
            self.mixed = mixed

        results = _caption_in_parallel(
            captioner, calls, _resolve_max_in_flight(captioner))
        misses: list = []
        self.segment_captioner = _MemoizingCaptioner(
            captioner, results, misses)
        try:
            return super().run(**arguments)
        finally:
            self.segment_captioner = captioner
