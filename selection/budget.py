"""Recaption-budget enforcement (PHASE2_3 §10).

When a selection union exceeds the budget, clips are kept in provenance
priority order (frame inspections first, temporal neighbors last); within one
priority class higher retrieval score wins, then earlier clip start (fully
deterministic). Dropped clips are recorded in metadata — never silently lost.
"""

from __future__ import annotations

from surrogate_rollout.evaluation.rollout_evaluator import SelectionBudget
from surrogate_rollout.selection.base import (
    SOURCE_FRAME_INSPECTION,
    SOURCE_NEIGHBOR,
    SOURCE_PROMPT_DELTA,
    SOURCE_QUESTION,
    SOURCE_RANDOM,
    SOURCE_TRACE_RETRIEVED,
    SOURCE_TRACE_RETURNED,
    SOURCE_TRACE_STRICT,
    SelectionResult,
)

# lower number = kept first
SOURCE_PRIORITY = {
    SOURCE_FRAME_INSPECTION: 0,
    SOURCE_TRACE_STRICT: 1,
    SOURCE_TRACE_RETURNED: 2,
    SOURCE_TRACE_RETRIEVED: 2,
    SOURCE_PROMPT_DELTA: 3,
    SOURCE_QUESTION: 4,
    SOURCE_RANDOM: 4,
    SOURCE_NEIGHBOR: 5,
}


def _clip_start(clip_id: str) -> float:
    return float(clip_id.split("_")[0])


def apply_budget(
    result: SelectionResult,
    budget: SelectionBudget,
    total_clips: int,
) -> SelectionResult:
    limit = budget.max_clips(total_clips)
    if len(result.clip_ids) <= limit:
        result.metadata.setdefault("budget", {}).update(
            {"limit": limit, "dropped": []})
        return result

    def rank(cid: str) -> tuple:
        prio = min(SOURCE_PRIORITY.get(s, 9) for s in result.sources.get(cid, []))
        score = result.scores.get(cid, 0.0)
        return (prio, -score, _clip_start(cid))

    ordered = sorted(result.clip_ids, key=rank)
    kept, dropped = ordered[:limit], ordered[limit:]

    out = SelectionResult(metadata=dict(result.metadata))
    for cid in sorted(kept, key=_clip_start):
        for src in result.sources[cid]:
            out.add(cid, src, result.scores.get(cid))
    out.metadata.setdefault("budget", {}).update(
        {"limit": limit, "dropped": sorted(dropped, key=_clip_start)})
    return out
