"""TraceReferenceSelector (PHASE2_3 §5): reasoning-trace references.

Reuses references/extractor + references/expansion over the *baseline*
trajectory and tool events. The reference policy (config.REFERENCE_POLICIES)
decides which evidence sets count; temporal-neighbor expansion radius is the
policy's radius capped by the budget's.
"""

from __future__ import annotations

from surrogate_rollout import config
from surrogate_rollout.references.expansion import expand_references
from surrogate_rollout.references.extractor import extract_references
from surrogate_rollout.selection.base import (
    SOURCE_FRAME_INSPECTION,
    SOURCE_NEIGHBOR,
    SOURCE_TRACE_RETRIEVED,
    SOURCE_TRACE_RETURNED,
    SOURCE_TRACE_STRICT,
    SelectionContext,
    SelectionResult,
)

_SET_TO_SOURCE = {
    "frame_inspected_segments": SOURCE_FRAME_INSPECTION,
    "explicitly_cited_segments": SOURCE_TRACE_STRICT,
    "returned_segments": SOURCE_TRACE_RETURNED,
    "retrieved_segments": SOURCE_TRACE_RETRIEVED,
    "consumed_segments": SOURCE_TRACE_RETURNED,
}


class TraceReferenceSelector:
    def select(self, context: SelectionContext) -> SelectionResult:
        policy = config.REFERENCE_POLICIES[context.reference_policy]
        refs = extract_references(
            list(context.baseline_trajectory),
            list(context.baseline_tool_events),
            list(context.all_clip_ids),
        )

        result = SelectionResult()
        base: set[str] = set()
        for set_name in policy.base_sets:
            members = getattr(refs, set_name)
            base |= members
            for cid in members:
                result.add(cid, _SET_TO_SOURCE[set_name])

        radius = min(policy.neighbor_radius, context.budget.neighbor_radius)
        if base and radius > 0:
            expanded, evidence = expand_references(
                base, list(context.all_clip_ids), radius=radius)
            for cid in expanded - base:
                result.add(cid, SOURCE_NEIGHBOR)
            result.metadata["expansion_evidence"] = evidence
        result.metadata["reference_policy"] = policy.name
        result.metadata["neighbor_radius"] = radius
        result.metadata["trace_reference_count"] = len(base)
        return result
