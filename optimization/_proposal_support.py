"""Shared structured-support checks for independent Stage 4.10 proposers."""

from __future__ import annotations

from typing import Sequence

from surrogate_rollout.optimization.schemas import (
    FailureAttribution,
    FeedbackBatch,
    FeedbackItem,
)


class ProposalInputError(RuntimeError):
    """Feedback/attribution lineage is incomplete or inconsistent."""


def component_feedback(
    feedback: FeedbackBatch,
    component: str,
    *,
    min_confidence: float,
) -> tuple[tuple[tuple[FeedbackItem, FailureAttribution], ...],
           tuple[tuple[FeedbackItem, FailureAttribution], ...]]:
    attributions = {item.attribution_id: item for item in feedback.attributions}
    if len(attributions) != len(feedback.attributions):
        raise ProposalInputError("duplicate attribution IDs")
    relevant = []
    supported = []
    issue_field = {
        "prompt_bank": "prompt_bank_issues",
        "router": "router_issues",
        "scaffold": "scaffold_issues",
    }[component]
    for item in feedback.items:
        attribution = attributions.get(item.attribution_id)
        if attribution is None:
            raise ProposalInputError(
                f"feedback item {item.feedback_id!r} has no attribution")
        direct = attribution.targets == (component,)
        multiple = attribution.targets == ("multiple",) and bool(
            getattr(attribution, issue_field))
        if not (direct or multiple):
            continue
        pair = (item, attribution)
        relevant.append(pair)
        if min(item.confidence, attribution.confidence) >= min_confidence:
            supported.append(pair)
    return tuple(relevant), tuple(supported)


def source_feedback_ids(
    pairs: Sequence[tuple[FeedbackItem, FailureAttribution]],
) -> tuple[str, ...]:
    return tuple(sorted({item.feedback_id for item, _ in pairs}))

