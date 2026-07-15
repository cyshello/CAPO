"""Deterministic Stage 4.9 feedback fixture policy."""

from __future__ import annotations

from typing import Mapping, Sequence

from surrogate_rollout.optimization.feedback_generator import (
    _stable_id,
    attribution_id_for_feedback,
)
from surrogate_rollout.optimization.schemas import (
    CounterfactualEvidence,
    FeedbackBatch,
    FeedbackItem,
    MetaKnowledgeItem,
)


MOCK_FEEDBACK_POLICY_VERSION = "mock_feedback_v0001"


def _default_targets(item: CounterfactualEvidence) -> tuple[str, ...]:
    missing_traces = tuple(item.metadata.get("missing_trace_sides", ()))
    if missing_traces or item.outcome_transition in ("unknown", "wrong_to_wrong"):
        return ("insufficient_evidence",)
    if item.score_delta == 0.0:
        return ("insufficient_evidence",)
    # A score change alone does not establish which component caused it.
    return ("insufficient_evidence",)


class MockFeedbackGenerator:
    """Deterministic fixture policy; never performs a model call."""

    def __init__(
        self,
        target_overrides: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._target_overrides = dict(target_overrides or {})

    def generate(
        self,
        evidence: Sequence[CounterfactualEvidence],
        meta_knowledge: Sequence[MetaKnowledgeItem],
    ) -> FeedbackBatch:
        del meta_knowledge
        ordered = sorted(evidence, key=lambda item: item.evidence_id)
        if len({item.evidence_id for item in ordered}) != len(ordered):
            raise ValueError("evidence contains duplicate evidence_id values")
        items = []
        for item in ordered:
            targets = self._target_overrides.get(
                item.evidence_id, _default_targets(item))
            feedback_id = _stable_id("feedback_", {
                "policy": MOCK_FEEDBACK_POLICY_VERSION,
                "evidence_id": item.evidence_id,
                "targets": targets,
            })
            insufficient = targets == ("insufficient_evidence",)
            items.append(FeedbackItem(
                feedback_id=feedback_id,
                evidence_ids=(item.evidence_id,),
                attribution_id=attribution_id_for_feedback(
                    feedback_id, (item.evidence_id,)),
                target_components=tuple(targets),
                failure_modes=(("causal_component_not_isolated",)
                               if insufficient else ("mock_targeted_failure",)),
                successful_behaviors=(("candidate_improved_saved_score",)
                                      if item.score_delta > 0 else ()),
                desired_behaviors=("collect_repeated_component_specific_evidence",),
                avoid_behaviors=("promote_single_example_to_global_knowledge",),
                applicable_segment_traits={
                    "segment_ids": item.segment_ids,
                    "outcome_transition": item.outcome_transition,
                },
                confidence=(0.35 if insufficient else 0.8),
                rationale=(
                    "Saved evidence does not isolate a causal component."
                    if insufficient else
                    "Deterministic fixture target supplied for attribution testing."
                ),
            ))
        return FeedbackBatch(
            feedback_policy="deterministic_mock",
            feedback_policy_version=MOCK_FEEDBACK_POLICY_VERSION,
            input_evidence_ids=tuple(item.evidence_id for item in ordered),
            attributions=(), items=tuple(items), raw_response_artifact=None,
            parse_errors=(),
        )

