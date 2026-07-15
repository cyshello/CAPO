"""Structured deterministic feedback overrides for saved Stage 4 fixtures."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from surrogate_rollout.optimization.policies.mock_feedback import (
    MockFeedbackGenerator,
)
from surrogate_rollout.optimization.schemas import (
    CounterfactualEvidence,
    FeedbackBatch,
    MetaKnowledgeItem,
)


class SavedFixtureFeedbackGenerator:
    """Wrap the mock policy with explicit saved, non-prose fixture labels."""

    def __init__(self, overrides: Mapping[str, Mapping]) -> None:
        self._overrides = {str(key): dict(value)
                           for key, value in overrides.items()}

    def generate(
        self,
        evidence: Sequence[CounterfactualEvidence],
        meta_knowledge: Sequence[MetaKnowledgeItem],
    ) -> FeedbackBatch:
        missing = {item.question_id for item in evidence} - set(self._overrides)
        if missing:
            raise ValueError(
                f"saved feedback overrides missing questions: {sorted(missing)}")
        targets = {
            item.evidence_id: tuple(self._overrides[item.question_id]["targets"])
            for item in evidence
        }
        batch = MockFeedbackGenerator(targets).generate(evidence, meta_knowledge)
        evidence_by_id = {item.evidence_id: item for item in evidence}
        items = []
        for item in batch.items:
            record = evidence_by_id[item.evidence_ids[0]]
            override = self._overrides[record.question_id]
            items.append(replace(
                item,
                failure_modes=tuple(override["failure_modes"]),
                desired_behaviors=tuple(override["desired_behaviors"]),
                avoid_behaviors=tuple(override.get("avoid_behaviors") or ()),
                applicable_segment_traits=override.get("applicable_segment_traits")
                or {},
                confidence=float(override.get("confidence", 0.9)),
                rationale="Deterministic labels loaded from a saved fixture.",
            ))
        return replace(batch, items=tuple(items))
