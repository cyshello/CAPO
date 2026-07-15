"""Stage 4.9 feedback policies and strict saved-response parsing."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Mapping, Protocol, Sequence

from surrogate_rollout.optimization.schemas import (
    CounterfactualEvidence,
    FailureAttribution,
    FeedbackBatch,
    FeedbackItem,
    MetaKnowledgeItem,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_text


class FeedbackGenerator(Protocol):
    def generate(
        self,
        evidence: Sequence[CounterfactualEvidence],
        meta_knowledge: Sequence[MetaKnowledgeItem],
    ) -> FeedbackBatch:
        ...


class FeedbackParseError(RuntimeError):
    def __init__(self, message: str, *, raw_response_artifact: str):
        super().__init__(message)
        self.raw_response_artifact = raw_response_artifact


def _stable_id(prefix: str, payload: object) -> str:
    return prefix + sha256_text(dumps_canonical(payload))[:24]


def attribution_id_for_feedback(
    feedback_id: str, evidence_ids: Sequence[str],
) -> str:
    return _stable_id("attribution_", {
        "feedback_id": feedback_id,
        "evidence_ids": tuple(evidence_ids),
    })


def failure_attribution_from_json(data: Mapping) -> FailureAttribution:
    return FailureAttribution(
        attribution_id=data["attribution_id"],
        evidence_ids=tuple(data["evidence_ids"]),
        targets=tuple(data["targets"]),
        prompt_bank_issues=tuple(data.get("prompt_bank_issues") or ()),
        router_issues=tuple(data.get("router_issues") or ()),
        scaffold_issues=tuple(data.get("scaffold_issues") or ()),
        supporting_facts=tuple(data.get("supporting_facts") or ()),
        confidence=float(data["confidence"]),
        rationale=data.get("rationale", ""),
    )


def feedback_item_from_json(data: Mapping) -> FeedbackItem:
    return FeedbackItem(
        feedback_id=data["feedback_id"],
        evidence_ids=tuple(data["evidence_ids"]),
        attribution_id=data["attribution_id"],
        target_components=tuple(data["target_components"]),
        failure_modes=tuple(data.get("failure_modes") or ()),
        successful_behaviors=tuple(data.get("successful_behaviors") or ()),
        desired_behaviors=tuple(data.get("desired_behaviors") or ()),
        avoid_behaviors=tuple(data.get("avoid_behaviors") or ()),
        applicable_segment_traits=data.get("applicable_segment_traits") or {},
        confidence=float(data["confidence"]),
        rationale=data.get("rationale", ""),
    )


def feedback_batch_from_json(
    data: Mapping, *, raw_response_artifact: str | None = None,
) -> FeedbackBatch:
    return FeedbackBatch(
        feedback_policy=data["feedback_policy"],
        feedback_policy_version=data["feedback_policy_version"],
        input_evidence_ids=tuple(data["input_evidence_ids"]),
        attributions=tuple(failure_attribution_from_json(item)
                           for item in data.get("attributions") or ()),
        items=tuple(feedback_item_from_json(item)
                    for item in data.get("items") or ()),
        raw_response_artifact=(raw_response_artifact
                               if raw_response_artifact is not None
                               else data.get("raw_response_artifact")),
        parse_errors=tuple(data.get("parse_errors") or ()),
    )


def attach_attributions(
    batch: FeedbackBatch, attributions: Sequence[FailureAttribution],
) -> FeedbackBatch:
    return replace(batch, attributions=tuple(attributions))


def write_feedback_batch(batch: FeedbackBatch, path: str) -> str:
    _atomic_write_text(path, dumps_canonical(batch) + "\n")
    return path


def read_feedback_batch(path: str) -> FeedbackBatch:
    with open(path) as f:
        data = json.load(f)
    return feedback_batch_from_json(data)
