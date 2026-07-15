"""Offline deterministic empirical validation for Stage 4.11."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from surrogate_rollout.optimization.schemas import ComponentValidationResult
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import QAExample, sha256_text


@dataclass(frozen=True)
class ValidationThresholds:
    min_confirmation_gain: Mapping[str, float] = field(default_factory=lambda: {
        "prompt_bank": 0.01,
        "router": 0.01,
        "scaffold": 0.01,
    })
    max_accuracy_drop: Mapping[str, float] = field(default_factory=lambda: {
        "prompt_bank": 0.0,
        "router": 0.0,
        "scaffold": 0.0,
    })
    max_correct_to_wrong_regressions: Mapping[str, int] = field(
        default_factory=lambda: {
            "prompt_bank": 0,
            "router": 0,
            "scaffold": 0,
        })
    min_scaffold_support_videos: int = 3


class UpdateValidator(Protocol):
    def validate(
        self,
        *,
        incumbent_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
        component: str,
        confirmation_examples: Sequence[QAExample],
        regression_examples: Sequence[QAExample],
    ) -> ComponentValidationResult:
        ...


def _average(scores: Mapping[str, float], examples: Sequence[QAExample]) -> float:
    return sum(float(scores[item.question_id]) for item in examples) / len(examples)


class DeterministicUpdateValidator:
    def __init__(self, thresholds: ValidationThresholds | None = None) -> None:
        self.thresholds = thresholds or ValidationThresholds()

    def validate(
        self,
        *,
        incumbent_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
        component: str,
        confirmation_examples: Sequence[QAExample],
        regression_examples: Sequence[QAExample],
    ) -> ComponentValidationResult:
        if component not in ("prompt_bank", "router", "scaffold"):
            raise ValueError("invalid validation component")
        before = dumps_canonical({
            "incumbent": incumbent_state, "candidate": candidate_state})
        incumbent_version = str(incumbent_state["version"])
        candidate_version = (str(candidate_state["version"])
                             if candidate_state.get("version") else None)
        evidence_ids = tuple(candidate_state.get("proposal_example_ids") or ())
        proposal_videos = set(candidate_state.get("proposal_video_ids") or ())
        confirmation = tuple(confirmation_examples)
        regression = tuple(regression_examples)
        confirmation_ids = tuple(item.question_id for item in confirmation)
        regression_ids = tuple(item.question_id for item in regression)
        reasons = []
        regressions = []
        decision = None

        if not confirmation or not regression:
            decision = "defer"
            reasons.append("confirmation and regression splits must both be non-empty")
        if len(set(confirmation_ids)) != len(confirmation_ids) or \
                len(set(regression_ids)) != len(regression_ids):
            decision = "defer"
            reasons.append("validation splits contain duplicate question IDs")
        overlap = set(evidence_ids) & set(confirmation_ids)
        if overlap:
            decision = "defer"
            reasons.append(
                f"proposal and confirmation example IDs overlap: {sorted(overlap)}")
        video_overlap = proposal_videos & {item.video_id for item in confirmation}
        if video_overlap:
            decision = "defer"
            reasons.append(
                f"proposal and confirmation videos overlap: {sorted(video_overlap)}")
        regression_evidence_overlap = set(evidence_ids) & set(regression_ids)
        if regression_evidence_overlap:
            decision = "defer"
            reasons.append(
                "proposal and regression example IDs overlap: "
                f"{sorted(regression_evidence_overlap)}")
        regression_video_overlap = proposal_videos & {
            item.video_id for item in regression}
        if regression_video_overlap:
            decision = "defer"
            reasons.append(
                "proposal and regression videos overlap: "
                f"{sorted(regression_video_overlap)}")
        split_overlap = set(confirmation_ids) & set(regression_ids)
        if split_overlap:
            decision = "defer"
            reasons.append(
                f"confirmation and regression IDs overlap: {sorted(split_overlap)}")

        all_examples = (*confirmation, *regression)
        incumbent_scores = incumbent_state.get("scores") or {}
        candidate_scores = candidate_state.get("scores") or {}
        errors = candidate_state.get("errors") or {}
        if not isinstance(errors, Mapping):
            raise TypeError("candidate_state.errors must be a mapping")
        missing_scores = [item.question_id for item in all_examples
                          if item.question_id not in incumbent_scores
                          or item.question_id not in candidate_scores]
        evaluation_errors = [f"{key}: {value}" for key, value in errors.items()
                             if key in {item.question_id for item in all_examples}]
        if missing_scores or evaluation_errors:
            decision = "evaluation_failed"
            if missing_scores:
                reasons.append(f"missing saved scores: {sorted(missing_scores)}")
            reasons.extend(evaluation_errors)

        if component == "scaffold" and decision != "evaluation_failed":
            support_videos = set(candidate_state.get("support_video_ids") or ())
            if len(support_videos) < self.thresholds.min_scaffold_support_videos:
                decision = "reject"
                reasons.append(
                    "scaffold candidate has insufficient distinct support videos: "
                    f"{len(support_videos)} < "
                    f"{self.thresholds.min_scaffold_support_videos}")
            incumbent_contract = incumbent_state.get("contract_version")
            candidate_contract = candidate_state.get("contract_version")
            if not incumbent_contract or candidate_contract != incumbent_contract:
                decision = "reject"
                reasons.append("scaffold candidate changes or omits the fixed contract")

        incumbent_metrics = {}
        candidate_metrics = {}
        if decision != "evaluation_failed" and confirmation and regression and \
                not missing_scores:
            incumbent_confirmation = _average(incumbent_scores, confirmation)
            candidate_confirmation = _average(candidate_scores, confirmation)
            incumbent_regression = _average(incumbent_scores, regression)
            candidate_regression = _average(candidate_scores, regression)
            confirmation_gain = candidate_confirmation - incumbent_confirmation
            regression_drop = incumbent_regression - candidate_regression
            correct_to_wrong = tuple(
                item.question_id for item in regression
                if float(incumbent_scores[item.question_id]) > 0.0
                and float(candidate_scores[item.question_id]) <= 0.0)
            incumbent_metrics = {
                "confirmation_accuracy": incumbent_confirmation,
                "regression_accuracy": incumbent_regression,
            }
            candidate_metrics = {
                "confirmation_accuracy": candidate_confirmation,
                "regression_accuracy": candidate_regression,
                "confirmation_gain": confirmation_gain,
                "regression_drop": regression_drop,
                "correct_to_wrong_regressions": float(len(correct_to_wrong)),
            }
            regressions.extend(correct_to_wrong)
            if decision is None:
                if regression_drop > self.thresholds.max_accuracy_drop[component] or \
                        len(correct_to_wrong) > \
                        self.thresholds.max_correct_to_wrong_regressions[component]:
                    decision = "reject"
                    reasons.append("candidate exceeds component regression threshold")
                elif confirmation_gain >= \
                        self.thresholds.min_confirmation_gain[component]:
                    decision = "accept"
                    reasons.append("confirmation improvement meets configured threshold")
                else:
                    decision = "defer"
                    reasons.append("confirmation result is inconclusive")

        if decision is None:
            decision = "defer"
            reasons.append("validation did not produce a conclusive decision")
        identity = {
            "component": component,
            "incumbent_version": incumbent_version,
            "candidate_version": candidate_version,
            "evidence_ids": evidence_ids,
            "confirmation_ids": confirmation_ids,
            "regression_ids": regression_ids,
            "incumbent_metrics": incumbent_metrics,
            "candidate_metrics": candidate_metrics,
            "decision": decision,
            "reasons": tuple(reasons),
        }
        result = ComponentValidationResult(
            validation_id="validation_" + sha256_text(
                dumps_canonical(identity))[:24],
            component=component,
            incumbent_version=incumbent_version,
            candidate_version=candidate_version,
            evidence_example_ids=evidence_ids,
            confirmation_example_ids=confirmation_ids,
            regression_example_ids=regression_ids,
            incumbent_metrics=incumbent_metrics,
            candidate_metrics=candidate_metrics,
            regressions=tuple(regressions),
            decision=decision,
            reasons=tuple(reasons),
        )
        after = dumps_canonical({
            "incumbent": incumbent_state, "candidate": candidate_state})
        if after != before:
            raise RuntimeError("validation mutated saved input states")
        return result
