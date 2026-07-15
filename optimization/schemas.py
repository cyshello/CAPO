"""Frozen Phase 4 evidence, feedback, attribution, and knowledge records."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

from surrogate_rollout.prompt_routing.schemas import (
    ScaffoldPolicySnapshot,
    validate_component_version,
)


OUTCOME_TRANSITIONS = (
    "correct_to_correct",
    "correct_to_wrong",
    "wrong_to_correct",
    "wrong_to_wrong",
    "unknown",
)

ATTRIBUTION_TARGETS = (
    "prompt_bank",
    "router",
    "scaffold",
    "multiple",
    "insufficient_evidence",
)

KNOWLEDGE_TYPES = (
    "successful_behavior",
    "failure_pattern",
    "routing_pattern",
    "composition_pattern",
    "rejected_update",
    "accepted_update",
    "conflict",
)

KNOWLEDGE_SCOPES = ("local_prompt", "routing", "global_scaffold", "meta_only")
KNOWLEDGE_STATUSES = ("candidate", "confirmed", "rejected", "deprecated")
PROMPT_BANK_OPERATION_TYPES = (
    "add_entry", "revise_entry", "merge_entries", "retire_entry", "no_op",
)
SCAFFOLD_SKIP_REASONS = (
    "disabled_by_config", "no_scaffold_attribution", "insufficient_support", "none",
)


def _freeze_json(value: Any, where: str) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): _freeze_json(item, f"{where}[{key!r}]")
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{where}[]") for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{where} contains non-JSON value {type(value).__name__}")


def _freeze_mapping(record: Any, name: str) -> None:
    value = getattr(record, name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{type(record).__name__}.{name} must be a Mapping")
    object.__setattr__(record, name, _freeze_json(
        value, f"{type(record).__name__}.{name}"))


def _require_str_tuple(value: tuple[str, ...], name: str) -> None:
    if not isinstance(value, tuple) or any(not isinstance(v, str) for v in value):
        raise TypeError(f"{name} must be a tuple of str")


def _require_confidence(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _require_nonempty_str(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class CaptionDifference:
    segment_id: str
    baseline_caption: str
    candidate_caption: str
    difference_summary: str | None
    structured_differences: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("CaptionDifference.segment_id must be non-empty")
        if not isinstance(self.baseline_caption, str) or \
                not isinstance(self.candidate_caption, str):
            raise TypeError("CaptionDifference captions must be strings")
        _freeze_mapping(self, "structured_differences")


@dataclass(frozen=True)
class CounterfactualEvidence:
    evidence_id: str
    video_id: str
    question_id: str
    ground_truth: str | None

    baseline_bank_version: str
    baseline_router_version: str
    baseline_scaffold_version: str
    candidate_bank_version: str
    candidate_router_version: str
    candidate_scaffold_version: str

    baseline_selected_prompt_ids: Mapping[str, tuple[str, ...]]
    candidate_selected_prompt_ids: Mapping[str, tuple[str, ...]]
    baseline_composed_prompt_refs: Mapping[str, str]
    candidate_composed_prompt_refs: Mapping[str, str]

    baseline_answer: str | None
    candidate_answer: str | None
    baseline_score: float
    candidate_score: float
    score_delta: float
    outcome_transition: Literal[
        "correct_to_correct",
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_wrong",
        "unknown",
    ]

    segment_ids: tuple[str, ...]
    caption_differences: tuple[CaptionDifference, ...]
    baseline_trajectory_refs: tuple[str, ...]
    candidate_trajectory_refs: tuple[str, ...]
    rollout_artifact_refs: Mapping[str, str]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "evidence_id", "video_id", "question_id",
            "baseline_bank_version", "baseline_router_version",
            "baseline_scaffold_version", "candidate_bank_version",
            "candidate_router_version", "candidate_scaffold_version",
        ):
            if not getattr(self, name):
                raise ValueError(f"CounterfactualEvidence.{name} must be non-empty")
        if self.outcome_transition not in OUTCOME_TRANSITIONS:
            raise ValueError(
                f"invalid outcome_transition {self.outcome_transition!r}")
        if abs(self.score_delta - (
                self.candidate_score - self.baseline_score)) > 1e-12:
            raise ValueError("score_delta must equal candidate_score - baseline_score")
        for name in (
            "segment_ids", "baseline_trajectory_refs",
            "candidate_trajectory_refs",
        ):
            _require_str_tuple(getattr(self, name), name)
        if len(self.segment_ids) != len(set(self.segment_ids)):
            raise ValueError("CounterfactualEvidence.segment_ids contains duplicates")
        if not isinstance(self.caption_differences, tuple) or any(
                not isinstance(item, CaptionDifference)
                for item in self.caption_differences):
            raise TypeError("caption_differences must contain CaptionDifference records")
        for name in (
            "baseline_selected_prompt_ids", "candidate_selected_prompt_ids",
            "baseline_composed_prompt_refs", "candidate_composed_prompt_refs",
            "rollout_artifact_refs", "metadata",
        ):
            _freeze_mapping(self, name)


@dataclass(frozen=True)
class FailureAttribution:
    attribution_id: str
    evidence_ids: tuple[str, ...]
    targets: tuple[Literal[
        "prompt_bank", "router", "scaffold", "multiple",
        "insufficient_evidence",
    ], ...]
    prompt_bank_issues: tuple[str, ...]
    router_issues: tuple[str, ...]
    scaffold_issues: tuple[str, ...]
    supporting_facts: tuple[str, ...]
    confidence: float
    rationale: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.attribution_id, "attribution_id")
        for name in (
            "evidence_ids", "targets", "prompt_bank_issues", "router_issues",
            "scaffold_issues", "supporting_facts",
        ):
            _require_str_tuple(getattr(self, name), name)
        if not self.evidence_ids:
            raise ValueError("FailureAttribution.evidence_ids must not be empty")
        if not self.targets or any(target not in ATTRIBUTION_TARGETS
                                   for target in self.targets):
            raise ValueError("FailureAttribution.targets contains an invalid target")
        if len(self.targets) != len(set(self.targets)):
            raise ValueError("FailureAttribution.targets contains duplicates")
        _require_confidence(self.confidence, "FailureAttribution.confidence")
        if not isinstance(self.rationale, str):
            raise TypeError("FailureAttribution.rationale must be a string")


@dataclass(frozen=True)
class FeedbackItem:
    feedback_id: str
    evidence_ids: tuple[str, ...]
    attribution_id: str
    target_components: tuple[str, ...]
    failure_modes: tuple[str, ...]
    successful_behaviors: tuple[str, ...]
    desired_behaviors: tuple[str, ...]
    avoid_behaviors: tuple[str, ...]
    applicable_segment_traits: Mapping[str, Any]
    confidence: float
    rationale: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.feedback_id, "feedback_id")
        _require_nonempty_str(self.attribution_id, "attribution_id")
        for name in (
            "evidence_ids", "target_components", "failure_modes",
            "successful_behaviors", "desired_behaviors", "avoid_behaviors",
        ):
            _require_str_tuple(getattr(self, name), name)
        if not self.evidence_ids:
            raise ValueError("FeedbackItem.evidence_ids must not be empty")
        allowed = {"prompt_bank", "router", "scaffold", "insufficient_evidence"}
        if not self.target_components or any(
                target not in allowed for target in self.target_components):
            raise ValueError("FeedbackItem.target_components contains an invalid target")
        if len(self.target_components) != len(set(self.target_components)):
            raise ValueError("FeedbackItem.target_components contains duplicates")
        _freeze_mapping(self, "applicable_segment_traits")
        _require_confidence(self.confidence, "FeedbackItem.confidence")
        if not isinstance(self.rationale, str):
            raise TypeError("FeedbackItem.rationale must be a string")


@dataclass(frozen=True)
class FeedbackBatch:
    feedback_policy: str
    feedback_policy_version: str
    input_evidence_ids: tuple[str, ...]
    attributions: tuple[FailureAttribution, ...]
    items: tuple[FeedbackItem, ...]
    raw_response_artifact: str | None
    parse_errors: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty_str(self.feedback_policy, "feedback_policy")
        _require_nonempty_str(
            self.feedback_policy_version, "feedback_policy_version")
        _require_str_tuple(self.input_evidence_ids, "input_evidence_ids")
        _require_str_tuple(self.parse_errors, "parse_errors")
        if len(self.input_evidence_ids) != len(set(self.input_evidence_ids)):
            raise ValueError("FeedbackBatch.input_evidence_ids contains duplicates")
        if not isinstance(self.attributions, tuple) or any(
                not isinstance(item, FailureAttribution) for item in self.attributions):
            raise TypeError("FeedbackBatch.attributions must contain FailureAttribution")
        if not isinstance(self.items, tuple) or any(
                not isinstance(item, FeedbackItem) for item in self.items):
            raise TypeError("FeedbackBatch.items must contain FeedbackItem")
        if self.raw_response_artifact is not None and not isinstance(
                self.raw_response_artifact, str):
            raise TypeError("FeedbackBatch.raw_response_artifact must be str or None")


@dataclass(frozen=True)
class MetaKnowledgeItem:
    knowledge_id: str
    knowledge_type: Literal[
        "successful_behavior", "failure_pattern", "routing_pattern",
        "composition_pattern", "rejected_update", "accepted_update", "conflict",
    ]
    condition: Mapping[str, Any]
    principle: str
    positive_support_ids: tuple[str, ...]
    negative_support_ids: tuple[str, ...]
    distinct_video_ids: tuple[str, ...]
    scope: Literal["local_prompt", "routing", "global_scaffold", "meta_only"]
    confidence: float
    status: Literal["candidate", "confirmed", "rejected", "deprecated"]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_nonempty_str(self.knowledge_id, "knowledge_id")
        _require_nonempty_str(self.principle, "principle")
        if self.knowledge_type not in KNOWLEDGE_TYPES:
            raise ValueError("MetaKnowledgeItem.knowledge_type is invalid")
        if self.scope not in KNOWLEDGE_SCOPES:
            raise ValueError("MetaKnowledgeItem.scope is invalid")
        if self.status not in KNOWLEDGE_STATUSES:
            raise ValueError("MetaKnowledgeItem.status is invalid")
        for name in (
            "positive_support_ids", "negative_support_ids", "distinct_video_ids",
        ):
            value = getattr(self, name)
            _require_str_tuple(value, name)
            if len(value) != len(set(value)):
                raise ValueError(f"MetaKnowledgeItem.{name} contains duplicates")
        _freeze_mapping(self, "condition")
        _freeze_mapping(self, "provenance")
        _require_confidence(self.confidence, "MetaKnowledgeItem.confidence")


@dataclass(frozen=True)
class PromptBankOperation:
    operation_id: str
    operation_type: Literal[
        "add_entry", "revise_entry", "merge_entries", "retire_entry", "no_op",
    ]
    target_ids: tuple[str, ...]
    payload: Mapping[str, Any]
    source_feedback_ids: tuple[str, ...]
    confidence: float

    def __post_init__(self) -> None:
        _require_nonempty_str(self.operation_id, "operation_id")
        if self.operation_type not in PROMPT_BANK_OPERATION_TYPES:
            raise ValueError("PromptBankOperation.operation_type is invalid")
        _require_str_tuple(self.target_ids, "target_ids")
        _require_str_tuple(self.source_feedback_ids, "source_feedback_ids")
        if not self.source_feedback_ids:
            raise ValueError("PromptBankOperation.source_feedback_ids must not be empty")
        if self.operation_type == "add_entry" and self.target_ids:
            raise ValueError("add_entry must not specify existing target_ids")
        if self.operation_type == "revise_entry" and len(self.target_ids) != 1:
            raise ValueError("revise_entry requires exactly one target_id")
        if self.operation_type == "merge_entries" and len(self.target_ids) < 2:
            raise ValueError("merge_entries requires at least two target_ids")
        if self.operation_type == "retire_entry" and not self.target_ids:
            raise ValueError("retire_entry requires target_ids")
        _freeze_mapping(self, "payload")
        _require_confidence(self.confidence, "PromptBankOperation.confidence")


@dataclass(frozen=True)
class PromptBankUpdateProposal:
    proposal_id: str
    input_bank_version: str
    operations: tuple[PromptBankOperation, ...]
    validation_errors: tuple[str, ...]
    is_valid: bool
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_nonempty_str(self.proposal_id, "proposal_id")
        validate_component_version(
            "bank", self.input_bank_version,
            field_name="PromptBankUpdateProposal.input_bank_version")
        if not isinstance(self.operations, tuple) or any(
                not isinstance(item, PromptBankOperation) for item in self.operations):
            raise TypeError("operations must contain PromptBankOperation records")
        _require_str_tuple(self.validation_errors, "validation_errors")
        if self.is_valid and self.validation_errors:
            raise ValueError("valid bank proposal cannot contain validation_errors")
        if not self.is_valid and not self.validation_errors:
            raise ValueError("invalid bank proposal requires validation_errors")
        _freeze_mapping(self, "provenance")


@dataclass(frozen=True)
class RouterUpdateProposal:
    proposal_id: str
    input_router_version: str
    operations: tuple[Mapping[str, Any], ...]
    source_feedback_ids: tuple[str, ...]
    validation_errors: tuple[str, ...]
    is_valid: bool
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_nonempty_str(self.proposal_id, "proposal_id")
        validate_component_version(
            "router", self.input_router_version,
            field_name="RouterUpdateProposal.input_router_version")
        if not isinstance(self.operations, tuple) or any(
                not isinstance(item, Mapping) for item in self.operations):
            raise TypeError("RouterUpdateProposal.operations must contain mappings")
        object.__setattr__(self, "operations", _freeze_json(
            self.operations, "RouterUpdateProposal.operations"))
        _require_str_tuple(self.source_feedback_ids, "source_feedback_ids")
        _require_str_tuple(self.validation_errors, "validation_errors")
        if self.is_valid and self.validation_errors:
            raise ValueError("valid router proposal cannot contain validation_errors")
        if not self.is_valid and not self.validation_errors:
            raise ValueError("invalid router proposal requires validation_errors")
        _freeze_mapping(self, "provenance")


@dataclass(frozen=True)
class ScaffoldUpdateProposal:
    proposal_id: str
    input_scaffold_version: str
    candidate_policy: ScaffoldPolicySnapshot | None
    source_feedback_ids: tuple[str, ...]
    validation_errors: tuple[str, ...]
    is_valid: bool
    skipped_reason: Literal[
        "disabled_by_config", "no_scaffold_attribution",
        "insufficient_support", "none",
    ]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_nonempty_str(self.proposal_id, "proposal_id")
        validate_component_version(
            "scaffold", self.input_scaffold_version,
            field_name="ScaffoldUpdateProposal.input_scaffold_version")
        if self.candidate_policy is not None and not isinstance(
                self.candidate_policy, ScaffoldPolicySnapshot):
            raise TypeError("candidate_policy must be ScaffoldPolicySnapshot or None")
        _require_str_tuple(self.source_feedback_ids, "source_feedback_ids")
        _require_str_tuple(self.validation_errors, "validation_errors")
        if self.skipped_reason not in SCAFFOLD_SKIP_REASONS:
            raise ValueError("ScaffoldUpdateProposal.skipped_reason is invalid")
        if self.skipped_reason != "none" and self.candidate_policy is not None:
            raise ValueError("skipped scaffold proposal cannot carry candidate_policy")
        if self.is_valid and self.validation_errors:
            raise ValueError("valid scaffold proposal cannot contain validation_errors")
        if not self.is_valid and not self.validation_errors:
            raise ValueError("invalid scaffold proposal requires validation_errors")
        _freeze_mapping(self, "provenance")
