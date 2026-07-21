"""Frozen Phase 4 optimization records.

The legacy property-codebook records and the prompt-delta records intentionally
coexist in this module.  Prompt deltas are episode-local interventions; none of
their records are property-bank entries or router inputs.
"""

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
REVIEW_DECISIONS = ("accept_for_validation", "revise", "reject", "defer")
VALIDATION_DECISIONS = ("accept", "reject", "defer", "evaluation_failed")
UPDATE_COMPONENTS = ("prompt_bank", "router", "scaffold")
ITERATION_STATUSES = (
    "dry_run", "partially_committed", "committed", "all_rejected",
    "evaluation_failed",
)
META_PROMPT_STATUSES = ("parent", "provisional", "confirmed", "rejected")
EPISODE_EVIDENCE_TYPES = (
    "caption_change", "trajectory", "qa_transition", "mixed",
)
QA_TRANSITION_TYPES = (
    "correct_to_correct", "wrong_to_wrong", "wrong_to_correct",
    "correct_to_wrong",
)
META_PROMPT_UPDATE_DECISIONS = ("update", "no_update")


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


def _freeze_opaque_json(record: Any, name: str) -> None:
    """Preserve an otherwise uninterpreted JSON value on a frozen record."""
    object.__setattr__(record, name, _freeze_json(
        getattr(record, name), f"{type(record).__name__}.{name}"))


def _require_optional_str(value: str | None, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be str or None")


@dataclass(frozen=True)
class MetaPromptVersion:
    """One immutable lineage node for the prompt-delta meta-prompt path."""

    meta_prompt_id: str
    parent_meta_prompt_id: str | None
    text: str
    created_at: str
    status: Literal["parent", "provisional", "confirmed", "rejected"]

    def __post_init__(self) -> None:
        _require_nonempty_str(self.meta_prompt_id, "MetaPromptVersion.meta_prompt_id")
        _require_optional_str(
            self.parent_meta_prompt_id,
            "MetaPromptVersion.parent_meta_prompt_id")
        if not isinstance(self.text, str):
            raise TypeError("MetaPromptVersion.text must be a string")
        if not isinstance(self.created_at, str):
            raise TypeError("MetaPromptVersion.created_at must be a string")
        if self.status not in META_PROMPT_STATUSES:
            raise ValueError("MetaPromptVersion.status is invalid")


@dataclass(frozen=True)
class PromptDelta:
    """An ephemeral correction for one prompt-intervention episode."""

    delta_id: str
    instruction: str
    source_qa_ids: tuple[str, ...]
    proposer_diagnosis: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.delta_id, "PromptDelta.delta_id")
        if not isinstance(self.instruction, str):
            raise TypeError("PromptDelta.instruction must be a string")
        _require_str_tuple(self.source_qa_ids, "PromptDelta.source_qa_ids")
        if not isinstance(self.proposer_diagnosis, str):
            raise TypeError("PromptDelta.proposer_diagnosis must be a string")


@dataclass(frozen=True)
class InterventionClipRecord:
    """Before/after caption evidence for one clip in an intervention.

    The repository has no dedicated type with exactly the semantics of a clip
    time range or a frozen history snapshot.  Both fields therefore retain an
    opaque, recursively frozen JSON value.  This preserves existing serialized
    payloads without inventing a new interval or history contract.
    """

    segment_id: str
    time_range: Any
    history_snapshot: Any
    base_prompt: str
    prompt_delta: PromptDelta
    baseline_caption: str
    intervention_caption: str

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.segment_id, "InterventionClipRecord.segment_id")
        _freeze_opaque_json(self, "time_range")
        _freeze_opaque_json(self, "history_snapshot")
        if not isinstance(self.base_prompt, str):
            raise TypeError("InterventionClipRecord.base_prompt must be a string")
        if not isinstance(self.prompt_delta, PromptDelta):
            raise TypeError(
                "InterventionClipRecord.prompt_delta must be a PromptDelta")
        if not isinstance(self.baseline_caption, str) or not isinstance(
                self.intervention_caption, str):
            raise TypeError("InterventionClipRecord captions must be strings")


@dataclass(frozen=True)
class QAInterventionOutcome:
    """Set-level QA result; trajectory references are explicitly nullable."""

    qa_id: str
    is_source_qa: bool
    baseline_answer: str | None
    intervention_answer: str | None
    baseline_correct: bool | None
    intervention_correct: bool | None
    baseline_trajectory_ref: str | None
    intervention_trajectory_ref: str | None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.qa_id, "QAInterventionOutcome.qa_id")
        if not isinstance(self.is_source_qa, bool):
            raise TypeError("QAInterventionOutcome.is_source_qa must be bool")
        for name in (
            "baseline_answer", "intervention_answer",
            "baseline_trajectory_ref", "intervention_trajectory_ref",
        ):
            _require_optional_str(
                getattr(self, name), f"QAInterventionOutcome.{name}")
        for name in ("baseline_correct", "intervention_correct"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(
                    f"QAInterventionOutcome.{name} must be bool or None")


@dataclass(frozen=True)
class InterventionEpisode:
    """One delta applied to one video-level intervention scope."""

    episode_id: str
    video_id: str
    parent_meta_prompt_id: str
    prompt_delta: PromptDelta
    clips: tuple[InterventionClipRecord, ...]
    qa_outcomes: tuple[QAInterventionOutcome, ...]
    baseline_run_ref: str
    intervention_run_ref: str

    def __post_init__(self) -> None:
        for name in (
            "episode_id", "video_id", "parent_meta_prompt_id",
            "baseline_run_ref", "intervention_run_ref",
        ):
            _require_nonempty_str(
                getattr(self, name), f"InterventionEpisode.{name}")
        if not isinstance(self.prompt_delta, PromptDelta):
            raise TypeError("InterventionEpisode.prompt_delta must be a PromptDelta")
        if not isinstance(self.clips, tuple) or any(
                not isinstance(item, InterventionClipRecord)
                for item in self.clips):
            raise TypeError(
                "InterventionEpisode.clips must contain InterventionClipRecord")
        if not isinstance(self.qa_outcomes, tuple) or any(
                not isinstance(item, QAInterventionOutcome)
                for item in self.qa_outcomes):
            raise TypeError(
                "InterventionEpisode.qa_outcomes must contain QAInterventionOutcome")


@dataclass(frozen=True)
class EpisodeFeedbackEvidence:
    """One evidence-linked observation or counterevidence item."""

    statement: str
    supporting_segment_ids: tuple[str, ...]
    supporting_qa_ids: tuple[str, ...]
    evidence_type: Literal[
        "caption_change", "trajectory", "qa_transition", "mixed",
    ]
    transition_type: Literal[
        "correct_to_correct", "wrong_to_wrong", "wrong_to_correct",
        "correct_to_wrong",
    ] | None
    confidence: Any

    def __post_init__(self) -> None:
        if not isinstance(self.statement, str):
            raise TypeError("EpisodeFeedbackEvidence.statement must be a string")
        _require_str_tuple(
            self.supporting_segment_ids,
            "EpisodeFeedbackEvidence.supporting_segment_ids")
        _require_str_tuple(
            self.supporting_qa_ids,
            "EpisodeFeedbackEvidence.supporting_qa_ids")
        if self.evidence_type not in EPISODE_EVIDENCE_TYPES:
            raise ValueError("EpisodeFeedbackEvidence.evidence_type is invalid")
        if self.evidence_type == "qa_transition":
            if self.transition_type not in QA_TRANSITION_TYPES:
                raise ValueError(
                    "EpisodeFeedbackEvidence.transition_type must be a "
                    "supported QA transition for qa_transition evidence")
        elif self.transition_type is not None:
            raise ValueError(
                "EpisodeFeedbackEvidence.transition_type must be None for "
                "non-qa_transition evidence")
        # The specification deliberately leaves the confidence scale undefined.
        # Preserve any JSON value without assigning a range or threshold.
        _freeze_opaque_json(self, "confidence")


@dataclass(frozen=True)
class EpisodeFeedback:
    feedback_id: str
    episode_id: str
    outcome_summary: str
    observations: tuple[EpisodeFeedbackEvidence, ...]
    counterevidence: tuple[EpisodeFeedbackEvidence, ...]
    generator_diagnosis: str
    recommended_strategy_change: str
    confidence: Any

    def __post_init__(self) -> None:
        _require_nonempty_str(self.feedback_id, "EpisodeFeedback.feedback_id")
        _require_nonempty_str(self.episode_id, "EpisodeFeedback.episode_id")
        for name in (
            "outcome_summary", "generator_diagnosis",
            "recommended_strategy_change",
        ):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"EpisodeFeedback.{name} must be a string")
        for name in ("observations", "counterevidence"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(
                    not isinstance(item, EpisodeFeedbackEvidence)
                    for item in value):
                raise TypeError(
                    f"EpisodeFeedback.{name} must contain "
                    "EpisodeFeedbackEvidence")
        _freeze_opaque_json(self, "confidence")


@dataclass(frozen=True)
class MetaPromptUpdateDecision:
    """Provider-independent E1 decision before persistence or confirmation."""

    decision: Literal["update", "no_update"]
    candidate_meta_prompt: str | None
    change_summary: str
    rationale: str
    supporting_feedback_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.decision not in META_PROMPT_UPDATE_DECISIONS:
            raise ValueError("MetaPromptUpdateDecision.decision is invalid")
        if self.decision == "update":
            _require_nonempty_str(
                self.candidate_meta_prompt,
                "MetaPromptUpdateDecision.candidate_meta_prompt")
        elif self.candidate_meta_prompt is not None:
            raise ValueError(
                "MetaPromptUpdateDecision.candidate_meta_prompt must be None "
                "for no_update")
        for name in ("change_summary", "rationale"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"MetaPromptUpdateDecision.{name} must be a string")
        _require_str_tuple(
            self.supporting_feedback_ids,
            "MetaPromptUpdateDecision.supporting_feedback_ids")


def meta_prompt_version_from_json(data: Mapping[str, Any]) -> MetaPromptVersion:
    return MetaPromptVersion(
        meta_prompt_id=data["meta_prompt_id"],
        parent_meta_prompt_id=data["parent_meta_prompt_id"],
        text=data["text"], created_at=data["created_at"], status=data["status"])


def prompt_delta_from_json(data: Mapping[str, Any]) -> PromptDelta:
    return PromptDelta(
        delta_id=data["delta_id"], instruction=data["instruction"],
        source_qa_ids=tuple(data["source_qa_ids"]),
        proposer_diagnosis=data["proposer_diagnosis"])


def intervention_clip_record_from_json(
    data: Mapping[str, Any],
) -> InterventionClipRecord:
    return InterventionClipRecord(
        segment_id=data["segment_id"], time_range=data["time_range"],
        history_snapshot=data["history_snapshot"],
        base_prompt=data["base_prompt"],
        prompt_delta=prompt_delta_from_json(data["prompt_delta"]),
        baseline_caption=data["baseline_caption"],
        intervention_caption=data["intervention_caption"])


def qa_intervention_outcome_from_json(
    data: Mapping[str, Any],
) -> QAInterventionOutcome:
    return QAInterventionOutcome(
        qa_id=data["qa_id"], is_source_qa=data["is_source_qa"],
        baseline_answer=data["baseline_answer"],
        intervention_answer=data["intervention_answer"],
        baseline_correct=data["baseline_correct"],
        intervention_correct=data["intervention_correct"],
        baseline_trajectory_ref=data["baseline_trajectory_ref"],
        intervention_trajectory_ref=data["intervention_trajectory_ref"])


def intervention_episode_from_json(
    data: Mapping[str, Any],
) -> InterventionEpisode:
    return InterventionEpisode(
        episode_id=data["episode_id"], video_id=data["video_id"],
        parent_meta_prompt_id=data["parent_meta_prompt_id"],
        prompt_delta=prompt_delta_from_json(data["prompt_delta"]),
        clips=tuple(intervention_clip_record_from_json(item)
                    for item in data["clips"]),
        qa_outcomes=tuple(qa_intervention_outcome_from_json(item)
                          for item in data["qa_outcomes"]),
        baseline_run_ref=data["baseline_run_ref"],
        intervention_run_ref=data["intervention_run_ref"])


def episode_feedback_evidence_from_json(
    data: Mapping[str, Any],
) -> EpisodeFeedbackEvidence:
    return EpisodeFeedbackEvidence(
        statement=data["statement"],
        supporting_segment_ids=tuple(data["supporting_segment_ids"]),
        supporting_qa_ids=tuple(data["supporting_qa_ids"]),
        evidence_type=data["evidence_type"],
        transition_type=data["transition_type"],
        confidence=data["confidence"])


def episode_feedback_from_json(data: Mapping[str, Any]) -> EpisodeFeedback:
    return EpisodeFeedback(
        feedback_id=data["feedback_id"], episode_id=data["episode_id"],
        outcome_summary=data["outcome_summary"],
        observations=tuple(episode_feedback_evidence_from_json(item)
                           for item in data["observations"]),
        counterevidence=tuple(episode_feedback_evidence_from_json(item)
                              for item in data["counterevidence"]),
        generator_diagnosis=data["generator_diagnosis"],
        recommended_strategy_change=data["recommended_strategy_change"],
        confidence=data["confidence"])


def meta_prompt_update_decision_from_json(
    data: Mapping[str, Any],
) -> MetaPromptUpdateDecision:
    return MetaPromptUpdateDecision(
        decision=data["decision"],
        candidate_meta_prompt=data["candidate_meta_prompt"],
        change_summary=data["change_summary"],
        rationale=data["rationale"],
        supporting_feedback_ids=tuple(data["supporting_feedback_ids"]),
    )


def validate_meta_prompt_update_decision(
    decision: MetaPromptUpdateDecision,
    feedbacks: tuple[EpisodeFeedback, ...],
) -> None:
    """Validate E1 provenance without assigning an evidence threshold."""
    if not isinstance(decision, MetaPromptUpdateDecision):
        raise TypeError("decision must be a MetaPromptUpdateDecision")
    if not isinstance(feedbacks, tuple) or not feedbacks or any(
            not isinstance(item, EpisodeFeedback) for item in feedbacks):
        raise TypeError("feedbacks must be a non-empty tuple of EpisodeFeedback")
    known = {item.feedback_id for item in feedbacks}
    unknown = set(decision.supporting_feedback_ids) - known
    if unknown:
        raise ValueError(
            "MetaPromptUpdateDecision references unknown feedback IDs: "
            f"{sorted(unknown)}")


def validate_episode_feedback(
    feedback: EpisodeFeedback,
    episode: InterventionEpisode,
) -> None:
    """Validate only cross-record identity references required by Checkpoint A."""
    if feedback.episode_id != episode.episode_id:
        raise ValueError(
            "EpisodeFeedback.episode_id does not match InterventionEpisode")
    segment_ids = {clip.segment_id for clip in episode.clips}
    qa_ids = {outcome.qa_id for outcome in episode.qa_outcomes}
    transition_by_qa_id = {}
    for outcome in episode.qa_outcomes:
        before = outcome.baseline_correct
        after = outcome.intervention_correct
        if not isinstance(before, bool) or not isinstance(after, bool):
            continue
        transition_by_qa_id[outcome.qa_id] = (
            "correct_to_correct" if before and after else
            "correct_to_wrong" if before and not after else
            "wrong_to_correct" if not before and after else
            "wrong_to_wrong")
    for collection_name in ("observations", "counterevidence"):
        for index, item in enumerate(getattr(feedback, collection_name)):
            unknown_segments = set(item.supporting_segment_ids) - segment_ids
            if unknown_segments:
                raise ValueError(
                    f"EpisodeFeedback.{collection_name}[{index}] references "
                    f"unknown segment IDs: {sorted(unknown_segments)}")
            unknown_qas = set(item.supporting_qa_ids) - qa_ids
            if unknown_qas:
                raise ValueError(
                    f"EpisodeFeedback.{collection_name}[{index}] references "
                    f"unknown QA IDs: {sorted(unknown_qas)}")
            if item.evidence_type == "qa_transition":
                mismatched = {
                    qa_id: transition_by_qa_id.get(qa_id)
                    for qa_id in item.supporting_qa_ids
                    if transition_by_qa_id.get(qa_id) != item.transition_type
                }
                if mismatched:
                    raise ValueError(
                        f"EpisodeFeedback.{collection_name}[{index}] "
                        "transition_type conflicts with stored QA outcomes: "
                        f"{mismatched}")


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


@dataclass(frozen=True)
class UpdateReview:
    review_id: str
    component: Literal["prompt_bank", "router", "scaffold"]
    proposal_id: str
    decision: Literal["accept_for_validation", "revise", "reject", "defer"]
    recommended_scope: Literal[
        "local_prompt", "routing", "global_scaffold", "meta_only",
    ]
    conflicts: tuple[str, ...]
    required_confirmations: tuple[str, ...]
    supporting_knowledge_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.review_id, "review_id")
        _require_nonempty_str(self.proposal_id, "proposal_id")
        if self.component not in UPDATE_COMPONENTS:
            raise ValueError("UpdateReview.component is invalid")
        if self.decision not in REVIEW_DECISIONS:
            raise ValueError("UpdateReview.decision is invalid")
        if self.recommended_scope not in KNOWLEDGE_SCOPES:
            raise ValueError("UpdateReview.recommended_scope is invalid")
        for name in (
            "conflicts", "required_confirmations", "supporting_knowledge_ids",
        ):
            _require_str_tuple(getattr(self, name), name)
        if not isinstance(self.rationale, str):
            raise TypeError("UpdateReview.rationale must be a string")


@dataclass(frozen=True)
class ComponentValidationResult:
    validation_id: str
    component: Literal["prompt_bank", "router", "scaffold"]
    incumbent_version: str
    candidate_version: str | None
    evidence_example_ids: tuple[str, ...]
    confirmation_example_ids: tuple[str, ...]
    regression_example_ids: tuple[str, ...]
    incumbent_metrics: Mapping[str, float]
    candidate_metrics: Mapping[str, float]
    regressions: tuple[str, ...]
    decision: Literal["accept", "reject", "defer", "evaluation_failed"]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty_str(self.validation_id, "validation_id")
        _require_nonempty_str(self.incumbent_version, "incumbent_version")
        if self.component not in UPDATE_COMPONENTS:
            raise ValueError("ComponentValidationResult.component is invalid")
        kind = {"prompt_bank": "bank", "router": "router",
                "scaffold": "scaffold"}[self.component]
        validate_component_version(
            kind, self.incumbent_version,
            field_name="ComponentValidationResult.incumbent_version")
        validate_component_version(
            kind, self.candidate_version,
            field_name="ComponentValidationResult.candidate_version",
            allow_none=True)
        if self.decision not in VALIDATION_DECISIONS:
            raise ValueError("ComponentValidationResult.decision is invalid")
        if self.candidate_version is not None and not isinstance(
                self.candidate_version, str):
            raise TypeError("candidate_version must be str or None")
        for name in (
            "evidence_example_ids", "confirmation_example_ids",
            "regression_example_ids", "regressions", "reasons",
        ):
            _require_str_tuple(getattr(self, name), name)
        for name in ("incumbent_metrics", "candidate_metrics"):
            metrics = getattr(self, name)
            if not isinstance(metrics, Mapping) or any(
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                    for value in metrics.values()):
                raise TypeError(f"{name} must map strings to numeric values")
            _freeze_mapping(self, name)


@dataclass(frozen=True)
class NoChangeEvaluationResult:
    status: Literal["no_change"]
    reason: Literal["equivalent_components_and_composed_prompts"]
    component_equivalence: Mapping[str, bool]
    incumbent_composed_prompt_hashes: Mapping[str, tuple[str, ...]]
    candidate_composed_prompt_hashes: Mapping[str, tuple[str, ...]]
    candidate_dvd_executed: bool
    eligible_for_validation: bool
    eligible_for_regression_evidence: bool

    def __post_init__(self) -> None:
        if self.status != "no_change":
            raise ValueError("NoChangeEvaluationResult.status must be no_change")
        if self.reason != "equivalent_components_and_composed_prompts":
            raise ValueError("NoChangeEvaluationResult.reason is invalid")
        expected = {"prompt_bank", "router", "scaffold", "contract"}
        if set(self.component_equivalence) != expected or any(
                not isinstance(value, bool)
                for value in self.component_equivalence.values()):
            raise ValueError(
                "component_equivalence must contain four boolean component flags")
        for name in (
            "incumbent_composed_prompt_hashes",
            "candidate_composed_prompt_hashes",
        ):
            values = getattr(self, name)
            if any(not isinstance(video_id, str)
                   or not isinstance(hashes, tuple)
                   or any(not isinstance(value, str) for value in hashes)
                   for video_id, hashes in values.items()):
                raise TypeError(f"{name} must map video IDs to prompt-hash tuples")
            _freeze_mapping(self, name)
        for name in (
            "candidate_dvd_executed", "eligible_for_validation",
            "eligible_for_regression_evidence",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.candidate_dvd_executed or self.eligible_for_validation or \
                self.eligible_for_regression_evidence:
            raise ValueError("a no-change result cannot be evaluated or reused")
        _freeze_mapping(self, "component_equivalence")


@dataclass(frozen=True)
class OptimizationIterationResult:
    iteration_id: str
    input_bank_version: str
    input_router_version: str
    input_scaffold_version: str
    optimize_prompt_bank: bool
    optimize_router: bool
    optimize_scaffold: bool
    evidence_artifact: str
    feedback_artifact: str
    meta_knowledge_artifact: str
    bank_proposal_artifact: str | None
    router_proposal_artifact: str | None
    scaffold_proposal_artifact: str | None
    review_artifacts: Mapping[str, str]
    validation_artifacts: Mapping[str, str]
    committed_bank_version: str | None
    committed_router_version: str | None
    committed_scaffold_version: str | None
    status: Literal[
        "dry_run", "partially_committed", "committed", "all_rejected",
        "evaluation_failed",
    ]
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "iteration_id", "evidence_artifact", "feedback_artifact",
            "meta_knowledge_artifact",
        ):
            _require_nonempty_str(getattr(self, name), name)
        validate_component_version(
            "bank", self.input_bank_version,
            field_name="OptimizationIterationResult.input_bank_version")
        validate_component_version(
            "router", self.input_router_version,
            field_name="OptimizationIterationResult.input_router_version")
        validate_component_version(
            "scaffold", self.input_scaffold_version,
            field_name="OptimizationIterationResult.input_scaffold_version")
        for name in (
            "optimize_prompt_bank", "optimize_router", "optimize_scaffold",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        for name in (
            "bank_proposal_artifact", "router_proposal_artifact",
            "scaffold_proposal_artifact", "committed_bank_version",
            "committed_router_version", "committed_scaffold_version",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be str or None")
        if self.status not in ITERATION_STATUSES:
            raise ValueError("OptimizationIterationResult.status is invalid")
        _require_str_tuple(self.errors, "errors")
        _freeze_mapping(self, "review_artifacts")
        _freeze_mapping(self, "validation_artifacts")
