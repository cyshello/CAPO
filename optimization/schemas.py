"""Frozen Phase 4 optimization records.

The legacy property-codebook records and the prompt-delta records intentionally
coexist in this module.  Prompt deltas are episode-local interventions; none of
their records are property-bank entries or router inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
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
COMPACT_FEEDBACK_EFFECTS = (
    "wrong_to_correct", "correct_to_wrong", "correct_to_correct",
    "wrong_to_wrong", "mixed", "no_effect",
)
COMPACT_FEEDBACK_ATTRIBUTIONS = (
    "direct", "indirect", "unresolved", "no_op", "invalid",
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


def _freeze_opaque_json(record: Any, name: str) -> None:
    """Preserve an otherwise uninterpreted JSON value on a frozen record."""
    object.__setattr__(record, name, _freeze_json(
        getattr(record, name), f"{type(record).__name__}.{name}"))


def _require_optional_str(value: str | None, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be str or None")


def _require_compact_memory_text(value: str, name: str) -> None:
    _require_nonempty_str(value, name)
    if len(re.findall(r"\b[\w'-]+\b", value, flags=re.UNICODE)) > 30:
        raise ValueError(f"{name} exceeds 30 words")
    if "\n" in value or re.search(r"(?:```|^\s*[-*#>]\s)", value):
        raise ValueError(f"{name} must be one plain-text sentence")
    if len(re.findall(r"[.!?](?=\s|$)", value)) > 1:
        raise ValueError(f"{name} must contain at most one sentence")
    if re.search(r"\b(?:caused|led\s+to|corrected|resulted\s+in)\b",
                 value, flags=re.IGNORECASE):
        raise ValueError(f"{name} contains prohibited causal wording")


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
    mixed_view_identity: Any | None = None

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
        if self.mixed_view_identity is not None:
            _freeze_opaque_json(self, "mixed_view_identity")


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
        if self.transition_type is not None and \
                self.transition_type not in QA_TRANSITION_TYPES:
            raise ValueError(
                "EpisodeFeedbackEvidence.transition_type is invalid")
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
    compact_memory_text: str | None = None

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
        memory = self.compact_memory_text
        if memory is not None:
            if not isinstance(memory, str):
                raise TypeError(
                    "EpisodeFeedback.compact_memory_text must be a string or None")
            memory = memory.strip() or None
            object.__setattr__(self, "compact_memory_text", memory)


@dataclass(frozen=True)
class EpisodeFeedbackMemoryRecord:
    """One provider-authored episode memory with separate provenance."""

    memory_id: str
    parent_meta_prompt_id: str
    iteration_id: str
    episode_id: str
    candidate_id: str
    feedback_id: str
    memory_text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
                "memory_id", "parent_meta_prompt_id", "iteration_id",
                "episode_id", "candidate_id", "feedback_id"):
            _require_nonempty_str(
                getattr(self, name), f"EpisodeFeedbackMemoryRecord.{name}")
        _require_nonempty_str(
            self.memory_text, "EpisodeFeedbackMemoryRecord.memory_text")
        object.__setattr__(self, "memory_text", self.memory_text.strip())
        _freeze_mapping(self, "metadata")


@dataclass(frozen=True)
class CompactFeedbackMemory:
    """Short updater-facing semantic memory; provenance lives separately."""

    runtime_condition: str
    description_change: str
    effect: Literal[
        "wrong_to_correct", "correct_to_wrong", "correct_to_correct",
        "wrong_to_wrong", "mixed", "no_effect"]
    attribution: Literal[
        "direct", "indirect", "unresolved", "no_op", "invalid"]

    def __post_init__(self) -> None:
        _require_compact_memory_text(
            self.runtime_condition, "CompactFeedbackMemory.runtime_condition")
        _require_compact_memory_text(
            self.description_change,
            "CompactFeedbackMemory.description_change")
        if self.effect not in COMPACT_FEEDBACK_EFFECTS:
            raise ValueError("CompactFeedbackMemory.effect is invalid")
        if self.attribution not in COMPACT_FEEDBACK_ATTRIBUTIONS:
            raise ValueError("CompactFeedbackMemory.attribution is invalid")


@dataclass(frozen=True)
class CompactFeedbackProvenance:
    parent_meta_prompt_id: str
    iteration_id: str
    episode_id: str
    video_id: str
    qa_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    feedback_id: str

    def __post_init__(self) -> None:
        for name in (
                "parent_meta_prompt_id", "iteration_id", "episode_id",
                "video_id", "feedback_id"):
            _require_nonempty_str(
                getattr(self, name), f"CompactFeedbackProvenance.{name}")
        _require_str_tuple(self.qa_ids, "CompactFeedbackProvenance.qa_ids")
        _require_str_tuple(
            self.segment_ids, "CompactFeedbackProvenance.segment_ids")
        if not self.qa_ids:
            raise ValueError("CompactFeedbackProvenance.qa_ids is required")


@dataclass(frozen=True)
class CompactFeedbackMemoryRecord:
    memory_id: str
    memory: CompactFeedbackMemory
    provenance: CompactFeedbackProvenance
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.memory_id, "CompactFeedbackMemoryRecord.memory_id")
        if not isinstance(self.memory, CompactFeedbackMemory):
            raise TypeError("CompactFeedbackMemoryRecord.memory is invalid")
        if not isinstance(self.provenance, CompactFeedbackProvenance):
            raise TypeError("CompactFeedbackMemoryRecord.provenance is invalid")
        _freeze_mapping(self, "metadata")


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
        intervention_run_ref=data["intervention_run_ref"],
        mixed_view_identity=data.get("mixed_view_identity"))


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
        confidence=data["confidence"],
        compact_memory_text=data.get("compact_memory_text"))


def episode_feedback_memory_record_from_json(
    data: Mapping[str, Any],
) -> EpisodeFeedbackMemoryRecord:
    return EpisodeFeedbackMemoryRecord(
        memory_id=data["memory_id"],
        parent_meta_prompt_id=data["parent_meta_prompt_id"],
        iteration_id=data["iteration_id"],
        episode_id=data["episode_id"],
        candidate_id=data["candidate_id"],
        feedback_id=data["feedback_id"],
        memory_text=data["memory_text"],
        metadata=data.get("metadata", {}),
    )


def compact_feedback_memory_record_from_json(
    data: Mapping[str, Any],
) -> CompactFeedbackMemoryRecord:
    memory = data["memory"]
    provenance = data["provenance"]
    return CompactFeedbackMemoryRecord(
        memory_id=data["memory_id"],
        memory=CompactFeedbackMemory(
            runtime_condition=memory["runtime_condition"],
            description_change=memory["description_change"],
            effect=memory["effect"], attribution=memory["attribution"]),
        provenance=CompactFeedbackProvenance(
            parent_meta_prompt_id=provenance["parent_meta_prompt_id"],
            iteration_id=provenance["iteration_id"],
            episode_id=provenance["episode_id"],
            video_id=provenance["video_id"],
            qa_ids=tuple(provenance["qa_ids"]),
            segment_ids=tuple(provenance["segment_ids"]),
            feedback_id=provenance["feedback_id"]),
        metadata=data.get("metadata", {}),
    )


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
    """Validate only the episode identity; do not judge evidence semantics."""
    if not isinstance(feedback, EpisodeFeedback):
        raise TypeError("feedback must be an EpisodeFeedback")
    if not isinstance(episode, InterventionEpisode):
        raise TypeError("episode must be an InterventionEpisode")
    if feedback.episode_id != episode.episode_id:
        raise ValueError(
            "EpisodeFeedback.episode_id does not match InterventionEpisode")
