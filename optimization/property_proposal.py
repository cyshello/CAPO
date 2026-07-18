"""Typed zero-or-multiple property proposal boundary for one source video."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from surrogate_rollout.prompt_routing.schemas import PromptBankSnapshot


ALLOWED_PROPERTY_MODALITIES = frozenset({
    "frames", "transcript", "caption_history",
})


def _normalized_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = " ".join(value.split()).strip()
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized


@dataclass(frozen=True)
class PropertyApplicability:
    """Observable input conditions under which a caption property applies."""

    when: str
    positive_cues: tuple[str, ...]
    negative_cues: tuple[str, ...]
    required_modalities: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "when", _normalized_string(self.when, field="applicability.when"))
        for name in ("positive_cues", "negative_cues"):
            values = tuple(
                _normalized_string(item, field=f"applicability.{name}")
                for item in getattr(self, name))
            if name == "positive_cues" and not values:
                raise ValueError("applicability.positive_cues must be non-empty")
            if len(values) != len({item.casefold() for item in values}):
                raise ValueError(f"applicability.{name} must not contain duplicates")
            object.__setattr__(self, name, values)
        modalities = tuple(
            _normalized_string(item, field="applicability.required_modalities")
            for item in self.required_modalities)
        if not modalities:
            raise ValueError("applicability.required_modalities must be non-empty")
        if len(modalities) != len({item.casefold() for item in modalities}):
            raise ValueError(
                "applicability.required_modalities must not contain duplicates")
        unsupported = set(modalities) - ALLOWED_PROPERTY_MODALITIES
        if unsupported:
            raise ValueError(
                f"unsupported applicability modalities: {sorted(unsupported)}")
        object.__setattr__(self, "required_modalities", modalities)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PropertyApplicability":
        fields = {"when", "positive_cues", "negative_cues", "required_modalities"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError(
                f"applicability must contain exactly {sorted(fields)}")
        for name in ("positive_cues", "negative_cues", "required_modalities"):
            if not isinstance(value[name], (list, tuple)):
                raise ValueError(f"applicability.{name} must be an array")
        return cls(
            when=value["when"],
            positive_cues=tuple(value["positive_cues"]),
            negative_cues=tuple(value["negative_cues"]),
            required_modalities=tuple(value["required_modalities"]),
        )


def legacy_property_applicability() -> PropertyApplicability:
    """Explicit compatibility value for artifacts predating applicability."""
    return PropertyApplicability(
        when="Apply when available captioning input supports this legacy instruction.",
        positive_cues=("Available captioning input supports the instruction.",),
        negative_cues=(),
        required_modalities=("frames",),
    )


@dataclass(frozen=True, init=False)
class CandidatePropertyProposal:
    candidate_property_id: str
    suggested_property_id: str
    property_text: str
    applicability: PropertyApplicability
    source_video_id: str
    source_question_ids: tuple[str, ...]
    source_qa_baseline_correct: bool | None
    coverage_hints: tuple[str, ...]
    failure_analysis: str
    proposer_policy_version: str

    def __init__(
        self,
        *,
        candidate_property_id: str | None = None,
        suggested_property_id: str | None = None,
        property_text: str | None = None,
        applicability: PropertyApplicability | Mapping[str, Any] | None = None,
        source_video_id: str,
        source_question_ids: tuple[str, ...] | None = None,
        source_qa_baseline_correct: bool | None = None,
        motivating_failure_types: tuple[str, ...] | None = None,
        coverage_hints: tuple[str, ...] | None = None,
        possible_coverage_by_existing_property_ids: (
            tuple[str, ...] | None) = None,
        covered_by_existing_property_ids: tuple[str, ...] | None = None,
        failure_analysis: str | None = None,
        proposal_rationale: str | None = None,
        proposer_policy_version: str,
        # Checkpoint 2 compatibility aliases.
        property_id: str | None = None,
        source_qa_ids: tuple[str, ...] | None = None,
        instruction: str | None = None,
        failure_evidence: str | None = None,
        coverage_assessment: str | None = None,
    ) -> None:
        del coverage_assessment
        supplied_hints = tuple(coverage_hints or ())
        possible_hints = tuple(possible_coverage_by_existing_property_ids or ())
        legacy_hints = tuple(covered_by_existing_property_ids or ())
        nonempty_hint_values = tuple(
            value for value in (supplied_hints, possible_hints, legacy_hints)
            if value)
        if nonempty_hint_values and any(
                value != nonempty_hint_values[0]
                for value in nonempty_hint_values[1:]):
            raise ValueError("proposal coverage hint aliases disagree")
        normalized_hints = nonempty_hint_values[0] if nonempty_hint_values else ()
        values = {
            "candidate_property_id": candidate_property_id or property_id or "",
            "suggested_property_id": (
                suggested_property_id or candidate_property_id or property_id or ""),
            "property_text": property_text or instruction or "",
            "applicability": (
                applicability if isinstance(applicability, PropertyApplicability)
                else PropertyApplicability.from_mapping(applicability)
                if applicability is not None else legacy_property_applicability()),
            "source_video_id": source_video_id,
            "source_question_ids": tuple(source_question_ids or source_qa_ids or ()),
            "source_qa_baseline_correct": source_qa_baseline_correct,
            "coverage_hints": normalized_hints,
            "failure_analysis": (
                failure_analysis or proposal_rationale or failure_evidence
                or "; ".join(motivating_failure_types or ())),
            "proposer_policy_version": proposer_policy_version,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        for name in ("candidate_property_id", "suggested_property_id",
                     "property_text", "source_video_id", "failure_analysis",
                     "proposer_policy_version"):
            if not getattr(self, name):
                raise ValueError(f"CandidatePropertyProposal.{name} must be non-empty")
        if not self.source_question_ids or len(self.source_question_ids) != len(
                set(self.source_question_ids)):
            raise ValueError("proposal source question IDs must be non-empty and unique")
        if len(self.coverage_hints) != len(set(self.coverage_hints)):
            raise ValueError("proposal coverage hints must be unique")
        if self.source_qa_baseline_correct is not None and not isinstance(
                self.source_qa_baseline_correct, bool):
            raise ValueError(
                "proposal source QA baseline correctness must be bool or null")

    @property
    def property_id(self) -> str:
        return self.candidate_property_id

    @property
    def instruction(self) -> str:
        return self.property_text

    @property
    def source_qa_ids(self) -> tuple[str, ...]:
        return self.source_question_ids

    @property
    def coverage_assessment(self) -> str:
        return "deferred_to_intervention"

    @property
    def possible_coverage_by_existing_property_ids(self) -> tuple[str, ...]:
        return self.coverage_hints

    @property
    def covered_by_existing_property_ids(self) -> tuple[str, ...]:
        """Legacy read alias; these IDs are non-binding proposal hints."""
        return self.coverage_hints


def candidate_property_proposal_from_json(
    value: Mapping[str, Any],
) -> CandidatePropertyProposal:
    """Load canonical proposals and explicitly migrate legacy artifacts."""
    applicability = value.get("applicability")
    return CandidatePropertyProposal(
        candidate_property_id=str(
            value.get("candidate_property_id") or value.get("property_id") or ""),
        suggested_property_id=str(
            value.get("suggested_property_id")
            or value.get("candidate_property_id")
            or value.get("property_id") or ""),
        property_text=str(value.get("property_text") or value.get("instruction") or ""),
        applicability=(
            PropertyApplicability.from_mapping(applicability)
            if applicability is not None else legacy_property_applicability()),
        source_video_id=str(value.get("source_video_id") or ""),
        source_question_ids=tuple(
            value.get("source_question_ids") or value.get("source_qa_ids") or ()),
        source_qa_baseline_correct=value.get("source_qa_baseline_correct"),
        coverage_hints=tuple(
            value.get("coverage_hints")
            or value.get("possible_coverage_by_existing_property_ids")
            or value.get("covered_by_existing_property_ids") or ()),
        failure_analysis=str(
            value.get("failure_analysis") or value.get("proposal_rationale")
            or value.get("failure_evidence")
            or "; ".join(value.get("motivating_failure_types") or ())),
        proposer_policy_version=str(value.get("proposer_policy_version") or "legacy"),
    )


@dataclass(frozen=True)
class VideoPropertyProposalRecord:
    video_id: str
    baseline_run_id: str
    baseline_qa_ids: tuple[str, ...]
    proposals: tuple[CandidatePropertyProposal, ...]
    proposer_policy_version: str

    def __post_init__(self) -> None:
        if not self.video_id or not self.baseline_run_id or \
                not self.proposer_policy_version:
            raise ValueError("video proposal record identifiers must be non-empty")
        if len(self.baseline_qa_ids) != 3 or len(set(self.baseline_qa_ids)) != 3:
            raise ValueError("video proposal record requires all three baseline QAs")
        if len({item.property_id for item in self.proposals}) != len(self.proposals):
            raise ValueError("property IDs must be unique within a source video")
        for item in self.proposals:
            if item.source_video_id != self.video_id:
                raise ValueError("proposal source video does not match record")
            if not set(item.source_qa_ids) <= set(self.baseline_qa_ids):
                raise ValueError("proposal lineage references a non-baseline QA")


@dataclass(frozen=True)
class VideoProposalContext:
    video_id: str
    baseline_run_id: str
    baseline_qa_results: tuple[Mapping[str, Any], ...]
    captions: Mapping[str, Any]
    frame_references: Mapping[str, tuple[str, ...]]
    frozen_histories: tuple[Mapping[str, Any], ...]
    prompt_bank: PromptBankSnapshot
    proposal_artifact_dir: str | None = None


class PropertyProposalPolicy(Protocol):
    policy_version: str

    def propose(
        self, context: VideoProposalContext,
    ) -> Sequence[CandidatePropertyProposal]:
        ...


class NoOpPropertyProposalPolicy:
    """Checkpoint 2 default: records a valid zero-proposal result."""

    policy_version = "checkpoint2_noop_property_proposer_v1"

    def propose(
        self, context: VideoProposalContext,
    ) -> tuple[CandidatePropertyProposal, ...]:
        return ()
