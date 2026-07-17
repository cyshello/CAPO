"""Typed zero-or-multiple property proposal boundary for one source video."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from surrogate_rollout.prompt_routing.schemas import PromptBankSnapshot


@dataclass(frozen=True, init=False)
class CandidatePropertyProposal:
    candidate_property_id: str
    property_text: str
    source_video_id: str
    source_question_ids: tuple[str, ...]
    motivating_failure_types: tuple[str, ...]
    coverage_hints: tuple[str, ...]
    proposal_rationale: str
    proposer_policy_version: str

    def __init__(
        self,
        *,
        candidate_property_id: str | None = None,
        property_text: str | None = None,
        source_video_id: str,
        source_question_ids: tuple[str, ...] | None = None,
        motivating_failure_types: tuple[str, ...] | None = None,
        coverage_hints: tuple[str, ...] | None = None,
        possible_coverage_by_existing_property_ids: (
            tuple[str, ...] | None) = None,
        covered_by_existing_property_ids: tuple[str, ...] | None = None,
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
            "property_text": property_text or instruction or "",
            "source_video_id": source_video_id,
            "source_question_ids": tuple(source_question_ids or source_qa_ids or ()),
            "motivating_failure_types": tuple(
                motivating_failure_types
                or ((failure_evidence,) if failure_evidence else ())),
            "coverage_hints": normalized_hints,
            "proposal_rationale": proposal_rationale or failure_evidence or "",
            "proposer_policy_version": proposer_policy_version,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        for name in ("candidate_property_id", "property_text", "source_video_id",
                     "proposal_rationale", "proposer_policy_version"):
            if not getattr(self, name):
                raise ValueError(f"CandidatePropertyProposal.{name} must be non-empty")
        if not self.source_question_ids or len(self.source_question_ids) != len(
                set(self.source_question_ids)):
            raise ValueError("proposal source question IDs must be non-empty and unique")
        if not self.motivating_failure_types:
            raise ValueError("proposal motivating failure types must be non-empty")
        if len(self.motivating_failure_types) != len(set(
                self.motivating_failure_types)) or any(
                not isinstance(item, str) or not item
                for item in self.motivating_failure_types):
            raise ValueError("motivating failure types must be unique strings")
        if len(self.coverage_hints) != len(set(self.coverage_hints)):
            raise ValueError("proposal coverage hints must be unique")

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
    def failure_evidence(self) -> str:
        return self.proposal_rationale

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
