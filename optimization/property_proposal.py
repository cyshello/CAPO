"""Typed zero-or-multiple property proposal boundary for one source video."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from surrogate_rollout.prompt_routing.schemas import PromptBankSnapshot


@dataclass(frozen=True)
class CandidatePropertyProposal:
    property_id: str
    source_video_id: str
    source_qa_ids: tuple[str, ...]
    instruction: str
    failure_evidence: str
    coverage_assessment: str
    proposer_policy_version: str

    def __post_init__(self) -> None:
        for name in ("property_id", "source_video_id", "instruction",
                     "failure_evidence", "coverage_assessment",
                     "proposer_policy_version"):
            if not getattr(self, name):
                raise ValueError(f"CandidatePropertyProposal.{name} must be non-empty")
        if not self.source_qa_ids or len(self.source_qa_ids) != len(set(self.source_qa_ids)):
            raise ValueError("proposal source QA IDs must be non-empty and unique")


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
