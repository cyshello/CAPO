"""Deterministic Stage 4.11 meta-knowledge proposal reviewer."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from surrogate_rollout.optimization.schemas import (
    MetaKnowledgeItem,
    PromptBankUpdateProposal,
    RouterUpdateProposal,
    ScaffoldUpdateProposal,
    UpdateReview,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_text


PROPOSAL_TYPES = (
    PromptBankUpdateProposal, RouterUpdateProposal, ScaffoldUpdateProposal,
)


class UpdateReviewer(Protocol):
    def review(
        self,
        *,
        proposal: Any,
        component: str,
        meta_knowledge: Sequence[MetaKnowledgeItem],
    ) -> UpdateReview:
        ...


def _source_feedback_ids(proposal: Any) -> tuple[str, ...]:
    if isinstance(proposal, PromptBankUpdateProposal):
        return tuple(sorted({value for operation in proposal.operations
                             for value in operation.source_feedback_ids}))
    return tuple(sorted(proposal.source_feedback_ids))


def _knowledge_relevant(item: MetaKnowledgeItem, component: str) -> bool:
    scope = {
        "prompt_bank": "local_prompt",
        "router": "routing",
        "scaffold": "global_scaffold",
    }[component]
    return (item.scope == scope
            or item.condition.get("component") == component
            or item.condition.get("attribution_target") == component)


class DeterministicUpdateReviewer:
    def review(
        self,
        *,
        proposal: Any,
        component: str,
        meta_knowledge: Sequence[MetaKnowledgeItem],
    ) -> UpdateReview:
        if not isinstance(proposal, PROPOSAL_TYPES):
            raise TypeError("proposal must be a typed component proposal")
        expected_type = {
            "prompt_bank": PromptBankUpdateProposal,
            "router": RouterUpdateProposal,
            "scaffold": ScaffoldUpdateProposal,
        }.get(component)
        if expected_type is None or not isinstance(proposal, expected_type):
            raise ValueError("proposal type does not match component")

        relevant = tuple(item for item in meta_knowledge
                         if _knowledge_relevant(item, component))
        confirmed = tuple(item for item in relevant if item.status == "confirmed")
        confirmed_conflicts = tuple(item for item in relevant
                                    if item.knowledge_type == "conflict"
                                    and item.status == "confirmed")
        candidate_conflicts = tuple(item for item in relevant
                                    if item.knowledge_type == "conflict"
                                    and item.status == "candidate")
        conflicts = tuple(item.principle for item in (
            *confirmed_conflicts, *candidate_conflicts))
        feedback_ids = _source_feedback_ids(proposal)

        if not proposal.is_valid:
            decision = "reject"
            rationale = "Proposal failed typed pre-validation."
        elif isinstance(proposal, ScaffoldUpdateProposal) and \
                proposal.skipped_reason != "none":
            decision = "defer"
            rationale = f"Scaffold proposal was skipped: {proposal.skipped_reason}."
        elif confirmed_conflicts:
            decision = "reject"
            rationale = "Confirmed meta-knowledge conflicts with the proposal."
        elif candidate_conflicts:
            decision = "revise"
            rationale = "Candidate conflict requires proposal revision."
        elif component == "scaffold" and not any(
                item.scope == "global_scaffold"
                and item.status == "confirmed"
                and len(item.distinct_video_ids) >= 3
                for item in relevant):
            decision = "defer"
            rationale = (
                "Global scaffold review requires confirmed support from at "
                "least three distinct videos.")
        elif len(feedback_ids) < 2 and not confirmed:
            decision = "defer"
            rationale = "One weak proposal sample lacks confirmed support."
        else:
            decision = "accept_for_validation"
            rationale = "Structured support is sufficient for empirical validation."

        scope = {
            "prompt_bank": "local_prompt",
            "router": "routing",
            "scaffold": "global_scaffold",
        }[component]
        required = [
            "confirmation_examples_disjoint_from_proposal_examples",
            "confirmation_videos_disjoint_from_proposal_videos",
            "regression_examples_and_videos_disjoint_from_proposal_evidence",
            "regression_examples_preserve_incumbent_behavior",
        ]
        if component == "scaffold":
            required.extend((
                "at_least_three_distinct_support_videos",
                "scaffold_contract_unchanged",
            ))
        identity = {
            "component": component,
            "proposal_id": proposal.proposal_id,
            "decision": decision,
            "conflicts": conflicts,
            "knowledge_ids": tuple(item.knowledge_id for item in relevant),
        }
        return UpdateReview(
            review_id="review_" + sha256_text(dumps_canonical(identity))[:24],
            component=component,
            proposal_id=proposal.proposal_id,
            decision=decision,
            recommended_scope=scope,
            conflicts=conflicts,
            required_confirmations=tuple(required),
            supporting_knowledge_ids=tuple(sorted(
                item.knowledge_id for item in relevant)),
            rationale=rationale,
        )
