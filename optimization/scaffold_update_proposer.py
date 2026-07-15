"""Preview-only Stage 4.10 scaffold-policy update proposals."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Callable, Mapping, Protocol, Sequence

from surrogate_rollout.optimization._proposal_support import (
    component_feedback,
    source_feedback_ids,
)
from surrogate_rollout.optimization.schemas import (
    FeedbackBatch,
    FeedbackItem,
    MetaKnowledgeItem,
    ScaffoldUpdateProposal,
)
from surrogate_rollout.prompt_routing.schemas import (
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    dumps_canonical,
    make_component_version,
)
from surrogate_rollout.prompt_routing.validators import (
    CONTRACT_OWNED_KEYS,
    validate_scaffold_policy,
)
from surrogate_rollout.schemas import sha256_text


SCAFFOLD_POLICY_KEYS = frozenset({
    "composition_instruction", "ordering_policy", "conflict_resolution_rules",
    "compression_policy", "configuration",
})
SCAFFOLD_FOREIGN_CONFIGURATION_KEYS = frozenset({
    "prompt_text", "target_prompt_ids", "routing_conditions", "rule_id",
})

ScaffoldProposalPolicy = Callable[
    [ScaffoldPolicySnapshot, Sequence[FeedbackItem], Sequence[MetaKnowledgeItem]],
    Mapping[str, Any],
]


class ScaffoldUpdateProposer(Protocol):
    def propose(
        self,
        *,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        feedback: FeedbackBatch,
        meta_knowledge: Sequence[MetaKnowledgeItem],
        enabled: bool,
    ) -> ScaffoldUpdateProposal:
        ...


def _next_scaffold_version(version: str) -> str:
    match = re.search(r"_v(\d+)", version)
    if not match:
        raise ValueError(f"cannot increment scaffold version {version!r}")
    return make_component_version("scaffold", int(match.group(1)) + 1)


def _default_policy(
    scaffold: ScaffoldPolicySnapshot,
    feedback_items: Sequence[FeedbackItem],
    meta_knowledge: Sequence[MetaKnowledgeItem],
) -> Mapping[str, Any]:
    del meta_knowledge
    desired = (feedback_items[0].desired_behaviors[0]
               if feedback_items[0].desired_behaviors
               else "preserve supported selected instructions")
    return {
        "composition_instruction": (
            f"{scaffold.composition_instruction.rstrip()} {desired}".strip()),
    }


class DeterministicScaffoldUpdateProposer:
    def __init__(
        self,
        *,
        min_confidence: float = 0.7,
        min_distinct_videos: int = 3,
        proposal_policy: ScaffoldProposalPolicy | None = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_distinct_videos = min_distinct_videos
        self.proposal_policy = proposal_policy or _default_policy

    @staticmethod
    def _skipped(
        scaffold_policy: ScaffoldPolicySnapshot,
        reason: str,
        *,
        feedback_ids: tuple[str, ...] = (),
    ) -> ScaffoldUpdateProposal:
        identity = {
            "input_scaffold_version": scaffold_policy.scaffold_version,
            "skipped_reason": reason,
            "source_feedback_ids": feedback_ids,
        }
        return ScaffoldUpdateProposal(
            proposal_id="scaffold_proposal_" + sha256_text(
                dumps_canonical(identity))[:20],
            input_scaffold_version=scaffold_policy.scaffold_version,
            candidate_policy=None,
            source_feedback_ids=feedback_ids,
            validation_errors=(),
            is_valid=True,
            skipped_reason=reason,
            provenance={"preview_only": True, "proposal_policy_calls": 0},
        )

    def propose(
        self,
        *,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        feedback: FeedbackBatch,
        meta_knowledge: Sequence[MetaKnowledgeItem],
        enabled: bool,
    ) -> ScaffoldUpdateProposal:
        # This branch precedes feedback inspection and policy invocation by design.
        if not enabled:
            return self._skipped(scaffold_policy, "disabled_by_config")

        relevant, supported = component_feedback(
            feedback, "scaffold", min_confidence=self.min_confidence)
        if not relevant:
            return self._skipped(scaffold_policy, "no_scaffold_attribution")
        if not supported:
            return self._skipped(
                scaffold_policy, "insufficient_support",
                feedback_ids=source_feedback_ids(relevant))

        confirmed = tuple(item for item in meta_knowledge
                          if item.knowledge_type == "composition_pattern"
                          and item.scope == "global_scaffold"
                          and item.status == "confirmed"
                          and len(item.distinct_video_ids) >= self.min_distinct_videos)
        if not confirmed:
            return self._skipped(
                scaffold_policy, "insufficient_support",
                feedback_ids=source_feedback_ids(supported))

        feedback_items = tuple(item for item, _ in supported)
        updates = dict(self.proposal_policy(
            scaffold_policy, feedback_items, confirmed))
        errors = []
        hard_keys = set(updates) & CONTRACT_OWNED_KEYS
        unknown = set(updates) - SCAFFOLD_POLICY_KEYS - CONTRACT_OWNED_KEYS
        if hard_keys:
            errors.append(
                f"proposal attempts to modify contract-owned fields: "
                f"{sorted(hard_keys)}")
        if unknown:
            errors.append(
                f"scaffold proposal contains foreign fields: {sorted(unknown)}")
        configuration = updates.get("configuration")
        if isinstance(configuration, Mapping):
            contract_config = set(configuration) & CONTRACT_OWNED_KEYS
            if contract_config:
                errors.append(
                    "scaffold configuration attempts to modify contract-owned "
                    f"fields: {sorted(contract_config)}")
            foreign_config = set(configuration) & SCAFFOLD_FOREIGN_CONFIGURATION_KEYS
            if foreign_config:
                errors.append(
                    "scaffold configuration contains foreign component fields: "
                    f"{sorted(foreign_config)}")
        elif configuration is not None:
            errors.append("scaffold configuration update must be a mapping")

        candidate = None
        if not errors:
            merged_configuration = dict(scaffold_policy.configuration)
            merged_configuration.update(configuration or {})
            candidate = replace(
                scaffold_policy,
                scaffold_version=_next_scaffold_version(
                    scaffold_policy.scaffold_version),
                parent_scaffold_version=scaffold_policy.scaffold_version,
                composition_instruction=updates.get(
                    "composition_instruction",
                    scaffold_policy.composition_instruction),
                ordering_policy=updates.get(
                    "ordering_policy", scaffold_policy.ordering_policy),
                conflict_resolution_rules=tuple(updates.get(
                    "conflict_resolution_rules",
                    scaffold_policy.conflict_resolution_rules)),
                compression_policy=updates.get(
                    "compression_policy", scaffold_policy.compression_policy),
                configuration=merged_configuration,
                provenance={
                    "preview_only": True,
                    "input_scaffold_version": scaffold_policy.scaffold_version,
                    "contract_version": scaffold_contract.contract_version,
                    "source_feedback_ids": source_feedback_ids(supported),
                    "knowledge_ids": tuple(item.knowledge_id for item in confirmed),
                },
            )
            errors.extend(validate_scaffold_policy(candidate))
            if errors:
                candidate = None

        identity = {
            "input_scaffold_version": scaffold_policy.scaffold_version,
            "candidate_policy": candidate,
            "errors": tuple(errors),
            "source_feedback_ids": source_feedback_ids(supported),
        }
        return ScaffoldUpdateProposal(
            proposal_id="scaffold_proposal_" + sha256_text(
                dumps_canonical(identity))[:20],
            input_scaffold_version=scaffold_policy.scaffold_version,
            candidate_policy=candidate,
            source_feedback_ids=source_feedback_ids(supported),
            validation_errors=tuple(errors),
            is_valid=candidate is not None and not errors,
            skipped_reason="none",
            provenance={
                "preview_only": True,
                "proposal_policy_calls": 1,
                "contract_version": scaffold_contract.contract_version,
            },
        )
