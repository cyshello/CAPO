"""Preview-only Stage 4.10 router-policy update proposals."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from surrogate_rollout.optimization._proposal_support import (
    component_feedback,
    source_feedback_ids,
)
from surrogate_rollout.optimization.schemas import (
    FeedbackBatch,
    MetaKnowledgeItem,
    RouterUpdateProposal,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    RouterPolicySnapshot,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_text


ROUTER_OPERATION_KEYS = frozenset({
    "operation_id", "operation_type", "rule_id", "priority", "conditions",
    "target_prompt_ids", "source_feedback_ids", "confidence",
})
ROUTER_FOREIGN_CONDITION_KEYS = frozenset({
    "prompt_text", "composition_instruction", "ordering_policy",
    "compression_policy", "required_placeholders", "required_sections",
    "output_schema", "forbidden_patterns", "max_prompt_tokens",
})


class RouterUpdateProposer(Protocol):
    def propose(
        self,
        *,
        prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
        feedback: FeedbackBatch,
        meta_knowledge: Sequence[MetaKnowledgeItem],
    ) -> RouterUpdateProposal:
        ...


class DeterministicRouterUpdateProposer:
    def __init__(self, *, min_confidence: float = 0.7,
                 max_operations: int = 4) -> None:
        self.min_confidence = min_confidence
        self.max_operations = max_operations

    def propose(
        self,
        *,
        prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
        feedback: FeedbackBatch,
        meta_knowledge: Sequence[MetaKnowledgeItem],
    ) -> RouterUpdateProposal:
        _, supported = component_feedback(
            feedback, "router", min_confidence=self.min_confidence)
        active = {entry.prompt_id for entry in prompt_bank.entries
                  if entry.status == "active"}
        operations = []
        errors = []
        for index, (item, attribution) in enumerate(
                supported[:self.max_operations]):
            traits = item.applicable_segment_traits
            targets = traits.get("target_prompt_ids")
            if targets is None:
                single = traits.get("target_prompt_id")
                targets = ((single,) if single else prompt_bank.default_entry_ids)
            elif isinstance(targets, str):
                targets = (targets,)
            targets = tuple(str(value) for value in targets)
            unknown = set(targets) - active
            if not targets or unknown:
                errors.append(
                    f"router proposal has non-active targets: {sorted(unknown)}")
                continue
            conditions = traits.get("routing_conditions") or {
                "feedback_trait": item.failure_modes[0]
                if item.failure_modes else "targeted_failure"
            }
            if not isinstance(conditions, Mapping):
                errors.append("router conditions must be a mapping")
                continue
            leaked_conditions = set(conditions) & ROUTER_FOREIGN_CONDITION_KEYS
            if leaked_conditions:
                errors.append(
                    "router conditions contain foreign component keys: "
                    f"{sorted(leaked_conditions)}")
                continue
            rule_id = "proposed_rule_" + sha256_text(item.feedback_id)[:8]
            operation = {
                "operation_type": "add_rule",
                "rule_id": rule_id,
                "priority": 1000 + index,
                "conditions": conditions,
                "target_prompt_ids": targets,
                "source_feedback_ids": (item.feedback_id,),
                "confidence": min(item.confidence, attribution.confidence),
            }
            operation["operation_id"] = "router_op_" + sha256_text(
                dumps_canonical(operation))[:20]
            leaked = set(operation) - ROUTER_OPERATION_KEYS
            if leaked:
                errors.append(f"router operation contains foreign keys: {sorted(leaked)}")
                continue
            operations.append(operation)
        if not supported:
            errors.append(
                f"insufficient router feedback at confidence >= {self.min_confidence}")
        if not operations and not errors:
            errors.append("no router operations generated")
        identity = {
            "input_router_version": router_policy.router_version,
            "operations": tuple(operations),
            "errors": tuple(errors),
        }
        return RouterUpdateProposal(
            proposal_id="router_proposal_" + sha256_text(
                dumps_canonical(identity))[:20],
            input_router_version=router_policy.router_version,
            operations=tuple(operations),
            source_feedback_ids=source_feedback_ids(supported),
            validation_errors=tuple(errors),
            is_valid=bool(operations) and not errors,
            provenance={
                "preview_only": True,
                "input_bank_version": prompt_bank.bank_version,
                "feedback_policy_version": feedback.feedback_policy_version,
                "knowledge_ids": tuple(sorted(
                    item.knowledge_id for item in meta_knowledge
                    if item.scope == "routing"
                    or item.condition.get("attribution_target") == "router")),
            },
        )
