"""Preview-only Stage 4.10 prompt-bank update proposals."""

from __future__ import annotations

from typing import Protocol, Sequence

from surrogate_rollout.optimization._proposal_support import component_feedback
from surrogate_rollout.optimization.schemas import (
    FeedbackBatch,
    MetaKnowledgeItem,
    PromptBankOperation,
    PromptBankUpdateProposal,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_text


BANK_PAYLOAD_KEYS = frozenset({
    "prompt_id", "prompt_text", "name", "description", "tags",
    "applicability_traits", "parent_prompt_ids",
})
BANK_FOREIGN_TRAIT_KEYS = frozenset({
    "target_prompt_id", "target_prompt_ids", "routing_conditions", "rule_id",
    "priority", "ordering_policy", "compression_policy",
})


class PromptBankUpdateProposer(Protocol):
    def propose(
        self,
        *,
        prompt_bank: PromptBankSnapshot,
        feedback: FeedbackBatch,
        meta_knowledge: Sequence[MetaKnowledgeItem],
    ) -> PromptBankUpdateProposal:
        ...


class DeterministicPromptBankUpdateProposer:
    def __init__(self, *, min_confidence: float = 0.7,
                 max_operations: int = 2) -> None:
        self.min_confidence = min_confidence
        self.max_operations = max_operations

    def propose(
        self,
        *,
        prompt_bank: PromptBankSnapshot,
        feedback: FeedbackBatch,
        meta_knowledge: Sequence[MetaKnowledgeItem],
    ) -> PromptBankUpdateProposal:
        _, supported = component_feedback(
            feedback, "prompt_bank", min_confidence=self.min_confidence)
        existing_ids = {entry.prompt_id for entry in prompt_bank.entries}
        operations = []
        errors = []
        for item, attribution in supported[:self.max_operations]:
            traits = item.applicable_segment_traits
            target_id = traits.get("target_prompt_id")
            instruction = (item.desired_behaviors[0]
                           if item.desired_behaviors else "clarify reusable behavior")
            confidence = min(item.confidence, attribution.confidence)
            if target_id:
                if target_id not in existing_ids:
                    errors.append(f"unknown prompt-bank revision target {target_id!r}")
                    continue
                operation_type = "revise_entry"
                target_ids = (str(target_id),)
                payload = {
                    "prompt_text": instruction,
                    "description": "Preview revision from structured feedback.",
                    "parent_prompt_ids": (str(target_id),),
                }
            else:
                operation_type = "add_entry"
                target_ids = ()
                prompt_id = "pe_proposed_" + sha256_text(item.feedback_id)[:8]
                payload = {
                    "prompt_id": prompt_id,
                    "prompt_text": instruction,
                    "name": "Proposed reusable behavior",
                    "description": "Preview addition from structured feedback.",
                    "tags": ("proposed",),
                    "applicability_traits": {
                        key: value for key, value in traits.items()
                        if key not in BANK_FOREIGN_TRAIT_KEYS
                    },
                    "parent_prompt_ids": (),
                }
            leaked = set(payload) - BANK_PAYLOAD_KEYS
            if leaked:
                errors.append(f"prompt-bank payload contains foreign keys: {sorted(leaked)}")
                continue
            operation_id = "bank_op_" + sha256_text(dumps_canonical({
                "type": operation_type,
                "targets": target_ids,
                "payload": payload,
                "feedback_id": item.feedback_id,
            }))[:20]
            operations.append(PromptBankOperation(
                operation_id=operation_id,
                operation_type=operation_type,
                target_ids=target_ids,
                payload=payload,
                source_feedback_ids=(item.feedback_id,),
                confidence=confidence,
            ))
        if not supported:
            errors.append(
                f"insufficient prompt-bank feedback at confidence >= "
                f"{self.min_confidence}")
        if not operations and not errors:
            errors.append("no prompt-bank operations generated")
        identity = {
            "input_bank_version": prompt_bank.bank_version,
            "operations": tuple(operations),
            "errors": tuple(errors),
        }
        return PromptBankUpdateProposal(
            proposal_id="bank_proposal_" + sha256_text(
                dumps_canonical(identity))[:20],
            input_bank_version=prompt_bank.bank_version,
            operations=tuple(operations),
            validation_errors=tuple(errors),
            is_valid=bool(operations) and not errors,
            provenance={
                "preview_only": True,
                "feedback_policy_version": feedback.feedback_policy_version,
                "knowledge_ids": tuple(sorted(
                    item.knowledge_id for item in meta_knowledge
                    if item.scope == "local_prompt"
                    or item.condition.get("attribution_target") == "prompt_bank")),
            },
        )
