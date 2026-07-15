"""Rollback-safe in-memory candidate snapshots for Stage 4.11."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from surrogate_rollout.optimization.schemas import (
    PromptBankUpdateProposal,
    RouterUpdateProposal,
    ScaffoldUpdateProposal,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    PromptEntry,
    RouterPolicySnapshot,
    RoutingRule,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    make_component_version,
)
from surrogate_rollout.prompt_routing.validators import (
    validate_prompt_bank,
    validate_router_against_bank,
    validate_scaffold_policy,
)
from surrogate_rollout.schemas import sha256_text


class CandidatePreviewError(RuntimeError):
    """A proposal cannot produce a valid isolated candidate snapshot."""


@dataclass(frozen=True)
class CandidatePreview:
    component: str
    proposal_id: str
    incumbent_version: str
    candidate_version: str
    candidate_state: Any
    contract_version: str | None = None
    preview_only: bool = True
    canonical_state_mutated: bool = False


def _next_version(kind: str, version: str) -> str:
    match = re.search(r"_v(\d+)", version)
    if not match:
        raise CandidatePreviewError(f"cannot increment version {version!r}")
    return make_component_version(kind, int(match.group(1)) + 1)


class CandidatePreviewBuilder:
    def build_prompt_bank(
        self,
        incumbent: PromptBankSnapshot,
        proposal: PromptBankUpdateProposal,
    ) -> CandidatePreview:
        if not proposal.is_valid or proposal.input_bank_version != \
                incumbent.bank_version:
            raise CandidatePreviewError("invalid or stale prompt-bank proposal")
        entries = list(incumbent.entries)
        by_id = {entry.prompt_id: index for index, entry in enumerate(entries)}
        for operation in proposal.operations:
            payload = operation.payload
            if operation.operation_type == "add_entry":
                prompt_id = payload["prompt_id"]
                if prompt_id in by_id:
                    raise CandidatePreviewError(f"duplicate prompt_id {prompt_id!r}")
                entry = PromptEntry(
                    prompt_id=prompt_id,
                    prompt_text=payload["prompt_text"],
                    prompt_hash=sha256_text(payload["prompt_text"]),
                    name=payload["name"], description=payload["description"],
                    tags=tuple(payload.get("tags") or ()),
                    applicability_traits=payload.get("applicability_traits") or {},
                    parent_prompt_ids=tuple(payload.get("parent_prompt_ids") or ()),
                    created_by="stage4_11_preview",
                    provenance={"proposal_id": proposal.proposal_id,
                                "preview_only": True},
                )
                by_id[prompt_id] = len(entries)
                entries.append(entry)
            elif operation.operation_type == "revise_entry":
                prompt_id = operation.target_ids[0]
                if prompt_id not in by_id:
                    raise CandidatePreviewError(
                        f"unknown revision target {prompt_id!r}")
                current = entries[by_id[prompt_id]]
                prompt_text = payload.get("prompt_text", current.prompt_text)
                entries[by_id[prompt_id]] = replace(
                    current,
                    prompt_text=prompt_text,
                    prompt_hash=sha256_text(prompt_text),
                    description=payload.get("description", current.description),
                    parent_prompt_ids=tuple(payload.get(
                        "parent_prompt_ids", (current.prompt_id,))),
                    created_by="stage4_11_preview",
                    provenance={"proposal_id": proposal.proposal_id,
                                "preview_only": True},
                )
            elif operation.operation_type == "retire_entry":
                for prompt_id in operation.target_ids:
                    if prompt_id not in by_id:
                        raise CandidatePreviewError(
                            f"unknown retirement target {prompt_id!r}")
                    entries[by_id[prompt_id]] = replace(
                        entries[by_id[prompt_id]], status="retired")
            elif operation.operation_type == "no_op":
                continue
            else:
                raise CandidatePreviewError(
                    f"unsupported preview operation {operation.operation_type!r}")
        candidate = replace(
            incumbent,
            bank_version=_next_version("bank", incumbent.bank_version),
            parent_bank_version=incumbent.bank_version,
            entries=tuple(entries),
            created_by="stage4_11_preview",
            provenance={"proposal_id": proposal.proposal_id, "preview_only": True},
        )
        errors = validate_prompt_bank(candidate)
        if errors:
            raise CandidatePreviewError("; ".join(errors))
        return CandidatePreview(
            component="prompt_bank", proposal_id=proposal.proposal_id,
            incumbent_version=incumbent.bank_version,
            candidate_version=candidate.bank_version, candidate_state=candidate)

    def build_router(
        self,
        incumbent: RouterPolicySnapshot,
        proposal: RouterUpdateProposal,
        *,
        prompt_bank: PromptBankSnapshot,
    ) -> CandidatePreview:
        if not proposal.is_valid or proposal.input_router_version != \
                incumbent.router_version:
            raise CandidatePreviewError("invalid or stale router proposal")
        rules = list(incumbent.rules)
        known = {rule.rule_id for rule in rules}
        for operation in proposal.operations:
            if operation.get("operation_type") != "add_rule":
                raise CandidatePreviewError("unsupported router preview operation")
            rule_id = operation["rule_id"]
            if rule_id in known:
                raise CandidatePreviewError(f"duplicate rule_id {rule_id!r}")
            rules.append(RoutingRule(
                rule_id=rule_id, priority=int(operation["priority"]),
                conditions=operation["conditions"],
                target_prompt_ids=tuple(operation["target_prompt_ids"]),
                provenance={"proposal_id": proposal.proposal_id,
                            "preview_only": True},
            ))
            known.add(rule_id)
        candidate = replace(
            incumbent,
            router_version=_next_version("router", incumbent.router_version),
            parent_router_version=incumbent.router_version,
            rules=tuple(rules),
            provenance={"proposal_id": proposal.proposal_id, "preview_only": True},
        )
        errors = validate_router_against_bank(candidate, prompt_bank)
        if errors:
            raise CandidatePreviewError("; ".join(errors))
        return CandidatePreview(
            component="router", proposal_id=proposal.proposal_id,
            incumbent_version=incumbent.router_version,
            candidate_version=candidate.router_version, candidate_state=candidate)

    def build_scaffold(
        self,
        incumbent: ScaffoldPolicySnapshot,
        proposal: ScaffoldUpdateProposal,
        *,
        scaffold_contract: ScaffoldContract,
    ) -> CandidatePreview:
        if not proposal.is_valid or proposal.skipped_reason != "none" or \
                proposal.input_scaffold_version != incumbent.scaffold_version or \
                proposal.candidate_policy is None:
            raise CandidatePreviewError("invalid, skipped, or stale scaffold proposal")
        candidate = proposal.candidate_policy
        if candidate.parent_scaffold_version != incumbent.scaffold_version:
            raise CandidatePreviewError("scaffold candidate parent version mismatch")
        if candidate.provenance.get("contract_version") != \
                scaffold_contract.contract_version:
            raise CandidatePreviewError("scaffold contract lineage mismatch")
        errors = validate_scaffold_policy(candidate)
        if errors:
            raise CandidatePreviewError("; ".join(errors))
        return CandidatePreview(
            component="scaffold", proposal_id=proposal.proposal_id,
            incumbent_version=incumbent.scaffold_version,
            candidate_version=candidate.scaffold_version,
            candidate_state=candidate,
            contract_version=scaffold_contract.contract_version)
