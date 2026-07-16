"""Deterministic force-add composition for one ephemeral candidate property."""

from __future__ import annotations

from typing import Any
from surrogate_rollout.prompt_routing.policies.deterministic_scaffold import (
    INSERTION_MARKER,
)
from surrogate_rollout.prompt_routing.scaffold_applier import validate_composed_text
from surrogate_rollout.prompt_routing.schemas import (
    ComposedCaptionPrompt,
    CompositionTrace,
    PromptBankSnapshot,
    RouterPolicySnapshot,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
)
from surrogate_rollout.schemas import sha256_text


class InterventionCompositionError(RuntimeError):
    pass


def compose_forced_property(
    *,
    incumbent: ComposedCaptionPrompt,
    candidate: Any,
    prompt_bank: PromptBankSnapshot,
    router_policy: RouterPolicySnapshot,
    scaffold_policy: ScaffoldPolicySnapshot,
    scaffold_contract: ScaffoldContract,
    base_prompt_template: str,
) -> ComposedCaptionPrompt:
    """Append exactly one candidate after all incumbent routed properties."""
    if not incumbent.is_valid:
        raise InterventionCompositionError("incumbent composed prompt is invalid")
    if incumbent.bank_version != prompt_bank.bank_version or \
            incumbent.router_version != router_policy.router_version or \
            incumbent.scaffold_version != scaffold_policy.scaffold_version or \
            incumbent.contract_version != scaffold_contract.contract_version:
        raise InterventionCompositionError("incumbent component versions changed")
    if incumbent.composition_trace.omitted_prompt_ids:
        raise InterventionCompositionError(
            "incumbent composition omitted routed properties; force-add would "
            "not preserve the incumbent sequence")
    active = {entry.prompt_id: entry for entry in prompt_bank.entries
              if entry.status == "active"}
    unknown = set(incumbent.selected_prompt_ids) - set(active)
    if unknown:
        raise InterventionCompositionError(
            f"incumbent references inactive or unknown properties: {sorted(unknown)}")
    if candidate.candidate_property_id in set(incumbent.selected_prompt_ids) or \
            candidate.candidate_property_id in active:
        raise InterventionCompositionError(
            "ephemeral candidate ID collides with incumbent codebook state")
    incumbent_limit = min(
        prompt_bank.max_selected_entries, router_policy.max_selected_entries)
    if len(incumbent.selected_prompt_ids) > incumbent_limit:
        raise InterventionCompositionError(
            "incumbent selection exceeds configured property limit")
    sequence = (*incumbent.selected_prompt_ids, candidate.candidate_property_id)
    if len(sequence) > incumbent_limit + 1:
        raise InterventionCompositionError(
            "temporary intervention sequence exceeds max_selected_properties + 1")

    header = scaffold_policy.configuration.get(
        "instructions_header", "Additional instructions:")
    bullet = scaffold_policy.configuration.get("bullet_prefix", "- ")
    texts = [active[property_id].prompt_text
             for property_id in incumbent.selected_prompt_ids]
    texts.append(candidate.property_text)
    block = "\n".join([header, *(f"{bullet}{text}" for text in texts)])
    if INSERTION_MARKER in base_prompt_template:
        prompt_text = base_prompt_template.replace(INSERTION_MARKER, block, 1)
    else:
        prompt_text = base_prompt_template.rstrip("\n") + "\n\n" + block
    errors = validate_composed_text(prompt_text, scaffold_contract)
    if errors:
        raise InterventionCompositionError(
            "forced intervention prompt violates scaffold contract: "
            + "; ".join(errors))
    trace = CompositionTrace(
        selected_prompt_ids=tuple(sequence),
        preserved_prompt_ids=tuple(sequence),
        ordering_decisions=(
            "incumbent routing order preserved",
            "one ephemeral candidate appended after incumbent properties",
        ),
        conflict_resolutions=("semantic conflict resolution disabled",),
    )
    return ComposedCaptionPrompt(
        video_id=incumbent.video_id,
        segment_id=incumbent.segment_id,
        bank_version=incumbent.bank_version,
        router_version=incumbent.router_version,
        scaffold_version=incumbent.scaffold_version,
        contract_version=incumbent.contract_version,
        selected_prompt_ids=tuple(sequence),
        prompt_text=prompt_text,
        prompt_hash=sha256_text(prompt_text),
        composition_trace=trace,
    )
