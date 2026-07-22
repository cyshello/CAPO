"""Replace-body scaffold applier (parity with the private baseline repo).

Where `DeterministicScaffoldApplier` keeps the base template and adds the
selected instructions as a bullet block, this applier lets the selected
instruction OWN the task section: the composed prompt is the instruction text,
then the transcript block, then the base template's JSON output contract
verbatim. The contract is never regenerated, so caption parsing is unchanged.

Intended for the free-form (static meta-prompt) condition, where exactly one
generated entry is selected. Multiple entries are composed in the router's
selection order, separated by a blank line — the same text the baseline repo
would produce for a single entry, and a defined (not silently truncated)
behaviour for the multi-entry case.
"""
from __future__ import annotations

from typing import Sequence

from surrogate_rollout.prompt_routing import static_meta_replace_body as smrb
from surrogate_rollout.prompt_routing.scaffold_applier import (
    finalize_composed_prompt,
)
from surrogate_rollout.prompt_routing.schemas import (
    ComposedCaptionPrompt,
    CompositionTrace,
    PromptEntry,
    RoutingDecision,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    SegmentContext,
)


class ReplaceBodyScaffoldApplier:
    """ScaffoldApplier that replaces the base template's task section."""

    policy_type = "replace_body"

    def __init__(self, *, base_prompt_template: str) -> None:
        if not base_prompt_template:
            raise ValueError("base_prompt_template must be non-empty")
        self.base_prompt_template = base_prompt_template

    def apply(
        self,
        *,
        context: SegmentContext,
        selected_entries: Sequence[PromptEntry],
        routing_decision: RoutingDecision,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
    ) -> ComposedCaptionPrompt:
        given_ids = tuple(entry.prompt_id for entry in selected_entries)
        if given_ids != routing_decision.selected_prompt_ids:
            raise ValueError(
                f"selected_entries {given_ids} do not match routing decision "
                f"selected_prompt_ids {routing_decision.selected_prompt_ids}")
        if not selected_entries:
            raise ValueError(
                "replace_body composition requires at least one selected entry "
                "(there is no task section to fall back to)")

        kept: list[PromptEntry] = []
        omitted: list[str] = []
        compression_actions: list[str] = []
        seen: dict[str, str] = {}
        for entry in selected_entries:
            duplicate_of = seen.get(entry.prompt_hash)
            if duplicate_of is not None:
                omitted.append(entry.prompt_id)
                compression_actions.append(
                    f"omitted {entry.prompt_id}: redundant with "
                    f"{duplicate_of} (identical prompt_text)")
                continue
            seen[entry.prompt_hash] = entry.prompt_id
            kept.append(entry)

        body = "\n\n".join(entry.prompt_text.strip() for entry in kept)
        try:
            prompt_text, _ = smrb.compose_caption_prompt(
                self.base_prompt_template, body)
        except smrb.StaticMetaPromptError as exc:
            # A contract-breaking generation is a validation outcome, not a
            # crash: surface it as an invalid composed prompt.
            return finalize_composed_prompt(
                context=context, routing_decision=routing_decision,
                scaffold_policy=scaffold_policy,
                scaffold_contract=scaffold_contract,
                prompt_text="",
                trace=CompositionTrace(
                    selected_prompt_ids=routing_decision.selected_prompt_ids,
                    preserved_prompt_ids=(),
                    omitted_prompt_ids=given_ids,
                    conflicts_detected=(),
                    conflict_resolutions=(),
                    ordering_decisions=("router_selection_order preserved",),
                    compression_actions=tuple(
                        compression_actions + [f"composition failed: {exc}"]),
                ))

        trace = CompositionTrace(
            selected_prompt_ids=routing_decision.selected_prompt_ids,
            preserved_prompt_ids=tuple(entry.prompt_id for entry in kept),
            omitted_prompt_ids=tuple(omitted),
            conflicts_detected=(),
            conflict_resolutions=(),
            ordering_decisions=("router_selection_order preserved",
                                "generated task section replaces base body"),
            compression_actions=tuple(compression_actions),
        )
        return finalize_composed_prompt(
            context=context, routing_decision=routing_decision,
            scaffold_policy=scaffold_policy,
            scaffold_contract=scaffold_contract,
            prompt_text=prompt_text, trace=trace)
