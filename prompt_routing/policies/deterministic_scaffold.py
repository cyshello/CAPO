"""Deterministic scaffold applier (Stage 4.4) — the initial real
implementation AND the test implementation (§11.2, CLAUDE.md: initial
implementation may use a deterministic scaffold).

Composition algorithm (every step deterministic and recorded in the
CompositionTrace, §3.3 ownership):

1. structural checks abort (ValueError): the given entries must match the
   routing decision's selected_prompt_ids in order, and the context must match
   the decision's video/segment — misuse is a caller bug, not a validation
   outcome;
2. ordering: the router's selection order is preserved verbatim
   (`ordering_policy=router_selection_order`); recorded in
   `ordering_decisions`;
3. redundancy removal: a later entry whose prompt_text hash equals an earlier
   kept entry's is omitted (recorded in `compression_actions`) — identical
   wording never appears twice;
4. conflict resolution: an entry conflicting (via `conflicts_with`, either
   direction) with an earlier kept entry is omitted — earlier wins, matching
   the router's own rule (`drop_lower_priority_on_conflict`); the router
   already prevents this for its own selections, so this is defense against
   hand-built or replayed decisions; recorded in `conflicts_detected` +
   `conflict_resolutions`;
5. composition: kept instructions become one block — a header line followed
   by one bullet per entry (header and bullet prefix configurable through
   `scaffold_policy.configuration`: `instructions_header`, `bullet_prefix`).
   If the base template contains the insertion marker
   (`ROUTED_INSTRUCTIONS_PLACEHOLDER`) the block replaces it, otherwise the
   block is appended after the template. Zero kept entries -> the marker is
   removed and the base template stands alone;
6. length: while the whitespace-token proxy exceeds
   `contract.max_prompt_tokens`, the LAST kept entry (lowest router priority)
   is omitted and the prompt recomposed (recorded in `compression_actions`).
   If the base template alone still exceeds the limit, composition stops and
   validation flags the prompt invalid — never silent truncation of the base
   contract text;
7. the final text goes through `finalize_composed_prompt`, so contract
   violations surface as `is_valid=False` + `validation_errors`.

Selected-instruction preservation: every preserved entry's prompt_text appears
verbatim in the final prompt; every omitted entry is accounted for in the
trace (preserved + omitted exactly partition selected).
"""

from __future__ import annotations

from typing import Sequence

from surrogate_rollout.prompt_routing.scaffold_applier import (
    estimate_prompt_tokens,
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

INSERTION_MARKER = "ROUTED_INSTRUCTIONS_PLACEHOLDER"
_DEFAULT_HEADER = "Additional instructions:"
_DEFAULT_BULLET = "- "


class DeterministicScaffoldApplier:
    """ScaffoldApplier implementation with fully deterministic composition."""

    policy_type = "deterministic"

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
        given_ids = tuple(e.prompt_id for e in selected_entries)
        if given_ids != routing_decision.selected_prompt_ids:
            raise ValueError(
                f"selected_entries {given_ids} do not match routing decision "
                f"selected_prompt_ids {routing_decision.selected_prompt_ids}")
        if (context.video_id != routing_decision.video_id
                or context.segment_id != routing_decision.segment_id):
            raise ValueError(
                f"context ({context.video_id!r}, {context.segment_id!r}) does "
                f"not match routing decision ({routing_decision.video_id!r}, "
                f"{routing_decision.segment_id!r})")

        ordering_decisions = ("router_selection_order preserved",)
        compression_actions: list[str] = []
        conflicts_detected: list[str] = []
        conflict_resolutions: list[str] = []
        omitted: list[str] = []
        kept: list[PromptEntry] = []

        seen_hashes: dict[str, str] = {}
        for entry in selected_entries:
            duplicate_of = seen_hashes.get(entry.prompt_hash)
            if duplicate_of is not None:
                omitted.append(entry.prompt_id)
                compression_actions.append(
                    f"omitted {entry.prompt_id}: redundant with "
                    f"{duplicate_of} (identical prompt_text)")
                continue
            blocker = self._conflicting_kept(entry, kept)
            if blocker is not None:
                omitted.append(entry.prompt_id)
                conflicts_detected.append(
                    f"{entry.prompt_id}<->{blocker}")
                conflict_resolutions.append(
                    f"dropped {entry.prompt_id}, kept {blocker} "
                    "(drop_lower_priority_on_conflict)")
                continue
            seen_hashes[entry.prompt_hash] = entry.prompt_id
            kept.append(entry)

        prompt_text = self._compose(kept, scaffold_policy)
        limit = scaffold_contract.max_prompt_tokens
        if limit is not None:
            while kept and estimate_prompt_tokens(prompt_text) > limit:
                dropped = kept.pop()
                omitted.append(dropped.prompt_id)
                compression_actions.append(
                    f"omitted {dropped.prompt_id}: prompt exceeded "
                    f"max_prompt_tokens={limit}")
                prompt_text = self._compose(kept, scaffold_policy)

        trace = CompositionTrace(
            selected_prompt_ids=routing_decision.selected_prompt_ids,
            preserved_prompt_ids=tuple(e.prompt_id for e in kept),
            omitted_prompt_ids=tuple(omitted),
            conflicts_detected=tuple(conflicts_detected),
            conflict_resolutions=tuple(conflict_resolutions),
            ordering_decisions=ordering_decisions,
            compression_actions=tuple(compression_actions),
        )
        return finalize_composed_prompt(
            context=context,
            routing_decision=routing_decision,
            scaffold_policy=scaffold_policy,
            scaffold_contract=scaffold_contract,
            prompt_text=prompt_text,
            trace=trace,
        )

    # ------------------------------------------------------------------ #
    def _compose(self, kept: Sequence[PromptEntry],
                 scaffold_policy: ScaffoldPolicySnapshot) -> str:
        header = scaffold_policy.configuration.get(
            "instructions_header", _DEFAULT_HEADER)
        bullet = scaffold_policy.configuration.get(
            "bullet_prefix", _DEFAULT_BULLET)
        block = ""
        if kept:
            lines = [header] + [f"{bullet}{e.prompt_text}" for e in kept]
            block = "\n".join(lines)
        template = self.base_prompt_template
        if INSERTION_MARKER in template:
            return template.replace(INSERTION_MARKER, block, 1)
        if not block:
            return template
        return template.rstrip("\n") + "\n\n" + block

    @staticmethod
    def _conflicting_kept(entry: PromptEntry,
                          kept: Sequence[PromptEntry]) -> str | None:
        """First kept entry conflicting with `entry` in either direction."""
        entry_conflicts = set(entry.conflicts_with)
        for k in kept:
            if k.prompt_id in entry_conflicts or \
                    entry.prompt_id in set(k.conflicts_with):
                return k.prompt_id
        return None
