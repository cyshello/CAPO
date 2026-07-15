"""Interface-compatible SLM scaffold applier stub (Stage 4.4, §11.2).

A small-language-model composer will implement the ScaffoldApplier protocol
here in a later phase. Until then this stub fails clearly and immediately —
before any file write, model call, or other side effect — so a
misconfiguration can never silently produce prompts.
"""

from __future__ import annotations

from typing import Sequence

from surrogate_rollout.prompt_routing.schemas import (
    ComposedCaptionPrompt,
    PromptEntry,
    RoutingDecision,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    SegmentContext,
)


class SLMScaffoldApplier:
    """ScaffoldApplier-shaped stub; every apply() raises NotImplementedError."""

    policy_type = "slm"

    def apply(
        self,
        *,
        context: SegmentContext,
        selected_entries: Sequence[PromptEntry],
        routing_decision: RoutingDecision,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
    ) -> ComposedCaptionPrompt:
        raise NotImplementedError(
            "SLMScaffoldApplier is an interface-compatible stub (Stage 4.4): "
            "no SLM composition is implemented yet. Use scaffold policy_type "
            "'deterministic' or implement the SLM applier in a later phase.")
