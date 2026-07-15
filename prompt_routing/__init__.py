"""Phase 4 prompt-routing package (PHASE4_PROMPT_ROUTING.md).

Stage 4.1: foundational typed records only. Routing logic, scaffold
application, persistence, and caption integration arrive in later stages.
"""

from surrogate_rollout.prompt_routing.schemas import (
    COMPONENT_KINDS,
    ComposedCaptionPrompt,
    CompositionTrace,
    Phase4Config,
    Phase4OptimizationConfig,
    PromptBankSnapshot,
    PromptEntry,
    RouterPolicySnapshot,
    RoutingDecision,
    RoutingRule,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    SegmentContext,
    as_json_dict,
    dumps_canonical,
    make_component_version,
    validate_component_version,
)

__all__ = [
    "COMPONENT_KINDS",
    "ComposedCaptionPrompt",
    "CompositionTrace",
    "Phase4Config",
    "Phase4OptimizationConfig",
    "PromptBankSnapshot",
    "PromptEntry",
    "RouterPolicySnapshot",
    "RoutingDecision",
    "RoutingRule",
    "ScaffoldContract",
    "ScaffoldPolicySnapshot",
    "SegmentContext",
    "as_json_dict",
    "dumps_canonical",
    "make_component_version",
    "validate_component_version",
]
