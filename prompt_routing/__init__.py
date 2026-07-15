"""Phase 4 prompt-routing package (PHASE4_PROMPT_ROUTING.md).

Stage 4.1: foundational typed records. Stage 4.2: versioned component
persistence and cross-record validation. Stage 4.3: routing. Stage 4.4:
scaffold application. Stage 4.5: offline routing-to-composition dry runs.
Caption integration arrives in later stages.
"""

from surrogate_rollout.prompt_routing.offline_dry_run import (
    OfflineDryRunError,
    OfflineDryRunResult,
    load_fixture_bundle,
    run_offline_dry_run,
    segment_context_from_json,
)

from surrogate_rollout.prompt_routing.policies.deterministic_scaffold import (
    INSERTION_MARKER,
    DeterministicScaffoldApplier,
)
from surrogate_rollout.prompt_routing.policies.rule_based_router import (
    RuleBasedPromptRouter,
)
from surrogate_rollout.prompt_routing.policies.slm_scaffold import (
    SLMScaffoldApplier,
)
from surrogate_rollout.prompt_routing.scaffold_applier import (
    ScaffoldApplier,
    composed_prompt_from_json,
    composition_trace_from_json,
    create_scaffold_applier,
    estimate_prompt_tokens,
    finalize_composed_prompt,
    read_composed_prompts,
    validate_composed_text,
    write_composed_prompts,
)
from surrogate_rollout.prompt_routing.router import (
    PromptRouter,
    read_routing_decisions,
    routing_decision_from_json,
    write_routing_decisions,
)
from surrogate_rollout.prompt_routing.persistence import (
    ComponentSnapshotStore,
    SnapshotStoreError,
    SnapshotValidationError,
    VersionConflictError,
    prompt_bank_from_json,
    prompt_bank_store,
    router_policy_from_json,
    router_policy_store,
    scaffold_contract_from_json,
    scaffold_contract_store,
    scaffold_policy_from_json,
    scaffold_policy_store,
)
from surrogate_rollout.prompt_routing.validators import (
    assert_router_compatible,
    validate_prompt_bank,
    validate_router_against_bank,
    validate_scaffold_policy,
)
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
    "ComponentSnapshotStore",
    "DeterministicScaffoldApplier",
    "INSERTION_MARKER",
    "OfflineDryRunError",
    "OfflineDryRunResult",
    "PromptRouter",
    "RuleBasedPromptRouter",
    "SLMScaffoldApplier",
    "ScaffoldApplier",
    "composed_prompt_from_json",
    "composition_trace_from_json",
    "create_scaffold_applier",
    "estimate_prompt_tokens",
    "finalize_composed_prompt",
    "load_fixture_bundle",
    "read_composed_prompts",
    "validate_composed_text",
    "write_composed_prompts",
    "read_routing_decisions",
    "routing_decision_from_json",
    "write_routing_decisions",
    "run_offline_dry_run",
    "segment_context_from_json",
    "SnapshotStoreError",
    "SnapshotValidationError",
    "VersionConflictError",
    "assert_router_compatible",
    "prompt_bank_from_json",
    "prompt_bank_store",
    "router_policy_from_json",
    "router_policy_store",
    "scaffold_contract_from_json",
    "scaffold_contract_store",
    "scaffold_policy_from_json",
    "scaffold_policy_store",
    "validate_prompt_bank",
    "validate_router_against_bank",
    "validate_scaffold_policy",
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
