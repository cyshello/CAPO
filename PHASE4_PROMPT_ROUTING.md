# Phase 4: Compositional Prompt Routing and Component-Wise Optimization

## Status

- **Prerequisite:** Phase 0–3 selective-surrogate-rollout infrastructure is implemented.
- **Active goal:** Add a compositional prompt bank, per-segment prompt routing, and component-wise optimization on top of the existing evaluators.
- **Implementation policy:** This phase must be implemented incrementally. Do not implement the entire phase in one task or one commit.
- **Default experiment setting:** Prompt-bank and router updates may be enabled, while scaffold optimization is disabled by default and exposed as an explicit ablation parameter.

Refer to `PHASE2_3_SURROGATE.md` for the completed rollout implementation, invariants, commands, and test expectations.

---

## Implementation progress (updated 2026-07-16)

### Completed stages and commits

| Stage | Content | Commit |
|---|---|---|
| 4.0 | Inspection + integration map (`PHASE4_STAGE40_INTEGRATION_MAP.md`) | `c6e8bd2` (baseline), `69bca67` (CLAUDE.md role fix) |
| 4.1 | Foundational schemas (`prompt_routing/schemas.py`) | `668f54f` |
| 4.1h | Hardening: opaque `segment_id`, recursive deep-freeze, version/digest + `prompt_scores` semantics documented | `4d5c416` |
| 4.2 | Versioned persistence (`persistence.py`) + cross-record validation (`validators.py`) | `bcc5bdb` |
| 4.3 | Multi-entry rule-based router (`router.py`, `policies/rule_based_router.py`) + numeric `list_versions` sort | `ce8c776` |
| 4.4 | Fixed scaffold application (`scaffold_applier.py`, deterministic applier, SLM stub) | `cad027a` |
| 4.5 | Offline routing-to-composition dry run (`offline_dry_run.py`, deterministic fixture CLI) | `f192433` |
| 4.6 | Routed caption-view adapter (`routed_caption_view.py`) with temporal whole-video registry merge | this commit |

Stages 4.7+ NOT started. Stop-gate protocol: after each stage, run tests,
produce the Section 20 checkpoint report, stop, wait for approval.

### Current configuration state

`Phase4Config` defaults (typed only; no YAML yet, deferred by review):
`optimize_prompt_bank=True`, `optimize_router=True`, `optimize_scaffold=False`,
`dry_run=True`, `commit=False`. `optimize_scaffold` exposed as top-level
`Phase4Config.optimize_scaffold` property.

### Fixed interfaces and invariants (do not change without review)

- Records: frozen dataclasses in `prompt_routing/schemas.py`; structured
  values recursively frozen (mappings→MappingProxyType, lists→tuples,
  sets→sorted tuples); only JSON-vocabulary leaves accepted;
  `as_json_dict`/`dumps_canonical` = deterministic serialization.
- `segment_id` is opaque non-empty; clip-key format ("{start}_{end}")
  validated later by the DVD/routed-caption adapter, not the schema.
- Versions: `<kind>_v<NNNN>[_<digest8>]`, kinds bank/router/scaffold/contract;
  monotonic component-local number; digest = sha256 over canonical record
  JSON with the record's own version field removed; versions are provenance
  IDs, never caption-cache keys; caption-cache identity = composed prompt
  text/hash only (byte-identical composed text may share a cache entry across
  component versions).
- Persistence: `ComponentSnapshotStore` — write-once
  `<root>/snapshots/<version>.json` + `current.json` pointer manifest
  ({component, version, updated_at} only); same-dir tmp + `os.replace`
  everywhere; identical re-save = idempotent no-op; different content under
  existing version = `VersionConflictError`; `write_preview` only outside the
  store root, never touches committed state; `list_versions` sorts by parsed
  numeric version.
- Validators: duplicate ACTIVE prompt hashes rejected; scaffold-policy
  configuration must not carry contract-owned keys (`CONTRACT_OWNED_KEYS`);
  router-vs-bank: targets/fallbacks must exist, enabled-rule targets and
  fallbacks must be active (retired tolerated only in disabled rules),
  `router.max_selected_entries <= bank.max_selected_entries`.
- Router: `PromptRouter` protocol `route(context, bank, policy) ->
  RoutingDecision`. RuleBasedPromptRouter: (priority asc, rule_id) order;
  first-appearance dedup; bidirectional `conflicts_with`, earlier wins, drops
  recorded in `decision_payload`; budget truncation recorded; fallback only
  on empty selection (`used_fallback` + reason); `prompt_scores` covers all
  considered candidates (rule 1.0, fallback 0.5); question conditions skipped
  unless policy configuration `question_conditioned: true` (only
  `question_contains` supported).
- Stage 4.6 subject-registry invariant recorded in
  `PHASE4_STAGE40_INTEGRATION_MAP.md` §8b (group captioning by composed-prompt
  hash, restore temporal order, reuse unchanged whole-video DVD merge).

### Unresolved assumptions

- Canonical repo locations for committed component state (e.g.
  `artifacts/prompt_banks/`) not yet created — decided at optimization-loop/CLI
  stage.
- `max_prompt_tokens=2048` in the Stage 4.2 example contract is a placeholder;
  production value must be derived from the actual captioner configuration.
- Rule-condition semantics = exact equality only; router scores are
  uncalibrated membership constants (1.0/0.5).
- `current.json` `updated_at` is intentionally nondeterministic pointer
  metadata.
- Tests run from the repo parent dir with the `local_llm_vllm` conda env
  (the `youngseo` env lacks pytest).

### Exact next step (pending approval)

Stage 4.7: routed evaluator smoke test — freeze one bank, router policy,
scaffold policy, and scaffold contract; compare fallback/default composition
against routed multi-entry composition on one small video and one to three QAs
through existing DVD reasoning. Record routing, composition, cache, prediction,
score, cost, and timing artifacts; do not implement evidence or updates.

### Test commands

```bash
# focused (per stage)
/home/intern/.conda/envs/local_llm_vllm/bin/python -m pytest \
    surrogate_rollout/tests/test_phase4_schemas.py \
    surrogate_rollout/tests/test_phase4_persistence.py \
    surrogate_rollout/tests/test_prompt_router.py \
    surrogate_rollout/tests/test_scaffold_applier.py \
    surrogate_rollout/tests/test_offline_dry_run.py \
    surrogate_rollout/tests/test_routed_caption_view.py -q
# complete suite (run from /home/intern/youngseo, the repo parent)
/home/intern/.conda/envs/local_llm_vllm/bin/python -m pytest surrogate_rollout/tests -q
```

Suite status after Stage 4.6: 208 passed (78 Phase 0–3 + 130 Phase 4).

---

## 1. Correct system interpretation

The prompt bank does **not** contain only mutually exclusive complete captioning prompts.

A prompt-bank entry is a reusable captioning instruction, behavior, or specialization that can be composed with other entries.

For each video segment:

1. the router selects zero or more prompt-bank entries;
2. the scaffold applier combines the selected entries with a fixed captioning contract;
3. the resulting composed prompt is sent to the captioner;
4. the generated caption is evaluated through the existing DVD pipeline.

The core inference path is:

```text
SegmentContext
        ↓
PromptRouter
        ↓
selected prompt-entry IDs
        ↓
PromptBank lookup
        ↓
ScaffoldApplier
        ↓
one composed captioning prompt
        ↓
Captioner
        ↓
existing DVD reasoning and rollout evaluation
```

The optimization path is separate:

```text
rollout and counterfactual evidence
        ↓
FeedbackGenerator
        ↓
FailureAttributor
        ↓
component-specific update proposals
        ├── PromptBankUpdateProposer
        ├── RouterUpdateProposer
        └── ScaffoldUpdateProposer
        ↓
MetaKnowledgeReviewer
        ↓
UpdateValidator
        ↓
accept / reject / revise / defer
        ↓
versioned component snapshots
```

`ScaffoldApplier` and `PromptBankUpdateProposer` are distinct components and must never be conflated.

---

## 2. Objective

Phase 4 adds a higher-level compositional prompt-routing system that:

1. maintains a versioned bank of reusable captioning instructions;
2. selects a subset of instructions for each video segment;
3. composes the selected instructions into one valid captioning prompt;
4. evaluates routed and composed prompts using the existing full and selective rollout infrastructure;
5. constructs structured counterfactual evidence from matched evaluations;
6. attributes failures to the prompt bank, router, scaffold, or multiple components;
7. proposes component-specific updates;
8. uses global meta-knowledge and held-out confirmation evidence to prevent sample-specific overfitting;
9. commits only validated, versioned component updates;
10. supports an ablation in which scaffold optimization is disabled or enabled through configuration.

The first milestone is not a large optimization experiment. It is a reproducible, inspectable, one-iteration implementation with explicit stop gates.

---

## 3. Component ownership and boundaries

### 3.1 Prompt bank

The prompt bank owns reusable captioning instructions.

Examples:

```text
Preserve the exact temporal order of actions.
Track visually similar entities consistently.
Describe meaningful state changes explicitly.
Avoid inferring identity without visual evidence.
Emphasize text that appears briefly on screen.
```

A prompt-bank entry should be:

- reusable across multiple segments;
- independently selectable;
- composable with other entries;
- versioned and traceable;
- narrower than the full captioning contract.

Prompt-bank entries must not silently duplicate the fixed output schema or required placeholders.

### 3.2 Prompt router

The router selects a subset of active prompt-bank entries for one segment.

The router owns:

- which entries are selected;
- selection scores or confidence;
- ordering or priority hints supplied to the scaffold;
- fallback behavior;
- selection provenance.

The router does **not** write the final captioning prompt.

### 3.3 Scaffold applier

The scaffold applier is an inference-time composer.

It receives:

- the segment context;
- the selected prompt-bank entries;
- a versioned scaffold policy;
- a fixed scaffold contract.

It returns one composed captioning prompt.

The scaffold applier owns:

- preserving selected instructions;
- removing redundant wording;
- resolving instruction conflicts;
- ordering and emphasizing instructions;
- combining global and specialized instructions;
- respecting prompt-length limits;
- producing one valid prompt under the fixed contract.

It does **not** update the prompt bank.

### 3.4 Prompt-bank update proposer

The prompt-bank update proposer is an optimization-time component.

It proposes operations such as:

- add a reusable prompt entry;
- revise an existing entry by creating a new version;
- merge redundant entries;
- retire an entry;
- attach or revise metadata and applicability traits.

It does not compose prompts for caption generation.

### 3.5 Router update proposer

The router update proposer changes the routing policy, rule set, or router parameters.

It does not modify prompt text or scaffold composition behavior.

### 3.6 Scaffold update proposer

The scaffold update proposer changes the **soft composition policy** used by the scaffold applier.

It must never modify the fixed hard contract.

Scaffold optimization is optional and controlled by configuration.

---

## 4. Optimization targets and ablation switch

Expose optimization targets through typed configuration and YAML.

At minimum:

```yaml
phase4:
  optimization:
    optimize_prompt_bank: true
    optimize_router: true
    optimize_scaffold: false
```

The default must be:

```yaml
optimize_scaffold: false
```

### 4.1 Behavior when `optimize_scaffold: false`

The system must:

- continue to execute the scaffold applier for every segment;
- use one frozen scaffold-policy version throughout the run;
- log scaffold-related failure attribution when detected;
- retain scaffold-update evidence for later analysis;
- skip `ScaffoldUpdateProposer`;
- record the skip reason as `disabled_by_config`;
- prohibit any scaffold-policy commit;
- keep all other optimization components functional.

### 4.2 Behavior when `optimize_scaffold: true`

The system may:

- generate scaffold-specific feedback;
- invoke `ScaffoldUpdateProposer`;
- create a provisional scaffold-policy version;
- validate the candidate scaffold on separate examples;
- commit a new scaffold-policy version only after acceptance;
- roll back or defer the update without modifying the incumbent policy.

### 4.3 Required scaffold ablation

The implementation must support two matched conditions:

```text
A. Fixed scaffold
   optimize_scaffold = false

B. Optimized scaffold
   optimize_scaffold = true
```

Hold constant where possible:

- initial prompt bank;
- initial router policy;
- initial scaffold policy;
- evidence batches;
- confirmation and regression batches;
- caption model and decoding;
- reasoning model and decoding;
- optimization budget;
- random seeds;
- evaluator configuration;
- update thresholds;
- prompt-bank and router optimization settings.

The intended experimental difference is whether scaffold-policy updates are proposed, validated, and committed.

Do not compare a deterministic fixed scaffold against an SLM scaffold while also changing unrelated components. Scaffold implementation and scaffold optimization are separate axes.

---

## 5. Replaceable policy boundaries

The following components must be independently replaceable.

### 5.1 Feedback generation

`FeedbackGenerator` converts rollout evidence into structured feedback.

The initial real implementation may use a large language model, but:

- evidence collection must not depend on a particular feedback model;
- downstream components must consume typed fields;
- raw model responses must never directly mutate state;
- prompts, settings, responses, parsing errors, and retries must be logged;
- another feedback policy must be usable without modifying rollout evaluation.

### 5.2 Prompt routing

`PromptRouter` selects multiple prompt entries.

Initial implementations may include:

- deterministic rules;
- a heuristic router;
- a large-language-model router.

A future SLM router must implement the same interface and output schema.

### 5.3 Scaffold application

`ScaffoldApplier` composes selected entries into a final captioning prompt.

Provide:

- a deterministic scaffold applier for tests and debugging;
- an initial real scaffold applier selected through configuration;
- an interface-compatible future SLM scaffold applier.

Replacing the scaffold implementation must not require changes to:

- the prompt bank;
- router output schema;
- caption-view adapter;
- evaluator;
- optimization loop.

### 5.4 Update proposal policies

The following must be independent:

- `PromptBankUpdateProposer`;
- `RouterUpdateProposer`;
- `ScaffoldUpdateProposer`.

Do not implement one monolithic optimizer that always rewrites all three components.

### 5.5 Update review and validation

A language-model reviewer may inspect global knowledge and update scope, but final acceptance must depend on structured validation results.

The reviewer and empirical validator must remain separate.

---

## 6. Non-goals for the first implementation

Do not implement the following in the initial pass:

- a large multi-iteration optimization run;
- online SLM training;
- end-to-end joint gradient training;
- a replacement DVD reasoning loop;
- a replacement caption cache;
- a replacement full or selective evaluator;
- automatic global update after one sample;
- unconstrained free-form scaffold rewriting;
- modification of output schemas or required placeholders;
- concurrent workers mutating the same component snapshots;
- full GEPA integration;
- query-conditioned routing unless separately enabled;
- simultaneous implementation of every stage below.

The initial milestone ends after:

1. a routed-caption smoke test;
2. an offline component-wise update dry run;
3. one fixed-scaffold optimization iteration;
4. one separately reviewed scaffold-enabled iteration.

---

## 7. Frozen Phase 0–3 boundary

Treat the following as existing services whenever possible:

- `evaluation/rollout_evaluator.py`;
- `evaluation/full_rollout.py`;
- `evaluation/selective_rollout.py`;
- `evaluation/dvd_qa.py`;
- `captioning/candidate_captions.py`;
- `mixed_views/builder.py`;
- `selection/`;
- `references/`;
- `retrieval/`;
- existing cache and run-artifact code.

Do not duplicate or broadly refactor:

- DVD reasoning;
- candidate caption generation;
- full rollout evaluation;
- selective rollout evaluation;
- mixed-view construction;
- caption-cache keying;
- reference extraction;
- selection-budget enforcement.

Any Phase 0–3 modification must be:

- minimal;
- documented;
- backward compatible;
- covered by focused tests;
- followed by the complete existing test suite.

---

## 8. Suggested repository layout

Adapt names to the repository where needed, but preserve component separation.

```text
prompt_routing/
    __init__.py
    schemas.py
    prompt_bank.py
    router_policy.py
    scaffold_contract.py
    scaffold_policy.py
    persistence.py
    validators.py

    router.py
    scaffold_applier.py

    policies/
        __init__.py
        rule_based_router.py
        deterministic_scaffold.py
        llm_scaffold.py
        slm_router.py                 # interface-compatible stub initially
        slm_scaffold.py               # interface-compatible stub initially

optimization/
    __init__.py
    schemas.py
    evidence_builder.py
    feedback_generator.py
    failure_attributor.py
    meta_knowledge.py

    prompt_bank_update_proposer.py
    router_update_proposer.py
    scaffold_update_proposer.py

    update_reviewer.py
    update_validator.py
    component_commit.py
    optimization_loop.py

    policies/
        __init__.py
        mock_feedback.py
        llm_feedback.py
        deterministic_bank_update.py
        deterministic_router_update.py
        deterministic_scaffold_update.py

scripts/
    inspect_phase4_integration.py
    run_routing_dry_run.py
    build_counterfactual_evidence.py
    run_component_update_dry_run.py
    run_routed_caption_smoke_test.py
    run_phase4_single_iteration.py
    compare_scaffold_ablation.py

tests/
    test_phase4_schemas.py
    test_prompt_bank.py
    test_router_policy.py
    test_scaffold_policy.py
    test_prompt_router.py
    test_scaffold_applier.py
    test_composed_prompt_validation.py
    test_routed_caption_adapter.py
    test_evidence_builder.py
    test_feedback_generator.py
    test_failure_attributor.py
    test_meta_knowledge.py
    test_update_proposers.py
    test_update_reviewer.py
    test_update_validator.py
    test_component_commit.py
    test_phase4_offline_pipeline.py
```

Do not create duplicated records in both schema modules. Routing-owned records belong in `prompt_routing/schemas.py`; optimization-owned records belong in `optimization/schemas.py`.

---

## 9. Core terminology

Use these terms consistently.

- `prompt_entry`: one reusable, composable captioning instruction.
- `prompt_bank`: a versioned snapshot of prompt entries.
- `router_policy`: a versioned policy that selects a subset of entries.
- `scaffold_contract`: immutable hard requirements for a valid captioning prompt.
- `scaffold_policy`: versioned soft composition behavior.
- `selected_prompt_ids`: prompt entries chosen for one segment.
- `composed_caption_prompt`: final prompt produced by the scaffold applier.
- `routing_decision`: structured selection output for one segment.
- `composition_trace`: structured evidence showing how selected entries became the final prompt.
- `counterfactual_evidence`: matched comparison between baseline and candidate behavior.
- `failure_attribution`: structured assignment of observed failure to one or more components.
- `meta_knowledge`: cross-example, cross-iteration knowledge used to review update scope and conflicts.
- `update_proposal`: uncommitted component-specific change.
- `confirmation_examples`: examples not used to propose the update.
- `regression_examples`: examples where incumbent behavior should be preserved.
- `component_snapshot`: immutable prompt-bank, router-policy, or scaffold-policy version.
- `optimization_iteration`: one frozen input state, evidence set, feedback set, proposals, validations, and optional commits.

---

## 10. Required typed records

Use dataclasses, Pydantic models, or equivalent typed records.

Exact names may follow repository conventions, but the same information must be retained.

### 10.1 Prompt-bank records

```python
@dataclass(frozen=True)
class PromptEntry:
    prompt_id: str
    prompt_text: str
    prompt_hash: str
    name: str
    description: str
    tags: tuple[str, ...]
    applicability_traits: Mapping[str, Any]
    conflicts_with: tuple[str, ...]
    status: Literal["active", "retired"]
    parent_prompt_ids: tuple[str, ...]
    created_by: str
    created_at: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class PromptBankSnapshot:
    bank_version: str
    parent_bank_version: str | None
    entries: tuple[PromptEntry, ...]
    default_entry_ids: tuple[str, ...]
    max_selected_entries: int
    created_at: str
    created_by: str
    provenance: Mapping[str, Any]
```

Prompt entries are composable instructions, not necessarily standalone complete prompts.

### 10.2 Router records

```python
@dataclass(frozen=True)
class RoutingRule:
    rule_id: str
    priority: int
    conditions: Mapping[str, Any]
    target_prompt_ids: tuple[str, ...]
    enabled: bool
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class RouterPolicySnapshot:
    router_version: str
    parent_router_version: str | None
    policy_type: str
    rules: tuple[RoutingRule, ...]
    fallback_prompt_ids: tuple[str, ...]
    max_selected_entries: int
    configuration: Mapping[str, Any]
    created_at: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class SegmentContext:
    video_id: str
    segment_id: str
    timestamp_start: float | None
    timestamp_end: float | None
    segment_features: Mapping[str, Any]
    history_summary: str | None
    question: str | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RoutingDecision:
    video_id: str
    segment_id: str
    bank_version: str
    router_version: str
    selected_prompt_ids: tuple[str, ...]
    prompt_scores: Mapping[str, float]
    matched_rule_ids: tuple[str, ...]
    selection_order: tuple[str, ...]
    confidence: float | None
    used_fallback: bool
    fallback_reason: str | None
    decision_payload: Mapping[str, Any]
```

The router may select multiple entries. It must not return only unstructured prose.

### 10.3 Scaffold records

```python
@dataclass(frozen=True)
class ScaffoldContract:
    contract_version: str
    required_placeholders: tuple[str, ...]
    required_sections: tuple[str, ...]
    output_schema: Mapping[str, Any]
    forbidden_patterns: tuple[str, ...]
    max_prompt_tokens: int | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ScaffoldPolicySnapshot:
    scaffold_version: str
    parent_scaffold_version: str | None
    policy_type: str
    composition_instruction: str
    ordering_policy: str
    conflict_resolution_rules: tuple[str, ...]
    compression_policy: str
    configuration: Mapping[str, Any]
    created_at: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class CompositionTrace:
    selected_prompt_ids: tuple[str, ...]
    preserved_prompt_ids: tuple[str, ...]
    omitted_prompt_ids: tuple[str, ...]
    conflicts_detected: tuple[str, ...]
    conflict_resolutions: tuple[str, ...]
    ordering_decisions: tuple[str, ...]
    compression_actions: tuple[str, ...]
    raw_policy_output_artifact: str | None


@dataclass(frozen=True)
class ComposedCaptionPrompt:
    video_id: str
    segment_id: str
    bank_version: str
    router_version: str
    scaffold_version: str
    contract_version: str
    selected_prompt_ids: tuple[str, ...]
    prompt_text: str
    prompt_hash: str
    composition_trace: CompositionTrace
    validation_errors: tuple[str, ...]
    is_valid: bool
```

### 10.4 Evidence records

```python
@dataclass(frozen=True)
class CaptionDifference:
    segment_id: str
    baseline_caption: str
    candidate_caption: str
    difference_summary: str | None
    structured_differences: Mapping[str, Any]


@dataclass(frozen=True)
class CounterfactualEvidence:
    evidence_id: str
    video_id: str
    question_id: str
    ground_truth: str | None

    baseline_bank_version: str
    baseline_router_version: str
    baseline_scaffold_version: str
    candidate_bank_version: str
    candidate_router_version: str
    candidate_scaffold_version: str

    baseline_selected_prompt_ids: Mapping[str, tuple[str, ...]]
    candidate_selected_prompt_ids: Mapping[str, tuple[str, ...]]
    baseline_composed_prompt_refs: Mapping[str, str]
    candidate_composed_prompt_refs: Mapping[str, str]

    baseline_answer: str | None
    candidate_answer: str | None
    baseline_score: float
    candidate_score: float
    score_delta: float

    outcome_transition: Literal[
        "correct_to_correct",
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_wrong",
        "unknown",
    ]

    segment_ids: tuple[str, ...]
    caption_differences: tuple[CaptionDifference, ...]
    baseline_trajectory_refs: tuple[str, ...]
    candidate_trajectory_refs: tuple[str, ...]
    rollout_artifact_refs: Mapping[str, str]
    metadata: Mapping[str, Any]
```

Retain beneficial, harmful, and neutral changes.

### 10.5 Failure attribution and feedback

```python
@dataclass(frozen=True)
class FailureAttribution:
    attribution_id: str
    evidence_ids: tuple[str, ...]
    targets: tuple[
        Literal[
            "prompt_bank",
            "router",
            "scaffold",
            "multiple",
            "insufficient_evidence",
        ],
        ...,
    ]
    prompt_bank_issues: tuple[str, ...]
    router_issues: tuple[str, ...]
    scaffold_issues: tuple[str, ...]
    supporting_facts: tuple[str, ...]
    confidence: float
    rationale: str


@dataclass(frozen=True)
class FeedbackItem:
    feedback_id: str
    evidence_ids: tuple[str, ...]
    attribution_id: str
    target_components: tuple[str, ...]
    failure_modes: tuple[str, ...]
    successful_behaviors: tuple[str, ...]
    desired_behaviors: tuple[str, ...]
    avoid_behaviors: tuple[str, ...]
    applicable_segment_traits: Mapping[str, Any]
    confidence: float
    rationale: str


@dataclass(frozen=True)
class FeedbackBatch:
    feedback_policy: str
    feedback_policy_version: str
    input_evidence_ids: tuple[str, ...]
    attributions: tuple[FailureAttribution, ...]
    items: tuple[FeedbackItem, ...]
    raw_response_artifact: str | None
    parse_errors: tuple[str, ...]
```

Downstream code must not parse `rationale` to determine operations.

### 10.6 Meta-knowledge records

```python
@dataclass(frozen=True)
class MetaKnowledgeItem:
    knowledge_id: str
    knowledge_type: Literal[
        "successful_behavior",
        "failure_pattern",
        "routing_pattern",
        "composition_pattern",
        "rejected_update",
        "accepted_update",
        "conflict",
    ]
    condition: Mapping[str, Any]
    principle: str
    positive_support_ids: tuple[str, ...]
    negative_support_ids: tuple[str, ...]
    distinct_video_ids: tuple[str, ...]
    scope: Literal[
        "local_prompt",
        "routing",
        "global_scaffold",
        "meta_only",
    ]
    confidence: float
    status: Literal["candidate", "confirmed", "rejected", "deprecated"]
    provenance: Mapping[str, Any]
```

Meta-knowledge stores abstractions and provenance, not an uncontrolled dump of complete trajectories.

### 10.7 Component-specific update proposals

```python
@dataclass(frozen=True)
class PromptBankOperation:
    operation_id: str
    operation_type: Literal[
        "add_entry",
        "revise_entry",
        "merge_entries",
        "retire_entry",
        "no_op",
    ]
    target_ids: tuple[str, ...]
    payload: Mapping[str, Any]
    source_feedback_ids: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class PromptBankUpdateProposal:
    proposal_id: str
    input_bank_version: str
    operations: tuple[PromptBankOperation, ...]
    validation_errors: tuple[str, ...]
    is_valid: bool
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class RouterUpdateProposal:
    proposal_id: str
    input_router_version: str
    operations: tuple[Mapping[str, Any], ...]
    source_feedback_ids: tuple[str, ...]
    validation_errors: tuple[str, ...]
    is_valid: bool
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ScaffoldUpdateProposal:
    proposal_id: str
    input_scaffold_version: str
    candidate_policy: ScaffoldPolicySnapshot | None
    source_feedback_ids: tuple[str, ...]
    validation_errors: tuple[str, ...]
    is_valid: bool
    skipped_reason: Literal[
        "disabled_by_config",
        "no_scaffold_attribution",
        "insufficient_support",
        "none",
    ]
    provenance: Mapping[str, Any]
```

### 10.8 Review and validation records

```python
@dataclass(frozen=True)
class UpdateReview:
    review_id: str
    component: Literal["prompt_bank", "router", "scaffold"]
    proposal_id: str
    decision: Literal["accept_for_validation", "revise", "reject", "defer"]
    recommended_scope: Literal[
        "local_prompt",
        "routing",
        "global_scaffold",
        "meta_only",
    ]
    conflicts: tuple[str, ...]
    required_confirmations: tuple[str, ...]
    supporting_knowledge_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class ComponentValidationResult:
    validation_id: str
    component: Literal["prompt_bank", "router", "scaffold"]
    incumbent_version: str
    candidate_version: str | None
    evidence_example_ids: tuple[str, ...]
    confirmation_example_ids: tuple[str, ...]
    regression_example_ids: tuple[str, ...]
    incumbent_metrics: Mapping[str, float]
    candidate_metrics: Mapping[str, float]
    regressions: tuple[str, ...]
    decision: Literal["accept", "reject", "defer", "evaluation_failed"]
    reasons: tuple[str, ...]
```

### 10.9 Iteration record

```python
@dataclass(frozen=True)
class OptimizationIterationResult:
    iteration_id: str

    input_bank_version: str
    input_router_version: str
    input_scaffold_version: str

    optimize_prompt_bank: bool
    optimize_router: bool
    optimize_scaffold: bool

    evidence_artifact: str
    feedback_artifact: str
    meta_knowledge_artifact: str

    bank_proposal_artifact: str | None
    router_proposal_artifact: str | None
    scaffold_proposal_artifact: str | None

    review_artifacts: Mapping[str, str]
    validation_artifacts: Mapping[str, str]

    committed_bank_version: str | None
    committed_router_version: str | None
    committed_scaffold_version: str | None

    status: Literal[
        "dry_run",
        "partially_committed",
        "committed",
        "all_rejected",
        "evaluation_failed",
    ]
    errors: tuple[str, ...]
```

---

## 11. Required public interfaces

### 11.1 Prompt router

```python
class PromptRouter(Protocol):
    def route(
        self,
        context: SegmentContext,
        prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
    ) -> RoutingDecision:
        ...
```

The router selects zero or more prompt entries up to a configured maximum.

### 11.2 Scaffold applier

```python
class ScaffoldApplier(Protocol):
    def apply(
        self,
        *,
        context: SegmentContext,
        selected_entries: Sequence[PromptEntry],
        routing_decision: RoutingDecision,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
    ) -> ComposedCaptionPrompt:
        ...
```

The scaffold applier must preserve traceability from selected entries to final text.

Provide at least:

- `DeterministicScaffoldApplier` for tests;
- the selected initial real implementation;
- `SLMScaffoldApplier` as an interface-compatible stub until implemented.

### 11.3 Evidence builder

```python
class CounterfactualEvidenceBuilder(Protocol):
    def build(
        self,
        baseline_results: Sequence[EvaluationResult],
        candidate_results: Sequence[EvaluationResult],
        *,
        component_snapshots: Mapping[str, Any],
    ) -> Sequence[CounterfactualEvidence]:
        ...
```

The builder must consume saved or in-memory results. It must not launch captioning, reasoning, or feedback calls.

### 11.4 Feedback generator

```python
class FeedbackGenerator(Protocol):
    def generate(
        self,
        evidence: Sequence[CounterfactualEvidence],
        meta_knowledge: Sequence[MetaKnowledgeItem],
    ) -> FeedbackBatch:
        ...
```

Provide deterministic fixtures and one isolated real policy implementation.

### 11.5 Failure attributor

```python
class FailureAttributor(Protocol):
    def attribute(
        self,
        evidence: Sequence[CounterfactualEvidence],
        feedback: FeedbackBatch,
    ) -> Sequence[FailureAttribution]:
        ...
```

Attribution may be produced jointly with feedback internally, but the external typed record must be preserved.

### 11.6 Component update proposers

```python
class PromptBankUpdateProposer(Protocol):
    def propose(
        self,
        *,
        prompt_bank: PromptBankSnapshot,
        feedback: FeedbackBatch,
        meta_knowledge: Sequence[MetaKnowledgeItem],
    ) -> PromptBankUpdateProposal:
        ...


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
```

When `enabled=False`, `ScaffoldUpdateProposer` must not call a model and must return a typed skipped proposal with `skipped_reason="disabled_by_config"`.

### 11.7 Meta-knowledge reviewer

```python
class UpdateReviewer(Protocol):
    def review(
        self,
        *,
        proposal: Any,
        component: Literal["prompt_bank", "router", "scaffold"],
        meta_knowledge: Sequence[MetaKnowledgeItem],
    ) -> UpdateReview:
        ...
```

The reviewer decides whether a proposal should proceed to empirical validation. It does not commit updates.

### 11.8 Empirical validator

```python
class UpdateValidator(Protocol):
    def validate(
        self,
        *,
        incumbent_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
        component: Literal["prompt_bank", "router", "scaffold"],
        confirmation_examples: Sequence[QAExample],
        regression_examples: Sequence[QAExample],
    ) -> ComponentValidationResult:
        ...
```

The final acceptance decision must be based on structured metrics and configured thresholds.

### 11.9 Optimization loop

```python
class PromptRoutingOptimizationLoop:
    def run_iteration(
        self,
        *,
        input_bank: PromptBankSnapshot,
        input_router: RouterPolicySnapshot,
        input_scaffold: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        evidence_results: Sequence[EvaluationResult],
        confirmation_examples: Sequence[QAExample],
        regression_examples: Sequence[QAExample],
        config: Phase4Config,
        commit: bool,
    ) -> OptimizationIterationResult:
        ...
```

The loop orchestrates components. It must not contain policy-specific prompts, parsing logic, routing rules, or direct model calls.

---

## 12. Scaffold contract versus scaffold policy

This distinction is mandatory.

### 12.1 Hard scaffold contract

The following are fixed and not optimization targets:

- required input placeholders;
- video/frame input bindings;
- required timestamp fields;
- required subject-registry fields;
- output JSON schema;
- parser compatibility;
- forbidden output patterns;
- required top-level prompt sections;
- hard prompt-length ceiling.

A candidate scaffold that violates the contract must be rejected before expensive evaluation.

### 12.2 Soft scaffold policy

The following may be optimized when `optimize_scaffold=true`:

- ordering of selected instructions;
- grouping of related instructions;
- redundancy removal;
- conflict resolution;
- compression or summarization behavior;
- emphasis allocation;
- placement of global versus specialized instructions;
- context-dependent wording;
- how multiple selected entries are fused into one coherent prompt.

Scaffold optimization must never modify the hard contract.

### 12.3 Scaffold attribution rule

A failure may be attributed to the scaffold only when there is evidence that:

1. appropriate prompt-bank entries existed;
2. the router selected appropriate entries;
3. the composed prompt omitted, weakened, contradicted, or distorted them;
4. the observed failure is not better explained only by caption-model stochasticity or insufficient visual evidence.

A single uncertain example should normally produce `defer`, not a global scaffold update.

---

## 13. Routing requirements

The router must support multiple selected entries.

The initial rule-based router should:

1. evaluate enabled rules in deterministic order;
2. collect all matching target entries;
3. deduplicate selected entries;
4. respect conflict metadata;
5. enforce `max_selected_entries`;
6. use a deterministic tie-breaker;
7. append configured fallback entries when needed;
8. validate that every entry is active;
9. output selection order and scores;
10. avoid question-conditioned routing by default.

Question-conditioned routing must be separately configurable:

```yaml
router:
  question_conditioned: false
```

The default should remain query-agnostic to avoid generating different captions for every downstream question unless the experiment explicitly studies that setting.

---

## 14. Routed caption integration

Phase 4 requires different composed prompts for different segments.

Add the smallest adapter that accepts:

```python
segment_composed_prompt_map: Mapping[str, ComposedCaptionPrompt]
```

The adapter must:

1. reuse existing caption generation;
2. group segments by composed prompt hash when batching is possible;
3. key every caption by the actual composed prompt hash;
4. retain bank, router, scaffold, and contract versions;
5. preserve segment order;
6. log routing decisions and composition traces;
7. prevent routed views from being mislabeled as single-prompt caches;
8. preserve existing full and selective rollout semantics.

Use distinct terminology:

- `single_prompt_candidate_view`: all segments use one complete candidate prompt;
- `routed_prompt_view`: segments use composed prompts produced from bank entries;
- `surrogate_mixed_view`: only selected segments use candidate captions and other captions are reused.

These artifact types must not be conflated.

---

## 15. Counterfactual evidence and component attribution

Evidence must allow diagnosis of three distinct failure sources.

### 15.1 Prompt-bank failure

Examples:

- the required reusable instruction does not exist;
- an entry is underspecified or misleading;
- two entries are redundant or internally contradictory;
- an entry is too sample-specific to be reusable.

### 15.2 Router failure

Examples:

- the correct entry exists but is not selected;
- an irrelevant entry is selected;
- too many entries are selected;
- a conflicting combination is selected;
- selection conditions are too broad or too narrow.

### 15.3 Scaffold failure

Examples:

- selected entries disappear from the final prompt;
- one selected entry dominates and suppresses another;
- the composed prompt introduces a contradiction;
- composition becomes excessively verbose;
- the fixed output contract is weakened or obscured;
- instruction ordering causes the model to ignore important behavior.

### 15.4 Multiple-component failure

Use `multiple` only when the evidence genuinely supports more than one component. Do not use it as a default escape hatch.

### 15.5 Insufficient evidence

Use `insufficient_evidence` when:

- the causal component cannot be isolated;
- caption differences are not informative;
- the rollout failed;
- model stochasticity is a plausible explanation;
- only one weak sample supports a global update.

These cases may be added to meta-knowledge as candidates but must not be committed automatically.

---

## 16. Meta-knowledge and overfitting control

Global meta-knowledge is used as an update reviewer, not as the final performance judge.

Store:

- recurring successful behaviors;
- recurring failure patterns;
- known routing conditions;
- composition patterns;
- accepted update outcomes;
- rejected update outcomes;
- known conflicts;
- number of distinct supporting videos;
- contradictory evidence.

Do not store only the latest sample.

### 16.1 Scope recommendation

The reviewer should recommend one of:

```text
local_prompt
routing
global_scaffold
meta_only
```

Guidance:

- one or two sample-specific observations should remain `local_prompt` or `meta_only`;
- routing failures should not trigger prompt-text changes by default;
- global scaffold changes require repeated support across distinct videos;
- contradictory evidence should usually produce `defer` or `reject`.

### 16.2 Proposal examples and validation examples must be separated

The examples used to create an update must not be reused as the only confirmation evidence.

Prefer video-level separation:

```text
Evidence batch:
  used to discover and propose the update

Confirmation batch:
  different train videos used to test generalization

Regression batch:
  different examples where incumbent behavior should remain correct
```

Questions from the same video are not considered fully independent confirmation.

### 16.3 Default acceptance principles

Make thresholds configurable, but the initial validator should enforce:

1. the target failure improves on the evidence or confirmation set;
2. confirmation aggregate score does not degrade beyond tolerance;
3. correct-to-wrong regressions remain below a configured limit;
4. global scaffold updates require support from multiple distinct videos;
5. invalid prompts or contracts are rejected before rollout;
6. inconclusive candidates are deferred rather than promoted.

---

## 17. Configuration

Use typed configuration objects and serializable configuration files.

Minimum example:

```yaml
phase4:
  seed: 0
  dry_run: true
  commit: false

  optimization:
    optimize_prompt_bank: true
    optimize_router: true
    optimize_scaffold: false

    max_iterations: 1
    max_new_entries_per_iteration: 2
    max_router_operations_per_iteration: 4
    min_feedback_confidence: 0.70

  prompt_bank:
    path: artifacts/prompt_banks/current.json
    max_active_entries: 12
    max_selected_entries: 4

  router:
    policy: rule_based
    policy_path: artifacts/router_policies/current.json
    question_conditioned: false
    fallback_prompt_ids: []
    deterministic_tie_break: prompt_id

  scaffold:
    applier: llm
    policy_path: artifacts/scaffold_policies/current.json
    contract_path: configs/scaffold_contract.json

    optimize: ${phase4.optimization.optimize_scaffold}
    update_frequency: 1
    min_distinct_videos_for_update: 3
    require_confirmation_batch: true
    require_regression_batch: true
    require_full_validation_for_commit: false

  evidence:
    include_transitions:
      - wrong_to_correct
      - correct_to_wrong
      - wrong_to_wrong
      - correct_to_correct
    max_caption_differences_per_example: 20

  feedback:
    policy: llm
    model: null
    prompt_template_path: configs/feedback_prompt.txt
    max_retries: 1

  meta_knowledge:
    path: artifacts/meta_knowledge/current.json
    min_support_for_confirmed: 2

  validation:
    rollout_mode: surrogate
    full_rollout_for_global_scaffold: true
    max_accuracy_drop: 0.0
    max_correct_to_wrong_regressions: 0

  evaluation:
    final_evaluation_rollout_mode: full
```

Do not hide `optimize_scaffold` only inside a scaffold module. It must be visible in the top-level run configuration and copied into every run artifact.

---

## 18. Run artifacts

Every Phase 4 run must create a self-contained directory.

Suggested layout:

```text
runs/phase4_<timestamp>/
    run_config.yaml
    environment.json
    git_commit.txt

    input_state/
        prompt_bank.json
        router_policy.json
        scaffold_policy.json
        scaffold_contract.json
        meta_knowledge.json

    routing/
        decisions.jsonl

    composition/
        composed_prompts.jsonl
        traces.jsonl
        raw_policy_outputs/

    evidence/
        counterfactual_evidence.jsonl
        evidence_summary.json

    feedback/
        request.json
        raw_response.txt
        parsed_feedback.json

    attribution/
        failure_attributions.jsonl

    proposals/
        prompt_bank_proposal.json
        router_proposal.json
        scaffold_proposal.json

    reviews/
        prompt_bank_review.json
        router_review.json
        scaffold_review.json

    validation/
        prompt_bank_validation.json
        router_validation.json
        scaffold_validation.json

    output_state/
        preview_prompt_bank.json
        preview_router_policy.json
        preview_scaffold_policy.json
        committed_prompt_bank.json
        committed_router_policy.json
        committed_scaffold_policy.json

    evaluation/
        incumbent_results.jsonl
        candidate_results.jsonl
        comparison.json

    iteration_result.json
```

When scaffold optimization is disabled, still save:

```text
proposals/scaffold_proposal.json
```

with:

```json
{
  "skipped_reason": "disabled_by_config"
}
```

Log at minimum:

- all component versions;
- `optimize_prompt_bank`;
- `optimize_router`;
- `optimize_scaffold`;
- selected prompt IDs per segment;
- composed prompt text and hash;
- composition trace;
- actual caption cache key;
- evidence-to-feedback lineage;
- feedback-to-proposal lineage;
- reviewer knowledge references;
- confirmation and regression example IDs;
- acceptance or rejection reason;
- token use, timing, and model settings;
- fallbacks and errors.

---

## 19. Incremental implementation sequence

## Critical rule

**Do not implement all stages in one task.**

After each stage:

1. run the required focused tests;
2. run the existing Phase 0–3 tests when code paths overlap;
3. produce the checkpoint report in Section 20;
4. stop editing;
5. wait for human review before starting the next stage.

Do not preemptively implement the next stage.

Do not combine checkpoints unless explicitly requested.

---

### Stage 4.0: Repository inspection and corrected integration map

**Code changes:** none, except an optional inspection script or notes artifact.

Inspect:

1. `PHASE2_3_SURROGATE.md`;
2. existing evaluation schemas;
3. prompt entry points in caption generation;
4. caption cache key construction;
5. segment materialization order;
6. existing prompt placeholders and output contracts;
7. whether current captioning supports per-call prompts;
8. the smallest adapter point for per-segment composed prompts;
9. existing run-artifact schemas;
10. conflicts with the proposed package layout.

Required output:

- exact current execution flow;
- exact integration points;
- exact files likely to change;
- discrepancies between documentation and code;
- proposed Stage 4.1 file list.

**Stop gate:** Do not create schemas or packages until reviewed.

---

### Stage 4.1: Foundational schemas only

Implement only typed records for:

- prompt entries and prompt-bank snapshots;
- router-policy snapshots and routing decisions;
- scaffold contracts and scaffold-policy snapshots;
- composition traces and composed prompts;
- top-level Phase 4 configuration;
- component version identifiers.

Do not implement:

- persistence;
- routing logic;
- scaffold model calls;
- caption generation;
- evidence;
- feedback;
- optimization.

Tests:

1. schema validation;
2. immutable records;
3. deterministic serialization;
4. multiple selected prompt IDs;
5. invalid confidence ranges;
6. invalid component versions;
7. invalid composed-prompt state;
8. `optimize_scaffold` defaulting to `false`;
9. configuration serialization preserving all optimization flags.

**Checkpoint artifact:** example JSON for every foundational record.

**Stop gate:** Review schemas before persistence or policies.

---

### Stage 4.2: Component persistence and invariants

Implement:

- prompt-bank load/save;
- router-policy load/save;
- scaffold-policy load/save;
- scaffold-contract load/save;
- atomic versioned snapshots;
- validation and dry-run preview.

Do not implement routing or composition.

Tests:

1. deterministic round trip;
2. immutable snapshots;
3. atomic writes;
4. invalid active-entry references;
5. retired-entry handling;
6. scaffold policy cannot modify contract;
7. failed write leaves incumbent unchanged;
8. preview does not alter canonical state;
9. parent-version provenance;
10. duplicate active prompt hashes.

**Checkpoint artifact:** one initial bank, router policy, scaffold policy, and contract.

**Stop gate:** Do not implement inference policies until snapshots are inspected.

---

### Stage 4.3: Multi-entry rule-based router

Implement:

- `PromptRouter` protocol;
- `RuleBasedPromptRouter`;
- multiple-entry selection;
- priorities and deterministic tie-breaking;
- conflict checks;
- fallback behavior;
- routing-decision logging.

Use synthetic segment contexts.

Tests:

1. zero matching rules;
2. one matching rule;
3. multiple matching rules;
4. duplicate target deduplication;
5. conflict handling;
6. maximum selected-entry enforcement;
7. disabled rules;
8. retired target rejection;
9. deterministic selection order;
10. query-agnostic default behavior.

**Checkpoint artifact:** routing decisions over synthetic segments.

**Stop gate:** Do not implement the scaffold yet.

---

### Stage 4.4: Fixed scaffold application

Implement:

- `ScaffoldApplier` protocol;
- `DeterministicScaffoldApplier` for tests;
- the selected initial real scaffold applier behind configuration;
- composed-prompt validation;
- composition-trace logging;
- SLM scaffold interface-compatible stub only.

At this stage, the scaffold policy is frozen. Do not implement scaffold updates.

Tests:

1. zero selected entries with fallback;
2. one selected entry;
3. multiple selected entries;
4. selected-instruction preservation;
5. redundancy removal;
6. deterministic ordering;
7. conflict resolution;
8. contract preservation;
9. prompt-length enforcement;
10. invalid raw model output;
11. trace completeness;
12. SLM stub fails clearly without side effects.

**Checkpoint artifact:** composed prompts and traces for Stage 4.3 decisions.

**Stop gate:** Manually inspect whether selected entries are preserved before caption integration.

---

### Stage 4.5: Offline routing-to-composition dry run

Connect only:

```text
saved SegmentContext
→ PromptRouter
→ selected entries
→ ScaffoldApplier
→ ComposedCaptionPrompt
```

Requirements:

- no captioning calls;
- no DVD reasoning calls;
- no optimization;
- one documented CLI;
- deterministic fixture mode;
- complete artifacts.

Tests:

1. successful offline run;
2. invalid routing decision rejection;
3. invalid composed prompt rejection;
4. reproducible rerun;
5. complete manifest;
6. `optimize_scaffold` has no effect on inference output when no update stage runs.

**Stop gate:** Review outputs before touching caption generation.

---

### Stage 4.6: Routed caption-view adapter

Implement the smallest adapter accepting:

```python
segment_composed_prompt_map: Mapping[str, ComposedCaptionPrompt]
```

Requirements:

- reuse existing caption generation;
- group identical prompt hashes where possible;
- preserve cache isolation;
- retain all component versions;
- create a distinct routed-view artifact;
- preserve existing full and surrogate semantics.

Tests:

1. all segments using the same composed prompt matches existing single-prompt behavior;
2. multiple prompt hashes across segments;
3. prompt-hash cache separation;
4. segment-order preservation;
5. no incumbent-cache contamination;
6. routed view not labeled as full single-prompt cache;
7. missing composed prompt handling;
8. invalid prompt contract handling.

**Checkpoint demonstration:** one tiny routed caption view.

**Stop gate:** Do not run a complete DVD evaluation yet.

---

### Stage 4.7: Routed evaluator smoke test

Run:

- one frozen bank;
- one frozen router;
- one frozen scaffold;
- one small video;
- one to three QAs;
- existing DVD reasoning.

Compare:

1. default/fallback composition;
2. routed multi-entry composition.

Record:

- routing decisions;
- selected entries;
- composition traces;
- composed prompt hashes;
- caption cache keys;
- predictions and scores;
- cost and timing.

This is only an integration test.

**Stop gate:** Do not implement feedback or updates until the inference path is manually verified.

---

### Stage 4.8: Offline counterfactual evidence builder

Implement evidence construction from saved Stage 4.7 or Phase 2–3 artifacts.

Requirements:

- no captioning calls;
- no reasoning calls;
- no feedback calls;
- stable ID matching;
- deterministic caption differences;
- all outcome transitions retained;
- component versions and selection/composition traces retained.

Tests:

1. matched pairing;
2. configuration mismatch rejection;
3. all correctness transitions;
4. missing trace handling;
5. fallback handling;
6. deterministic evidence IDs;
7. selected-entry differences;
8. composed-prompt differences;
9. serialization round trip.

**Checkpoint artifact:** a small evidence JSONL.

**Stop gate:** Manually inspect evidence before feedback implementation.

---

### Stage 4.9: Feedback, attribution, and meta-knowledge boundary

Implement:

- `FeedbackGenerator` protocol;
- `MockFeedbackGenerator`;
- isolated real feedback interface;
- `FailureAttributor`;
- `MetaKnowledgeStore`;
- strict validation and artifact logging.

The real model call may remain disabled for this checkpoint.

Tests:

1. deterministic mock output;
2. prompt-bank attribution;
3. router attribution;
4. scaffold attribution;
5. multiple attribution;
6. insufficient-evidence attribution;
7. missing evidence references;
8. invalid confidence;
9. malformed model output;
10. meta-knowledge deduplication;
11. contradictory support handling;
12. raw-response preservation.

**Checkpoint demonstration:** Stage 4.8 evidence through mock feedback and attribution.

**Stop gate:** Review attribution quality before update proposers.

---

### Stage 4.10: Component update proposers

Implement separately:

- `PromptBankUpdateProposer`;
- `RouterUpdateProposer`;
- `ScaffoldUpdateProposer`.

At this stage proposals are preview-only.

Required scaffold behavior:

```text
optimize_scaffold = false
→ no model call
→ typed skipped proposal
→ skipped_reason = disabled_by_config

optimize_scaffold = true
→ scaffold proposal may be generated
→ no commit yet
```

Tests:

1. bank add-entry proposal;
2. bank revise-entry proposal;
3. router-rule proposal;
4. scaffold proposal when enabled;
5. typed scaffold skip when disabled;
6. hard-contract modification rejection;
7. insufficient support;
8. low-confidence handling;
9. deterministic mock proposals;
10. no input-state mutation;
11. no cross-component operation leakage.

**Checkpoint artifact:** three proposal files for both scaffold-disabled and scaffold-enabled configurations.

**Stop gate:** Do not commit or evaluate proposals yet.

---

### Stage 4.11: Meta-knowledge review and empirical validator

Implement:

- `UpdateReviewer`;
- confirmation/regression split validation;
- accept/reject/revise/defer;
- component-specific thresholds;
- candidate-state preview;
- rollback-safe validation artifacts.

Requirements:

- proposal examples and confirmation examples must be separate;
- video-level separation should be enforced when possible;
- global scaffold updates require stricter support;
- scaffold contract remains fixed;
- no canonical commit.

Tests:

1. accept-for-validation;
2. reviewer rejection due to known conflict;
3. defer due to one weak sample;
4. confirmation improvement;
5. regression rejection;
6. scaffold update rejected for insufficient distinct videos;
7. scaffold update rejected for contract violation;
8. evaluation failure;
9. candidate preview leaves incumbent untouched;
10. deterministic decision under fixed metrics.

**Checkpoint artifact:** review and validation reports for synthetic proposals.

**Stop gate:** Do not connect the complete optimization loop yet.

---

### Stage 4.12: Offline one-iteration dry run

Connect:

```text
saved evaluation results
→ evidence
→ mock feedback
→ attribution
→ meta-knowledge review
→ component proposals
→ preview states
→ validation using saved fixtures
→ iteration result
```

Requirements:

- no model calls;
- no captioning;
- no DVD reasoning;
- no canonical commits;
- one documented CLI;
- run once with `optimize_scaffold=false`;
- run once with `optimize_scaffold=true`.

Verify that:

- disabled scaffold optimization creates no candidate scaffold;
- enabled scaffold optimization creates a proposal only when attribution supports it;
- other component outputs remain structurally comparable.

**Stop gate:** Review the complete dry-run artifacts before real evaluation.

---

### Stage 4.13: One real iteration with fixed scaffold

Run one complete small iteration with:

```yaml
optimize_prompt_bank: true
optimize_router: true
optimize_scaffold: false
```

Procedure:

1. freeze all input snapshots;
2. freeze evidence, confirmation, and regression batches;
3. generate real structured feedback;
4. propose bank and router updates;
5. create a typed skipped scaffold proposal;
6. review and validate;
7. optionally commit only to a temporary experiment path;
8. run one small routed evaluation;
9. save one iteration result.

Do not run multiple iterations.

**Stop gate:** Inspect the learned bank/router changes and regressions before enabling scaffold optimization.

---

### Stage 4.14: One real iteration with scaffold optimization enabled

Run as a separate reviewed experiment:

```yaml
optimize_prompt_bank: true
optimize_router: true
optimize_scaffold: true
```

Use the same frozen starting state and matched data budget as Stage 4.13 when possible.

Requirements:

- scaffold update must pass stricter review;
- hard contract remains unchanged;
- candidate scaffold gets a new version;
- candidate and incumbent scaffold are compared on confirmation and regression data;
- full rollout validation may be required by configuration;
- rejected scaffold updates must not block separately accepted bank/router updates;
- component commits may be partial and must be logged explicitly.

This is still a one-iteration smoke test, not the final ablation conclusion.

**Stop gate:** Do not begin multi-iteration optimization without a separate experiment plan.

---

### Later stage: matched multi-iteration ablation

Before implementation, create a separate reviewed document covering:

- update scheduling;
- component credit assignment;
- incumbent definition;
- evidence sampling;
- scaffold update frequency;
- bank-size control;
- router-update frequency;
- acceptance thresholds;
- periodic full rollout;
- stopping and rollback;
- matched cost budget;
- held-out full evaluation.

The final scaffold ablation should compare fixed versus optimized scaffold under a matched protocol, not merely compare one anecdotal iteration.

---

## 20. Mandatory checkpoint report

At every stop gate, Claude must report:

1. **Files inspected**
   - exact paths;
   - relevant interfaces found.

2. **Files changed**
   - exact paths;
   - one-sentence purpose.

3. **Behavior added**
   - public interfaces;
   - data flow;
   - explicitly deferred behavior.

4. **Tests and commands run**
   - exact commands;
   - pass/fail counts;
   - skipped tests and reasons.

5. **Artifacts produced**
   - exact paths;
   - short description.

6. **Backward compatibility**
   - Phase 0–3 test result;
   - existing CLI or format changes.

7. **Configuration state**
   - values of all optimization flags;
   - especially `optimize_scaffold`.

8. **Unresolved assumptions**
   - code/document discrepancies;
   - design choices requiring review;
   - temporary stubs.

9. **Diff summary**
   - architectural impact;
   - confirmation that later stages were not implemented.

10. **Next proposed stage**
    - only the next checkpoint;
    - no implementation until approved.

A checkpoint is incomplete if it reports only that tests passed.

---

## 21. Cross-cutting tests

Retain these tests throughout Phase 4.

1. Existing Phase 0–3 tests remain green.
2. Component snapshots are immutable.
3. Hashes and IDs are stable across process restarts.
4. Serialization is deterministic.
5. Router output supports multiple entries.
6. Every selected entry is traceable into the composition trace.
7. Every composed prompt is traceable to bank, router, scaffold, and contract versions.
8. No raw model response bypasses schema validation.
9. No policy mutates its input records.
10. Failed composition cannot enter caption generation.
11. Failed update validation cannot create a committed version.
12. Scaffold-disabled mode never invokes `ScaffoldUpdateProposer` model logic.
13. Scaffold-disabled mode never commits a new scaffold version.
14. Scaffold-enabled mode cannot modify the hard contract.
15. A rejected scaffold update does not roll back accepted bank/router updates.
16. Routed prompt views and surrogate mixed views remain distinguishable.
17. All-default routing reproduces the fixed baseline behavior.
18. Selecting the same entry set for every segment is deterministic.
19. Re-running offline stages with the same inputs yields identical structured outputs, excluding timestamps.
20. Every update is traceable to feedback, evidence, review, and validation IDs.

---

## 22. Error handling

- Do not silently repair malformed model output.
- Do not silently drop selected prompt entries during composition.
- Do not silently alter the scaffold contract.
- Do not silently enable scaffold optimization.
- Do not silently route to a replacement entry.
- Do not silently commit low-confidence updates.
- Do not reuse proposal examples as the only confirmation data.
- Do not mutate incumbent component snapshots.
- Do not promote a routed prompt view into a single-prompt cache.
- Abort on incompatible bank/router/scaffold/contract versions.
- Distinguish schema, routing, composition, captioning, feedback, review, validation, and commit failures.
- Preserve failed artifacts for debugging.
- Record all fallbacks in typed records.

---

## 23. Coding instructions for Claude

- Start each stage by inspecting the actual repository.
- Treat this document as a design target, not proof that existing interfaces match it.
- Prefer adapters and composition over Phase 0–3 refactors.
- Make small, reviewable commits.
- Use typed public interfaces.
- Keep model calls inside policy-specific modules.
- Keep orchestration free of policy-specific parsing.
- Make dry-run and preview mode the default.
- Require an explicit flag for persistent commits.
- Never implement the next stage without explicit approval.
- Do not train or integrate an SLM until the fixed interfaces and artifacts are reviewed.
- Preserve exact prompt texts, composed prompts, routing decisions, model responses, and version lineage.
- Report `optimize_scaffold` explicitly at every run and checkpoint.

---

## 24. First instruction to Claude

Begin with Stage 4.0 only.

Use:

```text
Read CLAUDE.md, PHASE2_3_SURROGATE.md, and PHASE4_PROMPT_ROUTING.md.

Perform Stage 4.0 only.

Do not create prompt_routing/ or optimization/ implementation modules yet.
Do not edit Phase 0–3 logic.
Do not begin Stage 4.1.

Produce the mandatory checkpoint report and stop.
```

---

## 25. Definition of the first Phase 4 milestone

The first milestone is complete only when:

- multiple prompt entries can be selected per segment;
- selected entries are composed into one valid prompt;
- composition is traceable and contract-safe;
- routed captions use correct cache keys;
- the existing DVD reasoning path evaluates routed caption views;
- evidence distinguishes bank, router, and scaffold behavior;
- component-specific proposals are independently generated;
- meta-knowledge review and held-out validation are operational;
- `optimize_scaffold=false` and `true` both execute correctly;
- scaffold-disabled mode never changes scaffold state;
- scaffold-enabled mode can propose, validate, accept, reject, or defer a scaffold update;
- one fixed-scaffold and one scaffold-enabled iteration are reproducible;
- no Phase 0–3 regression is introduced;
- no large multi-iteration run has been started prematurely.
