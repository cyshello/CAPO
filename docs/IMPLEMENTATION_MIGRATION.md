# Phase 4 Implementation Migration

## Active Checkpoint 2 correction

The older pilot role layout and any separate regression-video or
per-iteration-confirmation flow are superseded.

- Keep `split_manifest.json` unchanged: 10 videos / 30 QAs in each of train,
  validation, and test.
- Derive the evidence pool from the eight `previously_cached` train videos.
- Derive confirmation from the two remaining train videos.
- Select `K` unique evidence videos per iteration and run all `3K` QAs. `K=3`
  is the conservative default, not a method constraint.
- Rotate deterministically, prioritizing videos unused since the last
  confirmation.
- When all eight evidence videos have appeared, confirmation becomes due and
  the tracker resets only after an accept/rollback decision.
- Do not reserve separate regression videos. Use `correct_to_wrong` flips on
  the current `K` source videos.
- Updates between confirmations are provisional.
- Validation and test remain outside component-update feedback.

Checkpoint 2 implements only role derivation, rotation/coverage state,
provisional/confirmed schemas, the reusable configurable-size baseline phase, and
zero-or-multiple property proposal records. Real VLM routing, property
retrieval, interventions, feedback calls, confirmation evaluation, and main
execution remain deferred.

## 1. Current implementation to preserve

Preserve the existing:

- mixed-caption view construction;
- selective caption replacement and incumbent fallback;
- caption cache isolation;
- unchanged downstream DVD reasoning path;
- prompt-bank/router snapshot versioning;
- candidate artifact persistence;
- validation infrastructure;
- stable segment IDs and trajectory artifacts.

Do not rewrite Phase 0–3.

## 2. Required architectural correction

The current pilot evaluates one joint bank/router/scaffold candidate. Replace
that Phase 4 control flow with:

```text
current policy snapshot
→ full-caption K rotating evidence videos
→ run all baseline QAs
→ propose multiple properties per video
→ evaluate each property on its source video in an independent selective run
→ aggregate interventional feedback
→ update codebook and router once
```

The fixed scaffold is not a candidate component.

A candidate property is not inserted into the persistent codebook or routed by
the candidate router before evaluation. It is an ephemeral intervention that
is force-added to retrieved segments.

## 3. File-level changes

Paths follow the current implementation report. Inspect actual interfaces
before editing.

### 3.1 `scripts/run_phase4_matched_pilot.py`

Refactor the pilot runner into explicit iteration phases:

1. snapshot current codebook/router/scaffold;
2. select `K` source videos from the rotating evidence pool;
3. full-caption those selected videos with the current policy;
4. run all baseline QAs for every video;
5. invoke the property proposer separately per source video;
6. schedule every property-source-video intervention;
7. execute interventions with bounded parallelism;
8. rerun all QAs of each source video for every property-specific mixed view;
9. build flip-only feedback;
10. aggregate feedback and propose one iteration-level codebook/router update;
11. persist next state without launching another iteration automatically.

Remove the methodological use of:

```python
max_new_entries_per_iteration = 1
```

Replace it with explicit controls such as:

```yaml
videos_per_iteration: K  # default 3, up to the evidence-pool size
max_parallel_videos: P  # independent of K; P <= persistent worker count
max_property_proposals_per_video: B
max_parallel_property_interventions: N
max_selected_segments_per_property: M
```

Do not run the user's main experiment.

### 3.2 Baseline full-caption orchestration

Add or expose a reusable function that full-captions each source video under
the current policy and persists:

- incumbent full caption view;
- frame references;
- local-history snapshots;
- router decisions;
- composed prompts;
- policy versions;
- cache statistics.

Do not derive candidate histories from recaptioned outputs.

`HistoryAwareBaselineCaptionViewBuilder` supports optional history-block data
parallelism. With more than one configured GPU it spawns one process per active
GPU, initializes one persistent local Qwen/vLLM backend in that process, and
assigns complete history blocks round-robin. Segments remain sequential within
a block. The parent performs a deterministic source-order merge only after all
blocks complete. Per-worker cache-manifest fragments avoid concurrent append
corruption and are checked and merged into the configured manifest by the
parent. Block artifacts are stable resume units; changing worker/GPU assignment
does not change semantic cache keys.

Workers remain alive for the complete run and accept both full-history-block
and selective-segment caption commands. Do not join them between stages: the
parent advances after receiving task completion records and reuses the loaded
engines for later videos/interventions. At the run boundary, request an
explicit worker shutdown, call the vLLM EngineCore shutdown path, then join with
a bounded termination fallback. A worker task failure fails that task without
silently loading a second parent-process Qwen instance.

Route DVD `frame_inspect_tool` raw vision requests through the same pool by
installing a parent-process captioner proxy only while the pool is active.
Preserve standalone lazy Qwen initialization when no pool exists. Baseline QA
completion is fail-closed: any runtime/tool error, null prediction, or parse
failure prevents the property proposer from being called, even though all
three QA attempts and their failure artifacts are persisted first.

The history-aware Qwen router must pass a per-codebook JSON Schema to vLLM
structured decoding. The schema allows only `property_ids`, restricts values to
the frozen active property IDs, and enforces the current maximum array length.
Use structured-output decoding with fallback disabled. Do not apply this
constraint to normal caption generation, and include the structured-output
policy/version in router configuration identity.

Caption generation itself uses the versioned `caption_output_contract_v2`
compatibility boundary. The rendered request reiterates that the top level is
one object and requires a non-empty `clip_description`. The default
`SR_CAPTION_SUBJECT_REGISTRY_MODE=empty` asks Qwen for `{}`; `optional` enables
registry generation for a more capable model. Parsing preserves a valid registry, defaults a
missing/null/non-object registry to `{}`, and unwraps only a single-element
object array. Raw output and `caption_parse_result_v2` are saved in
`history_aware_caption_cache_v2`; ambiguous arrays, invalid JSON, and blank
descriptions remain failures. Contract/parser versions enter cache and resume
identity so v1 cache files remain immutable and are not reinterpreted.

`caption_parse_retry_v1` adds up to five parse-failure retries after the
initial caption call. It does not retry runtime/backend exceptions and does not
alter property text, frames, transcript, frozen history, or decoding. New
artifacts use `history_aware_caption_cache_v3` and retain every raw attempt and
parse reason. Existing valid v2 cache entries remain compatible; existing
invalid v2 entries remain immutable and receive a versioned retry sidecar.
After five unsuccessful retries, the segment remains invalid and the existing
selected-segment intervention failure path is unchanged.

### 3.3 Property proposal stage

Adapt `optimization/policies/llm_feedback.py` or add a narrowly scoped property
proposal policy.

Input per source video:

- all baseline QA outcomes, ordered to prioritize incorrect QA evidence;
- question, choices, ground truth, and baseline prediction without question ID;
- at most three sanitized reasoning events per QA, excluding tool-call IDs and
  source/question/segment identifiers;
- the complete union of `explicitly_cited_segments` and
  `frame_inspected_segments` in private provenance, while the provider payload
  packs timestamp-ranged explicitly cited evidence first and compact
  inspected-only summaries second under a fixed evidence-character budget;
- structurally split `baseline_generated_description` and `source_transcript`
  when the known raw transcript suffix marker is present;
- no `returned_segments` or general consumed/used-segment fallback;
- current codebook.

Output:

- one or two candidate properties for every eligible correct or incorrect QA;
- a source-free `source_qa_slot` on each provider proposal, privately mapped to
  exactly one source QA ID and its baseline correctness;
- one readable non-binding `suggested_property_id` per candidate; the parser
  replaces it with a deterministic opaque proposal handle for all lineage;
- source video ID;
- source QA IDs;
- one natural-language `failure_analysis`, with no fixed failure taxonomy;
- non-binding possible-codebook-coverage hints;
- reusable property instruction;
- strict applicability with `when`, positive/negative observable cues, and
  required modalities limited to `frames`, `transcript`, and
  `caption_history`.

Do not directly mutate the codebook.

Treat proposer `covered_by_existing_property_ids` as hints and persist them
internally as `coverage_hints`; never veto a candidate merely because this list
is non-empty. Pre-intervention rejection is structural only: malformed or
missing fields, empty text, unresolved placeholders, exact duplicate proposals,
invalid QA-slot lineage, or instructions requiring tools/inputs unavailable to
the captioner. Do not reject correct-sample candidates, lexical overlap with QA
text, active-property similarity, uncertain benefit, weak generality, or likely
repetition. Defer all semantic suitability and coverage
to Checkpoint 3D `coverage_assessment` and `covered_by_property_ids`, which feed
the Checkpoint 3E add/revise/merge/router/no-op rules.

Do not send source-video ID, question ID, segment ID, provider priority rank,
tool-call ID, or payload-truncation metadata. Preserve these only in a private
`input_identity.json` used for lineage and resume identity. Bound reasoning
event count/characters, caption characters, transformed image bytes, and total
text characters before the provider call. Compact payload evidence uses
timestamp ranges: cited intervals retain fuller generated-description and
non-duplicate transcript text, while inspected-only intervals use one-line
250-character description and 120-character transcript summaries. Pack cited
evidence first, then inspected-only evidence, beneath a 200,000-character
evidence cap and the existing 250,000-character total envelope. The complete
original union is retained privately with inclusion/omission decisions and
hashes; `evidence_packing.json` records aggregate and per-QA counts. `request.json`
stores the logical source-free multimodal input; `provider_request.json` stores
the exact secret-free OpenAI request body, including data-URL image blocks.
Attach the source video and the one actual source QA ID/baseline-correctness pair
internally from `source_qa_slot` after strict output parsing rather than asking
the model to repeat private IDs.

The proposer now treats candidates as executable intervention hypotheses. Every
eligible QA requires one or two candidates whether its baseline answer is correct
or incorrect. Downstream intervention and the existing updater decide each
candidate's fate. Zero remains valid only when no QA is eligible because of
runtime failure, missing focused evidence, malformed input, or clearly unreliable
annotation. The canonical proposal and opaque identity include applicability,
the exact source QA, and baseline correctness;
legacy failure fields are accepted only by artifact loader/constructor adapters
and are never serialized by the new provider contract. Applicability and
`failure_analysis` are retained in proposal, feedback, intervention, property-
memory, and codebook-updater artifacts. `property_intervention_transitions_v2`
retains the original proposal plus candidate QA outcomes and transitions;
`property_compact_summary_v2` and `memory_codebook_updater_request_v2` carry the
bounded provenance to the updater. Router interpretation remains out of scope.

`multi_property_proposer_v10` adds `missing_qa_slot_retry_v1`. A structurally
valid initial response that omits a required QA slot gets at most two follow-up
requests containing only the still-missing slots. Existing candidates remain
unchanged. Retry request/body/raw and per-row validation artifacts are
immutable and reusable for exact resume. The combined output uses the same
1–2-per-QA validation; retry exhaustion still fails closed. Intervention and
updater acceptance semantics are unchanged.

### 3.4 Similarity retrieval

Implement property-conditioned retrieval for every candidate property:

```text
exact normalized candidate property text + sampled source-video segment frames
→ SigLIP frame-level cosine similarity
→ intersect with frozen baseline caption-view segment IDs
→ maximum frame pooling per segment
→ deterministic top-M ranked source-video segments
→ S_sim
```

Reuse the existing SigLIP visual index and compatible cached frame embeddings.
The retrieval query must not include frozen history, baseline captions,
questions, answer choices, answers, correctness, traces, or used segments.
History remains available only to baseline routing/captioning and later
selective re-captioning/feedback.

Persist:

- candidate property ID;
- source video ID;
- ranked segment IDs and scores;
- selected `S_sim`;
- retrieval budget and model/version.
- property-text hash, frame-sampling identity, visual-index identity, maximum
  frame-pooling rule, and deterministic ranking version.
- baseline segment-universe hash/count and excluded visual-index segment IDs.

The baseline passes `caption_view.segment_ids` as a structural allowlist. The
retriever filters the visual index against it before top-M selection, so a tail
frame binned into a segment without an incumbent caption cannot consume a
retrieval slot or fail mixed-view construction. Caption strings remain
prohibited retrieval inputs.

### 3.5 `prompt_routing/routed_caption_view.py`

Preserve mixed-view behavior.

Add support for an ephemeral intervention property:

- load the incumbent routed property set for each selected segment;
- append exactly one candidate property;
- compose the fixed base prompt plus incumbent properties plus candidate
  property;
- recaption only `S_sim`;
- reuse incumbent captions elsewhere;
- fail the entire candidate if any selected segment cannot be re-captioned or
  validated; never fall back selected segments to incumbent captions.

The temporary sequence may contain `max_selected_properties + 1` entries.
Append the candidate after all incumbent properties, do not remove incumbents
or resolve semantic conflicts, and fail explicitly on prompt-budget overflow.

The candidate property must not be committed to the codebook during this step.

### 3.6 Frozen history

Build local-history snapshots during the baseline full-caption pass.

For every candidate intervention:

- reuse the exact baseline history snapshot;
- record its hash/version;
- do not propagate changed captions to later segments;
- do not let one candidate intervention affect another.

### 3.7 Parallel intervention scheduler

Represent one work item as:

```text
(iteration_id, source_video_id, candidate_property_id)
```

Each work item owns:

- one `S_sim`;
- one selective recaption job;
- one mixed caption view;
- one rerun of all source-video QAs;
- one feedback bundle.

Allow different work items to run in parallel with configurable resource
limits. Ensure output paths and cache namespaces cannot collide.

### 3.8 `optimization/evidence_builder.py`

Build feedback evidence per:

```text
(source video, candidate property, QA)
```

For each QA:

- compare incumbent and property-specific candidate results;
- classify correctness transition;
- retain only `wrong_to_correct` and `correct_to_wrong` for optimization;
- extract `S_used` from the incumbent trace;
- extract `S_usedagain` from the candidate trace;
- compute:

```text
S_feedback = S_sim ∩ (S_used ∪ S_usedagain)
```

An empty intersection is persisted as `empty_s_feedback` and makes no feedback
model call.

- load relevant trace contents, not only filesystem paths;
- include only `S_feedback` segments;
- resolve incumbent property IDs and candidate property to full text;
- include frames, concise frozen history, and before/after captions;
- enforce deterministic size limits and log truncation.

Bound frame payloads with a persisted deterministic resize/JPEG-compression
configuration and transformed-frame hash. Reject a frame only after exhausting
that transformation. Keep failed property artifacts immutable; explicit
`retry_failed` execution writes to an isolated retry namespace.

Keep non-flip records in analysis artifacts.

### 3.9 Interventional feedback model

The feedback model should return compact property credit/blame and codebook
coverage judgment.

Its evidence sentence must be one concise, generalizable sentence and should
not copy full histories, captions, traces, or answer text.

Aggregate QA-level outputs into one property-source-video result containing:

- helped QA IDs;
- harmed QA IDs;
- positive evidence;
- negative evidence;
- attributed segments;
- source lineage;
- coverage assessment.

Recommendation labels are limited to `add`, `revise`, `merge`, `retire`,
`router_positive`, `router_negative`, and `no_op`; this stage records but does
not apply them.

### 3.10 `optimization/prompt_bank_update_proposer.py`

Replace first-supported-item behavior with iteration-level aggregation across
all `K` current evidence videos and all candidate properties.

Support:

- `add_entry`;
- `revise_entry`;
- `merge_entries`;
- `retire_entry`;
- `no_op`.

Multiple `add_entry` operations may be proposed in one iteration.

Rules:

- existing coverage -> router update, not duplicate entry;
- useful missing behavior -> add;
- partially correct entry -> revise;
- semantically duplicate proposals -> merge;
- repeated harmful/redundant behavior -> revise or retire;
- weak evidence -> no-op.

### 3.11 `optimization/router_update_proposer.py`

Generate router updates only after codebook decisions.

Support:

- positive applicability;
- negative/avoid applicability;
- revise existing rule/example;
- disable harmful rule/example;
- no-op.

New properties receive router supervision from their successful
source-video/segment contexts.

### 3.12 `optimization/candidate_preview.py`

The intervention property itself should use an ephemeral preview, not a
persistent candidate bank/router/scaffold snapshot.

Use persistent candidate previews only for the final iteration-level
codebook/router update.

### 3.13 `optimization/update_validator.py`

Validate the aggregated iteration update.

This validation produces provisional state. Confirmation evaluation is a
coverage-cycle boundary, not a per-iteration step. The old requirement for a
separate regression-video batch is superseded.

Reject:

- duplicate codebook additions;
- unsupported merges or merge cycles;
- retirements without repeated evidence;
- router updates targeting missing/retired properties;
- scaffold changes;
- updates without cited intervention evidence IDs.

At coverage completion, evaluate the two confirmation videos only. Accept when
all six QAs are error-free, there are no `correct_to_wrong` transitions, and
mean accuracy does not decrease. Promote or roll back the bank/router pair
atomically while preserving the fixed scaffold and contract versions.

`optimization/confirmation_evaluator.py` is the concrete confirmation adapter.
It creates one immutable input bundle for both policies, runs the existing
`HistoryAwareBaselineCaptionViewBuilder` independently for parent and candidate,
validates complete per-segment cache identities and paths, then executes all six
QAs per policy through `run_dvd_qa`. Both policies share frames and runtime
configuration, but construct separate on-policy caption histories. Confirmation
never invokes proposal, intervention, or feedback stages.

### 3.14 Output artifacts

Add explicit lineage and nesting:

```text
iteration/
├── policy_snapshot/
├── baseline/
│   └── <video_id>/
├── property_proposals/
│   └── <video_id>.json
├── interventions/
│   └── <video_id>/
│       └── <property_id>/
├── feedback/
│   └── <video_id>/
│       └── <property_id>/
└── next_state/
```

### 3.15 Checkpoint 1 implementation log: compact property memory

Exact before-versus-after state:

| Concern | Before Checkpoint 1 | After Checkpoint 1 |
|---|---|---|
| Raw evidence | Complete artifacts persisted | Unchanged and still authoritative |
| Cross-iteration compact state | Codebook, router supervision, coverage and policy lineage only | Optional parent-linked bounded property/candidate memory |
| Correct-QA successes | No property-level retained credit | Strong/weak/none summaries with a routed-caption-reasoning-answer chain |
| Candidate interventions | Flip feedback could update components; no compact all-transition history | Positive/negative/mixed/no-effect summary from all four counts |
| Seed-property origin | Only codebook provenance | Explicit `seed_or_legacy` memory origin when creation evidence is absent |
| Bounded selection | Router kept its own supervision list; no property example policy | Strength, distinct video, diversity, then recency with retention/eviction audit |
| Candidate promotion | Existing updater decided add/merge | Decision unchanged; memory records promotion only after that decision exists |

Files and interfaces added or changed:

- `optimization/property_memory.py` adds `CompactPropertyMemoryRunner`,
  `PropertyMemoryBounds`, immutable artifact/hash helpers, conservative
  correct-QA credit extraction, deterministic intervention summaries, bounded
  selection, candidate separation/promotion recording, resume, and fail-closed
  schema validation.
- `tests/test_checkpoint1_property_memory.py` provides fixture-only coverage;
  it performs no GPU, paid API, or real-model call.
- `docs/CURRENT_METHOD.md`, `docs/IMPLEMENTATION_MIGRATION.md`, and
  `docs/RUNBOOK.md` record the active contract and operation.

The runner interface consumes completed baseline, intervention, and optional
feedback manifests plus a frozen codebook. `parent_memory_path` enables bounded
accumulation. `update_plan`, `update_plan_path`, and
`resulting_prompt_bank` are optional, observational inputs: they permit
recording an updater decision but never alter it. The compact-summary schema is
`property_compact_summary_v1`; the snapshot/record schema is
`property_memory_v1`; the completion manifest is
`property_memory_manifest_v1`; selection is
`property_memory_selection_v1`.

Intentionally deferred:

- replacing the deterministic codebook updater with an LLM updater;
- adding compact memory to the router prompt or changing router updates;
- changing property proposal acceptance, prompting, or retrieval semantics.

### 3.16 Checkpoint 2 implementation log: memory-conditioned codebook updater

Previous behavior: `Checkpoint3EOrchestrator.aggregate_updates()` converted
accepted flip-only feedback into deterministic codebook decisions and router
supervision, and `apply_update_plan()` applied both components to one
provisional pair. It had no `property_memory_v1` input, no tunable centralized
LLM planning prompt, and no candidate-codebook-only artifact boundary.

Current behavior: `run_memory_conditioned_codebook_checkpoint()` resolves the
parent memory snapshot from `state_dir/property_memory/current.json`, invokes
`CompactPropertyMemoryRunner` on the completed iteration artifacts, and passes
the resulting bounded snapshot and compact intervention summaries directly to
`MemoryConditionedLLMCodebookUpdater`. It persists the new memory lineage
pointer separately from production policy pointers. Exact repeat returns the
completed checkpoint without provider calls. Changed/missing artifacts,
incompatible schemas, a bank/memory-lineage mismatch, or partial output without
the new matching input identity fails closed; a matching strict-retry partial
output resumes its saved attempts.

Files and interfaces changed:

- `optimization/llm_codebook_updater.py`: strict request/response parsing,
  bounded request building, action-by-action deterministic validation,
  candidate bank application, ID mapping, memory promotion, artifact identity,
  and exact resume. The real provider now uses updater-specific strict JSON
  Schema output (`openai_strict_component_update_v2`) and the updater preserves
  up to three immutable parse attempts. `input_identity.json` permits an
  interrupted attempt sequence to resume without repeating saved provider
  calls; pre-identity partial directories fail closed.
- `optimization/prompts/codebook_updater_v1.txt`: centralized tunable system
  prompt, version `memory_codebook_updater_prompt_v1`.
- `optimization/final_iteration.py`: optional memory/updater stage injection and
  `run_memory_conditioned_codebook_checkpoint()` orchestration boundary.
- `optimization/property_memory.py`: preserves categorized candidate examples
  across parent-memory accumulation.
- `tests/test_checkpoint2_memory_codebook_updater.py`: mock-only updater and
  orchestration coverage.
- the three authoritative Phase 4 markdown files: active behavior, artifacts,
  commands, lineage, and deferred boundary.

Artifacts use these schemas:

- request: `memory_codebook_updater_request_v2`;
- response: `memory_codebook_updater_response_v1`;
- validation: `memory_codebook_validation_report_v1`;
- applied plan: `memory_codebook_applied_plan_v1`;
- candidate codebook: `memory_candidate_codebook_v1`;
- ID mapping: `property_id_mapping_v1`;
- updater manifest: `memory_codebook_checkpoint_manifest_v2`;
- retry audit: `memory_codebook_updater_attempts_v1`;
- partial-resume identity: `memory_codebook_updater_input_identity_v1`;
- orchestration manifest: `memory_conditioned_codebook_iteration_v1`;
- memory lineage pointer: `property_memory_lineage_pointer_v1`.

At the Checkpoint 2 boundary, router-prompt updating, ID remapping, candidate
bank/router validation, and atomic pair commit were intentionally deferred.
Checkpoint 3 below resolves those items; confirmation and production-pointer
mutation remain deferred.

### 3.17 Checkpoint 3 implementation log: router prompt and atomic pair

Previous behavior: deterministic updates accumulated bounded
`supervision_examples`, while `HistoryAwareVLMRouter` still rendered its fixed
hard-coded instruction plus the request. There was no structured editable
routing policy, rendered-prompt artifact/hash, memory-conditioned router LLM,
or atomic continuation from the Checkpoint 2 candidate codebook.

Current behavior: `MemoryConditionedLLMRouterUpdater` reads the validated
candidate codebook and complete ID mapping, remaps legacy guidance
deterministically, joins bounded positive/negative routing memory with current
candidate intervention effects, and sends only those summaries plus validated
codebook actions to `optimization/prompts/router_updater_v1.txt`. Strict JSON
actions are individually validated. The LLM never writes a prompt or router
snapshot directly.

`structured_router_policy_v1` keeps the immutable protocol separately from
per-property selection/avoidance guidance, two positive/two negative examples,
aliases, and remapped IDs. `history_aware_router_prompt_renderer_v1` produces
`rendered_router_prompt_v1`; the exact text and hash are installed in the
candidate `RouterPolicySnapshot` and consumed by the real history-aware VLM
router. Invalid/stale IDs, unsupported example polarity, question/answer/gold/
prediction/reasoning leakage, conflicts, retired guidance, protocol changes,
and hash mismatch fail closed.

`Checkpoint3EOrchestrator.run_memory_conditioned_router_checkpoint()` validates
the complete Checkpoint 2 artifact closure and creates
`atomic_provisional_policy_pair_v1` only after bank and router share the same
active property-ID space and both updater plans/validation reports are present.
It writes only `state_dir/memory_conditioned_provisional/current.json`. Router
failure leaves the candidate codebook uncommitted as a pair, records immutable
failure evidence, and preserves the parent policy. Exact resume verifies every
artifact and the rendered prompt hash before making zero provider calls.

Files and interfaces changed:

- `prompt_routing/structured_router_policy.py`: structured policy schema,
  deterministic ID remapping, fixed scaffold validation, rendering, prompt
  hashing, and candidate-router installation;
- `prompt_routing/policies/history_aware_vlm_router.py`: loads the installed
  rendered prompt and includes its hash/version in request and decision
  identity while preserving the legacy path;
- `optimization/llm_router_updater.py` and
  `optimization/prompts/router_updater_v1.txt`: bounded request, strict plan,
  deterministic validation/application, complete audit artifacts, and resume;
- `optimization/final_iteration.py`: separate atomic provisional-pair boundary;
- `optimization/llm_codebook_updater.py`: preserves bounded routing examples
  when property memories merge;
- `tests/test_checkpoint3_memory_router_updater.py`: mock-only checkpoint tests.
- `optimization/bounded_smoke.py`, `scripts/run_phase4_bounded_smoke.py`, and
  `optimization/policies/openai_update.py`: explicit one-video real-smoke
  opt-in, strict-schema OpenAI updater providers with at most three parse
  attempts, atomic checkpoint chaining,
  rendered-prompt consumption probe, confirmed-pointer guard, exact resume,
  and worker-cleanup audit. Legacy fixture callers remain deterministic.

Schemas/versions are `memory_router_updater_request_v1`,
`memory_router_updater_response_v1`, `memory_router_updater_plan_v1`,
`memory_router_validation_report_v1`, `memory_router_applied_plan_v1`,
`memory_router_updater_manifest_v2`, `memory_router_updater_attempts_v1`,
`memory_router_updater_input_identity_v1`, `structured_router_policy_v1`,
`rendered_router_prompt_v1`, `atomic_provisional_policy_pair_v1`, and
`memory_conditioned_atomic_policy_pair_v1`; the system prompt is
`memory_router_updater_prompt_v1`.

The router updater execution policy is
`no_routing_evidence_empty_plan_v1`. Previously, an iteration whose codebook
plan contained only target-free candidate `no_op` decisions could lead the LLM
to copy those decisions into router actions with `target_property_id: ""`; the
strict parser correctly rejected the response and prevented the atomic pair.
Now zero routing evidence deterministically produces an empty router plan
without a provider call. `execution.json`, the updater manifest, the atomic
pair, and the operational log record the execution mode and provider-call
flag. Target validation is unchanged for every LLM-generated action.

Intentionally deferred: confirmation of this provisional pair, promotion to a
confirmed production pointer, and any regular production or real-model run.

### 3.17.1 Strict updater envelope and bounded parse recovery

Previously both production updater adapters requested only
`response_format=json_object`. A response could therefore be valid JSON while
omitting `schema_version`, or a router no-op could omit/nullable its required
target. The first strict parser error terminated the iteration and only one
top-level `raw_response.txt` remained.

Now `optimization/policies/openai_update.py` supplies separate strict response
schemas for the codebook and router planners. The unchanged canonical response
schemas remain `memory_codebook_updater_response_v1` and
`memory_router_updater_response_v1`; provider enforcement and execution
semantics are versioned separately. Both updaters retain raw/result artifacts
for attempts 1 through 3, write a response-attempt summary, and use an immutable
input fingerprint to continue a partial attempt sequence. They never inject a
missing version, repair model JSON locally, retry runtime/API failures, or
reinterpret a completed legacy artifact. The codebook candidate, router
validation, rendering, and atomic-pair semantics are unchanged.

### 3.18 Production memory-conditioned iteration launcher

The previous repository exposed the latest memory-conditioned path only through
the isolated one-video bounded smoke. The obsolete Stage 4.13/4.14 launcher did
not construct the property-memory, LLM codebook, rendered-router, and atomic
pair chain and remains unused.

`scripts/run_phase4_memory_iteration.py` now constructs the reviewed real
baseline, intervention, feedback, memory, codebook-updater, and router-updater
stages directly. `optimization/production_iteration.py` freezes the ordered
selection and complete resume identity before model startup, runs per-video
work in deterministic waves, calls both iteration-level updaters once, and
writes a completed manifest only after the atomic pair exists. A router-stage
failure can leave candidate diagnostics but cannot create the completed
iteration or a codebook-only policy pair.

`--num-videos K` defaults logically to three when neither count nor explicit
IDs are supplied. `--video-ids` preserves explicit order. `--max-parallel-videos
P` controls only wave width and is bounded by the unique `--gpus` worker set;
it does not constrain `K`. Selection is seeded, rotating, evidence-pool-only,
and persists its next position only after atomic success. Exact resume restores
the frozen parent pair and ordered videos from `iteration_identity.json`, so a
newer state pointer cannot silently change an existing run.

Files and interfaces changed:

- `optimization/production_iteration.py`: selection, waves, planning,
  orchestration, atomic completion, and exact-resume validation;
- `scripts/run_phase4_memory_iteration.py`: reviewed real component assembly,
  CLI validation, parent-pair resolution, GPU preflight, and cleanup audit;
- `optimization/train_roles.py` and `optimization/iteration_state.py`:
  variable-size evidence batches while retaining the default of three;
- `captioning/history_aware_baseline.py` and `optimization/baseline_phase.py`:
  persistent single-worker affinity per scheduled video and concurrent
  request/result demultiplexing across workers;
- `optimization/final_iteration.py`: real-constructor memory-stage injection
  and promoted property-memory lineage after atomic pair commit;
- `tests/test_production_memory_iteration.py`: mock-only production-launcher
  coverage.

Schemas are `memory_production_iteration_plan_v1`,
`memory_production_iteration_identity_v1`,
`memory_production_iteration_manifest_v1`,
`memory_production_selection_pointer_v1`, and
`production_worker_cleanup_v1`. Confirmation and confirmed-pointer promotion
remain explicit, separate, and are not run by this launcher.

The launcher also accepts an optional dedicated `--embedding-gpu`. It rejects
overlap with Qwen worker GPUs and validates availability/free memory before
model startup. GPU embedding is owned by a spawned child process whose
bootstrap environment exposes only that physical GPU; inside the child SigLIP
uses `cuda:0`. CPU remains the compatibility default when omitted. Physical
GPU, isolated-process mode, logical device, child PID, and cleanup status are
saved in the plan, execution identity, and `production_worker_cleanup_v2`
record. Requests are serialized, and the child/model are explicitly closed
before parent-process exit. This checkpoint does not add asynchronous
pre-index scheduling.

### 3.19 Fixed downstream DVD clip-search budget

Previously, the DVD tool schema advised the QA model to use its default
`clip_search_tool` budget, but the model could explicitly request any `top_k`.
The vendored `OVERWRITE_CLIP_SEARCH_TOPK` path also checks the misspelled key
`topk`, so it does not constrain the actual `top_k` argument. The harness now
forces every executed DVD clip search to `top_k=16` at the instrumentation
boundary without editing vendored DVD code. Raw model arguments stay in the
trajectory; tool events retain both requested and executed arguments plus
`fixed_clip_search_top_k_v1`. The value and policy version enter baseline,
intervention, confirmation, and production resume identities. Phase 4
property-frame retrieval top-k remains a separate setting.

### 3.20 Optional subject registry and caption-object fallback

Before this correction, the DVD caption template asked every model to build a
detailed `subject_registry`, although Phase 4 success actually consumed only
`clip_description`. The parser silently converted any top-level list to `{}`;
Qwen 2.5-VL could return a valid single caption object inside a list, producing
empty baseline captions and selected-segment intervention failures.

After this correction, `clip_description` is the required semantic field. The
default prompt asks for an empty registry; setting
`SR_CAPTION_SUBJECT_REGISTRY_MODE=optional` enables larger models to populate
it. Missing or invalid registries become `{}`, valid populated registries
remain available for merging, and exactly one caption object in
an array is safely unwrapped. All other arrays, invalid JSON, and blank
descriptions fail closed. Raw responses are unchanged and parse decisions are
auditable. Output-contract and parser versions invalidate semantic resume and
cache identity without overwriting or reinterpreting legacy artifacts.

Changed interfaces are `captioning/history_aware_baseline.py` and the
baseline, intervention, and confirmation identity builders. Router, proposal,
retrieval, feedback, updater, and confirmation-decision semantics are
unchanged.

### 3.21 Opaque proposal handles and isolated embedding worker

Before this correction, proposer-controlled `candidate_property_id` was used
directly as cross-stage lineage. Two source videos could independently return
the same readable name, making distinct interventions collide in the
iteration-level updater. A later hotfix prefixed duplicate names during memory
construction, but that still conflated a naming suggestion with proposal
identity. Also, `--embedding-gpu 4` constructed `cuda:4` in a parent process
whose DVD setup could expose only physical GPU 5 as logical `cuda:0`, causing
`invalid device ordinal`.

`multi_property_proposer_v5` and
`multimodal_property_proposal_request_v4` now ask for
`suggested_property_id`. The strict parser validates and preserves that hint,
then derives an `opaque_candidate_proposal_id_v1` handle from private frozen
lineage and normalized property text. Retrieval, intervention, feedback,
memory, and both updaters use the opaque handle; the codebook updater remains
the only stage that selects a final active property ID. Legacy duplicate names
receive deterministic opaque migration handles without changing raw artifacts.

`ProcessIsolatedSiglipEmbedder` owns dedicated-GPU SigLIP in one spawned child,
with the physical GPU visible before bootstrap and logical `cuda:0` inside.
The parent environment is restored immediately after spawn. Exact execution
identity records the isolation mapping, and cleanup records the child PID and
release result. CPU embedding and compatible read-only visual-index caches are
preserved.

DVD search uses another CPU BGE embedder, independent of the isolated SigLIP
worker. The production launcher now preloads it once under a lock before
parallel evidence-video QA (`dvd_bge_parent_preload_v1`). This replaces unsafe
concurrent lazy initialization without changing DVD retrieval or QA semantics.
The policy version participates in resume identity, so an incomplete run made
under the previous initialization behavior is not silently reinterpreted.

`serialized_dvd_qa_execution_v1` additionally guards DVD's mutable global
`VIDEO_FPS`, prompt reset, and instrumentation installation. Evidence videos
may finish captioning concurrently, but their downstream DVD QA calls enter one
process-local critical section. Database construction must persist the exact
per-video effective FPS or the QA fails before tool execution. This prevents
one video's frame inspection from addressing another video's frame indices and
prevents nested/cross-wired instrumentation recorders. Captioning and
intervention worker parallelism are unchanged.

### 3.22 Production iteration operational logging

Before this correction, model/backend output reached stdout but the production
orchestrator did not provide one readable stage timeline. Parallel video output
was difficult to associate with an iteration, video, candidate, or update
boundary.

`optimization/stage_logging.py` adds `phase4_iteration_stage_log_v1`, a
thread-safe append-only operational logger mirrored to stdout.
`MemoryConditionedProductionIterationRunner` creates
`<output_dir>/iteration.log`, reports the iteration ordinal and deterministic
video waves, and passes the logger into the baseline, intervention, memory,
codebook, and router boundaries. `BaselinePhaseRunner` reports captioning, QA,
proposal, and similarity results per video; `PropertyInterventionBatchRunner`
reports recaption/cache results and all candidate QA transitions;
`Checkpoint3EOrchestrator` reports memory, validated codebook actions, rendered
router prompt identity, and atomic pair completion. Resume is explicitly
reported without changing semantic identity. The operational log is not added
to immutable artifact hashes.

### 3.23 Compact property-proposer evidence serialization

Before this correction, `multi_property_proposer_v10` placed a structured
caption/transcript object and representative image into the provider request
for every cited or inspected interval. A QA that inspected hundreds of
intervals could exceed the fixed 250,000-character text envelope before the
proposal provider was called.

`multi_property_proposer_v11` preserves the complete original evidence union
only in private provenance and sends a deterministic compact view. Provider
evidence is identified by `[start–end]` timestamp range, never stable segment ID.
Explicitly cited intervals are ordered first and retain normalized fuller
generated description and non-duplicate transcript. Inspected-only intervals
follow as one-line summaries with 250-character description and 120-character
transcript limits. Empty or substantially duplicate transcripts are omitted.
No additional model is called.

The evidence-text budget is at most 200,000 characters and is reduced when
needed to reserve the remainder of the existing envelope for the fixed prompt,
schema, QA context, and JSON structure. Packing stops deterministically at the
budget. `input_identity.json` retains every original caption, split transcript,
segment ID, timestamp, evidence role, representative-frame path/transform/hash,
and inclusion or omission reason. `evidence_packing.json` records aggregate and
per-QA included/omitted interval counts, and the same counts are logged.

Versions are `multimodal_property_proposal_request_v10`,
`multimodal_property_proposal_identity_v6`,
`multi_property_proposal_artifact_v6`, and
`compact_cited_then_inspected_evidence_v2`. Existing completed proposal
artifacts remain immutable and require a fresh output/state directory;
compatible caption caches remain reusable. Proposal parsing, candidate
semantics, intervention, memory, updater, and router behavior are unchanged.

### 3.24 Percent-escape recovery and valid-caption retrieval universe

A completed real run exposed two coupled boundary errors. Qwen emitted an
otherwise-valid fenced caption object containing the invalid JSON escape `\%`.
All deterministic retries repeated it, leaving `parsed={}` and omitting that
segment from `captions.json`. However, retrieval intersected the visual index
with `caption_view.segment_ids`, which describes every scheduled source segment,
not the subset with a valid incumbent caption. The invalid segment could
therefore enter `S_sim` and fail later during mixed-view replacement.

`caption_parse_normalization_v2_percent_escape` removes only an unescaped `\%`
before strict JSON loading and records `replace_invalid_percent_escape`. Valid
escaped backslashes and unrelated invalid escapes are unchanged. Raw model
output remains immutable. `history_aware_caption_cache_v4` and
`caption_parse_retry_v2_percent_escape` reuse earlier valid caches; an earlier
invalid raw response is reparsed in the new versioned retry sidecar, without a
model call when that narrow normalization is sufficient.

`valid_baseline_caption_segment_intersection_v2` passes only source-ordered,
non-empty actual `captions.json` entries to `property_retrieval_batch_v3` and
`property_frame_retrieval_v3`. Failed caption segments remain in raw routing and
cache artifacts but cannot be retrieved or selectively replaced. The valid
caption-universe hash and count preserve exact resume and make older retrieval
artifacts incompatible rather than silently reinterpreted.

## 4. Required tests

Add focused tests for:

1. one iteration full-captions each of `K` selected videos once;
2. all baseline QAs run for each video;
3. one video can produce multiple property candidates;
4. candidate lineage records source video and source QAs;
5. every property candidate becomes an independent work item;
6. candidate property is force-added only to `S_sim`;
7. incumbent router, codebook, scaffold, and history remain fixed;
8. candidate interventions do not affect each other;
9. all source-video QAs rerun for every property intervention;
10. non-flips are excluded from optimization feedback;
11. `S_feedback` set computation is exact;
12. feedback payload contains frames, full property text, concise history, and
    relevant traces;
13. multiple accepted properties can be added in one iteration;
14. codebook coverage prevents duplicate additions;
15. router receives positive and negative contextual supervision;
16. mixed-view, fallback, cache, and registry invariants remain green;
17. no main experiment is launched by tests or implementation scripts.

Run the focused tests and complete existing suite.

## 5. Minimal implementation order

1. Checkpoint current repository and update documentation.
2. Add explicit configurable-`K` baseline iteration orchestration.
3. Add per-video multi-property proposal output and lineage.
4. Implement property-conditioned source-video retrieval.
5. Add ephemeral force-add property intervention.
6. Add parallel work-item scheduling and isolated artifacts.
7. Rerun all source-video QAs per intervention.
8. Implement flip-only compact feedback and `S_feedback`.
9. Aggregate iteration feedback.
10. Update codebook and router once.
11. Run focused tests and complete regression suite.
12. Prepare exact user-run commands; do not run the main experiment.

## 6. Do not implement now

- history propagation;
- automatic evaluation on unrelated videos;
- per-segment QA rollouts;
- SLM-only router;
- scaffold optimization;
- trust regions;
- broad renaming;
- automatic main-run execution.

## 7. One-video bounded-smoke boundary

The isolated bounded-smoke adapter is not a replacement for the production
configurable-`K` coverage cycle. It selects one frozen evidence video and its three
QAs, limits proposals and candidate interventions to one, and uses retrieval
top-k one. `post_intervention_mode` controls whether execution stops after QA,
after feedback aggregation, or after isolated provisional update artifacts.
Changing the mode preserves compatible upstream caches. The adapter never
writes coverage-cycle state, confirmed checkpoints, or canonical pointers and
never calls confirmation.
