# Phase 4 Implementation Migration

## Active Checkpoint 2 correction

The older pilot role layout and any separate regression-video or
per-iteration-confirmation flow are superseded.

- Keep `split_manifest.json` unchanged: 10 videos / 30 QAs in each of train,
  validation, and test.
- Derive the evidence pool from the eight `previously_cached` train videos.
- Derive confirmation from the two remaining train videos.
- Select exactly three unique evidence videos per iteration and run all nine
  QAs.
- Rotate deterministically, prioritizing videos unused since the last
  confirmation.
- When all eight evidence videos have appeared, confirmation becomes due and
  the tracker resets only after an accept/rollback decision.
- Do not reserve separate regression videos. Use `correct_to_wrong` flips on
  the current three source videos.
- Updates between confirmations are provisional.
- Validation and test remain outside component-update feedback.

Checkpoint 2 implements only role derivation, rotation/coverage state,
provisional/confirmed schemas, the reusable three-video baseline phase, and
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
→ full-caption three rotating evidence videos
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
2. select exactly three source videos from the rotating evidence pool;
3. full-caption those three videos with the current policy;
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
videos_per_iteration: 3
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

### 3.3 Property proposal stage

Adapt `optimization/policies/llm_feedback.py` or add a narrowly scoped property
proposal policy.

Input per source video:

- all baseline QA outcomes;
- prioritize incorrect QA evidence;
- relevant reasoning excerpts;
- relevant incumbent captions and frames;
- current codebook.

Output:

- zero or more candidate properties;
- source video ID;
- source QA IDs;
- concise failure evidence;
- codebook coverage assessment;
- reusable property instruction.

Do not directly mutate the codebook.

### 3.4 Similarity retrieval

Implement property-conditioned retrieval for every candidate property:

```text
exact normalized candidate property text + sampled source-video segment frames
→ SigLIP frame-level cosine similarity
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

- load relevant trace contents, not only filesystem paths;
- include only `S_feedback` segments;
- resolve incumbent property IDs and candidate property to full text;
- include frames, concise frozen history, and before/after captions;
- enforce deterministic size limits and log truncation.

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

### 3.10 `optimization/prompt_bank_update_proposer.py`

Replace first-supported-item behavior with iteration-level aggregation across
all three current evidence videos and all candidate properties.

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

## 4. Required tests

Add focused tests for:

1. one iteration full-captions each of three selected videos once;
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
2. Add explicit three-video baseline iteration orchestration.
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
