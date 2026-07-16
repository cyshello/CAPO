# Phase 4 Runbook

## Active data policy

Older commands or artifacts that use three proposal videos plus separate
confirmation, regression, and final-evaluation train videos describe the
superseded matched pilot.

The frozen split remains:

```text
train:       10 videos / 30 QAs
validation:  10 videos / 30 QAs
test:        10 videos / 30 QAs
```

Checkpoint 2 derives eight `previously_cached` train videos as the evidence
pool and the other two train videos as confirmation. Each iteration selects
three unique evidence videos and runs all nine QAs. Confirmation runs only
after all eight evidence videos have appeared since the last confirmation.
There is no separate regression-video subset.

The main experiment, real feedback calls, component updates, and confirmation
evaluation are not available through Checkpoint 3D. Any older command that
launches those paths is superseded and must not be used as the active method.

Exact active train roles:

```text
Evidence:     0RxMZBLeqRI 7D-gxaie6UI GLW9omJfAdk TGom0uiW130
              pU_yyadYgG8 w0Wmc8C0Eq0 wCkQ138sg6M xKiRmesHWIA
Confirmation: g1VFfVsZt7w jIx5Zi84Z3Q
```

Checkpoint 2 focused tests:

```bash
conda run -n local_llm_vllm python -m pytest \
  tests/test_checkpoint2_train_roles.py \
  tests/test_checkpoint2_baseline_phase.py -q
```

Checkpoint 3B retrieval is authoritative and frame-only:

```text
exact candidate property text + sampled segment frames
→ SigLIP frame cosine scores
→ maximum frame pooling
→ top-M source-video segments (default M=5)
```

Frozen history is not part of the retrieval request, score, cache identity, or
resume identity. It remains used only by baseline routing, baseline captioning,
later selective re-captioning, and later multimodal feedback. Questions,
answers, correctness, traces, captions, and used segments are also prohibited
as direct retrieval inputs.

Checkpoint 3B artifacts:

```text
<iteration>/
├── property_proposals/<video_id>/model_artifacts/
│   ├── request.json
│   ├── raw_output.txt
│   ├── parsed_output.json
│   ├── rejections.json
│   └── completed.json
└── property_retrieval/<video_id>/
    ├── manifest.json
    └── <candidate_property_id>/retrieval.json
```

An exact completed proposal skips the optimization-LLM provider. An exact
retrieval artifact skips the SigLIP text encoder. Conflicting property text,
source video, visual index, model/sampling identity, top-k, pooling, or ranking
configuration fails closed rather than overwriting the artifact.

Checkpoint 3C consumes the frozen baseline and one Checkpoint 3B retrieval
artifact per candidate. Its artifact layout is:

```text
<iteration>/property_interventions/<video_id>/
├── manifest.json
└── <candidate_property_id>_p<property_text_hash>/
    ├── work_item.json
    ├── composed_prompts.jsonl
    ├── frozen_histories.jsonl
    ├── caption_cache_keys.jsonl
    ├── mixed_view/
    ├── qa/<question_id>/
    ├── transitions.json
    └── result.json
```

Every candidate is independent. The candidate is appended after the incumbent
routed properties only for `S_sim`; its temporary sequence may be one entry
over the router maximum. Exact frozen baseline histories are reused without
candidate-caption propagation. Unselected segments retain incumbent captions.
Any selected-segment caption/validation failure or prompt-budget overflow fails
the complete candidate without incumbent fallback and does not abort siblings.

Completed candidate results resume without captioning or QA calls when the
candidate ID/text, parent baseline, retrieval, composed prompts, frozen
histories, caption configuration, and QA configuration match. Any collision on
those identities fails closed. Mixed-view registries are rebuilt from restored
per-segment incumbent/candidate registries in temporal order.

Checkpoint 3C fixture tests do not invoke a GPU or external model:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint3c_property_intervention.py
```

Checkpoint 3D reads completed Checkpoint 3C artifacts and emits feedback only
for `wrong_to_correct` and `correct_to_wrong`. It computes, in retrieval order:

```text
S_feedback = S_sim ∩ (S_used_before ∪ S_used_after)
```

An empty `S_feedback` is an explicit rejection with no provider call. Unchanged
correctness remains in analysis records but never creates a feedback request.
The default request bounds are five segments, two frames per segment, 64 KiB
per encoded frame, three history items, 1,200 characters per caption, three
reasoning events per side, four relevant codebook entries, and 400,000 total
serialized characters. All limits are configurable through
`FeedbackEvidenceBounds`.

Frames are EXIF-normalized, converted to RGB, resized with deterministic
LANCZOS sampling, and encoded using a fixed JPEG quality ladder and subsampling
policy. `FrameTransformConfig`, the Pillow version, source hash, transformed
hash, dimensions, selected quality, and resize step are persisted in every
frame payload and in run identity. A frame is rejected only if the configured
quality and resize ladder is exhausted while it remains over the byte limit.

```text
<iteration>/property_feedback/
├── manifest.json
└── <video_id>/<candidate_property_id>_p<property_text_hash>/
    ├── input_identity.json
    ├── qa/<question_id>/
    │   ├── request.json
    │   ├── raw_output.txt
    │   ├── parsed_output.json
    │   └── rejection.json
    └── result.json
```

`result.json` aggregates positive/negative flip counts, source and flip-QA
lineage, attributed segments, accepted evidence references, codebook coverage,
and recommendation evidence. It never applies a recommendation. Exact complete
or per-QA resume performs no provider call; conflicting or partial artifacts
fail closed. Failed property results are immutable and resume by default.
Passing `retry_failed=True` creates an isolated
`retries/retry_001/` artifact tree and never overwrites the original failure.

Checkpoint 3D fixture tests use only mocked feedback and fixture frame payloads:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint3d_interventional_feedback.py
```

Checkpoint 2 baseline artifacts:

```text
<iteration>/
├── policy_snapshot/
├── coverage/{before,after}.json
├── baseline/<video_id>/
│   ├── routing/
│   ├── caption_view/
│   ├── frames.json
│   ├── frozen_histories.jsonl
│   ├── baseline_qas.jsonl
│   ├── qa/<question_id>/
│   └── video_complete.json
├── property_proposals/<video_id>.json
├── iteration_state/provisional.json
└── manifest.json
```

A completed `manifest.json` with the same frozen-input fingerprint returns
without routing, captioning, QA, or proposal calls. Partial resume skips only
videos whose `video_complete.json` fingerprint matches and whose referenced
artifacts all still exist. A conflicting fingerprint or incomplete completion
marker fails closed rather than overwriting results.

## 1. Repository and environment

```bash
cd /home/intern/youngseo/surrogate_rollout
git status
git diff
git log --oneline -5
```

Known test environment:

```bash
conda run -n local_llm_vllm python -m pytest tests
```

Confirm the actual working directory and import path.

The fixture smoke requires no model credentials or GPU. A later real run must
set the GPU and feedback model explicitly; the API key is read by the selected
provider and is never persisted:

```bash
export CUDA_VISIBLE_DEVICES=0
export SR_FEEDBACK_MODEL=gpt-4o
export OPENAI_API_KEY='<set in the shell only>'
```

Do not persist secrets in configs or artifacts.

## 2. Focused tests

The final command must cover:

- three-video baseline orchestration;
- per-video multi-property proposals;
- frame-only property-conditioned retrieval;
- independent frozen-history interventions;
- all-QA reruns;
- flip-only feedback;
- multi-property codebook update;
- mixed-view/cache regression.

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint3e_final_iteration.py \
  tests/test_checkpoint3d_interventional_feedback.py \
  tests/test_checkpoint3c_property_intervention.py \
  tests/test_checkpoint3b_property_proposal.py \
  tests/test_checkpoint3b_property_retrieval.py \
  tests/test_checkpoint3a_history_aware_baseline.py \
  tests/test_checkpoint2_baseline_phase.py \
  tests/test_checkpoint2_train_roles.py
```

## 3. Complete regression suite

```bash
conda run -n local_llm_vllm python -m pytest tests
```

Record the exact pass count.

## 4. Offline dry run

The dry run must build one iteration plan without captioning, QA APIs, or
persistent state mutation.

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint3e_final_iteration.py::test_fixture_end_to_end_accept_reject_rollback_and_resume
```

Confirm that artifacts show:

- three source videos;
- multiple candidate properties allowed per video;
- one work item per property-source-video pair;
- isolated intervention output paths;
- frozen policy/history versions;
- no scaffold candidate;
- no main experiment launch.

## 5. One-video bounded smoke

Prerequisite: export `OPENAI_API_KEY`. All three commands use evidence video
`0RxMZBLeqRI` and its three frozen train QAs, `max_proposals=1`, retrieval
top-k one, and at most one intervention. Output, state, and cache roots are
separate siblings. Reuse the same three paths when progressing modes.

QA only:

```bash
conda run -n local_llm_vllm python \
  scripts/run_phase4_bounded_smoke.py \
  --post-intervention-mode qa_only \
  --video-id 0RxMZBLeqRI --gpu 0 \
  --output-dir runs/phase4_one_video_smoke_output \
  --state-dir runs/phase4_one_video_smoke_state \
  --cache-dir runs/phase4_one_video_smoke_cache
```

Continue through flip-only feedback and property aggregation:

```bash
conda run -n local_llm_vllm python \
  scripts/run_phase4_bounded_smoke.py \
  --post-intervention-mode feedback_only \
  --video-id 0RxMZBLeqRI --gpu 0 \
  --output-dir runs/phase4_one_video_smoke_output \
  --state-dir runs/phase4_one_video_smoke_state \
  --cache-dir runs/phase4_one_video_smoke_cache
```

Continue through isolated provisional bank/router artifacts:

```bash
conda run -n local_llm_vllm python \
  scripts/run_phase4_bounded_smoke.py \
  --post-intervention-mode provisional_update \
  --video-id 0RxMZBLeqRI --gpu 0 \
  --output-dir runs/phase4_one_video_smoke_output \
  --state-dir runs/phase4_one_video_smoke_state \
  --cache-dir runs/phase4_one_video_smoke_cache
```

Each invocation writes `run_manifest.json` and an immutable mode manifest under
`mode_manifests/`. Later modes reuse completed baseline, proposal, retrieval,
intervention, QA, and feedback artifacts. Repeating a completed mode performs
an exact resume. Calling an earlier mode does not remove or overwrite later
artifacts. `state-dir/provisional_update/` is smoke-local: no coverage-cycle,
confirmed checkpoint, confirmation evaluation, or canonical pointer is used.

## 6. Main experiment

No automatic main-experiment CLI is provided. The active execution boundary is
`Checkpoint3EOrchestrator.run`. Construct it through
`Checkpoint3EOrchestrator.with_real_confirmation(...)`, which wires
`HistoryAwareDVDConfirmationEvaluator` to the existing sequential
history-aware caption builder and DVD QA path. The caller still explicitly
freezes model/provider settings, component snapshots, coverage state, and
output root before a real run. The fixture smoke above is the minimal executable
command until that run configuration is reviewed.

Use the same configured history builder for evidence baselines and confirmation:

```python
history_builder = HistoryAwareBaselineCaptionViewBuilder.from_local_qwen()
baseline_runner = BaselinePhaseRunner(
    history_aware_builder=history_builder,
    proposal_policy=proposal_policy,
    property_retrieval_runner=property_retrieval_runner,
)
orchestrator = Checkpoint3EOrchestrator.with_real_confirmation(
    baseline_runner=baseline_runner,
    intervention_runner=intervention_runner,
    feedback_runner=feedback_runner,
    confirmation_kwargs={
        "sample_loader": sample_loader,
        "history_aware_builder": history_builder,
        "base_prompt_template": base_prompt_template,
        "merge_prompt": merge_prompt,
        "sample_source_identity": split_manifest_hash,
        "cache_root": confirmation_cache_root,
        "cache_manifest_path": confirmation_cache_manifest_path,
        "history_block_seconds": 300.0,
        "max_history_captions": 30,
        "dvd_max_iterations": 10,
        "gpu": gpu,
        "downstream_qa_configuration": frozen_dvd_configuration,
    },
)
```

Every real `run()` invocation performs a fail-closed model audit before resume
resolution or any stage call. Standard output begins with one
`STARTUP_MODELS {...}` JSON line, and the same payload is written immutably to
`<output_dir>/startup_models.json`. It names the router, captioner, property
proposer, feedback provider, and downstream QA models. The active real path
requires the local Qwen history-aware router/captioner, OpenAI API property and
feedback providers, and the existing DVD QA runner. A missing model identity,
mock/fixture/stub implementation, provider mismatch, or a changed startup
manifest aborts the run before captioning, proposal, feedback, or QA calls.
Resume prints the same line again and validates it against the saved file.

The fixture-only concrete evaluator command is:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint3e_confirmation_evaluator.py
```

### 6.1 Coverage-cycle commands

The configured driver must use one stable `state_dir` for every iteration and
one distinct `output_dir` per iteration. These are the exact orchestrator call
contracts; `<configured-driver>` is responsible only for constructing the four
reviewed stage runners and loading the serialized arguments.

First iteration after confirmed checkpoint `C0` (explicit confirmed pair):

```python
result = orchestrator.run(
    iteration_id="cycle-0000-iteration-01",
    roles=roles,
    coverage_state=coverage_state,
    parent_confirmed=confirmed_c0,
    prompt_bank=confirmed_bank_c0,
    router_policy=confirmed_router_c0,
    confirmed_prompt_bank=confirmed_bank_c0,
    confirmed_router_policy=confirmed_router_c0,
    scaffold_policy=fixed_scaffold,
    scaffold_contract=fixed_contract,
    state_dir="<run-root>/policy_state",
    output_dir="<run-root>/iterations/cycle-0000-iteration-01",
    execution_identity=frozen_execution_identity,
    baseline_kwargs=baseline_kwargs,
    intervention_kwargs=intervention_kwargs,
)
```

Subsequent iteration in the same cycle (bank/router omitted deliberately):

```python
result = orchestrator.run(
    iteration_id="cycle-0000-iteration-02",
    roles=roles,
    coverage_state=previous_result.next_coverage_state,
    parent_confirmed=confirmed_c0,
    scaffold_policy=fixed_scaffold,
    scaffold_contract=fixed_contract,
    state_dir="<run-root>/policy_state",
    output_dir="<run-root>/iterations/cycle-0000-iteration-02",
    execution_identity=frozen_execution_identity,
    baseline_kwargs=baseline_kwargs,
    intervention_kwargs=intervention_kwargs,
)
```

The second call resolves the exact bank/router pair referenced by
`coverage_cycles/cycle_0000/active_provisional.json`. It never scans sibling
directories or chooses a file by modification time. Supplying any explicit
bank/router pair while this reference is active fails closed.

Confirmation is not a separate feedback command. Use the same subsequent-call
form for the iteration that completes coverage. The orchestrator evaluates the
latest provisional pair against `C0`, closes the active reference as `accepted`
or `rejected`, resets coverage, and atomically writes `confirmed/current.json`.

Resume command: rerun the exact call for the same `iteration_id`, input
coverage state, `output_dir`, `state_dir`, parent checkpoint, stage identities,
and execution identity. A matching completed manifest returns without stage
calls. For a completed provisional iteration, this also verifies that the
cycle-local reference still names that iteration and its complete lineage.

## 7. Expected artifact layout

```text
<iteration>/
├── manifest.json
├── baseline_stage/
│   ├── policy_snapshot/
│   ├── baseline/<video_id>/
│   ├── property_proposals/
│   ├── property_retrieval/
│   └── manifest.json
├── interventions/<video_id>/
├── feedback/<video_id>/
├── next_state/
│   ├── update_plan.json
│   ├── provisional_bank.json
│   ├── provisional_router.json
│   ├── fixed_scaffold_reference.json
│   └── provisional_state.json
└── confirmation/                  # only at a completed coverage cycle
    ├── evaluation.json
    ├── decision.json
    ├── active_confirmed_checkpoint.json
    ├── history_aware_evaluator/
    │   ├── input_bundle.json
    │   ├── manifest.json
    │   ├── parent/
    │   │   ├── caption_state.json
    │   │   ├── videos/<video_id>/
    │   │   └── qa/<question_id>/
    │   └── candidate/
    │       ├── caption_state.json
    │       ├── videos/<video_id>/
    │       └── qa/<question_id>/
    ├── rollback_bank.json         # rejection only
    └── rollback_router.json       # rejection only

<state_dir>/
├── confirmed/
│   ├── current.json               # atomic canonical confirmed pointer
│   └── checkpoints/<checkpoint_id>/
│       ├── checkpoint.json
│       ├── bank.json
│       └── router.json
└── coverage_cycles/cycle_<NNNN>/
    └── active_provisional.json    # atomic active/accepted/rejected reference
```

At minimum locate:

- iteration policy snapshot;
- complete baseline captions and histories;
- baseline QA outputs and traces;
- per-video property proposals;
- per-property retrieval results;
- per-property mixed caption views;
- all rerun QA outputs;
- QA-level and property-level feedback;
- aggregated codebook/router update;
- next-state versions;
- token, latency, GPU, and cache summaries.

## 8. Resume and recovery

The top-level resume key hashes the parent confirmed checkpoint, current and
confirmed component snapshots, coverage state, train roles, explicit execution
identity, stage configuration identities, fixed-scaffold setting, and update
thresholds. A matching completed manifest returns before calling baseline,
proposal, retrieval, caption, intervention, feedback, QA, or confirmation
stages. Stage runners retain their own finer-grained resume contracts.

Every final update artifact is write-once. A mismatched top-level manifest,
incomplete completion marker, or conflicting immutable artifact fails closed.
Provisional snapshots never update canonical `confirmed/current.json`.
`active_provisional.json` is a separate cycle-local atomic reference containing
the cycle ID, parent confirmed ID and hashes, exact active bank/router paths,
versions and hashes, ordered lineage, and coverage-state hash. Each provisional
state records the same parent confirmed checkpoint and accumulated lineage.
Confirmation accepts the complete latest bank/router pair or restores the exact
bank and router snapshots already referenced by the parent pointer. Acceptance
or rejection closes the cycle-local reference; the next cycle starts only from
the resulting canonical confirmed pointer.

The confirmation bundle is write-once and shared by parent and candidate. It
fixes video/QA IDs, ordered segments and timestamps, sampled-frame paths and
content hashes, transcripts, prompt text/hashes, model/backend/decoding,
sampling and history configuration, fixed scaffold/contract versions, and DVD
QA configuration. Parent and candidate construct separate sequential on-policy
histories. Their caption cache records may share one physical root, but reuse
requires equality of the complete history-aware key and bundle hash. A changed
frame, sampling setting, model/backend, decoding setting, history setting,
component version, composed prompt, or history resolves to a distinct identity
or fails closed before QA.

## 9. Success criteria

A healthy run satisfies:

- current policy is frozen within an iteration;
- each of the three batch videos has one complete incumbent caption view;
- one video may generate multiple candidate properties;
- every candidate retains source-video and source-QA lineage;
- every property intervention is independent;
- candidate properties are force-applied, not prematurely routed;
- frozen history is reused;
- unselected captions equal incumbent captions;
- selected-segment failures fail the candidate explicitly without incumbent
  fallback;
- all source-video QAs rerun for each property;
- only correctness flips enter optimization feedback;
- feedback evidence is concise;
- multiple supported properties may be accepted;
- codebook/router update cites intervention evidence;
- scaffold version does not change.
