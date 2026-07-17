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
pool and the other two train videos as confirmation. Each production iteration
selects `K` unique evidence videos and runs all `3K` QAs. `K=3` is the default
pilot setting, not a method constraint. Confirmation runs only
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
│   ├── input_identity.json
│   ├── provider_request.json
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

## Checkpoint 1 compact property-memory sidecar

Checkpoint 1 is a fixture-tested, model-free sidecar. It does not run
captioning, QA, proposal, retrieval, feedback, the updater, confirmation, or a
production iteration. Construct `CompactPropertyMemoryRunner` and call
`run(...)` only after the referenced baseline and intervention artifacts are
complete. For an updater-decided iteration, pass the already-written update
plan and resulting bank so candidate promotion is recorded after, rather than
decided by, the memory layer.

```python
from surrogate_rollout.optimization.property_memory import (
    CompactPropertyMemoryRunner,
)

memory_result = CompactPropertyMemoryRunner().run(
    iteration_id=iteration_id,
    iteration_ordinal=iteration_ordinal,
    prompt_bank=input_prompt_bank,
    baseline_video_manifest_paths=baseline_video_manifest_paths,
    intervention_manifest_paths=intervention_manifest_paths,
    feedback_manifest_paths=feedback_manifest_paths,
    output_dir=f"{iteration_output}/compact_property_memory",
    parent_memory_path=parent_memory_snapshot_path,  # None for seed bootstrap
    update_plan=completed_update_plan,               # optional
    update_plan_path=completed_update_plan_path,     # optional, hashed lineage
    resulting_prompt_bank=completed_resulting_bank,  # optional
)
```

Artifact layout:

```text
<iteration>/compact_property_memory/
├── manifest.json                         # property_memory_manifest_v1
├── compact_summaries/
│   ├── correct_qa_property_credit.jsonl  # property_compact_summary_v1
│   └── intervention_effects.jsonl        # property_compact_summary_v1
└── property_memory/
    ├── snapshot.json                     # property_memory_v1 parent unit
    ├── selection_audit.json
    ├── properties/<property_id>.json
    └── candidates/<video_id>__<candidate_id>.json
```

Defaults are three strong and two weak positives, three harmful, two no-effect,
and two positive/two negative routing examples. Candidate positive, negative,
mixed, and no-effect categories use the corresponding small bounds. Ranking is
strength, distinct video, representative-signature diversity, then recency.
The selection audit explains retention and eviction; evicted compact examples
do not delete raw artifacts or immutable summary rows.

Every example contains source paths and SHA-256 hashes. The manifest binds raw
manifest hashes, parent snapshot hash, input/result bank hashes, optional
update-plan content/path hash, schema versions, bounds, and selection version.
Repeat the exact call for exact resume. A partial output directory, a changed
source or parent, an incompatible memory schema, or a missing/hash-mismatched
completed artifact fails closed. Do not point `parent_memory_path` at a legacy
Phase 4 artifact; legacy artifacts remain raw inputs and are not overwritten.

Focused fixture-only command:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint1_property_memory.py
```

The memory-conditioned LLM codebook updater is the following checkpoint below.

## Checkpoint 2 memory-conditioned LLM codebook updater

Configure the existing orchestrator with `CompactPropertyMemoryRunner` and
`MemoryConditionedLLMCodebookUpdater`, then invoke the post-feedback checkpoint
with already-completed iteration manifests. Tests use a callable mock provider;
no real provider is configured or called by this checkpoint implementation.

```python
orchestrator = Checkpoint3EOrchestrator(
    baseline_runner=baseline_runner,
    intervention_runner=intervention_runner,
    feedback_runner=feedback_runner,
    confirmation_evaluator=confirmation_evaluator,
    property_memory_runner=CompactPropertyMemoryRunner(),
    llm_codebook_updater=MemoryConditionedLLMCodebookUpdater(
        response_provider=reviewed_codebook_provider,
    ),
)

result = orchestrator.run_memory_conditioned_codebook_checkpoint(
    iteration_id=iteration_id,
    iteration_ordinal=iteration_ordinal,
    prompt_bank=input_prompt_bank,
    baseline_video_manifest_paths=baseline_video_manifest_paths,
    intervention_manifest_paths=intervention_manifest_paths,
    feedback_manifest_paths=feedback_manifest_paths,
    output_dir=f"{iteration_output}/memory_codebook_checkpoint",
    state_dir=policy_state_dir,
)
```

Prompt:

```text
optimization/prompts/codebook_updater_v1.txt
memory_codebook_updater_prompt_v1
```

Artifacts:

```text
<iteration>/memory_codebook_checkpoint/
├── manifest.json
├── compact_property_memory/
│   ├── manifest.json
│   ├── compact_summaries/
│   └── property_memory/snapshot.json
└── llm_codebook_updater/
    ├── manifest.json
    ├── system_prompt.txt
    ├── request.json
    ├── raw_response.txt
    ├── parsed_plan.json
    ├── validation_report.json
    ├── rejected_actions.json
    ├── applied_plan.json
    ├── candidate_codebook.json
    ├── property_id_mapping.json
    └── promoted_property_memory.json

<state_dir>/property_memory/current.json
```

`property_id_mapping.json` contains:

```json
{
  "schema_version": "property_id_mapping_v1",
  "old_to_new_property_ids": {
    "unchanged_or_canonical_id": "same_or_canonical_id",
    "retired_id": null
  },
  "candidate_promotions": {
    "candidate_id": "validated_active_property_id"
  }
}
```

The state pointer tracks only compact-memory lineage for the unchanged active
input bank. It is not `confirmed/current.json` or
`active_provisional.json`. Candidate codebook output is not a production policy
and must not be passed to confirmation before the deferred router updater builds
and validates the matching router.

Focused mock-only command:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint1_property_memory.py \
  tests/test_checkpoint2_memory_codebook_updater.py
```

## Checkpoint 3 memory-conditioned router prompt and atomic pair

After the Checkpoint 2 manifest is complete, configure the same orchestrator
with `MemoryConditionedLLMRouterUpdater` and invoke the router checkpoint. This
is a mock/provider integration boundary; this implementation task does not run
a real provider, GPU, confirmation, or regular production iteration.

```python
orchestrator = Checkpoint3EOrchestrator(
    baseline_runner=baseline_runner,
    intervention_runner=intervention_runner,
    feedback_runner=feedback_runner,
    confirmation_evaluator=confirmation_evaluator,
    llm_router_updater=MemoryConditionedLLMRouterUpdater(
        response_provider=reviewed_router_update_provider,
    ),
)

result = orchestrator.run_memory_conditioned_router_checkpoint(
    iteration_id=iteration_id,
    parent_router_policy=input_router_policy,
    codebook_checkpoint_manifest_path=(
        f"{iteration_output}/memory_codebook_checkpoint/manifest.json"),
    output_dir=f"{iteration_output}/memory_router_checkpoint",
    state_dir=policy_state_dir,
)
```

Prompt and versions:

```text
optimization/prompts/router_updater_v1.txt
memory_router_updater_prompt_v1
structured_router_policy_v1
history_aware_router_prompt_renderer_v1
rendered_router_prompt_v1
```

Artifacts:

```text
<iteration>/memory_router_checkpoint/
├── manifest.json                         # written only after atomic success
├── failure.json                          # failure only; no pair pointer
├── llm_router_updater/
│   ├── manifest.json
│   ├── parent_structured_router_policy.json
│   ├── system_prompt.txt
│   ├── request.json
│   ├── raw_response.txt
│   ├── parsed_plan.json
│   ├── validation_report.json
│   ├── rejected_actions.json
│   ├── applied_plan.json
│   ├── structured_router_policy.json
│   ├── rendered_router_prompt.json
│   ├── rendered_router_prompt.txt
│   ├── router_prompt.diff
│   └── candidate_router_policy.json
└── provisional_policy_pair/
    ├── provisional_codebook.json
    ├── provisional_router.json
    └── policy_pair.json                  # atomic_provisional_policy_pair_v1

<state_dir>/memory_conditioned_provisional/current.json
```

The updater accepts only bounded memory/effect IDs and targets active IDs in
the candidate codebook. Selection guidance and positive examples require
positive demonstrated effects; avoidance guidance and negative examples
require negative demonstrated effects. The validator rejects stale IDs,
unsupported examples, question/answer/gold/prediction/reasoning leakage,
conflicting writes, retired guidance, mapping inconsistencies, protocol or
selection-limit changes, and rendered-prompt hash mismatches.

The provisional pair references/hashes its parent pair, property memory,
candidate codebook, complete ID mapping, structured policy, rendered prompt,
and both updater plans and validation reports. The final manifest and separate
pointer are written only after both components validate in one property-ID
space. A failure preserves all diagnostic artifacts, does not write the pair
pointer, and cannot resume as success. Exact repeat of a completed invocation
verifies closure and performs no provider call. This path never modifies
`confirmed/current.json` or coverage-cycle `active_provisional.json`.

Focused mock-only command:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint3_memory_router_updater.py
```

### One-video Checkpoint 3 real smoke

The bounded runner now opts into the memory-conditioned path only in
`scripts/run_phase4_bounded_smoke.py`; fixture and legacy callers retain the
deterministic provisional-update path unless they explicitly set
`memory_conditioned_update=True`. The real smoke uses one call each for the
codebook and router updater, then makes exactly one post-commit router call
through the still-active persistent GPU pool to prove that the rendered prompt
is consumed. Worker cleanup is recorded in `worker_cleanup.json`.

Use isolated output/state roots and the compatible read-only caption cache:

```bash
conda run -n local_llm_vllm python -m dotenv run -- python \
  scripts/run_phase4_bounded_smoke.py \
  --post-intervention-mode provisional_update \
  --video-id wCkQ138sg6M --gpu 4 --gpus 4,5 \
  --output-dir runs/phase4_memory_router_smoke_wCkQ138sg6M_v2_output \
  --state-dir runs/phase4_memory_router_smoke_wCkQ138sg6M_v2_state \
  --cache-dir runs/phase4_one_video_smoke_json_cache
```

Repeat that exact command for the resume audit. A successful repeat performs
no proposal, feedback, codebook-updater, router-updater, routing-probe,
captioning, or QA model call. Inspect:

```text
runs/phase4_memory_router_smoke_wCkQ138sg6M_v2_output/
├── mode_manifests/provisional_update.json
├── memory_codebook_checkpoint/
├── memory_router_checkpoint/
│   └── router_prompt_consumption_probe.json
└── worker_cleanup.json
```

`mode_manifests/provisional_update.json` records property-memory, both updater
manifests, the rendered-prompt hash, atomic pair paths, updater call counts,
and the before/after confirmed-pointer hash. The smoke fails if that pointer
changes or if the post-commit router call does not carry the new prompt hash.

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

- configurable-size evidence baseline orchestration;
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
  tests/test_startup_models.py \
  tests/test_bounded_smoke.py \
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

- the default three source videos;
- multiple candidate properties allowed per video;
- one work item per property-source-video pair;
- isolated intervention output paths;
- frozen policy/history versions;
- no scaffold candidate;
- no main experiment launch.

## 5. One-video bounded smoke

Prerequisite: store `OPENAI_API_KEY` in the repository-local `.env` with file
mode `600`. The commands use `python-dotenv` to inject it only into the child
process; they never print or persist the key. All three commands use evidence video
`0RxMZBLeqRI` and its three frozen train QAs, `max_proposals=1`, retrieval
top-k one, and at most one intervention. Output, state, and cache roots are
separate siblings. Reuse the same three paths when progressing modes.

The first 2026-07-17 pre-schema attempt reached real-model startup validation on GPU
4 but stopped on the first segment because unconstrained Qwen output was not
strict JSON. Its partial immutable roots are
`runs/phase4_one_video_smoke_{output,state}` and must not be reused or
overwritten. Router version `router_v9002` uses vLLM JSON Schema structured
decoding with fallback disabled.

The second 2026-07-17 attempt validated schema-constrained routing over all 184
segments and completed all three baseline QAs. It then stopped before the
OpenAI property-proposal call because the 71,767-character request exceeded the
40,000-character hard limit. Its partial immutable output/state roots are
`runs/phase4_one_video_smoke_json_{output,state}`. Proposer version
`multi_property_proposer_v2` now persists deterministic truncation metadata and
reduces that frozen request to 23,878 characters by retaining three reasoning
events per QA and twelve relevant captions. Use the new `*_json_v2_*`
output/state roots below. Reuse the identity-matched
`runs/phase4_one_video_smoke_json_cache`; completed caption entries are
immutable and avoid regenerating the same 184 captions.

The third 2026-07-17 `qa_only` invocation completed successfully at
`runs/phase4_one_video_smoke_json_v2_output/run_manifest.json`. It validated
184 schema-constrained router calls, 184 caption-cache hits, three baseline QAs,
the 23,783-character persisted proposal request, and zero confirmation calls.
The provider returned one proposal, which the strict proposal policy rejected
as already covered by active `pe_default`; therefore retrieval, intervention,
and candidate-QA work-item lists are valid but empty. This is an allowed
zero-proposal completion, not evidence that the real intervention path ran.

`multi_property_proposer_v3` supersedes that text-only proposal request. Its
model-visible input contains no source-video, question, segment, priority,
tool-call, or truncation identifiers. It sends, per QA, question and
answer/prediction context, three bounded sanitized reasoning events, and three
actual used-segment representative images paired with baseline captions, plus
the current codebook. Offline reconstruction from the v2 baseline produced a
13,388-character text payload and nine bounded image blocks. Use new
`*_multimodal_*` output/state roots below; reuse the identity-matched caption
cache.

`multi_property_proposer_v4` changes coverage handling without weakening
deterministic duplicate or knowledge-grounding checks. The provider field
`covered_by_existing_property_ids` now means possible relation or coverage and
is parsed and persisted as non-binding `coverage_hints`. Exact normalized text
matches and active-ID collisions still reject before retrieval, as do malformed
lineage, instance leakage, and non-visual/external/background/historical
knowledge instructions. Semantic coverage is assessed only from post-
intervention correctness-flip feedback using Checkpoint 3D
`coverage_assessment` and `covered_by_property_ids`. Existing serialized
`covered_by_existing_property_ids` proposal records remain readable through a
legacy alias; completed top-level bounded-smoke manifests retain their normal
exact-resume behavior.

`flip_only_property_feedback_v2` carries the proposal `coverage_hints` into
the bounded feedback request and aggregate artifact as context and lineage.
They do not populate `covered_by_property_ids`; only the post-intervention
feedback response can do that.

History-block-parallel baseline captioning is available through `--gpus`. The
comma-separated list creates at most one run-scoped persistent Qwen/vLLM worker
per GPU;
segments inside each 300-second history block stay sequential, while distinct
blocks run concurrently. `--gpu` selects the primary GPU for DVD QA; selective
caption interventions are sent back to the persistent caption pool so the
parent never loads a competing Qwen instance. Workers remain loaded across
baseline videos, proposals, retrieval, interventions, and confirmation, and
shut down only at the explicit run boundary. Each block resumes from
`history_aware_baseline/parallel_history_blocks/block_<index>/segment_state/`.
Workers write `worker_<index>_cache_manifest.jsonl` fragments, and only the
parent merges them into the configured caption manifest after all workers
succeed. `routing_manifest.json` records the scheduler version, worker/GPU and
block counts, and deterministic merge rule. GPU assignment is not part of the
semantic caption-cache key.

The first multi-GPU attempt on 2026-07-17 completed all 184 segment states but
stalled before the parent merge because the parent joined one-build workers
whose vLLM EngineCore processes remained alive. The run-scoped worker protocol
supersedes that lifecycle: the parent consumes completion messages without a
join, and the bounded runner closes workers in `finally`. Stop the old hung
invocation, then repeat the exact QA-only command below with the same three
roots. All completed block states and caption caches resume; do not delete or
rename them.

That resumed attempt reached QA but exposed a second lifecycle issue: DVD
`frame_inspect_tool` tried to initialize a parent Qwen on GPU 4 while the
persistent worker already occupied it. Two QAs failed with null predictions,
yet the older baseline contract treated them as ordinary incorrect answers and
called the proposer. The current adapter routes raw frame inspection through
the worker pool and fails before proposal on runtime errors, null predictions,
or parse failures. When the exact command below sees the invalid completed QA
mode, it moves only invalid QA/proposal/retrieval/intervention/mode artifacts to
`invalid_attempts/qa_execution_failure_<NNN>/`, leaves the 184 caption artifacts
in place, reruns all three QAs, and regenerates the proposal only after all
three succeed.

The repaired 2026-07-17 resume completed successfully in `qa_only` mode at
`runs/phase4_one_video_smoke_parallel_multimodal_output/run_manifest.json`.
Its merged routing manifest records 184 resumed segments, zero router calls,
and zero caption calls. All three QA executions returned non-null parsed
answers with empty error lists: QAs 9 and 11 were correct, and QA 10 was a
valid incorrect answer. Only then did the v3 proposer regenerate its request
and provider artifacts. The provider returned `pe_historical_context`, which
the strict policy rejected as covered by active `pe_default`, so the completed
mode contains zero accepted proposals and no intervention. The archived failed
attempt remains recoverable under
`invalid_attempts/qa_execution_failure_001/`. Confirmation was disabled and no
production state or coverage pointer was written.

Secure the environment file once:

```bash
chmod 600 .env
```

QA only:

```bash
conda run -n local_llm_vllm python -m dotenv run -- python \
  scripts/run_phase4_bounded_smoke.py \
  --post-intervention-mode qa_only \
  --video-id 0RxMZBLeqRI --gpu 4 --gpus 4,5,6,7 \
  --output-dir runs/phase4_one_video_smoke_parallel_multimodal_output \
  --state-dir runs/phase4_one_video_smoke_parallel_multimodal_state \
  --cache-dir runs/phase4_one_video_smoke_json_cache
```

Continue through flip-only feedback and property aggregation:

```bash
conda run -n local_llm_vllm python -m dotenv run -- python \
  scripts/run_phase4_bounded_smoke.py \
  --post-intervention-mode feedback_only \
  --video-id 0RxMZBLeqRI --gpu 4 --gpus 4,5,6,7 \
  --output-dir runs/phase4_one_video_smoke_parallel_multimodal_output \
  --state-dir runs/phase4_one_video_smoke_parallel_multimodal_state \
  --cache-dir runs/phase4_one_video_smoke_json_cache
```

Continue through isolated provisional bank/router artifacts:

```bash
conda run -n local_llm_vllm python -m dotenv run -- python \
  scripts/run_phase4_bounded_smoke.py \
  --post-intervention-mode provisional_update \
  --video-id 0RxMZBLeqRI --gpu 4 --gpus 4,5,6,7 \
  --output-dir runs/phase4_one_video_smoke_parallel_multimodal_output \
  --state-dir runs/phase4_one_video_smoke_parallel_multimodal_state \
  --cache-dir runs/phase4_one_video_smoke_json_cache
```

Each invocation writes `run_manifest.json` and an immutable mode manifest under
`mode_manifests/`. Later modes reuse completed baseline, proposal, retrieval,
intervention, QA, and feedback artifacts. Repeating a completed mode performs
an exact resume. Calling an earlier mode does not remove or overwrite later
artifacts. `state-dir/provisional_update/` is smoke-local: no coverage-cycle,
confirmed checkpoint, confirmation evaluation, or canonical pointer is used.
The bounded smoke keeps its shared SigLIP image/text retrieval embedder on CPU;
GPUs listed by `--gpus` remain reserved for the persistent Qwen worker pool.

## 6. Production memory-conditioned iteration launcher

The active launcher is `scripts/run_phase4_memory_iteration.py`. It constructs
the latest Checkpoint 3E component path and does not import or wrap the obsolete
Stage 4.13/4.14 launcher:

```text
baseline/intervention/feedback artifacts
→ property_memory_v1
→ LLM codebook plan and candidate codebook/ID mapping
→ LLM router plan and rendered real-router prompt
→ atomic provisional codebook/router pair
```

It never runs confirmation and never writes `confirmed/current.json`.
Production experiments remain manually launched by the user.

CLI selection and scheduling:

- `--num-videos K`: logical iteration size; omitted means `K=3` unless
  `--video-ids` is supplied;
- `--video-ids id1,id2,...`: explicit ordered evidence list; if
  `--num-videos` is also supplied, the count must agree;
- `--max-parallel-videos P`: deterministic video-wave width;
- `--gpus 4,5,6,7`: unique iteration-scoped persistent worker set;
- `--selection-seed`: deterministic initial rotation offset, default zero;
- `--dry-run-plan`: save selection, waves, identities, paths, and expected
  stages with zero model calls.

`K` and `P` are independent. For example, `K=8, P=4` produces two ordered
four-video waves. `P` may not exceed the usable worker count. The launcher
rejects duplicate/unavailable/busy GPUs before model startup.

Dry-run plan for the default three-video pilot:

```bash
conda run --no-capture-output -n local_llm_vllm \
  python -m dotenv run -- python -u \
  scripts/run_phase4_memory_iteration.py \
  --dry-run-plan \
  --iteration-id phase4-memory-pilot-k3-plan \
  --num-videos 3 --max-parallel-videos 3 --gpus 4,5,6,7 \
  --output-dir runs/phase4_memory_pilot_k3_plan_output \
  --state-dir runs/phase4_memory_pilot_k3_plan_state \
  --cache-dir runs/phase4_memory_pilot_k3_plan_cache
```

Real `K=3` pilot:

```bash
conda run --no-capture-output -n local_llm_vllm \
  python -m dotenv run -- python -u \
  scripts/run_phase4_memory_iteration.py \
  --iteration-id phase4-memory-pilot-k3-iteration-001 \
  --num-videos 3 --max-parallel-videos 3 --gpus 4,5,6,7 \
  --output-dir runs/phase4_memory_pilot_k3_output \
  --state-dir runs/phase4_memory_pilot_k3_state \
  --cache-dir runs/phase4_memory_pilot_k3_cache
```

Real `K=5` pilot:

```bash
conda run --no-capture-output -n local_llm_vllm \
  python -m dotenv run -- python -u \
  scripts/run_phase4_memory_iteration.py \
  --iteration-id phase4-memory-pilot-k5-iteration-001 \
  --num-videos 5 --max-parallel-videos 4 --gpus 4,5,6,7 \
  --output-dir runs/phase4_memory_pilot_k5_output \
  --state-dir runs/phase4_memory_pilot_k5_state \
  --cache-dir runs/phase4_memory_pilot_k5_cache
```

Full evidence pool in two four-video waves:

```bash
conda run --no-capture-output -n local_llm_vllm \
  python -m dotenv run -- python -u \
  scripts/run_phase4_memory_iteration.py \
  --iteration-id phase4-memory-full-k8-iteration-001 \
  --num-videos 8 --max-parallel-videos 4 --gpus 4,5,6,7 \
  --output-dir runs/phase4_memory_full_k8_output \
  --state-dir runs/phase4_memory_full_k8_state \
  --cache-dir runs/phase4_memory_full_k8_cache
```

The examples intentionally use distinct state roots; they are alternatives,
not three sequential calls in one coverage cycle. To resume, repeat the exact
same command. The launcher restores the ordered videos, `K`, parent pair, and
coverage input from `iteration_identity.json` before consulting newer state
pointers. Any changed ordered list, count, seed, split hash, parent identity,
GPU/model/decoding configuration, updater version, or prompt version fails
closed. A completed repeat returns before worker startup and performs no model
or updater call.

Artifacts:

```text
<output_dir>/
├── iteration_identity.json
├── iteration_plan.json
├── startup_models.json                  # real run only
├── baseline_videos/<video_id>/
├── baseline_batch_manifest.json
├── interventions/<video_id>/
├── feedback/<video_id>/
├── memory_codebook_checkpoint/
├── memory_router_checkpoint/
│   └── provisional_policy_pair/
├── worker_cleanup/attempt_<NNN>.json
└── manifest.json                        # only after atomic pair success

<state_dir>/
├── property_memory/current.json
├── memory_conditioned_provisional/current.json
└── production_selection/current.json
```

The worker cleanup artifact is written from `finally` on success, failure,
interruption, and completed resume. Router/update failure cannot create the
top-level completed manifest or a codebook-only policy pair. Raw stage
artifacts remain in place for diagnosis and exact stage resume.

The underlying active execution boundary remains
`Checkpoint3EOrchestrator.with_real_confirmation(...)`, which wires
`HistoryAwareDVDConfirmationEvaluator` to the same history-aware builder and
DVD QA path while injecting the property-memory and both LLM updater stages.

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
- each of the selected `K` batch videos has one complete incumbent caption view;
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
