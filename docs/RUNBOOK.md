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

The main experiment, real VLM router, property interventions, feedback calls,
and confirmation evaluation are not available at Checkpoint 2. Any older
command that launches those paths is superseded and must not be used as the
active method.

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

> The coding agent must inspect the actual CLI and replace every
> `INSPECT_AND_FILL` marker before declaring the implementation complete.

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

Required environment variables:

```bash
# INSPECT_AND_FILL: exact router, captioner, feedback, and downstream model vars.
export CUDA_VISIBLE_DEVICES=INSPECT_AND_FILL
```

Do not persist secrets in configs or artifacts.

## 2. Focused tests

The final command must cover:

- three-video baseline orchestration;
- per-video multi-property proposals;
- property-conditioned retrieval;
- independent frozen-history interventions;
- all-QA reruns;
- flip-only feedback;
- multi-property codebook update;
- mixed-view/cache regression.

```bash
INSPECT_AND_FILL
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
INSPECT_AND_FILL
```

Confirm that artifacts show:

- three source videos;
- multiple candidate properties allowed per video;
- one work item per property-source-video pair;
- isolated intervention output paths;
- frozen policy/history versions;
- no scaffold candidate;
- no main experiment launch.

## 5. Small bounded smoke test

The user may run:

```bash
INSPECT_AND_FILL
```

The coding agent may run it only if its GPU/API cost is explicitly bounded.

Expected checks:

- each source video is full-captioned once;
- all baseline QAs run;
- candidate properties retain source lineage;
- each property retrieves its own `S_sim`;
- candidate property is force-added only to selected segments;
- candidate runs share incumbent history and do not affect each other;
- all source-video QAs rerun for each intervention;
- only correctness flips enter feedback;
- `S_sim`, `S_used`, `S_usedagain`, and `S_feedback` are saved;
- codebook/router update occurs only after all work items finish;
- scaffold version remains unchanged.

## 6. Main experiment

The user executes the final command manually.

```bash
INSPECT_AND_FILL
```

Document every argument, including:

- videos per iteration (fixed at three);
- number of iterations;
- maximum property proposals per video;
- maximum parallel property interventions;
- segment retrieval budget;
- history block size;
- model IDs;
- GPU assignment;
- output root;
- resume/overwrite behavior.

## 7. Expected artifact layout

Replace with the actual implementation layout:

```text
INSPECT_AND_FILL
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

```bash
INSPECT_AND_FILL
```

Document:

- resume key and exact command;
- completed work-item detection;
- partial JSONL recovery;
- behavior after caption, QA, or feedback failure;
- collision-safe output paths;
- prevention of accidental overwrite.

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
- failures have explicit incumbent fallback records;
- all source-video QAs rerun for each property;
- only correctness flips enter optimization feedback;
- feedback evidence is concise;
- multiple supported properties may be accepted;
- codebook/router update cites intervention evidence;
- scaffold version does not change.
