# Prompt-Delta Meta-Prompt Optimization

## 0. Status

This document defines the **new prompt-delta optimization path**.

The existing property-codebook documents and implementation remain unchanged and must continue to work as a reproducible legacy baseline. Migration to this path must be incremental. Do not delete, rename, or silently repurpose legacy artifacts until the new path passes its own smoke tests and confirmation checks.

This specification intentionally defines only the minimum implementation required for one complete optimization iteration.

---

## 1. Objective

We optimize a meta-prompt that governs a frozen multimodal prompt generator.

\[
p_i = G_M(F_i, H_i)
\]

- `M`: persistent meta-prompt being optimized
- `G`: frozen multimodal prompt generator
- `F_i`: frames for the current clip
- `H_i`: bounded preceding caption history
- `p_i`: free-form captioning prompt generated for clip `i`

The optimization target is the **clip-and-history-to-prompt function**, not a persistent property codebook.

Prompt deltas are temporary interventions used to collect feedback. They are not reusable codebook entries and are never retrieved directly at inference time.

---

## 2. Minimal iteration

One iteration consists of the following steps.

1. Freeze the parent meta-prompt `M_t`.
2. Generate one free-form prompt `p_i` per clip from frames and bounded history.
3. Run one baseline full rollout and save captions, QA outputs, and trajectories.
4. Propose one or more ephemeral prompt deltas `delta_p` from training feedback.
5. For each delta, selectively recaption the chosen clips using `p_i + delta_p`.
6. Reuse all unaffected baseline captions and rerun every QA associated with the video.
7. Build one intervention episode per applied delta.
8. Generate compressed, evidence-linked feedback from the complete episode.
9. Use one or more episode feedback records to propose a candidate meta-prompt.
10. Save the candidate as provisional. Confirmation and promotion are separate operations.

The first implementation does not need to optimize clip selection, batch scheduling, or acceptance policy.

---

## 3. Component responsibilities

### 3.1 Free-form prompt generator

Input:

- current clip frames
- bounded preceding caption history
- current meta-prompt

Output:

- exactly one free-form captioning prompt for the current clip

The generator must not access:

- downstream questions
- answer choices
- ground-truth answers
- training intervention memory

### 3.2 Prompt-delta proposer

Input may include, during training only:

- question and answer choices
- ground-truth answer or correctness signal
- baseline answer
- baseline trajectory
- relevant frames and history
- generated prompt and baseline caption

Output:

- one or more temporary prompt deltas

A prompt delta is an executable correction appended to or composed with an already generated clip prompt. It must not be assigned a persistent property ID or inserted into a reusable codebook.

### 3.3 Selective intervention runner

For one prompt delta:

- apply the same delta to the selected generated prompts;
- recaption only the selected clips;
- reuse all unaffected baseline captions;
- rerun all QAs associated with the video using the same modified caption set.

The runner records set-level downstream outcomes. It must not claim that a particular recaptioned clip caused a QA transition.

### 3.4 Episode feedback generator

The feedback generator receives the complete textual intervention episode.

For every recaptioned clip, it may receive:

- segment ID and time range
- bounded history used by the generator
- base generated prompt
- applied prompt delta
- baseline caption
- intervention caption

For every QA, it may receive:

- question and answer choices
- ground-truth answer or correctness signal
- baseline and intervention answers
- baseline and intervention trajectories
- referenced or retrieved segment IDs, when available

Its output must:

- summarize what changed across the intervention scope;
- identify repeated relationships among history, generated prompts, prompt deltas, and caption changes;
- report positive outcomes and counterevidence;
- cite supporting segment and QA IDs;
- distinguish observed facts from hypotheses;
- state uncertainty when exact credit assignment is unavailable;
- recommend a generator-level strategy change rather than a reusable property.

It must not:

- identify a single causal clip unless independently established by an explicit atomic intervention;
- copy the prompt delta directly into the persistent meta-prompt as an unconditional rule;
- invent visual conditions that are not supported by the textual episode;
- modify the meta-prompt itself.

Frames are not required in the first feedback-generator implementation. The runtime generator and delta proposer remain multimodal.

### 3.5 Meta-prompt updater

Input:

- parent meta-prompt
- one or more compressed episode feedback records

Output:

- candidate revised meta-prompt
- concise revision rationale
- IDs of feedback records used

The updater should modify the procedure used to inspect frames and history and generate a clip-specific prompt. It must not build a hidden list of static properties.

### 3.6 Confirmation evaluator

Confirmation is a separate stage. It compares parent and candidate meta-prompts under matched conditions.

Matched conditions include:

- identical videos and frames
- identical generator and captioner models
- identical decoding settings
- identical history construction
- identical downstream inference configuration

The initial migration may stop after provisional candidate creation. It must not silently promote a candidate without an implemented confirmation policy.

---

## 4. Minimal data contracts

Only the following new records are required initially.

### 4.1 MetaPromptVersion

Required fields:

- `meta_prompt_id`
- `parent_meta_prompt_id`
- `text`
- `created_at`
- `status`: `parent`, `provisional`, `confirmed`, or `rejected`

### 4.2 PromptDelta

Required fields:

- `delta_id`
- `instruction`
- `source_qa_ids`
- `proposer_diagnosis`

No property ID, lifecycle action, merge target, or codebook membership field is allowed.

### 4.3 InterventionClipRecord

Required fields:

- `segment_id`
- `time_range`
- `history_snapshot`
- `base_prompt`
- `prompt_delta`
- `baseline_caption`
- `intervention_caption`

### 4.4 QAInterventionOutcome

Required fields:

- `qa_id`
- `is_source_qa`
- `baseline_answer`
- `intervention_answer`
- `baseline_correct`
- `intervention_correct`
- `baseline_trajectory_ref`
- `intervention_trajectory_ref`

Trajectory payloads should remain separate artifacts and be referenced rather than duplicated.

### 4.5 InterventionEpisode

Required fields:

- `episode_id`
- `video_id`
- `parent_meta_prompt_id`
- `prompt_delta`
- `clips`
- `qa_outcomes`
- `baseline_run_ref`
- `intervention_run_ref`

One episode represents one delta applied to one intervention scope.

### 4.6 EpisodeFeedback

Required fields:

- `feedback_id`
- `episode_id`
- `outcome_summary`
- `observations`
- `counterevidence`
- `generator_diagnosis`
- `recommended_strategy_change`
- `confidence`

Each observation and counterevidence item must include:

- statement
- supporting segment IDs
- supporting QA IDs
- evidence type: `caption_change`, `trajectory`, `qa_transition`, or `mixed`
- confidence

---

## 5. Validation invariants

The implementation must enforce the following invariants.

1. Every referenced segment ID and QA ID must exist in the corresponding episode.
2. A QA transition belongs to the intervention episode, not automatically to each recaptioned clip.
3. Missing trajectories must be represented explicitly as unavailable, not fabricated or replaced by empty evidence claims.
4. Prompt deltas must remain ephemeral and must not be written into the legacy property codebook.
5. The updater must preserve parent-child lineage for every candidate meta-prompt.
6. A provisional candidate must not become active through file ordering, latest-file lookup, or other implicit behavior.
7. Legacy and prompt-delta artifacts must use distinct namespaces or output directories.
8. Existing legacy execution must remain unchanged unless the user explicitly requests migration of that component.

---

## 6. No-hardcoding policy

Codex or any implementation agent must not resolve an important design ambiguity by inventing a default.

The following decisions are considered design decisions and must not be hardcoded without an explicit value in configuration or this document:

- number of prompt deltas proposed per QA or video
- number or percentage of clips selected for intervention
- clip-selection or retrieval policy
- prompt-delta composition operator
- bounded-history length or time window
- whether intervention captions propagate into later history
- trajectory fields supplied to the feedback generator
- maximum episode size or truncation strategy
- feedback-generator model and decoding settings
- number of episode feedback records used per update
- meta-prompt revision size or editing strategy
- confirmation sample size
- confirmation acceptance, tie, and rollback rules
- automatic promotion of provisional candidates

Required behavior when such a value is missing:

1. reuse an already explicit legacy/config value only if semantics are unchanged;
2. otherwise expose the decision as a named configuration field;
3. fail fast with a clear `NotConfigured` or validation error;
4. document the unresolved decision in the run manifest.

Prohibited behavior:

- choosing a convenient constant inside implementation code;
- inferring a policy from fixture size;
- silently truncating records to fit a context window;
- using lexical similarity, property overlap, or placeholder heuristics unless explicitly configured;
- adding restrictive validation rules because the intended policy is unclear;
- treating example schema values as valid production values.

Every nontrivial default must be declared in one central configuration object and serialized into the run manifest.

---

## 7. Migration plan

Migration must proceed in small checkpoints.

### Checkpoint A — documentation and schemas

Add this document and the new records. Add schema round-trip and validation tests. Do not change runtime behavior.

### Checkpoint B — legacy artifact adapter

Convert an existing saved property-intervention artifact into an `InterventionEpisode` without rerunning the main experiment.

Temporary mapping:

- legacy candidate property → ephemeral prompt delta
- legacy selective intervention → intervention episode
- legacy QA transitions → QA intervention outcomes

The adapter must not write to the codebook.

### Checkpoint C — mock feedback path

Implement a deterministic mock feedback generator and produce a valid `EpisodeFeedback` from a saved fixture.

### Checkpoint D — real feedback generator

Replace the mock with an LLM-backed feedback generator. It may consume all clip histories, prompts, caption pairs, QA outcomes, and available trajectories. Add strict evidence-reference validation.

### Checkpoint E — provisional meta-prompt updater

Implement the updater and version store. Save candidates as provisional only. Do not implement automatic promotion yet.

### Checkpoint F — runtime generator migration

Only after the previous checkpoints pass, connect the free-form prompt generator as the active prompt source for the new mode.

Legacy property routing remains available as a baseline.

---

## 8. Initial smoke test

The first smoke test must use an existing saved intervention fixture and must not run the main experiment.

Success criteria:

- the fixture converts into one valid intervention episode;
- all clip histories, prompts, caption pairs, and QA transitions are preserved;
- trajectory references resolve or are explicitly marked unavailable;
- mock feedback contains valid supporting IDs;
- one provisional meta-prompt version is saved with correct lineage;
- no legacy codebook or router artifact changes;
- rerunning the smoke test is deterministic or safely idempotent.

Suggested interface:

```bash
python scripts/run_phase4_memory_iteration.py \
  --optimization-mode prompt_delta \
  --dry-run \
  --input-fixture <saved_intervention_fixture> \
  --output-dir runs/prompt_delta_smoke
```

The actual command, arguments, environment variables, output paths, and resume behavior must be documented after inspecting the existing CLI. Do not create parallel scripts unless the current entry point cannot support the new mode cleanly.

---

## 9. Explicit non-goals for the first implementation

Do not implement the following unless separately requested:

- deletion or broad renaming of legacy property code
- frame aggregation for the feedback generator
- learned clip attribution
- automatic response clustering
- intervention-memory retrieval at inference time
- codebook conversion or migration
- asynchronous optimization
- multi-node scheduling
- automatic main-experiment execution
- automatic candidate promotion
- elaborate dashboards or reports

The initial goal is only:

> saved baseline/intervention artifacts → intervention episode → evidence-linked feedback → provisional meta-prompt candidate

---

## 10. Open decisions

The following remain intentionally unresolved and must stay configurable or unimplemented until explicitly decided:

- final clip-selection policy
- prompt-delta composition semantics
- history propagation during intervention
- trajectory compression format
- LLM context-overflow strategy for large episodes
- number of episodes consolidated per meta-prompt update
- confirmation protocol and acceptance threshold
- whether a later multimodal feedback-generator ablation will receive frames

Do not treat these open decisions as permission to invent implementation defaults.
