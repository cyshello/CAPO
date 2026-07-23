# caption_prompt_opt

An independent optimization path that runs the **same** Checkpoint F/G loop as
`surrogate_rollout.optimization`, but optimizes the **caption prompt itself**
(the task body the captioner receives) instead of the meta-prompt that a
prompt-generator LLM expands per segment.

It lives at `surrogate_rollout/caption_prompt_opt/` and **only adds** files —
no existing `surrogate_rollout` file is modified, and it imports that package
read-only. So it travels with a normal CAPO `git clone`, and it can run
**concurrently** with a live meta-prompt optimization as long as its artifact
directories and cache identities are disjoint. (Paths in the manual command
examples below are relative to this package dir; the go script uses absolute
paths and needs none of them.)

## What is trained, and the initial value

Training optimizes **one** caption prompt applied identically to every segment.
Only the **task body (top part)** is trainable; the fixed output-format block
(the `Transcript of current clip:` section and the strict JSON template with
`clip_start_time / clip_end_time / subject_registry / clip_description`) is
**never trained** — `compose_caption_prompt` appends it verbatim. This mirrors
the meta-prompt split and keeps the output format stable across iterations.

The initial value equals DVD's caption prompt exactly: the seed
`prompts/init_caption_prompt.json` holds only DVD's top sentence
(`There are consecutive frames from a video. Please understand the video clip
with the given transcript then output JSON in the template below.`), and at
iteration 0 `compose_caption_prompt(dvd.caption_prompt, seed)` reproduces
`dvd.caption_prompt` **byte-for-byte** (verified). After training, the promoted
artifact is that one trained top body; the effective caption prompt is
`trained_top + fixed_DVD_format`.

## What is reused vs. new

| Stage | Source |
|-------|--------|
| Feedback generator | reused verbatim (`build_checkpoint_g_components`) |
| Proposer | reused verbatim (unchanged) |
| Updater **code** (`LLMMetaPromptUpdater`, request builder, response schema) | reused |
| Updater **system prompt** | new — `prompts/caption_prompt_updater_system_v1.txt` |
| Per-segment generator | new — `StaticInstructionGenerator` (emits the caption prompt verbatim; **no model call**) |
| Confirmation evaluator | new subclass — `StaticGeneratorConfirmationEvaluator` (overrides only `_generator`) |
| Rollout / evidence generator | new — `run_static_evidence.py` rebinds `build_free_form_generator` to the static generator (no model call) |
| Iteration entry | reused — `surrogate_rollout/scripts/run_prompt_delta_iteration.py` |
| Evidence entry | reused — `surrogate_rollout/scripts/run_fresh_prompt_delta_evidence.py`, wrapped by `run_static_evidence.py` |
| Sample config | `config/component_config.sample.json` (fully isolated cache/identities) |

The updater's baked meta-prompt is swapped at the **backend boundary**
(`SystemPromptOverrideUpdaterBackend`): `LLMMetaPromptUpdater.update` calls
`self.backend(system_instruction, ...)`, so the wrapper replaces the system
prompt with the caption-prompt updater prompt without editing any file under
`surrogate_rollout`. The JSON response contract is unchanged — the parser still
reads `candidate_meta_prompt`, whose value is now the rewritten caption prompt.

## Run

The evidence and confirmation stages **must use the same (static) generator**,
otherwise the evidence policy and the confirmation policy differ. Step 1 below
produces the rollout episodes with the static generator; step 2 consumes them.

### Step 1 — rollout / evidence (static generator, no prompt-generator call)

```bash
cd /home/intern/youngseo
python caption_prompt_opt/run_static_evidence.py \
  --prepared-inputs <dir with manifest.json + component_config.json> \
  --parent-meta-prompt caption_prompt_opt/prompts/init_caption_prompt.json \
  --split-manifest <split.json> \
  --output-dir runs/caption_prompt/evidence \
  --source-revision <rev> \
  --worker-result-timeout-seconds 2400
```

`run_static_evidence.py` wraps the incumbent evidence entry and rebinds
`build_free_form_generator` to `StaticInstructionGenerator` **in this process
only**, so every segment's instruction is the caption prompt verbatim and no
prompt-generator model is called. Use `config/component_config.sample.json` as
the `component_config.json` in the prepared-inputs directory.

### Step 2 — iteration (feedback → update → confirm)

```bash
cd /home/intern/youngseo
python surrogate_rollout/scripts/run_prompt_delta_iteration.py \
  --iteration-id caption_iter_0001 \
  --parent-meta-prompt caption_prompt_opt/prompts/init_caption_prompt.json \
  --component-factory caption_prompt_opt.factory:build_caption_prompt_components \
  --component-config caption_prompt_opt/config/component_config.sample.json \
  --update-episode runs/caption_prompt/evidence/<episode>.json [--update-episode ...] \
  --confirmation-cases <cases.json> \
  --output-dir  runs/caption_prompt/output \
  --state-dir   runs/caption_prompt/state \
  --feedback-memory-bank-dir runs/caption_prompt/memory_bank \
  --measurement-queue-dir    runs/caption_prompt/queue \
  --candidate-created-at 2026-07-24T00:00:00Z \
  --model-identity "captioner=Qwen/Qwen2.5-VL-7B-Instruct;prompt_generator=static;dvd_tool=gpt-5-mini;dvd_fallback=gpt-5.5" \
  --decoding-settings <paired_decoding.json> \
  --cache-reset-identity caption_prompt_static_clean_SAMPLE \
  --evaluation-pipeline-identity dvd_history_aware_static_caption_prompt_paired_v1 \
  --minimum-confirmation-samples ... --minimum-accuracy-delta ... \
  --maximum-correct-to-wrong ... --require-no-execution-failures true \
  --promotion-policy <policy>
```

The `--output-dir / --state-dir / --feedback-memory-bank-dir /
--measurement-queue-dir` above all live under a dedicated
`runs/caption_prompt/` root so they never collide with an incumbent run.

## Concurrency isolation (required)

To run alongside a live meta-prompt optimization **without corrupting its
artifacts or caches**, every path and identity below must differ from the
meta-prompt run:

- entry flags: `--output-dir`, `--state-dir`, `--feedback-memory-bank-dir`,
  `--measurement-queue-dir`;
- in `--component-config` `runtime`: `cache_root`, `cache_manifest_path`,
  `sample_source_identity`, `cache_reset_identity`,
  `evaluation_pipeline_identity`, `paired_model_identity`.

The caption-prompt caches are already namespaced away from any OpenAI-generator
run at the router-version level (reserved `router_v8888_<static-identity>`
carrying `provider=static`, `real_model=false`), but the operator config still
must not point the two runs at the same `cache_root`/manifest — per the repo
rule *never overwrite incumbent or baseline caption caches*.

## Known caveat

The persisted `MetaPromptUpdateResult.request.system_instruction` still records
the baked meta-prompt text (the request is built inside `update()` before the
backend swap). The **actual** provider call uses the caption-prompt updater
prompt, and the produced candidate is genuinely a caption prompt. This is a
cosmetic provenance mismatch only; fixing it fully would require reimplementing
`LLMMetaPromptUpdater.update`, which this path deliberately avoids.
