# Free-form caption-instruction baseline (experimental, ablation only)

Training-free comparison baseline for the prompt-bank + router system.
It is NOT the main deployment path; the maintained method remains
`property_bank` routing. Main experiments must be launched manually.

## What differs

```
Ours (default, unchanged):
  frames + frozen history -> PromptRouter (property IDs)
    -> prompt-bank instructions -> fixed scaffold -> captioner [+ transcript]

Baseline (opt-in):
  frames + frozen history -> VLMFreeFormInstructionGenerator
    -> one generated instruction -> SAME fixed scaffold -> SAME captioner [+ transcript]
```

Only the adaptive-instruction source changes. The generator receives the
EXACT same context the router does — sampled frames + frozen caption history,
and nothing else. Like the router, it never sees the segment transcript; the
transcript still reaches the captioner downstream in both modes, so caption
behavior is unchanged. Shared and unchanged: segment boundaries, frame
sampling, history snapshot construction, transcript policy
(`config.USE_TRANSCRIPT`), scaffold contract (`CAPTION_OUTPUT_CONTRACT`),
captioner model/decoding, history-aware caption cache, routed-caption
assembly, downstream DVD inference and QA, cost/latency logging.

QA-leakage guard: the generator request serializes only video/segment IDs,
timestamps, frame references, and the frozen caption history. It never
receives the segment transcript, questions, answer options, reference answers,
correctness labels, feedback, intervention results, or any QA-conditioned
retrieval. See
`prompt_routing/free_form_instruction_generator.py` and the leak test in
`tests/test_free_form_instruction.py`.

## Enabling

Builder level (the integration hook):

```python
builder = HistoryAwareBaselineCaptionViewBuilder.from_local_qwen(
    parallel_gpus=("4", "5"),
    routing_mode="free_form_generator",   # default: "property_bank"
)
```

CLI (production memory-iteration entry point):

```bash
/home/intern/.conda/envs/local_llm_vllm/bin/python -m \
  surrogate_rollout.scripts.run_phase4_memory_iteration \
  --iteration-id freeform-ablation-001 \
  --output-dir  surrogate_rollout/runs/freeform_ablation_001_output \
  --state-dir   surrogate_rollout/runs/freeform_ablation_001_state \
  --cache-dir   surrogate_rollout/runs/freeform_ablation_001_cache \
  --num-videos 4 --gpus 4,5,6,7 \
  --routing-mode free_form_generator
```

The matching property-bank comparison is the same command with
`--routing-mode property_bank` (or the flag omitted) and its own
`--output-dir/--state-dir/--cache-dir`. NOTE: this entry point also executes
the optimizer stages (proposal/feedback/updaters, paid gpt-4o calls); for the
captioning-vs-QA ablation, compare the baseline-phase artifacts
(`baseline_videos/<video>/baseline/<video>/baseline_qas.jsonl`) between the
two runs and ignore the update stages of the free-form run.

Matched comparison guarantees: keep identical `--num-videos`,
`--selection-seed`, `--gpus`, split manifest, components file, and the same
global transcript setting (`config.USE_TRANSCRIPT`) across both runs. Frames,
segment boundaries, history construction, captioner, and the DVD evaluator
are code-identical in both modes.

## Generator configuration

- Backend: the same process-shared local Qwen2.5-VL backend as router and
  captioner (`config.CAPTION_MODEL_ID`); no separate inference stack.
- Template: `GENERATOR_TEMPLATES["v2_plain_text"]` in
  `prompt_routing/free_form_instruction_generator.py`
  (`template_version="v2_plain_text"`, `max_tokens=192` by default; constructor
  parameters of `VLMFreeFormInstructionGenerator`).
- Output parsing: strict plain text in
  `prompt_routing/free_form_instruction_parser.py`. Surrounding whitespace is
  removed and the complete non-empty response is retained literally. JSON-like
  text and Markdown fences are not interpreted or repaired. Empty output raises
  `FreeFormGenerationError`. The path is logged as `parser_path=plain_text`.

## Transcript behavior

The generator never receives the segment transcript, matching the router's
selection call exactly. This is independent of `config.USE_TRANSCRIPT`: the
flag governs only the captioner's own prompt (identical in both modes), not
the instruction producer. Keeping transcript out of the generator makes the
routing-vs-generation comparison apples-to-apples.

## Artifacts, logs, auditability

Per work root (same layout as property-bank runs):

- `routing_decisions.jsonl` — free-form decisions carry the audit trail in
  `decision_payload`: `mode`, `generator_model`, `template_version`,
  `parser_path`, `request_hash`, `raw_generator_response`,
  `generated_instruction`, `generated_instruction_hash`,
  `generation_seconds`. Runs are auditable without regenerating instructions.
- `composed_prompts.jsonl` — final composed prompts (instruction included;
  `prompt_hash` = final composed prompt hash).
- `segment_state/<segment>.json` — resume sidecar, including the generator
  exchange (request + hashes + raw output).
- `caption_cache_keys.jsonl`, `frozen_histories.jsonl`, `frames.json`,
  `routing_manifest.json` — unchanged conventions.

## Cache and resume behavior

- Reuses `build_history_aware_cache_key` / `new_history_aware_cache_dir`
  unchanged. Free-form caption caches carry the reserved router version
  `router_v8888_<generator-identity-hash8>` (identity covers mode, model,
  backend, template version+hash, max_tokens, parser version), and the
  generated instruction is part of the composed prompt hash — free-form keys
  can never resolve against property-router caches, and changing the
  instruction, frames-source, history, template version, scaffold/contract
  version, captioner model, or decoding changes the identity.
- Resume: the per-segment `state_identity` additionally includes
  `free_form_generator_identity` (ONLY in free-form mode, so property-bank
  state identities are byte-stable). A rerun with unchanged identity resumes
  from sidecars with zero generator/captioner calls.

## Validation status

Mock-only validation: unit + mocked integration tests
(`tests/test_free_form_instruction.py`, no real model inference). Real
Qwen2.5-VL compatibility of the generator call path has NOT been executed;
the call shape matches the router's existing `vlm.caption(frames, prompt,
max_tokens=...)` usage.
