# Phase 4 Runbook

The active run path is fresh prompt-delta-only. The user runs main experiments
manually.

## Environment

Required environment:

```bash
cd <path to>/surrogate_rollout
export PYTHONPATH="$(dirname "$PWD"):$PWD"
export OPENAI_API_KEY=...
```

Use the existing conda environment:

```bash
conda run -n local_llm_vllm python -m pytest -q
```

## Prepare Inputs

Use the existing prepared-input launcher from the shell wrapper:

```bash
FRESH_PROMPT_DELTA_TIMESTAMP=<timestamp> \
bash scripts/run_fresh_prompt_delta_iteration.sh
```

The wrapper prepares immutable inputs under:

```text
runs/fresh_prompt_delta_iteration_<timestamp>_inputs/
```

and writes outputs under:

```text
runs/fresh_prompt_delta_iteration_<timestamp>_output/
```

## Resume

Resume by rerunning the same timestamp:

```bash
FRESH_PROMPT_DELTA_TIMESTAMP=<timestamp> \
bash scripts/run_fresh_prompt_delta_iteration.sh
```

Completed immutable artifacts are reused only when identity and hashes match.
If regenerating feedback only, move aside the output `feedback/` directory and
the matching `input_identity.json`; do not delete evidence episodes.

## Verification

Focused model-free verification:

```bash
cd <path to>/surrogate_rollout
PYTHONPATH="$(dirname "$PWD"):$PWD" \
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_static_meta_replace_body.py \
  tests/test_fresh_prompt_delta_evidence.py \
  tests/test_llm_episode_feedback.py \
  tests/test_episode_feedback_provider.py \
  tests/test_meta_prompt_updater.py \
  tests/test_meta_prompt_update_execution.py \
  tests/test_prompt_delta_iteration.py
```

Full model-free verification:

```bash
cd <path to>/surrogate_rollout
PYTHONPATH="$(dirname "$PWD"):$PWD" \
conda run -n local_llm_vllm python -m pytest -q
```

Static checks:

```bash
python -m compileall -q captioning optimization prompt_routing retrieval selection evaluation scripts tests
bash -n scripts/run_fresh_prompt_delta_iteration.sh
git diff --check
```

## Success Checks

- No import of removed codebook/router/property/GEPA modules remains in active
  code.
- Fresh proposer requests are per-QA isolated.
- Candidate mixed views rerun all sibling QAs.
- Feedback request artifacts are immutable and resume-safe.
- Meta-prompt confirmation uses `replace_body` free-form captioning and does
  not construct compatibility bank/router objects.
