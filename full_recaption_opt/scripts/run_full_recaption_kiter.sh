#!/usr/bin/env bash
# Full-recaption baseline = the META prompt-delta pipeline, unchanged, with the
# evidence intervention runner swapped to apply the delta to EVERY baseline
# segment (full recaption) instead of the source-QA-localized subset.
#
# It reuses the meta K-iteration driver (scripts/run_prompt_delta_*_pool.sh ->
# run_fresh_prompt_delta_iteration.sh) verbatim, so every setting is IDENTICAL to
# a meta run -- LLM free-form generator, v5 updater, captioner, DVD tool,
# optimizer models/tokens, cohorts, iterations. The ONLY difference is the
# evidence entry (FRESH_PROMPT_DELTA_EVIDENCE_ENTRY), which patches the runner.
#
# Pass the same env as a meta run (PROMPT_DELTA_ITERATION_COUNT,
# PROMPT_DELTA_VIDEOS_PER_ITERATION, cohorts, GPUs, SR_*_MODEL_ID, generator
# tokens, promotion policy, ...). This wrapper only sets the evidence entry and a
# distinct experiment label so its runs/state/memory never collide with a meta
# run.

set -euo pipefail

if [[ -n "${SR_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$SR_PROJECT_ROOT"
else
  _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # full_recaption_opt/scripts
  _fro="$(dirname "$_here")"                                     # full_recaption_opt
  if [[ -f "$(dirname "$_fro")/scripts/run_prompt_delta_four_iteration_20video_pool.sh" ]]; then
    PROJECT_ROOT="$(dirname "$_fro")"                            # nested in repo
  elif [[ -f "$(dirname "$_fro")/surrogate_rollout/scripts/run_prompt_delta_four_iteration_20video_pool.sh" ]]; then
    PROJECT_ROOT="$(dirname "$_fro")/surrogate_rollout"          # sibling
  else
    echo "cannot locate surrogate_rollout checkout; set SR_PROJECT_ROOT" >&2; exit 2
  fi
fi

# The one change vs a meta run: the evidence step applies the delta to all
# segments. Path is relative to PROJECT_ROOT (the iteration shell prepends it).
export FRESH_PROMPT_DELTA_EVIDENCE_ENTRY="full_recaption_opt/run_full_recaption_evidence.py"
# Distinct label -> isolated experiment/state/cache/memory dirs from any meta run.
export PROMPT_DELTA_EXPERIMENT_LABEL="${PROMPT_DELTA_EXPERIMENT_LABEL:-full_recaption_20video_4iteration}"

exec bash "$PROJECT_ROOT/scripts/run_prompt_delta_four_iteration_20video_pool.sh" "$@"
