#!/usr/bin/env bash
# Relaunch LVBench free-form FULL eval (free_form_generator, 103 videos / 1549 QAs).
# Resumable: reuses on-disk caption + per-QA artifacts, so it continues from the
# last finished video (65/103 as of the Jul 21 restart) with no repeated work.
#
# Robustness vs. the run that died Jul 21 ~01:48:
#   * NO `conda run` wrapper  -> stdout/stderr are not buffered/swallowed, so a
#     crash traceback actually lands in the log (the previous run left a 0-byte log).
#   * `python -u`             -> unbuffered, progress visible live.
#   * launched via `setsid`   -> new session, survives terminal / SSH disconnect.
#
# Override GPUs with:  GPUS=0,1,2,3 bash scripts/run_freeform_lvbench_full.sh
set -u
PROJECT_ROOT="/home/intern/youngseo/surrogate_rollout"
cd "$PROJECT_ROOT"
[ -f .env ] && { set -a; source .env; set +a; }
GPUS="${GPUS:-0,1,2,3}"
exec env SR_BENCHMARK=lvbench SR_BENCHMARK_SPLIT=test \
  /home/intern/.conda/envs/local_llm_vllm/bin/python -u scripts/run_lvbench_eval.py \
    --output-dir runs/freeform_lvbench_full_output \
    --cache-dir  runs/freeform_lvbench_pilot_cache \
    --gpus "$GPUS" \
    --routing-mode free_form_generator
