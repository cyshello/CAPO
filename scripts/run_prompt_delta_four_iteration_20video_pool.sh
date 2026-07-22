#!/usr/bin/env bash
# Operator-run compatibility preset: four iterations x five evidence videos,
# with ten disjoint held-out videos. The generic launcher remains configurable.

set -euo pipefail

PROJECT_ROOT="${SR_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROMPT_DELTA_ITERATION_COUNT="${PROMPT_DELTA_ITERATION_COUNT:-4}"
export PROMPT_DELTA_VIDEOS_PER_ITERATION="${PROMPT_DELTA_VIDEOS_PER_ITERATION:-5}"
export PROMPT_DELTA_EVIDENCE_COHORT_FILE="${PROMPT_DELTA_EVIDENCE_COHORT_FILE:-$PROJECT_ROOT/train_set/20samples.txt}"
export PROMPT_DELTA_CONFIRMATION_COHORT_FILE="${PROMPT_DELTA_CONFIRMATION_COHORT_FILE:-$PROJECT_ROOT/train_set/confirmation_10samples.txt}"
export PROMPT_DELTA_EXPERIMENT_LABEL="${PROMPT_DELTA_EXPERIMENT_LABEL:-20video_4iteration}"

exec bash "$PROJECT_ROOT/scripts/run_prompt_delta_two_iteration_10video_pool.sh" "$@"
