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

# Held-out measurement is deferred to a queue for this preset. It is decision-free
# either way -- always_promote_measured_v1 promotes regardless of the numbers --
# but running it inline requires the confirmation cohort's media on the captioning
# host, which the evidence-only hosts do not carry. Deferring keeps the loop
# running on the evidence cohort alone; a measurement worker drains the queue
# wherever the held-out media lives.
#
# It lives here, in the preset both the meta run and its full-recaption variant
# exec, so the two share one policy -- a scope comparison is only readable when
# promotion is identical on both sides. The queue is derived from the experiment
# label so each variant keeps its own requests. Both are `:-` defaults, so an
# inline run stays available.
export FRESH_PROMPT_DELTA_PROMOTION_POLICY="${FRESH_PROMPT_DELTA_PROMOTION_POLICY:-promote_and_enqueue_measurement_v1}"
export FRESH_PROMPT_DELTA_MEASUREMENT_QUEUE_DIR="${FRESH_PROMPT_DELTA_MEASUREMENT_QUEUE_DIR:-$PROJECT_ROOT/runs/${PROMPT_DELTA_EXPERIMENT_LABEL}_measurement_queue}"

exec bash "$PROJECT_ROOT/scripts/run_prompt_delta_two_iteration_10video_pool.sh" "$@"
