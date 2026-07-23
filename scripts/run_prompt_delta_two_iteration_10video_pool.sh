#!/usr/bin/env bash
# Operator-run only: sequential configurable-K prompt-delta iterations over
# explicit frozen evidence and confirmation cohorts.

set -euo pipefail

PROJECT_ROOT="${SR_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/run_prompt_delta_two_iteration_10video_pool.sh

Environment:
  PROMPT_DELTA_ITERATION_TIMESTAMP      UTC YYYYMMDD_HHMMSS; reuse to resume.
  PROMPT_DELTA_WORKER_GPUS              Unique GPU IDs, one or more (default 0,1,2,3,4).
  PROMPT_DELTA_REQUIRED_GPU_COUNT       Optional exact GPU count to enforce.
  PROMPT_DELTA_ITERATION_COUNT          Positive iteration count (default 2).
  PROMPT_DELTA_VIDEOS_PER_ITERATION     Evidence videos per iteration (default 5).
  PROMPT_DELTA_EVIDENCE_COHORT_FILE     Ordered video-ID file (default train_set/10samples.txt).
  PROMPT_DELTA_CONFIRMATION_COHORT_FILE Disjoint held-out IDs (default confirmation_5samples.txt).
  PROMPT_DELTA_EXPERIMENT_LABEL         Output-name label.

Runs sequential prompt-delta iterations from explicit frozen cohorts.
This command performs local GPU work and paid provider calls.
EOF
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "unexpected arguments; use --help" >&2
  exit 2
fi
cd "$PROJECT_ROOT"

GPUS="${PROMPT_DELTA_WORKER_GPUS:-${PROMPT_DELTA_TWO_ITERATION_GPUS:-0,1,2,3,4}}"
ITERATION_COUNT="${PROMPT_DELTA_ITERATION_COUNT:-2}"
VIDEOS_PER_ITERATION="${PROMPT_DELTA_VIDEOS_PER_ITERATION:-5}"
# The worker count is a capacity choice, not a correctness constraint: the
# evidence runner only requires a non-empty unique GPU list. Pin it with
# PROMPT_DELTA_REQUIRED_GPU_COUNT when a run must reserve an exact number.
REQUIRED_GPU_COUNT="${PROMPT_DELTA_REQUIRED_GPU_COUNT:-}"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "PROMPT_DELTA_WORKER_GPUS must contain at least one GPU ID" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${GPU_IDS[@]}" | sort -u | wc -l)" -ne "${#GPU_IDS[@]}" ]]; then
  echo "PROMPT_DELTA_WORKER_GPUS must be unique" >&2
  exit 2
fi
if [[ -n "$REQUIRED_GPU_COUNT" ]]; then
  if [[ ! "$REQUIRED_GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "PROMPT_DELTA_REQUIRED_GPU_COUNT must be a positive integer" >&2
    exit 2
  fi
  if [[ "${#GPU_IDS[@]}" -ne "$REQUIRED_GPU_COUNT" ]]; then
    echo "PROMPT_DELTA_WORKER_GPUS must contain exactly $REQUIRED_GPU_COUNT GPU IDs" >&2
    exit 2
  fi
fi
if [[ ! "$VIDEOS_PER_ITERATION" =~ ^[1-9][0-9]*$ ]]; then
  echo "PROMPT_DELTA_VIDEOS_PER_ITERATION must be a positive integer" >&2
  exit 2
fi
if [[ ! "$ITERATION_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "PROMPT_DELTA_ITERATION_COUNT must be a positive integer" >&2
  exit 2
fi

EVIDENCE_COHORT="${PROMPT_DELTA_EVIDENCE_COHORT_FILE:-$PROJECT_ROOT/train_set/10samples.txt}"
CONFIRMATION_COHORT="${PROMPT_DELTA_CONFIRMATION_COHORT_FILE:-$PROJECT_ROOT/train_set/confirmation_5samples.txt}"
test -f "$EVIDENCE_COHORT"
test -f "$CONFIRMATION_COHORT"
mapfile -t EVIDENCE_IDS < "$EVIDENCE_COHORT"
mapfile -t CONFIRMATION_IDS < "$CONFIRMATION_COHORT"
if [[ "$((VIDEOS_PER_ITERATION * ITERATION_COUNT))" -gt "${#EVIDENCE_IDS[@]}" ]]; then
  echo "configured iterations exceed the frozen evidence cohort" >&2
  exit 2
fi
if printf '%s\n' "${EVIDENCE_IDS[@]}" "${CONFIRMATION_IDS[@]}" |
     sort | uniq -d | grep -q .; then
  echo "evidence and confirmation cohorts overlap or contain duplicates" >&2
  exit 2
fi

EXPERIMENT_TIMESTAMP="${PROMPT_DELTA_ITERATION_TIMESTAMP:-${PROMPT_DELTA_TWO_ITERATION_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}}"
if [[ ! "$EXPERIMENT_TIMESTAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "PROMPT_DELTA_TWO_ITERATION_TIMESTAMP must use YYYYMMDD_HHMMSS" >&2
  exit 2
fi
EXPERIMENT_LABEL="${PROMPT_DELTA_EXPERIMENT_LABEL:-${#EVIDENCE_IDS[@]}video_${ITERATION_COUNT}iteration}"
if [[ ! "$EXPERIMENT_LABEL" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "PROMPT_DELTA_EXPERIMENT_LABEL contains unsupported characters" >&2
  exit 2
fi
EXPERIMENT_ROOT="$PROJECT_ROOT/runs/prompt_delta_${EXPERIMENT_LABEL}_${EXPERIMENT_TIMESTAMP}"
STATE_ROOT="${EXPERIMENT_ROOT}_state"
CACHE_ROOT="${EXPERIMENT_ROOT}_cache"
MEMORY_ROOT="${EXPERIMENT_ROOT}_feedback_memory"
CURRENT_PARENT="$PROJECT_ROOT/optimization/prompts/init_meta_prompt.json"
COMPLETED_ITERATIONS=()
BASE_EPOCH="$(date -u -d "${EXPERIMENT_TIMESTAMP:0:4}-${EXPERIMENT_TIMESTAMP:4:2}-${EXPERIMENT_TIMESTAMP:6:2}T${EXPERIMENT_TIMESTAMP:9:2}:${EXPERIMENT_TIMESTAMP:11:2}:${EXPERIMENT_TIMESTAMP:13:2}Z" +%s)"

for ordinal in $(seq 1 "$ITERATION_COUNT"); do
  offset=$(( (ordinal - 1) * VIDEOS_PER_ITERATION ))
  selected=("${EVIDENCE_IDS[@]:offset:VIDEOS_PER_ITERATION}")
  previous=("${EVIDENCE_IDS[@]:0:offset}")
  run_timestamp="$(date -u -d "@$((BASE_EPOCH + ordinal - 1))" +%Y%m%d_%H%M%S)"
  selected_csv="$(IFS=,; echo "${selected[*]}")"
  previous_csv=""
  if [[ "${#previous[@]}" -gt 0 ]]; then
    previous_csv="$(IFS=,; echo "${previous[*]}")"
  fi

  FRESH_PROMPT_DELTA_MEASURE_PARENT="$([[ "$ordinal" -eq 1 ]] && echo true || echo false)" \
  FRESH_PROMPT_DELTA_TIMESTAMP="$run_timestamp" \
  FRESH_PROMPT_DELTA_WORKER_GPUS="$GPUS" \
  FRESH_PROMPT_DELTA_SELECTED_VIDEO_IDS="$selected_csv" \
  FRESH_PROMPT_DELTA_PREVIOUS_UPDATE_VIDEO_IDS="$previous_csv" \
  FRESH_PROMPT_DELTA_EVIDENCE_COHORT_FILE="$EVIDENCE_COHORT" \
  FRESH_PROMPT_DELTA_CONFIRMATION_COHORT_FILE="$CONFIRMATION_COHORT" \
  FRESH_PROMPT_DELTA_PARENT_META_PROMPT="$CURRENT_PARENT" \
  FRESH_PROMPT_DELTA_STATE_ROOT="$STATE_ROOT" \
  FRESH_PROMPT_DELTA_CACHE_ROOT="$CACHE_ROOT" \
  FRESH_PROMPT_DELTA_MEMORY_BANK_ROOT="$MEMORY_ROOT" \
    bash "$PROJECT_ROOT/scripts/run_fresh_prompt_delta_iteration.sh"

  evidence_manifest="$PROJECT_ROOT/runs/fresh_prompt_delta_iteration_${run_timestamp}_evidence/fresh_evidence_manifest.json"
  status="$(jq -r '.status' "$evidence_manifest")"
  if [[ "$status" != "no_eligible_proposal_evidence" ]]; then
    result="$PROJECT_ROOT/runs/fresh_prompt_delta_iteration_${run_timestamp}_output/iteration_result.json"
    jq -e '.status == "no_update" or .status == "promoted" or .status == "rolled_back"' "$result" >/dev/null
    # Not the live pointer: an iteration moves it at promotion time, before
    # its own measurement finishes. Replaying completed iterations after a
    # crash would then read the pointer this iteration itself advanced and
    # hand the next iteration a parent its prepared inputs never agreed to.
    # What carries forward is what THIS iteration ended with.
    active_id="$(jq -r '.active_meta_prompt_id' "$result")"
    if [[ -z "$active_id" || "$active_id" == "null" ]]; then
      echo "iteration result has no active meta-prompt id: $result" >&2
      exit 1
    fi
    CURRENT_PARENT="$STATE_ROOT/versions/${active_id}.json"
    test -f "$CURRENT_PARENT"
  fi
  COMPLETED_ITERATIONS+=("$run_timestamp")
done

mkdir -p "$EXPERIMENT_ROOT"
manifest_tmp="$EXPERIMENT_ROOT/experiment_manifest.json.tmp.$$"
jq -n \
  --arg timestamp "$EXPERIMENT_TIMESTAMP" \
  --arg gpus "$GPUS" \
  --arg evidence "$EVIDENCE_COHORT" \
  --arg confirmation "$CONFIRMATION_COHORT" \
  --arg state "$STATE_ROOT" \
  --arg cache "$CACHE_ROOT" \
  --arg memory "$MEMORY_ROOT" \
  --argjson iteration_count "$ITERATION_COUNT" \
  --argjson videos_per_iteration "$VIDEOS_PER_ITERATION" \
  --arg parent "$CURRENT_PARENT" \
  --argjson iterations "$(printf '%s\n' "${COMPLETED_ITERATIONS[@]}" | jq -R . | jq -s .)" \
  '{schema_version:"prompt_delta_multi_iteration_experiment_v1",status:"completed",experiment_timestamp:$timestamp,worker_gpus:($gpus|split(",")),iteration_count:$iteration_count,videos_per_iteration:$videos_per_iteration,evidence_cohort_path:$evidence,confirmation_cohort_path:$confirmation,iterations:$iterations,state_root:$state,cache_root:$cache,feedback_memory_bank_root:$memory,final_active_meta_prompt_artifact:$parent}' \
  > "$manifest_tmp"
if [[ -f "$EXPERIMENT_ROOT/experiment_manifest.json" ]]; then
  cmp -s "$manifest_tmp" "$EXPERIMENT_ROOT/experiment_manifest.json" || {
    echo "completed experiment manifest conflicts with resumed inputs" >&2
    exit 1
  }
  rm "$manifest_tmp"
else
  mv "$manifest_tmp" "$EXPERIMENT_ROOT/experiment_manifest.json"
fi
accuracy_tmp="$EXPERIMENT_ROOT/heldout_accuracy.json.tmp.$$"
{
# iteration 0: the starting meta-prompt, measured before any update
first_parent_measurement="$PROJECT_ROOT/runs/fresh_prompt_delta_iteration_${COMPLETED_ITERATIONS[0]}_output/parent_measurement/measurement_summary.json"
if [[ -f "$first_parent_measurement" ]]; then
  jq '{
    iteration_id:"iteration_0_initial_meta_prompt",
    status:"initial",
    heldout_evaluation:"completed_active_measurement",
    evaluated_qa_count:.evaluated_qa_count,
    case_count:.case_count,
    parent_accuracy:null,
    candidate_accuracy:null,
    active_accuracy:.accuracy,
    measurement_manifest_path:.measurement_manifest_path
  }' "$first_parent_measurement"
fi
printf '%s\n' "${COMPLETED_ITERATIONS[@]}" | while read -r run_timestamp; do
  output="$PROJECT_ROOT/runs/fresh_prompt_delta_iteration_${run_timestamp}_output"
  result="$output/iteration_result.json"
  if [[ ! -f "$result" ]]; then
    jq -n --arg iteration_id "fresh_prompt_delta_${run_timestamp}" \
      '{iteration_id:$iteration_id,status:"no_eligible_proposal_evidence",heldout_evaluation:"not_run",parent_accuracy:null,candidate_accuracy:null,active_accuracy:null}'
    continue
  fi
  status="$(jq -r '.status' "$result")"
  confirmation="$output/confirmation/dvd_confirmation_manifest.json"
  measurement="$output/measurement/dvd_measurement_manifest.json"
  if [[ "$status" == "no_update" ]]; then
    jq -n --arg iteration_id "fresh_prompt_delta_${run_timestamp}" \
      '{iteration_id:$iteration_id,status:"no_update",heldout_evaluation:"not_run",parent_accuracy:null,candidate_accuracy:null,active_accuracy:null}'
  elif [[ -f "$measurement" ]]; then
    # always_promote_measured_v1: the held-out set reports, it does not decide.
    jq --arg status "$status" --arg path "$measurement" \
      --arg iteration_id "$(jq -r '.iteration_id' "$result")" '{
      iteration_id:$iteration_id,
      status:$status,
      heldout_evaluation:"completed_active_measurement",
      evaluated_qa_count:.aggregate.evaluated_qa_count,
      case_count:.aggregate.case_count,
      parent_accuracy:null,
      candidate_accuracy:null,
      active_accuracy:.aggregate.accuracy,
      measurement_manifest_path:$path
    }' "$measurement"
  else
    test -f "$confirmation"
    jq --arg status "$status" --arg path "$confirmation" \
      --arg iteration_id "$(jq -r '.iteration_id' "$result")" '{
      iteration_id:$iteration_id,
      status:$status,
      heldout_evaluation:"completed_paired_confirmation",
      evaluated_qa_count:.aggregate.evaluated_qa_count,
      parent_accuracy:.aggregate.parent_accuracy,
      candidate_accuracy:.aggregate.candidate_accuracy,
      accuracy_delta:.aggregate.accuracy_delta,
      active_accuracy:(if $status == "promoted" then .aggregate.candidate_accuracy else .aggregate.parent_accuracy end),
      confirmation_manifest_path:$path
    }' "$confirmation"
  fi
done
} | jq -s '{schema_version:"prompt_delta_iteration_heldout_accuracy_v2",iterations:.}' > "$accuracy_tmp"
if [[ -f "$EXPERIMENT_ROOT/heldout_accuracy.json" ]]; then
  cmp -s "$accuracy_tmp" "$EXPERIMENT_ROOT/heldout_accuracy.json" || {
    echo "held-out accuracy report conflicts with resumed artifacts" >&2
    exit 1
  }
  rm "$accuracy_tmp"
else
  mv "$accuracy_tmp" "$EXPERIMENT_ROOT/heldout_accuracy.json"
fi
printf 'EXPERIMENT_ROOT=%s\nFINAL_PARENT=%s\n' "$EXPERIMENT_ROOT" "$CURRENT_PARENT"
