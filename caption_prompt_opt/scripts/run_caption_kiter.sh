#!/usr/bin/env bash
# Operator-run only: sequential configurable-K caption-PROMPT iterations over
# explicit frozen evidence and confirmation cohorts. Faithful copy of
# scripts/run_prompt_delta_two_iteration_10video_pool.sh with the caption swaps:
#   * per-iteration driver -> caption_prompt_opt/scripts/run_caption_iteration.sh
#   * parent chain starts at the caption prompt, advances by promoted id
#   * run/experiment dirs prefixed caption_prompt_
#   * promotion policy PINNED to always_promote_measured_v1 (held-out reports,
#     never gates) -- required for this experiment.

set -euo pipefail

CPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${SR_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$SR_PROJECT_ROOT"
elif [[ -f "$(dirname "$CPO_ROOT")/scripts/run_prompt_delta_iteration.py" ]]; then
  PROJECT_ROOT="$(dirname "$CPO_ROOT")"
elif [[ -f "$(dirname "$CPO_ROOT")/surrogate_rollout/scripts/run_prompt_delta_iteration.py" ]]; then
  PROJECT_ROOT="$(dirname "$CPO_ROOT")/surrogate_rollout"
else
  echo "cannot locate surrogate_rollout checkout; set SR_PROJECT_ROOT" >&2; exit 2
fi

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash caption_prompt_opt/scripts/run_caption_kiter.sh

Environment:
  PROMPT_DELTA_ITERATION_TIMESTAMP      UTC YYYYMMDD_HHMMSS; reuse to resume.
  PROMPT_DELTA_WORKER_GPUS              Unique GPU IDs (default 0,1,2,3).
  PROMPT_DELTA_REQUIRED_GPU_COUNT       Optional exact GPU count to enforce.
  PROMPT_DELTA_ITERATION_COUNT          Positive iteration count (default 5).
  PROMPT_DELTA_VIDEOS_PER_ITERATION     Evidence videos per iteration (default 4).
  PROMPT_DELTA_EVIDENCE_COHORT_FILE     Ordered video-ID file (default train_set/20samples.txt).
  PROMPT_DELTA_CONFIRMATION_COHORT_FILE Disjoint held-out IDs (default train_set/confirmation.txt).
  PROMPT_DELTA_EXPERIMENT_LABEL         Output-name label.

Sequential caption-prompt iterations from frozen cohorts. Promotion policy is
pinned to always_promote_measured_v1. Local GPU work + paid provider calls.
EOF
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "unexpected arguments; use --help" >&2
  exit 2
fi
cd "$PROJECT_ROOT"

GPUS="${PROMPT_DELTA_WORKER_GPUS:-0,1,2,3}"
ITERATION_COUNT="${PROMPT_DELTA_ITERATION_COUNT:-5}"
VIDEOS_PER_ITERATION="${PROMPT_DELTA_VIDEOS_PER_ITERATION:-4}"
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

EVIDENCE_COHORT="${PROMPT_DELTA_EVIDENCE_COHORT_FILE:-$PROJECT_ROOT/train_set/20samples.txt}"
CONFIRMATION_COHORT="${PROMPT_DELTA_CONFIRMATION_COHORT_FILE:-$PROJECT_ROOT/train_set/confirmation.txt}"
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

EXPERIMENT_TIMESTAMP="${PROMPT_DELTA_ITERATION_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ ! "$EXPERIMENT_TIMESTAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "PROMPT_DELTA_ITERATION_TIMESTAMP must use YYYYMMDD_HHMMSS" >&2
  exit 2
fi
EXPERIMENT_LABEL="${PROMPT_DELTA_EXPERIMENT_LABEL:-${#EVIDENCE_IDS[@]}video_${ITERATION_COUNT}iteration}"
if [[ ! "$EXPERIMENT_LABEL" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "PROMPT_DELTA_EXPERIMENT_LABEL contains unsupported characters" >&2
  exit 2
fi
EXPERIMENT_ROOT="$PROJECT_ROOT/runs/caption_prompt_${EXPERIMENT_LABEL}_${EXPERIMENT_TIMESTAMP}"
STATE_ROOT="${EXPERIMENT_ROOT}_state"
CACHE_ROOT="${EXPERIMENT_ROOT}_cache"
MEMORY_ROOT="${EXPERIMENT_ROOT}_feedback_memory"
CURRENT_PARENT="$CPO_ROOT/prompts/init_caption_prompt.json"
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
  FRESH_PROMPT_DELTA_PROMOTION_POLICY="always_promote_measured_v1" \
    bash "$CPO_ROOT/scripts/run_caption_iteration.sh"

  evidence_manifest="$PROJECT_ROOT/runs/caption_prompt_iteration_${run_timestamp}_evidence/fresh_evidence_manifest.json"
  status="$(jq -r '.status' "$evidence_manifest")"
  if [[ "$status" != "no_eligible_proposal_evidence" ]]; then
    result="$PROJECT_ROOT/runs/caption_prompt_iteration_${run_timestamp}_output/iteration_result.json"
    jq -e '.status == "no_update" or .status == "promoted" or .status == "rolled_back"' "$result" >/dev/null
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
  '{schema_version:"caption_prompt_multi_iteration_experiment_v1",status:"completed",experiment_timestamp:$timestamp,worker_gpus:($gpus|split(",")),iteration_count:$iteration_count,videos_per_iteration:$videos_per_iteration,evidence_cohort_path:$evidence,confirmation_cohort_path:$confirmation,iterations:$iterations,state_root:$state,cache_root:$cache,feedback_memory_bank_root:$memory,final_active_caption_prompt_artifact:$parent}' \
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
printf 'EXPERIMENT_ROOT=%s\nFINAL_PARENT=%s\n' "$EXPERIMENT_ROOT" "$CURRENT_PARENT"
