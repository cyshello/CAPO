#!/usr/bin/env bash
# Operator-run only: one fresh configurable-K prompt-delta iteration from the
# currently active parent.  This script performs paid/model work; Codex does not.

set -euo pipefail

PROJECT_ROOT="${SR_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${SR_CONDA_ENV:=local_llm_vllm}"
cd "$PROJECT_ROOT"
set -a
source "$PROJECT_ROOT/.env"
set +a
export PYTHONPATH="$(dirname "$PROJECT_ROOT"):$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# `conda run` does not put the env's lib dir on the loader path, so torchcodec's
# FFmpeg/TLS chain resolves the system libp11-kit/libffi and dies with
# "undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0". Prepend the env
# lib dir so conda's libffi wins.
_CONDA_ENV_PREFIX="$(conda info --base 2>/dev/null)/envs/$SR_CONDA_ENV"
[ -d "$_CONDA_ENV_PREFIX/lib" ] && export LD_LIBRARY_PATH="$_CONDA_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY is missing from $PROJECT_ROOT/.env}"

RUN_TIMESTAMP="${FRESH_PROMPT_DELTA_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
WORKER_GPUS="${FRESH_PROMPT_DELTA_WORKER_GPUS:-4,5,6,7}"
if [[ ! "$RUN_TIMESTAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "FRESH_PROMPT_DELTA_TIMESTAMP must use YYYYMMDD_HHMMSS" >&2
  exit 2
fi
if [[ ! "$WORKER_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "FRESH_PROMPT_DELTA_WORKER_GPUS must be a comma-separated GPU list" >&2
  exit 2
fi

RUN_ROOT="$PROJECT_ROOT/runs/fresh_prompt_delta_iteration_${RUN_TIMESTAMP}"
INPUT_ROOT="${RUN_ROOT}_inputs"
EVIDENCE_ROOT="${RUN_ROOT}_evidence"
OUTPUT_ROOT="${RUN_ROOT}_output"
STATE_ROOT="${FRESH_PROMPT_DELTA_STATE_ROOT:-${RUN_ROOT}_state}"
CACHE_ROOT="${FRESH_PROMPT_DELTA_CACHE_ROOT:-${RUN_ROOT}_cache}"
FEEDBACK_MEMORY_BANK_ROOT="${FRESH_PROMPT_DELTA_MEMORY_BANK_ROOT:-$PROJECT_ROOT/runs/prompt_delta_feedback_memory_bank}"

# Token accounting (additive only). Every model call in both phases and their
# GPU-worker subprocesses appends usage into this per-iteration ledger dir; the
# EXIT trap sums it and prints a per-source token report when the iteration ends.
# Set SR_TOKEN_LEDGER_DIR yourself to override, or unset all three below to skip.
export SR_TOKEN_LEDGER_DIR="${SR_TOKEN_LEDGER_DIR:-${RUN_ROOT}_tokens}"
mkdir -p "$SR_TOKEN_LEDGER_DIR"
export SR_QWEN_USAGE_LOG="${SR_QWEN_USAGE_LOG:-$SR_TOKEN_LEDGER_DIR/qwen}"
export SR_OPENAI_USAGE_LOG="${SR_OPENAI_USAGE_LOG:-$SR_TOKEN_LEDGER_DIR/dvd_openai}"
_report_token_usage() {
  python "$PROJECT_ROOT/scripts/report_token_usage.py" \
    "$SR_TOKEN_LEDGER_DIR" 2>/dev/null || true
}
trap _report_token_usage EXIT

PARENT="${FRESH_PROMPT_DELTA_PARENT_META_PROMPT:-$PROJECT_ROOT/optimization/prompts/init_meta_prompt.json}"
BASELINE_RESUME_DIR="${FRESH_PROMPT_DELTA_BASELINE_RESUME_DIR:-}"
BASELINE_RESUME_ARGS=()
if [[ -n "$BASELINE_RESUME_DIR" ]]; then
  BASELINE_RESUME_ARGS+=(--baseline-resume-dir "$BASELINE_RESUME_DIR")
fi
PARENT_PREPARE_ARGS=()
# Restart safety: worker_result_timeout_seconds is embedded in write-once
# artifacts (fresh_evidence_manifest.json, input_identity.json).  Changing the
# value under an iteration that already wrote one of them turns every resume
# into an immutable-artifact conflict, so an in-flight iteration keeps the value
# it was started with and only fresh iterations pick up the new default.
WORKER_RESULT_TIMEOUT_SECONDS="${FRESH_PROMPT_DELTA_WORKER_RESULT_TIMEOUT_SECONDS:-2400}"

# Model selection. Defaults reproduce the gpt-4o stack every completed
# iteration was measured with; the GPT-5 family needs a larger output budget
# because reasoning tokens are billed and capped as output.
OPTIMIZER_MODEL_ID="${FRESH_PROMPT_DELTA_OPTIMIZER_MODEL_ID:-gpt-4o}"
GENERATOR_MODEL_ID="${FRESH_PROMPT_DELTA_GENERATOR_MODEL_ID:-gpt-4o-mini}"
OPTIMIZER_MAX_OUTPUT_TOKENS="${FRESH_PROMPT_DELTA_OPTIMIZER_MAX_OUTPUT_TOKENS:-4096}"
GENERATOR_MAX_OUTPUT_TOKENS="${FRESH_PROMPT_DELTA_GENERATOR_MAX_OUTPUT_TOKENS:-512}"
for _budget in "$OPTIMIZER_MAX_OUTPUT_TOKENS" "$GENERATOR_MAX_OUTPUT_TOKENS"; do
  if [[ ! "$_budget" =~ ^[1-9][0-9]*$ ]]; then
    echo "output token budgets must be positive integers" >&2
    exit 2
  fi
done
MODEL_IDENTITY="captioner=${SR_CAPTION_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct};prompt_generator=${GENERATOR_MODEL_ID};dvd_tool=${SR_ORCHESTRATOR_TOOL_MODEL:-gpt-4o};dvd_fallback=gpt-5.5"

# Held-out measurement can run inline or on its own GPUs. Deferring it keeps a
# measurement failure -- or a measurement worker that is not running yet -- from
# stopping the optimization loop, which never reads the numbers anyway.
PROMOTION_POLICY="${FRESH_PROMPT_DELTA_PROMOTION_POLICY:-always_promote_measured_v1}"
MEASUREMENT_QUEUE_ARGS=()
if [[ "$PROMOTION_POLICY" == "promote_and_enqueue_measurement_v1" ]]; then
  MEASUREMENT_QUEUE_DIR="${FRESH_PROMPT_DELTA_MEASUREMENT_QUEUE_DIR:-$PROJECT_ROOT/runs/measurement_queue}"
  mkdir -p "$MEASUREMENT_QUEUE_DIR"
  MEASUREMENT_QUEUE_ARGS+=(--measurement-queue-dir "$MEASUREMENT_QUEUE_DIR")
fi
_frozen_worker_timeout() {
  local file="$1" value
  [[ -f "$file" ]] || return 1
  value="$(jq -r 'first(.. | objects | select(has("worker_result_timeout_seconds")) | .worker_result_timeout_seconds) // empty' "$file" 2>/dev/null)"
  [[ -n "$value" && "$value" != "null" ]] || return 1
  printf '%s' "$value"
}
for _frozen_source in \
    "$OUTPUT_ROOT/input_identity.json" \
    "$EVIDENCE_ROOT/fresh_evidence_manifest.json" \
    "$INPUT_ROOT/component_config.json"; do
  if _frozen_value="$(_frozen_worker_timeout "$_frozen_source")"; then
    WORKER_RESULT_TIMEOUT_SECONDS="$_frozen_value"
    echo "reusing frozen worker_result_timeout_seconds=$WORKER_RESULT_TIMEOUT_SECONDS from $_frozen_source" >&2
    break
  fi
done
if [[ ! "$WORKER_RESULT_TIMEOUT_SECONDS" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "FRESH_PROMPT_DELTA_WORKER_RESULT_TIMEOUT_SECONDS must be a positive number" >&2
  exit 2
fi

if [[ -n "${FRESH_PROMPT_DELTA_PARENT_META_PROMPT:-}" ]]; then
  PARENT_PREPARE_ARGS+=(--parent-meta-prompt "$PARENT")
fi
SPLIT="$PROJECT_ROOT/split_manifest.json"
COMPONENTS="$PROJECT_ROOT/prompt_routing/fixtures/static_meta_replace_body_components.json"
SOURCE_REVISION="$(git rev-parse HEAD)"
CANDIDATE_CREATED_AT="$(date -u -d "${RUN_TIMESTAMP:0:8} ${RUN_TIMESTAMP:9:2}:${RUN_TIMESTAMP:11:2}:${RUN_TIMESTAMP:13:2}" +%Y-%m-%dT%H:%M:%SZ)"

jq -e \
  '(.meta_prompt_id | type == "string" and length > 0) and (.status == "parent" or .status == "confirmed") and (.text | type == "string" and length > 0)' \
  "$PARENT" >/dev/null

SELECTED_VIDEO_IDS_CSV="${FRESH_PROMPT_DELTA_SELECTED_VIDEO_IDS:-0RxMZBLeqRI,TGom0uiW130,w0Wmc8C0Eq0}"
IFS=',' read -r -a SELECTED_VIDEO_IDS <<< "$SELECTED_VIDEO_IDS_CSV"
SELECTED_VIDEO_COUNT="${#SELECTED_VIDEO_IDS[@]}"
if [[ "$SELECTED_VIDEO_COUNT" -lt 1 ]]; then
  echo "FRESH_PROMPT_DELTA_SELECTED_VIDEO_IDS must be non-empty" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${SELECTED_VIDEO_IDS[@]}" | sort -u | wc -l)" -ne "$SELECTED_VIDEO_COUNT" ]]; then
  echo "FRESH_PROMPT_DELTA_SELECTED_VIDEO_IDS must be unique" >&2
  exit 2
fi
VIDEO_ARGS=()
for video_id in "${SELECTED_VIDEO_IDS[@]}"; do
  test -n "$video_id"
  VIDEO_ARGS+=(--video-id "$video_id")
done
PREVIOUS_VIDEO_ARGS=()
if [[ -n "${FRESH_PROMPT_DELTA_PREVIOUS_UPDATE_VIDEO_IDS+x}" ]]; then
  if [[ -n "$FRESH_PROMPT_DELTA_PREVIOUS_UPDATE_VIDEO_IDS" ]]; then
    IFS=',' read -r -a PREVIOUS_VIDEO_IDS <<< "$FRESH_PROMPT_DELTA_PREVIOUS_UPDATE_VIDEO_IDS"
    for video_id in "${PREVIOUS_VIDEO_IDS[@]}"; do
      test -n "$video_id"
      PREVIOUS_VIDEO_ARGS+=(--previous-update-video-id "$video_id")
    done
  fi
else
  PREVIOUS_VIDEO_ARGS=(
    --previous-update-video-id 7D-gxaie6UI
    --previous-update-video-id GLW9omJfAdk
    --previous-update-video-id pU_yyadYgG8
    --previous-update-video-id wCkQ138sg6M
    --previous-update-video-id xKiRmesHWIA)
fi
COHORT_ARGS=()
if [[ -n "${FRESH_PROMPT_DELTA_EVIDENCE_COHORT_FILE:-}" ||
      -n "${FRESH_PROMPT_DELTA_CONFIRMATION_COHORT_FILE:-}" ]]; then
  : "${FRESH_PROMPT_DELTA_EVIDENCE_COHORT_FILE:?both cohort files are required}"
  : "${FRESH_PROMPT_DELTA_CONFIRMATION_COHORT_FILE:?both cohort files are required}"
  test -f "$FRESH_PROMPT_DELTA_EVIDENCE_COHORT_FILE"
  test -f "$FRESH_PROMPT_DELTA_CONFIRMATION_COHORT_FILE"
  COHORT_ARGS=(
    --evidence-video-list "$FRESH_PROMPT_DELTA_EVIDENCE_COHORT_FILE"
    --confirmation-video-list "$FRESH_PROMPT_DELTA_CONFIRMATION_COHORT_FILE")
fi

if [[ ! -f "$INPUT_ROOT/manifest.json" ]]; then
  # A prepare that died midway leaves an input root without a manifest.  Failing
  # on it is deterministic, so the watcher would burn its whole restart budget
  # at the same point.  Move the partial root aside instead; nothing is deleted.
  if [[ -e "$INPUT_ROOT" ]]; then
    QUARANTINE="${INPUT_ROOT}.incomplete.$(date -u +%Y%m%d_%H%M%S)"
    test ! -e "$QUARANTINE"
    mv "$INPUT_ROOT" "$QUARANTINE"
    echo "quarantined incomplete prepared inputs: $QUARANTINE" >&2
  fi
  conda run --no-capture-output -n "$SR_CONDA_ENV" \
    python "$PROJECT_ROOT/scripts/prepare_fresh_prompt_delta_iteration.py" \
    "${PARENT_PREPARE_ARGS[@]}" \
    --split-manifest "$SPLIT" \
    --scaffold-components "$COMPONENTS" \
    "${VIDEO_ARGS[@]}" \
    "${PREVIOUS_VIDEO_ARGS[@]}" \
    "${COHORT_ARGS[@]}" \
    --output-dir "$INPUT_ROOT" \
    --api-endpoint https://api.openai.com/v1/chat/completions \
    --api-key-environment-variable OPENAI_API_KEY \
    --timeout-seconds 600 \
    --proposer-model-id "$OPTIMIZER_MODEL_ID" \
    --proposer-context-limit 128000 \
    --proposer-maximum-output-tokens "$OPTIMIZER_MAX_OUTPUT_TOKENS" \
    --proposer-temperature 0.0 \
    --proposer-policy-version fresh_prompt_delta_proposer_gpt4o_per_qa_isolated_v6 \
    --maximum-deltas-per-qa 2 \
    --proposal-target-policy "${FRESH_PROMPT_DELTA_PROPOSAL_TARGET_POLICY:-incorrect_baseline_qa_only_v1}" \
    --selection-policy source_qa_localized_trajectory_segments_v2_global_only_excluded \
    --global-inspection-boundary-tolerance-seconds 10 \
    --feedback-model-id "$OPTIMIZER_MODEL_ID" \
    --feedback-context-limit 128000 \
    --feedback-maximum-output-tokens "$OPTIMIZER_MAX_OUTPUT_TOKENS" \
    --feedback-temperature 0.0 \
    --feedback-policy-version episode_feedback_request_v12_directional_gpt4o \
    --updater-model-id "$OPTIMIZER_MODEL_ID" \
    --updater-context-limit 128000 \
    --updater-maximum-output-tokens "$OPTIMIZER_MAX_OUTPUT_TOKENS" \
    --updater-temperature 0.0 \
    --updater-policy-version meta_prompt_updater_v5_merge_gpt4o \
    --worker-gpus "$WORKER_GPUS" \
    --worker-result-timeout-seconds "$WORKER_RESULT_TIMEOUT_SECONDS" \
    --prompt-generator-model-id "$GENERATOR_MODEL_ID" \
    --prompt-generator-backend-id openai_chat_completions_vision_replace_body_v1 \
    --prompt-generator-max-tokens "$GENERATOR_MAX_OUTPUT_TOKENS" \
    --history-block-seconds 300 \
    --max-history-captions 30 \
    --dvd-max-iterations 15 \
    --cache-root "$CACHE_ROOT" \
    --cache-manifest-path "$CACHE_ROOT/caption_cache_manifest.jsonl" \
    --paired-model-identity "$MODEL_IDENTITY" \
    --cache-reset-identity "fresh_prompt_delta_clean_${RUN_TIMESTAMP}" \
    --evaluation-pipeline-identity dvd_history_aware_free_form_paired_v1
fi

jq -e '.status == "prepared" and .model_or_api_calls == 0 and .legacy_property_codebook_or_router_used == false and .source_hashes_before == .source_hashes_after' "$INPUT_ROOT/manifest.json" >/dev/null

if [[ "${FRESH_PROMPT_DELTA_MEASURE_PARENT:-false}" == "true" && "$PROMOTION_POLICY" == "promote_and_enqueue_measurement_v1" ]]; then
  # The iteration-0 point is scored by the measurement worker, not here, and
  # it is queued before the evidence phase so the worker has something to do
  # in the window before the first candidate exists.
  PARENT_MEASUREMENT_ROOT="${OUTPUT_ROOT}/parent_measurement"
  conda run --no-capture-output -n "$SR_CONDA_ENV" \
    python "$PROJECT_ROOT/scripts/enqueue_measurement.py" \
    --queue-dir "$MEASUREMENT_QUEUE_DIR" \
    --meta-prompt "$PARENT" \
    --confirmation-cases "$INPUT_ROOT/confirmation_cases.json" \
    --output-dir "$PARENT_MEASUREMENT_ROOT" \
      --iteration-id "iteration_0_initial_meta_prompt_${RUN_TIMESTAMP}" \
    --model-identity "$MODEL_IDENTITY" \
    --decoding-settings "$INPUT_ROOT/paired_decoding_settings.json" \
      --cache-reset-identity "fresh_prompt_delta_clean_${RUN_TIMESTAMP}" \
    --evaluation-pipeline-identity dvd_history_aware_free_form_paired_v1
fi


conda run --no-capture-output -n "$SR_CONDA_ENV" \
  python "$PROJECT_ROOT/scripts/run_fresh_prompt_delta_evidence.py" \
  --prepared-inputs "$INPUT_ROOT" \
  --parent-meta-prompt "$PARENT" \
  --split-manifest "$SPLIT" \
  --output-dir "$EVIDENCE_ROOT" \
  "${BASELINE_RESUME_ARGS[@]}" \
  --source-revision "$SOURCE_REVISION" \
  --worker-gpus "$WORKER_GPUS" \
  --worker-result-timeout-seconds "$WORKER_RESULT_TIMEOUT_SECONDS" \
  --proposer-policy-version-override fresh_prompt_delta_proposer_gpt4o_per_qa_isolated_v6 \
  --selection-policy-override source_qa_localized_trajectory_segments_v2_global_only_excluded \
  --global-inspection-boundary-tolerance-seconds-override 10 \
  --proposer-maximum-calls-override "$((SELECTED_VIDEO_COUNT * 3))" \
  --maximum-deltas-per-qa-override 2

jq -e '(.status == "completed" or .status == "no_eligible_proposal_evidence") and .legacy_property_codebook_or_router_used == false and .source_hashes_before == .source_hashes_after' "$EVIDENCE_ROOT/fresh_evidence_manifest.json" >/dev/null
if [[ "$(jq -r '.status' "$EVIDENCE_ROOT/fresh_evidence_manifest.json")" == "no_eligible_proposal_evidence" ]]; then
  printf 'RUN_TIMESTAMP=%s\nEVIDENCE_ROOT=%s\nSTATUS=no_eligible_proposal_evidence\n' "$RUN_TIMESTAMP" "$EVIDENCE_ROOT"
  exit 0
fi
jq -e '(.episode_paths | length) >= 2' "$EVIDENCE_ROOT/fresh_evidence_manifest.json" >/dev/null
mapfile -t EPISODES < <(jq -r '.episode_paths[]' "$EVIDENCE_ROOT/fresh_evidence_manifest.json")
RESOLVED_COMPONENT_CONFIG="$(jq -r '.resolved_component_config_path' "$EVIDENCE_ROOT/fresh_evidence_manifest.json")"
test -f "$RESOLVED_COMPONENT_CONFIG"
EPISODE_ARGS=()
for episode in "${EPISODES[@]}"; do
  test -f "$episode"
  EPISODE_ARGS+=(--update-episode "$episode")
done
HISTORICAL_MEMORY_ARGS=()
if [[ -n "${FRESH_PROMPT_DELTA_HISTORICAL_MEMORY_CHARACTER_BUDGET:-}" ]]; then
  if [[ ! "$FRESH_PROMPT_DELTA_HISTORICAL_MEMORY_CHARACTER_BUDGET" =~ ^[1-9][0-9]*$ ]]; then
    echo "FRESH_PROMPT_DELTA_HISTORICAL_MEMORY_CHARACTER_BUDGET must be a positive integer" >&2
    exit 2
  fi
  HISTORICAL_MEMORY_ARGS+=(
    --historical-memory-character-budget
    "$FRESH_PROMPT_DELTA_HISTORICAL_MEMORY_CHARACTER_BUDGET")
fi

# Iteration-0 point of the learning curve: score the starting meta-prompt on
# the held-out cases before any update touches it. Decision-free, so it never
# feeds back into the optimization. Under the deferred policy this already
# went to the measurement queue before the evidence phase started.
if [[ "${FRESH_PROMPT_DELTA_MEASURE_PARENT:-false}" == "true" && \
      "$PROMOTION_POLICY" != "promote_and_enqueue_measurement_v1" ]]; then
  PARENT_MEASUREMENT_ROOT="${OUTPUT_ROOT}/parent_measurement"
  if [[ ! -f "$PARENT_MEASUREMENT_ROOT/measurement_summary.json" ]]; then
    conda run --no-capture-output -n "$SR_CONDA_ENV" \
      python "$PROJECT_ROOT/scripts/measure_meta_prompt.py" \
      --meta-prompt "$PARENT" \
      --confirmation-cases "$INPUT_ROOT/confirmation_cases.json" \
      --output-dir "$PARENT_MEASUREMENT_ROOT" \
      --component-factory surrogate_rollout.optimization.checkpoint_g_factory:build_checkpoint_g_components \
      --component-config "$RESOLVED_COMPONENT_CONFIG" \
      --decoding-settings "$INPUT_ROOT/paired_decoding_settings.json" \
      --model-identity "$MODEL_IDENTITY" \
      --cache-reset-identity "fresh_prompt_delta_clean_${RUN_TIMESTAMP}" \
      --evaluation-pipeline-identity dvd_history_aware_free_form_paired_v1
  fi
fi

conda run --no-capture-output -n "$SR_CONDA_ENV" \
  python "$PROJECT_ROOT/scripts/run_prompt_delta_iteration.py" \
  --iteration-id "fresh_prompt_delta_${RUN_TIMESTAMP}" \
  --parent-meta-prompt "$PARENT" \
  "${EPISODE_ARGS[@]}" \
  --confirmation-cases "$INPUT_ROOT/confirmation_cases.json" \
  --output-dir "$OUTPUT_ROOT" \
  --state-dir "$STATE_ROOT" \
  --feedback-memory-bank-dir "$FEEDBACK_MEMORY_BANK_ROOT" \
  "${HISTORICAL_MEMORY_ARGS[@]}" \
  --candidate-created-at "$CANDIDATE_CREATED_AT" \
  --model-identity "$MODEL_IDENTITY" \
  --decoding-settings "$INPUT_ROOT/paired_decoding_settings.json" \
  --cache-reset-identity "fresh_prompt_delta_clean_${RUN_TIMESTAMP}" \
  --evaluation-pipeline-identity dvd_history_aware_free_form_paired_v1 \
  --worker-gpus "$WORKER_GPUS" \
  --worker-result-timeout-seconds "$WORKER_RESULT_TIMEOUT_SECONDS" \
  --promotion-policy "$PROMOTION_POLICY" \
  "${MEASUREMENT_QUEUE_ARGS[@]}" \
  --minimum-confirmation-samples "$(jq 'length' "$INPUT_ROOT/confirmation_cases.json")" \
  --minimum-accuracy-delta 0.0 \
  --maximum-correct-to-wrong 0 \
  --require-no-execution-failures true \
  --initialize-parent-pointer \
  --component-factory surrogate_rollout.optimization.checkpoint_g_factory:build_checkpoint_g_components \
  --component-config "$RESOLVED_COMPONENT_CONFIG"

jq -e '.status == "no_update" or .status == "promoted" or .status == "rolled_back"' "$OUTPUT_ROOT/iteration_result.json" >/dev/null
conda run --no-capture-output -n "$SR_CONDA_ENV" python -c '
import hashlib,json,sys
from pathlib import Path
for manifest_path in sys.argv[1:]:
    m=json.loads(Path(manifest_path).read_text())
    actual={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in m["source_hashes_before"]}
    assert actual == m["source_hashes_before"] == m["source_hashes_after"]
' "$INPUT_ROOT/manifest.json" "$EVIDENCE_ROOT/fresh_evidence_manifest.json"
printf 'RUN_TIMESTAMP=%s\nEVIDENCE_ROOT=%s\nOUTPUT_ROOT=%s\nSTATE_ROOT=%s\n' "$RUN_TIMESTAMP" "$EVIDENCE_ROOT" "$OUTPUT_ROOT" "$STATE_ROOT"
