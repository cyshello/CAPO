#!/usr/bin/env bash
# Operator-run only: one fresh caption-PROMPT optimization iteration (static
# generator) end to end: prepare -> static evidence -> feedback/update/confirm.
# Mirrors surrogate_rollout/scripts/run_fresh_prompt_delta_iteration.sh but:
#   * evidence uses caption_prompt_opt/run_static_evidence.py (no generator call)
#   * iteration uses caption_prompt_opt.factory:build_caption_prompt_components
#   * parent is the caption prompt (init_caption_prompt.json)
#   * cache/pipeline identities are the isolated static caption-prompt namespace
# This script performs paid/model work; do not let an assistant run it.

set -euo pipefail

# caption_prompt_opt package root (this script's parent dir).
CPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Locate the surrogate_rollout checkout, whether caption_prompt_opt is nested
# inside it or a sibling of it. Override with SR_PROJECT_ROOT if neither matches.
if [[ -n "${SR_PROJECT_ROOT:-}" ]]; then
  SR_ROOT="$SR_PROJECT_ROOT"
elif [[ -f "$(dirname "$CPO_ROOT")/scripts/run_prompt_delta_iteration.py" ]]; then
  SR_ROOT="$(dirname "$CPO_ROOT")"                                   # nested
elif [[ -f "$(dirname "$CPO_ROOT")/surrogate_rollout/scripts/run_prompt_delta_iteration.py" ]]; then
  SR_ROOT="$(dirname "$CPO_ROOT")/surrogate_rollout"                 # sibling
else
  echo "cannot locate surrogate_rollout checkout; set SR_PROJECT_ROOT" >&2; exit 2
fi
# PYTHONPATH gets both, so top-level `import caption_prompt_opt` and
# `import surrogate_rollout` both resolve in either layout.
REPO_PARENT="$(dirname "$SR_ROOT")"

: "${SR_CONDA_ENV:=local_llm_vllm}"
cd "$SR_ROOT"
set -a
source "$SR_ROOT/.env"
set +a
export PYTHONPATH="$REPO_PARENT:$SR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY is missing from $SR_ROOT/.env}"

# --- Captioner selection (Qwen VL, served by vLLM). Swaps model AND cache key. -
: "${SR_CAPTION_MODEL_ID:?export SR_CAPTION_MODEL_ID (e.g. Qwen/Qwen2.5-VL-7B-Instruct)}"
export SR_CAPTION_MODEL_ID
CAPTIONER_MODEL_ID="$SR_CAPTION_MODEL_ID"
# Captioner (Qwen25VLCaptioner) needs a VISION-language model. The class is
# model-agnostic (AutoProcessor + model's own chat template + qwen_vl_utils), so
# any Qwen-family VL id works; just confirm the id is the multimodal one and that
# transformers/qwen_vl_utils/vLLM(nightly) recognize it.
echo "captioner: $CAPTIONER_MODEL_ID (must be a vision-language model)"

RUN_TIMESTAMP="${CAPTION_PROMPT_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
WORKER_GPUS="${CAPTION_PROMPT_WORKER_GPUS:-0,1,2,3}"
if [[ ! "$RUN_TIMESTAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "CAPTION_PROMPT_TIMESTAMP must use YYYYMMDD_HHMMSS" >&2; exit 2
fi
if [[ ! "$WORKER_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "CAPTION_PROMPT_WORKER_GPUS must be a comma-separated GPU list" >&2; exit 2
fi

# --- Isolated run roots (never collide with a meta-prompt run) ------------------
RUN_ROOT="$SR_ROOT/runs/caption_prompt_iteration_${RUN_TIMESTAMP}"
INPUT_ROOT="${RUN_ROOT}_inputs"
EVIDENCE_ROOT="${RUN_ROOT}_evidence"
OUTPUT_ROOT="${RUN_ROOT}_output"
STATE_ROOT="${CAPTION_PROMPT_STATE_ROOT:-${RUN_ROOT}_state}"
CACHE_ROOT="${CAPTION_PROMPT_CACHE_ROOT:-${RUN_ROOT}_cache}"
FEEDBACK_MEMORY_BANK_ROOT="${CAPTION_PROMPT_MEMORY_BANK_ROOT:-$SR_ROOT/runs/caption_prompt_feedback_memory_bank}"
PARENT="${CAPTION_PROMPT_PARENT:-$CPO_ROOT/prompts/init_caption_prompt.json}"

# --- Isolated identities (static caption-prompt namespace) ----------------------
EVAL_PIPELINE_IDENTITY="dvd_history_aware_static_caption_prompt_paired_v1"
CACHE_RESET_IDENTITY="caption_prompt_static_clean_${RUN_TIMESTAMP}"
UPDATER_POLICY_VERSION="caption_prompt_updater_v1"

# Optimizer (feedback/proposer/updater) stack — reused, paid. GPT-5 family needs
# a larger output budget (reasoning billed as output).
OPTIMIZER_MODEL_ID="${CAPTION_PROMPT_OPTIMIZER_MODEL_ID:-gpt-5-mini}"
OPTIMIZER_MAX_OUTPUT_TOKENS="${CAPTION_PROMPT_OPTIMIZER_MAX_OUTPUT_TOKENS:-32000}"
if [[ ! "$OPTIMIZER_MAX_OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "optimizer output token budget must be a positive integer" >&2; exit 2
fi

# prompt_generator is VESTIGIAL under the static path (no model call). Marked
# 'static' so provenance and cache identity say so.
MODEL_IDENTITY="captioner=${CAPTIONER_MODEL_ID};prompt_generator=static;dvd_tool=${SR_ORCHESTRATOR_TOOL_MODEL:-gpt-5-mini};dvd_fallback=gpt-5.5"

WORKER_RESULT_TIMEOUT_SECONDS="${CAPTION_PROMPT_WORKER_RESULT_TIMEOUT_SECONDS:-2400}"
PROMOTION_POLICY="${CAPTION_PROMPT_PROMOTION_POLICY:-always_promote_measured_v1}"
MEASUREMENT_QUEUE_ARGS=()
if [[ "$PROMOTION_POLICY" == "promote_and_enqueue_measurement_v1" ]]; then
  MEASUREMENT_QUEUE_DIR="${CAPTION_PROMPT_MEASUREMENT_QUEUE_DIR:-$SR_ROOT/runs/caption_prompt_measurement_queue}"
  mkdir -p "$MEASUREMENT_QUEUE_DIR"
  MEASUREMENT_QUEUE_ARGS+=(--measurement-queue-dir "$MEASUREMENT_QUEUE_DIR")
fi

SPLIT="$SR_ROOT/split_manifest.json"
COMPONENTS="$SR_ROOT/prompt_routing/fixtures/static_meta_replace_body_components.json"
SOURCE_REVISION="$(git -C "$SR_ROOT" rev-parse HEAD)"
CANDIDATE_CREATED_AT="$(date -u -d "${RUN_TIMESTAMP:0:8} ${RUN_TIMESTAMP:9:2}:${RUN_TIMESTAMP:11:2}:${RUN_TIMESTAMP:13:2}" +%Y-%m-%dT%H:%M:%SZ)"

jq -e '(.meta_prompt_id|type=="string" and length>0) and (.status=="parent" or .status=="confirmed") and (.text|type=="string" and length>0)' "$PARENT" >/dev/null

SELECTED_VIDEO_IDS_CSV="${CAPTION_PROMPT_SELECTED_VIDEO_IDS:-0RxMZBLeqRI,TGom0uiW130,w0Wmc8C0Eq0}"
IFS=',' read -r -a SELECTED_VIDEO_IDS <<< "$SELECTED_VIDEO_IDS_CSV"
SELECTED_VIDEO_COUNT="${#SELECTED_VIDEO_IDS[@]}"
[[ "$SELECTED_VIDEO_COUNT" -ge 1 ]] || { echo "CAPTION_PROMPT_SELECTED_VIDEO_IDS must be non-empty" >&2; exit 2; }
VIDEO_ARGS=(); for v in "${SELECTED_VIDEO_IDS[@]}"; do test -n "$v"; VIDEO_ARGS+=(--video-id "$v"); done

PARENT_PREPARE_ARGS=()
if [[ -n "${CAPTION_PROMPT_PARENT:-}" ]]; then PARENT_PREPARE_ARGS+=(--parent-meta-prompt "$PARENT"); fi

# 1) PREPARE — build inputs (manifest, component_config, confirmation_cases,
#    paired_decoding_settings). prompt-generator args are static (vestigial).
if [[ ! -f "$INPUT_ROOT/manifest.json" ]]; then
  if [[ -e "$INPUT_ROOT" ]]; then
    Q="${INPUT_ROOT}.incomplete.$(date -u +%Y%m%d_%H%M%S)"; test ! -e "$Q"; mv "$INPUT_ROOT" "$Q"
    echo "quarantined incomplete prepared inputs: $Q" >&2
  fi
  conda run --no-capture-output -n "$SR_CONDA_ENV" \
    python "$SR_ROOT/scripts/prepare_fresh_prompt_delta_iteration.py" \
    "${PARENT_PREPARE_ARGS[@]}" \
    --split-manifest "$SPLIT" \
    --scaffold-components "$COMPONENTS" \
    "${VIDEO_ARGS[@]}" \
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
    --proposal-target-policy "${CAPTION_PROMPT_PROPOSAL_TARGET_POLICY:-incorrect_baseline_qa_only_v1}" \
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
    --updater-policy-version "$UPDATER_POLICY_VERSION" \
    --worker-gpus "$WORKER_GPUS" \
    --worker-result-timeout-seconds "$WORKER_RESULT_TIMEOUT_SECONDS" \
    --prompt-generator-model-id static \
    --prompt-generator-backend-id static \
    --prompt-generator-max-tokens 1 \
    --history-block-seconds 300 \
    --max-history-captions 30 \
    --dvd-max-iterations 15 \
    --cache-root "$CACHE_ROOT" \
    --cache-manifest-path "$CACHE_ROOT/caption_cache_manifest.jsonl" \
    --paired-model-identity "$MODEL_IDENTITY" \
    --cache-reset-identity "$CACHE_RESET_IDENTITY" \
    --evaluation-pipeline-identity "$EVAL_PIPELINE_IDENTITY"
fi
jq -e '.status=="prepared" and .model_or_api_calls==0' "$INPUT_ROOT/manifest.json" >/dev/null

# 2) EVIDENCE — static generator; rollout episodes carry provider=static.
conda run --no-capture-output -n "$SR_CONDA_ENV" \
  python "$CPO_ROOT/run_static_evidence.py" \
  --prepared-inputs "$INPUT_ROOT" \
  --parent-meta-prompt "$PARENT" \
  --split-manifest "$SPLIT" \
  --output-dir "$EVIDENCE_ROOT" \
  --source-revision "$SOURCE_REVISION" \
  --worker-gpus "$WORKER_GPUS" \
  --worker-result-timeout-seconds "$WORKER_RESULT_TIMEOUT_SECONDS" \
  --proposer-policy-version-override fresh_prompt_delta_proposer_gpt4o_per_qa_isolated_v6 \
  --selection-policy-override source_qa_localized_trajectory_segments_v2_global_only_excluded \
  --global-inspection-boundary-tolerance-seconds-override 10 \
  --proposer-maximum-calls-override "$((SELECTED_VIDEO_COUNT * 3))" \
  --maximum-deltas-per-qa-override 2

jq -e '(.status=="completed" or .status=="no_eligible_proposal_evidence")' "$EVIDENCE_ROOT/fresh_evidence_manifest.json" >/dev/null
if [[ "$(jq -r '.status' "$EVIDENCE_ROOT/fresh_evidence_manifest.json")" == "no_eligible_proposal_evidence" ]]; then
  printf 'RUN_TIMESTAMP=%s\nEVIDENCE_ROOT=%s\nSTATUS=no_eligible_proposal_evidence\n' "$RUN_TIMESTAMP" "$EVIDENCE_ROOT"; exit 0
fi
jq -e '(.episode_paths|length)>=2' "$EVIDENCE_ROOT/fresh_evidence_manifest.json" >/dev/null
mapfile -t EPISODES < <(jq -r '.episode_paths[]' "$EVIDENCE_ROOT/fresh_evidence_manifest.json")
RESOLVED_COMPONENT_CONFIG="$(jq -r '.resolved_component_config_path' "$EVIDENCE_ROOT/fresh_evidence_manifest.json")"
test -f "$RESOLVED_COMPONENT_CONFIG"
EPISODE_ARGS=(); for e in "${EPISODES[@]}"; do test -f "$e"; EPISODE_ARGS+=(--update-episode "$e"); done

# 3) ITERATION — feedback (reused) -> caption-prompt updater -> static confirm.
conda run --no-capture-output -n "$SR_CONDA_ENV" \
  python "$SR_ROOT/scripts/run_prompt_delta_iteration.py" \
  --iteration-id "caption_prompt_${RUN_TIMESTAMP}" \
  --parent-meta-prompt "$PARENT" \
  "${EPISODE_ARGS[@]}" \
  --confirmation-cases "$INPUT_ROOT/confirmation_cases.json" \
  --output-dir "$OUTPUT_ROOT" \
  --state-dir "$STATE_ROOT" \
  --feedback-memory-bank-dir "$FEEDBACK_MEMORY_BANK_ROOT" \
  --candidate-created-at "$CANDIDATE_CREATED_AT" \
  --model-identity "$MODEL_IDENTITY" \
  --decoding-settings "$INPUT_ROOT/paired_decoding_settings.json" \
  --cache-reset-identity "$CACHE_RESET_IDENTITY" \
  --evaluation-pipeline-identity "$EVAL_PIPELINE_IDENTITY" \
  --worker-gpus "$WORKER_GPUS" \
  --worker-result-timeout-seconds "$WORKER_RESULT_TIMEOUT_SECONDS" \
  --promotion-policy "$PROMOTION_POLICY" \
  "${MEASUREMENT_QUEUE_ARGS[@]}" \
  --minimum-confirmation-samples "$(jq 'length' "$INPUT_ROOT/confirmation_cases.json")" \
  --minimum-accuracy-delta 0.0 \
  --maximum-correct-to-wrong 0 \
  --require-no-execution-failures true \
  --initialize-parent-pointer \
  --component-factory caption_prompt_opt.factory:build_caption_prompt_components \
  --component-config "$RESOLVED_COMPONENT_CONFIG"

jq -e '.status=="no_update" or .status=="promoted" or .status=="rolled_back"' "$OUTPUT_ROOT/iteration_result.json" >/dev/null
printf 'RUN_TIMESTAMP=%s\nEVIDENCE_ROOT=%s\nOUTPUT_ROOT=%s\nSTATE_ROOT=%s\nCAPTIONER=%s\n' \
  "$RUN_TIMESTAMP" "$EVIDENCE_ROOT" "$OUTPUT_ROOT" "$STATE_ROOT" "$CAPTIONER_MODEL_ID"
