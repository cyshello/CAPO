#!/usr/bin/env bash
# Operator-run only: one fresh three-video prompt-delta iteration from the
# currently active parent.  This script performs paid/model work; Codex does not.

set -euo pipefail

PROJECT_ROOT="/home/intern/youngseo/surrogate_rollout"
cd "$PROJECT_ROOT"
set -a
source "$PROJECT_ROOT/.env"
set +a
export PYTHONPATH="/home/intern/youngseo:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
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
STATE_ROOT="${RUN_ROOT}_state"
CACHE_ROOT="${RUN_ROOT}_cache"
PARENT="$PROJECT_ROOT/runs/checkpoint_g_pilot_20260720_115145_state/versions/meta_prompt_42bb23b19a51450d6a9c.json"
ACTIVE_POINTER="$PROJECT_ROOT/runs/checkpoint_g_pilot_20260720_115145_state/current_meta_prompt.json"
SPLIT="$PROJECT_ROOT/split_manifest.json"
COMPONENTS="$PROJECT_ROOT/prompt_routing/fixtures/stage4_7_components.json"
SOURCE_REVISION="$(git rev-parse HEAD)"
CANDIDATE_CREATED_AT="$(date -u -d "${RUN_TIMESTAMP:0:8} ${RUN_TIMESTAMP:9:2}:${RUN_TIMESTAMP:11:2}:${RUN_TIMESTAMP:13:2}" +%Y-%m-%dT%H:%M:%SZ)"

if [[ ! -f "$INPUT_ROOT/manifest.json" ]]; then
  test ! -e "$INPUT_ROOT"
  conda run --no-capture-output -n local_llm_vllm \
    python "$PROJECT_ROOT/scripts/prepare_fresh_prompt_delta_iteration.py" \
    --parent-meta-prompt "$PARENT" \
    --active-pointer "$ACTIVE_POINTER" \
    --split-manifest "$SPLIT" \
    --scaffold-components "$COMPONENTS" \
    --video-id 0RxMZBLeqRI \
    --video-id TGom0uiW130 \
    --video-id w0Wmc8C0Eq0 \
    --previous-update-video-id 7D-gxaie6UI \
    --previous-update-video-id GLW9omJfAdk \
    --previous-update-video-id pU_yyadYgG8 \
    --previous-update-video-id wCkQ138sg6M \
    --previous-update-video-id xKiRmesHWIA \
    --output-dir "$INPUT_ROOT" \
    --api-endpoint https://api.openai.com/v1/chat/completions \
    --api-key-environment-variable OPENAI_API_KEY \
    --timeout-seconds 600 \
    --proposer-model-id gpt-4o \
    --proposer-context-limit 128000 \
    --proposer-maximum-output-tokens 4096 \
    --proposer-temperature 0.0 \
    --proposer-policy-version fresh_prompt_delta_proposer_gpt4o_localized_inspection_v4 \
    --maximum-deltas-per-video 3 \
    --selection-policy source_qa_localized_trajectory_segments_only_v1 \
    --global-inspection-boundary-tolerance-seconds 10 \
    --feedback-model-id gpt-4o \
    --feedback-context-limit 128000 \
    --feedback-maximum-output-tokens 4096 \
    --feedback-temperature 0.0 \
    --feedback-policy-version episode_feedback_request_v4_grounded_gpt4o_checkpoint_g \
    --updater-model-id gpt-4o \
    --updater-maximum-output-tokens 4096 \
    --updater-temperature 0.0 \
    --updater-policy-version meta_prompt_updater_v2_grounded_gpt4o_checkpoint_g \
    --worker-gpus "$WORKER_GPUS" \
    --worker-result-timeout-seconds 900 \
    --prompt-generator-model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --prompt-generator-backend-id local_qwen_vllm_history_worker_pool_v1 \
    --prompt-generator-max-tokens 192 \
    --history-block-seconds 300 \
    --max-history-captions 30 \
    --dvd-max-iterations 15 \
    --cache-root "$CACHE_ROOT" \
    --cache-manifest-path "$CACHE_ROOT/caption_cache_manifest.jsonl" \
    --paired-model-identity 'captioner=Qwen/Qwen2.5-VL-7B-Instruct;prompt_generator=Qwen/Qwen2.5-VL-7B-Instruct;dvd_tool=gpt-4o-mini;dvd_fallback=gpt-5.5' \
    --cache-reset-identity "fresh_prompt_delta_clean_${RUN_TIMESTAMP}" \
    --evaluation-pipeline-identity dvd_history_aware_free_form_paired_v1
fi

jq -e '.status == "prepared" and .model_or_api_calls == 0 and .legacy_property_codebook_or_router_used == false and .source_hashes_before == .source_hashes_after' "$INPUT_ROOT/manifest.json" >/dev/null

conda run --no-capture-output -n local_llm_vllm \
  python "$PROJECT_ROOT/scripts/run_fresh_prompt_delta_evidence.py" \
  --prepared-inputs "$INPUT_ROOT" \
  --parent-meta-prompt "$PARENT" \
  --split-manifest "$SPLIT" \
  --output-dir "$EVIDENCE_ROOT" \
  --source-revision "$SOURCE_REVISION" \
  --worker-gpus "$WORKER_GPUS" \
  --worker-result-timeout-seconds 900 \
  --proposer-policy-version-override fresh_prompt_delta_proposer_gpt4o_localized_inspection_v4 \
  --selection-policy-override source_qa_localized_trajectory_segments_only_v1 \
  --global-inspection-boundary-tolerance-seconds-override 10 \
  --proposer-maximum-calls-override 9 \
  --maximum-deltas-per-video-override 3

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

conda run --no-capture-output -n local_llm_vllm \
  python "$PROJECT_ROOT/scripts/run_prompt_delta_iteration.py" \
  --iteration-id "fresh_prompt_delta_${RUN_TIMESTAMP}" \
  --parent-meta-prompt "$PARENT" \
  "${EPISODE_ARGS[@]}" \
  --confirmation-cases "$INPUT_ROOT/confirmation_cases.json" \
  --output-dir "$OUTPUT_ROOT" \
  --state-dir "$STATE_ROOT" \
  --candidate-created-at "$CANDIDATE_CREATED_AT" \
  --model-identity 'captioner=Qwen/Qwen2.5-VL-7B-Instruct;prompt_generator=Qwen/Qwen2.5-VL-7B-Instruct;dvd_tool=gpt-4o-mini;dvd_fallback=gpt-5.5' \
  --decoding-settings "$INPUT_ROOT/paired_decoding_settings.json" \
  --cache-reset-identity "fresh_prompt_delta_clean_${RUN_TIMESTAMP}" \
  --evaluation-pipeline-identity dvd_history_aware_free_form_paired_v1 \
  --worker-gpus "$WORKER_GPUS" \
  --worker-result-timeout-seconds 900 \
  --minimum-confirmation-samples 6 \
  --minimum-accuracy-delta 0.0 \
  --maximum-correct-to-wrong 0 \
  --require-no-execution-failures true \
  --initialize-parent-pointer \
  --component-factory surrogate_rollout.optimization.checkpoint_g_factory:build_checkpoint_g_components \
  --component-config "$RESOLVED_COMPONENT_CONFIG"

jq -e '.status == "no_update" or .status == "promoted" or .status == "rolled_back"' "$OUTPUT_ROOT/iteration_result.json" >/dev/null
conda run --no-capture-output -n local_llm_vllm python -c '
import hashlib,json,sys
from pathlib import Path
for manifest_path in sys.argv[1:]:
    m=json.loads(Path(manifest_path).read_text())
    actual={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in m["source_hashes_before"]}
    assert actual == m["source_hashes_before"] == m["source_hashes_after"]
' "$INPUT_ROOT/manifest.json" "$EVIDENCE_ROOT/fresh_evidence_manifest.json"
printf 'RUN_TIMESTAMP=%s\nEVIDENCE_ROOT=%s\nOUTPUT_ROOT=%s\nSTATE_ROOT=%s\n' "$RUN_TIMESTAMP" "$EVIDENCE_ROOT" "$OUTPUT_ROOT" "$STATE_ROOT"
