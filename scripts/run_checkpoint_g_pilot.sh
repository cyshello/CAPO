#!/usr/bin/env bash
# Operator-run Checkpoint G pilot: five saved update episodes plus the active
# two-video/six-QA confirmation holdout.  Codex must not execute this script.

set -euo pipefail

PROJECT_ROOT="/home/intern/youngseo/surrogate_rollout"
cd "$PROJECT_ROOT"
set -a
source "/home/intern/youngseo/surrogate_rollout/.env"
set +a
export PYTHONPATH="/home/intern/youngseo:/home/intern/youngseo/surrogate_rollout${PYTHONPATH:+:$PYTHONPATH}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY is missing from the project .env}"

PILOT_TIMESTAMP="${CHECKPOINT_G_PILOT_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ ! "$PILOT_TIMESTAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "CHECKPOINT_G_PILOT_TIMESTAMP must use YYYYMMDD_HHMMSS" >&2
  exit 2
fi

INPUT_ROOT="$PROJECT_ROOT/runs/checkpoint_g_pilot_${PILOT_TIMESTAMP}_inputs"
OUTPUT_ROOT="$PROJECT_ROOT/runs/checkpoint_g_pilot_${PILOT_TIMESTAMP}_output"
STATE_ROOT="$PROJECT_ROOT/runs/checkpoint_g_pilot_${PILOT_TIMESTAMP}_state"
CACHE_ROOT="$PROJECT_ROOT/runs/checkpoint_g_pilot_${PILOT_TIMESTAMP}_cache"
PARENT="$PROJECT_ROOT/runs/meta_prompt_bootstrap_20260720_074042/parent_meta_prompt.json"
CANDIDATE_CREATED_AT="$(date -u -d "${PILOT_TIMESTAMP:0:8} ${PILOT_TIMESTAMP:9:2}:${PILOT_TIMESTAMP:11:2}:${PILOT_TIMESTAMP:13:2}" +%Y-%m-%dT%H:%M:%SZ)"

if [[ ! -f "$INPUT_ROOT/manifest.json" ]]; then
  test ! -e "$INPUT_ROOT"
  conda run --no-capture-output -n local_llm_vllm \
    python "$PROJECT_ROOT/scripts/prepare_checkpoint_g_pilot.py" \
    --split-manifest "$PROJECT_ROOT/split_manifest.json" \
    --parent-meta-prompt-id meta_prompt_42bb23b19a51450d6a9c \
    --intervention-result "$PROJECT_ROOT/runs/phase4_memory_k4_retrieval50_004_output/interventions/xKiRmesHWIA/xKiRmesHWIA/candidate_7ed6e9003137df81fe4e_pf98b2b2e263f/result.json" \
    --baseline-manifest "$PROJECT_ROOT/runs/phase4_memory_k4_retrieval50_004_output/baseline_videos/xKiRmesHWIA/baseline/xKiRmesHWIA/video_complete.json" \
    --intervention-result "$PROJECT_ROOT/runs/phase4_memory_k4_retrieval50_009_output/interventions/wCkQ138sg6M/wCkQ138sg6M/candidate_9efb1738da7d6449d420_pe0ec0a4dba64/result.json" \
    --baseline-manifest "$PROJECT_ROOT/runs/phase4_memory_k4_retrieval50_009_output/baseline_videos/wCkQ138sg6M/baseline/wCkQ138sg6M/video_complete.json" \
    --intervention-result "$PROJECT_ROOT/runs/phase4_memory_k4_retrieval50_004_output/interventions/pU_yyadYgG8/pU_yyadYgG8/candidate_8c19888d541e62fc2dc2_p027e7d36938e/result.json" \
    --baseline-manifest "$PROJECT_ROOT/runs/phase4_memory_k4_retrieval50_004_output/baseline_videos/pU_yyadYgG8/baseline/pU_yyadYgG8/video_complete.json" \
    --intervention-result "$PROJECT_ROOT/runs/phase4_memory_k4_isolated_001_output/interventions/GLW9omJfAdk/GLW9omJfAdk/candidate_4f539a8f20911b379a17_p166023aa756f/result.json" \
    --baseline-manifest "$PROJECT_ROOT/runs/phase4_memory_k4_isolated_001_output/baseline_videos/GLW9omJfAdk/baseline/GLW9omJfAdk/video_complete.json" \
    --intervention-result "$PROJECT_ROOT/runs/phase4_memory_k4_isolated_001_output/interventions/7D-gxaie6UI/7D-gxaie6UI/candidate_284d53fde9070f949063_pbad21bc654ed/result.json" \
    --baseline-manifest "$PROJECT_ROOT/runs/phase4_memory_k4_isolated_001_output/baseline_videos/7D-gxaie6UI/baseline/7D-gxaie6UI/video_complete.json" \
    --output-dir "$INPUT_ROOT" \
    --api-endpoint https://api.openai.com/v1/chat/completions \
    --api-key-environment-variable OPENAI_API_KEY \
    --timeout-seconds 600 \
    --feedback-model-id gpt-4o \
    --feedback-context-limit 128000 \
    --feedback-maximum-output-tokens 4096 \
    --feedback-temperature 0.0 \
    --feedback-policy-version episode_feedback_request_v4_grounded_gpt4o_checkpoint_g \
    --updater-model-id gpt-4o \
    --updater-maximum-output-tokens 4096 \
    --updater-temperature 0.0 \
    --updater-policy-version meta_prompt_updater_v2_grounded_gpt4o_checkpoint_g \
    --worker-gpus 4,5,6,7 \
    --prompt-generator-backend-id local_qwen_vllm_history_worker_pool_v1 \
    --prompt-generator-max-tokens 192 \
    --scaffold-components "$PROJECT_ROOT/prompt_routing/fixtures/stage4_7_components.json" \
    --cache-root "$CACHE_ROOT/captions" \
    --cache-manifest-path "$CACHE_ROOT/caption_cache_manifest.jsonl" \
    --history-block-seconds 300 \
    --max-history-captions 30 \
    --dvd-max-iterations 15 \
    --paired-model-identity 'captioner=Qwen/Qwen2.5-VL-7B-Instruct;prompt_generator=Qwen/Qwen2.5-VL-7B-Instruct;dvd_tool=gpt-4o-mini;dvd_fallback=gpt-5.5' \
    --cache-reset-identity "checkpoint_g_clean_paired_${PILOT_TIMESTAMP}" \
    --evaluation-pipeline-identity dvd_history_aware_free_form_paired_v1
fi

jq -e '.status == "prepared" and .model_or_api_calls == 0 and .source_hashes_before == .source_hashes_after' "$INPUT_ROOT/manifest.json" >/dev/null
mapfile -t UPDATE_EPISODES < <(jq -r '.update_episode_paths[]' "$INPUT_ROOT/manifest.json")
if [[ "${#UPDATE_EPISODES[@]}" -ne 5 ]]; then
  echo "expected exactly five prepared update episodes" >&2
  exit 2
fi
UPDATE_ARGS=()
for episode in "${UPDATE_EPISODES[@]}"; do
  test -f "$episode"
  UPDATE_ARGS+=(--update-episode "$episode")
done

conda run --no-capture-output -n local_llm_vllm \
  python "$PROJECT_ROOT/scripts/run_prompt_delta_iteration.py" \
  --iteration-id "checkpoint_g_pilot_${PILOT_TIMESTAMP}" \
  --parent-meta-prompt "$PARENT" \
  "${UPDATE_ARGS[@]}" \
  --confirmation-cases "$INPUT_ROOT/confirmation_cases.json" \
  --output-dir "$OUTPUT_ROOT" \
  --state-dir "$STATE_ROOT" \
  --candidate-created-at "$CANDIDATE_CREATED_AT" \
  --model-identity 'captioner=Qwen/Qwen2.5-VL-7B-Instruct;prompt_generator=Qwen/Qwen2.5-VL-7B-Instruct;dvd_tool=gpt-4o-mini;dvd_fallback=gpt-5.5' \
  --decoding-settings "$INPUT_ROOT/paired_decoding_settings.json" \
  --cache-reset-identity "checkpoint_g_clean_paired_${PILOT_TIMESTAMP}" \
  --evaluation-pipeline-identity dvd_history_aware_free_form_paired_v1 \
  --minimum-confirmation-samples 6 \
  --minimum-accuracy-delta 0.0 \
  --maximum-correct-to-wrong 0 \
  --require-no-execution-failures true \
  --initialize-parent-pointer \
  --component-factory surrogate_rollout.optimization.checkpoint_g_factory:build_checkpoint_g_components \
  --component-config "$INPUT_ROOT/component_config.json"

jq -e '.status == "no_update" or .status == "promoted" or .status == "rolled_back"' "$OUTPUT_ROOT/iteration_result.json" >/dev/null
conda run --no-capture-output -n local_llm_vllm python -c '
import hashlib,json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in m["source_hashes_before"]}
assert after == m["source_hashes_before"] == m["source_hashes_after"], "source artifact mutation detected"
' "$INPUT_ROOT/manifest.json"
printf 'CHECKPOINT_G_PILOT_TIMESTAMP=%s\nOUTPUT_ROOT=%s\nSTATE_ROOT=%s\n' "$PILOT_TIMESTAMP" "$OUTPUT_ROOT" "$STATE_ROOT"
