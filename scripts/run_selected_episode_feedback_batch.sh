#!/usr/bin/env bash
# Run the reviewed five-episode feedback batch. Each child command owns exactly
# one provider attempt and implements no retry, repair, or fallback.

set -euo pipefail

PROJECT_ROOT="/home/intern/youngseo/surrogate_rollout"
cd "$PROJECT_ROOT"

set -a
source "/home/intern/youngseo/surrogate_rollout/.env"
set +a

export PYTHONPATH="/home/intern/youngseo${PYTHONPATH:+:$PYTHONPATH}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY is missing from the project .env}"

BATCH_TIMESTAMP="${SELECTED_FEEDBACK_BATCH_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ ! "$BATCH_TIMESTAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "SELECTED_FEEDBACK_BATCH_TIMESTAMP must use YYYYMMDD_HHMMSS" >&2
  exit 2
fi

OUTPUT_ROOT="$PROJECT_ROOT/runs/selected_episode_feedback_$BATCH_TIMESTAMP"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "output root already exists; refusing overwrite: $OUTPUT_ROOT" >&2
  exit 2
fi

PARENT_META_PROMPT_ID="meta_prompt_42bb23b19a51450d6a9c"

INTERVENTION_RESULTS=(
  "/home/intern/youngseo/surrogate_rollout/runs/phase4_memory_k4_retrieval50_004_output/interventions/xKiRmesHWIA/xKiRmesHWIA/candidate_7ed6e9003137df81fe4e_pf98b2b2e263f/result.json"
  "/home/intern/youngseo/surrogate_rollout/runs/phase4_memory_k4_retrieval50_009_output/interventions/wCkQ138sg6M/wCkQ138sg6M/candidate_9efb1738da7d6449d420_pe0ec0a4dba64/result.json"
  "/home/intern/youngseo/surrogate_rollout/runs/phase4_memory_k4_retrieval50_004_output/interventions/pU_yyadYgG8/pU_yyadYgG8/candidate_8c19888d541e62fc2dc2_p027e7d36938e/result.json"
  "/home/intern/youngseo/surrogate_rollout/runs/phase4_memory_k4_isolated_001_output/interventions/GLW9omJfAdk/GLW9omJfAdk/candidate_4f539a8f20911b379a17_p166023aa756f/result.json"
  "/home/intern/youngseo/surrogate_rollout/runs/phase4_memory_k4_isolated_001_output/interventions/7D-gxaie6UI/7D-gxaie6UI/candidate_284d53fde9070f949063_pbad21bc654ed/result.json"
)

BASELINE_MANIFESTS=(
  "/home/intern/youngseo/surrogate_rollout/runs/phase4_memory_k4_retrieval50_004_output/baseline_videos/xKiRmesHWIA/baseline/xKiRmesHWIA/video_complete.json"
  "/home/intern/youngseo/surrogate_rollout/runs/phase4_memory_k4_retrieval50_009_output/baseline_videos/wCkQ138sg6M/baseline/wCkQ138sg6M/video_complete.json"
  "/home/intern/youngseo/surrogate_rollout/runs/phase4_memory_k4_retrieval50_004_output/baseline_videos/pU_yyadYgG8/baseline/pU_yyadYgG8/video_complete.json"
  "/home/intern/youngseo/surrogate_rollout/runs/phase4_memory_k4_isolated_001_output/baseline_videos/GLW9omJfAdk/baseline/GLW9omJfAdk/video_complete.json"
  "/home/intern/youngseo/surrogate_rollout/runs/phase4_memory_k4_isolated_001_output/baseline_videos/7D-gxaie6UI/baseline/7D-gxaie6UI/video_complete.json"
)

VIDEO_IDS=(
  "xKiRmesHWIA"
  "wCkQ138sg6M"
  "pU_yyadYgG8"
  "GLW9omJfAdk"
  "7D-gxaie6UI"
)

DELTA_IDS=(
  "candidate_7ed6e9003137df81fe4e"
  "candidate_9efb1738da7d6449d420"
  "candidate_8c19888d541e62fc2dc2"
  "candidate_4f539a8f20911b379a17"
  "candidate_284d53fde9070f949063"
)

OUTPUT_NAMES=(
  "01_xKiRmesHWIA_candidate_7ed6e9003137df81fe4e"
  "02_wCkQ138sg6M_candidate_9efb1738da7d6449d420"
  "03_pU_yyadYgG8_candidate_8c19888d541e62fc2dc2"
  "04_GLW9omJfAdk_candidate_4f539a8f20911b379a17"
  "05_7D-gxaie6UI_candidate_284d53fde9070f949063"
)

for index in "${!INTERVENTION_RESULTS[@]}"; do
  intervention_result="${INTERVENTION_RESULTS[$index]}"
  baseline_manifest="${BASELINE_MANIFESTS[$index]}"
  video_id="${VIDEO_IDS[$index]}"
  delta_id="${DELTA_IDS[$index]}"
  output_directory="$OUTPUT_ROOT/${OUTPUT_NAMES[$index]}"

  test -f "$intervention_result"
  test -f "$baseline_manifest"
  test ! -e "$output_directory"
  jq -e --arg video_id "$video_id" --arg delta_id "$delta_id" '
    .status == "completed"
    and .source_video_id == $video_id
    and .candidate_property_id == $delta_id
  ' "$intervention_result" >/dev/null

  conda run --no-capture-output -n local_llm_vllm \
    python "$PROJECT_ROOT/scripts/run_episode_feedback_once.py" \
    --intervention-result "$intervention_result" \
    --baseline-manifest "$baseline_manifest" \
    --parent-meta-prompt-id "$PARENT_META_PROMPT_ID" \
    --output-dir "$output_directory" \
    --provider openai_api \
    --api-endpoint https://api.openai.com/v1/chat/completions \
    --model-id gpt-4o \
    --request-representation model_compact \
    --context-limit 128000 \
    --maximum-output-tokens 4096 \
    --temperature 0.0 \
    --feedback-policy-version episode_feedback_request_v4_grounded_gpt4o_selected_v2 \
    --timeout-seconds 600

  test -f "$output_directory/parsed_feedback.json"
  test -f "$output_directory/semantic_eligibility.json"
  jq -e '
    .status == "success"
    and .provider_call_count == 1
    and .adapter_call_count == 1
    and .supporting_id_validation == "passed"
    and .semantic_eligibility == "passed"
    and .source_aggregate_sha256_before == .source_aggregate_sha256_after
  ' "$output_directory/manifest.json" >/dev/null
  jq -e '
    .eligible == true
    and (.reasons | length) == 0
  ' "$output_directory/semantic_eligibility.json" >/dev/null
  conda run -n local_llm_vllm python -c '
import json
import sys
from pathlib import Path
from surrogate_rollout.schemas import sha256_json
from surrogate_rollout.optimization.schemas import episode_feedback_from_json
from surrogate_rollout.prompt_routing.schemas import dumps_canonical

path = Path(sys.argv[1])
feedback = episode_feedback_from_json(json.loads(path.read_text(encoding="utf-8")))
eligibility = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert feedback.feedback_id
assert feedback.episode_id
assert eligibility["feedback_id"] == feedback.feedback_id
assert eligibility["episode_id"] == feedback.episode_id
assert eligibility["feedback_sha256"] == sha256_json(
    json.loads(dumps_canonical(feedback)))
' "$output_directory/parsed_feedback.json" \
  "$output_directory/semantic_eligibility.json"
done

printf 'SELECTED_FEEDBACK_ROOT=%s\n' "$OUTPUT_ROOT"
