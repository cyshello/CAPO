#!/usr/bin/env bash
# Consolidate one completed selected-feedback batch with exactly one updater
# provider attempt. This script never promotes or activates the candidate.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/run_selected_meta_prompt_update.sh <selected-feedback-root>" >&2
  exit 2
fi

PROJECT_ROOT="/home/intern/youngseo/surrogate_rollout"
cd "$PROJECT_ROOT"

set -a
source "/home/intern/youngseo/surrogate_rollout/.env"
set +a

export PYTHONPATH="/home/intern/youngseo${PYTHONPATH:+:$PYTHONPATH}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY is missing from the project .env}"

FEEDBACK_ROOT="$(realpath "$1")"
PARENT_META_PROMPT_ARTIFACT="/home/intern/youngseo/surrogate_rollout/runs/meta_prompt_bootstrap_20260720_074042/parent_meta_prompt.json"
UPDATE_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_DIRECTORY="$PROJECT_ROOT/runs/selected_meta_prompt_update_$UPDATE_TIMESTAMP"

FEEDBACK_ARTIFACTS=(
  "$FEEDBACK_ROOT/01_xKiRmesHWIA_candidate_7ed6e9003137df81fe4e/parsed_feedback.json"
  "$FEEDBACK_ROOT/02_wCkQ138sg6M_candidate_9efb1738da7d6449d420/parsed_feedback.json"
  "$FEEDBACK_ROOT/03_pU_yyadYgG8_candidate_8c19888d541e62fc2dc2/parsed_feedback.json"
  "$FEEDBACK_ROOT/04_GLW9omJfAdk_candidate_4f539a8f20911b379a17/parsed_feedback.json"
  "$FEEDBACK_ROOT/05_7D-gxaie6UI_candidate_284d53fde9070f949063/parsed_feedback.json"
)

MANIFESTS=(
  "$FEEDBACK_ROOT/01_xKiRmesHWIA_candidate_7ed6e9003137df81fe4e/manifest.json"
  "$FEEDBACK_ROOT/02_wCkQ138sg6M_candidate_9efb1738da7d6449d420/manifest.json"
  "$FEEDBACK_ROOT/03_pU_yyadYgG8_candidate_8c19888d541e62fc2dc2/manifest.json"
  "$FEEDBACK_ROOT/04_GLW9omJfAdk_candidate_4f539a8f20911b379a17/manifest.json"
  "$FEEDBACK_ROOT/05_7D-gxaie6UI_candidate_284d53fde9070f949063/manifest.json"
)

ELIGIBILITY_ARTIFACTS=(
  "$FEEDBACK_ROOT/01_xKiRmesHWIA_candidate_7ed6e9003137df81fe4e/semantic_eligibility.json"
  "$FEEDBACK_ROOT/02_wCkQ138sg6M_candidate_9efb1738da7d6449d420/semantic_eligibility.json"
  "$FEEDBACK_ROOT/03_pU_yyadYgG8_candidate_8c19888d541e62fc2dc2/semantic_eligibility.json"
  "$FEEDBACK_ROOT/04_GLW9omJfAdk_candidate_4f539a8f20911b379a17/semantic_eligibility.json"
  "$FEEDBACK_ROOT/05_7D-gxaie6UI_candidate_284d53fde9070f949063/semantic_eligibility.json"
)

test -f "$PARENT_META_PROMPT_ARTIFACT"
test ! -e "$OUTPUT_DIRECTORY"

ELIGIBLE_FEEDBACK_ARTIFACTS=()
INVALID_FEEDBACK_REPORTS=()
for index in "${!FEEDBACK_ARTIFACTS[@]}"; do
  if [[ ! -f "${FEEDBACK_ARTIFACTS[$index]}" ]]; then
    INVALID_FEEDBACK_REPORTS+=(
      "${FEEDBACK_ARTIFACTS[$index]}: missing parsed_feedback.json")
    continue
  fi
  if [[ ! -f "${MANIFESTS[$index]}" ]]; then
    INVALID_FEEDBACK_REPORTS+=(
      "${FEEDBACK_ARTIFACTS[$index]}: missing manifest.json")
    continue
  fi
  if [[ ! -f "${ELIGIBILITY_ARTIFACTS[$index]}" ]]; then
    INVALID_FEEDBACK_REPORTS+=(
      "${FEEDBACK_ARTIFACTS[$index]}: missing semantic_eligibility.json")
    continue
  fi
  if ! jq -e '
    .status == "success"
    and .provider_call_count == 1
    and .adapter_call_count == 1
    and .supporting_id_validation == "passed"
    and .semantic_eligibility == "passed"
    and .source_aggregate_sha256_before == .source_aggregate_sha256_after
  ' "${MANIFESTS[$index]}" >/dev/null; then
    manifest_reason="$(jq -c '{status, error, semantic_eligibility}' \
      "${MANIFESTS[$index]}")"
    INVALID_FEEDBACK_REPORTS+=(
      "${FEEDBACK_ARTIFACTS[$index]}: invalid run manifest $manifest_reason")
    continue
  fi
  if conda run -n local_llm_vllm python -c '
import json
import sys
from pathlib import Path
from surrogate_rollout.optimization.schemas import episode_feedback_from_json
from surrogate_rollout.schemas import sha256_json
from surrogate_rollout.prompt_routing.schemas import dumps_canonical

feedback = episode_feedback_from_json(
    json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert report.get("eligible") is True
assert report.get("reasons") == []
assert report.get("feedback_id") == feedback.feedback_id
assert report.get("episode_id") == feedback.episode_id
assert report.get("feedback_sha256") == sha256_json(
    json.loads(dumps_canonical(feedback)))
' "${FEEDBACK_ARTIFACTS[$index]}" "${ELIGIBILITY_ARTIFACTS[$index]}"; then
    ELIGIBLE_FEEDBACK_ARTIFACTS+=("${FEEDBACK_ARTIFACTS[$index]}")
  else
    reasons="$(jq -c '.reasons // ["eligibility artifact validation failed"]' \
      "${ELIGIBILITY_ARTIFACTS[$index]}")"
    INVALID_FEEDBACK_REPORTS+=(
      "${FEEDBACK_ARTIFACTS[$index]}: $reasons")
  fi
done

if ((${#INVALID_FEEDBACK_REPORTS[@]})); then
  echo "Ineligible feedback artifacts (excluded explicitly):" >&2
  printf '  %s\n' "${INVALID_FEEDBACK_REPORTS[@]}" >&2
fi
if ((${#ELIGIBLE_FEEDBACK_ARTIFACTS[@]} < 2)); then
  echo "fewer than two semantically eligible feedback artifacts; updater not called" >&2
  exit 1
fi

conda run -n local_llm_vllm python -c '
import json
import sys
from pathlib import Path
from surrogate_rollout.optimization.schemas import (
    episode_feedback_from_json,
    meta_prompt_version_from_json,
)

parent = meta_prompt_version_from_json(
    json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
feedbacks = [
    episode_feedback_from_json(
        json.loads(Path(value).read_text(encoding="utf-8")))
    for value in sys.argv[2:]
]
assert parent.status == "parent"
assert len(feedbacks) >= 2
assert len({item.feedback_id for item in feedbacks}) == len(feedbacks)
' "$PARENT_META_PROMPT_ARTIFACT" "${ELIGIBLE_FEEDBACK_ARTIFACTS[@]}"

FEEDBACK_ARGS=()
for feedback_artifact in "${ELIGIBLE_FEEDBACK_ARTIFACTS[@]}"; do
  FEEDBACK_ARGS+=(--feedback-artifact "$feedback_artifact")
done

conda run --no-capture-output -n local_llm_vllm \
  python "$PROJECT_ROOT/scripts/run_meta_prompt_update_once.py" \
  --provider openai_api \
  --api-endpoint https://api.openai.com/v1/chat/completions \
  --model-id gpt-4o \
  --parent-meta-prompt "$PARENT_META_PROMPT_ARTIFACT" \
  "${FEEDBACK_ARGS[@]}" \
  --updater-policy-version meta_prompt_updater_e2_grounded_multi_episode_v2 \
  --temperature 0.0 \
  --maximum-output-tokens 4096 \
  --candidate-created-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --output-dir "$OUTPUT_DIRECTORY" \
  --timeout-seconds 600

jq -e '
  .status == "succeeded"
  and .provider_call_count == 1
  and .source_artifacts == .source_artifacts_after
' "$OUTPUT_DIRECTORY/run_manifest.json" >/dev/null

decision="$(jq -r '.decision' "$OUTPUT_DIRECTORY/run_manifest.json")"
case "$decision" in
  update)
    test -f "$OUTPUT_DIRECTORY/provisional_meta_prompt.json"
    test ! -e "$OUTPUT_DIRECTORY/no_update.json"
    ;;
  no_update)
    test -f "$OUTPUT_DIRECTORY/no_update.json"
    test ! -e "$OUTPUT_DIRECTORY/provisional_meta_prompt.json"
    ;;
  *)
    echo "unexpected updater decision: $decision" >&2
    exit 1
    ;;
esac

printf 'META_PROMPT_UPDATE_OUTPUT=%s\n' "$OUTPUT_DIRECTORY"
