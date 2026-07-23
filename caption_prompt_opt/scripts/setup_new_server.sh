#!/usr/bin/env bash
# Fresh-server bring-up for the caption-prompt path. Mirrors docs/PORTABLE_SETUP.md
# (conda env + pip install -r requirements.txt), then OVERRIDES the captioner
# stack because Qwen3.x-VL needs newer vllm/transformers than requirements.txt
# pins (vllm==0.11.2 / transformers==4.57.6 target Qwen2.5-VL).
#
# Version-sensitive step is parametrized: set CAPTION_STACK_SPEC / VLLM_PIP_EXTRA
# to the exact combo you verified for your model. Defaults just upgrade to latest.
#
# Does NOT sync the Video-MME media (that comes from a machine that already has
# it) and does NOT create GPUs. See the printed next-steps.
set -euo pipefail

CPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${SR_PROJECT_ROOT:-}" ]]; then
  SR_ROOT="$SR_PROJECT_ROOT"
elif [[ -f "$(dirname "$CPO_ROOT")/requirements.txt" ]]; then
  SR_ROOT="$(dirname "$CPO_ROOT")"                                   # nested
elif [[ -f "$(dirname "$CPO_ROOT")/surrogate_rollout/requirements.txt" ]]; then
  SR_ROOT="$(dirname "$CPO_ROOT")/surrogate_rollout"                 # sibling
else
  echo "cannot locate surrogate_rollout checkout; set SR_PROJECT_ROOT" >&2; exit 2
fi
REPO_PARENT="$(dirname "$SR_ROOT")"

: "${SR_CONDA_ENV:=local_llm_vllm}"
: "${SR_CAPTION_MODEL_ID:?export SR_CAPTION_MODEL_ID=<Qwen3.x-VL repo id, e.g. Qwen/Qwen3.5-9B>}"

# The checkout dir MUST be named surrogate_rollout (package imports itself as
# surrogate_rollout.*), per docs/PORTABLE_SETUP.md.
if [[ "$(basename "$SR_ROOT")" != "surrogate_rollout" ]]; then
  echo "checkout must be named 'surrogate_rollout' (is '$(basename "$SR_ROOT")')" >&2; exit 2
fi

echo "== 1. conda env ($SR_CONDA_ENV) =="
if conda env list | awk '{print $1}' | grep -qx "$SR_CONDA_ENV"; then
  echo "env exists, reusing"
else
  conda create -n "$SR_CONDA_ENV" python=3.11 -y
fi

echo "== 2. base deps (pinned) =="
conda run --no-capture-output -n "$SR_CONDA_ENV" pip install -r "$SR_ROOT/requirements.txt"

echo "== 3. captioner stack override for Qwen3.x-VL (VERSION-SENSITIVE) =="
# requirements.txt pins vllm==0.11.2 (Qwen2.5-VL). Qwen3.x-VL needs newer. Set
# these to the combo you verified; defaults upgrade to latest / nightly wheels.
#   CAPTION_STACK_SPEC='vllm==<ver> transformers==<ver> qwen-vl-utils==<ver>'
#   VLLM_PIP_EXTRA='--pre --extra-index-url https://wheels.vllm.ai/nightly'
CAPTION_STACK_SPEC="${CAPTION_STACK_SPEC:-vllm transformers qwen-vl-utils}"
VLLM_PIP_EXTRA="${VLLM_PIP_EXTRA:-}"
echo "   installing: -U ${VLLM_PIP_EXTRA} ${CAPTION_STACK_SPEC}"
# shellcheck disable=SC2086
conda run --no-capture-output -n "$SR_CONDA_ENV" pip install -U ${VLLM_PIP_EXTRA} ${CAPTION_STACK_SPEC}

echo "== 4. .env =="
if [[ -f "$SR_ROOT/.env" ]]; then
  echo ".env exists, leaving as is"
else
  : "${OPENAI_API_KEY:?no .env present; export OPENAI_API_KEY so one can be written}"
  printf 'OPENAI_API_KEY=%s\n' "$OPENAI_API_KEY" > "$SR_ROOT/.env"
  echo "wrote $SR_ROOT/.env"
fi

echo "== 5. captioner model download ($SR_CAPTION_MODEL_ID) =="
conda run --no-capture-output -n "$SR_CONDA_ENV" \
  huggingface-cli download "$SR_CAPTION_MODEL_ID"

echo "== 6. import check (no GPU, no provider calls) =="
PYTHONPATH="$REPO_PARENT:$SR_ROOT" conda run --no-capture-output -n "$SR_CONDA_ENV" \
  python -c "import caption_prompt_opt.factory as f; import surrogate_rollout; print('import OK:', callable(f.build_caption_prompt_components))"

cat <<EOF

== bring-up done. Remaining (data + run) ==
1. Sync Video-MME cohort media onto this box (~8.4 GB), from a machine that has it:
     export SR_VIDEOMME_DATA_ROOT=<dest>/videomme_data
     bash "$SR_ROOT/scripts/sync_cohort_data.sh" <dest>
   (sync_cohort_data.sh copies FROM its SR_VIDEOMME_DATA_ROOT source; on a brand
    new box, rsync the media from the source host first, then point at it.)
2. Verify the provider sees the media (from docs/PORTABLE_SETUP.md "provider check").
3. Run:
     export SR_CAPTION_MODEL_ID=$SR_CAPTION_MODEL_ID
     export CAPTION_PROMPT_WORKER_GPUS=0,1,2,3
     bash "$CPO_ROOT/scripts/go_caption_prompt.sh"

NOTE: step 3 (captioner stack) is the only part that differs from the pinned
setup. If vLLM/transformers you installed do not recognize the model arch, pin
CAPTION_STACK_SPEC / VLLM_PIP_EXTRA to the combo that works and re-run this.
EOF
