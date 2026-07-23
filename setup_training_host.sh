#!/usr/bin/env bash
# ============================================================================
# Training-host setup for the CAPO meta-prompt optimization loop.
#
# Brings up the 5-iteration prompt-delta run on a machine that is not the one
# the reference iterations ran on: different GPUs (Blackwell), a different
# caption model, a cold cache. Held-out measurement is NOT part of this host --
# the loop enqueues those requests and a separate measurement worker scores
# them, so only the evidence cohort's videos need to be present here.
#
# Usage (from inside this repository, on the training host):
#   bash setup_training_host.sh all        # every stage, stopping at manual gates
#   bash setup_training_host.sh <stage>    # one stage
#
# Stages: check env install models data creds smoke-vllm smoke-tests launch
#
# Automatable stages run themselves. Stages that need something only a human can
# supply (the video files, a funded API key) print exactly what is missing and
# stop. Idempotent: re-running skips work that is already done.
# ============================================================================
set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="${SR_CONDA_ENV:-capo}"
PYVER="${PYVER:-3.11}"
CAPTION_MODEL="${SR_CAPTION_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
EMBED_MODEL="BAAI/bge-small-en-v1.5"
GPUS="${PROMPT_DELTA_WORKER_GPUS:-0,1,2,3}"
VIDEOMME_DATA_ROOT="${SR_VIDEOMME_DATA_ROOT:-/hub_data3/videomme_data}"
EVIDENCE_COHORT="${PROMPT_DELTA_EVIDENCE_COHORT_FILE:-$REPO_DIR/train_set/20samples.txt}"
ENV_FILE="${SR_ENV_FILE:-$REPO_DIR/.env}"

say(){ printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok(){ printf '  \033[32mOK\033[0m %s\n' "$*"; }
warn(){ printf '  \033[33m!!\033[0m %s\n' "$*"; }
die(){ printf '  \033[31mXX %s\033[0m\n' "$*"; exit 1; }
py(){ conda run -n "$CONDA_ENV" python "$@"; }

stage_check(){
  say "check: GPUs / arch / conda"
  command -v nvidia-smi >/dev/null || die "no nvidia-smi"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  local cc count
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  echo "  compute capability: $cc  (Blackwell = 12.0; needs driver 570+ and an sm_120 vLLM)"
  echo "  GPUs visible: $count  (PROMPT_DELTA_WORKER_GPUS=$GPUS)"
  local want; want=$(awk -F, '{print NF}' <<<"$GPUS")
  [ "$count" -ge "$want" ] || warn "fewer GPUs than PROMPT_DELTA_WORKER_GPUS asks for"
  command -v conda >/dev/null || die "no conda (install miniconda first)"
  command -v jq >/dev/null || warn "no jq -- the run scripts require it (apt install jq / conda install jq)"
  ok "system ready"
}

stage_env(){
  say "env: conda '$CONDA_ENV' + pinned stack"
  conda env list | grep -qE "^\s*$CONDA_ENV\s" || conda create -y -n "$CONDA_ENV" "python=$PYVER"
  # requirements.txt pins the versions the reference runs used. Blackwell may
  # need a newer vllm/torch than the pin; if the vLLM gate below fails, relax
  # those two lines rather than the rest of the file.
  conda run -n "$CONDA_ENV" pip install -r "$REPO_DIR/requirements.txt" || \
    warn "pinned install failed -- on Blackwell try: pip install -U vllm torch, then re-run"
  conda run -n "$CONDA_ENV" pip install huggingface_hub
  py -c "import vllm,torch;print('  vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda)"
  ok "env ready"
}

stage_install(){
  say "install: this repository as the 'surrogate_rollout' package"
  # Every module imports the absolute name `surrogate_rollout`. Installing is
  # what supplies that name -- without it the checkout directory would have to
  # be called `surrogate_rollout` for any entry point to import.
  conda run -n "$CONDA_ENV" pip install -e "$REPO_DIR" || die "editable install failed"
  (cd / && py -c "import surrogate_rollout, surrogate_rollout.prompt_routing.schemas; print('  import OK from', '$REPO_DIR')") \
    || die "package import failed"
  ok "installed (checkout directory name no longer matters)"
}

stage_models(){
  say "models: $CAPTION_MODEL + $EMBED_MODEL (HF download)"
  py -c "from huggingface_hub import snapshot_download as d; d('$CAPTION_MODEL'); d('$EMBED_MODEL'); print('  downloaded')" \
    || die "model download failed (HF token / internet?)"
  ok "models cached"
}

stage_data(){
  say "data: Video-MME long split, evidence cohort only"
  test -f "$EVIDENCE_COHORT" || die "no evidence cohort file at $EVIDENCE_COHORT"
  local repo="$VIDEOMME_DATA_ROOT/Video-MME"
  local parquet="$repo/videomme/test-00000-of-00001.parquet"
  local videos="$repo/videos/long"
  local total missing=0 id
  total=$(grep -c . "$EVIDENCE_COHORT")
  echo "  data root : $VIDEOMME_DATA_ROOT  (override with SR_VIDEOMME_DATA_ROOT)"
  echo "  cohort    : $EVIDENCE_COHORT ($total videos)"
  if [ -f "$parquet" ]; then ok "QA parquet present"; else warn "missing $parquet"; missing=1; fi
  while read -r id; do
    [ -n "$id" ] || continue
    [ -f "$videos/$id.mp4" ] || { warn "missing video: $videos/$id.mp4"; missing=1; }
  done < "$EVIDENCE_COHORT"
  if [ "$missing" -ne 0 ]; then
    echo "  -> copy the missing files from the source host, e.g."
    echo "     rsync -av --files-from=<(sed 's#^#Video-MME/videos/long/#;s#\$#.mp4#' $EVIDENCE_COHORT) \\"
    echo "           SRC:$VIDEOMME_DATA_ROOT/ $VIDEOMME_DATA_ROOT/"
    echo "     plus Video-MME/videomme/test-00000-of-00001.parquet"
    echo "  Frames decode on first use; do not transfer frame caches."
    echo "  The confirmation cohort's videos are NOT needed here -- the"
    echo "  measurement worker scores those, wherever it runs."
    return 1
  fi
  ok "every evidence video present"
}

stage_creds(){
  say "creds: OPENAI_API_KEY (DVD orchestrator, generator, optimizer)"
  if [ -f "$ENV_FILE" ] && grep -q OPENAI_API_KEY "$ENV_FILE"; then ok "$ENV_FILE"
  else
    warn "no OPENAI_API_KEY"
    echo "  -> echo 'OPENAI_API_KEY=sk-...' > $ENV_FILE   (funded key)"
    echo "     This run makes paid calls on every iteration: gpt-5-mini for the"
    echo "     DVD orchestrator, the prompt generator, and the optimizer."
    return 1
  fi
}

stage_smoke_vllm(){
  say "GATE: vLLM can load $CAPTION_MODEL on this GPU"
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" py - <<PY || die "vLLM failed to load the caption model -- bump vllm/torch, then re-run 'env' and this stage"
from vllm import LLM, SamplingParams
llm = LLM(model="$CAPTION_MODEL", dtype="bfloat16", max_model_len=8192,
          limit_mm_per_prompt={"image": 1}, enforce_eager=True,
          gpu_memory_utilization=0.85)
out = llm.generate(["hi"], SamplingParams(max_tokens=4))
print("  generated:", repr(out[0].outputs[0].text))
PY
  ok "caption model loads on this hardware"
}

stage_smoke_tests(){
  say "smoke: unit tests (no GPU, no paid calls)"
  conda run -n "$CONDA_ENV" pip install -q pytest
  conda run --no-capture-output -n "$CONDA_ENV" python -m pytest -q "$REPO_DIR/tests" \
    || die "unit tests failed -- fix before spending money on a run"
  ok "tests pass"
}

stage_launch(){
  say "launch: the 5-iteration run"
  cat <<EOF
  set -a
  source scripts/env/gpt5mini_stack.sh
  source scripts/env/training_host.sh
  set +a
  bash scripts/run_prompt_delta_two_iteration_10video_pool.sh

  Resume: re-run the same command with PROMPT_DELTA_ITERATION_TIMESTAMP set to
  the timestamp of the run directory under runs/. Completed iterations are
  skipped; a half-written one resumes from its own artifacts.

  Held-out scoring (separate process, may be another machine):
  python scripts/run_measurement_worker.py --queue-dir runs/measurement_queue
EOF
}

case "${1:-all}" in
  check) stage_check;; env) stage_env;; install) stage_install;; models) stage_models;;
  data) stage_data;; creds) stage_creds;; smoke-vllm) stage_smoke_vllm;;
  smoke-tests) stage_smoke_tests;; launch) stage_launch;;
  all)
    stage_check && stage_env && stage_install && stage_models
    stage_data  || warn "provide the videos, then: bash setup_training_host.sh data"
    stage_creds || warn "provide the key, then: bash setup_training_host.sh creds"
    stage_smoke_vllm
    stage_smoke_tests
    stage_launch
    ;;
  *) die "unknown stage '$1' (check|env|install|models|data|creds|smoke-vllm|smoke-tests|launch|all)";;
esac
