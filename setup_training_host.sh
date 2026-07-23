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
#   SR_CAPTION_MODEL_ID=<hf-model> bash setup_training_host.sh go
#         set up everything and start the run; stops before spending anything
#         if the videos or the API key are missing
#   bash setup_training_host.sh all        # set up only, never launches
#   bash setup_training_host.sh <stage>    # one stage
#
# Stages: check env install models data creds smoke-vllm smoke-tests launch go
#
# Anything already exported wins over the profiles in scripts/env, so the
# caption model, the GPU list, and the schedule are all overridable from the
# command line without editing a file.
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

  Or let this script do it: bash setup_training_host.sh go

  Resume: re-run with PROMPT_DELTA_ITERATION_TIMESTAMP set to the timestamp of
  the run directory under runs/. Completed iterations are skipped; a half-written
  one resumes from its own artifacts.

  Held-out scoring (separate process, may be another machine):
  python scripts/run_measurement_worker.py --queue-dir runs/measurement_queue \\
    --component-config <evidence-run>/resolved_component_config.json
EOF
}

# The whole thing: every setup stage, then the run. The gates are hard here --
# a missing video or a missing key stops it before anything is spent, because
# the alternative is discovering it an hour into a paid run.
stage_go(){
  stage_check    || die "environment not ready"
  stage_env      || die "dependency install failed"
  stage_install  || exit 1
  stage_models   || exit 1
  stage_data     || die "evidence videos missing (see above); nothing was started"
  stage_creds    || die "no API key (see above); nothing was started"
  stage_smoke_vllm
  stage_smoke_tests

  say "run: ${PROMPT_DELTA_ITERATION_COUNT:-5} iterations, captioner $CAPTION_MODEL, GPUs $GPUS"
  cd "$REPO_DIR" || die "cannot enter $REPO_DIR"
  # The profiles only fill in what the environment has not already set, so a
  # variable exported before this script wins over both of them.
  set -a
  # shellcheck source=scripts/env/gpt5mini_stack.sh
  source "$REPO_DIR/scripts/env/gpt5mini_stack.sh"
  # shellcheck source=scripts/env/training_host.sh
  source "$REPO_DIR/scripts/env/training_host.sh"
  set +a

  if [ -n "${CAPO_FOREGROUND:-}" ]; then
    warn "foreground: no watcher, so a crash ends the run until you restart it"
    exec bash "$REPO_DIR/scripts/run_prompt_delta_two_iteration_10video_pool.sh"
  fi

  # Every stage is write-once and skips when its artifact exists, so a crashed
  # driver relaunched on the same timestamp resumes rather than recomputes. That
  # is what makes an unattended restart safe -- and necessary: a vLLM engine can
  # die mid-iteration, and without something to relaunch the driver the run just
  # stops until a person notices. The watcher does that, drains any vLLM orphan
  # still holding the worker GPUs first, and gives up after MAX_RESTARTS so a
  # deterministic failure cannot spin forever.
  export PROMPT_DELTA_ITERATION_TIMESTAMP="${PROMPT_DELTA_ITERATION_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
  local ts="$PROMPT_DELTA_ITERATION_TIMESTAMP"
  mkdir -p "$REPO_DIR/runs"
  local log="$REPO_DIR/runs/experiment_${ts}.log"
  local watchlog="$REPO_DIR/runs/watch_${ts}.log"

  # The driver starts first. The watcher's loop waits for a driver to disappear
  # before doing anything, so starting it alone would spend a restart and a
  # backoff before the first iteration ever began.
  setsid nohup bash "$REPO_DIR/scripts/run_prompt_delta_two_iteration_10video_pool.sh" \
    >> "$log" 2>&1 < /dev/null &
  local pid=$!
  sleep 5
  if ! kill -0 "$pid" 2>/dev/null; then
    warn "the run exited within five seconds -- last lines:"
    tail -20 "$log"
    return 1
  fi
  # The watcher relaunches the driver with its own environment, so it has to
  # inherit the profiles sourced above; that is why it is started from here and
  # not from a bare shell.
  setsid nohup bash "$REPO_DIR/scripts/watch_prompt_delta_experiment.sh" "$ts" "$log" \
    >> "$watchlog" 2>&1 < /dev/null &
  ok "running: driver pid $pid, watcher pid $!, timestamp $ts"
  echo "  log:     tail -f $log"
  echo "  watcher: tail -f $watchlog   (restarts, GPU drains, give-up)"
  echo "  restarts allowed: ${PROMPT_DELTA_MAX_RESTARTS:-20}"
  echo "  stop:    pkill -f watch_prompt_delta_experiment && kill $pid"
  echo "  resume after a full stop: re-run with PROMPT_DELTA_ITERATION_TIMESTAMP=$ts"
  echo "  (CAPO_FOREGROUND=1 runs the driver in this shell, unwatched)"
  if ! grep -q TELEGRAM_BOT_TOKEN "$ENV_FILE" 2>/dev/null; then
    echo "  no TELEGRAM_BOT_TOKEN in $ENV_FILE -- restart alerts stay silent, which"
    echo "  is fine; the watcher log records them either way."
  fi
}

case "${1:-all}" in
  check) stage_check;; env) stage_env;; install) stage_install;; models) stage_models;;
  data) stage_data;; creds) stage_creds;; smoke-vllm) stage_smoke_vllm;;
  smoke-tests) stage_smoke_tests;; launch) stage_launch;; go) stage_go;;
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
