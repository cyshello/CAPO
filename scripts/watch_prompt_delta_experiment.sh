#!/usr/bin/env bash
# Keep a prompt-delta multi-iteration experiment alive across crashes.
#
# Usage: watch_prompt_delta_experiment.sh EXPERIMENT_TIMESTAMP LOG_PATH
#
# Every stage of an iteration is write-once and skips when its artifact already
# exists, so restarting the driver with the same PROMPT_DELTA_ITERATION_TIMESTAMP
# resumes instead of recomputing. This watcher does exactly that: it waits for
# the driver to disappear, and if the experiment manifest is not there yet it
# starts the driver again on the same timestamp.
#
# A deterministic bug will fail at the same point every time, so restarts are
# capped and each one is announced on Telegram.
#
# A dead driver can leave vLLM engine cores holding the worker GPUs. Relaunching
# into that state fails with "Free memory on device ... is less than desired GPU
# memory utilization", which burns another restart and leaves the next engine
# cores behind in turn. So the watcher drains the worker GPUs before it relaunches.
set -u

EXPERIMENT_TIMESTAMP="${1:?EXPERIMENT_TIMESTAMP required}"
LOG="${2:?LOG_PATH required}"
MAX_RESTARTS="${PROMPT_DELTA_MAX_RESTARTS:-20}"
BACKOFF_SECONDS="${PROMPT_DELTA_RESTART_BACKOFF_SECONDS:-180}"
GPU_DRAIN_SECONDS="${PROMPT_DELTA_GPU_DRAIN_SECONDS:-300}"

PROJECT_ROOT="${SR_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LABEL="${PROMPT_DELTA_EXPERIMENT_LABEL:-20video_5iter_val15}"
WORKER_GPUS="${PROMPT_DELTA_WORKER_GPUS:-4,5,6,7}"
EXPERIMENT_ROOT="$PROJECT_ROOT/runs/prompt_delta_${LABEL}_${EXPERIMENT_TIMESTAMP}"
NOTIFY="$PROJECT_ROOT/scripts/notify_prompt_delta_experiment.sh"

send() {
    local text="$1"
    local token chat
    token="$(grep -E '^TELEGRAM_BOT_TOKEN=' "$PROJECT_ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2-)"
    chat="$(grep -E '^TELEGRAM_CHAT_ID=' "$PROJECT_ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2-)"
    [ -n "${token:-}" ] && [ -n "${chat:-}" ] || return 0
    curl -s -m 20 -X POST "https://api.telegram.org/bot${token}/sendMessage" \
        --data-urlencode "chat_id=${chat}" --data-urlencode "text=${text}" \
        >/dev/null 2>&1
}

# UUIDs of the GPUs this experiment owns, so nothing outside them is inspected.
worker_gpu_uuids() {
    local want=",${WORKER_GPUS},"
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null |
        tr -d ' ' |
        while IFS=, read -r index uuid; do
            case "$want" in *,"$index",*) printf '%s\n' "$uuid" ;; esac
        done
}

# PIDs still holding those GPUs, restricted to this user's own captioning stack.
# Called only after the driver is gone, so a survivor here is an orphan of it.
worker_gpu_orphan_pids() {
    local uuids pid uuid command
    uuids="$(worker_gpu_uuids)"
    [ -n "$uuids" ] || return 0
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null |
        tr -d ' ' |
        while IFS=, read -r uuid pid; do
            printf '%s\n' "$uuids" | grep -qxF "$uuid" || continue
            [ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" = "$(id -un)" ] || continue
            command="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
            case "$command" in
                *VLLM*|*vllm*|*surrogate_rollout*|*multiprocessing-fork*|*spawn_main*)
                    printf '%s\n' "$pid" ;;
            esac
        done
}

drain_worker_gpus() {
    local waited=0 pids
    while [ "$waited" -lt "$GPU_DRAIN_SECONDS" ]; do
        pids="$(worker_gpu_orphan_pids | sort -u | tr '\n' ' ')"
        pids="${pids% }"
        [ -z "$pids" ] && return 0
        sleep 15
        waited=$((waited + 15))
    done
    pids="$(worker_gpu_orphan_pids | sort -u | tr '\n' ' ')"
    pids="${pids% }"
    [ -z "$pids" ] && return 0
    echo "watch: terminating orphan GPU processes on $WORKER_GPUS: $pids"
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
    sleep 30
    pids="$(worker_gpu_orphan_pids | sort -u | tr '\n' ' ')"
    pids="${pids% }"
    if [ -n "$pids" ]; then
        echo "watch: killing orphan GPU processes on $WORKER_GPUS: $pids"
        # shellcheck disable=SC2086
        kill -KILL $pids 2>/dev/null || true
        sleep 15
    fi
}

restarts=0
while true; do
    while pgrep -f run_prompt_delta_two_iteration >/dev/null 2>&1; do
        sleep 60
    done

    if [ -f "$EXPERIMENT_ROOT/experiment_manifest.json" ]; then
        echo "watch: experiment complete"
        bash "$NOTIFY" "$LABEL" "$LOG" 5 >/dev/null 2>&1
        exit 0
    fi

    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
        send "[$LABEL] ⛔ giving up after ${MAX_RESTARTS} restarts.
$(grep -aoE '(FreshPromptDeltaError|PromptDelta[A-Za-z]*Error|RuntimeError|ValueError|KeyError|TypeError): .*' "$LOG" 2>/dev/null | tail -1 | cut -c1-200)
$(date '+%m-%d %H:%M')"
        echo "watch: giving up after $MAX_RESTARTS restarts" >&2
        exit 1
    fi

    failure="$(grep -aoE '(FreshPromptDeltaError|PromptDelta[A-Za-z]*Error|RuntimeError|ValueError|KeyError|TypeError): .*' "$LOG" 2>/dev/null | tail -1 | cut -c1-200)"
    send "[$LABEL] 🔁 driver stopped, resuming (restart ${restarts}/${MAX_RESTARTS})
${failure:-no error line found}
$(date '+%m-%d %H:%M')"

    drain_worker_gpus
    sleep "$BACKOFF_SECONDS"
    env PROMPT_DELTA_ITERATION_TIMESTAMP="$EXPERIMENT_TIMESTAMP" \
        PROMPT_DELTA_ITERATION_COUNT="${PROMPT_DELTA_ITERATION_COUNT:-5}" \
        PROMPT_DELTA_VIDEOS_PER_ITERATION="${PROMPT_DELTA_VIDEOS_PER_ITERATION:-4}" \
        PROMPT_DELTA_WORKER_GPUS="$WORKER_GPUS" \
        PROMPT_DELTA_EVIDENCE_COHORT_FILE="${PROMPT_DELTA_EVIDENCE_COHORT_FILE:-$PROJECT_ROOT/train_set/20samples.txt}" \
        PROMPT_DELTA_CONFIRMATION_COHORT_FILE="${PROMPT_DELTA_CONFIRMATION_COHORT_FILE:-$PROJECT_ROOT/train_set/confirmation.txt}" \
        PROMPT_DELTA_EXPERIMENT_LABEL="$LABEL" \
        nohup bash "$PROJECT_ROOT/scripts/run_prompt_delta_two_iteration_10video_pool.sh" \
        >> "$LOG" 2>&1 &
    sleep 30
done
