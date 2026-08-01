#!/usr/bin/env bash
# Move the full-recaption run from 4 GPUs (5,7,4,6) to 7 (0,1,2,4,5,6,7).
#
# Replaces logs/expand_full_recaption_to_7gpu.sh, whose LVBench wait never
# fired: `pgrep -fc PAT || echo 0` emits "0\n0" when nothing matches (pgrep -fc
# prints 0 AND exits 1), so `[[ "$alive" -eq 0 ]]` was a syntax error and always
# false. LVBench is already finished (last shard 2026-07-26 06:11, 1535 QA --
# 1535 not 1549 because TiQBTesZUJQ's view failed), so this script does not wait
# for anything; it switches immediately.
#
# GPU set is 0,1,2,4,5,6,7 per the user's instruction on 2026-07-26. GPU 3 is
# left out.
#
# Not a live resize: the worker GPU list is part of the baseline input
# fingerprint, so the driver must stop, the in-flight iteration's evidence root
# must be set aside, and the driver relaunched on the new list. Completed
# iterations are read and skipped by the pool driver; the in-flight iteration
# replays from the caption and generator-response caches, whose keys carry no
# GPU.
set -uo pipefail

PROJECT_ROOT=/home/seungmin/youngseo/surrogate_rollout
LAUNCHER="$PROJECT_ROOT/logs/resume_full_recaption_gpu57_20260725_080528.sh"
LOG="$PROJECT_ROOT/logs/switch_to_7gpu.log"
NEW_GPUS="0,1,2,4,5,6,7"

say() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

driver_pgid() {
  ps -eo pgid,cmd | grep '[r]un_full_recaption_evidence' | awk '{print $1}' | sort -u | head -1
}

say "switch to $NEW_GPUS starting"

# 1. The added GPUs must be genuinely free -- other users grab freed cards fast.
for g in 0 1 2; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
  if [[ -z "$used" ]]; then say "GPU $g unreadable; aborting"; exit 1; fi
  if (( used > 1000 )); then
    say "GPU $g has ${used} MiB in use (someone else took it); aborting without touching anything"
    exit 1
  fi
  say "GPU $g free (${used} MiB)"
done

PGID="$(driver_pgid)"
if [[ -z "$PGID" ]]; then say "driver not running; nothing to stop"; else
  say "stopping driver pgid $PGID"
  kill -TERM -"$PGID"
  sleep 25
  kill -KILL -"$PGID" 2>/dev/null
  sleep 6
  if [[ -n "$(driver_pgid)" ]]; then
    say "driver did not stop; aborting so nothing is half-switched"
    exit 1
  fi
  say "driver stopped"
fi

# 2. Quarantine the in-flight iteration: the newest evidence root with no
#    iteration_result.json. Its baseline artifacts are fingerprinted to the old
#    GPU list and would raise "completed video has stale inputs" on resume.
INFLIGHT=""
for root in $(ls -dt "$PROJECT_ROOT"/runs/fresh_prompt_delta_iteration_*_evidence 2>/dev/null); do
  ts="$(basename "$root" | sed -E 's/fresh_prompt_delta_iteration_(.*)_evidence/\1/')"
  if [[ ! -f "$PROJECT_ROOT/runs/fresh_prompt_delta_iteration_${ts}_output/iteration_result.json" ]]; then
    INFLIGHT="$root"; break
  fi
done
if [[ -n "$INFLIGHT" ]]; then
  say "in-flight: $(basename "$INFLIGHT") ($(find "$INFLIGHT" -name video_complete.json | wc -l) baselines complete, $(find "$INFLIGHT" -path '*interventions*' -name '*.json' | wc -l) intervention files)"
  QUARANTINE="${INFLIGHT}.gpu4_$(date +%Y%m%d_%H%M%S)"
  mv "$INFLIGHT" "$QUARANTINE" && say "quarantined -> $(basename "$QUARANTINE")"
else
  say "no in-flight evidence root found (nothing to quarantine)"
fi

# 3. Repoint the launcher.
sed -i -E "s|^export PROMPT_DELTA_WORKER_GPUS=.*$|export PROMPT_DELTA_WORKER_GPUS=$NEW_GPUS|" "$LAUNCHER"
say "launcher now: $(grep -E '^export PROMPT_DELTA_WORKER_GPUS=' "$LAUNCHER")"

# 4. Relaunch detached, so it survives this shell.
RUNLOG="$PROJECT_ROOT/logs/full_recaption_gpu7_$(date -u +%Y%m%d_%H%M%S).log"
setsid nohup bash "$LAUNCHER" > "$RUNLOG" 2>&1 < /dev/null &
say "relaunched on $NEW_GPUS; log $RUNLOG"

# 5. Verify it actually came up.
sleep 180
say "driver pgid after restart: $(driver_pgid || echo NONE)"
say "GPU memory: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
say "errors in new log: $(grep -icE 'Traceback|stale inputs|conflict' "$RUNLOG")"
say "switch done"
