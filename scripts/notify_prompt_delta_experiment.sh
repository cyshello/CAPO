#!/usr/bin/env bash
# Prompt-delta multi-iteration experiment status -> Telegram.
#
# Usage: notify_prompt_delta_experiment.sh EXPERIMENT_LABEL LOG_PATH [ITERATION_COUNT]
#
# Composes the message from run artifacts rather than the log: under background
# `conda run` the redirected stdout stays buffered for long stretches even while
# the run writes per-segment sidecars every few seconds.
#
# Reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from surrogate_rollout/.env.
# Never echoes secrets; always exits 0 so a scheduling loop survives a bad tick.
set -u

LABEL="${1:?EXPERIMENT_LABEL required}"
LOG="${2:?LOG_PATH required}"
TOTAL_ITERATIONS="${3:-}"

PROJECT_ROOT="${SR_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="$PROJECT_ROOT/.env"
TOKEN="$(grep -E '^TELEGRAM_BOT_TOKEN=' "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2-)"
CHAT="$(grep -E '^TELEGRAM_CHAT_ID=' "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2-)"
if [ -z "${TOKEN:-}" ] || [ -z "${CHAT:-}" ]; then
    echo "notify: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing in $ENV_FILE" >&2
    exit 0
fi

now="$(date '+%m-%d %H:%M')"
runs="$PROJECT_ROOT/runs"

# This experiment's timestamp, from the roots the driver names after it. The
# driver derives each iteration's timestamp from that same base epoch, so
# iterations belonging to this experiment are exactly those at or after it.
# Without this bound the earlier standalone runs in runs/ get counted too.
experiment_stamp=""
for root in $(ls -dt "$runs"/prompt_delta_"$LABEL"_*_state \
                     "$runs"/prompt_delta_"$LABEL"_*_cache \
                     "$runs"/prompt_delta_"$LABEL"_*_feedback_memory 2>/dev/null); do
    base="$(basename "$root")"
    base="${base#prompt_delta_${LABEL}_}"
    experiment_stamp="${base%_*}"
    break
done

# ----- finished experiment -------------------------------------------------- #
experiment_root="$(ls -dt "$runs"/prompt_delta_"$LABEL"_* 2>/dev/null \
    | grep -vE '_(state|cache|feedback_memory)$' | head -1)"
if [ -n "$experiment_root" ] && [ -f "$experiment_root/experiment_manifest.json" ]; then
    lines="$(jq -r '.iterations[] | "  " + .iteration_id[-24:] + " " + .status
        + (if .active_accuracy != null
           then " acc=" + (.active_accuracy * 1000 | round / 1000 | tostring)
           else "" end)' \
        "$experiment_root/heldout_accuracy.json" 2>/dev/null)"
    msg="[$LABEL] ✅ DONE
$lines
$experiment_root
$now"
    curl -s -m 20 -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${CHAT}" --data-urlencode "text=${msg}" \
        >/dev/null 2>&1
    exit 0
fi

# ----- completed iterations so far ------------------------------------------ #
completed=""
completed_count=0
for result in $(ls -tr "$runs"/fresh_prompt_delta_iteration_*_output/iteration_result.json 2>/dev/null); do
    stamp="$(basename "$(dirname "$result")")"
    stamp="${stamp#fresh_prompt_delta_iteration_}"; stamp="${stamp%_output}"
    if [ -n "$experiment_stamp" ] && [ "$stamp" \< "$experiment_stamp" ]; then
        continue
    fi
    status="$(jq -r '.status // "?"' "$result" 2>/dev/null)"
    # always_promote_measured_v1 records the held-out accuracy of the active
    # prompt on the iteration result itself; the paired path leaves a
    # confirmation manifest instead.
    acc="$(jq -r 'if .held_out_accuracy != null
        then "acc=" + (.held_out_accuracy * 1000 | round / 1000 | tostring)
             + " (" + (.held_out_evaluated_count|tostring) + "/"
             + (.held_out_case_count|tostring) + ")"
        else "" end' "$result" 2>/dev/null)"
    if [ -z "$acc" ]; then
        conf="$(dirname "$result")/confirmation/dvd_confirmation_manifest.json"
        [ -f "$conf" ] && acc="$(jq -r '"parent=" + (.aggregate.parent_accuracy|tostring)
            + " cand=" + (.aggregate.candidate_accuracy|tostring)' "$conf" 2>/dev/null)"
    fi
    completed="${completed}  ${stamp} ${status}${acc:+ | $acc}"$'\n'
    completed_count=$((completed_count + 1))
done

# iteration 0: the starting meta-prompt, measured before any update
initial_line=""
for summary in $(ls -tr "$runs"/fresh_prompt_delta_iteration_*_output/parent_measurement/measurement_summary.json 2>/dev/null); do
    stamp="$(basename "$(dirname "$(dirname "$summary")")")"
    stamp="${stamp#fresh_prompt_delta_iteration_}"; stamp="${stamp%_output}"
    if [ -n "$experiment_stamp" ] && [ "$stamp" \< "$experiment_stamp" ]; then
        continue
    fi
    initial_line="  iter0 (initial) | acc=$(jq -r '.accuracy * 1000 | round / 1000' "$summary" 2>/dev/null)"
    break
done

# ----- the iteration currently running -------------------------------------- #
current_root=""
for candidate in $(ls -dt "$runs"/fresh_prompt_delta_iteration_*_evidence 2>/dev/null); do
    candidate_stamp="$(basename "$candidate")"
    candidate_stamp="${candidate_stamp#fresh_prompt_delta_iteration_}"
    candidate_stamp="${candidate_stamp%_evidence}"
    if [ -n "$experiment_stamp" ] && [ "$candidate_stamp" \< "$experiment_stamp" ]; then
        continue
    fi
    current_root="$candidate"
    break
done
stage="starting"; detail=""
if [ -n "$current_root" ]; then
    stamp="$(basename "$current_root")"
    stamp="${stamp#fresh_prompt_delta_iteration_}"; stamp="${stamp%_evidence}"
    out="$runs/fresh_prompt_delta_iteration_${stamp}_output"
    segments="$(find "$current_root" -path '*segment_state*' -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
    videos="$(ls -d "$current_root"/baseline/baseline/*/ 2>/dev/null | wc -l | tr -d ' ')"
    episodes="$(ls "$current_root"/episodes/*/*/*.json 2>/dev/null | wc -l | tr -d ' ')"
    deltas="$(jq -s 'map(.plans | length) | add // 0' \
        "$current_root"/proposals/*/*/proposal_plans.json 2>/dev/null)"
    if [ -d "$out/measurement" ]; then
        stage="held-out measurement"
    elif [ -d "$out/confirmation" ]; then
        stage="confirmation"
    elif [ -d "$out/parent_measurement" ] && [ ! -d "$out/feedback" ]; then
        stage="held-out measurement (initial prompt)"
    elif [ -d "$out/feedback" ]; then
        stage="feedback/updater"
        detail="feedback $(ls -d "$out"/feedback/*/ 2>/dev/null | wc -l | tr -d ' ')"
    elif [ "${episodes:-0}" -gt 0 ]; then
        stage="interventions"; detail="episodes ${episodes}"
    elif [ "${deltas:-0}" != "0" ] && [ -n "${deltas:-}" ]; then
        stage="interventions"; detail="deltas ${deltas}"
    elif [ -d "$current_root/proposals" ]; then
        stage="proposals"
    else
        stage="baseline captions"; detail="${segments} segments, ${videos} videos"
    fi
    detail="iter ${stamp} | ${detail}"
fi

# freshness from the newest artifact, not the buffered log
newest="$(find "$runs" -newer "$runs" -maxdepth 4 -name '*.json*' -printf '%T@\n' 2>/dev/null | sort -n | tail -1)"
newest="${newest%.*}"
[ -f "$LOG" ] && { lts="$(stat -c %Y "$LOG" 2>/dev/null)"; [ -n "$lts" ] && [ "${newest:-0}" -lt "$lts" ] && newest="$lts"; }
age_min="?"
[ -n "${newest:-}" ] && [ "${newest:-0}" -gt 0 ] && age_min=$(( ($(date +%s) - newest) / 60 ))

if pgrep -f run_prompt_delta_two_iteration >/dev/null 2>&1; then
    if [ "$age_min" != "?" ] && [ "$age_min" -ge 45 ]; then
        status="⚠️ RUNNING but idle ${age_min}m"
    else
        status="🟢 RUNNING"
    fi
else
    status="❌ NOT RUNNING"
fi

failure="$(grep -aoE '(FreshPromptDeltaError|PromptDeltaIteration[A-Za-z]*Error|RuntimeError|ValueError): .*' \
    "$LOG" 2>/dev/null | tail -1 | cut -c1-160)"

msg="[$LABEL] ${status}
${completed_count}${TOTAL_ITERATIONS:+/${TOTAL_ITERATIONS}} iterations done | stage: ${stage}
${detail}
${initial_line:+
$initial_line}${completed:+
$completed}${failure:+
last error: $failure}
+${age_min}m | $now"

curl -s -m 20 -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${CHAT}" --data-urlencode "text=${msg}" \
    >/dev/null 2>&1 \
    && echo "notify: sent [$LABEL] ($status)" \
    || echo "notify: telegram send failed [$LABEL]" >&2
exit 0
