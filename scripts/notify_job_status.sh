#!/usr/bin/env bash
# Generic job health ping -> Telegram (shared by all monitored runs).
#
# Usage: notify_job_status.sh LABEL PROC_PATTERN LOG_PATH [DONE_FILE] [STALE_MIN]
#   LABEL         human name shown in the message
#   PROC_PATTERN  pgrep -f pattern that uniquely matches the run's process
#   LOG_PATH      log file to tail for the latest progress line + freshness
#   DONE_FILE     optional; if it exists the run is treated as finished
#   STALE_MIN     optional; log idle >= this many minutes => ⚠️ (default 30)
#
# Reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from surrogate_rollout/.env.
# Safe for cron: always exits 0 on a normal tick, never echoes secrets.
set -u

LABEL="${1:?LABEL required}"
PATTERN="${2:?PROC_PATTERN required}"
LOG="${3:?LOG_PATH required}"
DONE_FILE="${4:-}"
STALE_MIN="${5:-30}"
TOTAL="${6:-}"     # optional total unit count, e.g. 103, shown as done/TOTAL

ENV_FILE="${SR_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/.env"
TOKEN="$(grep -E '^TELEGRAM_BOT_TOKEN=' "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2-)"
CHAT="$(grep -E '^TELEGRAM_CHAT_ID=' "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2-)"
if [ -z "${TOKEN:-}" ] || [ -z "${CHAT:-}" ]; then
    echo "notify: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing in $ENV_FILE" >&2
    exit 0
fi

now="$(date '+%Y-%m-%d %H:%M:%S')"

# Progress + freshness come from ARTIFACTS, not the log: under background
# `conda run` the redirected stdout can stay buffered/empty even while the run
# writes per-video sidecars every few minutes. done = number of finished
# videos; freshness = newest file mtime under the output dir (updates each QA).
outdir=""; done_units=""; newest_ts=0
if [ -n "$DONE_FILE" ]; then
    outdir="$(dirname "$DONE_FILE")"
    done_units="$(ls "$outdir"/videos/*/video_eval.json 2>/dev/null | wc -l | tr -d ' ')"
    af="$(find "$outdir/videos" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)"
    af="${af%.*}"; [ -n "$af" ] && newest_ts="$af"
fi
[ -f "$LOG" ] && { lts="$(stat -c %Y "$LOG")"; [ "$lts" -gt "$newest_ts" ] && newest_ts="$lts"; }
now_s="$(date +%s)"
if [ "$newest_ts" -gt 0 ]; then age_min=$(( (now_s - newest_ts) / 60 )); else age_min="?"; fi
prog_str=""
[ -n "$done_units" ] && prog_str="videos ${done_units}${TOTAL:+/${TOTAL}} done"
# Last meaningful log line (drops vLLM / progress-bar noise). Useful for jobs
# that write a real progress.log; empty for jobs whose stdout stays buffered.
logline="$(grep -avE 'it/s|Processed prompts|Adding requests|^Batches|INFO:nano|Loading|Capturing|Warning|graph' "$LOG" 2>/dev/null | grep -v '^[[:space:]]*$' | tail -1 | tr -d '\r' | cut -c1-170)"

if [ -n "$DONE_FILE" ] && [ -f "$DONE_FILE" ]; then
    acc="$(grep -oE '"accuracy": [0-9.]+' "$DONE_FILE" 2>/dev/null | head -1 | awk '{print $2}')"
    nv="$(grep -oE '"num_videos": [0-9]+' "$DONE_FILE" 2>/dev/null | head -1 | awk '{print $2}')"
    status="✅ DONE"
    detail="${nv:+$nv videos | }accuracy=${acc:-?}"
elif pgrep -f "$PATTERN" >/dev/null 2>&1; then
    if [ "$age_min" != "?" ] && [ "$age_min" -ge "$STALE_MIN" ]; then
        status="⚠️ RUNNING but idle ${age_min}m"
    else
        status="🟢 RUNNING"
    fi
    main="$prog_str"
    [ -n "$logline" ] && main="${main:+$main | }$logline"
    detail="${main:-starting up} | +${age_min}m"
else
    status="❌ NOT RUNNING"
    detail="${prog_str:+$prog_str | }${logline:-$(tail -n 3 "$LOG" 2>/dev/null | tr '\n' ' ' | tail -c 200)}"
fi

msg="[${LABEL}] ${status}
${detail}
${now}"

curl -s -m 20 -X POST \
    "https://api.telegram.org/bot${TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${CHAT}" \
    --data-urlencode "text=${msg}" >/dev/null 2>&1 \
    && echo "notify: sent [${LABEL}] (${status})" \
    || echo "notify: telegram send failed [${LABEL}]" >&2
exit 0
