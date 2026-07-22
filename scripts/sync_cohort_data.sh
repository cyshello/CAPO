#!/usr/bin/env bash
# Copy only the Video-MME files the meta-prompt optimization cohorts need into
# a portable data root, so a fresh machine gets ~8.4 GB instead of 300 GB.
#
#   bash scripts/sync_cohort_data.sh <destination_data_root>
#
# Produces:
#   <destination>/videomme_data/Video-MME/videomme/*.parquet
#   <destination>/videomme_data/Video-MME/videos/long/<cohort id>.mp4
#   <destination>/videomme_data/Video-MME/subtitles/subtitle/<cohort id>.srt
#
# Point the target machine at it with SR_VIDEOMME_DATA_ROOT=<destination>/videomme_data.

set -euo pipefail

PROJECT_ROOT="${SR_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_ROOT="${SR_VIDEOMME_DATA_ROOT:-/hub_data3/videomme_data}"

if [[ "$#" -ne 1 ]]; then
  echo "usage: bash scripts/sync_cohort_data.sh <destination_data_root>" >&2
  exit 2
fi
DESTINATION="$1"

COHORT_FILES=(
  "${SR_EVIDENCE_COHORT_FILE:-$PROJECT_ROOT/train_set/20samples.txt}"
  "${SR_CONFIRMATION_COHORT_FILE:-$PROJECT_ROOT/train_set/confirmation_10samples.txt}"
)
for path in "${COHORT_FILES[@]}"; do
  test -f "$path"
done
test -d "$SOURCE_ROOT/Video-MME"

mapfile -t VIDEO_IDS < <(cat "${COHORT_FILES[@]}" | sed '/^\s*$/d' | sort -u)
echo "cohort videos: ${#VIDEO_IDS[@]}"

TARGET="$DESTINATION/videomme_data/Video-MME"
mkdir -p "$TARGET/videomme" "$TARGET/videos/long" "$TARGET/subtitles/subtitle"

cp -n "$SOURCE_ROOT/Video-MME/videomme/"*.parquet "$TARGET/videomme/"

missing=0
for video_id in "${VIDEO_IDS[@]}"; do
  source_video="$(find "$SOURCE_ROOT/Video-MME/videos" -name "${video_id}.mp4" -print -quit)"
  if [[ -z "$source_video" ]]; then
    echo "missing video: $video_id" >&2
    missing=$((missing + 1))
    continue
  fi
  cp -n "$source_video" "$TARGET/videos/long/${video_id}.mp4"
  source_srt="$SOURCE_ROOT/Video-MME/subtitles/subtitle/${video_id}.srt"
  if [[ -f "$source_srt" ]]; then
    cp -n "$source_srt" "$TARGET/subtitles/subtitle/"
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "$missing cohort videos were not found under $SOURCE_ROOT" >&2
  exit 1
fi

printf 'destination=%s\ntotal=%s\n' "$TARGET" "$(du -sh "$DESTINATION/videomme_data" | cut -f1)"
