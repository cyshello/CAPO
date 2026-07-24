#!/usr/bin/env python3
"""Download ONLY a cohort's Video-MME media from HuggingFace into the layout the
DVD provider expects. For a fresh host with no local Video-MME and no rsync
access to one that has it.

lmms-lab/Video-MME stores videos in 20 archives (videos_chunked_01..20.zip,
~5 GB each); there is no per-video file, so a targeted fetch must download a zip,
extract the cohort ids found inside, delete the zip, and move to the next —
stopping once every cohort id is found. Peak disk stays ~one zip (~5 GB) instead
of the ~100 GB full set. The QA parquet and subtitle archive are small and are
fetched whole.

Produces (matching scripts/sync_cohort_data.sh / the provider's expectations):
  <data_root>/Video-MME/videomme/test-00000-of-00001.parquet
  <data_root>/Video-MME/videos/long/<id>.mp4
  <data_root>/Video-MME/subtitles/subtitle/<id>.srt

Then point the run at it: export SR_VIDEOMME_DATA_ROOT=<data_root>

Usage:
  pip install huggingface_hub
  # Video-MME is gated: accept terms on the HF page, then:
  export HF_TOKEN=hf_...            # or `huggingface-cli login`
  python scripts/download_videomme_cohort.py \
      --cohort train_set/20samples.txt \
      --data-root /workspace/videomme_data
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

REPO_ID = "lmms-lab/Video-MME"
PARQUET_NAME = "videomme/test-00000-of-00001.parquet"
SUBTITLE_ZIP = "subtitle.zip"
VIDEO_ZIPS = [f"videos_chunked_{i:02d}.zip" for i in range(1, 21)]


def _hf_download(filename: str, token: str | None):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=filename, token=token)


def _read_cohort(path: str) -> list[str]:
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        vid = line.strip()
        if vid and not vid.startswith("#"):
            ids.append(vid)
    if not ids:
        raise SystemExit(f"no video ids in {path}")
    # de-dup, preserve order
    seen: set[str] = set()
    return [v for v in ids if not (v in seen or seen.add(v))]


def _extract_matching(zip_path: str, wanted: set[str], suffix: str,
                      dest_dir: Path) -> set[str]:
    """Extract members whose basename is <id><suffix> for id in wanted.

    Returns the set of ids extracted. Matches by basename so internal zip
    layout (flat or nested) does not matter.
    """
    found: set[str] = set()
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        by_name = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = os.path.basename(info.filename)
            if base.endswith(suffix):
                by_name[base[: -len(suffix)]] = info
        for vid in list(wanted):
            info = by_name.get(vid)
            if info is None:
                continue
            target = dest_dir / f"{vid}{suffix}"
            with zf.open(info) as src, open(target, "wb") as out:
                out.write(src.read())
            found.add(vid)
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="train_set/20samples.txt",
                        help="Ordered video-id file (one id per line).")
    parser.add_argument("--data-root", required=True,
                        help="Destination; Video-MME/ is created under it.")
    parser.add_argument("--keep-zips", action="store_true",
                        help="Do not delete each video zip after extraction.")
    args = parser.parse_args(argv)

    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN")
    ids = _read_cohort(args.cohort)
    root = Path(args.data_root) / "Video-MME"
    videos_dir = root / "videos" / "long"
    subs_dir = root / "subtitles" / "subtitle"
    parquet_dir = root / "videomme"
    print(f"cohort: {len(ids)} videos from {args.cohort}", flush=True)

    # 1) QA parquet (small, whole)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    dst_parquet = parquet_dir / "test-00000-of-00001.parquet"
    if dst_parquet.exists():
        print("parquet: already present", flush=True)
    else:
        print("parquet: downloading...", flush=True)
        src = _hf_download(PARQUET_NAME, token)
        dst_parquet.write_bytes(Path(src).read_bytes())
        print(f"parquet: {dst_parquet}", flush=True)

    # 2) subtitles (small zip, whole; extract cohort .srt)
    need_subs = {v for v in ids if not (subs_dir / f"{v}.srt").exists()}
    if need_subs:
        print(f"subtitles: downloading subtitle.zip for {len(need_subs)} ids...",
              flush=True)
        sub_zip = _hf_download(SUBTITLE_ZIP, token)
        got = _extract_matching(sub_zip, need_subs, ".srt", subs_dir)
        print(f"subtitles: extracted {len(got)}/{len(need_subs)}", flush=True)
        missing_subs = need_subs - got
        if missing_subs:
            print(f"subtitles: MISSING {sorted(missing_subs)}", flush=True)
    else:
        print("subtitles: all present", flush=True)

    # 3) videos: fetch each chunk, extract cohort mp4s, delete, early-stop
    need_vids = {v for v in ids if not (videos_dir / f"{v}.mp4").exists()}
    if not need_vids:
        print("videos: all present", flush=True)
    for zip_name in VIDEO_ZIPS:
        if not need_vids:
            break
        print(f"videos: {zip_name} (need {len(need_vids)} more)...", flush=True)
        try:
            zpath = _hf_download(zip_name, token)
        except Exception as exc:  # noqa: BLE001
            print(f"videos: {zip_name} download failed: {exc}", file=sys.stderr)
            continue
        got = _extract_matching(zpath, need_vids, ".mp4", videos_dir)
        need_vids -= got
        print(f"videos: {zip_name} -> {len(got)} extracted, {len(need_vids)} left",
              flush=True)
        if not args.keep_zips:
            try:
                os.remove(zpath)  # hf cache copy; frees ~5 GB before next chunk
            except OSError:
                pass

    if need_vids:
        print(f"\nDONE with MISSING videos: {sorted(need_vids)}", file=sys.stderr)
        print("These ids were in no chunk (deleted/renamed on HF?).",
              file=sys.stderr)
        return 1
    print(f"\nDONE. data root: {args.data_root}")
    print(f"export SR_VIDEOMME_DATA_ROOT={os.path.abspath(args.data_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
