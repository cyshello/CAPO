"""Build DVD-format captions.json from pre-sampled frame_cache frames.

DVD's own process_video parallelises captioning with mp.Pool(16), which cannot
coexist with an in-process vLLM engine. We reproduce the same steps serially,
reusing DVD's `_caption_clip` (which reads PROMPTS.caption_prompt and routes to
Qwen once the backend is installed) and `merge_subject_registries`
(PROMPTS.merge_prompt -> codex).

Frame timestamps come from the pre-sampled cache: frame i of N over a video of
duration D seconds maps to t_i = i * D / N.
"""

from __future__ import annotations

import json
import math
import os
import re

import cv2

import dvd.config as config
from dvd.build_database import convert_seconds_to_hhmmss
from dvd.frame_caption import merge_subject_registries, parse_srt_to_dict
from dvd.prompts import get_prompts


def video_duration_seconds(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0 or n <= 0:
        raise RuntimeError(f"cannot read duration from {video_path}")
    return n / fps


def _sorted_frames(frames_dir: str) -> list[str]:
    files = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
    files.sort()
    return [os.path.join(frames_dir, f) for f in files]


def decode_frames_at_fps(video_path: str, out_dir: str, target_fps: float) -> int:
    """Decode `video_path` into JPEGs sampled at `target_fps`, named
    frame_n{i:06d}.jpg (i = sequential index at target_fps, so frame i is at
    t = i / target_fps seconds). Uniform for every video, unlike the frame
    cache. Skips work if already decoded. Returns the frame count.
    """
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir)
                if f.startswith("frame_n") and f.endswith(".jpg")]
    if existing:
        return len(existing)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if src_fps <= 0:
        cap.release()
        raise RuntimeError(f"cannot read fps from {video_path}")
    interval = int(round(src_fps / target_fps)) if target_fps < src_fps else 1

    frame_count = saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            cv2.imwrite(os.path.join(out_dir, f"frame_n{saved:06d}.jpg"), frame)
            saved += 1
        frame_count += 1
    cap.release()
    return saved


_TAG_RE = re.compile(r"<[^>]+>")


def _clean_subtitle(text: str) -> str:
    """Strip HTML/font tags and collapse whitespace from an .srt line."""
    return re.sub(r"\s+", " ", _TAG_RE.sub("", text)).strip()


def _clip_transcript(subtitle_map: dict, clip_start: float, clip_end: float) -> str:
    """Join cleaned subtitle text overlapping [clip_start, clip_end] (seconds)."""
    parts = []
    for key, text in subtitle_map.items():
        s, e = map(int, key.split("_"))
        if s <= clip_end and e >= clip_start:  # overlap
            cleaned = _clean_subtitle(text)
            if cleaned:
                parts.append(cleaned)
    return " ".join(parts).strip() or "No transcript."


def _build_clips(frame_paths: list[str], duration: float, clip_secs: int,
                 max_frames_per_clip: int,
                 subtitle_map: dict | None = None) -> list[tuple[str, dict]]:
    n = len(frame_paths)
    ts = [i * duration / n for i in range(n)]  # uniform sampling assumption
    subtitle_map = subtitle_map or {}
    clips = []
    start = 0
    while start < duration:
        end = min(start + clip_secs, duration)
        idx = [i for i in range(n) if start <= ts[i] < end]
        if idx:
            if len(idx) > max_frames_per_clip:  # uniform subsample
                step = len(idx) / max_frames_per_clip
                idx = [idx[int(k * step)] for k in range(max_frames_per_clip)]
            files = [frame_paths[i] for i in idx]
            key = f"{int(start)}_{int(end)}"
            transcript = _clip_transcript(subtitle_map, start, end)
            clips.append((key, {"files": files, "transcript": transcript}))
        start += clip_secs
    return clips


def _pending_tasks(clips: list[tuple[str, dict]], ckpt_folder: str) -> list[dict]:
    """Clips not yet cached, with their fully-rendered caption prompt."""
    prompts = get_prompts()
    pending = []
    for key, info in clips:
        if os.path.exists(os.path.join(ckpt_folder, f"{key}.json")):
            continue
        start_s, end_s = key.split("_")
        prompt = (
            prompts.caption_prompt
            .replace("TRANSCRIPT_PLACEHOLDER", info["transcript"])
            .replace("CLIP_START_TIME", convert_seconds_to_hhmmss(float(start_s)))
            .replace("CLIP_END_TIME", convert_seconds_to_hhmmss(float(end_s)))
        )
        pending.append({
            "key": key, "files": info["files"],
            "prompt": prompt, "transcript": info["transcript"],
        })
    return pending


def _caption_inprocess(pending: list[dict], ckpt_folder: str, max_tokens: int) -> None:
    """Caption all pending clips in ONE Qwen batch in this process (1 GPU)."""
    from dvd_backend import get_captioner, _strip_fences, _extract_json

    outputs = get_captioner().caption_batch(
        [t["files"] for t in pending], [t["prompt"] for t in pending],
        max_tokens=max_tokens,
    )
    for task, raw in zip(pending, outputs):
        try:
            parsed = _extract_json(_strip_fences(raw))
            parsed["clip_description"] = (
                parsed.get("clip_description", "")
                + f"\n\nTranscript during this video clip: {task['transcript']}."
            )
        except Exception:
            parsed = {}  # cache empty on deterministic parse failure (no retry loop)
        with open(os.path.join(ckpt_folder, f"{task['key']}.json"), "w") as f:
            json.dump(parsed, f)


def _caption_shards_dp(pending: list[dict], ckpt_folder: str, gpus: list[str],
                       max_tokens: int, max_images: int) -> None:
    """Data-parallel captioning: one worker subprocess per GPU, round-robin
    shards. Each worker builds its own engine, captions its shard, exits."""
    import subprocess
    import sys
    import tempfile

    shards: list[list[dict]] = [[] for _ in gpus]
    for i, task in enumerate(pending):
        shards[i % len(gpus)].append(task)

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "dvd_caption_worker.py")
    procs = []
    tmpfiles = []
    for gpu, shard in zip(gpus, shards):
        if not shard:
            continue
        tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(shard, tf)
        tf.close()
        tmpfiles.append(tf.name)
        procs.append(subprocess.Popen(
            [sys.executable, worker, tf.name, ckpt_folder, gpu,
             str(max_tokens), str(max_images)],
            cwd=here,
        ))
    try:
        rcs = [p.wait() for p in procs]
    finally:
        for f in tmpfiles:
            try:
                os.unlink(f)
            except OSError:
                pass
    for rc in rcs:
        if rc != 0:
            raise RuntimeError(f"caption worker failed with exit code {rc}")


def _caption_clips_batched(clips: list[tuple[str, dict]], ckpt_folder: str,
                           *, max_tokens: int = 1024,
                           gpus: list[str] | None = None,
                           max_images: int = 16) -> list[tuple[str, dict]]:
    """Caption uncached clips (data-parallel across `gpus` if >1, else one batch
    in-process). Per-clip JSON is cached under ckpt/ so reruns skip finished
    clips. Returns [(key, parsed_or_empty), ...] in clip order."""
    pending = _pending_tasks(clips, ckpt_folder)
    if pending:
        if gpus and len(gpus) > 1:
            _caption_shards_dp(pending, ckpt_folder, gpus, max_tokens, max_images)
        else:
            _caption_inprocess(pending, ckpt_folder, max_tokens)

    results = []
    for key, _ in clips:
        p = os.path.join(ckpt_folder, f"{key}.json")
        if os.path.exists(p):
            with open(p) as f:
                results.append((key, json.load(f)))
        else:
            results.append((key, {}))
    return results


def build_captions_json(video_path: str, frames_dir: str, out_captions_dir: str,
                        *, clip_secs: int | None = None,
                        max_frames_per_clip: int = 8,
                        subtitle_path: str | None = None,
                        gpus: list[str] | None = None,
                        max_images: int = 16,
                        max_tokens: int = 1024,
                        before_merge=None) -> tuple[str, float]:
    """Caption every clip with Qwen and write DVD's captions.json.

    If `subtitle_path` is given, each clip's overlapping .srt text is passed to
    the caption prompt (TRANSCRIPT_PLACEHOLDER) and appended to the stored
    caption. With >1 `gpus`, captioning is data-parallel across worker
    subprocesses. `before_merge` (if given) is called after captioning but
    before the codex merge step — used to build the frame_inspect engine in a
    clean process, before any in-process codex call. Returns
    (captions_json_path, effective_fps). Backend must be installed first.
    """
    clip_secs = clip_secs or config.CLIP_SECS
    os.makedirs(out_captions_dir, exist_ok=True)
    ckpt = os.path.join(out_captions_dir, "ckpt")
    os.makedirs(ckpt, exist_ok=True)

    duration = video_duration_seconds(video_path)
    frame_paths = _sorted_frames(frames_dir)
    effective_fps = len(frame_paths) / duration

    subtitle_map = parse_srt_to_dict(subtitle_path) if subtitle_path else {}
    clips = _build_clips(
        frame_paths, duration, clip_secs, max_frames_per_clip, subtitle_map
    )

    results = _caption_clips_batched(
        clips, ckpt, max_tokens=max_tokens, gpus=gpus, max_images=max_images
    )

    if before_merge is not None:
        before_merge()

    frame_captions: dict = {}
    partial_registries = []
    for ts_key, parsed in results:
        if parsed and parsed.get("clip_description"):
            frame_captions[ts_key] = {"caption": parsed["clip_description"]}
            if parsed.get("subject_registry"):
                partial_registries.append(parsed["subject_registry"])

    frame_captions["subject_registry"] = merge_subject_registries(partial_registries)

    out_path = os.path.join(out_captions_dir, "captions.json")
    with open(out_path, "w") as f:
        json.dump(frame_captions, f, indent=4)
    return out_path, effective_fps
