"""Candidate-prompt clip captioning with strong-key, prompt-versioned caching.

Reuses the vendored captioning path (dvd_captioning._build_clips /
_pending_tasks / _caption_inprocess) so a candidate caption is produced by the
exact pipeline the baseline used — only PROMPTS.caption_prompt differs, and it
is set/reset around the call. Per-clip results are cached under the harness
cache root keyed by the strong CaptionCacheKey (CLAUDE.md §11); legacy DVD
workspace caches are read, never written.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import dataclass, field

from surrogate_rollout import config
from surrogate_rollout.cache.caption_cache import (
    assert_writable,
    build_cache_key,
    key_as_dict,
    legacy_md5_tag,
    new_candidate_cache_dir,
    register_cache,
)

for _p in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REQUIRED_PLACEHOLDERS = (
    "TRANSCRIPT_PLACEHOLDER",
    "CLIP_START_TIME",
    "CLIP_END_TIME",
)


class CandidatePromptError(ValueError):
    """Candidate prompt fails structural validation (CLAUDE.md §17). Callers
    turn this into a documented failure score, never a silent repair."""


def validate_candidate_prompt(candidate_prompt: str) -> None:
    missing = [ph for ph in REQUIRED_PLACEHOLDERS if ph not in candidate_prompt]
    if missing:
        raise CandidatePromptError(
            f"candidate caption prompt missing placeholders: {missing}"
        )


@dataclass
class CandidateCaptionSet:
    """Per-clip parsed captioner outputs for one (video, candidate prompt)."""

    video_id: str
    prompt_hash: str
    ckpt_dir: str
    clips: list[str]                      # every clip key of the video, ordered
    parsed: dict[str, dict]               # clip key -> parsed captioner JSON
    caption_call_count: int = 0
    cache_hits: int = 0
    caption_seconds: float = 0.0
    cache_key: dict = field(default_factory=dict)


def _subtitle_path(sample: dict) -> str | None:
    if not config.USE_TRANSCRIPT:
        return None
    return sample.get("extra", {}).get("subtitle_path")


def build_clip_index(sample: dict, video_id: str) -> list[tuple[str, dict]]:
    """The video's (clip_key, {files, transcript}) list — identical clip
    boundaries and frame subsets to the baseline captioning run."""
    from dvd.frame_caption import parse_srt_to_dict
    from dvd_captioning import _build_clips, _sorted_frames, video_duration_seconds

    from surrogate_rollout.evaluation.dvd_qa import resolve_frames_dir

    frames_dir = resolve_frames_dir(sample, video_id)
    duration = video_duration_seconds(sample["video_path"])
    subtitle = _subtitle_path(sample)
    subtitle_map = parse_srt_to_dict(subtitle) if subtitle else {}
    frames_per_clip = max(1, int(round(config.SAMPLE_FPS * config.CLIP_SECS)))
    return _build_clips(_sorted_frames(frames_dir), duration, config.CLIP_SECS,
                        frames_per_clip, subtitle_map)


def legacy_ckpt_dir(sample: dict, video_id: str, caption_prompt: str,
                    merge_prompt: str) -> str | None:
    """The read-only legacy DVD workspace ckpt dir for these prompts, if any."""
    mode = "tx" if _subtitle_path(sample) else "notx"
    tag = f"{mode}_fps{config.SAMPLE_FPS:g}_{legacy_md5_tag(caption_prompt, merge_prompt)}"
    d = os.path.join(config.DVD_RUN_WORKSPACE, video_id, f"captions_{tag}", "ckpt")
    return d if os.path.isdir(d) else None


def _prefill_from_legacy(dest_ckpt: str, legacy_dir: str) -> int:
    """Copy per-clip JSONs out of a legacy cache (reading legacy is allowed;
    the copy lives in the writable harness cache). Returns files copied."""
    copied = 0
    for name in os.listdir(legacy_dir):
        if not name.endswith(".json"):
            continue
        dst = os.path.join(dest_ckpt, name)
        if not os.path.exists(dst):
            shutil.copyfile(os.path.join(legacy_dir, name), dst)
            copied += 1
    return copied


def caption_clips(
    *,
    sample: dict,
    candidate_prompt: str,
    merge_prompt: str | None = None,
    clip_ids: set[str] | None = None,
    cache_root: str | None = None,
) -> CandidateCaptionSet:
    """Caption `clip_ids` (None = every clip) of the sample's video under the
    candidate prompt, through the prompt-versioned harness cache.

    Requires ensure_backend() first (the Qwen engine is shared with
    frame_inspect). Parse failures are cached as {} exactly like the vendored
    path — deterministic, no retry loop."""
    from dvd_captioning import _caption_inprocess, _pending_tasks
    from dvd_prompt import get_prompts, reset_prompts, set_prompts

    validate_candidate_prompt(candidate_prompt)
    video_id = sample.get("extra", {}).get("videoID") or sample["sample_id"]
    merge = merge_prompt if merge_prompt is not None else get_prompts().merge_prompt

    key = build_cache_key(
        video_id=video_id, video_path=sample["video_path"],
        caption_prompt=candidate_prompt, merge_prompt=merge,
        subtitle_path=_subtitle_path(sample),
    )
    cache_dir = new_candidate_cache_dir(key, cache_root)
    ckpt = os.path.join(cache_dir, "ckpt")
    assert_writable(cache_dir)
    os.makedirs(ckpt, exist_ok=True)
    register_cache({
        "video_id": video_id, "cache_dir": cache_dir, "key": key_as_dict(key),
        "read_only": False, "legacy": False,
    })

    legacy = legacy_ckpt_dir(sample, video_id, candidate_prompt, merge)
    if legacy:
        _prefill_from_legacy(ckpt, legacy)

    clips = build_clip_index(sample, video_id)
    all_keys = [k for k, _ in clips]
    if clip_ids is not None:
        unknown = set(clip_ids) - set(all_keys)
        if unknown:
            raise ValueError(f"unknown clip ids for {video_id}: {sorted(unknown)[:5]}")
        wanted = [(k, info) for k, info in clips if k in clip_ids]
    else:
        wanted = clips

    set_prompts(caption_prompt=candidate_prompt, merge_prompt=merge)
    t0 = time.time()
    try:
        pending = _pending_tasks(wanted, ckpt)
        if pending:
            _caption_inprocess(pending, ckpt,
                               config.CAPTION_DECODING["max_tokens"])
    finally:
        reset_prompts()
    caption_seconds = time.time() - t0

    import json

    parsed: dict[str, dict] = {}
    for k, _ in wanted:
        p = os.path.join(ckpt, f"{k}.json")
        parsed[k] = json.load(open(p)) if os.path.exists(p) else {}

    return CandidateCaptionSet(
        video_id=video_id,
        prompt_hash=key.prompt_hash,
        ckpt_dir=ckpt,
        clips=all_keys,
        parsed=parsed,
        caption_call_count=len(pending),
        cache_hits=len(wanted) - len(pending),
        caption_seconds=caption_seconds,
        cache_key=key_as_dict(key),
    )
