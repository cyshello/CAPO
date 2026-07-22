"""Captioner — component 2.

Thin re-export of surrogate_rollout.captioning. The captioner is
Qwen2.5-VL-7B-Instruct served by vLLM.

Must run in the `local_llm_vllm` conda env (vLLM + transformers + qwen_vl_utils
+ opencv). Building the captioner loads the model onto the GPU, so build once
and reuse.
"""

from __future__ import annotations

import os
import sys
from glob import glob

from surrogate_rollout.captioning import (  # noqa: E402
    BaseCaptioner,
    Qwen25VLCaptioner,
    sample_video_frames,
)

__all__ = [
    "BaseCaptioner",
    "Qwen25VLCaptioner",
    "sample_video_frames",
    "build_captioner",
    "caption_video",
    "caption_frames",
    "list_frame_paths",
    "frames_dir_for",
    "VIDEOMME_FRAME_CACHE",
]

# Pre-sampled frames live at <root>/<video_id>/frames/frame_n######.jpg
# (name-sorted == time order). Frames are sampled in advance, so no decode.
VIDEOMME_FRAME_CACHE = os.environ.get(
    "SR_VIDEOMME_FRAME_CACHE",
    os.path.join(os.environ.get("SR_VIDEOMME_DATA_ROOT",
                                "/hub_data3/videomme_data"),
                 "Video-MME", "frame_cache"))


def build_captioner(**kwargs) -> Qwen25VLCaptioner:
    """Instantiate the Qwen2.5-VL captioner (loads the model onto GPU)."""
    return Qwen25VLCaptioner(**kwargs)


def caption_video(
    captioner: BaseCaptioner,
    video_path: str,
    prompt: str,
    *,
    max_frames: int = 8,
    max_tokens: int | None = None,
) -> str:
    """Uniformly sample frames from a video and caption them in one request."""
    frames = sample_video_frames(video_path, max_frames=max_frames)
    return captioner.caption(frames, prompt, max_tokens=max_tokens)


def list_frame_paths(
    frames_dir: str,
    *,
    max_frames: int | None = 8,
    pattern: str = "*.jpg",
) -> list[str]:
    """Return name-sorted (== time order) frame paths from a frames dir.

    If `max_frames` is set, uniformly subsample down to that many frames.
    Pass `max_frames=None` to keep every frame.
    """
    paths = sorted(glob(os.path.join(frames_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"no frames matching {pattern!r} in {frames_dir}")
    if max_frames is None or len(paths) <= max_frames:
        return paths
    if max_frames == 1:
        return [paths[len(paths) // 2]]
    last = len(paths) - 1
    idx = [round(i * last / (max_frames - 1)) for i in range(max_frames)]
    return [paths[i] for i in idx]


def caption_frames(
    captioner: BaseCaptioner,
    image_paths: list[str],
    prompt: str,
    *,
    max_tokens: int | None = None,
) -> str:
    """Caption an explicit list of pre-sampled frame image paths."""
    return captioner.caption(image_paths, prompt, max_tokens=max_tokens)


def frames_dir_for(video_id: str, cache_root: str = VIDEOMME_FRAME_CACHE) -> str:
    """Resolve the frames dir for a video id in the frame cache."""
    return os.path.join(cache_root, video_id, "frames")
