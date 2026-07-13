"""Central configuration for the surrogate-rollout harness.

Experiment choices live here (CLAUDE.md §26) — no hard-coded paths or model
names elsewhere. Values mirror the current prompt_sensitivity DVD setup; they
describe it, they do not change it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

HARNESS_ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPT_SENS_ROOT = os.environ.get(
    "SR_PROMPT_SENS_ROOT",
    os.path.abspath(os.path.join(HARNESS_ROOT, "..", "prompt_sensitivity")),
)
DVD_ROOT = os.path.join(PROMPT_SENS_ROOT, "dvd")
DVD_RUN_WORKSPACE = os.path.join(DVD_ROOT, "run_workspace")

RUNS_ROOT = os.environ.get("SR_RUNS_ROOT", os.path.join(HARNESS_ROOT, "runs"))
CAPTION_CACHE_ROOT = os.environ.get(
    "SR_CAPTION_CACHE_ROOT", os.path.join(HARNESS_ROOT, "caption_caches")
)
CACHE_MANIFEST_PATH = os.path.join(CAPTION_CACHE_ROOT, "cache_manifest.jsonl")

SPLIT_MANIFEST_PATH = os.path.join(HARNESS_ROOT, "split_manifest.json")

# ----------------------------- dataset ------------------------------------ #
BENCHMARK = "videomme"
BENCHMARK_SPLIT = "long"
SPLIT_SEED = 0
VIDEOS_PER_SPLIT = 10  # x 3 QAs per video = 30 QAs per split

# Videos already captioned/inspected by earlier prompt_sensitivity experiments.
# Pinned to train; must never enter validation or test (CLAUDE.md §5).
# (fFjv93ACGo8 also has a workspace cache but belongs to videomme-short — it is
# outside the long-split pool entirely, so it cannot leak into any split.)
PREVIOUSLY_CACHED_VIDEOS = (
    "0RxMZBLeqRI",
    "7D-gxaie6UI",
    "GLW9omJfAdk",
    "pU_yyadYgG8",
    "TGom0uiW130",
    "w0Wmc8C0Eq0",
    "wCkQ138sg6M",
    "xKiRmesHWIA",
)

# ------------------------------ models ------------------------------------ #
CAPTION_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ORCHESTRATOR_TOOL_MODEL = "gpt-4o-mini"
TEXT_FALLBACK_MODEL = "gpt-5.5"  # codex CLI
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"

# Decoding configuration used for clip captioning (mirrors dvd_captioning +
# captioning.Qwen25VLCaptioner defaults). Part of the strong cache key.
CAPTION_DECODING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1024,
    "max_frames_per_clip": None,  # run_dvd derives sample_fps * clip_secs
    "image_max_pixels": 200704,
}

# --------------------------- DVD run settings ------------------------------ #
SAMPLE_FPS = 1.0
CLIP_SECS = 10
MAX_ITERATIONS = 15
TOOL_VLM_MAX_FRAMES = 16
USE_TRANSCRIPT = True  # notx fallback happens automatically when no subtitle


# ------------------------- reference policies ------------------------------ #
@dataclass(frozen=True)
class ReferencePolicy:
    """Which evidence sets form the surrogate reference set, and how far
    temporal-neighbor expansion reaches. `base_sets` are ReferenceSets field
    names unioned before expansion."""

    name: str
    base_sets: tuple[str, ...]
    neighbor_radius: int = 1


REFERENCE_POLICIES: dict[str, ReferencePolicy] = {
    p.name: p
    for p in (
        # every clip any tool retrieved or inspected, radius-1 neighbors
        ReferencePolicy(
            "all_returned",
            ("retrieved_segments", "frame_inspected_segments"),
            neighbor_radius=1,
        ),
        # same, without neighbor expansion
        ReferencePolicy(
            "all_returned_without_neighbors",
            ("retrieved_segments", "frame_inspected_segments"),
            neighbor_radius=0,
        ),
        # only clips the agent demonstrably focused on
        ReferencePolicy(
            "explicit_citations_and_frame_inspection",
            ("explicitly_cited_segments", "frame_inspected_segments"),
            neighbor_radius=1,
        ),
        # middle ground: clips whose captions reached the orchestrator verbatim
        # (clip_search) or were inspected — excludes global_browse's top_k=100
        # bulk retrieval that would otherwise select most of the video
        ReferencePolicy(
            "returned_and_frame_inspection",
            ("returned_segments", "frame_inspected_segments"),
            neighbor_radius=1,
        ),
    )
}
DEFAULT_REFERENCE_POLICY = "all_returned"


def decoding_hash() -> str:
    from surrogate_rollout.schemas import sha256_json

    return sha256_json(CAPTION_DECODING)
