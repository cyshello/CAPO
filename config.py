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

# ----------------------------- dataset ------------------------------------ #
# Benchmark selection is env-overridable (SR_BENCHMARK / SR_BENCHMARK_SPLIT)
# so alternative datasets (e.g. lvbench) can run without editing defaults.
# Defaults stay videomme/long — existing runs and manifests are unaffected.
BENCHMARK = os.environ.get("SR_BENCHMARK", "videomme")
BENCHMARK_SPLIT = os.environ.get(
    "SR_BENCHMARK_SPLIT", "long" if BENCHMARK == "videomme" else "test")
SPLIT_SEED = 0
VIDEOS_PER_SPLIT = 10  # x 3 QAs per video = 30 QAs per split (videomme)

# The videomme manifest keeps its historical filename; other benchmarks get
# their own file so they can never clobber the videomme split.
SPLIT_MANIFEST_PATH = os.environ.get("SR_SPLIT_MANIFEST_PATH", os.path.join(
    HARNESS_ROOT,
    "split_manifest.json" if BENCHMARK == "videomme"
    else f"split_manifest_{BENCHMARK}.json"))

# Videos already captioned/inspected by earlier prompt_sensitivity experiments.
# Pinned to train; must never enter validation or test (CLAUDE.md §5).
# (fFjv93ACGo8 also has a workspace cache but belongs to videomme-short — it is
# outside the long-split pool entirely, so it cannot leak into any split.)
# videomme-specific: other benchmarks have no legacy caches, so no pins.
PREVIOUSLY_CACHED_VIDEOS = (
    "0RxMZBLeqRI",
    "7D-gxaie6UI",
    "GLW9omJfAdk",
    "pU_yyadYgG8",
    "TGom0uiW130",
    "w0Wmc8C0Eq0",
    "wCkQ138sg6M",
    "xKiRmesHWIA",
) if BENCHMARK == "videomme" else ()

# ------------------------------ models ------------------------------------ #
CAPTION_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ORCHESTRATOR_TOOL_MODEL = "gpt-4o-mini"
TEXT_FALLBACK_MODEL = "gpt-5.5"  # codex CLI
FEEDBACK_MODEL = os.environ.get("SR_FEEDBACK_MODEL", "gpt-4o")

# DVD text-reasoning / tool-calling backend. "openai" routes through the
# OpenAI API; "codex" uses the codex CLI. Default is the API path so runs do
# not depend on codex CLI account quota. Override with SR_DVD_TEXT_BACKEND=codex.
DVD_TEXT_BACKEND = os.environ.get("SR_DVD_TEXT_BACKEND", "openai")
DVD_USE_OPENAI_TOOLS = os.environ.get(
    "SR_DVD_USE_OPENAI_TOOLS", "1") not in ("0", "false", "False", "")
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"

# Decoding configuration used for clip captioning. The vLLM ``max_tokens``
# value is the maximum number of newly generated tokens. This versioned,
# deterministic plain-text policy is part of every strong caption-cache key.
CAPTION_DECODING_POLICY_VERSION = "qwen_caption_plain_text_decoding_v2"
CAPTION_DECODING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1024,
    "repetition_penalty": 1.05,
    "max_frames_per_clip": None,  # run_dvd derives sample_fps * clip_secs
    "image_max_pixels": 200704,
}
CAPTION_SUBJECT_REGISTRY_MODE = os.environ.get(
    "SR_CAPTION_SUBJECT_REGISTRY_MODE", "empty").strip().lower()
if CAPTION_SUBJECT_REGISTRY_MODE not in {"empty", "optional"}:
    raise ValueError(
        "SR_CAPTION_SUBJECT_REGISTRY_MODE must be 'empty' or 'optional'")
CAPTION_PARSE_MAX_RETRIES = 5

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


# --------------------------- Phase 2: retrieval ---------------------------- #
# Cached visual index (PHASE2_3 §6) — SigLIP text+image encoders, local HF cache.
VISUAL_INDEX_MODEL_ID = "google/siglip-so400m-patch14-384"
VISUAL_INDEX_PREPROCESSING_VERSION = "vi_v1"
VISUAL_INDEX_ROOT = os.environ.get(
    "SR_VISUAL_INDEX_ROOT", os.path.join(HARNESS_ROOT, "visual_indexes")
)
VISUAL_INDEX_BATCH_SIZE = 64

# Query generators (PHASE2_3 §7-8) — text-only LLM via codex CLI.
QUERY_GENERATOR_MODEL = TEXT_FALLBACK_MODEL
QUERY_CACHE_ROOT = os.environ.get(
    "SR_QUERY_CACHE_ROOT", os.path.join(HARNESS_ROOT, "query_caches")
)
QUESTION_QUERY_PROMPT_VERSION = "qq_v1"
PROMPT_DELTA_QUERY_PROMPT_VERSION = "pdq_v1"
MAX_RETRIEVAL_QUERIES = 8

# Phase 4 property proposal and frame-only retrieval.
MAX_PROPERTY_PROPOSALS_PER_VIDEO = 4
PROPERTY_PROPOSAL_MAX_PAYLOAD_CHARS = 250000
PROPERTY_PROPOSAL_MAX_TRACE_EVENTS_PER_QA = 20
PROPERTY_PROPOSAL_MAX_CAPTIONS = 30
PROPERTY_PROPOSAL_MAX_TEXT_CHARS = 240
PROPERTY_PROPOSAL_MISSING_SLOT_MAX_RETRIES = 2
PROPERTY_RETRIEVAL_TOP_K = 5

# Downstream DVD caption-database retrieval.  The model-facing tool still
# exposes top_k for compatibility, but the harness deterministically executes
# every clip_search_tool call with this value.
DVD_CLIP_SEARCH_TOP_K = 16
DVD_CLIP_SEARCH_POLICY_VERSION = "fixed_clip_search_top_k_v1"
DVD_FRAME_INSPECT_TOOL_CONTRACT_VERSION = (
    "strict_hhmmss_pair_with_one_corrective_retry_v1")
DVD_FRAME_INSPECT_CORRECTIVE_RETRY_LIMIT = 1

# CLIP retrieval defaults (PHASE2_3 §9-10)
RETRIEVAL_TOP_K = 8
DEFAULT_SELECTION_POLICY = "trace_plus_prompt_delta_clip"
DEFAULT_TRACE_POLICY_PHASE2 = "returned_and_frame_inspection"


def decoding_hash() -> str:
    from surrogate_rollout.schemas import sha256_json

    return sha256_json(CAPTION_DECODING)
