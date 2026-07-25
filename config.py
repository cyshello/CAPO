"""Central configuration for the surrogate-rollout harness.

Experiment choices live here (CLAUDE.md §26) — no hard-coded paths or model
names elsewhere. Values mirror the current prompt_sensitivity DVD setup; they
describe it, they do not change it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

HARNESS_ROOT = os.path.dirname(os.path.abspath(__file__))
# The DVD stack (data_provider / dvd_prompt / dvd_backend / the `dvd` package /
# the Video-MME provider) is vendored under vendor/dvd_stack so this repository
# clones and runs on its own. SR_PROMPT_SENS_ROOT still points the harness at an
# external checkout when one is available.
PROMPT_SENS_ROOT = os.environ.get(
    "SR_PROMPT_SENS_ROOT",
    os.path.join(HARNESS_ROOT, "vendor", "dvd_stack"),
)
DVD_ROOT = os.path.join(PROMPT_SENS_ROOT, "dvd")
DVD_RUN_WORKSPACE = os.environ.get(
    "SR_DVD_RUN_WORKSPACE", os.path.join(DVD_ROOT, "run_workspace"))

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
CAPTION_MODEL_ID = os.environ.get(
    "SR_CAPTION_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
PROMPT_GENERATOR_MODEL_ID = os.environ.get(
    "SR_PROMPT_GENERATOR_MODEL_ID", "gpt-4o-mini")
PROMPT_GENERATOR_BACKEND_ID = "openai_chat_completions_vision_replace_body_v1"
PROMPT_GENERATOR_MAX_TOKENS = 512
# The DVD agent: function-calling orchestrator and the plain-text reasoning it
# does between tool calls. frame_inspect's vision stays on the local Qwen
# captioner, so this is the only OpenAI model in the DVD loop.
ORCHESTRATOR_TOOL_MODEL = os.environ.get(
    "SR_ORCHESTRATOR_TOOL_MODEL", "gpt-4o")
# A second orchestrator for requests the first one refuses. The reasoning
# models apply a stricter prompt-policy filter than gpt-4o, and a transcript
# that trips it trips it on every retry, so without this the run stops on that
# video. The Codex shim cannot serve as the fallback: it is a ChatGPT-account
# path with its own quota, which can be -- and currently is -- exhausted.
ORCHESTRATOR_TOOL_FALLBACK_MODEL = os.environ.get(
    "SR_ORCHESTRATOR_TOOL_FALLBACK_MODEL", "gpt-4o")
TEXT_FALLBACK_MODEL = "gpt-5.5"  # codex CLI
FEEDBACK_MODEL = os.environ.get("SR_FEEDBACK_MODEL", "gpt-4o")


def _reasoning_effort(variable_name: str) -> str | None:
    """One reasoning-effort knob, unset by default.

    Only the GPT-5 family reads it; on gpt-4o the request body is unchanged, so
    leaving these unset keeps every request byte-identical to earlier runs.
    """
    value = os.environ.get(variable_name, "").strip()
    return value or None


# The DVD agent sometimes ends a run without naming an answer option, which
# cannot be scored. Re-running the QA costs seconds; letting it end the process
# costs a driver restart. Attempts include the first try.
INTERVENTION_QA_RETRY_ATTEMPTS = int(
    os.environ.get("SR_INTERVENTION_QA_RETRY_ATTEMPTS", "3"))

# How long a single provider request may keep retrying a transient transport
# failure (DNS, connection reset, 429/5xx) before the caller sees the error.
# Fifteen minutes rides out a resolver or edge blip without letting a real
# outage stall a run for an hour; a retried request is byte-identical to the
# first attempt, so nothing about the run's identity or accounting changes.
# See network_retry.py for the transient/permanent split.
NETWORK_RETRY_DEADLINE_SECONDS = float(
    os.environ.get("SR_NETWORK_RETRY_DEADLINE_SECONDS", "900"))
NETWORK_RETRY_INITIAL_DELAY_SECONDS = float(
    os.environ.get("SR_NETWORK_RETRY_INITIAL_DELAY_SECONDS", "2"))
NETWORK_RETRY_MAXIMUM_DELAY_SECONDS = float(
    os.environ.get("SR_NETWORK_RETRY_MAXIMUM_DELAY_SECONDS", "60"))

# Structural compaction of the tool evidence in the feedback payload.
COMPACT_TOOL_EVIDENCE = os.environ.get(
    "SR_COMPACT_TOOL_EVIDENCE", "0") not in ("0", "false", "False", "")

# The minimal feedback payload: delta, caption pairs, QA outcomes, one line per
# tool event. Off keeps the lean/compact payload.
EPISODE_FEEDBACK_VIEW = os.environ.get(
    "SR_EPISODE_FEEDBACK_VIEW", "0") not in ("0", "false", "False", "")

# minimal | low | medium | high. Reasoning tokens are billed as output and are
# spent from the same allowance as the visible answer, so raising the effort
# without raising the output budget truncates the reply.
GENERATOR_REASONING_EFFORT = _reasoning_effort("SR_GENERATOR_REASONING_EFFORT")
DVD_REASONING_EFFORT = _reasoning_effort("SR_DVD_REASONING_EFFORT")
# The optimization-side models are cheap (about fifteen calls per iteration), so
# effort here is a latency choice, not a cost one. They are set separately
# because the work differs: reading caption pairs is comparison, while turning a
# set of episodes into one general rule is where the reasoning earns its time.
OPTIMIZER_REASONING_EFFORT = _reasoning_effort("SR_OPTIMIZER_REASONING_EFFORT")
PROPOSER_REASONING_EFFORT = (
    _reasoning_effort("SR_PROPOSER_REASONING_EFFORT")
    or OPTIMIZER_REASONING_EFFORT)
FEEDBACK_REASONING_EFFORT = (
    _reasoning_effort("SR_FEEDBACK_REASONING_EFFORT")
    or OPTIMIZER_REASONING_EFFORT)
UPDATER_REASONING_EFFORT = (
    _reasoning_effort("SR_UPDATER_REASONING_EFFORT")
    or OPTIMIZER_REASONING_EFFORT)

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
# Reasoning-tuned captioners (Qwen3.5+) think before answering unless the chat
# template is told not to, and the same prompt then yields a different caption.
# That makes the flag part of the caption identity: a cache written with it one
# way must not answer a request made the other way. It joins the decoding dict
# only when enabled, so the default identity -- and every cache already written
# under it, including every Qwen2.5-VL cache -- is unchanged.
CAPTION_ENABLE_THINKING = os.environ.get(
    "SR_CAPTION_ENABLE_THINKING", "0") not in ("0", "false", "False", "")
if CAPTION_ENABLE_THINKING:
    CAPTION_DECODING["enable_thinking"] = True

CAPTION_SUBJECT_REGISTRY_MODE = os.environ.get(
    "SR_CAPTION_SUBJECT_REGISTRY_MODE", "empty").strip().lower()
if CAPTION_SUBJECT_REGISTRY_MODE not in {"empty", "optional"}:
    raise ValueError(
        "SR_CAPTION_SUBJECT_REGISTRY_MODE must be 'empty' or 'optional'")
CAPTION_PARSE_MAX_RETRIES = 5
# Attempts that request the canonical DVD registry JSON before the captioner
# falls back to a description-only plain-text contract for the remaining
# attempts. Must be >= 1 and <= CAPTION_PARSE_MAX_RETRIES + 1.
CAPTION_JSON_ATTEMPTS = int(os.environ.get("SR_CAPTION_JSON_ATTEMPTS", "2"))
if not 1 <= CAPTION_JSON_ATTEMPTS <= CAPTION_PARSE_MAX_RETRIES + 1:
    raise ValueError(
        "SR_CAPTION_JSON_ATTEMPTS must be between 1 and "
        f"{CAPTION_PARSE_MAX_RETRIES + 1}")

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
    "strict_hhmmss_pair_in_video_bounds_with_one_corrective_retry_v2")
DVD_FRAME_INSPECT_CORRECTIVE_RETRY_LIMIT = 1

# CLIP retrieval defaults (PHASE2_3 §9-10)
RETRIEVAL_TOP_K = 8
DEFAULT_SELECTION_POLICY = "trace_plus_prompt_delta_clip"
DEFAULT_TRACE_POLICY_PHASE2 = "returned_and_frame_inspection"


def decoding_hash() -> str:
    from surrogate_rollout.schemas import sha256_json

    return sha256_json(CAPTION_DECODING)
