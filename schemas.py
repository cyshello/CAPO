"""Typed records shared across the surrogate-rollout harness (CLAUDE.md §9).

These records standardize what every rollout produces so that full and
surrogate evaluators, the fidelity experiment, and the GEPA adapter all consume
the same shapes. Reference-set fields deliberately keep the raw evidence sets
separate (retrieved / returned / cited / inspected) so stricter reference
policies can be evaluated later without re-running DVD.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    """Stable hash of a JSON-serializable object (sorted keys)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class QAExample:
    video_id: str
    question_id: str          # "<benchmark>/<split>/<provider index>"
    question: str
    ground_truth: str
    provider_index: int
    options: tuple[str, ...] = ()
    subtitle_available: bool = False


@dataclass(frozen=True)
class CaptionCacheKey:
    """Strong identity of one cached caption set (CLAUDE.md §11).

    The vendored DVD cache tag is only md5(caption_prompt||merge_prompt)[:8];
    this key adds the caption model, decoding, and source identity. Optional
    history/component fields extend it for sequential baselines. Existing
    legacy caches are registered with reconstructed fields, marked read-only,
    and never rewritten.
    """

    video_id: str
    segment_id: str           # "{start}_{end}" clip key, or "*" for whole-video sets
    prompt_hash: str          # sha256 of caption_prompt||merge_prompt (full)
    caption_model_id: str
    decoding_hash: str        # sha256 of decoding-config JSON
    source_hash: str          # sha256 of frame/subtitle source identity
    history_hash: str | None = None
    composed_prompt_hash: str | None = None
    bank_version: str | None = None
    router_version: str | None = None
    scaffold_version: str | None = None
    contract_version: str | None = None
    backend_id: str | None = None
    history_config_hash: str | None = None
    intervention_identity_hash: str | None = None

    @property
    def legacy_tag8(self) -> str:
        """First 8 hex of the *sha256* prompt hash — used only for new harness
        cache dirs. The vendored DVD md5 tag is stored separately in the
        manifest for legacy caches."""
        return self.prompt_hash[:8]


@dataclass
class ReferenceSets:
    """Raw evidence sets extracted from one trajectory. All members are DVD
    clip keys ("{start}_{end}")."""

    retrieved_segments: set[str] = field(default_factory=set)
    returned_segments: set[str] = field(default_factory=set)
    frame_inspected_segments: set[str] = field(default_factory=set)
    explicitly_cited_segments: set[str] = field(default_factory=set)
    consumed_segments: set[str] = field(default_factory=set)
    evidence: list[dict] = field(default_factory=list)  # why each segment was selected

    def as_json(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, set):
                d[k] = sorted(v)
        return d


@dataclass
class DVDRunResult:
    question_id: str
    video_id: str
    prediction: str | None
    parsed_answer: str | None
    ground_truth: str | None
    score: float
    trajectory: list[dict]
    references: ReferenceSets
    total_segments: int
    token_usage: dict[str, Any]      # per-route counts; token fields None when backend hides them
    latency_seconds: float
    caption_cache_tag: str
    captions_path: str | None
    database_path: str | None
    errors: list[dict] = field(default_factory=list)

    def as_json(self) -> dict:
        d = asdict(self)
        d["references"] = self.references.as_json()
        return d


@dataclass
class CandidateEvaluation:
    prompt: str
    prompt_hash: str
    rollout_mode: Literal["full", "surrogate"]
    aggregate_score: float
    examples: list[DVDRunResult]
    recaptioned_segments: int
    total_segments: int
    caption_cache_hits: int
    caption_cache_misses: int
    caption_seconds: float
    reasoning_seconds: float
    fallback_reasons: list[str] = field(default_factory=list)
