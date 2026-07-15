"""Routed caption-view adapter for Phase 4 Stage 4.6.

The adapter groups segments by composed-prompt hash for caption generation,
then discards group order: all parsed per-clip outputs are restored to the
original DVD clip-index order before the existing whole-video subject-registry
merge and captions.json writer are invoked.  Per-group registries are never
merged or treated as final video state.

This module materializes captions only.  It does not build a vector database,
run DVD reasoning, evaluate a QA, or invoke any optimization component.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from surrogate_rollout import config
from surrogate_rollout.captioning.candidate_captions import (
    CandidateCaptionSet,
    CandidatePromptError,
    build_clip_index,
    caption_clips,
    validate_candidate_prompt,
)
from surrogate_rollout.mixed_views.builder import (
    caption_entry_from_parsed,
    default_merge_fn,
    write_captions_json,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import (
    ComposedCaptionPrompt,
    CompositionTrace,
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_text


ROUTED_PROMPT_VIEW = "routed_prompt_view"


class RoutedCaptionViewError(RuntimeError):
    """Invalid routed input or an inconsistent caption-service result."""


@dataclass(frozen=True)
class RoutedSegmentProvenance:
    segment_id: str
    composed_prompt_hash: str
    bank_version: str
    router_version: str
    scaffold_version: str
    contract_version: str
    selected_prompt_ids: tuple[str, ...]
    composition_trace: CompositionTrace


@dataclass(frozen=True)
class RoutedPromptGroup:
    composed_prompt_hash: str
    segment_ids: tuple[str, ...]
    caption_cache_key: Mapping[str, Any]
    caption_cache_prompt_hash: str
    caption_ckpt_dir: str
    caption_call_count: int
    caption_cache_hits: int
    caption_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "caption_cache_key",
                           MappingProxyType(dict(self.caption_cache_key)))


@dataclass(frozen=True)
class RoutedCaptionViewArtifact:
    artifact_type: str
    video_id: str
    view_hash: str
    captions_path: str
    captions_hash: str
    database_path: str
    routed_view_path: str
    segment_ids: tuple[str, ...]
    segments: tuple[RoutedSegmentProvenance, ...]
    prompt_groups: tuple[RoutedPromptGroup, ...]
    caption_call_count: int
    caption_cache_hits: int
    caption_seconds: float

    def as_json_dict(self) -> dict[str, Any]:
        return as_json_dict(self)


class RoutedCaptionViewBuilder:
    """Build a DVD-compatible caption view from per-segment composed prompts."""

    def __init__(
        self,
        *,
        caption_fn: Callable[..., CandidateCaptionSet] = caption_clips,
        clip_index_fn: Callable[..., list[tuple[str, dict]]] = build_clip_index,
        merge_fn: Callable[[list], object] | None = None,
        captions_writer: Callable[[dict, str], tuple[str, str]] = write_captions_json,
    ) -> None:
        self.caption_fn = caption_fn
        self.clip_index_fn = clip_index_fn
        self.merge_fn = merge_fn or default_merge_fn
        self.captions_writer = captions_writer

    def build(
        self,
        *,
        sample: dict,
        segment_composed_prompt_map: Mapping[str, ComposedCaptionPrompt],
        work_root: str,
        merge_prompt: str | None = None,
        candidate_cache_root: str | None = None,
    ) -> RoutedCaptionViewArtifact:
        video_id = sample.get("extra", {}).get("videoID") or sample.get("sample_id")
        if not video_id:
            raise RoutedCaptionViewError(
                "sample must provide extra.videoID or sample_id")

        clip_index = self.clip_index_fn(sample, video_id)
        clip_keys = tuple(key for key, _ in clip_index)
        if not clip_keys:
            raise RoutedCaptionViewError(f"video {video_id!r} has no DVD clips")
        if len(clip_keys) != len(set(clip_keys)):
            raise RoutedCaptionViewError(
                f"video {video_id!r} clip index contains duplicate keys")
        self._validate_prompt_map(
            video_id, clip_keys, segment_composed_prompt_map)

        # Group in temporal first-appearance order.  Mapping insertion order is
        # irrelevant; the DVD clip index is the sole ordering authority.
        grouped_ids: dict[str, list[str]] = {}
        grouped_prompts: dict[str, ComposedCaptionPrompt] = {}
        for segment_id in clip_keys:
            composed = segment_composed_prompt_map[segment_id]
            grouped_ids.setdefault(composed.prompt_hash, []).append(segment_id)
            representative = grouped_prompts.setdefault(
                composed.prompt_hash, composed)
            if representative.prompt_text != composed.prompt_text:
                raise RoutedCaptionViewError(
                    "composed prompt hash collision across different prompt text")

        segments = tuple(self._segment_provenance(
            segment_composed_prompt_map[segment_id]) for segment_id in clip_keys)
        view_hash = sha256_text(dumps_canonical({
            "artifact_type": ROUTED_PROMPT_VIEW,
            "video_id": video_id,
            "segments": segments,
        }))
        view_dir = os.path.join(
            work_root, video_id, f"captions_routed_{view_hash[:12]}")
        self._validate_view_isolation(
            view_dir, candidate_cache_root or config.CAPTION_CACHE_ROOT)

        parsed_by_segment: dict[str, dict] = {}
        groups: list[RoutedPromptGroup] = []
        cache_hash_owners: dict[str, str] = {}
        for composed_hash, segment_ids in grouped_ids.items():
            composed = grouped_prompts[composed_hash]
            result = self.caption_fn(
                sample=sample,
                candidate_prompt=composed.prompt_text,
                merge_prompt=merge_prompt,
                clip_ids=set(segment_ids),
                cache_root=candidate_cache_root,
            )
            self._validate_caption_result(
                result, video_id, clip_keys, tuple(segment_ids))
            cache_prompt_hash = result.cache_key.get("prompt_hash")
            if not isinstance(cache_prompt_hash, str) or not cache_prompt_hash:
                raise RoutedCaptionViewError(
                    f"caption group {composed_hash[:12]} returned no strong "
                    "cache prompt_hash")
            if result.prompt_hash != cache_prompt_hash:
                raise RoutedCaptionViewError(
                    f"caption group {composed_hash[:12]} returned inconsistent "
                    "cache prompt hashes")
            owner = cache_hash_owners.setdefault(cache_prompt_hash, composed_hash)
            if owner != composed_hash:
                raise RoutedCaptionViewError(
                    "distinct composed prompts resolved to the same caption "
                    "cache prompt hash")
            for segment_id in segment_ids:
                parsed_by_segment[segment_id] = result.parsed[segment_id]
            groups.append(RoutedPromptGroup(
                composed_prompt_hash=composed_hash,
                segment_ids=tuple(segment_ids),
                caption_cache_key=result.cache_key,
                caption_cache_prompt_hash=cache_prompt_hash,
                caption_ckpt_dir=result.ckpt_dir,
                caption_call_count=result.caption_call_count,
                caption_cache_hits=result.cache_hits,
                caption_seconds=result.caption_seconds,
            ))

        # This is deliberately the same assembly behavior as FullRolloutEvaluator:
        # valid clip captions and their registries are collected in DVD temporal
        # order, then the existing whole-video merge runs exactly once.
        captions: dict[str, dict | object] = {}
        ordered_registries: list[object] = []
        for segment_id in clip_keys:
            parsed = parsed_by_segment[segment_id]
            entry = caption_entry_from_parsed(parsed)
            if entry is None:
                continue
            captions[segment_id] = entry
            registry = parsed.get("subject_registry")
            if registry:
                ordered_registries.append(registry)
        captions["subject_registry"] = self.merge_fn(ordered_registries)

        captions_path, captions_hash = self.captions_writer(captions, view_dir)
        workdir = os.path.join(work_root, video_id)
        routed_view_path = os.path.join(view_dir, "routed_view.json")
        artifact = RoutedCaptionViewArtifact(
            artifact_type=ROUTED_PROMPT_VIEW,
            video_id=video_id,
            view_hash=view_hash,
            captions_path=captions_path,
            captions_hash=captions_hash,
            database_path=os.path.join(
                workdir, f"database_c{captions_hash[:16]}.json"),
            routed_view_path=routed_view_path,
            segment_ids=clip_keys,
            segments=segments,
            prompt_groups=tuple(groups),
            caption_call_count=sum(g.caption_call_count for g in groups),
            caption_cache_hits=sum(g.caption_cache_hits for g in groups),
            caption_seconds=sum(g.caption_seconds for g in groups),
        )
        _atomic_write_text(
            routed_view_path,
            json.dumps(artifact.as_json_dict(), sort_keys=True,
                       ensure_ascii=False, indent=2) + "\n",
        )
        return artifact

    @staticmethod
    def _validate_prompt_map(
        video_id: str,
        clip_keys: tuple[str, ...],
        prompt_map: Mapping[str, ComposedCaptionPrompt],
    ) -> None:
        if not isinstance(prompt_map, Mapping):
            raise TypeError("segment_composed_prompt_map must be a Mapping")
        missing = set(clip_keys) - set(prompt_map)
        extra = set(prompt_map) - set(clip_keys)
        if missing or extra:
            raise RoutedCaptionViewError(
                "composed prompt map must exactly cover DVD clips; "
                f"missing={sorted(missing)}, extra={sorted(extra)}")
        for segment_id in clip_keys:
            composed = prompt_map[segment_id]
            if not isinstance(composed, ComposedCaptionPrompt):
                raise RoutedCaptionViewError(
                    f"segment {segment_id!r} value is not ComposedCaptionPrompt")
            if composed.video_id != video_id or composed.segment_id != segment_id:
                raise RoutedCaptionViewError(
                    f"segment {segment_id!r} composed-prompt identity mismatch")
            if not composed.is_valid:
                raise RoutedCaptionViewError(
                    f"segment {segment_id!r} has invalid composed prompt: "
                    + "; ".join(composed.validation_errors))
            try:
                validate_candidate_prompt(composed.prompt_text)
            except CandidatePromptError as exc:
                raise RoutedCaptionViewError(
                    f"segment {segment_id!r} violates the caption prompt "
                    f"contract: {exc}") from exc

    @staticmethod
    def _validate_caption_result(
        result: Any,
        video_id: str,
        clip_keys: tuple[str, ...],
        requested_ids: tuple[str, ...],
    ) -> None:
        if not isinstance(result, CandidateCaptionSet):
            raise RoutedCaptionViewError(
                f"caption service returned {type(result).__name__}, "
                "expected CandidateCaptionSet")
        if result.video_id != video_id:
            raise RoutedCaptionViewError("caption service video_id mismatch")
        if tuple(result.clips) != clip_keys:
            raise RoutedCaptionViewError(
                "caption service clip order differs from the DVD clip index")
        missing = set(requested_ids) - set(result.parsed)
        if missing:
            raise RoutedCaptionViewError(
                f"caption service omitted requested clips: {sorted(missing)}")

    @staticmethod
    def _validate_view_isolation(view_dir: str, candidate_cache_root: str) -> None:
        view_dir = os.path.abspath(view_dir)
        protected_roots = (
            ("candidate cache", os.path.abspath(candidate_cache_root)),
            ("legacy DVD workspace", os.path.abspath(config.DVD_RUN_WORKSPACE)),
        )
        for label, root in protected_roots:
            if os.path.commonpath((view_dir, root)) in (view_dir, root):
                raise RoutedCaptionViewError(
                    f"routed view {view_dir} must be isolated from {label} "
                    f"root {root}")

    @staticmethod
    def _segment_provenance(
        composed: ComposedCaptionPrompt,
    ) -> RoutedSegmentProvenance:
        return RoutedSegmentProvenance(
            segment_id=composed.segment_id,
            composed_prompt_hash=composed.prompt_hash,
            bank_version=composed.bank_version,
            router_version=composed.router_version,
            scaffold_version=composed.scaffold_version,
            contract_version=composed.contract_version,
            selected_prompt_ids=composed.selected_prompt_ids,
            composition_trace=composed.composition_trace,
        )
