"""Frame-only SigLIP retrieval for one candidate caption property."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from surrogate_rollout import config
from surrogate_rollout.optimization.property_proposal import CandidatePropertyProposal
from surrogate_rollout.optimization.policies.property_proposal import normalize_property_text
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.retrieval.visual_index import (
    SiglipEmbedder,
    VisualIndex,
    load_or_build_visual_index,
)
from surrogate_rollout.schemas import sha256_json, sha256_text


RETRIEVAL_METHOD = "siglip_text_frame_max_pool"
POOLING_RULE = "maximum_frame_cosine_v1"
RANKING_VERSION = "score_desc_start_asc_segment_id_asc_v2"
SEGMENT_UNIVERSE_POLICY = "valid_baseline_caption_segment_intersection_v2"
RETRIEVAL_ARTIFACT_SCHEMA_VERSION = "property_frame_retrieval_v3"
RETRIEVAL_BATCH_SCHEMA_VERSION = "property_retrieval_batch_v3"


class PropertyRetrievalError(RuntimeError):
    pass


class PropertyRetrievalConflictError(PropertyRetrievalError):
    pass


@dataclass(frozen=True)
class PropertyRetrievalResult:
    artifact_path: str
    artifact: dict[str, Any]
    retrieval_identity: dict[str, Any]
    resumed: bool = False

    @property
    def s_sim(self) -> tuple[str, ...]:
        return tuple(self.artifact["s_sim"])


@dataclass(frozen=True)
class PropertyRetrievalBatchResult:
    video_id: str
    manifest_path: str
    retrieval_artifact_paths: tuple[str, ...]
    resumed: bool = False


def _segment_interval(segment_id: str) -> tuple[float, float]:
    try:
        start, end = (float(value) for value in segment_id.split("_", 1))
    except (TypeError, ValueError) as exc:
        raise PropertyRetrievalError(f"invalid segment ID {segment_id!r}") from exc
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise PropertyRetrievalError(f"invalid segment interval {segment_id!r}")
    return start, end


def visual_index_identity(index: VisualIndex) -> str:
    embeddings = np.asarray(index.embeddings, dtype=np.float32)
    digest = hashlib.sha256(embeddings.tobytes(order="C")).hexdigest()
    return sha256_json({
        "key": index.key,
        "video_id": index.video_id,
        "model_id": index.model_id,
        "frame_names": index.frame_names,
        "clip_ids": index.clip_ids,
        "timestamps": index.timestamps,
        "embedding_shape": embeddings.shape,
        "embedding_sha256": digest,
    })


def retrieval_identity(
    proposal: CandidatePropertyProposal,
    index: VisualIndex,
    *,
    top_k: int,
    baseline_segment_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    text = normalize_property_text(proposal.property_text)
    if not text:
        raise PropertyRetrievalError("candidate property text is empty")
    if not index.model_id:
        raise PropertyRetrievalError("SigLIP model identity is missing")
    sampling = index.key.get("frame_sampling_config")
    if not isinstance(sampling, dict) or not sampling:
        raise PropertyRetrievalError("frame-sampling identity is missing")
    if any("history" in str(key).casefold() for key in index.key):
        raise PropertyRetrievalError(
            "visual-index identity must not contain history fields")
    eligible = _validated_baseline_segment_ids(baseline_segment_ids, index)
    return {
        "candidate_property_id": proposal.candidate_property_id,
        "source_video_id": proposal.source_video_id,
        "property_text_hash": sha256_text(text),
        "siglip_model_id": index.model_id,
        "frame_sampling_identity": sampling,
        "visual_index_identity": visual_index_identity(index),
        "pooling_rule": POOLING_RULE,
        "retrieval_top_k": top_k,
        "deterministic_ranking_version": RANKING_VERSION,
        "segment_universe_policy": SEGMENT_UNIVERSE_POLICY,
        "baseline_segment_ids_hash": sha256_json(eligible),
        "baseline_segment_count": len(eligible),
    }


def _validated_baseline_segment_ids(
    baseline_segment_ids: tuple[str, ...] | None,
    index: VisualIndex,
) -> tuple[str, ...]:
    if baseline_segment_ids is None:
        # Compatibility for direct callers outside the Phase 4 baseline path.
        values = tuple(dict.fromkeys(index.clip_ids))
    else:
        values = tuple(baseline_segment_ids)
    if not values:
        raise PropertyRetrievalError("baseline caption segment universe is empty")
    if len(values) != len(set(values)):
        raise PropertyRetrievalError(
            "baseline caption segment universe contains duplicate IDs")
    for segment_id in values:
        _segment_interval(segment_id)
    if not set(values) & set(index.clip_ids):
        raise PropertyRetrievalError(
            "visual index has no segments in the baseline caption universe")
    return values


class PropertyFrameRetriever:
    """Retrieve source-video segments without QA, caption, trace, or history input."""

    def __init__(self, *, embedder: Any | None = None,
                 top_k: int = config.PROPERTY_RETRIEVAL_TOP_K) -> None:
        if top_k < 1:
            raise ValueError("property_retrieval_top_k must be positive")
        self.embedder = embedder
        self.top_k = top_k

    def _embedder(self, model_id: str):
        if self.embedder is None:
            self.embedder = SiglipEmbedder(model_id)
        return self.embedder

    def retrieve(
        self,
        *,
        proposal: CandidatePropertyProposal,
        visual_index: VisualIndex,
        output_dir: str,
        baseline_segment_ids: tuple[str, ...] | None = None,
    ) -> PropertyRetrievalResult:
        if visual_index is None:
            raise PropertyRetrievalError("compatible visual index is required")
        if proposal.source_video_id != visual_index.video_id:
            raise PropertyRetrievalError(
                "candidate property source video does not match visual index")
        if not visual_index.frame_names or not visual_index.clip_ids:
            raise PropertyRetrievalError("visual index contains no sampled frames")
        n = len(visual_index.frame_names)
        if len(visual_index.clip_ids) != n or len(visual_index.timestamps) != n:
            raise PropertyRetrievalError("visual index frame metadata lengths differ")
        if len(set(visual_index.frame_names)) != n:
            raise PropertyRetrievalError("visual index contains duplicate frame IDs")
        embeddings = np.asarray(visual_index.embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != n or not np.isfinite(
                embeddings).all():
            raise PropertyRetrievalError("visual embeddings are missing or non-finite")
        for segment_id in set(visual_index.clip_ids):
            _segment_interval(segment_id)

        text = normalize_property_text(proposal.property_text)
        eligible_ids = _validated_baseline_segment_ids(
            baseline_segment_ids, visual_index)
        eligible_set = set(eligible_ids)
        identity = retrieval_identity(
            proposal, visual_index, top_k=self.top_k,
            baseline_segment_ids=eligible_ids)
        fingerprint = sha256_json(identity)
        output_dir = os.path.abspath(output_dir)
        artifact_path = os.path.join(output_dir, "retrieval.json")
        if os.path.exists(artifact_path):
            with open(artifact_path) as f:
                artifact = json.load(f)
            if artifact.get("retrieval_identity") != identity or \
                    artifact.get("input_fingerprint") != fingerprint:
                raise PropertyRetrievalConflictError(
                    "retrieval artifact exists for different frozen inputs")
            self._validate_saved_artifact(
                artifact, visual_index, baseline_segment_ids=eligible_ids)
            return PropertyRetrievalResult(
                artifact_path=artifact_path, artifact=artifact,
                retrieval_identity=identity, resumed=True)

        encoded = np.asarray(
            self._embedder(visual_index.model_id).embed_texts([text]),
            dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] != 1 or \
                encoded.shape[1] != embeddings.shape[1] or \
                not np.isfinite(encoded).all():
            raise PropertyRetrievalError("invalid SigLIP property-text embedding")
        q = encoded[0]
        q_norm = float(np.linalg.norm(q))
        z_norms = np.linalg.norm(embeddings, axis=1)
        if q_norm <= 0 or np.any(z_norms <= 0):
            raise PropertyRetrievalError("zero-norm text or frame embedding")
        scores = (embeddings @ q) / (z_norms * q_norm)
        if not np.isfinite(scores).all():
            raise PropertyRetrievalError("retrieval produced non-finite scores")

        grouped: dict[str, list[dict[str, Any]]] = {}
        for frame_name, segment_id, timestamp, score in zip(
                visual_index.frame_names, visual_index.clip_ids,
                visual_index.timestamps, scores):
            timestamp = float(timestamp)
            score = float(score)
            if not math.isfinite(timestamp) or not math.isfinite(score):
                raise PropertyRetrievalError("frame timestamp or score is non-finite")
            if segment_id not in eligible_set:
                continue
            grouped.setdefault(segment_id, []).append({
                "frame_id": frame_name,
                "timestamp_sec": timestamp,
                "score": score,
            })

        ranked = []
        for segment_id, frame_scores in grouped.items():
            start, end = _segment_interval(segment_id)
            ordered_frames = sorted(
                frame_scores,
                key=lambda item: (item["timestamp_sec"], item["frame_id"]))
            maximum = sorted(
                ordered_frames,
                key=lambda item: (-item["score"], item["timestamp_sec"],
                                  item["frame_id"]))[0]
            ranked.append({
                "segment_id": segment_id,
                "start_sec": start,
                "end_sec": end,
                "segment_score": maximum["score"],
                "max_frame_id": maximum["frame_id"],
                "frame_scores": ordered_frames,
            })
        ranked.sort(key=lambda item: (
            -item["segment_score"], item["start_sec"], item["segment_id"]))
        selected_count = min(self.top_k, len(ranked))
        selected_ids = tuple(item["segment_id"] for item in ranked[:selected_count])
        ranked_output = []
        for rank, item in enumerate(ranked, 1):
            ranked_output.append({
                "rank": rank, **item,
                "selected": item["segment_id"] in selected_ids,
            })
        excluded_ids = tuple(sorted(
            set(visual_index.clip_ids) - eligible_set,
            key=lambda item: (*_segment_interval(item), item)))
        artifact = {
            "schema_version": RETRIEVAL_ARTIFACT_SCHEMA_VERSION,
            "candidate_property_id": proposal.candidate_property_id,
            "source_video_id": proposal.source_video_id,
            "property_text": text,
            "retrieval_method": RETRIEVAL_METHOD,
            "top_k": self.top_k,
            "segment_universe_policy": SEGMENT_UNIVERSE_POLICY,
            "baseline_segment_ids_hash": sha256_json(eligible_ids),
            "baseline_segment_count": len(eligible_ids),
            "eligible_index_segment_count": len(ranked),
            "excluded_visual_index_segment_ids": list(excluded_ids),
            "ranked_segments": ranked_output,
            "s_sim": list(selected_ids),
            "retrieval_identity": identity,
            "input_fingerprint": fingerprint,
        }
        self._validate_saved_artifact(
            artifact, visual_index, baseline_segment_ids=eligible_ids)
        os.makedirs(output_dir, exist_ok=True)
        _atomic_write_text(artifact_path, json.dumps(
            artifact, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
        return PropertyRetrievalResult(
            artifact_path=os.path.abspath(artifact_path), artifact=artifact,
            retrieval_identity=identity, resumed=False)

    def retrieve_from_frames(
        self,
        *,
        proposal: CandidatePropertyProposal,
        frames_dir: str,
        duration: float,
        output_dir: str,
        visual_index_root: str | None = None,
        model_id: str = config.VISUAL_INDEX_MODEL_ID,
    ) -> PropertyRetrievalResult:
        index = load_or_build_visual_index(
            video_id=proposal.source_video_id, frames_dir=frames_dir,
            duration=duration, embedder=self._embedder(model_id),
            root=visual_index_root, model_id=model_id)
        return self.retrieve(
            proposal=proposal, visual_index=index, output_dir=output_dir)

    @staticmethod
    def _validate_saved_artifact(artifact: dict[str, Any],
                                 index: VisualIndex, *,
                                 baseline_segment_ids: tuple[str, ...]) -> None:
        if artifact.get("schema_version") != RETRIEVAL_ARTIFACT_SCHEMA_VERSION:
            raise PropertyRetrievalError("incompatible retrieval artifact schema")
        if artifact.get("source_video_id") != index.video_id:
            raise PropertyRetrievalError("retrieval artifact source-video mismatch")
        eligible = _validated_baseline_segment_ids(baseline_segment_ids, index)
        eligible_set = set(eligible)
        expected_ids = set(index.clip_ids) & eligible_set
        excluded_ids = set(index.clip_ids) - eligible_set
        if artifact.get("segment_universe_policy") != SEGMENT_UNIVERSE_POLICY or \
                artifact.get("baseline_segment_ids_hash") != sha256_json(eligible) or \
                int(artifact.get("baseline_segment_count", -1)) != len(eligible):
            raise PropertyRetrievalError(
                "retrieval artifact baseline segment universe mismatch")
        if set(artifact.get("excluded_visual_index_segment_ids") or ()) != excluded_ids:
            raise PropertyRetrievalError(
                "retrieval artifact excluded segment set is inconsistent")
        ranked = artifact.get("ranked_segments")
        if not isinstance(ranked, list):
            raise PropertyRetrievalError("retrieval artifact has no ranked segments")
        ids = [item.get("segment_id") for item in ranked]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise PropertyRetrievalError(
                "retrieval artifact segment set is duplicated or inconsistent")
        expected = sorted(ranked, key=lambda item: (
            -float(item["segment_score"]), float(item["start_sec"]),
            str(item["segment_id"])))
        for item in ranked:
            if not math.isfinite(float(item["segment_score"])):
                raise PropertyRetrievalError("saved segment score is non-finite")
            frame_scores = item.get("frame_scores")
            if not isinstance(frame_scores, list) or not frame_scores:
                raise PropertyRetrievalError("saved segment has no frame scores")
            if any(not math.isfinite(float(frame["score"]))
                   for frame in frame_scores):
                raise PropertyRetrievalError("saved frame score is non-finite")
        if [item["segment_id"] for item in expected] != ids:
            raise PropertyRetrievalError("retrieval artifact ranking is not deterministic")
        selected = artifact.get("s_sim")
        if not isinstance(selected, list) or not set(selected) <= set(ids):
            raise PropertyRetrievalError("S_sim contains segments outside source video")
        if selected != ids[:min(int(artifact["top_k"]), len(ids))]:
            raise PropertyRetrievalError("S_sim does not match deterministic top-k")


class PropertyRetrievalBatchRunner:
    """Evaluate every accepted property from one source video independently."""

    policy_version = "property_retrieval_batch_v3"

    def __init__(self, *, retriever: PropertyFrameRetriever,
                 visual_index_loader: Any | None = None) -> None:
        self.retriever = retriever
        self.visual_index_loader = visual_index_loader or self._load_visual_index

    @property
    def configuration_identity(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "top_k": self.retriever.top_k,
            "retrieval_method": RETRIEVAL_METHOD,
            "pooling_rule": POOLING_RULE,
            "ranking_version": RANKING_VERSION,
            "segment_universe_policy": SEGMENT_UNIVERSE_POLICY,
        }

    def _load_visual_index(self, sample: dict, video_id: str) -> VisualIndex:
        from dvd_captioning import video_duration_seconds
        from surrogate_rollout.evaluation.dvd_qa import resolve_frames_dir

        frames_dir = resolve_frames_dir(sample, video_id)
        return load_or_build_visual_index(
            video_id=video_id, frames_dir=frames_dir,
            duration=video_duration_seconds(sample["video_path"]),
            embedder=self.retriever._embedder(config.VISUAL_INDEX_MODEL_ID),
            model_id=config.VISUAL_INDEX_MODEL_ID)

    def run(self, *, sample: dict,
            proposals: tuple[CandidatePropertyProposal, ...],
            output_dir: str,
            baseline_segment_ids: tuple[str, ...] | None = None,
            ) -> PropertyRetrievalBatchResult:
        video_id = str(sample.get("extra", {}).get("videoID")
                       or sample.get("sample_id") or "")
        if not video_id:
            raise PropertyRetrievalError("sample has no source-video identity")
        if any(item.source_video_id != video_id for item in proposals):
            raise PropertyRetrievalError("proposal batch source-video mismatch")
        if len({item.candidate_property_id for item in proposals}) != len(proposals):
            raise PropertyRetrievalError("proposal batch contains duplicate IDs")
        output_dir = os.path.abspath(output_dir)
        manifest_path = os.path.join(output_dir, "manifest.json")
        batch_input = {
            "policy_version": self.policy_version,
            "video_id": video_id,
            "properties": tuple({
                "candidate_property_id": item.candidate_property_id,
                "property_text_hash": sha256_text(
                    normalize_property_text(item.property_text)),
            } for item in proposals),
            "top_k": self.retriever.top_k,
        }
        index = (self.visual_index_loader(sample, video_id)
                 if proposals else None)
        eligible_ids = (_validated_baseline_segment_ids(
            baseline_segment_ids, index) if index is not None else ())
        batch_input["visual_index_identity"] = (
            visual_index_identity(index) if index is not None else None)
        batch_input["segment_universe_policy"] = SEGMENT_UNIVERSE_POLICY
        batch_input["baseline_segment_ids_hash"] = (
            sha256_json(eligible_ids) if index is not None else None)
        batch_fingerprint = sha256_json(batch_input)
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest.get("batch_fingerprint") != batch_fingerprint:
                raise PropertyRetrievalConflictError(
                    "retrieval batch manifest has different frozen inputs")
            paths = tuple(manifest.get("retrieval_artifact_paths") or ())
            if not all(os.path.exists(path) for path in paths):
                raise PropertyRetrievalError(
                    "completed retrieval batch is missing property artifacts")
            resumed_results = tuple(self.retriever.retrieve(
                proposal=item, visual_index=index,
                output_dir=os.path.join(output_dir, item.candidate_property_id),
                baseline_segment_ids=eligible_ids)
                for item in proposals) if index is not None else ()
            actual_paths = tuple(item.artifact_path for item in resumed_results)
            if actual_paths != paths:
                raise PropertyRetrievalConflictError(
                    "retrieval batch property paths do not match manifest")
            return PropertyRetrievalBatchResult(
                video_id=video_id, manifest_path=manifest_path,
                retrieval_artifact_paths=paths, resumed=True)

        if not proposals:
            results = ()
        else:
            results = tuple(self.retriever.retrieve(
                proposal=item, visual_index=index,
                output_dir=os.path.join(output_dir, item.candidate_property_id),
                baseline_segment_ids=eligible_ids)
                for item in proposals)
        paths = tuple(item.artifact_path for item in results)
        os.makedirs(output_dir, exist_ok=True)
        _atomic_write_text(manifest_path, json.dumps({
            "schema_version": RETRIEVAL_BATCH_SCHEMA_VERSION,
            "status": "completed", "video_id": video_id,
            "batch_fingerprint": batch_fingerprint,
            "segment_universe_policy": SEGMENT_UNIVERSE_POLICY,
            "baseline_segment_ids_hash": (
                sha256_json(eligible_ids) if index is not None else None),
            "baseline_segment_count": len(eligible_ids),
            "visual_index_identity": (
                batch_input["visual_index_identity"]),
            "retrieval_artifact_paths": paths,
        }, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
        return PropertyRetrievalBatchResult(
            video_id=video_id, manifest_path=os.path.abspath(manifest_path),
            retrieval_artifact_paths=paths)
