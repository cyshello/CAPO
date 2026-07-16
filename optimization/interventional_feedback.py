"""Flip-only multimodal feedback and property-level aggregation."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from PIL import Image, ImageOps, __version__ as pillow_version

from surrogate_rollout.optimization.property_proposal import CandidatePropertyProposal
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_json, sha256_text


FLIP_TRANSITIONS = ("wrong_to_correct", "correct_to_wrong")
RECOMMENDATION_LABELS = (
    "add", "revise", "merge", "retire", "router_positive",
    "router_negative", "no_op",
)
COVERAGE_LABELS = ("uncovered", "partially_covered", "covered", "uncertain")
OUTPUT_FIELDS = frozenset({
    "assessment", "coverage_assessment", "covered_by_property_ids",
    "evidence", "recommendation",
})


class InterventionalFeedbackError(RuntimeError):
    pass


class InterventionalFeedbackConflictError(InterventionalFeedbackError):
    pass


class InterventionalFeedbackParseError(InterventionalFeedbackError):
    pass


@dataclass(frozen=True)
class FeedbackEvidenceBounds:
    max_segments: int = 5
    max_frames_per_segment: int = 2
    max_frame_bytes: int = 65536
    max_history_items: int = 3
    max_caption_chars: int = 1200
    max_reasoning_events: int = 3
    max_reasoning_event_chars: int = 1200
    max_codebook_entries: int = 4
    max_payload_chars: int = 400000
    max_raw_output_chars: int = 5000
    max_evidence_words: int = 50
    max_evidence_chars: int = 400

    def __post_init__(self) -> None:
        for name, value in as_json_dict(self).items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"FeedbackEvidenceBounds.{name} must be positive")


@dataclass(frozen=True)
class FrameTransformConfig:
    max_dimension: int = 768
    initial_quality: int = 85
    minimum_quality: int = 35
    quality_step: int = 10
    max_resize_steps: int = 4
    resize_numerator: int = 3
    resize_denominator: int = 4
    jpeg_subsampling: int = 2
    implementation_version: str = "bounded_jpeg_v1"

    def __post_init__(self) -> None:
        if self.max_dimension < 1 or self.max_resize_steps < 0:
            raise ValueError("frame resize bounds are invalid")
        if not 1 <= self.minimum_quality <= self.initial_quality <= 95:
            raise ValueError("frame JPEG quality bounds are invalid")
        if self.quality_step < 1:
            raise ValueError("frame quality_step must be positive")
        if not 0 < self.resize_numerator < self.resize_denominator:
            raise ValueError("frame resize ratio must be between zero and one")
        if self.jpeg_subsampling not in (0, 1, 2):
            raise ValueError("frame JPEG subsampling must be 0, 1, or 2")


@dataclass(frozen=True)
class QAFlipFeedback:
    question_id: str
    transition: str
    status: str
    s_used_before: tuple[str, ...]
    s_used_after: tuple[str, ...]
    s_feedback: tuple[str, ...]
    request_path: str | None
    raw_output_path: str | None
    parsed_output_path: str | None
    rejection_path: str | None
    feedback_id: str | None = None
    assessment: str | None = None
    coverage_assessment: str | None = None
    covered_by_property_ids: tuple[str, ...] = ()
    evidence: str | None = None
    recommendation: str | None = None
    rejection_reason: str | None = None
    resumed: bool = False


@dataclass(frozen=True)
class PropertyFeedbackAggregate:
    candidate_property_id: str
    property_text: str
    source_video_id: str
    source_question_ids: tuple[str, ...]
    flip_question_ids: tuple[str, ...]
    helped_question_ids: tuple[str, ...]
    harmed_question_ids: tuple[str, ...]
    status: str
    positive_flip_count: int
    negative_flip_count: int
    unchanged_count: int
    accepted_feedback_count: int
    rejected_feedback_count: int
    supporting_s_feedback: tuple[str, ...]
    evidence_references: tuple[str, ...]
    coverage_assessment: str
    covered_by_property_ids: tuple[str, ...]
    recommendation_evidence: tuple[Mapping[str, str], ...]
    qa_feedback: tuple[QAFlipFeedback, ...]
    result_path: str
    input_fingerprint: str
    failure_reason: str | None = None
    resumed: bool = False


@dataclass(frozen=True)
class PropertyFeedbackBatchResult:
    source_video_id: str
    manifest_path: str
    properties: tuple[PropertyFeedbackAggregate, ...]


def _read_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise InterventionalFeedbackError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: str) -> tuple[dict[str, Any], ...]:
    with open(path) as f:
        return tuple(json.loads(line) for line in f if line.strip())


def _write_json(path: str, value: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, json.dumps(
        as_json_dict(value), sort_keys=True, ensure_ascii=False, indent=2) + "\n")
    return os.path.abspath(path)


def _write_text(path: str, value: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, value)
    return os.path.abspath(path)


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_"
                   for char in value)


def _caption_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("caption") or value.get("clip_description") or "")
    return str(value or "")


def _bounded_text(value: Any, limit: int, truncations: list[dict[str, Any]],
                  field: str) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    truncations.append({"field": field, "original_chars": len(text),
                        "retained_chars": limit})
    return text[:limit]


def _used_segments(row: Mapping[str, Any]) -> tuple[str, ...]:
    direct = row.get("used_segments")
    if direct:
        return tuple(dict.fromkeys(str(item) for item in direct))
    refs = row.get("reference_sets") or {}
    consumed = refs.get("consumed_segments") or ()
    if consumed:
        return tuple(dict.fromkeys(str(item) for item in consumed))
    fallback = set(refs.get("frame_inspected_segments") or ())
    fallback.update(refs.get("explicitly_cited_segments") or ())
    fallback.update(refs.get("returned_segments") or ())
    return tuple(sorted(str(item) for item in fallback))


def _reasoning_excerpt(path: str | None, segment_ids: tuple[str, ...],
                       bounds: FeedbackEvidenceBounds,
                       truncations: list[dict[str, Any]], side: str) -> tuple[str, ...]:
    if not path or not os.path.exists(path):
        return ()
    rows = _read_jsonl(path)
    serialized = tuple(dumps_canonical(row) for row in rows)
    relevant = tuple(text for text in serialized
                     if any(segment_id in text for segment_id in segment_ids))
    selected = (relevant or serialized)[-bounds.max_reasoning_events:]
    return tuple(_bounded_text(
        text, bounds.max_reasoning_event_chars, truncations,
        f"reasoning.{side}[{index}]") for index, text in enumerate(selected))


def load_bounded_frame(
    path: str, max_bytes: int, transform: FrameTransformConfig,
) -> Mapping[str, Any]:
    with open(path, "rb") as f:
        source = f.read()
    source_hash = hashlib.sha256(source).hexdigest()
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(source))).convert("RGB")
    except Exception as exc:
        raise InterventionalFeedbackError(
            f"cannot decode frame for bounded transformation: {path}") from exc
    original_size = image.size
    if max(image.size) > transform.max_dimension:
        image.thumbnail(
            (transform.max_dimension, transform.max_dimension),
            Image.Resampling.LANCZOS)
    qualities = list(range(
        transform.initial_quality, transform.minimum_quality - 1,
        -transform.quality_step))
    if qualities[-1] != transform.minimum_quality:
        qualities.append(transform.minimum_quality)
    for resize_step in range(transform.max_resize_steps + 1):
        for quality in qualities:
            output = io.BytesIO()
            image.save(
                output, format="JPEG", quality=quality,
                subsampling=transform.jpeg_subsampling,
                optimize=False, progressive=False)
            data = output.getvalue()
            if len(data) <= max_bytes:
                return {
                    "frame_id": os.path.basename(path),
                    "mime_type": "image/jpeg",
                    "source_sha256": source_hash,
                    "transformed_sha256": hashlib.sha256(data).hexdigest(),
                    "transformed_bytes": len(data),
                    "original_dimensions": original_size,
                    "transformed_dimensions": image.size,
                    "applied_quality": quality,
                    "resize_step": resize_step,
                    "transform_configuration": {
                        **as_json_dict(transform),
                        "pillow_version": pillow_version,
                    },
                    "base64_data": base64.b64encode(data).decode("ascii"),
                }
        if resize_step < transform.max_resize_steps:
            width = max(
                1, image.width * transform.resize_numerator
                // transform.resize_denominator)
            height = max(
                1, image.height * transform.resize_numerator
                // transform.resize_denominator)
            if (width, height) == image.size:
                break
            image = image.resize((width, height), Image.Resampling.LANCZOS)
    raise InterventionalFeedbackError(
        f"frame still exceeds max_frame_bytes={max_bytes} after bounded transformation")


def _provider_identity(provider: Callable[[str], str]) -> Mapping[str, Any]:
    metadata = getattr(provider, "metadata", None)
    if callable(metadata):
        return metadata()
    return {
        "provider_type": f"{type(provider).__module__}.{type(provider).__qualname__}",
        "model": getattr(provider, "model", None),
    }


def _leakage_reason(text: str, *, video_id: str, question_id: str,
                    segment_ids: tuple[str, ...], forbidden_answers: tuple[str, ...],
                    max_words: int, max_chars: int) -> str | None:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return "evidence is empty"
    if len(normalized) > max_chars or len(normalized.split()) > max_words:
        return "evidence exceeds configured length bounds"
    if normalized[-1] not in ".!?" or re.search(r"[.!?].+?[.!?]", normalized):
        return "evidence must be one concise sentence"
    lowered = normalized.casefold()
    forbidden_patterns = (
        r"\bthis video\b", r"\bthe question\b", r"\bquestion id\b",
        r"\bground[ -]?truth\b", r"\bcorrect answer\b",
        r"\banswer (?:is|option)\b", r"\boption\s+[a-z0-9]\b",
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b", r"\b\d+_\d+\b",
    )
    if video_id.casefold() in lowered or question_id.casefold() in lowered:
        return "evidence mentions source lineage identifiers"
    if any(segment_id.casefold() in lowered for segment_id in segment_ids):
        return "evidence mentions segment or timestamp identifiers"
    if any(re.search(pattern, lowered) for pattern in forbidden_patterns):
        return "evidence contains instance-specific language"
    for answer in forbidden_answers:
        answer = " ".join(str(answer).split()).strip()
        if len(answer) >= 3 and re.search(
                rf"(?<!\w){re.escape(answer.casefold())}(?!\w)", lowered):
            return "evidence copies an answer or prediction"
    return None


def parse_feedback_output(
    raw: str,
    *,
    transition: str,
    active_property_ids: tuple[str, ...],
    video_id: str,
    question_id: str,
    segment_ids: tuple[str, ...],
    forbidden_answers: tuple[str, ...],
    bounds: FeedbackEvidenceBounds,
) -> dict[str, Any]:
    if len(raw) > bounds.max_raw_output_chars:
        raise InterventionalFeedbackParseError("raw output exceeds configured bound")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InterventionalFeedbackParseError("feedback output is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != OUTPUT_FIELDS:
        raise InterventionalFeedbackParseError(
            f"feedback output must contain exactly {sorted(OUTPUT_FIELDS)}")
    expected_assessment = (
        "credit" if transition == "wrong_to_correct" else "blame")
    if value["assessment"] != expected_assessment:
        raise InterventionalFeedbackParseError(
            f"{transition} feedback must use assessment={expected_assessment}")
    coverage = value["coverage_assessment"]
    if coverage not in COVERAGE_LABELS:
        raise InterventionalFeedbackParseError("invalid coverage_assessment")
    covered = value["covered_by_property_ids"]
    if not isinstance(covered, list) or any(
            not isinstance(item, str) or not item for item in covered):
        raise InterventionalFeedbackParseError(
            "covered_by_property_ids must be a string array")
    if len(covered) != len(set(covered)) or not set(covered) <= set(active_property_ids):
        raise InterventionalFeedbackParseError(
            "covered_by_property_ids contains duplicates or unknown entries")
    if coverage in ("covered", "partially_covered") and not covered:
        raise InterventionalFeedbackParseError(
            "covered feedback must identify an active property")
    if coverage == "uncovered" and covered:
        raise InterventionalFeedbackParseError(
            "uncovered feedback cannot identify covering properties")
    recommendation = value["recommendation"]
    if recommendation not in RECOMMENDATION_LABELS:
        raise InterventionalFeedbackParseError("invalid recommendation")
    evidence = value["evidence"]
    if not isinstance(evidence, str):
        raise InterventionalFeedbackParseError("evidence must be a string")
    reason = _leakage_reason(
        evidence, video_id=video_id, question_id=question_id,
        segment_ids=segment_ids, forbidden_answers=forbidden_answers,
        max_words=bounds.max_evidence_words,
        max_chars=bounds.max_evidence_chars)
    if reason:
        raise InterventionalFeedbackParseError(reason)
    return {
        "assessment": expected_assessment,
        "coverage_assessment": coverage,
        "covered_by_property_ids": tuple(covered),
        "evidence": " ".join(evidence.split()),
        "recommendation": recommendation,
    }


class PropertyFeedbackRunner:
    policy_version = "flip_only_property_feedback_v1"

    def __init__(
        self,
        *,
        response_provider: Callable[[str], str],
        bounds: FeedbackEvidenceBounds | None = None,
        frame_loader: Callable[
            [str, int, FrameTransformConfig], Mapping[str, Any]
        ] = load_bounded_frame,
        frame_transform: FrameTransformConfig | None = None,
    ) -> None:
        self.response_provider = response_provider
        self.bounds = bounds or FeedbackEvidenceBounds()
        self.frame_loader = frame_loader
        self.frame_transform = frame_transform or FrameTransformConfig()

    def run(
        self,
        *,
        iteration_id: str,
        baseline_video_manifest_path: str,
        proposals: tuple[CandidatePropertyProposal, ...],
        retrieval_artifact_paths: tuple[str, ...],
        intervention_manifest_path: str,
        prompt_bank: PromptBankSnapshot,
        output_dir: str,
        retry_failed: bool = False,
    ) -> PropertyFeedbackBatchResult:
        baseline = _read_json(baseline_video_manifest_path)
        video_id = str(baseline["video_id"])
        if any(item.source_video_id != video_id for item in proposals):
            raise InterventionalFeedbackError("proposal source-video lineage mismatch")
        retrievals = self._indexed_artifacts(
            retrieval_artifact_paths, "candidate_property_id", "retrieval")
        intervention = _read_json(intervention_manifest_path)
        if intervention.get("source_video_id") != video_id:
            raise InterventionalFeedbackError("intervention source-video mismatch")
        intervention_results = {
            item["candidate_property_id"]: item
            for item in intervention.get("results") or ()
        }
        candidate_ids = {item.candidate_property_id for item in proposals}
        if set(retrievals) != candidate_ids or set(intervention_results) != candidate_ids:
            raise InterventionalFeedbackError(
                "proposal, retrieval, and intervention candidate sets differ")
        if len(candidate_ids) != len(proposals):
            raise InterventionalFeedbackError("duplicate candidate property IDs")

        identity = {
            "policy_version": self.policy_version,
            "iteration_id": iteration_id,
            "source_video_id": video_id,
            "baseline_manifest_hash": _file_hash(baseline_video_manifest_path),
            "baseline_artifact_hashes": {
                name: _file_hash(baseline[name]) for name in (
                    "baseline_qas_path", "frames_path", "captions_path",
                    "frozen_histories_path")
            },
            "proposal_hash": sha256_json(as_json_dict(proposals)),
            "retrieval_hashes": tuple(
                _file_hash(retrievals[item.candidate_property_id]) for item in proposals),
            "intervention_manifest_hash": _file_hash(intervention_manifest_path),
            "intervention_result_hashes": tuple(
                _file_hash(intervention_results[item.candidate_property_id]["result_path"])
                for item in proposals),
            "prompt_bank_version": prompt_bank.bank_version,
            "prompt_bank_hash": sha256_json(as_json_dict(prompt_bank)),
            "provider": _provider_identity(self.response_provider),
            "bounds": as_json_dict(self.bounds),
            "frame_loader": self._callable_identity(self.frame_loader),
            "frame_transform": {
                **as_json_dict(self.frame_transform),
                "pillow_version": pillow_version,
            },
        }
        fingerprint = sha256_json(identity)
        output_dir = os.path.abspath(output_dir)
        manifest_target = os.path.join(output_dir, "manifest.json")
        if os.path.exists(manifest_target):
            existing = _read_json(manifest_target)
            if existing.get("input_fingerprint") != fingerprint:
                raise InterventionalFeedbackConflictError(
                    "feedback output exists for different frozen inputs")
            if existing.get("status") == "completed":
                properties = tuple(self._aggregate_from_json(
                    _read_json(item["result_path"]), resumed=True)
                    for item in existing["properties"])
                if retry_failed and any(
                        item.status == "failed" for item in properties):
                    return self._retry_failed_properties(
                        source_video_id=video_id, parent_fingerprint=fingerprint,
                        base_properties=properties, proposals=proposals,
                        retrievals=retrievals,
                        intervention_results=intervention_results,
                        baseline=baseline, prompt_bank=prompt_bank,
                        output_dir=output_dir)
                return PropertyFeedbackBatchResult(
                    source_video_id=video_id,
                    manifest_path=os.path.abspath(manifest_target),
                    properties=properties)

        os.makedirs(output_dir, exist_ok=True)
        properties = []
        for proposal in proposals:
            try:
                aggregate = self._run_property(
                    proposal=proposal,
                    retrieval_path=retrievals[proposal.candidate_property_id],
                    intervention_result=intervention_results[
                        proposal.candidate_property_id],
                    baseline=baseline, prompt_bank=prompt_bank,
                    output_dir=output_dir,
                    parent_fingerprint=fingerprint)
            except InterventionalFeedbackConflictError:
                raise
            except Exception as exc:
                property_dir = self._property_dir(output_dir, proposal)
                result_target = os.path.join(property_dir, "result.json")
                if os.path.exists(result_target):
                    raise InterventionalFeedbackConflictError(
                        "refusing to overwrite an existing property feedback result")
                result_path = _write_json(result_target, {
                    "schema_version": "property_feedback_aggregate_v1",
                    "input_fingerprint": sha256_json({
                        "parent": fingerprint,
                        "candidate_property_id": proposal.candidate_property_id,
                        "property_text_hash": sha256_text(proposal.property_text),
                    }),
                    "candidate_property_id": proposal.candidate_property_id,
                    "property_text": proposal.property_text,
                    "source_video_id": proposal.source_video_id,
                    "source_question_ids": proposal.source_question_ids,
                    "flip_question_ids": (), "helped_question_ids": (),
                    "harmed_question_ids": (),
                    "status": "failed", "positive_flip_count": 0,
                    "negative_flip_count": 0, "unchanged_count": 0,
                    "accepted_feedback_count": 0, "rejected_feedback_count": 0,
                    "supporting_s_feedback": (), "evidence_references": (),
                    "coverage_assessment": "uncertain",
                    "covered_by_property_ids": (), "recommendation_evidence": (),
                    "qa_feedback": (),
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "result_path": os.path.abspath(os.path.join(
                        property_dir, "result.json")),
                })
                aggregate = self._aggregate_from_json(_read_json(result_path))
            properties.append(aggregate)
        manifest_path = _write_json(manifest_target, {
            "schema_version": "property_feedback_batch_v1",
            "status": "completed",
            "iteration_id": iteration_id,
            "source_video_id": video_id,
            "input_fingerprint": fingerprint,
            "identity": identity,
            "properties": properties,
        })
        return PropertyFeedbackBatchResult(
            source_video_id=video_id, manifest_path=manifest_path,
            properties=tuple(properties))

    def _retry_failed_properties(
        self, *, source_video_id: str, parent_fingerprint: str,
        base_properties: tuple[PropertyFeedbackAggregate, ...],
        proposals: tuple[CandidatePropertyProposal, ...],
        retrievals: Mapping[str, str],
        intervention_results: Mapping[str, Mapping[str, Any]],
        baseline: Mapping[str, Any], prompt_bank: PromptBankSnapshot,
        output_dir: str,
    ) -> PropertyFeedbackBatchResult:
        retry_root = os.path.join(output_dir, "retries", "retry_001")
        retry_manifest = os.path.join(retry_root, "manifest.json")
        failed_ids = tuple(item.candidate_property_id for item in base_properties
                           if item.status == "failed")
        retry_fingerprint = sha256_json({
            "parent_fingerprint": parent_fingerprint,
            "retry_policy": "explicit_failed_only_retry_v1",
            "failed_candidate_ids": failed_ids,
        })
        if os.path.exists(retry_manifest):
            existing = _read_json(retry_manifest)
            if existing.get("input_fingerprint") != retry_fingerprint:
                raise InterventionalFeedbackConflictError(
                    "failed-property retry exists for different inputs")
            properties = tuple(self._aggregate_from_json(
                _read_json(item["result_path"]), resumed=True)
                for item in existing["properties"])
            return PropertyFeedbackBatchResult(
                source_video_id=source_video_id,
                manifest_path=os.path.abspath(retry_manifest),
                properties=properties)

        base_by_id = {item.candidate_property_id: item for item in base_properties}
        retried = []
        for proposal in proposals:
            previous = base_by_id[proposal.candidate_property_id]
            if previous.status != "failed":
                retried.append(previous)
                continue
            try:
                result = self._run_property(
                    proposal=proposal,
                    retrieval_path=retrievals[proposal.candidate_property_id],
                    intervention_result=intervention_results[
                        proposal.candidate_property_id],
                    baseline=baseline, prompt_bank=prompt_bank,
                    output_dir=retry_root,
                    parent_fingerprint=parent_fingerprint)
            except InterventionalFeedbackConflictError:
                raise
            except Exception as exc:
                property_dir = self._property_dir(retry_root, proposal)
                target = os.path.join(property_dir, "result.json")
                if os.path.exists(target):
                    raise InterventionalFeedbackConflictError(
                        "refusing to overwrite an existing retry result")
                _write_json(target, {
                    "schema_version": "property_feedback_aggregate_v1",
                    "input_fingerprint": sha256_json({
                        "retry": retry_fingerprint,
                        "candidate_property_id": proposal.candidate_property_id,
                    }),
                    "candidate_property_id": proposal.candidate_property_id,
                    "property_text": proposal.property_text,
                    "source_video_id": proposal.source_video_id,
                    "source_question_ids": proposal.source_question_ids,
                    "flip_question_ids": (), "helped_question_ids": (),
                    "harmed_question_ids": (), "status": "failed",
                    "positive_flip_count": 0, "negative_flip_count": 0,
                    "unchanged_count": 0, "accepted_feedback_count": 0,
                    "rejected_feedback_count": 0,
                    "supporting_s_feedback": (), "evidence_references": (),
                    "coverage_assessment": "uncertain",
                    "covered_by_property_ids": (), "recommendation_evidence": (),
                    "qa_feedback": (),
                    "result_path": os.path.abspath(target),
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                })
                result = self._aggregate_from_json(_read_json(target))
            retried.append(result)
        manifest_path = _write_json(retry_manifest, {
            "schema_version": "property_feedback_retry_batch_v1",
            "status": "completed", "source_video_id": source_video_id,
            "input_fingerprint": retry_fingerprint,
            "parent_manifest_path": os.path.join(output_dir, "manifest.json"),
            "retry_failed": True, "properties": retried,
        })
        return PropertyFeedbackBatchResult(
            source_video_id=source_video_id, manifest_path=manifest_path,
            properties=tuple(retried))

    @staticmethod
    def _indexed_artifacts(paths: tuple[str, ...], key: str,
                           label: str) -> dict[str, str]:
        output = {}
        for path in paths:
            value = _read_json(path)
            item_id = value.get(key)
            if not item_id or item_id in output:
                raise InterventionalFeedbackError(
                    f"invalid or duplicate {label} candidate ID")
            output[str(item_id)] = os.path.abspath(path)
        return output

    @staticmethod
    def _property_dir(output_dir: str,
                      proposal: CandidatePropertyProposal) -> str:
        return os.path.join(
            output_dir, proposal.source_video_id,
            f"{_safe_id(proposal.candidate_property_id)}_"
            f"p{sha256_text(proposal.property_text)[:12]}")

    def _run_property(
        self,
        *,
        proposal: CandidatePropertyProposal,
        retrieval_path: str,
        intervention_result: Mapping[str, Any],
        baseline: Mapping[str, Any],
        prompt_bank: PromptBankSnapshot,
        output_dir: str,
        parent_fingerprint: str,
    ) -> PropertyFeedbackAggregate:
        if intervention_result.get("status") != "completed":
            raise InterventionalFeedbackError("candidate intervention is not completed")
        result_path = intervention_result.get("result_path")
        if not result_path or not os.path.exists(result_path):
            raise InterventionalFeedbackError("candidate intervention result is missing")
        result = _read_json(result_path)
        retrieval = _read_json(retrieval_path)
        if retrieval.get("candidate_property_id") != proposal.candidate_property_id or \
                retrieval.get("source_video_id") != proposal.source_video_id or \
                retrieval.get("property_text") != proposal.property_text:
            raise InterventionalFeedbackError("retrieval lineage mismatch")
        selected = tuple(retrieval.get("s_sim") or ())
        transitions_path = result.get("transitions_path")
        if not transitions_path or not os.path.exists(transitions_path):
            raise InterventionalFeedbackError("candidate transitions are missing")
        transitions = _read_json(transitions_path)
        if transitions.get("candidate_property_id") != proposal.candidate_property_id:
            raise InterventionalFeedbackError("transition candidate lineage mismatch")
        candidate_dir = os.path.dirname(os.path.abspath(result_path))
        work_identity_path = os.path.join(candidate_dir, "work_item.json")
        composed_path = os.path.join(candidate_dir, "composed_prompts.jsonl")
        histories_path = os.path.join(candidate_dir, "frozen_histories.jsonl")
        for path in (work_identity_path, composed_path, histories_path):
            if not os.path.exists(path):
                raise InterventionalFeedbackError(
                    f"candidate intervention artifact is missing: {path}")
        property_identity = {
            "parent_fingerprint": parent_fingerprint,
            "candidate_property_id": proposal.candidate_property_id,
            "property_text_hash": sha256_text(proposal.property_text),
            "retrieval_hash": _file_hash(retrieval_path),
            "intervention_result_hash": _file_hash(result_path),
            "transitions_hash": _file_hash(transitions_path),
            "work_item_hash": _file_hash(work_identity_path),
            "composed_prompts_hash": _file_hash(composed_path),
            "frozen_histories_hash": _file_hash(histories_path),
            "mixed_captions_hash": _file_hash(result["mixed_captions_path"]),
        }
        fingerprint = sha256_json(property_identity)
        property_dir = self._property_dir(output_dir, proposal)
        aggregate_path = os.path.join(property_dir, "result.json")
        if os.path.exists(aggregate_path):
            existing = _read_json(aggregate_path)
            if existing.get("input_fingerprint") != fingerprint:
                raise InterventionalFeedbackConflictError(
                    "property feedback exists for different frozen inputs")
            return self._aggregate_from_json(existing, resumed=True)
        identity_path = os.path.join(property_dir, "input_identity.json")
        if os.path.exists(identity_path):
            existing_identity = _read_json(identity_path)
            if existing_identity.get("input_fingerprint") != fingerprint:
                raise InterventionalFeedbackConflictError(
                    "property feedback directory has a conflicting identity")
        else:
            _write_json(identity_path, {
                "input_fingerprint": fingerprint,
                "identity": property_identity,
            })

        baseline_qas = {str(item["question_id"]): item for item in _read_jsonl(
            baseline["baseline_qas_path"])}
        transition_rows = {str(item["question_id"]): item
                           for item in transitions.get("qas") or ()}
        if len(baseline_qas) != 3 or set(baseline_qas) != set(transition_rows):
            raise InterventionalFeedbackError(
                "feedback requires the same three baseline and candidate QAs")
        if not set(proposal.source_question_ids) <= set(baseline_qas):
            raise InterventionalFeedbackError(
                "proposal source-QA lineage is outside the baseline batch")
        if proposal.property_text != " ".join(proposal.property_text.split()):
            raise InterventionalFeedbackError(
                "candidate property text is not in normalized form")
        frames = _read_json(baseline["frames_path"])
        baseline_captions = _read_json(baseline["captions_path"])
        candidate_captions = _read_json(result["mixed_captions_path"])
        histories = {str(item["segment_id"]): item
                     for item in _read_jsonl(histories_path)}
        composed = {str(item["segment_id"]): item
                    for item in _read_jsonl(composed_path)}
        active = {entry.prompt_id: entry for entry in prompt_bank.entries
                  if entry.status == "active"}
        qa_feedback = []
        for question_id in sorted(baseline_qas):
            before = baseline_qas[question_id]
            after = transition_rows[question_id]
            transition = str(after["transition"])
            qa_dir = os.path.join(property_dir, "qa", _safe_id(question_id))
            used_before = _used_segments(before)
            used_after = _used_segments(after)
            if transition not in FLIP_TRANSITIONS:
                qa_feedback.append(QAFlipFeedback(
                    question_id=question_id, transition=transition,
                    status="not_eligible_unchanged", s_used_before=used_before,
                    s_used_after=used_after, s_feedback=(), request_path=None,
                    raw_output_path=None, parsed_output_path=None,
                    rejection_path=None))
                continue
            used_union = set(used_before) | set(used_after)
            s_feedback = tuple(segment_id for segment_id in selected
                               if segment_id in used_union)
            if not s_feedback:
                qa_feedback.append(self._persist_rejection(
                    qa_dir=qa_dir, question_id=question_id,
                    transition=transition, used_before=used_before,
                    used_after=used_after, s_feedback=(),
                    request={
                        "schema_version": "property_flip_feedback_request_v1",
                        "candidate_property": {
                            "candidate_property_id": proposal.candidate_property_id,
                            "property_text": proposal.property_text,
                        },
                        "transition": transition,
                        "s_sim": selected, "s_used_before": used_before,
                        "s_used_after": used_after, "s_feedback": (),
                    }, reason="empty_s_feedback"))
                continue
            if len(s_feedback) > self.bounds.max_segments:
                raise InterventionalFeedbackError(
                    "S_feedback exceeds configured segment bound")
            request = self._build_request(
                proposal=proposal, question_id=question_id, transition=transition,
                baseline_qa=before, candidate_qa=after,
                s_feedback=s_feedback, frames=frames, histories=histories,
                composed=composed, baseline_captions=baseline_captions,
                candidate_captions=candidate_captions, active=active)
            qa_feedback.append(self._call_provider(
                qa_dir=qa_dir, request=request, proposal=proposal,
                question_id=question_id, transition=transition,
                used_before=used_before, used_after=used_after,
                s_feedback=s_feedback, active_property_ids=tuple(
                    item["property_id"]
                    for item in request["relevant_active_codebook"]),
                forbidden_answers=tuple(filter(None, (
                    str(before.get("ground_truth") or ""),
                    str(before.get("prediction") or ""),
                    str(after.get("candidate_prediction") or ""),
                    *(str(item) for item in before.get("options") or ()),
                )))))

        positive = sum(item.transition == "wrong_to_correct" for item in qa_feedback)
        negative = sum(item.transition == "correct_to_wrong" for item in qa_feedback)
        unchanged = len(qa_feedback) - positive - negative
        accepted = tuple(item for item in qa_feedback if item.status == "accepted")
        rejected = tuple(item for item in qa_feedback if item.status == "rejected")
        supporting = tuple(dict.fromkeys(
            segment_id for item in accepted for segment_id in item.s_feedback))
        references = tuple(item.parsed_output_path for item in accepted
                           if item.parsed_output_path)
        covered_ids = tuple(entry.prompt_id for entry in prompt_bank.entries
                            if any(entry.prompt_id in item.covered_by_property_ids
                                   for item in accepted))
        coverages = {item.coverage_assessment for item in accepted}
        if not coverages:
            coverage = "uncertain"
        elif len(coverages) == 1:
            coverage = next(iter(coverages)) or "uncertain"
        else:
            coverage = "mixed"
        recommendation_evidence = tuple({
            "recommendation": str(item.recommendation),
            "feedback_id": str(item.feedback_id),
            "question_id": item.question_id,
            "evidence": str(item.evidence),
        } for item in accepted)
        payload = {
            "schema_version": "property_feedback_aggregate_v1",
            "input_fingerprint": fingerprint,
            "candidate_property_id": proposal.candidate_property_id,
            "property_text": proposal.property_text,
            "source_video_id": proposal.source_video_id,
            "source_question_ids": proposal.source_question_ids,
            "flip_question_ids": tuple(
                item.question_id for item in qa_feedback
                if item.transition in FLIP_TRANSITIONS),
            "helped_question_ids": tuple(
                item.question_id for item in qa_feedback
                if item.transition == "wrong_to_correct"),
            "harmed_question_ids": tuple(
                item.question_id for item in qa_feedback
                if item.transition == "correct_to_wrong"),
            "status": "completed",
            "positive_flip_count": positive,
            "negative_flip_count": negative,
            "unchanged_count": unchanged,
            "accepted_feedback_count": len(accepted),
            "rejected_feedback_count": len(rejected),
            "supporting_s_feedback": supporting,
            "evidence_references": references,
            "coverage_assessment": coverage,
            "covered_by_property_ids": covered_ids,
            "recommendation_evidence": recommendation_evidence,
            "qa_feedback": qa_feedback,
            "failure_reason": None,
            "result_path": os.path.abspath(aggregate_path),
        }
        result_path = _write_json(aggregate_path, payload)
        return self._aggregate_from_json(_read_json(result_path))

    def _build_request(
        self, *, proposal: CandidatePropertyProposal, question_id: str,
        transition: str, baseline_qa: Mapping[str, Any],
        candidate_qa: Mapping[str, Any], s_feedback: tuple[str, ...],
        frames: Mapping[str, Any], histories: Mapping[str, Any],
        composed: Mapping[str, Any], baseline_captions: Mapping[str, Any],
        candidate_captions: Mapping[str, Any], active: Mapping[str, Any],
    ) -> dict[str, Any]:
        truncations: list[dict[str, Any]] = []
        relevant_ids = []
        for segment_id in s_feedback:
            if segment_id not in frames or segment_id not in histories or \
                    segment_id not in composed:
                raise InterventionalFeedbackError(
                    f"missing bounded evidence for segment {segment_id}")
            for property_id in composed[segment_id].get("selected_prompt_ids") or ():
                if property_id != proposal.candidate_property_id and \
                        property_id in active and property_id not in relevant_ids:
                    relevant_ids.append(property_id)
        for property_id in proposal.covered_by_existing_property_ids:
            if property_id in active and property_id not in relevant_ids:
                relevant_ids.append(property_id)
        relevant_ids = relevant_ids[:self.bounds.max_codebook_entries]
        codebook = tuple({
            "property_id": property_id,
            "property_text": active[property_id].prompt_text,
        } for property_id in relevant_ids)
        segments = []
        for segment_id in s_feedback:
            frame_paths = frames[segment_id]
            if isinstance(frame_paths, str):
                frame_paths = (frame_paths,)
            loaded_frames = tuple(self.frame_loader(
                os.fspath(path), self.bounds.max_frame_bytes,
                self.frame_transform)
                for path in tuple(frame_paths)[:self.bounds.max_frames_per_segment])
            if not loaded_frames:
                raise InterventionalFeedbackError(
                    f"no selected frames available for {segment_id}")
            for frame in loaded_frames:
                if not frame.get("transformed_sha256") or \
                        not frame.get("transform_configuration"):
                    raise InterventionalFeedbackError(
                        "frame loader omitted transformation identity")
                if int(frame.get("transformed_bytes", 0)) > \
                        self.bounds.max_frame_bytes:
                    raise InterventionalFeedbackError(
                        "transformed frame exceeds configured byte bound")
            history = histories[segment_id].get("history") or ()
            bounded_history = tuple({
                "segment_id": str(item.get("segment_id") or ""),
                "caption": _bounded_text(
                    item.get("caption"), self.bounds.max_caption_chars,
                    truncations, f"segments.{segment_id}.history"),
            } for item in tuple(history)[-self.bounds.max_history_items:])
            incumbent_ids = tuple(
                item for item in composed[segment_id].get("selected_prompt_ids") or ()
                if item != proposal.candidate_property_id)
            segments.append({
                "segment_id": segment_id,
                "frames": loaded_frames,
                "frozen_history": bounded_history,
                "frozen_history_hash": histories[segment_id]["history_hash"],
                "incumbent_properties": tuple({
                    "property_id": item,
                    "property_text": active[item].prompt_text,
                } for item in incumbent_ids if item in active),
                "force_added_candidate_property": {
                    "candidate_property_id": proposal.candidate_property_id,
                    "property_text": proposal.property_text,
                },
                "before_caption": _bounded_text(
                    _caption_text(baseline_captions.get(segment_id)),
                    self.bounds.max_caption_chars, truncations,
                    f"segments.{segment_id}.before_caption"),
                "after_caption": _bounded_text(
                    _caption_text(candidate_captions.get(segment_id)),
                    self.bounds.max_caption_chars, truncations,
                    f"segments.{segment_id}.after_caption"),
            })
        request = {
            "schema_version": "property_flip_feedback_request_v1",
            "task": (
                "Return strict JSON credit or blame for the reusable candidate "
                "captioning property. Use only supplied evidence. The evidence "
                "sentence must be general and must not mention lineage IDs, "
                "timestamps, answer choices, or ground-truth answers."),
            "candidate_property": {
                "candidate_property_id": proposal.candidate_property_id,
                "property_text": proposal.property_text,
            },
            "transition": transition,
            "qa_context": {
                "question": baseline_qa.get("question"),
                "ground_truth": baseline_qa.get("ground_truth"),
                "before_prediction": baseline_qa.get("prediction"),
                "after_prediction": candidate_qa.get("candidate_prediction"),
            },
            "relevant_active_codebook": codebook,
            "segments": tuple(segments),
            "reasoning_excerpts": {
                "before": _reasoning_excerpt(
                    baseline_qa.get("trajectory_path"), s_feedback,
                    self.bounds, truncations, "before"),
                "after": _reasoning_excerpt(
                    candidate_qa.get("trajectory_path"), s_feedback,
                    self.bounds, truncations, "after"),
            },
            "truncations": tuple(truncations),
            "output_schema": {
                "assessment": "credit | blame",
                "coverage_assessment": (
                    "uncovered | partially_covered | covered | uncertain"),
                "covered_by_property_ids": ["active_property_id"],
                "evidence": "one concise reusable sentence",
                "recommendation": " | ".join(RECOMMENDATION_LABELS),
            },
        }
        if len(dumps_canonical(request)) > self.bounds.max_payload_chars:
            raise InterventionalFeedbackError(
                "feedback request exceeds max_payload_chars")
        return request

    def _call_provider(
        self, *, qa_dir: str, request: Mapping[str, Any],
        proposal: CandidatePropertyProposal, question_id: str, transition: str,
        used_before: tuple[str, ...], used_after: tuple[str, ...],
        s_feedback: tuple[str, ...], active_property_ids: tuple[str, ...],
        forbidden_answers: tuple[str, ...],
    ) -> QAFlipFeedback:
        request_target = os.path.join(qa_dir, "request.json")
        resumed = self._resume_qa_artifact(
            qa_dir=qa_dir, request=request, question_id=question_id,
            transition=transition, used_before=used_before,
            used_after=used_after, s_feedback=s_feedback)
        if resumed is not None:
            return resumed
        request_path = _write_json(request_target, request)
        raw = self.response_provider(dumps_canonical(request))
        if not isinstance(raw, str):
            raw = str(raw)
        raw_path = _write_text(os.path.join(qa_dir, "raw_output.txt"), raw)
        parsed_path = os.path.abspath(os.path.join(qa_dir, "parsed_output.json"))
        rejection_path = os.path.abspath(os.path.join(qa_dir, "rejection.json"))
        try:
            parsed = parse_feedback_output(
                raw, transition=transition,
                active_property_ids=active_property_ids,
                video_id=proposal.source_video_id, question_id=question_id,
                segment_ids=s_feedback, forbidden_answers=forbidden_answers,
                bounds=self.bounds)
        except InterventionalFeedbackParseError as exc:
            _write_json(parsed_path, {"parsed": None})
            _write_json(rejection_path, {"rejection_reason": str(exc)})
            return QAFlipFeedback(
                question_id=question_id, transition=transition,
                status="rejected", s_used_before=used_before,
                s_used_after=used_after, s_feedback=s_feedback,
                request_path=request_path, raw_output_path=raw_path,
                parsed_output_path=parsed_path, rejection_path=rejection_path,
                rejection_reason=str(exc))
        feedback_id = "property_feedback_" + sha256_json({
            "request": request, "parsed": parsed})[:24]
        parsed_value = {"feedback_id": feedback_id, **parsed}
        _write_json(parsed_path, parsed_value)
        _write_json(rejection_path, {"rejection_reason": None})
        return QAFlipFeedback(
            question_id=question_id, transition=transition, status="accepted",
            s_used_before=used_before, s_used_after=used_after,
            s_feedback=s_feedback, request_path=request_path,
            raw_output_path=raw_path, parsed_output_path=parsed_path,
            rejection_path=rejection_path, feedback_id=feedback_id,
            assessment=parsed["assessment"],
            coverage_assessment=parsed["coverage_assessment"],
            covered_by_property_ids=parsed["covered_by_property_ids"],
            evidence=parsed["evidence"], recommendation=parsed["recommendation"])

    def _persist_rejection(
        self, *, qa_dir: str, question_id: str, transition: str,
        used_before: tuple[str, ...], used_after: tuple[str, ...],
        s_feedback: tuple[str, ...], request: Mapping[str, Any], reason: str,
    ) -> QAFlipFeedback:
        resumed = self._resume_qa_artifact(
            qa_dir=qa_dir, request=request, question_id=question_id,
            transition=transition, used_before=used_before,
            used_after=used_after, s_feedback=s_feedback)
        if resumed is not None:
            return resumed
        request_path = _write_json(os.path.join(qa_dir, "request.json"), request)
        raw_path = _write_text(os.path.join(qa_dir, "raw_output.txt"), "")
        parsed_path = _write_json(os.path.join(
            qa_dir, "parsed_output.json"), {"parsed": None})
        rejection_path = _write_json(os.path.join(
            qa_dir, "rejection.json"), {"rejection_reason": reason})
        return QAFlipFeedback(
            question_id=question_id, transition=transition, status="rejected",
            s_used_before=used_before, s_used_after=used_after,
            s_feedback=s_feedback, request_path=request_path,
            raw_output_path=raw_path, parsed_output_path=parsed_path,
            rejection_path=rejection_path, rejection_reason=reason)

    @staticmethod
    def _resume_qa_artifact(
        *, qa_dir: str, request: Mapping[str, Any], question_id: str,
        transition: str, used_before: tuple[str, ...],
        used_after: tuple[str, ...], s_feedback: tuple[str, ...],
    ) -> QAFlipFeedback | None:
        request_path = os.path.abspath(os.path.join(qa_dir, "request.json"))
        raw_path = os.path.abspath(os.path.join(qa_dir, "raw_output.txt"))
        parsed_path = os.path.abspath(os.path.join(qa_dir, "parsed_output.json"))
        rejection_path = os.path.abspath(os.path.join(qa_dir, "rejection.json"))
        paths = (request_path, raw_path, parsed_path, rejection_path)
        exists = tuple(os.path.exists(path) for path in paths)
        if not any(exists):
            return None
        if not all(exists):
            raise InterventionalFeedbackConflictError(
                "partial QA feedback artifact set cannot be overwritten")
        if _read_json(request_path) != as_json_dict(request):
            raise InterventionalFeedbackConflictError(
                "QA feedback request conflicts with frozen evidence")
        parsed = _read_json(parsed_path)
        rejection = _read_json(rejection_path)
        reason = rejection.get("rejection_reason")
        if reason is not None:
            if parsed != {"parsed": None}:
                raise InterventionalFeedbackConflictError(
                    "rejected QA feedback contains a parsed response")
            return QAFlipFeedback(
                question_id=question_id, transition=transition,
                status="rejected", s_used_before=used_before,
                s_used_after=used_after, s_feedback=s_feedback,
                request_path=request_path, raw_output_path=raw_path,
                parsed_output_path=parsed_path, rejection_path=rejection_path,
                rejection_reason=str(reason), resumed=True)
        required = {
            "feedback_id", "assessment", "coverage_assessment",
            "covered_by_property_ids", "evidence", "recommendation",
        }
        if set(parsed) != required:
            raise InterventionalFeedbackConflictError(
                "accepted QA feedback parsed artifact has an invalid schema")
        return QAFlipFeedback(
            question_id=question_id, transition=transition,
            status="accepted", s_used_before=used_before,
            s_used_after=used_after, s_feedback=s_feedback,
            request_path=request_path, raw_output_path=raw_path,
            parsed_output_path=parsed_path, rejection_path=rejection_path,
            feedback_id=parsed["feedback_id"], assessment=parsed["assessment"],
            coverage_assessment=parsed["coverage_assessment"],
            covered_by_property_ids=tuple(parsed["covered_by_property_ids"]),
            evidence=parsed["evidence"], recommendation=parsed["recommendation"],
            resumed=True)

    @staticmethod
    def _aggregate_from_json(value: Mapping[str, Any],
                             resumed: bool = False) -> PropertyFeedbackAggregate:
        qa_feedback = tuple(QAFlipFeedback(
            question_id=item["question_id"], transition=item["transition"],
            status=item["status"],
            s_used_before=tuple(item.get("s_used_before") or ()),
            s_used_after=tuple(item.get("s_used_after") or ()),
            s_feedback=tuple(item.get("s_feedback") or ()),
            request_path=item.get("request_path"),
            raw_output_path=item.get("raw_output_path"),
            parsed_output_path=item.get("parsed_output_path"),
            rejection_path=item.get("rejection_path"),
            feedback_id=item.get("feedback_id"), assessment=item.get("assessment"),
            coverage_assessment=item.get("coverage_assessment"),
            covered_by_property_ids=tuple(
                item.get("covered_by_property_ids") or ()),
            evidence=item.get("evidence"), recommendation=item.get("recommendation"),
            rejection_reason=item.get("rejection_reason"),
            resumed=resumed) for item in value.get("qa_feedback") or ())
        return PropertyFeedbackAggregate(
            candidate_property_id=value["candidate_property_id"],
            property_text=value["property_text"],
            source_video_id=value["source_video_id"],
            source_question_ids=tuple(value.get("source_question_ids") or ()),
            flip_question_ids=tuple(value.get("flip_question_ids") or ()),
            helped_question_ids=tuple(value.get("helped_question_ids") or ()),
            harmed_question_ids=tuple(value.get("harmed_question_ids") or ()),
            status=value["status"],
            positive_flip_count=int(value.get("positive_flip_count", 0)),
            negative_flip_count=int(value.get("negative_flip_count", 0)),
            unchanged_count=int(value.get("unchanged_count", 0)),
            accepted_feedback_count=int(value.get("accepted_feedback_count", 0)),
            rejected_feedback_count=int(value.get("rejected_feedback_count", 0)),
            supporting_s_feedback=tuple(value.get("supporting_s_feedback") or ()),
            evidence_references=tuple(value.get("evidence_references") or ()),
            coverage_assessment=value.get("coverage_assessment", "uncertain"),
            covered_by_property_ids=tuple(
                value.get("covered_by_property_ids") or ()),
            recommendation_evidence=tuple(
                value.get("recommendation_evidence") or ()),
            qa_feedback=qa_feedback, result_path=os.path.abspath(
                value.get("result_path") or ""),
            input_fingerprint=value["input_fingerprint"],
            failure_reason=value.get("failure_reason"), resumed=resumed)

    @staticmethod
    def _callable_identity(value: Callable[..., Any]) -> str:
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{name}"
