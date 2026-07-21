"""Per-video absolute scoring for GEPA over the free-form meta-prompt.

The official GEPA engine tracks an absolute per-example score vector, so the
atomic scoring unit here is ONE video (with its QAs), not the two-video paired
confirmation protocol. This module reuses the existing history-aware caption
state machine (``HistoryAwareDVDConfirmationEvaluator._caption_state``) and the
existing DVD QA function untouched; it only assembles a single-video input
bundle (the frozen paired ``_materialize_bundle`` hard-codes exactly two videos)
and memoizes QA results by caption identity so a video is not re-answered when
the same meta-prompt is scored again on a different batch.

Caption caching: the single-video bundle hash depends only on that video and the
runtime configuration, so the caption cache identity is stable for a given
``(video, meta-prompt)`` across every batch it appears in. Captions are recomputed
only when the meta-prompt text changes, which is the fundamental cost of the
search.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from surrogate_rollout.optimization.confirmation_evaluator import (
    HistoryAwareDVDConfirmationEvaluator,
    _file_hash,
    _read_json,
    _segment_times,
    _write_immutable,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    RouterPolicySnapshot,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    as_json_dict,
)
from surrogate_rollout.schemas import sha256_json, sha256_text


@dataclass(frozen=True)
class GepaVideoInstance:
    """One GEPA training example: a single video with exactly three QAs."""

    video_id: str
    provider_indices: tuple[int, ...]
    question_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("GepaVideoInstance.video_id is required")
        if len(self.provider_indices) != 3 or len(self.question_ids) != 3:
            raise ValueError("GepaVideoInstance requires exactly three QAs")
        if len(set(self.provider_indices)) != 3 or \
                len(set(self.question_ids)) != 3:
            raise ValueError("GepaVideoInstance QA identities must be unique")


@dataclass(frozen=True)
class GepaQAResult:
    question_id: str
    provider_index: int
    is_correct: bool | None
    prediction: Any
    parsed_answer: Any
    ground_truth: Any
    score: float | None
    errors: tuple[str, ...]
    captions_hash: str


@dataclass(frozen=True)
class GepaVideoScore:
    video_id: str
    accuracy: float
    evaluated_qa_count: int
    qa_results: tuple[GepaQAResult, ...]
    captions_hash: str
    caption_calls: int
    caption_cache_hits: int


def materialize_single_video_bundle(
    *, evaluator: HistoryAwareDVDConfirmationEvaluator,
    video: GepaVideoInstance, scaffold: ScaffoldPolicySnapshot,
    contract: ScaffoldContract, output_dir: str,
) -> tuple[str, Mapping[str, Any]]:
    """Assemble a one-video input bundle in the exact schema _caption_state expects.

    This mirrors ``HistoryAwareDVDConfirmationEvaluator._materialize_bundle`` for
    a single video (the paired version rejects anything but two videos). All
    caption/reference logic stays in the reused evaluator; this only builds the
    frozen input manifest.
    """

    runtime = evaluator._runtime_configuration(scaffold, contract)
    samples = tuple(evaluator.sample_loader(index)
                    for index in video.provider_indices)
    normalized_samples = tuple(as_json_dict(sample) for sample in samples)
    observed_ids = tuple(str(
        sample.get("extra", {}).get("videoID") or sample.get("sample_id"))
        for sample in samples)
    if set(observed_ids) != {video.video_id}:
        raise ValueError(f"sample/video mismatch: {video.video_id}")
    clip_index = evaluator.clip_index_fn(dict(samples[0]), video.video_id)
    segment_rows = []
    seen: set[str] = set()
    previous_start = -1.0
    for segment_id, info in clip_index:
        if segment_id in seen:
            raise ValueError(f"duplicate segment: {segment_id}")
        seen.add(segment_id)
        start, end = _segment_times(segment_id)
        if start < previous_start:
            raise ValueError("segments are not temporally ordered")
        previous_start = start
        frame_values = info.get("files") or info.get("frames") or ()
        if isinstance(frame_values, (str, os.PathLike)):
            frame_values = (frame_values,)
        frames = []
        for frame_path in frame_values:
            path = os.path.abspath(os.fspath(frame_path))
            if not os.path.isfile(path):
                raise ValueError(f"missing sampled frame: {path}")
            frames.append({"path": path, "content_hash": _file_hash(path)})
        if not frames:
            raise ValueError(f"segment has no sampled frames: {segment_id}")
        segment_rows.append({
            "segment_id": segment_id, "start_seconds": start,
            "end_seconds": end,
            "transcript": str(info.get("transcript") or "No transcript."),
            "frames": frames,
        })
    qas = tuple({
        "question_id": question_id,
        "provider_index": provider_index,
        "sample": sample,
    } for question_id, provider_index, sample in zip(
        video.question_ids, video.provider_indices, normalized_samples))
    video_path = os.path.abspath(str(samples[0]["video_path"]))
    videos = [{
        "video_id": video.video_id,
        "video_path": video_path,
        "video_content_hash": (_file_hash(video_path)
                               if os.path.isfile(video_path) else None),
        "question_ids": video.question_ids,
        "provider_indices": video.provider_indices,
        "qas": qas,
        "segments": segment_rows,
    }]
    payload = {
        "schema_version": "checkpoint3e_confirmation_input_bundle_v1",
        "status": "complete",
        "sample_source_identity": evaluator.sample_source_identity,
        "confirmation_video_ids": (video.video_id,),
        "runtime_configuration": runtime,
        "runtime_configuration_hash": sha256_json(runtime),
        "videos": videos,
    }
    bundle = {**payload, "bundle_hash": sha256_json(payload)}
    path = _write_immutable(
        os.path.join(output_dir, "input_bundle.json"), bundle)
    return path, bundle


def _memoized_qa(
    *, evaluator: HistoryAwareDVDConfirmationEvaluator,
    view: Mapping[str, Any], qa: Mapping[str, Any],
    qa_cache_root: str,
) -> Mapping[str, Any]:
    """Run one DVD QA, reusing a prior result keyed by caption identity.

    The QA is a function of the caption content (``captions_hash``) and the
    question, so results are cached under ``qa_cache_root`` and reused whenever
    the same meta-prompt (hence the same captions) is scored again.
    """

    captions_hash = view["captions_hash"]
    cache_dir = os.path.join(
        qa_cache_root, captions_hash, qa["question_id"].replace("/", "-"))
    result_path = os.path.join(cache_dir, "qa_result.json")
    if os.path.isfile(result_path):
        saved = _read_json(result_path)
        if saved.get("question_id") == qa["question_id"] and \
                saved.get("captions_hash") == captions_hash:
            return saved
    try:
        result = evaluator.qa_fn(
            captions_path=view["captions_path"], sample=dict(qa["sample"]),
            run_dir=os.path.join(cache_dir, "run"),
            question_id=qa["question_id"],
            database_path=view["database_path"],
            max_iterations=evaluator.dvd_max_iterations, gpu=evaluator.gpu)
        row = {
            "question_id": qa["question_id"],
            "provider_index": qa["provider_index"],
            "captions_hash": captions_hash,
            "prediction": result.prediction,
            "parsed_answer": result.parsed_answer,
            "ground_truth": result.ground_truth,
            "score": float(result.score),
            "is_correct": float(result.score) > 0.0,
            "errors": tuple(str(item) for item in result.errors),
            "latency_seconds": float(result.latency_seconds),
        }
    except Exception as exc:  # noqa: BLE001 - individual QA must not abort search
        row = {
            "question_id": qa["question_id"],
            "provider_index": qa["provider_index"],
            "captions_hash": captions_hash,
            "prediction": None, "parsed_answer": None, "ground_truth": None,
            "score": None, "is_correct": None,
            "errors": (f"{type(exc).__name__}: {exc}",),
            "latency_seconds": None,
        }
    _write_immutable(result_path, row)
    return row


def caption_and_score_video(
    *, evaluator: HistoryAwareDVDConfirmationEvaluator,
    video: GepaVideoInstance, generator: Any,
    bank: PromptBankSnapshot, router: RouterPolicySnapshot,
    scaffold: ScaffoldPolicySnapshot, contract: ScaffoldContract,
    work_root: str, qa_cache_root: str,
) -> GepaVideoScore:
    """Caption one video under ``generator`` and score its QAs absolutely."""

    work_dir = os.path.join(
        work_root, video.video_id, uuid.uuid4().hex[:12])
    _, bundle = materialize_single_video_bundle(
        evaluator=evaluator, video=video, scaffold=scaffold,
        contract=contract, output_dir=work_dir)
    state = evaluator._caption_state(
        state_name="candidate", bundle=bundle, bank=bank, router=router,
        scaffold=scaffold, contract=contract, output_dir=work_dir,
        free_form_generator=generator)
    view = {item["video_id"]: item for item in state["videos"]}[video.video_id]
    bundle_video = {item["video_id"]: item
                    for item in bundle["videos"]}[video.video_id]
    qa_results = []
    for qa in bundle_video["qas"]:
        row = _memoized_qa(
            evaluator=evaluator, view=view, qa=qa, qa_cache_root=qa_cache_root)
        qa_results.append(GepaQAResult(
            question_id=row["question_id"], provider_index=row["provider_index"],
            is_correct=row["is_correct"], prediction=row["prediction"],
            parsed_answer=row["parsed_answer"], ground_truth=row["ground_truth"],
            score=row["score"], errors=tuple(row["errors"]),
            captions_hash=view["captions_hash"]))
    total = len(qa_results)
    correct = sum(1 for row in qa_results if row.is_correct is True)
    return GepaVideoScore(
        video_id=video.video_id,
        accuracy=correct / total if total else 0.0,
        evaluated_qa_count=total, qa_results=tuple(qa_results),
        captions_hash=view["captions_hash"],
        caption_calls=int(view["caption_calls"]),
        caption_cache_hits=int(view["caption_cache_hits"]))


def generator_meta_prompt_id(text: str) -> str:
    return "gepa_meta_" + sha256_text(text)[:16]
