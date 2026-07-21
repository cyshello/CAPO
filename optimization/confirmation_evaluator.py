"""Concrete history-aware confirmation evaluation for Checkpoint 3E."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from surrogate_rollout import config
from surrogate_rollout.captioning.candidate_captions import build_clip_index
from surrogate_rollout.captioning.history_aware_baseline import (
    CAPTION_OUTPUT_CONTRACT_VERSION,
    CAPTION_PARSE_NORMALIZATION_VERSION,
    CAPTION_PARSE_SCHEMA_VERSION,
    CAPTION_REPETITION_POLICY_VERSION,
    DEFAULT_HISTORY_BLOCK_SECONDS,
    DEFAULT_MAX_HISTORY_CAPTIONS,
    HISTORY_SCHEMA_VERSION,
    HistoryAwareBaselineCaptionViewBuilder,
)
from surrogate_rollout.evaluation.dvd_qa import run_dvd_qa
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.free_form_instruction_generator import (
    free_form_router_version,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    RouterPolicySnapshot,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_json, sha256_text


class ConfirmationEvaluationError(RuntimeError):
    pass


class ConfirmationEvaluationConflictError(ConfirmationEvaluationError):
    pass


_FREE_FORM_GENERATOR_UNSET = object()


@dataclass(frozen=True)
class ConfirmationEvaluationResult:
    manifest_path: str
    input_bundle_path: str
    input_bundle_hash: str
    qas: tuple[Mapping[str, Any], ...]
    resumed: bool = False

    def as_mapping(self) -> Mapping[str, Any]:
        return {
            "schema_version": "history_aware_confirmation_result_v1",
            "status": "completed",
            "manifest_path": self.manifest_path,
            "input_bundle_path": self.input_bundle_path,
            "input_bundle_hash": self.input_bundle_hash,
            "qas": self.qas,
            "proposals_generated": 0,
            "interventions": 0,
            "feedback_calls": 0,
            "resumed": self.resumed,
        }


def _read_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ConfirmationEvaluationError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: str) -> tuple[dict[str, Any], ...]:
    with open(path) as f:
        values = tuple(json.loads(line) for line in f if line.strip())
    if not all(isinstance(value, dict) for value in values):
        raise ConfirmationEvaluationError(f"expected JSONL objects: {path}")
    return values


def _write_immutable(path: str, value: Any) -> str:
    text = json.dumps(
        as_json_dict(value), sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    if os.path.exists(path):
        with open(path) as f:
            if f.read() != text:
                raise ConfirmationEvaluationConflictError(
                    f"immutable confirmation artifact conflict: {path}")
        return os.path.abspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, text)
    return os.path.abspath(path)


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _callable_identity(value: Any) -> str:
    return f"{value.__module__}.{getattr(value, '__qualname__', type(value).__qualname__)}"


def _router_adapter_identity(value: Any) -> Mapping[str, Any]:
    configured = getattr(value, "configuration_identity", None)
    if configured is not None:
        return as_json_dict(configured)
    identity = {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
    for name in ("policy_type", "max_selected_properties", "max_tokens"):
        if hasattr(value, name):
            identity[name] = as_json_dict(getattr(value, name))
    vlm = getattr(value, "vlm", None)
    if vlm is not None:
        identity["vlm_type"] = f"{type(vlm).__module__}.{type(vlm).__qualname__}"
    return identity


def _segment_times(segment_id: str) -> tuple[float, float]:
    try:
        start, end = (float(value) for value in segment_id.split("_", 1))
    except (TypeError, ValueError) as exc:
        raise ConfirmationEvaluationError(
            f"invalid confirmation segment ID: {segment_id!r}") from exc
    if start < 0 or end <= start:
        raise ConfirmationEvaluationError(
            f"invalid confirmation segment interval: {segment_id!r}")
    return start, end


def _transition(before: bool, after: bool) -> str:
    return {
        (False, False): "wrong_to_wrong",
        (False, True): "wrong_to_correct",
        (True, False): "correct_to_wrong",
        (True, True): "correct_to_correct",
    }[(before, after)]


class HistoryAwareDVDConfirmationEvaluator:
    """Evaluate parent and candidate on one immutable two-video input bundle."""

    policy_version = "history_aware_dvd_confirmation_v1"

    def __init__(
        self, *, sample_loader: Callable[[int], Mapping[str, Any]],
        history_aware_builder: HistoryAwareBaselineCaptionViewBuilder,
        base_prompt_template: str, merge_prompt: str,
        sample_source_identity: str,
        cache_root: str, cache_manifest_path: str,
        clip_index_fn: Callable[..., list[tuple[str, dict]]] = build_clip_index,
        qa_fn: Callable[..., Any] = run_dvd_qa,
        history_block_seconds: float = DEFAULT_HISTORY_BLOCK_SECONDS,
        max_history_captions: int = DEFAULT_MAX_HISTORY_CAPTIONS,
        frame_sampling_configuration: Mapping[str, Any] | None = None,
        downstream_qa_configuration: Mapping[str, Any] | None = None,
        dvd_max_iterations: int = config.MAX_ITERATIONS,
        gpu: str | None = None,
    ) -> None:
        if not sample_source_identity:
            raise ValueError("sample_source_identity must be non-empty")
        if history_block_seconds <= 0 or max_history_captions < 1:
            raise ValueError("invalid confirmation history configuration")
        if dvd_max_iterations < 1:
            raise ValueError("dvd_max_iterations must be positive")
        self.sample_loader = sample_loader
        self.builder = history_aware_builder
        self.clip_index_fn = clip_index_fn
        self.qa_fn = qa_fn
        self.base_prompt_template = base_prompt_template
        self.merge_prompt = merge_prompt
        self.sample_source_identity = sample_source_identity
        self.cache_root = os.path.abspath(cache_root)
        self.cache_manifest_path = os.path.abspath(cache_manifest_path)
        self.history_block_seconds = float(history_block_seconds)
        self.max_history_captions = int(max_history_captions)
        self.dvd_max_iterations = int(dvd_max_iterations)
        self.gpu = gpu
        self._caption_model_id = self.builder.segment_captioner.caption_model_id
        self._caption_backend_id = self.builder.segment_captioner.backend_id
        self._router_identity = _router_adapter_identity(self.builder.router)
        self._caption_decoding = as_json_dict(config.CAPTION_DECODING)
        self._frame_sampling = as_json_dict(frame_sampling_configuration or {
            "sample_fps": config.SAMPLE_FPS,
            "clip_seconds": config.CLIP_SECS,
            "segment_boundary_rule": "dvd_build_clips_fixed_duration",
            "frame_order_rule": "sorted_decoded_frame_paths",
        })
        if float(self._frame_sampling.get("sample_fps", -1)) != \
                float(config.SAMPLE_FPS) or \
                float(self._frame_sampling.get("clip_seconds", -1)) != \
                float(config.CLIP_SECS):
            raise ValueError(
                "frame sampling configuration must match active DVD sampling")
        self._dvd_runtime_defaults = {
            "orchestrator_tool_model": config.ORCHESTRATOR_TOOL_MODEL,
            "text_fallback_model": config.TEXT_FALLBACK_MODEL,
            "tool_vlm_max_frames": config.TOOL_VLM_MAX_FRAMES,
            "use_transcript": config.USE_TRANSCRIPT,
            "clip_search_top_k": config.DVD_CLIP_SEARCH_TOP_K,
            "clip_search_policy_version": config.DVD_CLIP_SEARCH_POLICY_VERSION,
            "frame_inspect_tool_contract_version":
                config.DVD_FRAME_INSPECT_TOOL_CONTRACT_VERSION,
            "frame_inspect_corrective_retry_limit":
                config.DVD_FRAME_INSPECT_CORRECTIVE_RETRY_LIMIT,
        }
        downstream_extra = as_json_dict(downstream_qa_configuration or {})
        conflicting = {
            key for key in self._dvd_runtime_defaults
            if key in downstream_extra and
            downstream_extra[key] != self._dvd_runtime_defaults[key]
        }
        if conflicting:
            raise ValueError(
                "downstream QA configuration conflicts with active DVD settings: "
                f"{sorted(conflicting)}")
        self._qa_configuration = {
            "qa_fn": _callable_identity(qa_fn),
            "dvd_max_iterations": self.dvd_max_iterations,
            "gpu": gpu,
            **self._dvd_runtime_defaults,
            **downstream_extra,
        }

    @classmethod
    def from_local_qwen(cls, **kwargs: Any) -> "HistoryAwareDVDConfirmationEvaluator":
        return cls(
            history_aware_builder=
            HistoryAwareBaselineCaptionViewBuilder.from_local_qwen(),
            **kwargs)

    @property
    def configuration_identity(self) -> Mapping[str, Any]:
        return {
            "policy_version": self.policy_version,
            "sample_source_identity": self.sample_source_identity,
            "clip_index_fn": _callable_identity(self.clip_index_fn),
            "base_prompt_hash": sha256_text(self.base_prompt_template),
            "merge_prompt_hash": sha256_text(self.merge_prompt),
            "caption_model_id": self._caption_model_id,
            "caption_backend_id": self._caption_backend_id,
            "router_adapter": self._router_identity,
            "caption_decoding": self._caption_decoding,
            "caption_output_contract_version":
                CAPTION_OUTPUT_CONTRACT_VERSION,
            "caption_decoding_policy_version":
                config.CAPTION_DECODING_POLICY_VERSION,
            "caption_parse_schema_version": CAPTION_PARSE_SCHEMA_VERSION,
            "caption_parse_normalization_version":
                CAPTION_PARSE_NORMALIZATION_VERSION,
            "caption_repetition_policy_version":
                CAPTION_REPETITION_POLICY_VERSION,
            "frame_sampling": self._frame_sampling,
            "history": {
                "schema_version": HISTORY_SCHEMA_VERSION,
                "block_seconds": self.history_block_seconds,
                "max_history_captions": self.max_history_captions,
                "boundary_rule": "floor_segment_start_div_block_seconds",
            },
            "downstream_qa": self._qa_configuration,
            "cache_root": self.cache_root,
            "cache_manifest_path": self.cache_manifest_path,
        }

    def _validate_live_runtime(self) -> None:
        if self.builder.segment_captioner.caption_model_id != \
                self._caption_model_id:
            raise ConfirmationEvaluationConflictError(
                "caption model changed after confirmation configuration freeze")
        if self.builder.segment_captioner.backend_id != self._caption_backend_id:
            raise ConfirmationEvaluationConflictError(
                "caption backend changed after confirmation configuration freeze")
        if _router_adapter_identity(self.builder.router) != self._router_identity:
            raise ConfirmationEvaluationConflictError(
                "router adapter changed after confirmation configuration freeze")
        if as_json_dict(config.CAPTION_DECODING) != self._caption_decoding:
            raise ConfirmationEvaluationConflictError(
                "caption decoding changed after confirmation configuration freeze")
        if float(config.SAMPLE_FPS) != \
                float(self._frame_sampling["sample_fps"]) or \
                float(config.CLIP_SECS) != \
                float(self._frame_sampling["clip_seconds"]):
            raise ConfirmationEvaluationConflictError(
                "frame sampling changed after confirmation configuration freeze")
        live_dvd = {
            "orchestrator_tool_model": config.ORCHESTRATOR_TOOL_MODEL,
            "text_fallback_model": config.TEXT_FALLBACK_MODEL,
            "tool_vlm_max_frames": config.TOOL_VLM_MAX_FRAMES,
            "use_transcript": config.USE_TRANSCRIPT,
            "clip_search_top_k": config.DVD_CLIP_SEARCH_TOP_K,
            "clip_search_policy_version": config.DVD_CLIP_SEARCH_POLICY_VERSION,
            "frame_inspect_tool_contract_version":
                config.DVD_FRAME_INSPECT_TOOL_CONTRACT_VERSION,
            "frame_inspect_corrective_retry_limit":
                config.DVD_FRAME_INSPECT_CORRECTIVE_RETRY_LIMIT,
        }
        if live_dvd != self._dvd_runtime_defaults:
            raise ConfirmationEvaluationConflictError(
                "downstream DVD configuration changed after freeze")

    def _runtime_configuration(
        self, scaffold: ScaffoldPolicySnapshot,
        contract: ScaffoldContract,
    ) -> Mapping[str, Any]:
        return {
            **self.configuration_identity,
            "base_prompt": self.base_prompt_template,
            "merge_prompt": self.merge_prompt,
            "frame_sampling_identity": sha256_json(self._frame_sampling),
            "caption_model_hash": sha256_text(self._caption_model_id),
            "caption_backend_hash": sha256_text(self._caption_backend_id),
            "router_adapter_hash": sha256_json(self._router_identity),
            "caption_decoding_hash": sha256_json(self._caption_decoding),
            "history_configuration_hash": sha256_json(
                self.configuration_identity["history"]),
            "downstream_qa_configuration_hash": sha256_json(
                self._qa_configuration),
            "scaffold_version": scaffold.scaffold_version,
            "contract_version": contract.contract_version,
        }

    def _materialize_bundle(
        self, *, confirmation_videos: tuple[Any, ...],
        scaffold: ScaffoldPolicySnapshot, contract: ScaffoldContract,
        output_dir: str,
    ) -> tuple[str, Mapping[str, Any]]:
        if len(confirmation_videos) != 2 or len({
                item.video_id for item in confirmation_videos}) != 2:
            raise ConfirmationEvaluationError(
                "confirmation requires exactly two unique videos")
        runtime = self._runtime_configuration(scaffold, contract)
        videos = []
        for record in confirmation_videos:
            if len(record.provider_indices) != 3 or len(record.question_ids) != 3:
                raise ConfirmationEvaluationError(
                    "each confirmation video requires exactly three QAs")
            samples = tuple(self.sample_loader(index)
                            for index in record.provider_indices)
            normalized_samples = tuple(as_json_dict(sample) for sample in samples)
            observed_ids = tuple(str(
                sample.get("extra", {}).get("videoID") or sample.get("sample_id"))
                for sample in samples)
            if set(observed_ids) != {record.video_id}:
                raise ConfirmationEvaluationError(
                    f"confirmation sample/video mismatch: {record.video_id}")
            clip_index = self.clip_index_fn(dict(samples[0]), record.video_id)
            segment_rows = []
            seen = set()
            previous_start = -1.0
            for segment_id, info in clip_index:
                if segment_id in seen:
                    raise ConfirmationEvaluationError(
                        f"duplicate confirmation segment: {segment_id}")
                seen.add(segment_id)
                start, end = _segment_times(segment_id)
                if start < previous_start:
                    raise ConfirmationEvaluationError(
                        "confirmation segments are not temporally ordered")
                previous_start = start
                frame_values = info.get("files") or info.get("frames") or ()
                if isinstance(frame_values, (str, os.PathLike)):
                    frame_values = (frame_values,)
                frames = []
                for frame_path in frame_values:
                    path = os.path.abspath(os.fspath(frame_path))
                    if not os.path.isfile(path):
                        raise ConfirmationEvaluationError(
                            f"missing sampled confirmation frame: {path}")
                    frames.append({"path": path, "content_hash": _file_hash(path)})
                if not frames:
                    raise ConfirmationEvaluationError(
                        f"confirmation segment has no sampled frames: {segment_id}")
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
                record.question_ids, record.provider_indices, normalized_samples))
            video_path = os.path.abspath(str(samples[0]["video_path"]))
            videos.append({
                "video_id": record.video_id,
                "video_path": video_path,
                "video_content_hash": (_file_hash(video_path)
                                       if os.path.isfile(video_path) else None),
                "question_ids": record.question_ids,
                "provider_indices": record.provider_indices,
                "qas": qas,
                "segments": segment_rows,
            })
        payload = {
            "schema_version": "checkpoint3e_confirmation_input_bundle_v1",
            "status": "complete",
            "sample_source_identity": self.sample_source_identity,
            "confirmation_video_ids": tuple(
                item.video_id for item in confirmation_videos),
            "runtime_configuration": runtime,
            "runtime_configuration_hash": sha256_json(runtime),
            "videos": videos,
        }
        bundle = {**payload, "bundle_hash": sha256_json(payload)}
        path = _write_immutable(
            os.path.join(output_dir, "input_bundle.json"), bundle)
        return path, bundle

    def _validate_bundle(self, bundle: Mapping[str, Any]) -> None:
        payload = {key: value for key, value in bundle.items()
                   if key != "bundle_hash"}
        if bundle.get("bundle_hash") != sha256_json(payload):
            raise ConfirmationEvaluationConflictError(
                "confirmation input bundle hash mismatch")
        if bundle.get("runtime_configuration_hash") != sha256_json(
                bundle.get("runtime_configuration") or {}):
            raise ConfirmationEvaluationConflictError(
                "confirmation runtime configuration hash mismatch")
        for video in bundle.get("videos") or ():
            for segment in video.get("segments") or ():
                for frame in segment.get("frames") or ():
                    path = frame.get("path")
                    if not path or not os.path.isfile(path) or \
                            _file_hash(path) != frame.get("content_hash"):
                        raise ConfirmationEvaluationConflictError(
                            "sampled confirmation frame content changed")

    @staticmethod
    def _clip_index(video: Mapping[str, Any]) -> list[tuple[str, dict]]:
        return [(segment["segment_id"], {
            "files": tuple(frame["path"] for frame in segment["frames"]),
            "transcript": segment["transcript"],
        }) for segment in video["segments"]]

    def _caption_state(
        self, *, state_name: str, bundle: Mapping[str, Any],
        bank: PromptBankSnapshot, router: RouterPolicySnapshot,
        scaffold: ScaffoldPolicySnapshot, contract: ScaffoldContract,
        output_dir: str,
        free_form_generator: Any = _FREE_FORM_GENERATOR_UNSET,
    ) -> Mapping[str, Any]:
        self._validate_live_runtime()
        self._validate_bundle(bundle)
        videos = []
        for video in bundle["videos"]:
            work_root = os.path.join(
                output_dir, state_name, "videos", video["video_id"])
            previous_generator = self.builder.free_form_generator
            active_generator = (
                previous_generator if free_form_generator is
                _FREE_FORM_GENERATOR_UNSET else free_form_generator)
            self.builder.free_form_generator = active_generator
            try:
                view = self.builder.build(
                    sample=dict(video["qas"][0]["sample"]),
                    clip_index=self._clip_index(video),
                    prompt_bank=bank, router_policy=router,
                    scaffold_policy=scaffold, scaffold_contract=contract,
                    base_prompt_template=self.base_prompt_template,
                    merge_prompt=self.merge_prompt, work_root=work_root,
                    history_block_seconds=self.history_block_seconds,
                    max_history_captions=self.max_history_captions,
                    candidate_cache_root=self.cache_root,
                    cache_manifest_path=self.cache_manifest_path,
                    caption_run_identity_hash=bundle["bundle_hash"])
            finally:
                self.builder.free_form_generator = previous_generator
            routing = _read_json(view.routing_manifest_path)
            cache_rows = _read_jsonl(routing["caption_cache_keys_path"])
            histories = _read_jsonl(routing["frozen_histories_path"])
            composed = _read_jsonl(routing["composed_prompts_path"])
            if tuple(row["segment_id"] for row in cache_rows) != \
                    tuple(view.segment_ids) or \
                    tuple(row["segment_id"] for row in histories) != \
                    tuple(view.segment_ids) or \
                    tuple(row["segment_id"] for row in composed) != \
                    tuple(view.segment_ids):
                raise ConfirmationEvaluationError(
                    "confirmation caption artifacts have inconsistent segments")
            segments = []
            for cache_row, history, prompt in zip(
                    cache_rows, histories, composed):
                key = cache_row.get("cache_key") or {}
                required = (
                    "video_id", "segment_id", "prompt_hash",
                    "caption_model_id", "decoding_hash", "source_hash",
                    "history_hash", "composed_prompt_hash", "bank_version",
                    "router_version", "scaffold_version", "contract_version",
                    "backend_id", "history_config_hash",
                )
                if any(not key.get(name) for name in required):
                    raise ConfirmationEvaluationError(
                        "incomplete history-aware confirmation cache identity")
                expected_router_version = (
                    free_form_router_version(active_generator.identity_hash)
                    if active_generator is not None else
                    router.router_version)
                if key["bank_version"] != bank.bank_version or \
                        key["router_version"] != expected_router_version or \
                        key["scaffold_version"] != scaffold.scaffold_version or \
                        key["contract_version"] != contract.contract_version or \
                        key["caption_model_id"] != self._caption_model_id or \
                        key["backend_id"] != self._caption_backend_id or \
                        key["decoding_hash"] != sha256_json(self._caption_decoding) or \
                        key.get("intervention_identity_hash") != \
                        bundle["bundle_hash"]:
                    raise ConfirmationEvaluationError(
                        "confirmation caption cache identity/configuration mismatch")
                segments.append({
                    "segment_id": cache_row["segment_id"],
                    "cache_identity": key,
                    "cache_identity_hash": sha256_json(key),
                    "cache_dir": cache_row["cache_dir"],
                    "caption_result_path": cache_row["result_path"],
                    "cache_hit": bool(cache_row["cache_hit"]),
                    "history_hash": history["history_hash"],
                    "preceding_segment_ids": history["preceding_segment_ids"],
                    "preceding_captions": history["history"],
                    "composed_prompt_hash": prompt["prompt_hash"],
                })
            view_record = _read_json(view.routed_view_path)
            videos.append({
                "video_id": video["video_id"],
                "input_bundle_hash": bundle["bundle_hash"],
                "runtime_configuration_hash":
                    bundle["runtime_configuration_hash"],
                "caption_view_hash": view_record["view_hash"],
                "captions_path": view.captions_path,
                "captions_hash": view.captions_hash,
                "database_path": view.database_path,
                "routing_manifest_path": view.routing_manifest_path,
                "caption_calls": view.caption_call_count,
                "caption_cache_hits": view.caption_cache_hits,
                "segments": segments,
            })
        state = {
            "schema_version": "checkpoint3e_confirmation_caption_state_v1",
            "state_name": state_name,
            "input_bundle_hash": bundle["bundle_hash"],
            "runtime_configuration_hash": bundle["runtime_configuration_hash"],
            "bank_version": bank.bank_version,
            "bank_hash": sha256_json(as_json_dict(bank)),
            "router_version": router.router_version,
            "router_hash": sha256_json(as_json_dict(router)),
            "prompt_generator": (
                as_json_dict(active_generator.configuration_identity)
                if active_generator is not None else None),
            "videos": videos,
        }
        path = _write_immutable(
            os.path.join(output_dir, state_name, "caption_state.json"), state)
        return {**state, "artifact_path": path}

    @staticmethod
    def _validate_paired_states(
        parent: Mapping[str, Any], candidate: Mapping[str, Any],
        bundle: Mapping[str, Any],
    ) -> None:
        expected = bundle["bundle_hash"]
        runtime = bundle["runtime_configuration_hash"]
        if parent.get("input_bundle_hash") != expected or \
                candidate.get("input_bundle_hash") != expected or \
                parent.get("runtime_configuration_hash") != runtime or \
                candidate.get("runtime_configuration_hash") != runtime:
            raise ConfirmationEvaluationConflictError(
                "parent/candidate confirmation inputs or runtime differ")
        parent_videos = {item["video_id"]: item for item in parent["videos"]}
        candidate_videos = {item["video_id"]: item
                            for item in candidate["videos"]}
        if set(parent_videos) != set(candidate_videos):
            raise ConfirmationEvaluationConflictError(
                "parent/candidate confirmation videos differ")
        for video_id in parent_videos:
            before = {item["segment_id"]: item
                      for item in parent_videos[video_id]["segments"]}
            after = {item["segment_id"]: item
                     for item in candidate_videos[video_id]["segments"]}
            if set(before) != set(after):
                raise ConfirmationEvaluationConflictError(
                    "parent/candidate confirmation segments differ")
            for segment_id in before:
                left, right = before[segment_id], after[segment_id]
                same_identity = left["cache_identity"] == right["cache_identity"]
                same_path = os.path.abspath(left["caption_result_path"]) == \
                    os.path.abspath(right["caption_result_path"])
                if same_identity != same_path:
                    raise ConfirmationEvaluationConflictError(
                        "caption cache path does not match complete identity equality")

    def _run_qas(
        self, *, state_name: str, bundle: Mapping[str, Any],
        state: Mapping[str, Any], output_dir: str,
    ) -> tuple[Mapping[str, Any], ...]:
        by_video = {item["video_id"]: item for item in state["videos"]}
        rows = []
        for video in bundle["videos"]:
            view = by_video[video["video_id"]]
            for qa in video["qas"]:
                run_dir = os.path.join(
                    output_dir, state_name, "qa",
                    qa["question_id"].replace("/", "-"))
                result = self.qa_fn(
                    captions_path=view["captions_path"],
                    sample=dict(qa["sample"]), run_dir=run_dir,
                    question_id=qa["question_id"],
                    database_path=view["database_path"],
                    max_iterations=self.dvd_max_iterations, gpu=self.gpu)
                row = {
                    "state": state_name, "video_id": video["video_id"],
                    "question_id": qa["question_id"],
                    "provider_index": qa["provider_index"],
                    "prediction": result.prediction,
                    "parsed_answer": result.parsed_answer,
                    "ground_truth": result.ground_truth,
                    "score": float(result.score),
                    "is_correct": float(result.score) > 0.0,
                    "errors": tuple(result.errors),
                    "latency_seconds": float(result.latency_seconds),
                    "captions_path": view["captions_path"],
                    "caption_view_hash": view["caption_view_hash"],
                }
                rows.append(row)
        return tuple(rows)

    def evaluate(
        self, *, confirmation_videos: tuple[Any, ...],
        incumbent_bank: PromptBankSnapshot,
        incumbent_router: RouterPolicySnapshot,
        candidate_bank: PromptBankSnapshot,
        candidate_router: RouterPolicySnapshot,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        output_dir: str,
    ) -> Mapping[str, Any]:
        output_dir = os.path.abspath(output_dir)
        evaluator_dir = os.path.join(output_dir, "history_aware_evaluator")
        manifest_path = os.path.join(evaluator_dir, "manifest.json")
        self._validate_live_runtime()
        runtime = self._runtime_configuration(scaffold_policy, scaffold_contract)
        requested_identity = {
            "policy_version": self.policy_version,
            "confirmation_videos": confirmation_videos,
            "incumbent_bank": incumbent_bank,
            "incumbent_router": incumbent_router,
            "candidate_bank": candidate_bank,
            "candidate_router": candidate_router,
            "scaffold_policy": scaffold_policy,
            "scaffold_contract": scaffold_contract,
            "runtime_configuration": runtime,
        }
        request_hash = sha256_text(dumps_canonical(requested_identity))
        if os.path.exists(manifest_path):
            manifest = _read_json(manifest_path)
            if manifest.get("status") != "completed" or \
                    manifest.get("request_hash") != request_hash:
                raise ConfirmationEvaluationConflictError(
                    "completed confirmation belongs to different frozen inputs")
            bundle_path = manifest.get("input_bundle_path")
            if not bundle_path or not os.path.exists(bundle_path) or \
                    manifest.get("input_bundle_file_hash") != \
                    _file_hash(bundle_path):
                raise ConfirmationEvaluationConflictError(
                    "completed confirmation input bundle is missing or changed")
            self._validate_bundle(_read_json(bundle_path))
            return ConfirmationEvaluationResult(
                manifest_path=os.path.abspath(manifest_path),
                input_bundle_path=bundle_path,
                input_bundle_hash=manifest["input_bundle_hash"],
                qas=tuple(manifest["qas"]), resumed=True).as_mapping()

        bundle_path, bundle = self._materialize_bundle(
            confirmation_videos=confirmation_videos,
            scaffold=scaffold_policy, contract=scaffold_contract,
            output_dir=evaluator_dir)
        parent = self._caption_state(
            state_name="parent", bundle=bundle,
            bank=incumbent_bank, router=incumbent_router,
            scaffold=scaffold_policy, contract=scaffold_contract,
            output_dir=evaluator_dir)
        candidate = self._caption_state(
            state_name="candidate", bundle=bundle,
            bank=candidate_bank, router=candidate_router,
            scaffold=scaffold_policy, contract=scaffold_contract,
            output_dir=evaluator_dir)
        self._validate_live_runtime()
        self._validate_bundle(bundle)
        self._validate_paired_states(parent, candidate, bundle)
        parent_qas = self._run_qas(
            state_name="parent", bundle=bundle, state=parent,
            output_dir=evaluator_dir)
        candidate_qas = self._run_qas(
            state_name="candidate", bundle=bundle, state=candidate,
            output_dir=evaluator_dir)
        before = {row["question_id"]: row for row in parent_qas}
        after = {row["question_id"]: row for row in candidate_qas}
        expected_ids = tuple(
            qa["question_id"] for video in bundle["videos"] for qa in video["qas"])
        if set(before) != set(expected_ids) or set(after) != set(expected_ids):
            raise ConfirmationEvaluationError(
                "confirmation did not evaluate all six paired QAs")
        paired = tuple({
            "video_id": before[question_id]["video_id"],
            "question_id": question_id,
            "provider_index": before[question_id]["provider_index"],
            "incumbent_prediction": before[question_id]["prediction"],
            "candidate_prediction": after[question_id]["prediction"],
            "incumbent_score": before[question_id]["score"],
            "candidate_score": after[question_id]["score"],
            "transition": _transition(
                before[question_id]["is_correct"],
                after[question_id]["is_correct"]),
            "incumbent_errors": before[question_id]["errors"],
            "candidate_errors": after[question_id]["errors"],
            "error": (before[question_id]["errors"] or
                      after[question_id]["errors"] or None),
            "parent_caption_view_hash":
                before[question_id]["caption_view_hash"],
            "candidate_caption_view_hash":
                after[question_id]["caption_view_hash"],
        } for question_id in expected_ids)
        regressions = tuple(row["question_id"] for row in paired
                            if row["transition"] == "correct_to_wrong")
        parent_accuracy = sum(row["incumbent_score"] for row in paired) / 6
        candidate_accuracy = sum(row["candidate_score"] for row in paired) / 6
        errors = tuple(row["question_id"] for row in paired if row["error"])
        criterion = {
            "all_six_qas_present": len(paired) == 6,
            "all_evaluations_error_free": not errors,
            "zero_correct_to_wrong": not regressions,
            "candidate_mean_accuracy_not_lower":
                candidate_accuracy >= parent_accuracy,
            "accepted": not errors and not regressions and
                        candidate_accuracy >= parent_accuracy,
        }
        manifest = {
            "schema_version": "history_aware_dvd_confirmation_manifest_v1",
            "status": "completed", "request_hash": request_hash,
            "input_bundle_path": bundle_path,
            "input_bundle_file_hash": _file_hash(bundle_path),
            "input_bundle_hash": bundle["bundle_hash"],
            "runtime_configuration": runtime,
            "runtime_configuration_hash": bundle["runtime_configuration_hash"],
            "parent": parent, "candidate": candidate,
            "component_versions": {
                "parent_bank": incumbent_bank.bank_version,
                "parent_router": incumbent_router.router_version,
                "candidate_bank": candidate_bank.bank_version,
                "candidate_router": candidate_router.router_version,
                "scaffold": scaffold_policy.scaffold_version,
                "contract": scaffold_contract.contract_version,
            },
            "component_hashes": {
                "parent_bank": sha256_json(as_json_dict(incumbent_bank)),
                "parent_router": sha256_json(as_json_dict(incumbent_router)),
                "candidate_bank": sha256_json(as_json_dict(candidate_bank)),
                "candidate_router": sha256_json(as_json_dict(candidate_router)),
                "scaffold": sha256_json(as_json_dict(scaffold_policy)),
                "contract": sha256_json(as_json_dict(scaffold_contract)),
            },
            "parent_qas": parent_qas, "candidate_qas": candidate_qas,
            "qas": paired, "criterion": criterion,
            "parent_accuracy": parent_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "correct_to_wrong_question_ids": regressions,
            "proposals_generated": 0, "interventions": 0,
            "feedback_calls": 0,
        }
        completed_path = _write_immutable(manifest_path, manifest)
        return ConfirmationEvaluationResult(
            manifest_path=completed_path, input_bundle_path=bundle_path,
            input_bundle_hash=bundle["bundle_hash"], qas=paired).as_mapping()
