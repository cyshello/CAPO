"""Real DVD paired confirmation adapter for prompt-delta meta-prompts.

This module is the Checkpoint G bridge only.  It reuses the existing
history-aware caption builder and downstream ``run_dvd_qa`` evaluator while
injecting parent/candidate text exclusively into the free-form prompt
generator.  It never routes legacy properties and never promotes state itself.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from surrogate_rollout.optimization.confirmation_evaluator import (
    ConfirmationEvaluationConflictError,
    ConfirmationEvaluationError,
    HistoryAwareDVDConfirmationEvaluator,
    _file_hash,
    _read_json,
    _write_immutable,
)
from surrogate_rollout.optimization.prompt_delta_iteration import (
    MetaPromptConfirmationOutcome,
    MetaPromptConfirmationRequest,
    PairedMetaPromptConfirmationResult,
)
from surrogate_rollout.optimization.schemas import MetaPromptVersion
from surrogate_rollout.prompt_routing.free_form_instruction_generator import (
    VLMFreeFormInstructionGenerator,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    RouterPolicySnapshot,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_json


@dataclass(frozen=True)
class DVDMetaPromptConfirmationVideo:
    video_id: str
    provider_indices: tuple[int, ...]
    question_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.video_id or len(self.provider_indices) != 3 or \
                len(self.question_ids) != 3:
            raise ValueError(
                "confirmation video requires an ID and exactly three QAs")
        if len(set(self.provider_indices)) != 3 or \
                len(set(self.question_ids)) != 3:
            raise ValueError("confirmation video QA identities must be unique")


def _transition(before: bool, after: bool) -> str:
    return {
        (False, False): "wrong_to_wrong",
        (False, True): "wrong_to_correct",
        (True, False): "correct_to_wrong",
        (True, True): "correct_to_correct",
    }[(before, after)]


def _error_text(errors: Sequence[Any]) -> str | None:
    return "; ".join(str(item) for item in errors) if errors else None


class DVDMetaPromptConfirmationEvaluator:
    """Adapt the actual DVD runtime to the Checkpoint F paired protocol."""

    policy_version = "dvd_meta_prompt_confirmation_v1"

    def __init__(
        self, *, evaluator: HistoryAwareDVDConfirmationEvaluator,
        confirmation_videos: Sequence[DVDMetaPromptConfirmationVideo],
        compatibility_bank: PromptBankSnapshot,
        compatibility_router: RouterPolicySnapshot,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        prompt_generator_model_id: str,
        prompt_generator_backend_id: str,
        prompt_generator_max_tokens: int,
        model_identity: str,
        decoding_settings: Mapping[str, Any],
        cache_reset_identity: str,
        evaluation_pipeline_identity: str,
    ) -> None:
        self.evaluator = evaluator
        self.confirmation_videos = tuple(confirmation_videos)
        if len(self.confirmation_videos) != 2 or len({
                item.video_id for item in self.confirmation_videos}) != 2:
            raise ValueError(
                "active confirmation policy requires exactly two videos")
        self.bank = compatibility_bank
        self.router = compatibility_router
        self.scaffold = scaffold_policy
        self.contract = scaffold_contract
        for name, value in (
                ("prompt_generator_model_id", prompt_generator_model_id),
                ("prompt_generator_backend_id", prompt_generator_backend_id),
                ("model_identity", model_identity),
                ("cache_reset_identity", cache_reset_identity),
                ("evaluation_pipeline_identity", evaluation_pipeline_identity)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be explicitly configured")
        if not isinstance(prompt_generator_max_tokens, int) or \
                isinstance(prompt_generator_max_tokens, bool) or \
                prompt_generator_max_tokens <= 0:
            raise ValueError("prompt_generator_max_tokens must be positive")
        if not isinstance(decoding_settings, Mapping) or not decoding_settings:
            raise ValueError("decoding_settings must be explicitly configured")
        self.prompt_generator_model_id = prompt_generator_model_id
        self.prompt_generator_backend_id = prompt_generator_backend_id
        self.prompt_generator_max_tokens = prompt_generator_max_tokens
        self.model_identity = model_identity
        self.decoding_settings = json.loads(dumps_canonical(decoding_settings))
        self.cache_reset_identity = cache_reset_identity
        self.evaluation_pipeline_identity = evaluation_pipeline_identity
        self.call_count = 0

    @property
    def configuration_identity(self) -> Mapping[str, Any]:
        return {
            "policy_version": self.policy_version,
            "model_identity": self.model_identity,
            "decoding_settings": self.decoding_settings,
            "cache_reset_identity": self.cache_reset_identity,
            "evaluation_pipeline_identity": self.evaluation_pipeline_identity,
            "prompt_generator": {
                "model": self.prompt_generator_model_id,
                "backend": self.prompt_generator_backend_id,
                "max_tokens": self.prompt_generator_max_tokens,
            },
            "dvd_runtime": self.evaluator.configuration_identity,
            "confirmation_videos": self.confirmation_videos,
            "legacy_property_routing_used": False,
        }

    def _validate_request(self, request: MetaPromptConfirmationRequest) -> None:
        expected_cases = tuple(
            (video.video_id, qa_id, provider_index)
            for video in self.confirmation_videos
            for qa_id, provider_index in zip(
                video.question_ids, video.provider_indices))
        observed = tuple((case.video_id, case.qa_id) for case in request.cases)
        if observed != tuple((video_id, qa_id)
                             for video_id, qa_id, _ in expected_cases):
            raise ConfirmationEvaluationConflictError(
                "confirmation request does not match the frozen two-video/six-QA manifest")
        if request.model_identity != self.model_identity or \
                dumps_canonical(request.decoding_settings) != \
                dumps_canonical(self.decoding_settings) or \
                request.cache_reset_identity != self.cache_reset_identity or \
                request.evaluation_pipeline_identity != \
                self.evaluation_pipeline_identity:
            raise ConfirmationEvaluationConflictError(
                "confirmation request/runtime identity mismatch")

    def _generator(self, version: MetaPromptVersion) -> VLMFreeFormInstructionGenerator:
        return VLMFreeFormInstructionGenerator(
            self.evaluator.builder.router.vlm,
            max_tokens=self.prompt_generator_max_tokens,
            template_text=version.text,
            meta_prompt_id=version.meta_prompt_id,
            model_id=self.prompt_generator_model_id,
            backend_id=self.prompt_generator_backend_id)

    def _run_qas(
        self, *, state_name: str, bundle: Mapping[str, Any],
        state: Mapping[str, Any], output_directory: str,
    ) -> tuple[Mapping[str, Any], ...]:
        by_video = {item["video_id"]: item for item in state["videos"]}
        rows = []
        for video in bundle["videos"]:
            view = by_video[video["video_id"]]
            for qa in video["qas"]:
                run_dir = os.path.join(
                    output_directory, state_name, "qa",
                    qa["question_id"].replace("/", "-"))
                result_path = os.path.join(run_dir, "paired_qa_result.json")
                if os.path.isfile(result_path):
                    saved = _read_json(result_path)
                    if saved.get("question_id") != qa["question_id"] or \
                            saved.get("video_id") != video["video_id"] or \
                            saved.get("captions_hash") != view["captions_hash"]:
                        raise ConfirmationEvaluationConflictError(
                            "completed paired QA belongs to different inputs")
                    rows.append(saved)
                    continue
                try:
                    result = self.evaluator.qa_fn(
                        captions_path=view["captions_path"],
                        sample=dict(qa["sample"]), run_dir=run_dir,
                        question_id=qa["question_id"],
                        database_path=view["database_path"],
                        max_iterations=self.evaluator.dvd_max_iterations,
                        gpu=self.evaluator.gpu)
                    errors = tuple(str(item) for item in result.errors)
                    row = {
                        "prediction": result.prediction,
                        "parsed_answer": result.parsed_answer,
                        "ground_truth": result.ground_truth,
                        "score": float(result.score),
                        "is_correct": float(result.score) > 0.0,
                        "errors": errors,
                        "latency_seconds": float(result.latency_seconds),
                    }
                except Exception as exc:
                    row = {
                        "prediction": None, "parsed_answer": None,
                        "ground_truth": None, "score": None,
                        "is_correct": None,
                        "errors": (f"{type(exc).__name__}: {exc}",),
                        "latency_seconds": None,
                    }
                persisted = {
                    "state": state_name, "video_id": video["video_id"],
                    "question_id": qa["question_id"],
                    "provider_index": qa["provider_index"],
                    "captions_path": view["captions_path"],
                    "captions_hash": view["captions_hash"],
                    "routing_manifest_path": view["routing_manifest_path"],
                    "caption_view_hash": view["caption_view_hash"],
                    "caption_calls": view["caption_calls"],
                    "caption_cache_hits": view["caption_cache_hits"],
                    "cache_identity_hashes": tuple(
                        item["cache_identity_hash"] for item in view["segments"]),
                    "qa_run_directory": os.path.abspath(run_dir),
                    **row,
                }
                _write_immutable(result_path, persisted)
                rows.append(persisted)
        return tuple(rows)

    def evaluate(
        self, *, request: MetaPromptConfirmationRequest,
        parent: MetaPromptVersion, candidate: MetaPromptVersion,
        output_directory: str,
    ) -> PairedMetaPromptConfirmationResult:
        self._validate_request(request)
        if request.parent_meta_prompt_id != parent.meta_prompt_id or \
                request.candidate_meta_prompt_id != candidate.meta_prompt_id or \
                candidate.parent_meta_prompt_id != parent.meta_prompt_id:
            raise ConfirmationEvaluationConflictError(
                "parent/candidate lineage does not match confirmation request")
        output = os.path.abspath(output_directory)
        manifest_path = os.path.join(output, "dvd_confirmation_manifest.json")
        request_hash = sha256_json(json.loads(dumps_canonical({
            "request": request, "parent": parent, "candidate": candidate,
            "configuration": self.configuration_identity,
        })))
        if os.path.isfile(manifest_path):
            manifest = _read_json(manifest_path)
            if manifest.get("status") != "completed" or \
                    manifest.get("request_hash") != request_hash:
                raise ConfirmationEvaluationConflictError(
                    "completed DVD confirmation belongs to different inputs")
            return PairedMetaPromptConfirmationResult(
                request_id=request.request_id,
                confirmation_set_id=request.confirmation_set_id,
                parent_meta_prompt_id=parent.meta_prompt_id,
                candidate_meta_prompt_id=candidate.meta_prompt_id,
                model_identity=request.model_identity,
                decoding_settings=request.decoding_settings,
                cache_reset_identity=request.cache_reset_identity,
                evaluation_pipeline_identity=request.evaluation_pipeline_identity,
                outcomes=tuple(MetaPromptConfirmationOutcome(**item)
                               for item in manifest["protocol_outcomes"]))

        self.call_count += 1
        evaluator_dir = os.path.join(output, "dvd_runtime")
        bundle_path, bundle = self.evaluator._materialize_bundle(
            confirmation_videos=self.confirmation_videos,
            scaffold=self.scaffold, contract=self.contract,
            output_dir=evaluator_dir)
        parent_state = self.evaluator._caption_state(
            state_name="parent", bundle=bundle, bank=self.bank,
            router=self.router, scaffold=self.scaffold, contract=self.contract,
            output_dir=evaluator_dir, free_form_generator=self._generator(parent))
        candidate_state = self.evaluator._caption_state(
            state_name="candidate", bundle=bundle, bank=self.bank,
            router=self.router, scaffold=self.scaffold, contract=self.contract,
            output_dir=evaluator_dir,
            free_form_generator=self._generator(candidate))
        self.evaluator._validate_paired_states(
            parent_state, candidate_state, bundle)
        parent_rows = self._run_qas(
            state_name="parent", bundle=bundle, state=parent_state,
            output_directory=evaluator_dir)
        candidate_rows = self._run_qas(
            state_name="candidate", bundle=bundle, state=candidate_state,
            output_directory=evaluator_dir)
        before = {row["question_id"]: row for row in parent_rows}
        after = {row["question_id"]: row for row in candidate_rows}
        protocol = []
        rich = []
        for case in request.cases:
            left, right = before[case.qa_id], after[case.qa_id]
            captions_equal = left["captions_hash"] == right["captions_hash"]
            parent_error = _error_text(left["errors"])
            candidate_error = _error_text(right["errors"])
            protocol_row = MetaPromptConfirmationOutcome(
                case_id=case.case_id, video_id=case.video_id, qa_id=case.qa_id,
                parent_correct=left["is_correct"],
                candidate_correct=right["is_correct"],
                captions_equal=captions_equal,
                parent_error=parent_error, candidate_error=candidate_error)
            protocol.append(protocol_row)
            transition = (
                _transition(left["is_correct"], right["is_correct"])
                if isinstance(left["is_correct"], bool) and
                isinstance(right["is_correct"], bool) else None)
            rich.append({
                "case_id": case.case_id, "video_id": case.video_id,
                "qa_id": case.qa_id,
                "parent_prediction": left["prediction"],
                "candidate_prediction": right["prediction"],
                "parent_correct": left["is_correct"],
                "candidate_correct": right["is_correct"],
                "transition_type": transition,
                "captions_equal": captions_equal,
                "uncertain_noop": bool(captions_equal and transition in (
                    "wrong_to_correct", "correct_to_wrong")),
                "parent_prompt_artifact_ref": left["routing_manifest_path"],
                "candidate_prompt_artifact_ref": right["routing_manifest_path"],
                "parent_caption_artifact_ref": left["captions_path"],
                "candidate_caption_artifact_ref": right["captions_path"],
                "parent_cache_identity_hashes": left["cache_identity_hashes"],
                "candidate_cache_identity_hashes": right[
                    "cache_identity_hashes"],
                "parent_execution_error": parent_error,
                "candidate_execution_error": candidate_error,
                "parent_runtime_statistics": {
                    "caption_calls": left["caption_calls"],
                    "caption_cache_hits": left["caption_cache_hits"],
                    "qa_latency_seconds": left["latency_seconds"],
                    "qa_run_directory": left["qa_run_directory"],
                },
                "candidate_runtime_statistics": {
                    "caption_calls": right["caption_calls"],
                    "caption_cache_hits": right["caption_cache_hits"],
                    "qa_latency_seconds": right["latency_seconds"],
                    "qa_run_directory": right["qa_run_directory"],
                },
            })
        failures = tuple(row["qa_id"] for row in rich if
                         row["parent_execution_error"] or
                         row["candidate_execution_error"])
        valid = tuple(row for row in rich if row["transition_type"] is not None)
        count = len(request.cases)
        aggregate = {
            "evaluated_qa_count": len(valid),
            "parent_accuracy": sum(bool(row["parent_correct"])
                                   for row in valid) / count,
            "candidate_accuracy": sum(bool(row["candidate_correct"])
                                      for row in valid) / count,
            "accuracy_delta": (sum(bool(row["candidate_correct"])
                                   for row in valid) -
                               sum(bool(row["parent_correct"])
                                   for row in valid)) / count,
            "wrong_to_correct": sum(row["transition_type"] == "wrong_to_correct"
                                    for row in valid),
            "correct_to_wrong": sum(row["transition_type"] == "correct_to_wrong"
                                    for row in valid),
            "unchanged": sum(row["transition_type"] in (
                "correct_to_correct", "wrong_to_wrong") for row in valid),
            "execution_failures": failures,
            "uncertain_noop_qa_count": sum(row["uncertain_noop"] for row in rich),
        }
        manifest = {
            "schema_version": "dvd_meta_prompt_confirmation_manifest_v1",
            "status": "completed", "request_hash": request_hash,
            "request_id": request.request_id,
            "input_bundle_path": bundle_path,
            "input_bundle_file_sha256": _file_hash(bundle_path),
            "input_bundle_hash": bundle["bundle_hash"],
            "parent_caption_state_path": parent_state["artifact_path"],
            "candidate_caption_state_path": candidate_state["artifact_path"],
            "configuration": self.configuration_identity,
            "protocol_outcomes": tuple(as_json_dict(row) for row in protocol),
            "qa_results": rich, "aggregate": aggregate,
            "shared_immutable_inputs": (
                "input bundle, sampled frame files, transcripts, QA samples, "
                "captioner/DVD models and decoding configuration"),
            "independently_recomputed": (
                "prompt generations, on-policy histories, captions, caption "
                "cache identities, caption databases, and downstream QA runs"),
            "legacy_property_routing_used": False,
        }
        _write_immutable(manifest_path, manifest)
        return PairedMetaPromptConfirmationResult(
            request_id=request.request_id,
            confirmation_set_id=request.confirmation_set_id,
            parent_meta_prompt_id=parent.meta_prompt_id,
            candidate_meta_prompt_id=candidate.meta_prompt_id,
            model_identity=request.model_identity,
            decoding_settings=request.decoding_settings,
            cache_reset_identity=request.cache_reset_identity,
            evaluation_pipeline_identity=request.evaluation_pipeline_identity,
            outcomes=tuple(protocol))


class LazyDVDMetaPromptConfirmationEvaluator:
    """Delay GPU/DVD construction until an updater actually returns update."""

    policy_version = "lazy_dvd_meta_prompt_confirmation_v1"

    def __init__(self, *, configuration_identity: Mapping[str, Any], factory):
        if not isinstance(configuration_identity, Mapping) or not \
                configuration_identity:
            raise ValueError("configuration_identity must be explicit")
        if not callable(factory):
            raise TypeError("factory must be callable")
        self._configuration_identity = json.loads(dumps_canonical(
            configuration_identity))
        self._factory = factory
        self._delegate = None

    @property
    def configuration_identity(self) -> Mapping[str, Any]:
        return self._configuration_identity

    def evaluate(self, **kwargs):
        if self._delegate is None:
            delegate = self._factory()
            if not isinstance(delegate, DVDMetaPromptConfirmationEvaluator):
                raise TypeError(
                    "lazy factory must return DVDMetaPromptConfirmationEvaluator")
            self._delegate = delegate
        return self._delegate.evaluate(**kwargs)
