"""Checkpoint F prompt-delta update, confirmation, and promotion boundary.

The orchestrator is provider-independent. Feedback generation, meta-prompt
updating, and paired confirmation evaluation are injected. It never touches
the legacy property codebook/router and never silently selects thresholds.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from surrogate_rollout.optimization.episode_feedback import (
    EpisodeFeedbackGenerator,
    classify_qa_transition,
    evaluate_episode_feedback_eligibility,
)
from surrogate_rollout.optimization.meta_prompt_updater import MetaPromptUpdater
from surrogate_rollout.optimization.llm_episode_feedback import (
    LEAN_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION,
)
from surrogate_rollout.optimization.feedback_memory import (
    append_parent_feedback_memory_bank,
    archive_parent_feedback_memory_bank,
    build_episode_feedback_memory_record,
    initialize_parent_feedback_memory_bank,
    load_parent_feedback_memory_bank,
)
from surrogate_rollout.optimization.schemas import (
    EpisodeFeedback,
    InterventionEpisode,
    MetaPromptUpdateDecision,
    MetaPromptVersion,
    episode_feedback_from_json,
    meta_prompt_update_decision_from_json,
    meta_prompt_version_from_json,
    validate_meta_prompt_update_decision,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json


ITERATION_SCHEMA_VERSION = "prompt_delta_iteration_v1"
CONFIRMATION_REQUEST_SCHEMA_VERSION = "meta_prompt_confirmation_request_v1"
CONFIRMATION_RESULT_SCHEMA_VERSION = "meta_prompt_confirmation_result_v1"
PROMOTION_DECISION_SCHEMA_VERSION = "meta_prompt_promotion_decision_v1"
CURRENT_POINTER_SCHEMA_VERSION = "current_meta_prompt_pointer_v1"


class PromptDeltaIterationError(RuntimeError):
    pass


class PromptDeltaIterationConflictError(PromptDeltaIterationError):
    pass


@dataclass(frozen=True)
class MetaPromptConfirmationCase:
    case_id: str
    video_id: str
    qa_id: str
    input_ref: str

    def __post_init__(self) -> None:
        for name in ("case_id", "video_id", "qa_id", "input_ref"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"MetaPromptConfirmationCase.{name} is required")


@dataclass(frozen=True)
class MetaPromptConfirmationCriterion:
    minimum_sample_count: int
    minimum_accuracy_delta: float
    maximum_correct_to_wrong: int
    require_no_execution_failures: bool

    def __post_init__(self) -> None:
        if not isinstance(self.minimum_sample_count, int) or \
                isinstance(self.minimum_sample_count, bool) or \
                self.minimum_sample_count <= 0:
            raise ValueError("minimum_sample_count must be a positive integer")
        if not isinstance(self.minimum_accuracy_delta, (int, float)) or \
                isinstance(self.minimum_accuracy_delta, bool):
            raise TypeError("minimum_accuracy_delta must be numeric")
        if not isinstance(self.maximum_correct_to_wrong, int) or \
                isinstance(self.maximum_correct_to_wrong, bool) or \
                self.maximum_correct_to_wrong < 0:
            raise ValueError("maximum_correct_to_wrong must be non-negative")
        if not isinstance(self.require_no_execution_failures, bool):
            raise TypeError("require_no_execution_failures must be bool")


@dataclass(frozen=True)
class MetaPromptConfirmationRequest:
    request_id: str
    confirmation_set_id: str
    parent_meta_prompt_id: str
    candidate_meta_prompt_id: str
    cases: tuple[MetaPromptConfirmationCase, ...]
    model_identity: str
    decoding_settings: Mapping[str, Any]
    cache_reset_identity: str
    evaluation_pipeline_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.cases, tuple) or not self.cases or any(
                not isinstance(item, MetaPromptConfirmationCase)
                for item in self.cases):
            raise TypeError("confirmation request cases are invalid")


@dataclass(frozen=True)
class MetaPromptConfirmationOutcome:
    case_id: str
    video_id: str
    qa_id: str
    parent_correct: bool | None
    candidate_correct: bool | None
    captions_equal: bool
    parent_error: str | None = None
    candidate_error: str | None = None

    def __post_init__(self) -> None:
        for name in ("case_id", "video_id", "qa_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"confirmation outcome {name} is required")
        for name in ("parent_correct", "candidate_correct"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool or None")
        if not isinstance(self.captions_equal, bool):
            raise TypeError("captions_equal must be bool")
        for name in ("parent_error", "candidate_error"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be str or None")


@dataclass(frozen=True)
class PairedMetaPromptConfirmationResult:
    request_id: str
    confirmation_set_id: str
    parent_meta_prompt_id: str
    candidate_meta_prompt_id: str
    model_identity: str
    decoding_settings: Mapping[str, Any]
    cache_reset_identity: str
    evaluation_pipeline_identity: str
    outcomes: tuple[MetaPromptConfirmationOutcome, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.outcomes, tuple) or any(
                not isinstance(item, MetaPromptConfirmationOutcome)
                for item in self.outcomes):
            raise TypeError("confirmation result outcomes are invalid")


class MetaPromptConfirmationEvaluator(Protocol):
    def evaluate(
        self,
        *,
        request: MetaPromptConfirmationRequest,
        parent: MetaPromptVersion,
        candidate: MetaPromptVersion,
        output_directory: str,
    ) -> PairedMetaPromptConfirmationResult:
        ...


@dataclass(frozen=True)
class MetaPromptPromotionDecision:
    decision_id: str
    accepted: bool
    status: str
    parent_accuracy: float
    candidate_accuracy: float
    accuracy_delta: float
    attributable_correct_to_wrong_qa_ids: tuple[str, ...]
    uncertain_noop_qa_ids: tuple[str, ...]
    execution_failures: tuple[str, ...]
    criterion: MetaPromptConfirmationCriterion
    parent_meta_prompt_id: str
    candidate_meta_prompt_id: str


@dataclass(frozen=True)
class PromptDeltaIterationResult:
    iteration_id: str
    status: str
    output_directory: str
    active_meta_prompt_id: str
    candidate_meta_prompt_id: str | None
    final_manifest_path: str
    resumed: bool


def _plain(value: Any) -> Any:
    return json.loads(dumps_canonical(value))


def _read_object(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PromptDeltaIterationError(f"expected JSON object: {path}")
    return value


def _write_once(path: str, value: Any) -> str:
    text = dumps_canonical(value) + "\n"
    absolute = os.path.abspath(path)
    if os.path.exists(absolute):
        with open(absolute, encoding="utf-8") as handle:
            if handle.read() != text:
                raise PromptDeltaIterationConflictError(
                    f"immutable artifact conflict: {absolute}")
        return absolute
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    _atomic_write_text(absolute, text)
    return absolute


def _write_pointer(path: str, value: Any) -> str:
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    _atomic_write_text(absolute, dumps_canonical(value) + "\n")
    return absolute


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_valid_pointer(path: str) -> dict[str, Any]:
    value = _read_object(path)
    if value.get("schema_version") != CURRENT_POINTER_SCHEMA_VERSION:
        raise PromptDeltaIterationConflictError(
            "current meta-prompt pointer schema is invalid")
    artifact = value.get("artifact_path")
    if not isinstance(artifact, str) or not os.path.isfile(artifact) or \
            value.get("artifact_sha256") != _file_sha256(artifact):
        raise PromptDeltaIterationConflictError(
            "current meta-prompt pointer artifact hash is invalid")
    version = meta_prompt_version_from_json(_read_object(artifact))
    if version.meta_prompt_id != value.get("active_meta_prompt_id"):
        raise PromptDeltaIterationConflictError(
            "current pointer identity does not match its artifact")
    return value


def _source_hashes(episodes: Sequence[InterventionEpisode]) -> dict[str, str]:
    paths = {
        reference for episode in episodes
        for reference in (
            episode.baseline_run_ref, episode.intervention_run_ref,
            *(outcome.baseline_trajectory_ref for outcome in episode.qa_outcomes),
            *(outcome.intervention_trajectory_ref for outcome in episode.qa_outcomes),
        ) if reference is not None and os.path.isfile(reference)
    }
    return {os.path.abspath(path): _file_sha256(path) for path in sorted(paths)}


def _component_identity(value: Any) -> dict[str, Any]:
    identity = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }
    for name in ("policy_version", "updater_policy_version"):
        field = getattr(value, name, None)
        if isinstance(field, str) and field:
            identity[name] = field
    metadata = getattr(value, "metadata", None)
    if callable(metadata):
        raw = _plain(metadata())
        if isinstance(raw, dict):
            raw = {key: item for key, item in raw.items()
                   if not key.endswith("call_count")}
        identity["metadata"] = raw
    configured = getattr(value, "configuration_identity", None)
    if isinstance(configured, Mapping):
        identity["configuration_identity"] = _plain(configured)
    for nested_name in ("response_provider", "backend"):
        nested = getattr(value, nested_name, None)
        nested_metadata = getattr(nested, "metadata", None)
        if callable(nested_metadata):
            raw = _plain(nested_metadata())
            if isinstance(raw, dict):
                raw = {key: item for key, item in raw.items()
                       if not key.endswith("call_count")}
            identity[f"{nested_name}_metadata"] = raw
    return identity


def build_feedback_grounding(
    feedback: EpisodeFeedback,
    episode: InterventionEpisode,
) -> dict[str, Any]:
    changed = tuple(
        clip.segment_id for clip in episode.clips
        if clip.baseline_caption != clip.intervention_caption)
    transitions = {
        name: [] for name in (
            "correct_to_correct", "wrong_to_wrong", "wrong_to_correct",
            "correct_to_wrong")
    }
    for outcome in episode.qa_outcomes:
        transitions[classify_qa_transition(outcome)].append(outcome.qa_id)
    unchanged = not changed
    has_positive_flip = bool(transitions["wrong_to_correct"])
    return {
        "schema_version": "episode_feedback_grounding_v1",
        "feedback_id": feedback.feedback_id,
        "episode_id": episode.episode_id,
        "caption_change_status": "unchanged" if unchanged else "changed",
        "changed_segment_ids": list(changed),
        "qa_transition_summary": transitions,
        "qa_flip_attribution": (
            "positive_episode_signal_without_caption_change"
            if unchanged and has_positive_flip else
            "no_positive_signal_without_caption_change" if unchanged else
            "episode_level_with_caption_change"),
    }


def build_confirmation_request(
    *,
    parent: MetaPromptVersion,
    candidate: MetaPromptVersion,
    cases: Sequence[MetaPromptConfirmationCase],
    model_identity: str,
    decoding_settings: Mapping[str, Any],
    cache_reset_identity: str,
    evaluation_pipeline_identity: str,
) -> MetaPromptConfirmationRequest:
    ordered = tuple(cases)
    if not ordered or len({item.case_id for item in ordered}) != len(ordered):
        raise ValueError("confirmation cases must be non-empty with unique IDs")
    if len({(item.video_id, item.qa_id) for item in ordered}) != len(ordered):
        raise ValueError("confirmation video/QA pairs must be unique")
    for name, value in (
            ("model_identity", model_identity),
            ("cache_reset_identity", cache_reset_identity),
            ("evaluation_pipeline_identity", evaluation_pipeline_identity)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be explicitly configured")
    if not isinstance(decoding_settings, Mapping) or not decoding_settings:
        raise ValueError("decoding_settings must be explicitly configured")
    identity = {
        "schema_version": CONFIRMATION_REQUEST_SCHEMA_VERSION,
        "parent_meta_prompt_id": parent.meta_prompt_id,
        "candidate_meta_prompt_id": candidate.meta_prompt_id,
        "cases": ordered,
        "model_identity": model_identity,
        "decoding_settings": decoding_settings,
        "cache_reset_identity": cache_reset_identity,
        "evaluation_pipeline_identity": evaluation_pipeline_identity,
    }
    confirmation_set_id = "confirmation_set_" + sha256_json({
        "cases": _plain(ordered),
        "pipeline": evaluation_pipeline_identity,
        "cache_reset": cache_reset_identity,
    })[:20]
    return MetaPromptConfirmationRequest(
        request_id="meta_prompt_confirmation_" + sha256_json(
            _plain(identity))[:20],
        confirmation_set_id=confirmation_set_id,
        parent_meta_prompt_id=parent.meta_prompt_id,
        candidate_meta_prompt_id=candidate.meta_prompt_id,
        cases=ordered,
        model_identity=model_identity,
        decoding_settings=_plain(decoding_settings),
        cache_reset_identity=cache_reset_identity,
        evaluation_pipeline_identity=evaluation_pipeline_identity,
    )


def decide_meta_prompt_promotion(
    *,
    request: MetaPromptConfirmationRequest,
    result: PairedMetaPromptConfirmationResult,
    criterion: MetaPromptConfirmationCriterion,
) -> MetaPromptPromotionDecision:
    echoed = (
        result.request_id == request.request_id and
        result.confirmation_set_id == request.confirmation_set_id and
        result.parent_meta_prompt_id == request.parent_meta_prompt_id and
        result.candidate_meta_prompt_id == request.candidate_meta_prompt_id and
        result.model_identity == request.model_identity and
        dumps_canonical(result.decoding_settings) ==
        dumps_canonical(request.decoding_settings) and
        result.cache_reset_identity == request.cache_reset_identity and
        result.evaluation_pipeline_identity ==
        request.evaluation_pipeline_identity)
    if not echoed:
        raise PromptDeltaIterationError(
            "confirmation result does not match the paired request identity")
    expected = tuple((item.case_id, item.video_id, item.qa_id)
                     for item in request.cases)
    observed = tuple((item.case_id, item.video_id, item.qa_id)
                     for item in result.outcomes)
    if observed != expected:
        raise PromptDeltaIterationError(
            "confirmation result changed case order or identity")

    failures = []
    parent_values = []
    effective_candidate_values = []
    regressions = []
    uncertain_noops = []
    for row in result.outcomes:
        if row.parent_error or row.candidate_error or \
                not isinstance(row.parent_correct, bool) or \
                not isinstance(row.candidate_correct, bool):
            failures.append(row.case_id)
            continue
        parent_values.append(row.parent_correct)
        if row.captions_equal:
            effective_candidate_values.append(row.parent_correct)
            if row.parent_correct != row.candidate_correct:
                uncertain_noops.append(row.qa_id)
        else:
            effective_candidate_values.append(row.candidate_correct)
            if row.parent_correct and not row.candidate_correct:
                regressions.append(row.qa_id)
    denominator = len(request.cases)
    parent_accuracy = sum(parent_values) / denominator
    candidate_accuracy = sum(effective_candidate_values) / denominator
    delta = candidate_accuracy - parent_accuracy
    accepted = (
        len(request.cases) >= criterion.minimum_sample_count and
        delta >= criterion.minimum_accuracy_delta and
        len(regressions) <= criterion.maximum_correct_to_wrong and
        (not criterion.require_no_execution_failures or not failures))
    identity = {
        "request_id": request.request_id,
        "result": result,
        "criterion": criterion,
        "accepted": accepted,
    }
    return MetaPromptPromotionDecision(
        decision_id="promotion_" + sha256_json(_plain(identity))[:20],
        accepted=accepted,
        status="promoted" if accepted else "rolled_back",
        parent_accuracy=parent_accuracy,
        candidate_accuracy=candidate_accuracy,
        accuracy_delta=delta,
        attributable_correct_to_wrong_qa_ids=tuple(regressions),
        uncertain_noop_qa_ids=tuple(uncertain_noops),
        execution_failures=tuple(failures),
        criterion=criterion,
        parent_meta_prompt_id=request.parent_meta_prompt_id,
        candidate_meta_prompt_id=request.candidate_meta_prompt_id,
    )


class PromptDeltaIterationOrchestrator:
    def __init__(
        self,
        *,
        feedback_generator: EpisodeFeedbackGenerator,
        updater: MetaPromptUpdater,
        confirmation_evaluator: MetaPromptConfirmationEvaluator,
    ) -> None:
        self.feedback_generator = feedback_generator
        self.updater = updater
        self.confirmation_evaluator = confirmation_evaluator

    def run(
        self,
        *,
        iteration_id: str,
        parent: MetaPromptVersion,
        update_episodes: Sequence[InterventionEpisode],
        confirmation_cases: Sequence[MetaPromptConfirmationCase],
        criterion: MetaPromptConfirmationCriterion,
        model_identity: str,
        decoding_settings: Mapping[str, Any],
        cache_reset_identity: str,
        evaluation_pipeline_identity: str,
        candidate_created_at: str,
        output_directory: str,
        state_directory: str,
        feedback_memory_bank_directory: str,
        historical_memory_character_budget: int | None = None,
        initialize_parent_pointer: bool = False,
    ) -> PromptDeltaIterationResult:
        if not iteration_id:
            raise ValueError("iteration_id is required")
        episodes = tuple(update_episodes)
        cases = tuple(confirmation_cases)
        if not episodes or len({item.episode_id for item in episodes}) != len(episodes):
            raise ValueError("update episodes must be non-empty and unique")
        update_videos = {item.video_id for item in episodes}
        update_qas = {outcome.qa_id for item in episodes
                      for outcome in item.qa_outcomes}
        overlap = tuple(item.case_id for item in cases
                        if item.video_id in update_videos or item.qa_id in update_qas)
        if overlap:
            raise ValueError(
                f"confirmation set overlaps update evidence: {overlap}")
        if len(cases) < criterion.minimum_sample_count:
            raise ValueError(
                "confirmation set is smaller than minimum_sample_count")
        if not candidate_created_at:
            raise ValueError("candidate_created_at must be explicit")
        if not isinstance(feedback_memory_bank_directory, str) or not \
                feedback_memory_bank_directory:
            raise ValueError("feedback_memory_bank_directory is required")

        output = os.path.abspath(output_directory)
        state = os.path.abspath(state_directory)
        os.makedirs(output, exist_ok=True)
        input_identity = {
            "schema_version": ITERATION_SCHEMA_VERSION,
            "iteration_id": iteration_id,
            "parent": parent,
            "update_episode_hashes": [sha256_json(_plain(item)) for item in episodes],
            "confirmation_cases": cases,
            "criterion": criterion,
            "model_identity": model_identity,
            "decoding_settings": decoding_settings,
            "cache_reset_identity": cache_reset_identity,
            "evaluation_pipeline_identity": evaluation_pipeline_identity,
            "candidate_created_at": candidate_created_at,
            "feedback_memory_bank_directory": os.path.abspath(
                feedback_memory_bank_directory),
            "historical_memory_character_budget": (
                historical_memory_character_budget),
            "components": {
                "feedback_generator": _component_identity(
                    self.feedback_generator),
                "updater": _component_identity(self.updater),
                "confirmation_evaluator": _component_identity(
                    self.confirmation_evaluator),
            },
        }
        identity_path = _write_once(
            os.path.join(output, "input_identity.json"), input_identity)
        final_path = os.path.join(output, "iteration_result.json")
        if os.path.isfile(final_path):
            final = _read_object(final_path)
            current_sources = _source_hashes(episodes)
            if final.get("source_hashes_before") != current_sources or \
                    final.get("source_hashes_after") != current_sources:
                raise PromptDeltaIterationConflictError(
                    "completed iteration source artifact hashes changed")
            pointer = _read_valid_pointer(
                os.path.join(state, "current_meta_prompt.json"))
            if pointer.get("active_meta_prompt_id") != \
                    final.get("active_meta_prompt_id"):
                raise PromptDeltaIterationConflictError(
                    "completed iteration active pointer changed")
            return PromptDeltaIterationResult(
                iteration_id=iteration_id, status=final["status"],
                output_directory=output,
                active_meta_prompt_id=final["active_meta_prompt_id"],
                candidate_meta_prompt_id=final.get("candidate_meta_prompt_id"),
                final_manifest_path=final_path, resumed=True)

        source_before = _source_hashes(episodes)
        pointer_path = os.path.join(state, "current_meta_prompt.json")
        parent_state_path = os.path.join(
            state, "versions", f"{parent.meta_prompt_id}.json")
        if not os.path.exists(pointer_path):
            if not initialize_parent_pointer:
                raise PromptDeltaIterationError(
                    "current meta-prompt pointer is absent; explicit "
                    "initialize_parent_pointer is required")
            _write_once(parent_state_path, parent)
            _write_pointer(pointer_path, {
                "schema_version": CURRENT_POINTER_SCHEMA_VERSION,
                "active_meta_prompt_id": parent.meta_prompt_id,
                "artifact_path": parent_state_path,
                "artifact_sha256": _file_sha256(parent_state_path),
            })

        historical_memories = load_parent_feedback_memory_bank(
            feedback_memory_bank_directory, parent.meta_prompt_id)
        feedbacks = []
        grounding = []
        iteration_memories = []
        for index, episode in enumerate(episodes):
            stage = os.path.join(output, "feedback", f"{index:03d}_{episode.episode_id}")
            feedback_path = os.path.join(stage, "feedback.json")
            eligibility_path = os.path.join(stage, "eligibility.json")
            grounding_path = os.path.join(stage, "grounding.json")
            memory_path = os.path.join(stage, "compact_memory.json")
            if os.path.isfile(feedback_path):
                if callable(getattr(
                        self.feedback_generator, "generate_to_directory", None)):
                    required_trace_paths = {
                        name: os.path.join(stage, name) for name in (
                            "request.json", "provider_request.json",
                            "request_manifest.json", "raw_response.json")
                    }
                    missing = [name for name, path in required_trace_paths.items()
                               if not os.path.isfile(path)]
                    if missing:
                        raise PromptDeltaIterationConflictError(
                            "saved feedback predates the active lean request "
                            f"contract; remove this feedback stage and resume: "
                            f"{stage}; missing={missing!r}")
                    request_payload = _read_object(
                        required_trace_paths["request.json"])
                    request_manifest = _read_object(
                        required_trace_paths["request_manifest.json"])
                    if request_payload.get("schema_version") != \
                            LEAN_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION or \
                            request_manifest.get("request_schema_version") != \
                            LEAN_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION or \
                            request_manifest.get("episode_id") != \
                            episode.episode_id or \
                            request_manifest.get("model_payload_hash") != \
                            sha256_json(request_payload):
                        raise PromptDeltaIterationConflictError(
                            "saved feedback request trace does not match the "
                            "active lean request contract")
                saved_feedback_data = _read_object(feedback_path)
                feedback = episode_feedback_from_json(saved_feedback_data)
                eligibility = _read_object(eligibility_path)
                grounded = _read_object(grounding_path)
                if not eligibility.get("eligible"):
                    raise PromptDeltaIterationError(
                        f"saved feedback is ineligible: {episode.episode_id}")
                if eligibility.get("feedback_id") != feedback.feedback_id or \
                        eligibility.get("episode_id") != episode.episode_id or \
                        eligibility.get("feedback_sha256") != sha256_json(
                            saved_feedback_data):
                    raise PromptDeltaIterationConflictError(
                        "saved feedback eligibility provenance mismatch")
                if dumps_canonical(grounded) != dumps_canonical(
                        build_feedback_grounding(feedback, episode)):
                    raise PromptDeltaIterationConflictError(
                        "saved feedback grounding mismatch")
            else:
                generate_to_directory = getattr(
                    self.feedback_generator, "generate_to_directory", None)
                feedback = (
                    generate_to_directory(episode, stage)
                    if callable(generate_to_directory) else
                    self.feedback_generator.generate(episode))
                eligibility_record = evaluate_episode_feedback_eligibility(
                    feedback, episode)
                _write_once(eligibility_path, eligibility_record)
                if not eligibility_record.eligible:
                    raise PromptDeltaIterationError(
                        f"feedback is ineligible: {eligibility_record.reasons}")
                grounded = build_feedback_grounding(feedback, episode)
                _write_once(grounding_path, grounded)
                _write_once(feedback_path, feedback)
            feedbacks.append(feedback)
            grounding.append(grounded)
            expected_memory = build_episode_feedback_memory_record(
                feedback=feedback, episode=episode, iteration_id=iteration_id,
                parent_meta_prompt_id=parent.meta_prompt_id,
                feedback_artifact_ref=feedback_path)
            expected_payload = {
                "schema_version": "iteration_episode_feedback_memory_v2",
                "record": expected_memory,
            }
            if os.path.isfile(memory_path):
                if dumps_canonical(_read_object(memory_path)) != \
                        dumps_canonical(_plain(expected_payload)):
                    raise PromptDeltaIterationConflictError(
                        "saved compact feedback memory mismatch")
            else:
                _write_once(memory_path, expected_payload)
            if expected_memory is not None:
                iteration_memories.append(expected_memory)

        updater_path = os.path.join(output, "updater_result.json")
        if os.path.isfile(updater_path):
            saved_update = _read_object(updater_path)
            decision = meta_prompt_update_decision_from_json(
                saved_update["decision"])
            candidate_id = saved_update.get("candidate_meta_prompt_id")
            known_feedback_ids = {item.feedback_id for item in feedbacks}
            known_feedback_ids.update(
                item.feedback_id for item in historical_memories)
            unknown = set(decision.supporting_feedback_ids) - known_feedback_ids
            if unknown:
                raise PromptDeltaIterationConflictError(
                    f"saved updater result references unknown memory: {unknown}")
        else:
            update = self.updater.update(
                parent, tuple(feedbacks),
                feedback_grounding=tuple(grounding),
                historical_memories=historical_memories,
                current_iteration_id=iteration_id,
                historical_memory_character_budget=(
                    historical_memory_character_budget))
            decision = update.decision
            candidate_id = update.candidate_meta_prompt_id
            _write_once(updater_path, {
                "schema_version": "prompt_delta_iteration_updater_result_v1",
                "request_id": update.request.request_id,
                "request_payload_hash": update.request.payload_hash,
                "request": update.request.payload,
                "decision": decision,
                "candidate_meta_prompt_id": candidate_id,
                "candidate_status": update.candidate_status,
                "updater_policy_version": update.updater_policy_version,
                "backend_metadata": update.backend_metadata,
                "raw_response": update.raw_response,
            })

        # Current compact text is deliberately appended only after the single
        # updater decision. It therefore becomes historical evidence starting
        # with the next iteration under this same parent.
        memory_bank = append_parent_feedback_memory_bank(
            feedback_memory_bank_directory, parent.meta_prompt_id,
            tuple(iteration_memories))

        pointer = _read_valid_pointer(pointer_path)
        if decision.decision == "no_update":
            if pointer.get("active_meta_prompt_id") != parent.meta_prompt_id:
                raise PromptDeltaIterationConflictError(
                    "no_update parent pointer changed during iteration")
            source_after = _source_hashes(episodes)
            if source_after != source_before:
                raise PromptDeltaIterationConflictError("source artifacts changed")
            final = {
                "schema_version": ITERATION_SCHEMA_VERSION,
                "iteration_id": iteration_id,
                "status": "no_update",
                "active_meta_prompt_id": parent.meta_prompt_id,
                "candidate_meta_prompt_id": None,
                "input_identity_path": identity_path,
                "updater_result_path": updater_path,
                "feedback_memory_bank": memory_bank,
                "source_hashes_before": source_before,
                "source_hashes_after": source_after,
            }
            final_manifest = _write_once(final_path, final)
            return PromptDeltaIterationResult(
                iteration_id, "no_update", output, parent.meta_prompt_id,
                None, final_manifest, False)

        if not candidate_id or not decision.candidate_meta_prompt:
            raise PromptDeltaIterationError("update decision has no candidate")
        candidate = MetaPromptVersion(
            meta_prompt_id=candidate_id,
            parent_meta_prompt_id=parent.meta_prompt_id,
            text=decision.candidate_meta_prompt,
            created_at=candidate_created_at,
            status="provisional")
        provisional_path = _write_once(
            os.path.join(output, "provisional_meta_prompt.json"), candidate)
        request = build_confirmation_request(
            parent=parent, candidate=candidate, cases=cases,
            model_identity=model_identity,
            decoding_settings=decoding_settings,
            cache_reset_identity=cache_reset_identity,
            evaluation_pipeline_identity=evaluation_pipeline_identity)
        request_path = _write_once(
            os.path.join(output, "confirmation", "request.json"), request)
        confirmation_path = os.path.join(
            output, "confirmation", "paired_result.json")
        if os.path.isfile(confirmation_path):
            result = paired_confirmation_result_from_json(
                _read_object(confirmation_path))
        else:
            result = self.confirmation_evaluator.evaluate(
                request=request, parent=parent, candidate=candidate,
                output_directory=os.path.dirname(confirmation_path))
            _write_once(confirmation_path, result)
        promotion = decide_meta_prompt_promotion(
            request=request, result=result, criterion=criterion)
        decision_path = _write_once(
            os.path.join(output, "confirmation", "promotion_decision.json"),
            promotion)

        pointer = _read_valid_pointer(pointer_path)
        if promotion.accepted:
            if pointer.get("active_meta_prompt_id") not in (
                    parent.meta_prompt_id, candidate.meta_prompt_id):
                raise PromptDeltaIterationConflictError(
                    "current pointer is neither parent nor resumed candidate")
            confirmed = MetaPromptVersion(
                meta_prompt_id=candidate.meta_prompt_id,
                parent_meta_prompt_id=parent.meta_prompt_id,
                text=candidate.text, created_at=candidate.created_at,
                status="confirmed")
            active_path = _write_once(
                os.path.join(state, "versions", f"{confirmed.meta_prompt_id}.json"),
                confirmed)
            if pointer.get("active_meta_prompt_id") == parent.meta_prompt_id:
                _write_pointer(pointer_path, {
                    "schema_version": CURRENT_POINTER_SCHEMA_VERSION,
                    "active_meta_prompt_id": confirmed.meta_prompt_id,
                    "artifact_path": active_path,
                    "artifact_sha256": _file_sha256(active_path),
                    "parent_meta_prompt_id": parent.meta_prompt_id,
                    "promotion_decision_path": decision_path,
                })
            active_id = candidate.meta_prompt_id
            status = "promoted"
            archived_memory_bank = archive_parent_feedback_memory_bank(
                feedback_memory_bank_directory, parent.meta_prompt_id,
                promoted_meta_prompt_id=candidate.meta_prompt_id)
            new_parent_memory_bank = initialize_parent_feedback_memory_bank(
                feedback_memory_bank_directory, candidate.meta_prompt_id)
        else:
            if pointer.get("active_meta_prompt_id") != parent.meta_prompt_id:
                raise PromptDeltaIterationConflictError(
                    "rollback requires the parent pointer to remain active")
            rejected = MetaPromptVersion(
                meta_prompt_id=candidate.meta_prompt_id,
                parent_meta_prompt_id=parent.meta_prompt_id,
                text=candidate.text, created_at=candidate.created_at,
                status="rejected")
            _write_once(os.path.join(output, "rejected_meta_prompt.json"), rejected)
            active_id = parent.meta_prompt_id
            status = "rolled_back"
            archived_memory_bank = None
            new_parent_memory_bank = None

        source_after = _source_hashes(episodes)
        if source_after != source_before:
            raise PromptDeltaIterationConflictError("source artifacts changed")
        final = {
            "schema_version": ITERATION_SCHEMA_VERSION,
            "iteration_id": iteration_id,
            "status": status,
            "active_meta_prompt_id": active_id,
            "candidate_meta_prompt_id": candidate.meta_prompt_id,
            "input_identity_path": identity_path,
            "updater_result_path": updater_path,
            "feedback_memory_bank": memory_bank,
            "archived_parent_feedback_memory_bank": archived_memory_bank,
            "new_parent_feedback_memory_bank": new_parent_memory_bank,
            "provisional_meta_prompt_path": provisional_path,
            "confirmation_request_path": request_path,
            "confirmation_result_path": confirmation_path,
            "promotion_decision_path": decision_path,
            "current_pointer_path": pointer_path,
            "source_hashes_before": source_before,
            "source_hashes_after": source_after,
        }
        final_manifest = _write_once(final_path, final)
        return PromptDeltaIterationResult(
            iteration_id, status, output, active_id,
            candidate.meta_prompt_id, final_manifest, False)


def paired_confirmation_result_from_json(
    value: Mapping[str, Any],
) -> PairedMetaPromptConfirmationResult:
    return PairedMetaPromptConfirmationResult(
        request_id=value["request_id"],
        confirmation_set_id=value["confirmation_set_id"],
        parent_meta_prompt_id=value["parent_meta_prompt_id"],
        candidate_meta_prompt_id=value["candidate_meta_prompt_id"],
        model_identity=value["model_identity"],
        decoding_settings=value["decoding_settings"],
        cache_reset_identity=value["cache_reset_identity"],
        evaluation_pipeline_identity=value["evaluation_pipeline_identity"],
        outcomes=tuple(MetaPromptConfirmationOutcome(**item)
                       for item in value["outcomes"]),
    )


class DeterministicMockMetaPromptConfirmationEvaluator:
    """Caller-supplied paired rows for bounded offline tests and dry-runs."""

    def __init__(self, outcomes: Sequence[MetaPromptConfirmationOutcome]) -> None:
        self.outcomes = tuple(outcomes)
        self.call_count = 0

    def evaluate(
        self, *, request: MetaPromptConfirmationRequest,
        parent: MetaPromptVersion, candidate: MetaPromptVersion,
        output_directory: str,
    ) -> PairedMetaPromptConfirmationResult:
        self.call_count += 1
        return PairedMetaPromptConfirmationResult(
            request_id=request.request_id,
            confirmation_set_id=request.confirmation_set_id,
            parent_meta_prompt_id=parent.meta_prompt_id,
            candidate_meta_prompt_id=candidate.meta_prompt_id,
            model_identity=request.model_identity,
            decoding_settings=request.decoding_settings,
            cache_reset_identity=request.cache_reset_identity,
            evaluation_pipeline_identity=request.evaluation_pipeline_identity,
            outcomes=self.outcomes,
        )
