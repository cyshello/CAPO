"""Fail-closed model audit and startup logging for real optimization runs."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import as_json_dict, dumps_canonical
from surrogate_rollout.schemas import sha256_text


SCHEMA_VERSION = "checkpoint3e_startup_models_v1"
_DISALLOWED_MARKERS = (
    "mock", "fixture", "fake", "stub", "noop", "no_op", "dummy", "test_double",
)
_REAL_TYPES = {
    "router": (
        "surrogate_rollout.prompt_routing.policies.history_aware_vlm_router."
        "HistoryAwareVLMRouter"),
    "captioner": (
        "surrogate_rollout.captioning.history_aware_baseline."
        "HistoryAwareSegmentCaptioner"),
    "property_proposer": (
        "surrogate_rollout.optimization.policies.property_proposal."
        "MultiPropertyProposalPolicy"),
    "feedback_provider": (
        "surrogate_rollout.optimization.interventional_feedback."
        "PropertyFeedbackRunner"),
    "confirmation": (
        "surrogate_rollout.optimization.confirmation_evaluator."
        "HistoryAwareDVDConfirmationEvaluator"),
    "intervention": (
        "surrogate_rollout.optimization.property_intervention."
        "PropertyInterventionBatchRunner"),
}
_REAL_QA_FN = "surrogate_rollout.evaluation.dvd_qa.run_dvd_qa"


class StartupModelValidationError(RuntimeError):
    """A real run has a missing, conflicting, or non-real model component."""


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _metadata(value: Any) -> dict[str, Any]:
    metadata = getattr(value, "metadata", None)
    if callable(metadata):
        result = metadata()
        if isinstance(result, Mapping):
            stable = dict(result)
            stable.pop("call_count", None)
            return as_json_dict(stable)
    identity = getattr(value, "configuration_identity", None)
    if identity is not None:
        return as_json_dict(identity)
    return {
        "provider": _type_name(value),
        "model": getattr(value, "model", None),
        "real_model": False,
    }


def _require_type(name: str, value: Any) -> None:
    expected = _REAL_TYPES[name]
    actual = _type_name(value)
    if actual != expected:
        raise StartupModelValidationError(
            f"{name} must use {expected}, got {actual}")


def _callable_name(value: Any) -> str:
    return f"{value.__module__}.{getattr(value, '__qualname__', type(value).__qualname__)}"


def _require_real(name: str, value: Mapping[str, Any], *, provider: str | None = None) \
        -> dict[str, Any]:
    result = as_json_dict(value)
    model = result.get("model")
    if not isinstance(model, str) or not model.strip():
        raise StartupModelValidationError(f"{name} has no model name")
    if result.get("real_model") is not True:
        raise StartupModelValidationError(f"{name} is not marked as a real model")
    if provider is not None and result.get("provider") != provider:
        raise StartupModelValidationError(
            f"{name} must use provider {provider!r}, got {result.get('provider')!r}")
    serialized = dumps_canonical(result).lower()
    marker = next((item for item in _DISALLOWED_MARKERS if item in serialized), None)
    if marker:
        raise StartupModelValidationError(
            f"{name} contains disallowed non-real marker {marker!r}")
    return result


def build_startup_model_manifest(
    *, iteration_id: str, baseline_runner: Any, intervention_runner: Any,
    feedback_runner: Any, confirmation_evaluator: Any,
) -> dict[str, Any]:
    """Resolve and validate every model-bearing component before stage calls."""
    confirmation_builder = getattr(confirmation_evaluator, "builder", None)
    baseline_builder = getattr(baseline_runner, "history_aware_builder", None)
    if confirmation_builder is None or baseline_builder is None:
        raise StartupModelValidationError(
            "real execution requires explicit history-aware baseline and confirmation builders")
    if baseline_builder is not confirmation_builder:
        raise StartupModelValidationError(
            "baseline and confirmation must share the same history-aware builder")
    _require_type("confirmation", confirmation_evaluator)
    _require_type("intervention", intervention_runner)

    router = getattr(confirmation_builder, "router", None)
    captioner = getattr(confirmation_builder, "segment_captioner", None)
    if router is None or captioner is None:
        raise StartupModelValidationError("router or captioner is missing")
    _require_type("router", router)
    _require_type("captioner", captioner)
    intervention_captioner = getattr(intervention_runner, "segment_captioner", None)
    if intervention_captioner is not captioner:
        raise StartupModelValidationError(
            "baseline, intervention, and confirmation must share the same captioner")

    proposal_policy = getattr(baseline_runner, "proposal_policy", None)
    proposal_provider = getattr(proposal_policy, "response_provider", None)
    feedback_provider = getattr(feedback_runner, "response_provider", None)
    if proposal_provider is None or feedback_provider is None:
        raise StartupModelValidationError(
            "property proposer or feedback provider is missing")
    _require_type("property_proposer", proposal_policy)
    _require_type("feedback_provider", feedback_runner)

    router_identity = _require_real("router", _metadata(router))
    captioner_identity = _require_real("captioner", _metadata(captioner))
    proposer_identity = _require_real(
        "property_proposer", _metadata(proposal_provider), provider="openai_api")
    feedback_identity = _require_real(
        "feedback_provider", _metadata(feedback_provider), provider="openai_api")

    qa_configuration = getattr(confirmation_evaluator, "_qa_configuration", None)
    if not isinstance(qa_configuration, Mapping):
        raise StartupModelValidationError("downstream QA configuration is missing")
    qa_fn = str(qa_configuration.get("qa_fn") or "")
    if qa_fn != _REAL_QA_FN:
        raise StartupModelValidationError(
            f"downstream QA must use the real DVD path, got {qa_fn!r}")
    for owner_name, owner in (
        ("baseline", baseline_runner), ("intervention", intervention_runner),
    ):
        owner_qa = getattr(owner, "qa_fn", None)
        if owner_qa is None or _callable_name(owner_qa) != _REAL_QA_FN:
            raise StartupModelValidationError(
                f"{owner_name} downstream QA is not the real DVD path")
    tool_model = qa_configuration.get("orchestrator_tool_model")
    fallback_model = qa_configuration.get("text_fallback_model")
    downstream_identity = _require_real("downstream_qa", {
        "provider": "dvd_reasoning_path",
        "model": str(tool_model or ""),
        "fallback_model": str(fallback_model or ""),
        "qa_fn": qa_fn,
        "real_model": True,
    })
    if not isinstance(fallback_model, str) or not fallback_model:
        raise StartupModelValidationError(
            "downstream QA text fallback model name is missing")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "validated_real_models",
        "iteration_id": iteration_id,
        "models": {
            "router": router_identity,
            "captioner": captioner_identity,
            "property_proposer": proposer_identity,
            "feedback_provider": feedback_identity,
            "downstream_qa": downstream_identity,
        },
    }


def log_startup_models(output_dir: str, manifest: Mapping[str, Any]) -> str:
    """Persist an immutable audit record and print it on every invocation."""
    output_dir = os.path.abspath(output_dir)
    path = os.path.join(output_dir, "startup_models.json")
    payload = as_json_dict(manifest)
    if os.path.exists(path):
        with open(path) as handle:
            existing = json.load(handle)
        if existing != payload:
            raise StartupModelValidationError(
                "startup model manifest conflicts with the existing run")
    else:
        os.makedirs(output_dir, exist_ok=True)
        _atomic_write_text(
            path, json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
    print("STARTUP_MODELS " + dumps_canonical(payload), flush=True)
    return path


def startup_model_hash(manifest: Mapping[str, Any]) -> str:
    return sha256_text(dumps_canonical(manifest))
