"""Checkpoint E2 one-call provider execution and provisional persistence.

This module is intentionally separate from the provider-independent E1
updater.  It writes one reviewable local run directory, never promotes a
candidate, and never mutates parent or feedback artifacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from surrogate_rollout.optimization.meta_prompt_updater import (
    LLMMetaPromptUpdater,
    MetaPromptUpdateResult,
    build_meta_prompt_update_request,
    meta_prompt_update_response_json_schema,
)
from surrogate_rollout.optimization.context_budget import (
    ContextTruncationResult,
    fit_json_payload_to_token_budget,
)
from surrogate_rollout.optimization.schemas import (
    EpisodeFeedback,
    MetaPromptVersion,
    episode_feedback_from_json,
    meta_prompt_version_from_json,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json


META_PROMPT_UPDATE_STRICT_SCHEMA_NAME = "meta_prompt_update_response_v1"

_PARENT_FIELDS = {
    "meta_prompt_id", "parent_meta_prompt_id", "text", "created_at", "status",
}
_FEEDBACK_FIELDS = {
    "feedback_id", "episode_id", "outcome_summary", "observations",
    "counterevidence", "generator_diagnosis", "recommended_strategy_change",
    "confidence",
}
_EVIDENCE_FIELDS = {
    "statement", "supporting_segment_ids", "supporting_qa_ids",
    "evidence_type", "transition_type", "confidence",
}


class MetaPromptUpdateExecutionError(RuntimeError):
    """A one-call E2 run failed; its new output directory is inspectable."""

    def __init__(self, reason: str, *, output_directory: Path) -> None:
        self.output_directory = output_directory
        super().__init__(reason)


class MetaPromptUpdateProviderTransport(Protocol):
    def __call__(self, request_body: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class PreparedMetaPromptUpdateProviderRequest:
    provider: str
    model_id: str
    updater_policy_version: str
    messages: tuple[Mapping[str, str], ...]
    request_body: Mapping[str, Any]
    serialized_request: str
    request_hash: str
    response_schema: Mapping[str, Any]


def _configured_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be explicitly configured")
    return value


def _configured_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class OpenAICompatibleMetaPromptUpdaterBackend:
    """Explicit strict-chat backend implementing the E1 callable boundary.

    The injected transport performs the provider I/O.  This adapter invokes it
    at most once and does not implement retries, JSON repair, or fallback.
    """

    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        maximum_output_tokens: int,
        generation_settings: Mapping[str, Any],
        updater_policy_version: str,
        response_transport: MetaPromptUpdateProviderTransport,
        tokenizer_identity: str | None = None,
        exact_token_counter: Any | None = None,
        context_limit: int | None = None,
        context_safety_margin_tokens: int = 0,
    ) -> None:
        self.provider = _configured_string(provider, "provider")
        self.model_id = _configured_string(model_id, "model_id")
        self.maximum_output_tokens = _configured_positive_int(
            maximum_output_tokens, "maximum_output_tokens")
        if not isinstance(generation_settings, Mapping) or not generation_settings:
            raise ValueError("generation_settings must be an explicit object")
        reserved = {"model", "messages", "max_tokens", "response_format"}
        overlap = reserved.intersection(generation_settings)
        if overlap:
            raise ValueError(
                "generation_settings may not replace adapter-owned fields: "
                f"{sorted(overlap)}")
        self.generation_settings = json.loads(dumps_canonical(
            generation_settings))
        self.updater_policy_version = _configured_string(
            updater_policy_version, "updater_policy_version")
        if not callable(response_transport):
            raise ValueError("response_transport must be explicitly configured")
        self.response_transport = response_transport
        configured_budget = (
            tokenizer_identity is not None,
            exact_token_counter is not None,
            context_limit is not None,
        )
        if any(configured_budget) and not all(configured_budget):
            raise ValueError(
                "tokenizer_identity, exact_token_counter, and context_limit "
                "must be configured together")
        self.tokenizer_identity = tokenizer_identity
        self.exact_token_counter = exact_token_counter
        self.context_limit = context_limit
        if not isinstance(context_safety_margin_tokens, int) or isinstance(
                context_safety_margin_tokens, bool) or \
                context_safety_margin_tokens < 0:
            raise ValueError(
                "context_safety_margin_tokens must be a non-negative integer")
        self.context_safety_margin_tokens = context_safety_margin_tokens
        if context_limit is not None:
            _configured_positive_int(context_limit, "context_limit")
            if not callable(exact_token_counter):
                raise ValueError("exact_token_counter must be callable")
        self.call_count = 0
        self.response_schema = meta_prompt_update_response_json_schema()
        self.last_prepared_request: PreparedMetaPromptUpdateProviderRequest | None = None
        self.last_provider_response: Mapping[str, Any] | None = None
        self.last_raw_response: str | None = None
        self.last_usage: Mapping[str, Any] = {}
        self.last_context_truncation: Mapping[str, Any] | None = None

    def prepare_messages(
        self, system_instruction: str, user_request: str,
    ) -> PreparedMetaPromptUpdateProviderRequest:
        if not isinstance(system_instruction, str) or not system_instruction:
            raise TypeError("system_instruction must be a non-empty string")
        if not isinstance(user_request, str) or not user_request:
            raise TypeError("user_request must be a non-empty string")
        messages = (
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_request},
        )
        body = json.loads(dumps_canonical({
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.maximum_output_tokens,
            **self.generation_settings,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": META_PROMPT_UPDATE_STRICT_SCHEMA_NAME,
                    "strict": True,
                    "schema": self.response_schema,
                },
            },
        }))
        serialized = dumps_canonical(body)
        return PreparedMetaPromptUpdateProviderRequest(
            provider=self.provider,
            model_id=self.model_id,
            updater_policy_version=self.updater_policy_version,
            messages=messages,
            request_body=body,
            serialized_request=serialized,
            request_hash=sha256_json(body),
            response_schema=json.loads(dumps_canonical(self.response_schema)),
        )

    def __call__(self, system_instruction: str, user_request: str) -> str:
        if self.call_count:
            raise RuntimeError("updater backend is limited to exactly one call")
        user_request, fitted = self.fit_user_request(
            system_instruction, user_request)
        prepared = self.prepare_messages(system_instruction, user_request)
        self.last_prepared_request = prepared
        self.call_count += 1
        envelope = self.response_transport(prepared.request_body)
        if not isinstance(envelope, Mapping):
            raise TypeError("provider response envelope must be an object")
        self.last_provider_response = json.loads(dumps_canonical(envelope))
        usage = envelope.get("usage", {})
        if not isinstance(usage, Mapping):
            raise TypeError("provider usage must be an object when present")
        self.last_usage = json.loads(dumps_canonical(usage))
        try:
            raw = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "provider response does not contain choices[0].message.content") \
                from exc
        if not isinstance(raw, str):
            raise TypeError("provider response content must be a string")
        self.last_raw_response = raw
        return raw

    def fit_user_request(
        self, system_instruction: str, user_request: str,
    ) -> tuple[str, ContextTruncationResult | None]:
        if self.context_limit is None:
            return user_request, None
        try:
            payload = json.loads(user_request)
        except json.JSONDecodeError as exc:
            raise TypeError("updater user request must be strict JSON") from exc
        if not isinstance(payload, Mapping):
            raise TypeError("updater user request must be a JSON object")

        def measure(text: str) -> int:
            messages = (
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": text},
            )
            value = self.exact_token_counter(messages)
            total = getattr(value, "total_input_tokens", value)
            if not isinstance(total, int) or isinstance(total, bool):
                raise TypeError("exact_token_counter must return an exact count")
            return total

        maximum_input_tokens = (
            int(self.context_limit) - self.maximum_output_tokens -
            self.context_safety_margin_tokens)
        if maximum_input_tokens <= 0:
            raise ValueError(
                "context limit cannot cover output reservation and provider "
                "safety margin")
        fitted = fit_json_payload_to_token_budget(
            payload,
            measure_input_tokens=measure,
            maximum_input_tokens=maximum_input_tokens,
        )
        audit = fitted.audit_metadata()
        previous = self.last_context_truncation
        if not (previous and previous.get("transmitted_payload_hash") ==
                audit["original_payload_hash"] and
                previous.get("original_payload_hash") !=
                previous.get("transmitted_payload_hash")):
            self.last_context_truncation = audit
        return fitted.serialized_payload, fitted

    def metadata(self) -> Mapping[str, Any]:
        value: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model_id,
            "maximum_output_tokens": self.maximum_output_tokens,
            "generation_settings": self.generation_settings,
            "updater_policy_version": self.updater_policy_version,
            "provider_call_count": self.call_count,
            "tokenizer_identity": self.tokenizer_identity,
            "context_limit": self.context_limit,
            "context_truncation": self.last_context_truncation,
        }
        if self.last_provider_response is not None:
            value["provider_response_id"] = self.last_provider_response.get("id")
            value["provider_model"] = self.last_provider_response.get("model")
        return value


def _write_once(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    _atomic_write_text(str(path), text)


def _write_json_once(path: Path, value: Any) -> None:
    _write_once(path, dumps_canonical(value))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_records(paths: Sequence[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path.resolve()), "sha256": _sha256_file(path.resolve())}
        for path in paths
    ]


def _strict_object(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly {sorted(fields)}")
    return value


def _load_parent(path: Path) -> MetaPromptVersion:
    value = _strict_object(
        json.loads(path.read_text(encoding="utf-8")), _PARENT_FIELDS,
        "parent meta-prompt artifact")
    return meta_prompt_version_from_json(value)


def _load_feedback(path: Path) -> EpisodeFeedback:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) not in (
            _FEEDBACK_FIELDS, _FEEDBACK_FIELDS | {"compact_memory_text"}):
        raise ValueError(
            "episode feedback artifact must contain the detailed fields and "
            "only the optional compact_memory_text field")
    for collection in ("observations", "counterevidence"):
        if not isinstance(value[collection], list):
            raise TypeError(f"EpisodeFeedback.{collection} must be an array")
        for index, evidence in enumerate(value[collection]):
            _strict_object(
                evidence, _EVIDENCE_FIELDS,
                f"EpisodeFeedback.{collection}[{index}]")
    return episode_feedback_from_json(value)


def _failure_text(exc: BaseException, backend: Any) -> str:
    raw = getattr(exc, "raw_response", None)
    if raw is None:
        raw = getattr(backend, "last_raw_response", None)
    if raw is None:
        raw = getattr(exc, "raw_error", None)
    if raw is None:
        return f"{type(exc).__name__}: {exc}"
    return raw if isinstance(raw, str) else dumps_canonical(raw)


def execute_meta_prompt_update_once(
    *,
    parent_artifact_path: str | Path,
    feedback_artifact_paths: Sequence[str | Path],
    output_directory: str | Path,
    backend: OpenAICompatibleMetaPromptUpdaterBackend,
    updater_policy_version: str,
    candidate_created_at: str,
) -> MetaPromptUpdateResult:
    """Execute one call and atomically materialize a review-only E2 run."""
    output = Path(output_directory).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise MetaPromptUpdateExecutionError(
            f"output directory already exists; refusing overwrite: {output}",
            output_directory=output) from exc
    parent_path = Path(parent_artifact_path).resolve()
    feedback_paths = tuple(Path(item).resolve() for item in feedback_artifact_paths)
    source_paths = (parent_path, *feedback_paths)
    source_before: list[dict[str, str]] = []
    input_manifest: dict[str, Any] = {}
    try:
        if not feedback_paths:
            raise ValueError("one or more ordered feedback artifacts are required")
        if not isinstance(candidate_created_at, str) or not candidate_created_at:
            raise ValueError("candidate_created_at must be explicitly configured")
        if backend.updater_policy_version != updater_policy_version:
            raise ValueError(
                "backend and execution updater_policy_version must match")
        source_before = _source_records(source_paths)
        parent = _load_parent(parent_path)
        feedbacks = tuple(_load_feedback(path) for path in feedback_paths)
        ids = tuple(item.feedback_id for item in feedbacks)
        if len(ids) != len(set(ids)):
            raise ValueError("source feedback IDs must be unique")
        request = build_meta_prompt_update_request(
            parent, feedbacks, updater_policy_version=updater_policy_version)
        prepared = backend.prepare_messages(
            request.system_instruction, request.user_request)
        input_manifest = {
            "schema_version": "meta_prompt_update_run_input_v1",
            "request_id": request.request_id,
            "request_payload_hash": request.payload_hash,
            "provider_request_hash": prepared.request_hash,
            "provider": backend.provider,
            "model": backend.model_id,
            "updater_policy_version": updater_policy_version,
            "ordered_feedback_ids": ids,
            "parent_meta_prompt_id": parent.meta_prompt_id,
            "source_artifacts": source_before,
        }
        _write_json_once(output / "updater_request.json", request)
        _write_once(output / "provider_request.json", prepared.serialized_request)
        _write_json_once(output / "input_manifest.json", input_manifest)

        updater = LLMMetaPromptUpdater(
            backend=backend, updater_policy_version=updater_policy_version)
        result = updater.update(parent, feedbacks)
        if backend.call_count != 1:
            raise RuntimeError(
                f"expected exactly one provider call, observed {backend.call_count}")
        if result.request.request_id != request.request_id:
            raise RuntimeError("executed updater request identity changed")
        _write_once(output / "raw_response.txt", result.raw_response or "")
        if backend.last_provider_response is not None:
            _write_json_once(
                output / "provider_response.json",
                backend.last_provider_response)
        _write_json_once(output / "usage.json", backend.last_usage)
        _write_json_once(
            output / "parsed_meta_prompt_update_result.json", result)

        if result.decision.decision == "update":
            candidate = MetaPromptVersion(
                meta_prompt_id=result.candidate_meta_prompt_id or "",
                parent_meta_prompt_id=parent.meta_prompt_id,
                text=result.decision.candidate_meta_prompt or "",
                created_at=candidate_created_at,
                status="provisional",
            )
            if candidate.parent_meta_prompt_id != parent.meta_prompt_id:
                raise ValueError("provisional candidate parent lineage mismatch")
            _write_json_once(output / "provisional_meta_prompt.json", candidate)
        else:
            _write_json_once(output / "no_update.json", {
                "decision": result.decision,
                "parent_meta_prompt_id": parent.meta_prompt_id,
                "request_id": request.request_id,
                "request_payload_hash": request.payload_hash,
            })

        source_after = _source_records(source_paths)
        if source_after != source_before:
            raise RuntimeError("source parent or feedback artifact was modified")
        _write_json_once(output / "run_manifest.json", {
            **input_manifest,
            "status": "succeeded",
            "decision": result.decision.decision,
            "candidate_meta_prompt_id": result.candidate_meta_prompt_id,
            "candidate_status": result.candidate_status,
            "provider_call_count": backend.call_count,
            "source_artifacts_after": source_after,
        })
        return result
    except BaseException as exc:
        try:
            if backend.last_raw_response is not None and not \
                    (output / "raw_response.txt").exists():
                _write_once(output / "raw_response.txt", backend.last_raw_response)
            if backend.last_provider_response is not None and not \
                    (output / "provider_response.json").exists():
                _write_json_once(
                    output / "provider_response.json",
                    backend.last_provider_response)
            if backend.last_usage and not (output / "usage.json").exists():
                _write_json_once(output / "usage.json", backend.last_usage)
            if not (output / "raw_error.txt").exists():
                _write_once(output / "raw_error.txt", _failure_text(exc, backend))
            source_after = (
                _source_records(source_paths) if source_before else [])
            if not (output / "run_manifest.json").exists():
                _write_json_once(output / "run_manifest.json", {
                    **input_manifest,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "provider_call_count": getattr(backend, "call_count", 0),
                    "source_artifacts_before": source_before,
                    "source_artifacts_after": source_after,
                })
        except BaseException:
            pass
        raise MetaPromptUpdateExecutionError(
            str(exc), output_directory=output) from exc
