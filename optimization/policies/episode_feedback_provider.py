"""Lean episode-feedback provider and exact-token inspection boundary.

This module formats an OpenAI-compatible strict-JSON chat request, but never
selects a provider, model, tokenizer, context window, decoding setting, or
transport.  Every such value is constructor-injected.  Inspection builds and
measures the exact request body without calling the injected transport.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from surrogate_rollout.optimization.llm_episode_feedback import (
    EpisodeFeedbackArtifactResolver,
    EpisodeFeedbackBackendConfigurationError,
    LeanEpisodeFeedbackRequest,
    build_lean_episode_feedback_request,
)
from surrogate_rollout.optimization.schemas import InterventionEpisode
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json


EPISODE_FEEDBACK_STRICT_SCHEMA_NAME = \
    "episode_feedback_response_v5_mixed_view_memory"


class EpisodeFeedbackProviderNotConfiguredError(
        EpisodeFeedbackBackendConfigurationError):
    """A required D2 provider or tokenizer setting was not supplied."""


class EpisodeFeedbackProviderContextOverflowError(ValueError):
    """The exact provider request plus reserved output exceeds context."""

    def __init__(self, statistics: ExactEpisodeFeedbackTokenStatistics) -> None:
        self.statistics = statistics
        super().__init__(
            "episode feedback provider request exceeds configured context: "
            f"input_tokens={statistics.total_input_tokens}, "
            f"reserved_output_tokens={statistics.reserved_output_tokens}, "
            f"context_limit={statistics.context_limit}, "
            f"remaining_tokens={statistics.remaining_tokens}; no truncation, "
            "summarization, filtering, sampling, or splitting was performed")


@dataclass(frozen=True)
class ExactProviderInputTokenCount:
    """Exact tokenizer result for the final provider message sequence."""

    system_prompt_tokens: int
    user_payload_tokens: int
    total_input_tokens: int

    def __post_init__(self) -> None:
        for name in (
                "system_prompt_tokens", "user_payload_tokens",
                "total_input_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TypeError(f"{name} must be a non-negative integer")
        if self.total_input_tokens < max(
                self.system_prompt_tokens, self.user_payload_tokens):
            raise ValueError(
                "total_input_tokens cannot be smaller than a message count")


class ExactEpisodeFeedbackTokenCounter(Protocol):
    def __call__(
        self, messages: tuple[Mapping[str, str], ...],
    ) -> ExactProviderInputTokenCount:
        ...


class EpisodeFeedbackResponseTransport(Protocol):
    def __call__(self, request_body: Mapping[str, Any]) -> str:
        ...


@dataclass(frozen=True)
class ExactEpisodeFeedbackTokenStatistics:
    system_prompt_tokens: int
    user_payload_tokens: int
    total_input_tokens: int
    reserved_output_tokens: int
    context_limit: int
    remaining_tokens: int
    fits_context: bool


@dataclass(frozen=True)
class PreparedEpisodeFeedbackProviderRequest:
    provider: str
    model_id: str
    tokenizer_identity: str
    feedback_policy_version: str
    messages: tuple[Mapping[str, str], ...]
    request_body: Mapping[str, Any]
    serialized_request: str
    response_schema: Mapping[str, Any]
    request_identity: str


@dataclass(frozen=True)
class EpisodeFeedbackRequestInspection:
    lean_request: LeanEpisodeFeedbackRequest
    prepared_request: PreparedEpisodeFeedbackProviderRequest
    token_statistics: ExactEpisodeFeedbackTokenStatistics
    fits_context: bool


def episode_feedback_response_json_schema() -> dict[str, Any]:
    """Strict provider schema; parser still owns final validation and ID."""
    string_array = {"type": "array", "items": {"type": "string"}}
    confidence = {
        "type": "string",
        "description": (
            "An open-vocabulary confidence assessment. No fixed scale or "
            "enum is imposed."),
    }
    evidence = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "statement", "supporting_segment_ids", "supporting_qa_ids",
            "evidence_type", "confidence",
        ],
        "properties": {
            "statement": {"type": "string"},
            "supporting_segment_ids": string_array,
            "supporting_qa_ids": string_array,
            "evidence_type": {"type": "string", "enum": [
                "caption_change", "trajectory", "mixed"]},
            # Provider JSON is a string while its vocabulary remains opaque.
            "confidence": dict(confidence),
        },
    }
    fields = {
        "episode_id": {"type": "string"},
        "outcome_summary": {"type": "string"},
        "observations": {"type": "array", "items": evidence},
        "counterevidence": {"type": "array", "items": evidence},
        "generator_diagnosis": {"type": "string"},
        "recommended_strategy_change": {"type": "string"},
        "confidence": dict(confidence),
        "compact_memory_text": {
            "type": ["string", "null"],
            "description": (
                "A short provider-authored experience for historical memory; "
                "null only when no non-empty memory can be supplied."),
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(fields),
        "properties": fields,
    }


def _configured_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EpisodeFeedbackProviderNotConfiguredError(
            f"{name} must be explicitly configured")
    return value


def _configured_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise EpisodeFeedbackProviderNotConfiguredError(
            f"{name} must be explicitly configured as a positive integer")
    return value


class OpenAICompatibleEpisodeFeedbackProviderAdapter:
    """Explicit strict-chat adapter implementing the D1 backend protocol."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        model_id: str | None = None,
        tokenizer_identity: str | None = None,
        exact_token_counter: ExactEpisodeFeedbackTokenCounter | None = None,
        context_limit: int | None = None,
        maximum_output_tokens: int | None = None,
        generation_settings: Mapping[str, Any] | None = None,
        feedback_policy_version: str | None = None,
        response_transport: EpisodeFeedbackResponseTransport | None = None,
        context_safety_margin_tokens: int = 0,
    ) -> None:
        self.provider = _configured_string(provider, "provider")
        self.model_id = _configured_string(model_id, "model_id")
        self.tokenizer_identity = _configured_string(
            tokenizer_identity, "tokenizer_identity")
        if not callable(exact_token_counter):
            raise EpisodeFeedbackProviderNotConfiguredError(
                "exact_token_counter must be explicitly configured")
        self.exact_token_counter = exact_token_counter
        self.context_limit = _configured_positive_int(
            context_limit, "context_limit")
        self.maximum_output_tokens = _configured_positive_int(
            maximum_output_tokens, "maximum_output_tokens")
        if not isinstance(context_safety_margin_tokens, int) or isinstance(
                context_safety_margin_tokens, bool) or \
                context_safety_margin_tokens < 0:
            raise EpisodeFeedbackProviderNotConfiguredError(
                "context_safety_margin_tokens must be a non-negative integer")
        self.context_safety_margin_tokens = context_safety_margin_tokens
        if not isinstance(generation_settings, Mapping) or not \
                generation_settings:
            raise EpisodeFeedbackProviderNotConfiguredError(
                "generation_settings must be an explicit non-empty object")
        reserved = {"model", "messages", "max_tokens", "response_format"}
        overlap = reserved.intersection(generation_settings)
        if overlap:
            raise EpisodeFeedbackBackendConfigurationError(
                "generation_settings may not replace adapter-owned fields: "
                f"{sorted(overlap)!r}")
        self.generation_settings = json.loads(dumps_canonical(
            generation_settings))
        self.feedback_policy_version = _configured_string(
            feedback_policy_version, "feedback_policy_version")
        if not callable(response_transport):
            raise EpisodeFeedbackProviderNotConfiguredError(
                "response_transport must be explicitly configured")
        self.response_transport = response_transport
        self.call_count = 0
        self.response_schema = episode_feedback_response_json_schema()

    def prepare_messages(
        self, system_instruction: str, user_payload: str,
    ) -> tuple[PreparedEpisodeFeedbackProviderRequest,
               ExactEpisodeFeedbackTokenStatistics]:
        if not isinstance(system_instruction, str) or not system_instruction:
            raise TypeError("system_instruction must be a non-empty string")
        if not isinstance(user_payload, str) or not user_payload:
            raise TypeError("user_payload must be a non-empty string")
        messages = (
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_payload},
        )
        body = {
            "model": self.model_id,
            "messages": [dict(item) for item in messages],
            "max_tokens": self.maximum_output_tokens,
            **self.generation_settings,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": EPISODE_FEEDBACK_STRICT_SCHEMA_NAME,
                    "strict": True,
                    "schema": self.response_schema,
                },
            },
        }
        canonical_body = json.loads(dumps_canonical(body))
        exact = self.exact_token_counter(messages)
        if not isinstance(exact, ExactProviderInputTokenCount):
            raise TypeError(
                "exact_token_counter must return ExactProviderInputTokenCount")
        remaining = (
            self.context_limit - exact.total_input_tokens -
            self.maximum_output_tokens)
        statistics = ExactEpisodeFeedbackTokenStatistics(
            system_prompt_tokens=exact.system_prompt_tokens,
            user_payload_tokens=exact.user_payload_tokens,
            total_input_tokens=exact.total_input_tokens,
            reserved_output_tokens=self.maximum_output_tokens,
            context_limit=self.context_limit,
            remaining_tokens=remaining,
            fits_context=(remaining >= self.context_safety_margin_tokens),
        )
        identity_input = {
            "provider": self.provider,
            "model_id": self.model_id,
            "tokenizer_identity": self.tokenizer_identity,
            "feedback_policy_version": self.feedback_policy_version,
            "context_limit": self.context_limit,
            "maximum_output_tokens": self.maximum_output_tokens,
            "request_body": canonical_body,
        }
        prepared = PreparedEpisodeFeedbackProviderRequest(
            provider=self.provider,
            model_id=self.model_id,
            tokenizer_identity=self.tokenizer_identity,
            feedback_policy_version=self.feedback_policy_version,
            messages=messages,
            request_body=canonical_body,
            serialized_request=dumps_canonical(canonical_body),
            response_schema=json.loads(dumps_canonical(self.response_schema)),
            request_identity="episode_feedback_request_" + sha256_json(
                identity_input)[:20],
        )
        return prepared, statistics

    def count_tokens(
        self, messages: tuple[Mapping[str, str], ...],
    ) -> int:
        if len(messages) != 2 or tuple(
                item.get("role") for item in messages) != ("system", "user"):
            raise TypeError("episode feedback requires system and user messages")
        _, statistics = self.prepare_messages(
            messages[0]["content"], messages[1]["content"])
        return statistics.total_input_tokens

    def __call__(self, system_instruction: str, user_payload: str) -> str:
        prepared, statistics = self.preflight(system_instruction, user_payload)
        self.call_count += 1
        raw_response = self.response_transport(prepared.request_body)
        if not isinstance(raw_response, str):
            raise TypeError("response_transport must return a string")
        return raw_response

    def preflight(
        self, system_instruction: str, user_payload: str,
    ) -> tuple[PreparedEpisodeFeedbackProviderRequest,
               ExactEpisodeFeedbackTokenStatistics]:
        """Validate the exact budget without invoking the model transport."""
        prepared, statistics = self.prepare_messages(
            system_instruction, user_payload)
        if not statistics.fits_context:
            raise EpisodeFeedbackProviderContextOverflowError(statistics)
        return prepared, statistics

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model_id,
            "tokenizer_identity": self.tokenizer_identity,
            "generation_settings": self.generation_settings,
            "output_token_limit": self.maximum_output_tokens,
            "context_limit_tokens": self.context_limit,
            "feedback_policy_version": self.feedback_policy_version,
            "response_format": "strict_json_schema",
            "response_schema_name": EPISODE_FEEDBACK_STRICT_SCHEMA_NAME,
            "response_schema": self.response_schema,
            "call_count": self.call_count,
        }


def prepare_and_measure(
    episode: InterventionEpisode,
    *,
    provider_adapter: OpenAICompatibleEpisodeFeedbackProviderAdapter,
    artifact_resolver: EpisodeFeedbackArtifactResolver,
) -> EpisodeFeedbackRequestInspection:
    """Build and exactly measure the lean request without calling transport."""
    if not isinstance(
            provider_adapter, OpenAICompatibleEpisodeFeedbackProviderAdapter):
        raise TypeError(
            "provider_adapter must be an explicit episode feedback adapter")
    request = build_lean_episode_feedback_request(
        episode, artifact_resolver=artifact_resolver)
    prepared, statistics = provider_adapter.prepare_messages(
        request.system_instruction, request.user_request)
    return EpisodeFeedbackRequestInspection(
        lean_request=request,
        prepared_request=prepared,
        token_statistics=statistics,
        fits_context=statistics.fits_context,
    )
