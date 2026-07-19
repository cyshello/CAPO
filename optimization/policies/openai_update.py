"""Bounded real OpenAI JSON provider for Phase 4 component-update planners."""

from __future__ import annotations

from surrogate_rollout import config
from surrogate_rollout.optimization.policies.codex_feedback import (
    _load_openai_key,
    _openai_chat,
)


OPENAI_STRICT_UPDATE_POLICY_VERSION = "openai_strict_component_update_v3"


def codebook_updater_response_schema(*, max_actions: int) -> dict:
    string_array = {"type": "array", "items": {"type": "string"}}
    nullable_string = {"type": ["string", "null"]}
    action_fields = {
        "action_id": {"type": "string"},
        "action": {"type": "string", "enum": [
            "add", "revise", "merge", "preserve", "retire", "no_op"]},
        "target_property_ids": string_array,
        "candidate_ids": string_array,
        "proposed_property_text": nullable_string,
        "reasoning": {"type": "string"},
        "supporting_memory_example_ids": string_array,
        "supporting_evidence_ids": string_array,
        "behaviors_to_preserve": string_array,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["schema_version", "actions"],
        "properties": {
            "schema_version": {
                "type": "string",
                "enum": ["memory_codebook_updater_response_v2"]},
            "actions": {
                "type": "array", "maxItems": max_actions,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": list(action_fields), "properties": action_fields,
                },
            },
        },
    }


def router_updater_response_schema(*, max_actions: int) -> dict:
    string_array = {"type": "array", "items": {"type": "string"}}
    action_fields = {
        "action_id": {"type": "string"},
        "target_property_id": {"type": "string"},
        "action_type": {"type": "string", "enum": [
            "set_selection_guidance", "set_avoidance_guidance",
            "add_positive_example", "add_negative_example", "preserve", "no_op"]},
        "proposed_value": {"type": ["string", "null"]},
        "reason": {"type": "string"},
        "supporting_memory_example_ids": string_array,
        "supporting_evidence_ids": string_array,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["schema_version", "actions"],
        "properties": {
            "schema_version": {
                "type": "string",
                "enum": ["memory_router_updater_response_v1"]},
            "actions": {
                "type": "array", "maxItems": max_actions,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": list(action_fields), "properties": action_fields,
                },
            },
        },
    }


class OpenAIJSONUpdateProvider:
    """Two-message planner adapter with an explicit per-process call bound."""

    policy_version = OPENAI_STRICT_UPDATE_POLICY_VERSION

    def __init__(
        self, *, model: str = config.FEEDBACK_MODEL,
        api_key: str | None = None, max_calls: int = 3,
        response_schema_name: str,
        response_schema: dict,
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.model = model
        self.api_key = api_key
        self.max_calls = max_calls
        self.response_schema_name = response_schema_name
        self.response_schema = response_schema
        self.call_count = 0

    def __call__(self, system_prompt: str, request: str) -> str:
        if self.call_count >= self.max_calls:
            raise RuntimeError("component-update provider call limit reached")
        self.call_count += 1
        prompt = (
            system_prompt.rstrip() +
            "\n\nBounded versioned updater request:\n" + request)
        return _openai_chat(
            prompt, model=self.model,
            api_key=self.api_key or _load_openai_key(),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": self.response_schema_name,
                    "strict": True,
                    "schema": self.response_schema,
                },
            })

    def metadata(self) -> dict:
        return {
            "provider": "openai_api", "model": self.model,
            "policy_version": self.policy_version,
            "max_calls": self.max_calls, "call_count": self.call_count,
            "response_format": "strict_json_schema",
            "response_schema_name": self.response_schema_name,
            "response_schema": self.response_schema,
            "real_model": True,
        }
