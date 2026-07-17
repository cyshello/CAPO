"""Bounded real OpenAI JSON provider for Phase 4 component-update planners."""

from __future__ import annotations

from surrogate_rollout import config
from surrogate_rollout.optimization.policies.codex_feedback import (
    _load_openai_key,
    _openai_chat,
)


class OpenAIJSONUpdateProvider:
    """Two-message planner adapter with an explicit per-process call bound."""

    policy_version = "openai_json_component_update_v1"

    def __init__(
        self, *, model: str = config.FEEDBACK_MODEL,
        api_key: str | None = None, max_calls: int = 1,
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.model = model
        self.api_key = api_key
        self.max_calls = max_calls
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
            api_key=self.api_key or _load_openai_key())

    def metadata(self) -> dict:
        return {
            "provider": "openai_api", "model": self.model,
            "policy_version": self.policy_version,
            "max_calls": self.max_calls, "call_count": self.call_count,
            "real_model": True,
        }
