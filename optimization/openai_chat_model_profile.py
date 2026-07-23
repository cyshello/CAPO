"""Chat-completions request shaping per OpenAI model family.

The GPT-5 family rejects `temperature` and `top_p`, requires
`max_completion_tokens` where earlier models took `max_tokens`, and accepts a
`reasoning_effort` control. Every adapter in this repository builds the same
`{model, messages, max_tokens, **generation_settings, response_format}` body,
so the family difference is resolved in one place instead of five.

Bodies for non-reasoning models are returned unchanged, so switching a model id
is the only thing that changes a request.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
REASONING_EFFORTS = ("minimal", "low", "medium", "high")

# Rejected outright by the reasoning models rather than ignored.
_UNSUPPORTED_BY_REASONING_MODELS = ("temperature", "top_p")


class ChatModelProfileError(ValueError):
    """Raised for an unusable model id or reasoning-effort value."""


def is_reasoning_chat_model(model_id: str) -> bool:
    """True when `model_id` names a model that takes `reasoning_effort`."""
    if not isinstance(model_id, str) or not model_id:
        raise ChatModelProfileError("model_id must be a non-empty string")
    return model_id.startswith(REASONING_MODEL_PREFIXES)


def validate_reasoning_effort(effort: str | None) -> str | None:
    """Return `effort` unchanged, or raise when it is not a known level."""
    if effort is None:
        return None
    if effort not in REASONING_EFFORTS:
        raise ChatModelProfileError(
            f"reasoning_effort must be one of {REASONING_EFFORTS}: {effort!r}")
    return effort


def adapt_chat_completions_body(
    body: Mapping[str, Any], *, reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Return `body` shaped for the family its `model` belongs to.

    Non-reasoning models get an unchanged copy, including any `reasoning_effort`
    argument, which they would reject. Reasoning models lose the sampling
    controls they reject, take `max_completion_tokens` in place of `max_tokens`,
    and carry `reasoning_effort` when one was requested.
    """
    if not isinstance(body, Mapping):
        raise ChatModelProfileError("body must be a mapping")
    adapted = dict(body)
    model_id = adapted.get("model")
    if not isinstance(model_id, str) or not model_id:
        raise ChatModelProfileError("body must carry a non-empty model id")
    validate_reasoning_effort(reasoning_effort)
    if not is_reasoning_chat_model(model_id):
        # A configured effort must not reach a model that rejects the field.
        adapted.pop("reasoning_effort", None)
        return adapted
    for field in _UNSUPPORTED_BY_REASONING_MODELS:
        adapted.pop(field, None)
    if "max_tokens" in adapted:
        # The reasoning budget is spent from the same allowance, so this cap is
        # no longer a bound on visible output alone.
        adapted["max_completion_tokens"] = adapted.pop("max_tokens")
    if reasoning_effort is not None:
        adapted["reasoning_effort"] = reasoning_effort
    return adapted
