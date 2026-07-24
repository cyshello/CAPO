"""Context-fitting wrapper for the episode-feedback provider (caption path).

The incumbent feedback provider hard-fails when one episode's request (its
changed-caption pairs plus the DVD tool evidence) exceeds the model context: it
performs no truncation by design. On the caption-prompt path we accept lossy
truncation instead, so this wrapper fits the per-episode user payload to the
budget before the provider's preflight can reject it.

It reuses the incumbent ``fit_json_payload_to_token_budget`` (prefix-truncates
only long dynamic string leaves; ids, hashes, schema/type/tool fields and all
structure are preserved) and the wrapped adapter's own exact token counter and
budget, so the fitted request is measured exactly the way the provider measures
it. Nothing in ``surrogate_rollout`` is modified; the wrapper only sits in front
of the adapter the caption factory already built.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from surrogate_rollout.optimization.context_budget import (
    fit_json_payload_to_token_budget,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


class TruncatingEpisodeFeedbackProvider:
    """Wrap a feedback provider adapter, fitting the user payload to context."""

    def __init__(self, inner: Any) -> None:
        for attr in ("exact_token_counter", "context_limit",
                     "maximum_output_tokens", "context_safety_margin_tokens"):
            if not hasattr(inner, attr):
                raise TypeError(
                    f"inner feedback provider is missing required attr {attr!r}")
        if not callable(inner):
            raise TypeError("inner feedback provider must be callable")
        self._inner = inner

    # ------------------------------- fitting ------------------------------ #
    def _fit_user(self, system_instruction: str, user_payload: str) -> str:
        sys_msg = {"role": "system", "content": system_instruction}

        def _measure_total(text: str) -> int:
            exact = self._inner.exact_token_counter(
                (sys_msg, {"role": "user", "content": text}))
            return int(exact.total_input_tokens)

        def _measure_user(text: str) -> int:
            exact = self._inner.exact_token_counter(
                (sys_msg, {"role": "user", "content": text}))
            return int(exact.user_payload_tokens)

        exact = self._inner.exact_token_counter(
            (sys_msg, {"role": "user", "content": user_payload}))
        margin = int(self._inner.context_safety_margin_tokens)
        budget_total = (int(self._inner.context_limit)
                        - int(self._inner.maximum_output_tokens) - margin)
        if int(exact.total_input_tokens) <= budget_total:
            return user_payload

        # Leave the system prompt and safety margin room; fit only the user
        # payload's dynamic strings down to what remains.
        target_user = budget_total - int(exact.system_prompt_tokens)
        if target_user <= 0:
            # The system prompt alone already overruns the budget; truncation of
            # the user payload cannot help, so let the provider raise its exact
            # overflow error rather than silently sending an empty payload.
            return user_payload
        try:
            parsed = json.loads(user_payload)
        except json.JSONDecodeError:
            return user_payload
        if not isinstance(parsed, Mapping):
            return user_payload
        result = fit_json_payload_to_token_budget(
            parsed, measure_input_tokens=_measure_user,
            maximum_input_tokens=target_user)
        fitted = result.serialized_payload
        # Guard: if the exact total still overruns (counter/serialization edge),
        # fall back to the original and let the provider decide.
        if _measure_total(fitted) > budget_total:
            return user_payload
        return fitted

    # ------------------------- provider interface ------------------------- #
    def __call__(self, system_instruction: str, user_payload: str) -> str:
        return self._inner(system_instruction,
                           self._fit_user(system_instruction, user_payload))

    def preflight(self, system_instruction: str, user_payload: str):
        return self._inner.preflight(
            system_instruction, self._fit_user(system_instruction, user_payload))

    def count_tokens(self, messages: tuple[Mapping[str, str], ...]) -> int:
        if len(messages) != 2:
            return self._inner.count_tokens(messages)
        system_instruction = messages[0].get("content", "")
        user_payload = messages[1].get("content", "")
        fitted = (dict(messages[0]),
                  {"role": messages[1].get("role", "user"),
                   "content": self._fit_user(system_instruction, user_payload)})
        return self._inner.count_tokens(tuple(fitted))

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (metadata, tokenizer_identity, ...) to inner.
        return getattr(self._inner, name)
