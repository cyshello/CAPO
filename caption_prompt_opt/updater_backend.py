"""Zero-touch system-prompt override for the meta-prompt updater backend.

The updater's system prompt is baked into ``build_meta_prompt_update_request``
via the module-level ``META_PROMPT_UPDATER_SYSTEM_INSTRUCTION`` constant, so it
cannot be injected through ``LLMMetaPromptUpdater``'s constructor. Rather than
edit the running implementation, we intercept at the backend boundary:
``LLMMetaPromptUpdater.update`` calls ``self.backend(system_instruction,
user_request)``, so wrapping the backend lets us swap the baked meta-prompt for
the caption-prompt updater prompt without touching any file under
``surrogate_rollout``.

The wrapper preserves any suffix the updater appends to the system prompt (the
one-shot parse-failure retry note), and it forwards ``fit_user_request`` so the
context-budget fitting reflects the caption-prompt updater prompt length.
"""

from __future__ import annotations

from typing import Any


class SystemPromptOverrideUpdaterBackend:
    """Wrap an updater backend, replacing its baked system prompt at call time."""

    def __init__(
        self,
        inner: Any,
        system_instruction: str,
        *,
        replaced_base: str,
    ) -> None:
        if not callable(inner):
            raise TypeError("inner updater backend must be callable")
        if not isinstance(system_instruction, str) or not system_instruction.strip():
            raise ValueError("system_instruction must be a non-empty string")
        if not isinstance(replaced_base, str) or not replaced_base:
            raise ValueError("replaced_base must be the baked meta-prompt text")
        self._inner = inner
        self._text = system_instruction.strip()
        self._replaced_base = replaced_base

    def __getattr__(self, name: str) -> Any:
        # Transparent decorator: forward every attribute this wrapper does not
        # define itself (provider, model_id, updater_policy_version,
        # prepare_messages, call_count, last_* ...) to the wrapped backend, so
        # callers that inspect the backend see the inner implementation.
        # Only __call__ and fit_user_request are intercepted to swap the prompt.
        return getattr(object.__getattribute__(self, "_inner"), name)

    def _swap(self, system_instruction: str) -> str:
        # Preserve any appended text (e.g. the retry rejection note) by swapping
        # only the known baked prefix; fall back to our text otherwise.
        if system_instruction.startswith(self._replaced_base):
            return self._text + system_instruction[len(self._replaced_base):]
        return self._text

    def __call__(self, system_instruction: str, user_request: str) -> str:
        return self._inner(self._swap(system_instruction), user_request)

    def fit_user_request(self, system_instruction: str, user_request: str):
        inner_fit = getattr(self._inner, "fit_user_request", None)
        if inner_fit is None:
            return user_request, None
        if not callable(inner_fit):
            raise TypeError("inner backend fit_user_request must be callable")
        return inner_fit(self._swap(system_instruction), user_request)
