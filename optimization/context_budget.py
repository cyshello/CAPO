"""Deterministic JSON payload fitting while preserving system instructions."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json


TRUNCATION_MARKER = "[TRUNCATED_TO_CONTEXT]"
CONTEXT_TRUNCATION_POLICY_VERSION = "dynamic_json_strings_v1"
# Provider-side chat and structured-output accounting is not exposed exactly
# by local tokenizers. Never fit a request all the way to the advertised
# context boundary; the July 2026 gpt-4o run under-counted it by 428 tokens.
PROVIDER_CONTEXT_SAFETY_MARGIN_TOKENS = 2048


@dataclass(frozen=True)
class ContextTruncationResult:
    payload: Mapping[str, Any]
    serialized_payload: str
    original_payload_hash: str
    transmitted_payload_hash: str
    original_input_tokens: int
    transmitted_input_tokens: int
    maximum_input_tokens: int
    truncated_paths: tuple[str, ...]
    retention_ratio: float

    @property
    def truncated(self) -> bool:
        return bool(self.truncated_paths)

    def audit_metadata(self) -> dict[str, Any]:
        """Return bounded provenance without duplicating the fitted payload."""
        return {
            "policy_version": CONTEXT_TRUNCATION_POLICY_VERSION,
            "original_payload_hash": self.original_payload_hash,
            "transmitted_payload_hash": self.transmitted_payload_hash,
            "original_input_tokens": self.original_input_tokens,
            "transmitted_input_tokens": self.transmitted_input_tokens,
            "maximum_input_tokens": self.maximum_input_tokens,
            "truncated_paths": list(self.truncated_paths),
            "retention_ratio": self.retention_ratio,
        }


_PROTECTED_EXACT_KEYS = {
    "id", "ids", "status", "type", "availability", "schema_version",
    "evidence_type", "transition_type", "role", "tool", "name",
}
_PROTECTED_SUFFIXES = (
    "_id", "_ids", "_ref", "_refs", "_hash", "_sha256", "_path",
    "_version", "_index", "_count", "_timestamp",
)


def _protected(key: str) -> bool:
    normalized = key.lower()
    return normalized in _PROTECTED_EXACT_KEYS or normalized.endswith(
        _PROTECTED_SUFFIXES)


def _string_paths(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key, item in value.items():
            current = path + (str(key),)
            if isinstance(item, str) and not _protected(str(key)):
                yield current, item
            else:
                yield from _string_paths(item, current)
    elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            yield from _string_paths(item, path + (str(index),))


def _replace_path(value: Any, path: tuple[str, ...], replacement: str) -> None:
    cursor = value
    for part in path[:-1]:
        cursor = cursor[int(part)] if isinstance(cursor, list) else cursor[part]
    last = path[-1]
    if isinstance(cursor, list):
        cursor[int(last)] = replacement
    else:
        cursor[last] = replacement


def fit_json_payload_to_token_budget(
    payload: Mapping[str, Any], *,
    measure_input_tokens: Callable[[str], int],
    maximum_input_tokens: int,
) -> ContextTruncationResult:
    """Fit only dynamic JSON string leaves; templates live outside payload.

    Identifiers, references, hashes, paths, schema/version/type fields and all
    non-string structure are preserved. Long semantic strings are prefix-
    truncated at one common deterministic retention ratio.
    """
    if maximum_input_tokens <= 0 or not callable(measure_input_tokens):
        raise ValueError("positive token budget and token counter are required")
    original = copy.deepcopy(dict(payload))
    original_text = dumps_canonical(original)
    original_tokens = int(measure_input_tokens(original_text))
    if original_tokens <= maximum_input_tokens:
        return ContextTruncationResult(
            original, original_text, sha256_json(original), sha256_json(original),
            original_tokens, original_tokens, maximum_input_tokens, (), 1.0)
    candidates = tuple(_string_paths(original))
    if not candidates:
        raise ValueError(
            "dynamic payload exceeds context but has no truncatable strings")

    marker_cost = len(TRUNCATION_MARKER) + 1

    def render(maximum_characters: int):
        result = copy.deepcopy(original)
        changed = []
        for path, text in candidates:
            keep = min(len(text), maximum_characters)
            if keep + marker_cost < len(text):
                replacement = ((text[:keep].rstrip() + " ") if keep else "") + \
                    TRUNCATION_MARKER
                _replace_path(result, path, replacement)
                changed.append(".".join(path))
        serialized = dumps_canonical(result)
        return result, serialized, int(measure_input_tokens(serialized)), tuple(changed)

    minimal = render(0)
    if minimal[2] > maximum_input_tokens:
        raise ValueError(
            "payload structure and protected provenance exceed context even "
            "after all semantic strings were truncated")
    low, high = 0, max(len(text) for _path, text in candidates)
    best = minimal
    while low <= high:
        middle = (low + high) // 2
        candidate = render(middle)
        if candidate[2] <= maximum_input_tokens:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    fitted, fitted_text, fitted_tokens, paths = best
    original_candidate_characters = sum(len(text) for _path, text in candidates)
    retained_characters = sum(
        min(len(text), max(0, high)) for _path, text in candidates)
    retention_ratio = (
        retained_characters / original_candidate_characters
        if original_candidate_characters else 1.0)
    return ContextTruncationResult(
        fitted, fitted_text, sha256_json(original), sha256_json(fitted),
        original_tokens, fitted_tokens, maximum_input_tokens, paths,
        retention_ratio)
