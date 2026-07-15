"""Isolated, disabled-by-default real feedback policy boundary."""

from __future__ import annotations

import json
import os
from typing import Callable, Mapping, Sequence

from surrogate_rollout.optimization.feedback_generator import (
    FeedbackParseError,
    feedback_batch_from_json,
)
from surrogate_rollout.optimization.schemas import (
    CounterfactualEvidence,
    FeedbackBatch,
    MetaKnowledgeItem,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


class RealFeedbackDisabledError(RuntimeError):
    """The isolated real feedback policy was invoked while disabled."""


class LLMFeedbackGenerator:
    """Strict parser plus injectable provider; disabled unless explicit."""

    def __init__(
        self,
        *,
        artifact_dir: str,
        enabled: bool = False,
        response_provider: Callable[[str], str] | None = None,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.enabled = enabled
        self.response_provider = response_provider

    def parse_saved_response(self, raw_response: str) -> FeedbackBatch:
        os.makedirs(self.artifact_dir, exist_ok=True)
        raw_path = os.path.abspath(os.path.join(
            self.artifact_dir, "raw_response.txt"))
        _atomic_write_text(raw_path, raw_response)
        try:
            parsed = json.loads(raw_response)
            if not isinstance(parsed, Mapping):
                raise TypeError("feedback response must be a JSON object")
            return feedback_batch_from_json(
                parsed, raw_response_artifact=raw_path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FeedbackParseError(
                f"invalid feedback response: {exc}",
                raw_response_artifact=raw_path,
            ) from exc

    def generate(
        self,
        evidence: Sequence[CounterfactualEvidence],
        meta_knowledge: Sequence[MetaKnowledgeItem],
    ) -> FeedbackBatch:
        if not self.enabled:
            raise RealFeedbackDisabledError("real feedback generation is disabled")
        if self.response_provider is None:
            raise RealFeedbackDisabledError(
                "real feedback response provider is not configured")
        request = dumps_canonical({
            "evidence": tuple(evidence),
            "meta_knowledge": tuple(meta_knowledge),
        })
        os.makedirs(self.artifact_dir, exist_ok=True)
        _atomic_write_text(os.path.join(self.artifact_dir, "request.json"), request)
        return self.parse_saved_response(self.response_provider(request))
