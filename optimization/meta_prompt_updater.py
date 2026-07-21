"""Checkpoint E1 provider-independent provisional meta-prompt updater.

The updater consumes only a parent meta-prompt and validated episode feedback.
It does not load episodes, captions, trajectories, persist candidates, promote
versions, or select a model/provider.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from surrogate_rollout.optimization.schemas import (
    EpisodeFeedback,
    MetaPromptUpdateDecision,
    MetaPromptVersion,
    meta_prompt_update_decision_from_json,
    validate_meta_prompt_update_decision,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json


META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION = "meta_prompt_update_request_v1"
GROUNDED_META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION = \
    "meta_prompt_update_request_v2_grounded"
META_PROMPT_UPDATE_RESPONSE_SCHEMA_VERSION = "meta_prompt_update_response_v1"
MOCK_META_PROMPT_UPDATER_POLICY_VERSION = "mock_meta_prompt_updater_v1"

META_PROMPT_UPDATER_SYSTEM_INSTRUCTION = """You update the procedure used by a visual- and history-conditioned prompt generator.

Input contains one current parent meta-prompt and an ordered list of validated episode-feedback records. Use only those records. Raw episodes, captions, trajectories, QA answers, and prompt deltas are intentionally absent.

When feedback_grounding is present, it is deterministic metadata derived from
the corresponding stored episode. If caption_change_status is unchanged,
treat any QA flip as attribution-uncertain no-op evidence, not as direct
caption-strategy benefit or harm. Do not override these grounding facts from
the feedback prose.

Your goal is not to vote over QA transition counts. Determine why caption changes helped, harmed, or had no demonstrated utility by comparing the diagnoses, observations, counterevidence, and recommended strategies across feedback records.

Reason in this order:
1. identify beneficial caption behavior supported by the feedback;
2. identify harmful, neutral, or limiting behavior;
3. determine whether the difference can be expressed as a condition recognizable from current frames or bounded preceding history;
4. propose the smallest conditional change that preserves the beneficial behavior while avoiding the observed risk;
5. return no_update when no coherent and runtime-observable condition is supported.

QA transition counts are descriptive evidence, not a majority vote. Do not require every episode to improve. Harmful evidence may define when a strategy should not be applied rather than automatically ruling out an update. Mixed outcomes may support a conditional update when the feedback explains their difference. Return no_update when the records do not support a coherent mechanism, the required condition cannot be recognized from runtime inputs, or the harmful evidence cannot be addressed by a clear restriction.

Treat all conclusions as provisional. Do not turn one episode into a universal rule. Clearly explain which feedback records support the proposed behavior and which records define its boundary or risk.

The runtime prompt generator can use only current visual frames, bounded preceding caption history, and the current meta-prompt. Do not require QA information, correctness labels, trajectories, OCR metadata, external tools, or other unavailable inputs.

Do not place QA answers, dataset-specific wording, clip IDs, segment IDs, feedback IDs, or episode IDs in the candidate meta-prompt. Express any update in new, general wording rather than reproducing an individual intervention instruction. Make only the smallest necessary change to the parent meta-prompt; do not rewrite it wholesale.

A no_update decision is normal when evidence is insufficient or cannot support a safe conditional change.

Return exactly one strict JSON object with:
- decision
- candidate_meta_prompt
- change_summary
- rationale
- supporting_feedback_ids

For update, candidate_meta_prompt must be a non-empty string.
For no_update, candidate_meta_prompt must be null.
supporting_feedback_ids may contain only exact IDs from the input.

Do not return a candidate ID, request hash, status, Markdown fence, prefix, suffix, or explanatory prose outside the JSON object."""

_RESPONSE_FIELDS = {
    "decision", "candidate_meta_prompt", "change_summary", "rationale",
    "supporting_feedback_ids",
}


class MetaPromptUpdaterError(ValueError):
    pass


class MetaPromptUpdaterParseError(MetaPromptUpdaterError):
    def __init__(self, reason: str, *, raw_response: Any) -> None:
        self.raw_response = raw_response
        super().__init__(reason)


class MetaPromptUpdaterBackend(Protocol):
    def __call__(self, system_instruction: str, user_request: str) -> str:
        ...


class MetaPromptUpdater(Protocol):
    def update(
        self,
        parent: MetaPromptVersion,
        feedbacks: Sequence[EpisodeFeedback],
        *,
        feedback_grounding: Sequence[Mapping[str, Any]] | None = None,
    ) -> "MetaPromptUpdateResult":
        ...


@dataclass(frozen=True)
class MetaPromptUpdateRequest:
    system_instruction: str
    payload: Mapping[str, Any]
    payload_hash: str
    request_id: str
    messages: tuple[Mapping[str, str], ...]

    @property
    def user_request(self) -> str:
        return self.messages[1]["content"]


@dataclass(frozen=True)
class MetaPromptUpdateResult:
    request: MetaPromptUpdateRequest
    decision: MetaPromptUpdateDecision
    candidate_meta_prompt_id: str | None
    candidate_status: str | None
    raw_response: str | None
    updater_policy_version: str
    backend_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.decision.decision == "update":
            if not self.candidate_meta_prompt_id:
                raise ValueError("update result requires a candidate ID")
            if self.candidate_status != "provisional":
                raise ValueError("update result status must be provisional")
        elif self.candidate_meta_prompt_id is not None or \
                self.candidate_status is not None:
            raise ValueError("no_update result cannot contain a candidate")


def meta_prompt_update_response_json_schema() -> dict[str, Any]:
    fields = {
        "decision": {"type": "string", "enum": ["update", "no_update"]},
        "candidate_meta_prompt": {
            "anyOf": [{"type": "string"}, {"type": "null"}]},
        "change_summary": {"type": "string"},
        "rationale": {"type": "string"},
        "supporting_feedback_ids": {
            "type": "array", "items": {"type": "string"}},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(fields),
        "properties": fields,
    }


def _normalize_feedbacks(
    feedbacks: Sequence[EpisodeFeedback],
) -> tuple[EpisodeFeedback, ...]:
    if isinstance(feedbacks, (str, bytes)):
        raise TypeError("feedbacks must be an ordered sequence")
    values = tuple(feedbacks)
    if not values or any(not isinstance(item, EpisodeFeedback) for item in values):
        raise TypeError("feedbacks must contain one or more EpisodeFeedback")
    ids = tuple(item.feedback_id for item in values)
    if len(ids) != len(set(ids)):
        raise ValueError("feedback IDs must be unique within an updater request")
    return values


def build_meta_prompt_update_request(
    parent: MetaPromptVersion,
    feedbacks: Sequence[EpisodeFeedback],
    *,
    updater_policy_version: str,
    feedback_grounding: Sequence[Mapping[str, Any]] | None = None,
) -> MetaPromptUpdateRequest:
    if not isinstance(parent, MetaPromptVersion):
        raise TypeError("parent must be a MetaPromptVersion")
    if not isinstance(updater_policy_version, str) or not updater_policy_version:
        raise ValueError("updater_policy_version must be a non-empty string")
    ordered = _normalize_feedbacks(feedbacks)
    grounding = None
    if feedback_grounding is not None:
        if isinstance(feedback_grounding, (str, bytes)):
            raise TypeError("feedback_grounding must be an ordered sequence")
        grounding = tuple(feedback_grounding)
        if len(grounding) != len(ordered) or any(
                not isinstance(item, Mapping) for item in grounding):
            raise ValueError(
                "feedback_grounding must contain one mapping per feedback")
        expected_ids = tuple(item.feedback_id for item in ordered)
        observed_ids = tuple(item.get("feedback_id") for item in grounding)
        if observed_ids != expected_ids:
            raise ValueError(
                "feedback_grounding order or feedback IDs do not match")
        grounding = json.loads(dumps_canonical(grounding))
    payload = json.loads(dumps_canonical({
        "schema_version": (
            GROUNDED_META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION
            if grounding is not None else
            META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION),
        "updater_policy_version": updater_policy_version,
        "parent_meta_prompt": parent,
        "feedbacks": ordered,
        **({"feedback_grounding": grounding}
           if grounding is not None else {}),
    }))
    payload_hash = sha256_json(payload)
    user_request = dumps_canonical(payload)
    return MetaPromptUpdateRequest(
        system_instruction=META_PROMPT_UPDATER_SYSTEM_INSTRUCTION,
        payload=payload,
        payload_hash=payload_hash,
        request_id="meta_prompt_update_request_" + payload_hash[:20],
        messages=(
            {"role": "system", "content": META_PROMPT_UPDATER_SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_request},
        ),
    )


_RUNTIME_UNAVAILABLE_PATTERNS = (
    r"\bQA(?:s)?\b",
    r"\bcorrectness(?: labels?)?\b",
    r"\btrajector(?:y|ies)\b",
    r"\bOCR(?: metadata)?\b",
    r"\bexternal tools?\b",
    r"\bground[- ]truth answers?\b",
    r"\banswer choices?\b",
    r"\bdataset[- ]specific answers?\b",
)


def _contains_exact_identifier(text: str, identifier: str) -> bool:
    """Match a known provenance ID as a complete identifier token.

    Delimiters are defined explicitly instead of using substring matching, so
    a short ID such as ``qa-1`` does not reject unrelated ``qa-10`` text.
    """
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])",
        text,
    ) is not None


def validate_meta_prompt_candidate_content(
    decision: MetaPromptUpdateDecision,
    feedbacks: tuple[EpisodeFeedback, ...],
) -> None:
    if decision.candidate_meta_prompt is None:
        return
    forbidden_ids = {
        identifier
        for feedback in feedbacks
        for identifier in (
            feedback.feedback_id,
            feedback.episode_id,
            *(item for evidence in (
                *feedback.observations, *feedback.counterevidence)
              for item in (
                  *evidence.supporting_segment_ids,
                  *evidence.supporting_qa_ids)),
        )
        if identifier
    }
    present = sorted(
        identifier for identifier in forbidden_ids
        if _contains_exact_identifier(
            decision.candidate_meta_prompt, identifier))
    if present:
        raise ValueError(
            "candidate meta-prompt contains provenance-only identifiers: "
            f"{present}")
    unavailable = sorted({
        match.group(0)
        for pattern in _RUNTIME_UNAVAILABLE_PATTERNS
        for match in re.finditer(
            pattern, decision.candidate_meta_prompt, flags=re.IGNORECASE)
    })
    if unavailable:
        raise ValueError(
            "candidate meta-prompt requires runtime-unavailable or "
            f"dataset-specific inputs: {unavailable}")


def parse_meta_prompt_update_response(
    raw_response: Any,
    *,
    request: MetaPromptUpdateRequest,
    feedbacks: Sequence[EpisodeFeedback],
) -> tuple[MetaPromptUpdateDecision, str | None, str | None]:
    ordered = _normalize_feedbacks(feedbacks)
    try:
        if not isinstance(raw_response, str):
            raise TypeError("response must be a string")
        value = json.loads(raw_response)
        if not isinstance(value, dict) or set(value) != _RESPONSE_FIELDS:
            raise ValueError(
                f"response must contain exactly {sorted(_RESPONSE_FIELDS)}")
        decision = meta_prompt_update_decision_from_json(value)
        validate_meta_prompt_update_decision(decision, ordered)
        validate_meta_prompt_candidate_content(decision, ordered)
        restored = meta_prompt_update_decision_from_json(
            json.loads(dumps_canonical(decision)))
        if restored != decision:
            raise ValueError("decision failed canonical round-trip")
        if decision.decision == "no_update":
            return decision, None, None
        identity = {
            "parent_meta_prompt_id": request.payload["parent_meta_prompt"][
                "meta_prompt_id"],
            "updater_policy_version": request.payload[
                "updater_policy_version"],
            "request_payload_hash": request.payload_hash,
            "candidate_meta_prompt": decision.candidate_meta_prompt,
        }
        return (
            decision,
            "meta_prompt_" + sha256_json(identity)[:20],
            "provisional",
        )
    except MetaPromptUpdaterParseError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MetaPromptUpdaterParseError(
            f"invalid meta-prompt updater response: {exc}",
            raw_response=raw_response) from exc


def _backend_metadata(backend: Any) -> Mapping[str, Any]:
    metadata = getattr(backend, "metadata", None)
    if callable(metadata):
        value = metadata()
        if not isinstance(value, Mapping):
            raise TypeError("updater backend metadata must be a mapping")
        return json.loads(dumps_canonical(value))
    return {}


class LLMMetaPromptUpdater:
    """One-call updater with an injected provider-independent backend."""

    def __init__(
        self, *, backend: MetaPromptUpdaterBackend,
        updater_policy_version: str,
    ) -> None:
        if not callable(backend):
            raise TypeError("backend must be callable")
        if not isinstance(updater_policy_version, str) or not \
                updater_policy_version:
            raise ValueError("updater_policy_version must be a non-empty string")
        self.backend = backend
        self.updater_policy_version = updater_policy_version

    def update(
        self,
        parent: MetaPromptVersion,
        feedbacks: Sequence[EpisodeFeedback],
        *,
        feedback_grounding: Sequence[Mapping[str, Any]] | None = None,
    ) -> MetaPromptUpdateResult:
        ordered = _normalize_feedbacks(feedbacks)
        request = build_meta_prompt_update_request(
            parent, ordered,
            updater_policy_version=self.updater_policy_version,
            feedback_grounding=feedback_grounding)
        raw = self.backend(request.system_instruction, request.user_request)
        decision, candidate_id, status = parse_meta_prompt_update_response(
            raw, request=request, feedbacks=ordered)
        return MetaPromptUpdateResult(
            request=request,
            decision=decision,
            candidate_meta_prompt_id=candidate_id,
            candidate_status=status,
            raw_response=raw,
            updater_policy_version=self.updater_policy_version,
            backend_metadata=_backend_metadata(self.backend),
        )


class DeterministicMockMetaPromptUpdater:
    """Deterministic structural mock; caller supplies any update text."""

    def __init__(self, *, candidate_meta_prompt: str | None = None) -> None:
        if candidate_meta_prompt is not None and (
                not isinstance(candidate_meta_prompt, str) or
                not candidate_meta_prompt):
            raise ValueError("candidate_meta_prompt must be non-empty or None")
        self.candidate_meta_prompt = candidate_meta_prompt
        self.updater_policy_version = MOCK_META_PROMPT_UPDATER_POLICY_VERSION

    def update(
        self,
        parent: MetaPromptVersion,
        feedbacks: Sequence[EpisodeFeedback],
        *,
        feedback_grounding: Sequence[Mapping[str, Any]] | None = None,
    ) -> MetaPromptUpdateResult:
        ordered = _normalize_feedbacks(feedbacks)
        request = build_meta_prompt_update_request(
            parent, ordered,
            updater_policy_version=self.updater_policy_version,
            feedback_grounding=feedback_grounding)
        supporting_ids = tuple(item.feedback_id for item in ordered)
        if self.candidate_meta_prompt is None:
            decision = MetaPromptUpdateDecision(
                decision="no_update",
                candidate_meta_prompt=None,
                change_summary="No provisional meta-prompt change was produced.",
                rationale=(
                    "Deterministic mock no-update decision; no semantic "
                    "evidence synthesis was performed."),
                supporting_feedback_ids=supporting_ids,
            )
            candidate_id = status = None
        else:
            decision = MetaPromptUpdateDecision(
                decision="update",
                candidate_meta_prompt=self.candidate_meta_prompt,
                change_summary=(
                    "A caller-supplied deterministic provisional candidate "
                    "was produced."),
                rationale=(
                    "Deterministic mock update decision; no semantic evidence "
                    "synthesis was performed."),
                supporting_feedback_ids=supporting_ids,
            )
            validate_meta_prompt_candidate_content(decision, ordered)
            identity = {
                "parent_meta_prompt_id": parent.meta_prompt_id,
                "updater_policy_version": self.updater_policy_version,
                "request_payload_hash": request.payload_hash,
                "candidate_meta_prompt": self.candidate_meta_prompt,
            }
            candidate_id = "meta_prompt_" + sha256_json(identity)[:20]
            status = "provisional"
        validate_meta_prompt_update_decision(decision, ordered)
        return MetaPromptUpdateResult(
            request=request,
            decision=decision,
            candidate_meta_prompt_id=candidate_id,
            candidate_status=status,
            raw_response=None,
            updater_policy_version=self.updater_policy_version,
            backend_metadata={"provider": "deterministic_mock"},
        )
