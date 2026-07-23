"""Checkpoint E1 provider-independent provisional meta-prompt updater.

The updater consumes only a parent meta-prompt and validated episode feedback.
It does not load episodes, captions, trajectories, persist candidates, promote
versions, or select a model/provider.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from surrogate_rollout.optimization.schemas import (
    CompactFeedbackMemoryRecord,
    EpisodeFeedback,
    EpisodeFeedbackMemoryRecord,
    MetaPromptUpdateDecision,
    MetaPromptVersion,
    meta_prompt_update_decision_from_json,
    validate_meta_prompt_update_decision,
)
from surrogate_rollout.optimization.feedback_memory import (
    build_compact_feedback_updater_projection,
    select_historical_feedback_memories,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json
from surrogate_rollout.optimization.context_budget import ContextTruncationResult


META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION = "meta_prompt_update_request_v1"
GROUNDED_META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION = \
    "meta_prompt_update_request_v2_grounded"
COMPACT_MEMORY_META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION = \
    "meta_prompt_update_request_v3_compact_memory"
EPISODE_HISTORY_META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION = \
    "meta_prompt_update_request_v5_compact_id_free_current_feedback"
META_PROMPT_UPDATE_RESPONSE_SCHEMA_VERSION = "meta_prompt_update_response_v1"
MOCK_META_PROMPT_UPDATER_POLICY_VERSION = "mock_meta_prompt_updater_v1"

_PROMPT_DIRECTORY = Path(__file__).resolve().parent / "prompts"


def _load_prompt_text(filename: str) -> str:
    path = _PROMPT_DIRECTORY / filename
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"meta-prompt updater prompt unavailable: {path}") from exc
    if not value:
        raise RuntimeError(f"meta-prompt updater prompt is empty: {path}")
    return value


META_PROMPT_UPDATER_SYSTEM_INSTRUCTION = _load_prompt_text(
    "meta_prompt_updater_system_v5.txt")

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
        feedback_memories: Sequence[CompactFeedbackMemoryRecord] | None = None,
        historical_memories: Sequence[EpisodeFeedbackMemoryRecord] | None = None,
        current_iteration_id: str | None = None,
        historical_memory_character_budget: int | None = None,
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


_SEGMENT_IDENTIFIER_RE = re.compile(r"(?<![A-Za-z0-9_])\d+_\d+(?![A-Za-z0-9_])")
_QA_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:videomme(?:/|-)[A-Za-z0-9_/-]+|qa[-_/][A-Za-z0-9_-]+)",
    flags=re.IGNORECASE,
)


def _id_free_text(value: str | None) -> str | None:
    """Remove provenance tokens from model prose used by the updater.

    None survives as None: an abstaining feedback has no rule to clean, and
    turning that into the literal string "None" would hand the updater a
    sentence to act on.
    """
    if value is None:
        return None
    text = _SEGMENT_IDENTIFIER_RE.sub("a selected interval", str(value))
    text = _QA_IDENTIFIER_RE.sub("a QA", text)
    return " ".join(text.split())


def _transition_counts(
    feedback: EpisodeFeedback,
    grounded: Mapping[str, Any] | None,
) -> dict[str, int]:
    if grounded is not None:
        summary = grounded.get("qa_transition_summary")
        if isinstance(summary, Mapping) and all(
                isinstance(summary.get(name), list)
                for name in (
                    "correct_to_correct", "wrong_to_wrong",
                    "wrong_to_correct", "correct_to_wrong")):
            return {name: len(summary[name]) for name in (
                "correct_to_correct", "wrong_to_wrong",
                "wrong_to_correct", "correct_to_wrong")}
    by_type = {
        name: set() for name in (
            "correct_to_correct", "wrong_to_wrong",
            "wrong_to_correct", "correct_to_wrong")}
    for evidence in (*feedback.observations, *feedback.counterevidence):
        if evidence.evidence_type == "qa_transition" and \
                evidence.transition_type in by_type:
            by_type[evidence.transition_type].update(
                evidence.supporting_qa_ids)
    return {name: len(values) for name, values in by_type.items()}


def _updater_feedback_projection(
    feedback: EpisodeFeedback,
    grounded: Mapping[str, Any] | None,
) -> dict[str, Any]:
    counts = _transition_counts(feedback, grounded)
    improved = counts["wrong_to_correct"] > 0
    regressed = counts["correct_to_wrong"] > 0
    effect = (
        "mixed" if improved and regressed else
        "positive" if improved else
        "negative" if regressed else "neutral")
    # Feedback that did not establish an attribution still reports what
    # happened, but its prose must not reach the updater: an abstention was
    # reasoned about anyway last time, and its wording shaped the rule it was
    # not entitled to support.
    supported = feedback.attribution_status == "supported"
    evidence_statements = []
    if supported:
        for evidence in (*feedback.observations, *feedback.counterevidence):
            if evidence.evidence_type == "qa_transition":
                continue
            statement = _id_free_text(evidence.statement)
            if statement and statement not in evidence_statements:
                evidence_statements.append(statement)
    changed = (
        grounded.get("changed_segment_ids")
        if isinstance(grounded, Mapping) else None)
    return {
        "caption_change_status": (
            grounded.get("caption_change_status", "not_provided")
            if isinstance(grounded, Mapping) else "not_provided"),
        "changed_caption_count": (
            len(changed) if isinstance(changed, list) else None),
        "qa_transition_counts": counts,
        "episode_effect": effect,
        "caption_or_trajectory_evidence": evidence_statements,
        # Only a supported attribution may reach the updater as a rule. The
        # rest is still transmitted as an observation, so the updater sees the
        # episode happened, without a lesson it is not entitled to draw.
        "attribution_status": feedback.attribution_status,
        "recommended_strategy_change": (
            _id_free_text(feedback.recommended_strategy_change)
            if supported else None),
        "observable_trigger": (
            _id_free_text(feedback.observable_trigger)
            if supported else None),
        "caption_operation": (
            _id_free_text(feedback.caption_operation)
            if supported else None),
    }


def build_meta_prompt_update_request(
    parent: MetaPromptVersion,
    feedbacks: Sequence[EpisodeFeedback],
    *,
    updater_policy_version: str,
    feedback_grounding: Sequence[Mapping[str, Any]] | None = None,
    feedback_memories: Sequence[CompactFeedbackMemoryRecord] | None = None,
    historical_memories: Sequence[EpisodeFeedbackMemoryRecord] | None = None,
    current_iteration_id: str | None = None,
    historical_memory_character_budget: int | None = None,
) -> MetaPromptUpdateRequest:
    if not isinstance(parent, MetaPromptVersion):
        raise TypeError("parent must be a MetaPromptVersion")
    if not isinstance(updater_policy_version, str) or not updater_policy_version:
        raise ValueError("updater_policy_version must be a non-empty string")
    ordered = _normalize_feedbacks(feedbacks)
    if feedback_memories is not None and historical_memories is not None:
        raise ValueError(
            "legacy feedback_memories and historical_memories are mutually exclusive")
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
    memories = None
    if feedback_memories is not None:
        if isinstance(feedback_memories, (str, bytes)):
            raise TypeError("feedback_memories must be an ordered sequence")
        memory_records = tuple(feedback_memories)
        if not memory_records or any(
                not isinstance(item, CompactFeedbackMemoryRecord)
                for item in memory_records):
            raise TypeError(
                "feedback_memories must contain CompactFeedbackMemoryRecord")
        memories = build_compact_feedback_updater_projection(memory_records)
    historical = None
    if historical_memories is not None:
        if not isinstance(current_iteration_id, str) or not current_iteration_id:
            raise ValueError(
                "current_iteration_id is required with historical_memories")
        history_records = tuple(historical_memories)
        if any(not isinstance(item, EpisodeFeedbackMemoryRecord)
               for item in history_records):
            raise TypeError(
                "historical_memories must contain EpisodeFeedbackMemoryRecord")
        if any(item.parent_meta_prompt_id != parent.meta_prompt_id
               for item in history_records):
            raise ValueError("historical memory belongs to another parent")
        selected, selection = select_historical_feedback_memories(
            history_records,
            current_iteration_id=current_iteration_id,
            maximum_serialized_characters=historical_memory_character_budget)
        historical = {
            "memories": [
                {"memory_text": item["memory_text"]} for item in selected],
            "selection": {
                "selected_memory_count": len(selected),
                "excluded_memory_count": len(
                    selection["excluded_memory_ids"]),
                "selection_policy": selection["selection_policy"],
                "maximum_serialized_characters": (
                    selection["maximum_serialized_characters"]),
                "selected_serialized_characters": (
                    selection["selected_serialized_characters"]),
            },
        }
    current_feedback = [
        _updater_feedback_projection(
            item, grounding[index] if grounding is not None else None)
        for index, item in enumerate(ordered)
    ]
    payload = json.loads(dumps_canonical({
        "schema_version": (
            EPISODE_HISTORY_META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION
            if memories is None else
            COMPACT_MEMORY_META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION
        ),
        "updater_policy_version": updater_policy_version,
        "parent_meta_prompt": parent,
        **({
            "current_iteration_feedback": current_feedback,
            "historical_experience": historical,
        } if historical is not None else
           {"compact_feedback_memory": memories} if memories is not None else
           {"current_iteration_feedback": current_feedback}),
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
    feedback_memories: Sequence[CompactFeedbackMemoryRecord] | None = None,
    historical_memories: Sequence[EpisodeFeedbackMemoryRecord] | None = None,
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
    if feedback_memories is not None:
        forbidden_ids.update(
            identifier
            for record in feedback_memories
            for identifier in (
                record.memory_id, record.provenance.feedback_id,
                record.provenance.episode_id, record.provenance.video_id,
                *record.provenance.qa_ids, *record.provenance.segment_ids)
            if identifier)
    if historical_memories is not None:
        forbidden_ids.update(
            identifier for record in historical_memories for identifier in (
                record.memory_id, record.feedback_id, record.episode_id,
                record.candidate_id) if identifier)
    present = sorted(
        identifier for identifier in forbidden_ids
        if _contains_exact_identifier(
            decision.candidate_meta_prompt, identifier))
    if present:
        raise ValueError(
            "candidate meta-prompt contains provenance-only identifiers: "
            f"{present}")
    # Whether the candidate depends on runtime-unavailable inputs is stated in
    # the updater prompt, not matched here. A word search cannot tell "use the
    # QA answer" from "do not use the QA answer", so it rejected candidates
    # that were obeying the instruction.


def parse_meta_prompt_update_response(
    raw_response: Any,
    *,
    request: MetaPromptUpdateRequest,
    feedbacks: Sequence[EpisodeFeedback],
    feedback_memories: Sequence[CompactFeedbackMemoryRecord] | None = None,
    historical_memories: Sequence[EpisodeFeedbackMemoryRecord] | None = None,
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
        known = {item.feedback_id for item in ordered}
        if feedback_memories is not None:
            known.update(item.provenance.feedback_id
                         for item in feedback_memories)
        if historical_memories is not None:
            known.update(item.feedback_id for item in historical_memories)
        # The request is ID-free by construction: feedback, episode, QA and
        # segment identifiers are stripped before the model ever sees it. The
        # strict response schema still requires the key, so a model with no
        # identifiers to cite fills it with positional inventions. Those cite
        # nothing and are dropped; the harness attaches the real lineage. An ID
        # that does match stays, which keeps the older ID-carrying requests
        # exact.
        supported = tuple(item for item in decision.supporting_feedback_ids
                          if item in known)
        if supported != decision.supporting_feedback_ids:
            decision = replace(decision, supporting_feedback_ids=supported)
        validate_meta_prompt_candidate_content(
            decision, ordered, feedback_memories, historical_memories)
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
        feedback_memories: Sequence[CompactFeedbackMemoryRecord] | None = None,
        historical_memories: Sequence[EpisodeFeedbackMemoryRecord] | None = None,
        current_iteration_id: str | None = None,
        historical_memory_character_budget: int | None = None,
    ) -> MetaPromptUpdateResult:
        ordered = _normalize_feedbacks(feedbacks)
        request = build_meta_prompt_update_request(
            parent, ordered,
            updater_policy_version=self.updater_policy_version,
            feedback_grounding=feedback_grounding,
            feedback_memories=feedback_memories,
            historical_memories=historical_memories,
            current_iteration_id=current_iteration_id,
            historical_memory_character_budget=(
                historical_memory_character_budget))
        fit_user_request = getattr(self.backend, "fit_user_request", None)
        if fit_user_request is not None:
            if not callable(fit_user_request):
                raise TypeError("backend fit_user_request must be callable")
            transmitted, fitted = fit_user_request(
                request.system_instruction, request.user_request)
            if fitted is not None:
                if not isinstance(fitted, ContextTruncationResult):
                    raise TypeError("invalid updater context fitting result")
                if fitted.truncated:
                    payload = json.loads(transmitted)
                    request = replace(
                        request,
                        payload=payload,
                        payload_hash=fitted.transmitted_payload_hash,
                        request_id=("meta_prompt_update_request_" +
                                    fitted.transmitted_payload_hash[:20]),
                        messages=(
                            {"role": "system",
                             "content": request.system_instruction},
                            {"role": "user", "content": transmitted},
                        ),
                    )
        raw = self.backend(request.system_instruction, request.user_request)
        try:
            decision, candidate_id, status = parse_meta_prompt_update_response(
                raw, request=request, feedbacks=ordered,
                feedback_memories=feedback_memories,
                historical_memories=historical_memories)
        except MetaPromptUpdaterParseError as exc:
            # An unusable candidate is a reason to keep the parent, not to end
            # the run: a multi-iteration experiment would lose every completed
            # stage over one malformed response. One corrective retry states
            # the violation, and a second failure becomes an explicit
            # no_update carrying both raw responses.
            retry_instruction = (
                request.system_instruction +
                "\n\nThe previous response was rejected: " + str(exc) +
                "\nReturn a candidate that satisfies the stated contract, or "
                "return no_update with candidate_meta_prompt set to null.")
            retry_raw = self.backend(retry_instruction, request.user_request)
            try:
                decision, candidate_id, status = (
                    parse_meta_prompt_update_response(
                        retry_raw, request=request, feedbacks=ordered,
                        feedback_memories=feedback_memories,
                        historical_memories=historical_memories))
                raw = retry_raw
            except MetaPromptUpdaterParseError as retry_exc:
                decision = MetaPromptUpdateDecision(
                    decision="no_update",
                    candidate_meta_prompt=None,
                    change_summary=(
                        "No update: the updater response was unusable."),
                    rationale=(
                        "The response was rejected twice by the updater "
                        f"contract. First: {exc}. Retry: {retry_exc}."),
                    supporting_feedback_ids=())
                candidate_id, status = None, None
                raw = dumps_canonical({
                    "rejected_first_response": raw,
                    "rejected_retry_response": retry_raw,
                    "first_error": str(exc),
                    "retry_error": str(retry_exc),
                })
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
        feedback_memories: Sequence[CompactFeedbackMemoryRecord] | None = None,
        historical_memories: Sequence[EpisodeFeedbackMemoryRecord] | None = None,
        current_iteration_id: str | None = None,
        historical_memory_character_budget: int | None = None,
    ) -> MetaPromptUpdateResult:
        ordered = _normalize_feedbacks(feedbacks)
        request = build_meta_prompt_update_request(
            parent, ordered,
            updater_policy_version=self.updater_policy_version,
            feedback_grounding=feedback_grounding,
            feedback_memories=feedback_memories,
            historical_memories=historical_memories,
            current_iteration_id=current_iteration_id,
            historical_memory_character_budget=(
                historical_memory_character_budget))
        supporting_ids = tuple(dict.fromkeys(
            item.provenance.feedback_id for item in feedback_memories)) \
            if feedback_memories is not None else tuple(dict.fromkeys((
                *(item.feedback_id for item in ordered),
                *(item.feedback_id for item in (historical_memories or ())),
            )))
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
            validate_meta_prompt_candidate_content(
                decision, ordered, feedback_memories, historical_memories)
            identity = {
                "parent_meta_prompt_id": parent.meta_prompt_id,
                "updater_policy_version": self.updater_policy_version,
                "request_payload_hash": request.payload_hash,
                "candidate_meta_prompt": self.candidate_meta_prompt,
            }
            candidate_id = "meta_prompt_" + sha256_json(identity)[:20]
            status = "provisional"
        if feedback_memories is None and historical_memories is None:
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
