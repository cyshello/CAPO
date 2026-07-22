"""Checkpoint D1 request construction and strict episode-feedback boundary.

The module is deliberately runtime-only and read-only.  It resolves complete
saved QA/trajectory evidence, constructs one deterministic text request, and
parses one injected backend response.  It does not select evidence, truncate
payloads, call a configured model by itself, retry, or persist anything.
"""

from __future__ import annotations

import json
import hashlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from surrogate_rollout.optimization.episode_feedback import (
    classify_qa_transition,
)
from surrogate_rollout.optimization.schemas import (
    EpisodeFeedback,
    InterventionEpisode,
    episode_feedback_from_json,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json, sha256_text


LEAN_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION = \
    "episode_feedback_request_v6_candidate_mixed_view_sibling_outcomes"
EPISODE_FEEDBACK_RESPONSE_SCHEMA_VERSION = \
    "episode_feedback_response_v5_mixed_view_memory"

QA_TRANSITION_SUMMARY_ORDER = (
    "correct_to_correct", "wrong_to_wrong", "wrong_to_correct",
    "correct_to_wrong",
)


_PROMPT_DIRECTORY = Path(__file__).resolve().parent / "prompts"


def _load_prompt_text(filename: str) -> str:
    path = _PROMPT_DIRECTORY / filename
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"episode-feedback prompt unavailable: {path}") from exc
    if not value:
        raise RuntimeError(f"episode-feedback prompt is empty: {path}")
    return value


EPISODE_FEEDBACK_SYSTEM_INSTRUCTION = _load_prompt_text(
    "episode_feedback_system_v7.txt")


class EpisodeFeedbackRequestError(ValueError):
    """Saved episode evidence cannot be resolved without fabrication."""


class EpisodeFeedbackBackendConfigurationError(ValueError):
    """The injected backend omits required explicit configuration metadata."""


class EpisodeFeedbackParseError(ValueError):
    """Strict response parsing failed while preserving the raw response."""

    def __init__(self, reason: str, *, raw_response: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw_response = raw_response


class EpisodeFeedbackContextOverflowError(ValueError):
    """A complete request exceeds a known backend context limit."""

    def __init__(
        self, *, observed_tokens: int, configured_limit: int,
        clip_count: int, qa_count: int,
        largest_payload_sections: tuple[tuple[str, int], ...],
    ) -> None:
        self.observed_tokens = observed_tokens
        self.configured_limit = configured_limit
        self.clip_count = clip_count
        self.qa_count = qa_count
        self.largest_payload_sections = largest_payload_sections
        super().__init__(
            "episode feedback request exceeds configured context limit: "
            f"observed_tokens={observed_tokens}, limit={configured_limit}, "
            f"clip_count={clip_count}, qa_count={qa_count}, "
            f"largest_payload_sections={largest_payload_sections!r}; "
            "no truncation, sampling, summarization, or splitting was performed")


class EpisodeFeedbackArtifactResolver(Protocol):
    def resolve_qas(
        self, episode: InterventionEpisode,
    ) -> tuple[Mapping[str, Any], ...]:
        ...


class EpisodeFeedbackResponseProvider(Protocol):
    def __call__(self, system_instruction: str, request: str) -> str:
        ...

    def metadata(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)

class LeanEpisodeFeedbackRequestSize:
    changed_caption_count: int
    qa_count: int
    tool_call_count: int
    model_payload_character_count: int
    token_count: int | None


@dataclass(frozen=True)
class LeanEpisodeFeedbackRequest:
    system_instruction: str
    model_payload: Mapping[str, Any]
    model_payload_hash: str
    messages: tuple[Mapping[str, str], ...]
    size_statistics: LeanEpisodeFeedbackRequestSize

    @property
    def user_payload(self) -> Mapping[str, Any]:
        return self.model_payload

    @property
    def payload_hash(self) -> str:
        return self.model_payload_hash

    @property
    def user_request(self) -> str:
        return self.messages[1]["content"]




@dataclass(frozen=True)
class EpisodeFeedbackInvocationResult:
    request: LeanEpisodeFeedbackRequest
    raw_response: str
    feedback: EpisodeFeedback
    policy_version: str
    backend_metadata: Mapping[str, Any]

def _read_json_object(path: str, where: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeFeedbackRequestError(
            f"cannot read {where} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EpisodeFeedbackRequestError(f"{where} must be a JSON object")
    return value


def _read_jsonl_objects(path: str, where: str) -> tuple[dict[str, Any], ...]:
    rows = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise EpisodeFeedbackRequestError(
                        f"{where} row {line_number} must be a JSON object")
                rows.append(value)
    except EpisodeFeedbackRequestError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeFeedbackRequestError(
            f"cannot read {where} at {path}: {exc}") from exc
    return tuple(rows)


def _artifact_path(
    owner_path: str, record: Mapping[str, Any], key: str, where: str,
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise EpisodeFeedbackRequestError(
            f"{where}.{key} must identify an existing artifact")
    path = value if os.path.isabs(value) else os.path.join(
        os.path.dirname(owner_path), value)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise EpisodeFeedbackRequestError(
            f"{where}.{key} does not resolve to an existing artifact: {path}")
    return path


def _index_rows(
    rows: Sequence[Mapping[str, Any]], key: str, where: str,
) -> dict[str, Mapping[str, Any]]:
    output = {}
    for index, row in enumerate(rows):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise EpisodeFeedbackRequestError(
                f"{where}[{index}].{key} must be a non-empty string")
        if value in output:
            raise EpisodeFeedbackRequestError(
                f"duplicate {key} {value!r} in {where}")
        output[value] = row
    return output


def _reference_sets(
    row: Mapping[str, Any], trajectory_ref: str | None,
) -> tuple[dict[str, tuple[str, ...]], tuple[dict[str, Any], ...]]:
    value = row.get("reference_sets")
    sibling_value = None
    if trajectory_ref is not None:
        sibling = os.path.join(os.path.dirname(trajectory_ref), "references.json")
        if os.path.isfile(sibling):
            sibling_value = _read_json_object(sibling, "trajectory references")
    if value is not None and not isinstance(value, Mapping):
        raise EpisodeFeedbackRequestError("QA reference_sets must be an object")
    if sibling_value is not None:
        merged = dict(sibling_value)
        for key, items in (value or {}).items():
            if key in merged and json.loads(dumps_canonical(merged[key])) != \
                    json.loads(dumps_canonical(items)):
                raise EpisodeFeedbackRequestError(
                    f"QA reference_sets.{key} conflicts with references.json")
            merged[key] = items
        value = merged
    if value is None:
        return {}, ()
    output = {}
    evidence: tuple[dict[str, Any], ...] = ()
    for key, items in value.items():
        if key == "evidence":
            if not isinstance(items, (list, tuple)) or any(
                    not isinstance(item, Mapping) for item in items):
                raise EpisodeFeedbackRequestError(
                    "QA reference_sets.evidence must be an object array")
            evidence = tuple(json.loads(dumps_canonical(item)) for item in items)
            continue
        if not isinstance(items, (list, tuple)) or any(
                not isinstance(item, str) or not item for item in items):
            raise EpisodeFeedbackRequestError(
                f"QA reference_sets.{key} must be a string array")
        output[str(key)] = tuple(items)
    return output, evidence


def _same_reference(stored: Any, episode_reference: str | None) -> bool:
    if stored is None or episode_reference is None:
        return stored is None and episode_reference is None
    return isinstance(stored, str) and bool(stored) and \
        os.path.abspath(stored) == os.path.abspath(episode_reference)


def _resolve_trajectory(
    reference: str | None, reference_sets: Mapping[str, tuple[str, ...]],
    reference_evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if reference is None:
        if any(reference_sets.values()) or reference_evidence:
            raise EpisodeFeedbackRequestError(
                "unavailable trajectory conflicts with reference provenance")
        return {
            "availability": "unavailable",
            "content": None,
            "tool_events": [],
            "reference_sets": {},
            "reference_evidence": [],
            "referenced_segment_ids": [],
            "retrieved_segment_ids": [],
        }
    if not isinstance(reference, str) or not reference or not os.path.isfile(reference):
        raise EpisodeFeedbackRequestError(
            f"trajectory reference does not resolve: {reference!r}")
    content = _read_jsonl_objects(reference, "trajectory")
    tool_events_path = os.path.join(os.path.dirname(reference), "tool_events.jsonl")
    tool_events = (_read_jsonl_objects(tool_events_path, "trajectory tool events")
                   if os.path.isfile(tool_events_path) else ())
    return {
        "availability": "available",
        "content": list(content),
        "tool_events": list(tool_events),
        "reference_sets": {key: list(items)
                           for key, items in reference_sets.items()},
        "reference_evidence": [
            json.loads(dumps_canonical(item)) for item in reference_evidence],
        "referenced_segment_ids": list(
            reference_sets.get("explicitly_cited_segments", ())),
        "retrieved_segment_ids": list(
            reference_sets.get("retrieved_segments", ())),
    }


class FreshEpisodeFeedbackArtifactResolver:
    """Resolve QA evidence from a fresh prompt-delta intervention manifest."""

    def resolve_qas(
        self, episode: InterventionEpisode,
    ) -> tuple[Mapping[str, Any], ...]:
        baseline_path = os.path.abspath(episode.baseline_run_ref)
        intervention_path = os.path.abspath(episode.intervention_run_ref)
        baseline = _read_json_object(baseline_path, "baseline video manifest")
        intervention = _read_json_object(
            intervention_path, "fresh intervention manifest")
        if baseline.get("video_id") != episode.video_id or \
                intervention.get("video_id") != episode.video_id or \
                intervention.get("schema_version") != \
                "fresh_prompt_delta_intervention_v1":
            raise EpisodeFeedbackRequestError(
                "fresh episode video/schema lineage conflicts with run references")
        baseline_path_qas = _artifact_path(
            baseline_path, baseline, "baseline_qas_path", "baseline manifest")
        before_rows = _index_rows(
            _read_jsonl_objects(baseline_path_qas, "baseline QAs"),
            "question_id", "baseline QAs")
        after_rows = _index_rows(
            intervention.get("qa_records") or (), "qa_id",
            "fresh intervention qa_records")
        output = []
        for outcome in episode.qa_outcomes:
            before = before_rows.get(outcome.qa_id)
            after = after_rows.get(outcome.qa_id)
            if before is None or after is None:
                raise EpisodeFeedbackRequestError(
                    f"fresh saved QA metadata is missing {outcome.qa_id!r}")
            if (before.get("prediction"), before.get("is_correct"),
                    after.get("intervention_answer"),
                    after.get("intervention_correct")) != (
                    outcome.baseline_answer, outcome.baseline_correct,
                    outcome.intervention_answer, outcome.intervention_correct):
                raise EpisodeFeedbackRequestError(
                    f"fresh QA outcome conflicts for {outcome.qa_id!r}")
            if not _same_reference(before.get("trajectory_path"),
                                   outcome.baseline_trajectory_ref) or not \
                    _same_reference(after.get("intervention_trajectory_path"),
                                    outcome.intervention_trajectory_ref):
                raise EpisodeFeedbackRequestError(
                    f"fresh trajectory reference conflicts for {outcome.qa_id!r}")
            question = before.get("question")
            choices = before.get("options") or ()
            if not isinstance(question, str) or not question or not isinstance(
                    choices, (list, tuple)) or any(
                        not isinstance(item, str) for item in choices):
                raise EpisodeFeedbackRequestError(
                    f"fresh baseline QA metadata is invalid: {outcome.qa_id!r}")
            before_refs, before_reference_evidence = _reference_sets(
                before, outcome.baseline_trajectory_ref)
            after_refs, after_reference_evidence = _reference_sets(
                {}, outcome.intervention_trajectory_ref)
            output.append({
                "qa_id": outcome.qa_id,
                "is_source_qa": outcome.is_source_qa,
                "question": question, "answer_choices": list(choices),
                "gold_answer": before.get("ground_truth"),
                "baseline_answer": outcome.baseline_answer,
                "intervention_answer": outcome.intervention_answer,
                "baseline_correct": outcome.baseline_correct,
                "intervention_correct": outcome.intervention_correct,
                "transition": classify_qa_transition(outcome),
                "baseline_trajectory": _resolve_trajectory(
                    outcome.baseline_trajectory_ref, before_refs,
                    before_reference_evidence),
                "intervention_trajectory": _resolve_trajectory(
                    outcome.intervention_trajectory_ref, after_refs,
                    after_reference_evidence),
            })
        return tuple(output)


class SavedEpisodeFeedbackArtifactResolver:
    """Resolve saved prompt-delta feedback artifacts."""

    def __init__(self) -> None:
        self.fresh = FreshEpisodeFeedbackArtifactResolver()

    def resolve_qas(self, episode: InterventionEpisode):
        value = _read_json_object(
            os.path.abspath(episode.intervention_run_ref),
            "saved intervention manifest")
        if value.get("schema_version") != "fresh_prompt_delta_intervention_v1":
            raise EpisodeFeedbackRequestError(
                "saved feedback resolver only accepts fresh prompt-delta artifacts")
        return self.fresh.resolve_qas(episode)



def build_qa_transition_summary(
    episode: InterventionEpisode,
) -> dict[str, list[str]]:
    """Return canonical stored correctness facts without trajectory inference."""
    summary = {name: [] for name in QA_TRANSITION_SUMMARY_ORDER}
    for outcome in episode.qa_outcomes:
        summary[classify_qa_transition(outcome)].append(outcome.qa_id)
    return summary




def _plain_json(value: Any) -> Any:
    return json.loads(dumps_canonical(value))


def _parse_call_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        raise EpisodeFeedbackRequestError(
            "stored tool-call arguments must be a JSON string")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise EpisodeFeedbackRequestError(
            "stored tool-call arguments are not strict JSON") from exc


def _hit_segment_id(hit: Mapping[str, Any]) -> str | None:
    start = hit.get("time_start_secs")
    end = hit.get("time_end_secs")
    if not isinstance(start, (int, float)) or isinstance(start, bool) or \
            not isinstance(end, (int, float)) or isinstance(end, bool):
        return None
    start_text = str(int(start)) if float(start).is_integer() else f"{start:g}"
    end_text = str(int(end)) if float(end).is_integer() else f"{end:g}"
    return f"{start_text}_{end_text}"


def _project_tool_trajectory(
    resolved: Mapping[str, Any], reference: str | None,
) -> dict[str, Any]:
    """Project only executed calls, exact returned evidence, and segment IDs."""
    availability = resolved.get("availability")
    if availability == "unavailable":
        if reference is not None:
            raise EpisodeFeedbackRequestError(
                "unavailable trajectory conflicts with episode reference")
        return {"availability": "unavailable", "tool_events": []}
    if availability != "available" or reference is None:
        raise EpisodeFeedbackRequestError(
            "trajectory availability conflicts with episode reference")
    rows = resolved.get("content")
    events = resolved.get("tool_events")
    if not isinstance(rows, list) or not isinstance(events, list):
        raise EpisodeFeedbackRequestError(
            "available trajectory content and tool_events must be arrays")

    projected = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise EpisodeFeedbackRequestError(
                f"tool_events[{index}] must be an object")
        hits = event.get("hits") or ()
        if not isinstance(hits, (list, tuple)):
            raise EpisodeFeedbackRequestError(
                f"tool_events[{index}].hits must be an array")
        projected.append({
            "tool": event.get("tool"),
            "args": _plain_json(event.get("args") or {}),
            "returned_evidence": [],
            "returned_segment_ids": [
                segment_id for segment_id in (
                    _hit_segment_id(hit) if isinstance(hit, Mapping) else None
                    for hit in hits)
                if segment_id is not None
            ],
        })

    call_to_event: dict[str, int] = {}
    used_events: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("role") != "assistant":
            continue
        calls = row.get("tool_calls") or ()
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            call_id = call.get("id")
            if not isinstance(function, Mapping) or not isinstance(call_id, str):
                continue
            name = function.get("name")
            if name == "finish":
                continue
            arguments = _parse_call_arguments(function.get("arguments"))
            match = next((event_index for event_index, event in enumerate(events)
                          if event_index not in used_events
                          and event.get("tool") == name
                          and _plain_json(event.get(
                              "requested_args", event.get("args"))) ==
                          _plain_json(arguments)), None)
            if match is not None:
                used_events.add(match)
                call_to_event[call_id] = match

    for row in rows:
        if not isinstance(row, Mapping) or row.get("role") != "tool":
            continue
        event_index = call_to_event.get(row.get("tool_call_id"))
        if event_index is None or row.get("name") != projected[event_index]["tool"]:
            continue
        content = row.get("content")
        if not isinstance(content, str):
            raise EpisodeFeedbackRequestError(
                "stored returned tool evidence must be text")
        projected[event_index]["returned_evidence"].append(content)
    return {"availability": "available", "tool_events": projected}

def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_exact_compact_character_count(view: dict[str, Any]) -> dict[str, Any]:
    value = 0
    for _ in range(8):
        view["projection_statistics"]["compact_character_count"] = value
        observed = len(dumps_canonical(view))
        if observed == value:
            return view
        value = observed
    raise EpisodeFeedbackRequestError(
        "compact trajectory character count did not stabilize")


def _compact_trajectory(
    resolved: Mapping[str, Any], reference: str | None,
) -> dict[str, Any]:
    if resolved.get("availability") == "unavailable":
        if reference is not None:
            raise EpisodeFeedbackRequestError(
                "unavailable trajectory conflicts with episode reference")
        return _with_exact_compact_character_count({
            "availability": "unavailable",
            "source": None,
            "assistant_steps": [],
            "tool_events": [],
            "final_response": None,
            "unclassified_messages": [],
            "reference_sets": {},
            "reference_evidence": [],
            "referenced_segment_ids": [],
            "retrieved_segment_ids": [],
            "projection_statistics": {
                "raw_message_count": 0,
                "raw_character_count": 0,
                "included_assistant_step_count": 0,
                "included_tool_event_count": 0,
                "included_final_response_count": 0,
                "excluded_boilerplate_message_count": 0,
                "excluded_duplicate_wrapper_count": 0,
                "unclassified_message_count": 0,
                "compact_character_count": 0,
            },
        })
    if resolved.get("availability") != "available" or reference is None:
        raise EpisodeFeedbackRequestError(
            "trajectory availability conflicts with episode reference")
    reference = os.path.abspath(reference)
    try:
        with open(reference, encoding="utf-8") as handle:
            raw_text = handle.read()
    except OSError as exc:
        raise EpisodeFeedbackRequestError(
            f"cannot read raw trajectory {reference}: {exc}") from exc
    rows = list(_read_jsonl_objects(reference, "trajectory"))
    if _plain_json(rows) != _plain_json(resolved.get("content")):
        raise EpisodeFeedbackRequestError(
            "resolved trajectory content changed before compact projection")
    tool_events_path = os.path.join(os.path.dirname(reference), "tool_events.jsonl")
    if os.path.isfile(tool_events_path):
        events = list(_read_jsonl_objects(
            tool_events_path, "trajectory tool events"))
        if _plain_json(events) != _plain_json(resolved.get("tool_events")):
            raise EpisodeFeedbackRequestError(
                "resolved tool events changed before compact projection")
        tool_events_sha = _sha256_file(tool_events_path)
        tool_events_ref = os.path.abspath(tool_events_path)
    else:
        events = []
        if resolved.get("tool_events"):
            raise EpisodeFeedbackRequestError(
                "resolved tool events have no source artifact")
        tool_events_sha = None
        tool_events_ref = None

    compact_events = []
    for event_index, event in enumerate(events):
        projected = {
            key: _plain_json(value) for key, value in event.items()
            if key != "latency_seconds"
        }
        hits = event.get("hits") or ()
        if not isinstance(hits, (list, tuple)):
            raise EpisodeFeedbackRequestError(
                f"tool event {event_index} hits must be an array")
        projected["event_index"] = event_index
        projected["returned_segment_ids"] = [
            segment_id for segment_id in (
                _hit_segment_id(hit) if isinstance(hit, Mapping) else None
                for hit in hits) if segment_id is not None]
        projected["returned_evidence"] = []
        compact_events.append(projected)

    assistant_steps = []
    unclassified = []
    final_response = None
    call_to_event: dict[str, int] = {}
    duplicate_wrappers = 0
    boilerplate = 0
    used_events: set[int] = set()
    allowed_assistant_keys = {
        "role", "content", "tool_calls", "annotations", "refusal"}
    allowed_tool_keys = {"role", "content", "name", "tool_call_id"}
    for message_index, row in enumerate(rows):
        role = row.get("role")
        if role in ("system", "user") and set(row) <= {"role", "content"}:
            boilerplate += 1
            continue
        if role == "assistant" and set(row) <= allowed_assistant_keys:
            calls = row.get("tool_calls") or ()
            content = row.get("content")
            # OpenAI chat transport metadata is not semantic trace content.
            # Preserve any non-empty/unknown value instead of discarding it.
            if row.get("annotations") not in (None, []) or \
                    row.get("refusal") is not None:
                unclassified.append({
                    "message_index": message_index,
                    "message": _plain_json(row)})
                continue
            if calls:
                if not isinstance(calls, list) or any(
                        not isinstance(call, Mapping) for call in calls):
                    unclassified.append({
                        "message_index": message_index, "message": _plain_json(row)})
                    continue
                parsed_calls = []
                invalid_call = False
                for call in calls:
                    function = call.get("function")
                    call_id = call.get("id")
                    if not isinstance(function, Mapping) or \
                            not isinstance(call_id, str) or not call_id:
                        invalid_call = True
                        break
                    parsed_calls.append((
                        call_id, function.get("name"),
                        _parse_call_arguments(function.get("arguments"))))
                if invalid_call:
                    unclassified.append({
                        "message_index": message_index, "message": _plain_json(row)})
                    continue
                finish_calls = [item for item in parsed_calls
                                if item[1] == "finish"]
                if finish_calls:
                    if len(parsed_calls) != 1 or len(finish_calls) != 1:
                        unclassified.append({
                            "message_index": message_index,
                            "message": _plain_json(row)})
                        continue
                    if final_response is not None:
                        raise EpisodeFeedbackRequestError(
                            "trajectory contains multiple final responses")
                    final_response = {
                        "message_index": message_index,
                        "finish_arguments": finish_calls[0][2],
                        "response_text": content if isinstance(content, str) else None,
                    }
                    continue
                if content is not None and not isinstance(content, str):
                    unclassified.append({
                        "message_index": message_index, "message": _plain_json(row)})
                    continue
                pending_matches = []
                pending_used = set(used_events)
                for call_id, name, arguments in parsed_calls:
                    match = None
                    for event_index, event in enumerate(events):
                        comparable_args = event.get(
                            "requested_args", event.get("args"))
                        if event_index not in pending_used and \
                                event.get("tool") == name and \
                                _plain_json(comparable_args) == \
                                _plain_json(arguments):
                            match = event_index
                            break
                    if match is None:
                        pending_matches = []
                        break
                    pending_used.add(match)
                    pending_matches.append((call_id, match))
                if len(pending_matches) != len(parsed_calls):
                    unclassified.append({
                        "message_index": message_index, "message": _plain_json(row)})
                    continue
                used_events.update(match for _call_id, match in pending_matches)
                call_to_event.update(pending_matches)
                duplicate_wrappers += len(pending_matches)
                if isinstance(content, str) and content:
                    assistant_steps.append({
                        "message_index": message_index, "content": content})
                continue
            if isinstance(content, str):
                assistant_steps.append({
                    "message_index": message_index, "content": content})
                continue
            unclassified.append({
                "message_index": message_index, "message": _plain_json(row)})
            continue
        if role == "tool" and set(row) <= allowed_tool_keys:
            event_index = call_to_event.get(row.get("tool_call_id"))
            if event_index is None or row.get("name") != \
                    events[event_index].get("tool"):
                unclassified.append({
                    "message_index": message_index, "message": _plain_json(row)})
                continue
            compact_events[event_index]["returned_evidence"].append(
                _plain_json(row.get("content")))
            duplicate_wrappers += 1
            continue
        unclassified.append({
            "message_index": message_index, "message": _plain_json(row)})

    reference_sets = _plain_json(resolved.get("reference_sets") or {})
    reference_evidence = _plain_json(
        resolved.get("reference_evidence") or [])
    view = {
        "availability": "available",
        "source": {
            "trajectory_ref": reference,
            "trajectory_sha256": _sha256_file(reference),
            "tool_events_ref": tool_events_ref,
            "tool_events_sha256": tool_events_sha,
            "reference_sets_sha256": sha256_json(reference_sets),
            "reference_evidence_sha256": sha256_json(reference_evidence),
        },
        "assistant_steps": assistant_steps,
        "tool_events": compact_events,
        "final_response": final_response,
        "unclassified_messages": unclassified,
        "reference_sets": reference_sets,
        "reference_evidence": reference_evidence,
        "referenced_segment_ids": _plain_json(
            resolved.get("referenced_segment_ids") or []),
        "retrieved_segment_ids": _plain_json(
            resolved.get("retrieved_segment_ids") or []),
        "projection_statistics": {
            "raw_message_count": len(rows),
            "raw_character_count": len(raw_text),
            "included_assistant_step_count": len(assistant_steps),
            "included_tool_event_count": len(compact_events),
            "included_final_response_count": int(final_response is not None),
            "excluded_boilerplate_message_count": boilerplate,
            "excluded_duplicate_wrapper_count": duplicate_wrappers,
            "unclassified_message_count": len(unclassified),
            "compact_character_count": 0,
        },
    }
    return _with_exact_compact_character_count(view)



def _model_trajectory_projection(
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate semantic trajectory evidence from v2 audit/debug fields."""
    assistant_steps = []
    assistant_message_indices = []
    for step in trajectory.get("assistant_steps", ()):
        if not isinstance(step, Mapping):
            raise EpisodeFeedbackRequestError(
                "compact trajectory assistant step must be an object")
        assistant_steps.append({
            key: _plain_json(value) for key, value in step.items()
            if key != "message_index"
        })
        assistant_message_indices.append(step.get("message_index"))

    tool_events = []
    tool_event_indices = []
    for event in trajectory.get("tool_events", ()):
        if not isinstance(event, Mapping):
            raise EpisodeFeedbackRequestError(
                "compact trajectory tool event must be an object")
        tool_events.append({
            key: _plain_json(value) for key, value in event.items()
            if key not in {"event_index", "hits"}
        })
        tool_event_indices.append(event.get("event_index"))

    final_response = trajectory.get("final_response")
    final_message_index = None
    if final_response is not None:
        if not isinstance(final_response, Mapping):
            raise EpisodeFeedbackRequestError(
                "compact trajectory final response must be an object or null")
        final_message_index = final_response.get("message_index")
        final_response = {
            key: _plain_json(value) for key, value in final_response.items()
            if key != "message_index"
        }

    unclassified_messages = []
    unclassified_message_indices = []
    for row in trajectory.get("unclassified_messages", ()):
        if not isinstance(row, Mapping) or "message" not in row:
            raise EpisodeFeedbackRequestError(
                "compact unclassified trajectory message is invalid")
        unclassified_messages.append(_plain_json(row["message"]))
        unclassified_message_indices.append(row.get("message_index"))

    model_view = {
        "availability": trajectory.get("availability"),
        "assistant_steps": assistant_steps,
        "tool_events": tool_events,
        "final_response": final_response,
        "unclassified_messages": unclassified_messages,
        "referenced_segment_ids": _plain_json(
            trajectory.get("referenced_segment_ids") or []),
        "retrieved_segment_ids": _plain_json(
            trajectory.get("retrieved_segment_ids") or []),
    }
    audit_view = {
        "source": _plain_json(trajectory.get("source")),
        "projection_statistics": _plain_json(
            trajectory.get("projection_statistics") or {}),
        "reference_sets": _plain_json(
            trajectory.get("reference_sets") or {}),
        "reference_evidence": _plain_json(
            trajectory.get("reference_evidence") or []),
        "projection_debug": {
            "assistant_message_indices": assistant_message_indices,
            "tool_event_indices": tool_event_indices,
            "final_response_message_index": final_message_index,
            "unclassified_message_indices": unclassified_message_indices,
        },
    }
    return model_view, audit_view




_LEAN_QUERY_FIELD_BY_TOOL = {
    "global_browse_tool": "query",
    "clip_search_tool": "event_description",
    "frame_inspect_tool": "question",
}

def _lean_tool_calls(
    trajectory: Mapping[str, Any], *, changed_segment_ids: frozenset[str],
) -> list[dict[str, Any]]:
    calls = []
    for index, event in enumerate(trajectory.get("tool_events") or ()):
        if not isinstance(event, Mapping):
            raise EpisodeFeedbackRequestError(
                f"tool_events[{index}] must be an object")
        tool = event.get("tool")
        query_field = _LEAN_QUERY_FIELD_BY_TOOL.get(tool)
        if query_field is None:
            raise EpisodeFeedbackRequestError(
                f"unsupported feedback tool event: {tool!r}")
        args = event.get("args")
        if not isinstance(args, Mapping):
            raise EpisodeFeedbackRequestError(
                f"tool_events[{index}].args must be an object")
        query = args.get(query_field)
        if not isinstance(query, str) or not query.strip():
            raise EpisodeFeedbackRequestError(
                f"tool_events[{index}] has no exact {query_field!r} query")
        returned_evidence = event.get("returned_evidence") or ()
        if not isinstance(returned_evidence, (list, tuple)) or any(
                not isinstance(item, str) for item in returned_evidence):
            raise EpisodeFeedbackRequestError(
                f"tool_events[{index}].returned_evidence must be a string array")
        returned_ids = event.get("returned_segment_ids") or ()
        if not isinstance(returned_ids, (list, tuple)) or any(
                not isinstance(item, str) or not item for item in returned_ids):
            raise EpisodeFeedbackRequestError(
                f"tool_events[{index}].returned_segment_ids is invalid")
        calls.append({
            "query": query,
            "returned_evidence": list(returned_evidence),
            "segment_ids": [
                item for item in returned_ids if item in changed_segment_ids],
        })
    return calls


def build_lean_episode_feedback_request(
    episode: InterventionEpisode,
    *,
    artifact_resolver: EpisodeFeedbackArtifactResolver,
    token_counter: Callable[[tuple[Mapping[str, str], ...]], int] | None = None,
) -> LeanEpisodeFeedbackRequest:
    """Build the only active feedback request without lossy truncation."""
    resolved_qas = artifact_resolver.resolve_qas(episode)
    if len(resolved_qas) != len(episode.qa_outcomes):
        raise EpisodeFeedbackRequestError(
            "resolved QA count does not match the intervention episode")
    changed = tuple(
        clip for clip in episode.clips
        if clip.baseline_caption != clip.intervention_caption)
    changed_ids = frozenset(clip.segment_id for clip in changed)
    changed_captions = [{
        "segment_id": clip.segment_id,
        "baseline_caption": clip.baseline_caption,
        "intervention_caption": clip.intervention_caption,
    } for clip in changed]

    qas = []
    tool_call_count = 0
    for outcome, qa in zip(episode.qa_outcomes, resolved_qas):
        if qa.get("qa_id") != outcome.qa_id or \
                qa.get("is_source_qa") is not outcome.is_source_qa:
            raise EpisodeFeedbackRequestError(
                "resolved QA identity/order conflicts with the episode")
        question = qa.get("question")
        if not isinstance(question, str) or not question:
            raise EpisodeFeedbackRequestError(
                f"QA {outcome.qa_id!r} must have an exact non-empty question")
        choices = qa.get("answer_choices")
        if not isinstance(choices, list) or len(choices) != 4 or any(
                not isinstance(item, str) or not item for item in choices):
            raise EpisodeFeedbackRequestError(
                f"QA {outcome.qa_id!r} must have exactly four answer choices")
        for field in ("gold_answer", "baseline_answer", "intervention_answer"):
            if not isinstance(qa.get(field), str) or not qa[field]:
                raise EpisodeFeedbackRequestError(
                    f"QA {outcome.qa_id!r} must have an exact non-empty {field}")
        if qa.get("baseline_correct") is not outcome.baseline_correct or \
                qa.get("intervention_correct") is not \
                outcome.intervention_correct or qa.get("transition") != \
                classify_qa_transition(outcome):
            raise EpisodeFeedbackRequestError(
                f"QA {outcome.qa_id!r} outcome fields conflict with the episode")
        projected = {
            key: _plain_json(qa[key]) for key in (
                "qa_id", "is_source_qa", "question", "answer_choices",
                "gold_answer", "baseline_answer", "baseline_correct",
                "intervention_answer", "intervention_correct", "transition")
        }
        for side, reference in (
                ("baseline", outcome.baseline_trajectory_ref),
                ("intervention", outcome.intervention_trajectory_ref)):
            compact = _project_tool_trajectory(qa[f"{side}_trajectory"], reference)
            calls = _lean_tool_calls(
                compact, changed_segment_ids=changed_ids)
            projected[f"{side}_tool_calls"] = calls
            tool_call_count += len(calls)
        qas.append(projected)

    payload = json.loads(dumps_canonical({
        "schema_version": LEAN_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION,
        "episode": {
            "episode_id": episode.episode_id,
            "prompt_delta": _plain_json(episode.prompt_delta),
            "qa_transition_summary": build_qa_transition_summary(episode),
            "changed_captions": changed_captions,
            "qas": qas,
        },
    }))
    user_request = dumps_canonical(payload)
    messages = (
        {"role": "system", "content": EPISODE_FEEDBACK_SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_request},
    )
    token_count = token_counter(messages) if token_counter is not None else None
    if token_count is not None and (
            not isinstance(token_count, int) or isinstance(token_count, bool) or
            token_count < 0):
        raise EpisodeFeedbackBackendConfigurationError(
            "backend token counter must return a non-negative integer")
    payload_hash = sha256_json(payload)
    return LeanEpisodeFeedbackRequest(
        system_instruction=EPISODE_FEEDBACK_SYSTEM_INSTRUCTION,
        model_payload=payload,
        model_payload_hash=payload_hash,
        messages=messages,
        size_statistics=LeanEpisodeFeedbackRequestSize(
            changed_caption_count=len(changed),
            qa_count=len(qas),
            tool_call_count=tool_call_count,
            model_payload_character_count=len(user_request),
            token_count=token_count,
        ),
    )


_DETAILED_RESPONSE_FIELDS = {
    "episode_id", "outcome_summary", "observations", "counterevidence",
    "generator_diagnosis", "recommended_strategy_change", "confidence",
}
_RESPONSE_FIELDS = _DETAILED_RESPONSE_FIELDS | {"compact_memory_text"}
_EVIDENCE_FIELDS = {
    "statement", "supporting_segment_ids", "supporting_qa_ids",
    "evidence_type", "confidence",
}


def parse_episode_feedback_response(
    raw_response: Any,
    *,
    episode: InterventionEpisode,
    policy_version: str,
    request_payload_hash: str,
) -> EpisodeFeedback:
    """Accept exactly one JSON object; perform no extraction or repair."""
    try:
        if not isinstance(raw_response, str):
            raise TypeError("response must be a string")
        value = json.loads(raw_response)
        if not isinstance(value, dict) or set(value) not in (
                _DETAILED_RESPONSE_FIELDS, _RESPONSE_FIELDS):
            raise ValueError(
                "response must contain the detailed feedback fields and only "
                "the optional compact_memory_text field")
        compact_memory = value.get("compact_memory_text")
        if compact_memory is not None and not isinstance(compact_memory, str):
            raise TypeError("compact_memory_text must be a string or null")
        value["compact_memory_text"] = (
            compact_memory.strip() or None
            if isinstance(compact_memory, str) else None)
        if value.get("episode_id") != episode.episode_id:
            raise ValueError("response episode_id does not match input episode")
        for collection_name in ("observations", "counterevidence"):
            items = value[collection_name]
            if not isinstance(items, list):
                raise TypeError(f"{collection_name} must be an array")
            for index, item in enumerate(items):
                if not isinstance(item, dict) or set(item) != _EVIDENCE_FIELDS:
                    raise ValueError(
                        f"{collection_name}[{index}] does not match strict fields")
                segment_ids = item["supporting_segment_ids"]
                qa_ids = item["supporting_qa_ids"]
                for field_name, identifiers in (
                        ("supporting_segment_ids", segment_ids),
                        ("supporting_qa_ids", qa_ids)):
                    if not isinstance(identifiers, list) or any(
                            not isinstance(identifier, str) or not identifier
                            for identifier in identifiers):
                        raise TypeError(
                            f"{collection_name}[{index}].{field_name} must be "
                            "a non-empty-string array")
        identity = {
            "episode_id": episode.episode_id,
            "feedback_policy_version": policy_version,
            "request_payload_hash": request_payload_hash,
        }
        for collection_name in ("observations", "counterevidence"):
            for item in value[collection_name]:
                item["transition_type"] = None
        feedback = episode_feedback_from_json({
            "feedback_id": "episode_feedback_" + sha256_json(identity)[:20],
            **value,
        })
        restored = episode_feedback_from_json(
            json.loads(dumps_canonical(feedback)))
        if restored != feedback or dumps_canonical(restored) != \
                dumps_canonical(feedback):
            raise ValueError("EpisodeFeedback failed canonical round-trip")
        return feedback
    except EpisodeFeedbackParseError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EpisodeFeedbackParseError(
            f"invalid episode feedback response: {exc}",
            raw_response=raw_response) from exc


def _backend_metadata(
    provider: EpisodeFeedbackResponseProvider,
) -> dict[str, Any]:
    metadata_fn = getattr(provider, "metadata", None)
    if not callable(metadata_fn):
        raise EpisodeFeedbackBackendConfigurationError(
            "response provider must expose metadata()")
    metadata = metadata_fn()
    if not isinstance(metadata, Mapping):
        raise EpisodeFeedbackBackendConfigurationError(
            "response provider metadata must be an object")
    metadata = json.loads(dumps_canonical(metadata))
    if not isinstance(metadata.get("provider"), str) or not metadata["provider"]:
        raise EpisodeFeedbackBackendConfigurationError(
            "backend provider identity is required")
    if not isinstance(metadata.get("model"), str) or not metadata["model"]:
        raise EpisodeFeedbackBackendConfigurationError(
            "backend model identity is required")
    if not isinstance(metadata.get("generation_settings"), Mapping) or not \
            metadata["generation_settings"]:
        raise EpisodeFeedbackBackendConfigurationError(
            "explicit backend generation_settings are required")
    output_limit = metadata.get("output_token_limit")
    if not isinstance(output_limit, int) or isinstance(output_limit, bool) or \
            output_limit <= 0:
        raise EpisodeFeedbackBackendConfigurationError(
            "explicit positive output_token_limit is required")
    if "context_limit_tokens" not in metadata:
        raise EpisodeFeedbackBackendConfigurationError(
            "context_limit_tokens must be explicit, including null when unknown")
    context_limit = metadata["context_limit_tokens"]
    if context_limit is not None and (
            not isinstance(context_limit, int) or isinstance(context_limit, bool)
            or context_limit <= 0):
        raise EpisodeFeedbackBackendConfigurationError(
            "context_limit_tokens must be a positive integer or null")
    return metadata


class LLMEpisodeFeedbackGenerator:
    """One-call injected policy using the sole active lean request."""

    def __init__(
        self,
        *,
        response_provider: EpisodeFeedbackResponseProvider,
        artifact_resolver: EpisodeFeedbackArtifactResolver,
        policy_version: str,
    ) -> None:
        if not callable(response_provider):
            raise EpisodeFeedbackBackendConfigurationError(
                "response_provider must be callable")
        if artifact_resolver is None:
            raise EpisodeFeedbackBackendConfigurationError(
                "artifact_resolver is required")
        if not isinstance(policy_version, str) or not policy_version:
            raise EpisodeFeedbackBackendConfigurationError(
                "policy_version must be an explicit non-empty string")
        self.response_provider = response_provider
        self.artifact_resolver = artifact_resolver
        self.policy_version = policy_version

    def generate(self, episode: InterventionEpisode) -> EpisodeFeedback:
        return self.generate_with_trace(episode).feedback

    def generate_to_directory(
        self, episode: InterventionEpisode, artifact_directory: str,
    ) -> EpisodeFeedback:
        return self._generate_with_trace(
            episode, artifact_directory=artifact_directory).feedback

    def generate_with_trace(
        self, episode: InterventionEpisode,
    ) -> EpisodeFeedbackInvocationResult:
        return self._generate_with_trace(episode, artifact_directory=None)

    def _generate_with_trace(
        self, episode: InterventionEpisode, *, artifact_directory: str | None,
    ) -> EpisodeFeedbackInvocationResult:
        metadata = _backend_metadata(self.response_provider)
        token_counter = getattr(self.response_provider, "count_tokens", None)
        if token_counter is not None and not callable(token_counter):
            raise EpisodeFeedbackBackendConfigurationError(
                "backend count_tokens must be callable")
        preflight = getattr(self.response_provider, "preflight", None)
        if preflight is not None and not callable(preflight):
            raise EpisodeFeedbackBackendConfigurationError(
                "backend preflight must be callable")
        request = build_lean_episode_feedback_request(
            episode, artifact_resolver=self.artifact_resolver,
            token_counter=token_counter)
        if preflight is None:
            limit = metadata["context_limit_tokens"]
            observed = request.size_statistics.token_count
            if limit is not None and observed is not None and observed > limit:
                sections = request.user_payload["episode"]
                largest = tuple(sorted((
                    ("changed_captions", len(dumps_canonical(
                        sections["changed_captions"]))),
                    ("qas", len(dumps_canonical(sections["qas"]))),
                    ("system_instruction", len(request.system_instruction)),
                ), key=lambda item: (-item[1], item[0])))
                raise EpisodeFeedbackContextOverflowError(
                    observed_tokens=observed, configured_limit=limit,
                    clip_count=len(episode.clips),
                    qa_count=len(episode.qa_outcomes),
                    largest_payload_sections=largest)
        prepared = None
        if preflight is not None:
            prepared_value = preflight(
                request.system_instruction, request.user_request)
            if isinstance(prepared_value, tuple) and prepared_value:
                prepared = prepared_value[0]
        if artifact_directory is not None:
            _persist_feedback_request(
                artifact_directory, request=request, prepared=prepared,
                policy_version=self.policy_version,
                backend_metadata=metadata)
        raw_response_path = (
            os.path.join(artifact_directory, "raw_response.json")
            if artifact_directory is not None else None)
        if raw_response_path is not None and os.path.isfile(raw_response_path):
            with open(raw_response_path, encoding="utf-8") as handle:
                raw = handle.read()
        else:
            raw = self.response_provider(
                request.system_instruction, request.user_request)
        if raw_response_path is not None:
            _write_feedback_text_once(
                raw_response_path, raw)
        feedback = parse_episode_feedback_response(
            raw, episode=episode, policy_version=self.policy_version,
            request_payload_hash=request.payload_hash)
        return EpisodeFeedbackInvocationResult(
            request=request,
            raw_response=raw,
            feedback=feedback,
            policy_version=self.policy_version,
            backend_metadata=_backend_metadata(self.response_provider),
        )


def _write_feedback_text_once(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    value = text if text.endswith("\n") else text + "\n"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            if handle.read() != value:
                raise EpisodeFeedbackRequestError(
                    f"immutable feedback artifact conflict: {path}")
        return
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(value)
    os.replace(temporary, path)


def _persist_feedback_request(
    directory: str, *, request: LeanEpisodeFeedbackRequest, prepared: Any,
    policy_version: str, backend_metadata: Mapping[str, Any],
) -> None:
    _write_feedback_text_once(
        os.path.join(directory, "request.json"), request.user_request)
    provider_body = getattr(prepared, "request_body", None)
    if provider_body is not None:
        _write_feedback_text_once(
            os.path.join(directory, "provider_request.json"),
            dumps_canonical(provider_body))
    stable_backend_metadata = _plain_json(backend_metadata)
    stable_backend_metadata.pop("call_count", None)
    manifest = {
        "schema_version": "episode_feedback_request_artifact_v1",
        "request_schema_version": LEAN_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION,
        "episode_id": request.model_payload["episode"]["episode_id"],
        "policy_version": policy_version,
        "model_payload_hash": request.model_payload_hash,
        "system_prompt_hash": sha256_text(request.system_instruction),
        "backend": stable_backend_metadata,
        "provider_request_persisted": provider_body is not None,
    }
    _write_feedback_manifest_once(
        os.path.join(directory, "request_manifest.json"), manifest)


def _write_feedback_manifest_once(
    path: str, manifest: Mapping[str, Any],
) -> None:
    """Compare immutable request identity without process-local counters."""
    normalized = _plain_json(manifest)
    backend = normalized.get("backend")
    if isinstance(backend, dict):
        backend.pop("call_count", None)
    if os.path.exists(path):
        existing = _read_json_object(path, "feedback request manifest")
        existing_backend = existing.get("backend")
        if isinstance(existing_backend, dict):
            existing_backend.pop("call_count", None)
        if dumps_canonical(existing) != dumps_canonical(normalized):
            raise EpisodeFeedbackRequestError(
                f"immutable feedback artifact conflict: {path}")
        return
    _write_feedback_text_once(path, dumps_canonical(normalized))
