"""Fresh baseline-to-InterventionEpisode evidence path.

This is deliberately parallel to the legacy property path.  It consumes a
free-form, history-aware baseline, proposes ephemeral PromptDelta records, and
selectively recaptions frozen baseline prompts/histories.  It never reads or
writes a property codebook/router state.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

from surrogate_rollout.captioning.candidate_captions import build_clip_index
from surrogate_rollout.evaluation.dvd_qa import (
    dvd_qa_execution_identity,
    run_dvd_qa,
)
from surrogate_rollout.mixed_views.builder import MixedViewBuilder
from surrogate_rollout.optimization.llm_episode_feedback import (
    _compact_trajectory,
    _model_trajectory_projection,
    _resolve_trajectory,
)
from surrogate_rollout.optimization.schemas import (
    InterventionClipRecord,
    InterventionEpisode,
    PromptDelta,
    QAInterventionOutcome,
    prompt_delta_from_json,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.scaffold_applier import (
    read_composed_prompts,
    validate_composed_text,
)
from surrogate_rollout.prompt_routing.schemas import (
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_json, sha256_text
from surrogate_rollout.references.extractor import (
    clips_for_range, hhmmss_to_seconds,
)


PROMPT_DELTA_PROPOSAL_REQUEST_SCHEMA_VERSION = (
    "prompt_delta_proposal_request_v5_per_qa_isolated")
PROMPT_DELTA_PROPOSAL_REPRESENTATION_VERSION = (
    "trajectory_grounded_per_qa_catalog_v4_localized_inspection")
PROMPT_DELTA_PROPOSAL_EVIDENCE_SCOPE = (
    "assistant_timestamp_or_localized_frame_inspection_v1")
PROMPT_DELTA_PROPOSAL_SPLIT_POLICY = (
    "always_per_qa_isolated_v1")
PROMPT_DELTA_SEGMENT_SELECTION_POLICY = (
    "source_qa_localized_trajectory_segments_only_v1")
PROMPT_DELTA_FRAME_INSPECTION_CLASSIFICATION_POLICY = (
    "frame_inspection_global_boundary_tolerance_v1")
PROMPT_DELTA_POLICY_OUTPUT_NAMESPACE = (
    "per_qa_isolated_localized_trajectory_segments_v2")
PROMPT_DELTA_INTERVENTION_EXECUTION_NAMESPACE = (
    "dvd_strict_frame_inspect_corrective_retry_v1")


def correctness_transition(before: bool, after: bool) -> str:
    if not before and after:
        return "wrong_to_correct"
    if before and not after:
        return "correct_to_wrong"
    if before and after:
        return "correct_to_correct"
    return "wrong_to_wrong"


_PROMPT_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompts")


def _load_prompt_text(filename: str) -> str:
    path = os.path.join(_PROMPT_DIRECTORY, filename)
    try:
        with open(path, encoding="utf-8") as handle:
            value = handle.read().strip()
    except OSError as exc:
        raise RuntimeError(f"prompt-delta prompt unavailable: {path}") from exc
    if not value:
        raise RuntimeError(f"prompt-delta prompt is empty: {path}")
    return value


PROMPT_DELTA_PROPOSAL_SYSTEM_INSTRUCTION = _load_prompt_text(
    "prompt_delta_proposer_system_v6.txt")


class FreshPromptDeltaError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromptDeltaRequestPreflight:
    system_prompt_tokens: int
    user_payload_tokens: int
    total_input_tokens: int
    reserved_output_tokens: int
    context_limit: int
    remaining_tokens: int
    fits_context: bool


class PromptDeltaContextOverflowError(FreshPromptDeltaError):
    def __init__(self, preflight: PromptDeltaRequestPreflight):
        self.preflight = preflight
        super().__init__(
            "prompt-delta proposal exceeds configured context; no truncation, "
            "filtering, sampling, or intra-QA splitting was performed")


class PromptDeltaProposalBackend(Protocol):
    policy_version: str

    @property
    def configuration_identity(self) -> Mapping[str, Any]:
        ...

    def preflight(
        self, system_instruction: str, user_payload: str,
    ) -> PromptDeltaRequestPreflight:
        ...

    def generate(
        self, system_instruction: str, user_payload: str, *,
        minimum_proposals: int, maximum_proposals: int,
    ) -> str:
        ...


class OpenAICompatiblePromptDeltaProposalBackend:
    """Strict one-request-per-call backend with exact preflight accounting."""

    def __init__(
        self, *, provider: str, provider_endpoint: str,
        model_id: str, context_limit: int,
        maximum_output_tokens: int, generation_settings: Mapping[str, Any],
        policy_version: str, exact_token_counter: Callable[[tuple[
            Mapping[str, str], ...]], Any], response_transport: Callable[[
                Mapping[str, Any]], Mapping[str, Any]],
        tokenizer_identity: str, maximum_calls: int,
    ) -> None:
        if not provider or not provider_endpoint or not model_id or \
                context_limit <= 0 or maximum_output_tokens <= 0 or \
                maximum_calls <= 0 or \
                not policy_version or not generation_settings or \
                not tokenizer_identity or \
                not callable(exact_token_counter) or not callable(response_transport):
            raise ValueError("all prompt-delta provider settings are required")
        self.provider = provider
        self.provider_endpoint = provider_endpoint
        self.model_id = model_id
        self.context_limit = context_limit
        self.maximum_output_tokens = maximum_output_tokens
        self.generation_settings = json.loads(dumps_canonical(
            generation_settings))
        self.policy_version = policy_version
        self.tokenizer_identity = tokenizer_identity
        self.maximum_calls = maximum_calls
        self.exact_token_counter = exact_token_counter
        self.response_transport = response_transport
        self.call_count = 0

    @property
    def configuration_identity(self) -> Mapping[str, Any]:
        return {
            "provider": self.provider,
            "provider_endpoint": self.provider_endpoint,
            "model_id": self.model_id,
            "tokenizer_identity": self.tokenizer_identity,
            "context_limit": self.context_limit,
            "maximum_output_tokens": self.maximum_output_tokens,
            "generation_settings": self.generation_settings,
            "policy_version": self.policy_version,
            "maximum_calls": self.maximum_calls,
        }

    def preflight(
        self, system_instruction: str, user_payload: str,
    ) -> PromptDeltaRequestPreflight:
        messages = (
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_payload},
        )
        count = self.exact_token_counter(messages)
        system_tokens = int(getattr(count, "system_prompt_tokens"))
        user_tokens = int(getattr(count, "user_payload_tokens"))
        input_tokens = int(getattr(count, "total_input_tokens"))
        remaining = self.context_limit - input_tokens - self.maximum_output_tokens
        return PromptDeltaRequestPreflight(
            system_prompt_tokens=system_tokens,
            user_payload_tokens=user_tokens,
            total_input_tokens=input_tokens,
            reserved_output_tokens=self.maximum_output_tokens,
            context_limit=self.context_limit,
            remaining_tokens=remaining,
            fits_context=remaining >= 0,
        )

    def generate(
        self, system_instruction: str, user_payload: str, *,
        minimum_proposals: int, maximum_proposals: int,
    ) -> str:
        if minimum_proposals < 0 or maximum_proposals < minimum_proposals:
            raise ValueError("invalid prompt-delta proposal bounds")
        preflight = self.preflight(system_instruction, user_payload)
        if not preflight.fits_context:
            raise PromptDeltaContextOverflowError(preflight)
        messages = (
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_payload},
        )
        evidence = {
            "type": "object", "additionalProperties": False,
            "required": ["instruction", "source_qa_ids",
                         "proposer_diagnosis"],
            "properties": {
                "instruction": {"type": "string"},
                "source_qa_ids": {"type": "array",
                                  "items": {"type": "string"}},
                "proposer_diagnosis": {"type": "string"},
            },
        }
        body = json.loads(dumps_canonical({
            "model": self.model_id, "messages": messages,
            "max_tokens": self.maximum_output_tokens,
            **self.generation_settings,
            "response_format": {
                "type": "json_schema", "json_schema": {
                    "name": "prompt_delta_proposals_v1", "strict": True,
                    "schema": {
                        "type": "object", "additionalProperties": False,
                        "required": ["proposals"],
                        "properties": {"proposals": {
                            "type": "array", "items": evidence,
                            "minItems": minimum_proposals,
                            "maxItems": maximum_proposals}},
                    },
                },
            },
        }))
        self.call_count += 1
        envelope = self.response_transport(body)
        try:
            raw = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise FreshPromptDeltaError(
                "prompt-delta provider response has no message content") from exc
        if not isinstance(raw, str):
            raise FreshPromptDeltaError(
                "prompt-delta provider content must be a string")
        return raw


@dataclass(frozen=True)
class PromptDeltaExecutionPlan:
    prompt_delta: PromptDelta
    selected_segment_ids: tuple[str, ...]
    selection_policy: str
    frame_inspection_classification_hash: str
    global_inspection_boundary_tolerance_seconds: float


def _source_qa_classification_hash(
    row: Mapping[str, Any], *, tolerance_seconds: float,
) -> str:
    return sha256_json({
        "policy": PROMPT_DELTA_FRAME_INSPECTION_CLASSIFICATION_POLICY,
        "global_inspection_boundary_tolerance_seconds": tolerance_seconds,
        "qa_classification": {
            "qa_id": row["qa_id"],
            "assistant_timestamp_cited_segments":
                row["assistant_timestamp_cited_segments"],
            "localized_frame_inspected_segments":
                row["localized_frame_inspected_segments"],
            "global_frame_inspected_segments":
                row["global_frame_inspected_segments"],
            "frame_inspect_calls": row["frame_inspect_calls"],
        },
    })


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise FreshPromptDeltaError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: str) -> tuple[dict[str, Any], ...]:
    with open(path, encoding="utf-8") as handle:
        values = tuple(json.loads(line) for line in handle if line.strip())
    if any(not isinstance(value, dict) for value in values):
        raise FreshPromptDeltaError(f"expected JSONL objects: {path}")
    return values


def _write_once(path: str, value: Any) -> str:
    text = dumps_canonical(value) + "\n"
    absolute = os.path.abspath(path)
    if os.path.exists(absolute):
        with open(absolute, encoding="utf-8") as handle:
            if handle.read() != text:
                raise FreshPromptDeltaError(
                    f"immutable artifact conflict: {absolute}")
        return absolute
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    _atomic_write_text(absolute, text)
    return absolute


_SEQUENTIAL_HISTORY_SERIALIZED_FIELDS = (
    "schema_version", "block_index", "block_start_seconds",
    "block_end_seconds", "max_history_captions", "preceding_captions",
)
_SEQUENTIAL_HISTORY_DERIVED_FIELDS = {
    "segment_id", "history", "preceding_captions", "preceding_segment_ids",
    "serialized_history", "history_hash",
}


def _plain(value: Any) -> Any:
    return json.loads(dumps_canonical(value))


def _normalize_history_snapshot(
    snapshot: Mapping[str, Any], *, segment_id: str,
    item_rows: list[dict[str, Any]], item_by_ref: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    exact = _plain(snapshot)
    if exact.get("segment_id") != segment_id:
        raise FreshPromptDeltaError(
            f"frozen history segment identity differs for {segment_id!r}")
    history = exact.get("history")
    if not isinstance(history, list) or any(
            not isinstance(item, Mapping) for item in history):
        raise FreshPromptDeltaError(
            f"frozen history items are invalid for {segment_id!r}")
    item_refs = []
    for item in history:
        content = _plain(item)
        digest = sha256_json(content)
        stable_segment_id = content.get("segment_id")
        item_ref = ("history_item_segment_" + stable_segment_id
                    if isinstance(stable_segment_id, str) and stable_segment_id
                    else "history_item_" + digest[:20])
        previous = item_by_ref.get(item_ref)
        if previous is not None and dumps_canonical(previous) != \
                dumps_canonical(content):
            raise FreshPromptDeltaError(
                f"history item hash collision for {item_ref!r}")
        if previous is None:
            item_by_ref[item_ref] = content
            item_rows.append({
                "history_item_ref": item_ref,
                "content_sha256": digest,
                "content": content,
            })
        item_refs.append(item_ref)

    if exact.get("source") == "sequential_history_aware_baseline":
        if exact.get("preceding_captions") != history or \
                exact.get("preceding_segment_ids") != [
                    item.get("segment_id") for item in history]:
            raise FreshPromptDeltaError(
                f"frozen history derived fields conflict for {segment_id!r}")
        try:
            serialized_value = {
                key: (history if key == "preceding_captions" else exact[key])
                for key in _SEQUENTIAL_HISTORY_SERIALIZED_FIELDS
            }
        except KeyError as exc:
            raise FreshPromptDeltaError(
                f"frozen history missing {exc.args[0]!r} for {segment_id!r}") \
                from exc
        serialized = dumps_canonical(serialized_value)
        if exact.get("serialized_history") != serialized or \
                exact.get("history_hash") != sha256_text(serialized):
            raise FreshPromptDeltaError(
                f"frozen history serialization conflicts for {segment_id!r}")
        content = {
            "projection": "sequential_history_aware_baseline_v1",
            "metadata": {
                key: value for key, value in exact.items()
                if key not in _SEQUENTIAL_HISTORY_DERIVED_FIELDS
            },
            "ordered_history_item_refs": item_refs,
        }
    else:
        content = {
            "projection": "opaque_snapshot_without_owner_segment_v1",
            "snapshot_without_segment_id": {
                key: value for key, value in exact.items()
                if key != "segment_id"
            },
        }
    digest = sha256_json(content)
    history_ref = "history_snapshot_" + digest[:20]
    return history_ref, {
        "history_ref": history_ref,
        "content_sha256": digest,
        **content,
    }


def reconstruct_prompt_delta_history_snapshot(
    payload: Mapping[str, Any], *, segment_id: str,
) -> Mapping[str, Any]:
    """Losslessly rebuild one baseline frozen-history snapshot."""
    catalog = payload.get("history_catalog")
    segments = payload.get("segment_catalog")
    if not isinstance(catalog, Mapping) or not isinstance(segments, list):
        raise FreshPromptDeltaError("normalized history catalogs are invalid")
    item_by_ref = {
        row["history_item_ref"]: row["content"]
        for row in catalog.get("history_items", ())
        if isinstance(row, Mapping)
    }
    snapshot_by_ref = {
        row["history_ref"]: row
        for row in catalog.get("snapshots", ())
        if isinstance(row, Mapping)
    }
    matching = [row for row in segments if isinstance(row, Mapping) and
                row.get("segment_id") == segment_id]
    if len(matching) != 1 or matching[0].get("history_ref") not in snapshot_by_ref:
        raise FreshPromptDeltaError(
            f"normalized history reference is invalid for {segment_id!r}")
    row = snapshot_by_ref[matching[0]["history_ref"]]
    projection = row.get("projection")
    if projection == "opaque_snapshot_without_owner_segment_v1":
        result = _plain(row.get("snapshot_without_segment_id"))
        if not isinstance(result, dict):
            raise FreshPromptDeltaError("opaque history snapshot must be an object")
        result["segment_id"] = segment_id
        return result
    if projection != "sequential_history_aware_baseline_v1":
        raise FreshPromptDeltaError(
            f"unknown normalized history projection {projection!r}")
    refs = row.get("ordered_history_item_refs")
    if not isinstance(refs, list) or any(ref not in item_by_ref for ref in refs):
        raise FreshPromptDeltaError("normalized history item reference is invalid")
    history = [_plain(item_by_ref[ref]) for ref in refs]
    result = _plain(row.get("metadata"))
    if not isinstance(result, dict):
        raise FreshPromptDeltaError("normalized history metadata must be an object")
    result["segment_id"] = segment_id
    result["history"] = history
    result["preceding_captions"] = history
    result["preceding_segment_ids"] = [item["segment_id"] for item in history]
    serialized = dumps_canonical({
        key: (history if key == "preceding_captions" else result[key])
        for key in _SEQUENTIAL_HISTORY_SERIALIZED_FIELDS
    })
    result["serialized_history"] = serialized
    result["history_hash"] = sha256_text(serialized)
    return result


def _request_size_breakdown(payload: Mapping[str, Any]) -> dict[str, Any]:
    history = payload["history_catalog"]
    segments = payload["segment_catalog"]
    qas = payload["qa_records"]
    return {
        "total_user_payload_characters": len(dumps_canonical(payload)),
        "history_catalog_characters": len(dumps_canonical(history)),
        "segment_catalog_characters": len(dumps_canonical(segments)),
        "qa_records_characters": len(dumps_canonical(qas)),
        "trajectory_characters": sum(
            len(dumps_canonical(row["trajectory"])) for row in qas),
        "history_snapshot_count": len(history["snapshots"]),
        "unique_history_item_count": len(history["history_items"]),
        "segment_count": len(segments),
        "qa_count": len(qas),
    }


_QA_SEGMENT_PROVENANCE_FIELDS = (
    "used_segments", "explicitly_cited_segments", "frame_inspected_segments",
    "consumed_segments", "retrieved_segments", "returned_segments",
)
_ASSISTANT_TIMESTAMP = re.compile(r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\b")


def _ordered_subset(
    segment_order: Sequence[str], values: set[str],
) -> list[str]:
    return [segment_id for segment_id in segment_order if segment_id in values]


def _qa_segment_selection_records(
    baseline_video_manifest_path: str,
    *, global_inspection_boundary_tolerance_seconds: float,
) -> tuple[dict[str, Any], ...]:
    """Resolve the explicit-localization scope without changing provenance.

    The returned rows are private request/run audit data.  Only
    ``intervention_candidate_segments`` is permitted to enter proposal and
    selective-recaption scope.
    """
    if global_inspection_boundary_tolerance_seconds < 0:
        raise ValueError(
            "global inspection boundary tolerance must be non-negative")
    baseline = _read_json(baseline_video_manifest_path)
    routing = _read_json(baseline["routing_manifest_path"])
    prompts = _read_jsonl(routing["composed_prompts_path"])
    segment_order = tuple(row["segment_id"] for row in prompts)
    if not segment_order or len(set(segment_order)) != len(segment_order):
        raise FreshPromptDeltaError("baseline segment universe is invalid")
    try:
        boundaries = [tuple(float(value) for value in segment_id.split("_", 1))
                      for segment_id in segment_order]
    except (TypeError, ValueError) as exc:
        raise FreshPromptDeltaError(
            "baseline segment boundaries are invalid") from exc
    video_start = min(start for start, _end in boundaries)
    video_end = max(end for _start, end in boundaries)
    rows = []
    for qa in _read_jsonl(baseline["baseline_qas_path"]):
        qa_id = qa.get("question_id")
        refs = qa.get("reference_sets") or {}
        if not isinstance(qa_id, str) or not qa_id or not isinstance(refs, Mapping):
            raise FreshPromptDeltaError("baseline QA reference provenance is invalid")
        provenance: dict[str, list[str]] = {}
        for name in _QA_SEGMENT_PROVENANCE_FIELDS:
            values = qa.get(name) if name == "used_segments" else refs.get(name)
            if values is None:
                values = []
            if not isinstance(values, (list, tuple)) or any(
                    not isinstance(value, str) or not value for value in values):
                raise FreshPromptDeltaError(
                    f"baseline QA {qa_id!r} {name!r} provenance is invalid")
            if len(set(values)) != len(values):
                raise FreshPromptDeltaError(
                    f"baseline QA {qa_id!r} has duplicate {name!r} segments")
            provenance[name] = list(values)
        trajectory_path = qa.get("trajectory_path")
        if not isinstance(trajectory_path, str) or not trajectory_path or \
                not os.path.isfile(trajectory_path):
            raise FreshPromptDeltaError(
                f"baseline QA {qa_id!r} trajectory does not resolve")
        messages = _read_jsonl(trajectory_path)
        assistant_cited: set[str] = set()
        for message in messages:
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            for match in _ASSISTANT_TIMESTAMP.finditer(content):
                try:
                    second = hhmmss_to_seconds(match.group(1))
                except ValueError as exc:
                    raise FreshPromptDeltaError(
                        f"baseline QA {qa_id!r} has invalid assistant timestamp") \
                        from exc
                assistant_cited.update(clips_for_range(
                    second, second, list(segment_order)))

        events_path = os.path.join(
            os.path.dirname(trajectory_path), "tool_events.jsonl")
        events = _read_jsonl(events_path) if os.path.isfile(events_path) else ()
        localized_inspected: set[str] = set()
        global_inspected: set[str] = set()
        call_rows = []
        for event_index, event in enumerate(events):
            if event.get("tool") != "frame_inspect_tool":
                continue
            args = event.get("args")
            ranges = args.get("time_ranges_hhmmss") if isinstance(
                args, Mapping) else None
            # Invalid calls rejected by the strict execution boundary remain
            # audit provenance, but no frame inspection occurred. They cannot
            # localize proposer or intervention evidence.
            if event.get("execution_performed") is False or \
                    event.get("status") == "argument_validation_error":
                call_rows.append({
                    "tool_event_index": event_index,
                    "requested_ranges": ranges,
                    "classification": "invalid_not_executed",
                    "segment_ids": [],
                    "validation_error": event.get("error"),
                })
                continue
            if not isinstance(ranges, list) or not ranges:
                raise FreshPromptDeltaError(
                    f"baseline QA {qa_id!r} frame inspection has no ranges")
            parsed_ranges = []
            call_segments: set[str] = set()
            call_is_global = False
            for value in ranges:
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise FreshPromptDeltaError(
                        f"baseline QA {qa_id!r} frame inspection range is invalid")
                try:
                    start = hhmmss_to_seconds(str(value[0]))
                    end = hhmmss_to_seconds(str(value[1]))
                except ValueError as exc:
                    raise FreshPromptDeltaError(
                        f"baseline QA {qa_id!r} frame inspection range is invalid") \
                        from exc
                if end < start:
                    raise FreshPromptDeltaError(
                        f"baseline QA {qa_id!r} frame inspection range is reversed")
                range_segments = clips_for_range(
                    start, end, list(segment_order))
                call_segments.update(range_segments)
                is_global_range = (
                    start <= video_start +
                    global_inspection_boundary_tolerance_seconds and
                    end >= video_end -
                    global_inspection_boundary_tolerance_seconds)
                call_is_global = call_is_global or is_global_range
                parsed_ranges.append({
                    "requested_range": [str(value[0]), str(value[1])],
                    "start_seconds": start, "end_seconds": end,
                    "global_boundary_match": is_global_range,
                    "overlapping_segments": _ordered_subset(
                        segment_order, set(range_segments)),
                })
            ordered_call_segments = _ordered_subset(segment_order, call_segments)
            if call_is_global:
                global_inspected.update(call_segments)
                classification = "global"
            else:
                localized_inspected.update(call_segments)
                classification = "localized"
            call_rows.append({
                "tool_event_index": event_index,
                "requested_ranges": parsed_ranges,
                "classification": classification,
                "segment_ids": ordered_call_segments,
            })

        # A call is classified atomically. If malformed instrumentation places
        # one segment in both categories, global exclusion wins fail-closed.
        localized_inspected.difference_update(global_inspected)
        candidate = assistant_cited | localized_inspected
        assistant_ordered = _ordered_subset(segment_order, assistant_cited)
        localized_ordered = _ordered_subset(segment_order, localized_inspected)
        global_ordered = _ordered_subset(segment_order, global_inspected)
        candidate_ordered = _ordered_subset(segment_order, candidate)
        rows.append({
            "qa_id": qa_id,
            **provenance,
            "assistant_timestamp_cited_segments": assistant_ordered,
            "localized_frame_inspected_segments": localized_ordered,
            "global_frame_inspected_segments": global_ordered,
            "intervention_candidate_segments": candidate_ordered,
            "frame_inspect_calls": call_rows,
            "selection_reason": (
                "assistant_timestamp_or_localized_frame_inspection"
                if candidate_ordered else "no_localized_trajectory_evidence"),
            "skip": not bool(candidate_ordered),
            "skip_reason": (None if candidate_ordered else
                            "no_localized_evidence"),
            "no_localized_evidence": not bool(candidate_ordered),
            "context_ineligible": False,
            "token_preflight": None,
        })
    return tuple(rows)


def build_prompt_delta_proposal_request(
    baseline_video_manifest_path: str,
    *, parent_meta_prompt_id: str, parent_meta_prompt_text: str,
    global_inspection_boundary_tolerance_seconds: float,
) -> tuple[str, Mapping[str, Any], str]:
    if not parent_meta_prompt_id:
        raise ValueError("parent_meta_prompt_id is required")
    if not isinstance(parent_meta_prompt_text, str) or not \
            parent_meta_prompt_text.strip():
        raise ValueError("parent_meta_prompt_text is required")
    baseline = _read_json(baseline_video_manifest_path)
    routing = _read_json(baseline["routing_manifest_path"])
    prompts = _read_jsonl(routing["composed_prompts_path"])
    histories = _read_jsonl(baseline["frozen_histories_path"])
    captions = _read_json(baseline["captions_path"])
    prompt_ids = tuple(item["segment_id"] for item in prompts)
    history_ids = tuple(item["segment_id"] for item in histories)
    if prompt_ids != history_ids or len(set(prompt_ids)) != len(prompt_ids):
        raise FreshPromptDeltaError("baseline prompt/history order differs")
    prompt_by_id = {row["segment_id"]: row for row in prompts}
    history_by_id = {row["segment_id"]: row for row in histories}

    qa_records = []
    ordered_cited_ids = []
    selection_by_qa = {
        row["qa_id"]: row
        for row in _qa_segment_selection_records(
            baseline_video_manifest_path,
            global_inspection_boundary_tolerance_seconds=
                global_inspection_boundary_tolerance_seconds)
    }
    for qa in _read_jsonl(baseline["baseline_qas_path"]):
        qa_id = qa.get("question_id")
        if not isinstance(qa_id, str) or not qa_id or qa_id not in selection_by_qa:
            raise FreshPromptDeltaError("baseline QA identity is invalid")
        cited = selection_by_qa[qa_id]["intervention_candidate_segments"]
        question = qa.get("question")
        choices = qa.get("options") or []
        prediction = qa.get("prediction")
        correctness = qa.get("is_correct")
        if not isinstance(question, str) or not question or \
                not isinstance(choices, list) or any(
                    not isinstance(item, str) for item in choices) or \
                not isinstance(prediction, str) or not prediction or \
                not isinstance(correctness, bool):
            raise FreshPromptDeltaError(
                f"baseline QA {qa_id!r} metadata/outcome is invalid")
        unknown = set(cited) - set(prompt_by_id)
        if unknown:
            raise FreshPromptDeltaError(
                f"baseline QA {qa_id!r} references unknown segments {sorted(unknown)!r}")
        if not cited:
            continue
        ordered_cited_ids.extend(cited)
        refs = qa.get("reference_sets") or {}
        if not isinstance(refs, Mapping):
            raise FreshPromptDeltaError(
                f"baseline QA {qa_id!r} reference sets are invalid")
        normalized_refs = {}
        for key, values in refs.items():
            if not isinstance(values, (list, tuple)) or any(
                    not isinstance(item, str) or not item for item in values):
                raise FreshPromptDeltaError(
                    f"baseline QA {qa_id!r} reference set {key!r} is invalid")
            normalized_refs[str(key)] = tuple(values)
        resolved = _resolve_trajectory(
            qa.get("trajectory_path"), normalized_refs)
        compact = _compact_trajectory(resolved, qa.get("trajectory_path"))
        trajectory, _audit = _model_trajectory_projection(compact)
        qa_records.append({
            "qa_id": qa_id,
            "question": question,
            "answer_choices": choices,
            "baseline_prediction": prediction,
            "baseline_correct": correctness,
            "intervention_candidate_segment_refs": list(cited),
            "trajectory": trajectory,
        })

    included_ids = tuple(dict.fromkeys(ordered_cited_ids))
    history_items = []
    history_item_by_ref: dict[str, Any] = {}
    history_snapshots = []
    history_snapshot_by_ref: dict[str, Any] = {}
    segment_catalog = []
    original_history_by_segment = {}
    for segment_id in included_ids:
        caption = captions.get(segment_id)
        if not isinstance(caption, Mapping) or not isinstance(
                caption.get("caption"), str) or not caption["caption"]:
            raise FreshPromptDeltaError(
                f"referenced baseline caption unavailable: {segment_id}")
        history_ref, history_row = _normalize_history_snapshot(
            history_by_id[segment_id], segment_id=segment_id,
            item_rows=history_items, item_by_ref=history_item_by_ref)
        previous = history_snapshot_by_ref.get(history_ref)
        if previous is not None and dumps_canonical(previous) != \
                dumps_canonical(history_row):
            raise FreshPromptDeltaError(
                f"history snapshot hash collision for {history_ref!r}")
        if previous is None:
            history_snapshot_by_ref[history_ref] = history_row
            history_snapshots.append(history_row)
        prompt_text = prompt_by_id[segment_id].get("prompt_text")
        if not isinstance(prompt_text, str) or not prompt_text:
            raise FreshPromptDeltaError(
                f"referenced baseline prompt unavailable: {segment_id}")
        segment_catalog.append({
            "segment_id": segment_id,
            "history_ref": history_ref,
            "base_generated_prompt": prompt_text,
            "baseline_caption": caption["caption"],
        })
        original_history_by_segment[segment_id] = history_by_id[segment_id]

    payload = {
        "schema_version": PROMPT_DELTA_PROPOSAL_REQUEST_SCHEMA_VERSION,
        "representation": {
            "version": PROMPT_DELTA_PROPOSAL_REPRESENTATION_VERSION,
            "evidence_scope": PROMPT_DELTA_PROPOSAL_EVIDENCE_SCOPE,
            "split_policy": PROMPT_DELTA_PROPOSAL_SPLIT_POLICY,
            "frame_inspection_classification_policy":
                PROMPT_DELTA_FRAME_INSPECTION_CLASSIFICATION_POLICY,
            "global_inspection_boundary_tolerance_seconds":
                global_inspection_boundary_tolerance_seconds,
        },
        "request_scope": {"mode": "all_qas", "qa_ids": [
            row["qa_id"] for row in qa_records]},
        "video_id": baseline["video_id"],
        "parent_meta_prompt_id": parent_meta_prompt_id,
        "parent_meta_prompt_text": parent_meta_prompt_text,
        "history_catalog": {
            "history_items": history_items,
            "snapshots": history_snapshots,
        },
        "segment_catalog": segment_catalog,
        "qa_records": qa_records,
    }
    for segment_id, original in original_history_by_segment.items():
        if dumps_canonical(reconstruct_prompt_delta_history_snapshot(
                payload, segment_id=segment_id)) != dumps_canonical(original):
            raise FreshPromptDeltaError(
                f"normalized history reconstruction failed for {segment_id!r}")
    canonical = _plain(payload)
    return (PROMPT_DELTA_PROPOSAL_SYSTEM_INSTRUCTION, canonical,
            sha256_json(canonical))


def build_prompt_delta_single_qa_request(
    normalized_payload: Mapping[str, Any], *, qa_id: str,
) -> tuple[Mapping[str, Any], str]:
    qas = normalized_payload.get("qa_records")
    if not isinstance(qas, list):
        raise FreshPromptDeltaError("normalized QA records are invalid")
    matching = [row for row in qas if row.get("qa_id") == qa_id]
    if len(matching) != 1:
        raise FreshPromptDeltaError(f"unknown or duplicate QA ID {qa_id!r}")
    qa = matching[0]
    segment_refs = set(qa["intervention_candidate_segment_refs"])
    segments = [row for row in normalized_payload["segment_catalog"]
                if row["segment_id"] in segment_refs]
    if {row["segment_id"] for row in segments} != segment_refs:
        raise FreshPromptDeltaError(
            f"single-QA segment projection changed scope for {qa_id!r}")
    history_refs = {row["history_ref"] for row in segments}
    snapshots = [row for row in normalized_payload["history_catalog"]["snapshots"]
                 if row["history_ref"] in history_refs]
    item_refs = {
        ref for row in snapshots
        for ref in row.get("ordered_history_item_refs", ())
    }
    items = [row for row in normalized_payload["history_catalog"]["history_items"]
             if row["history_item_ref"] in item_refs]
    payload = {
        **{key: _plain(value) for key, value in normalized_payload.items()
           if key not in ("request_scope", "history_catalog",
                          "segment_catalog", "qa_records")},
        "request_scope": {"mode": "single_qa", "qa_ids": [qa_id]},
        "history_catalog": {"history_items": items, "snapshots": snapshots},
        "segment_catalog": segments,
        "qa_records": [_plain(qa)],
    }
    canonical = _plain(payload)
    return canonical, sha256_json(canonical)


class LLMPromptDeltaProposer:
    """Generate candidates through isolated, deterministic per-QA requests."""

    def __init__(
        self, *, backend: PromptDeltaProposalBackend,
        maximum_deltas_per_qa: int,
        selection_policy: str,
        global_inspection_boundary_tolerance_seconds: float,
    ) -> None:
        if not callable(getattr(backend, "preflight", None)) or not callable(
                getattr(backend, "generate", None)):
            raise TypeError("backend must provide preflight and generate")
        if maximum_deltas_per_qa not in (1, 2):
            raise ValueError("maximum_deltas_per_qa must be 1 or 2")
        if selection_policy != PROMPT_DELTA_SEGMENT_SELECTION_POLICY:
            raise ValueError("unsupported prompt-delta selection policy")
        if global_inspection_boundary_tolerance_seconds < 0:
            raise ValueError(
                "global inspection boundary tolerance must be non-negative")
        self.backend = backend
        self.maximum_deltas_per_qa = maximum_deltas_per_qa
        self.selection_policy = selection_policy
        self.global_inspection_boundary_tolerance_seconds = (
            global_inspection_boundary_tolerance_seconds)

    @staticmethod
    def _parse_response(
        raw: str, *, allowed_qa_ids: tuple[str, ...],
        minimum_proposals: int, maximum_proposals: int,
    ) -> tuple[dict[str, Any], ...]:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FreshPromptDeltaError(
                "delta response is not strict JSON") from exc
        if not isinstance(value, dict) or set(value) != {"proposals"} or \
                not isinstance(value["proposals"], list):
            raise FreshPromptDeltaError(
                "delta response must contain only proposals array")
        if not minimum_proposals <= len(value["proposals"]) <= maximum_proposals:
            raise FreshPromptDeltaError(
                "delta response violates configured proposal count")
        output = []
        for item in value["proposals"]:
            expected = {"instruction", "source_qa_ids", "proposer_diagnosis"}
            if not isinstance(item, dict) or set(item) != expected:
                raise FreshPromptDeltaError("delta proposal fields are invalid")
            instruction = item["instruction"]
            diagnosis = item["proposer_diagnosis"]
            source_ids = item["source_qa_ids"]
            if not isinstance(instruction, str) or not instruction.strip() or \
                    not isinstance(diagnosis, str) or not diagnosis.strip() or \
                    not isinstance(source_ids, list) or not source_ids or any(
                        not isinstance(value, str) or not value
                        for value in source_ids) or \
                    len(set(source_ids)) != len(source_ids) or \
                    set(source_ids) - set(allowed_qa_ids):
                raise FreshPromptDeltaError("delta proposal content is invalid")
            output.append({
                "instruction": instruction,
                "source_qa_ids": tuple(source_ids),
                "proposer_diagnosis": diagnosis,
            })
        return tuple(output)

    def propose(
        self, baseline_video_manifest_path: str,
        *, parent_meta_prompt_id: str, parent_meta_prompt_text: str,
        output_directory: str,
    ) -> tuple[PromptDeltaExecutionPlan, ...]:
        system, payload, request_hash = build_prompt_delta_proposal_request(
            baseline_video_manifest_path,
            parent_meta_prompt_id=parent_meta_prompt_id,
            parent_meta_prompt_text=parent_meta_prompt_text,
            global_inspection_boundary_tolerance_seconds=
                self.global_inspection_boundary_tolerance_seconds)
        output = os.path.abspath(output_directory)
        qa_ids = tuple(row["qa_id"] for row in payload["qa_records"])
        selection_records = list(_plain(_qa_segment_selection_records(
            baseline_video_manifest_path,
            global_inspection_boundary_tolerance_seconds=
                self.global_inspection_boundary_tolerance_seconds)))
        classification_input = {
            "policy": PROMPT_DELTA_FRAME_INSPECTION_CLASSIFICATION_POLICY,
            "global_inspection_boundary_tolerance_seconds":
                self.global_inspection_boundary_tolerance_seconds,
            "qa_classifications": [{
                "qa_id": row["qa_id"],
                "assistant_timestamp_cited_segments":
                    row["assistant_timestamp_cited_segments"],
                "localized_frame_inspected_segments":
                    row["localized_frame_inspected_segments"],
                "global_frame_inspected_segments":
                    row["global_frame_inspected_segments"],
                "frame_inspect_calls": row["frame_inspect_calls"],
            } for row in selection_records],
        }
        classification_hash = sha256_json(classification_input)
        classification_hash_by_qa = {
            row["qa_id"]: _source_qa_classification_hash(
                row, tolerance_seconds=
                    self.global_inspection_boundary_tolerance_seconds)
            for row in selection_records
        }
        selection_summary = {
            "schema_version": "prompt_delta_segment_selection_audit_v2",
            "selection_policy": self.selection_policy,
            "frame_inspection_classification_policy":
                PROMPT_DELTA_FRAME_INSPECTION_CLASSIFICATION_POLICY,
            "global_inspection_boundary_tolerance_seconds":
                self.global_inspection_boundary_tolerance_seconds,
            "global_inspection_classification_hash": classification_hash,
            "selection_is_causal_attribution": False,
            "qa_records": selection_records,
            "total_qa_count": len(selection_records),
            "eligible_qa_count": sum(
                not row["no_localized_evidence"] for row in selection_records),
            "qa_without_localized_evidence": sum(
                row["no_localized_evidence"] for row in selection_records),
            "context_ineligible_qa_count": 0,
            "excluded_global_inspection_segment_count": len({
                segment_id for row in selection_records
                for segment_id in row["global_frame_inspected_segments"]}),
            "total_unique_intervention_candidate_segments": len({
                segment_id for row in selection_records
                for segment_id in row["intervention_candidate_segments"]}),
        }
        if not qa_ids:
            _write_once(os.path.join(output, "selection_audit.json"),
                        selection_summary)
            identity = {
                "schema_version": "prompt_delta_proposal_identity_v4",
                "representation_version":
                    PROMPT_DELTA_PROPOSAL_REPRESENTATION_VERSION,
                "evidence_scope": PROMPT_DELTA_PROPOSAL_EVIDENCE_SCOPE,
                "split_policy": PROMPT_DELTA_PROPOSAL_SPLIT_POLICY,
                "split_mode": "no_eligible_proposal_evidence",
                "normalized_full_request_hash": request_hash,
                "request_hashes": [],
                "catalog_hashes": {
                    "history_catalog_hash": sha256_json(
                        payload["history_catalog"]),
                    "segment_catalog_hash": sha256_json(
                        payload["segment_catalog"]),
                    "qa_records_hash": sha256_json(payload["qa_records"]),
                },
                "selection_audit_hash": sha256_json(selection_summary),
                "global_inspection_classification_hash": classification_hash,
                "global_inspection_boundary_tolerance_seconds":
                    self.global_inspection_boundary_tolerance_seconds,
                "provider_settings": _plain(
                    self.backend.configuration_identity),
                "maximum_deltas_per_qa": self.maximum_deltas_per_qa,
                "selection_policy": self.selection_policy,
            }
            identity_hash = sha256_json(identity)
            _write_once(os.path.join(output, "request_manifest.json"), {
                "schema_version": "prompt_delta_proposal_request_manifest_v4",
                "status": "no_eligible_proposal_evidence",
                "request_identity": identity,
                "request_identity_hash": identity_hash,
                "full_request_preflight": None,
                "requests": [],
                "proposer_call_count": 0,
                "truncation_filtering_sampling_or_intra_qa_split": False,
            })
            _write_once(os.path.join(output, "proposal_plans.json"), {
                "schema_version": "prompt_delta_proposal_plans_v4",
                "status": "no_eligible_proposal_evidence",
                "request_hash": request_hash,
                "request_identity_hash": identity_hash,
                "backend_policy_version": self.backend.policy_version,
                "representation_version":
                    PROMPT_DELTA_PROPOSAL_REPRESENTATION_VERSION,
                "split_mode": "no_eligible_proposal_evidence",
                "catalog_hashes": identity["catalog_hashes"],
                "plans": [],
            })
            return ()
        single_requests = []
        single_by_qa = {}
        single_truncations = {}
        for qa_id in qa_ids:
            single, single_hash = build_prompt_delta_single_qa_request(
                payload, qa_id=qa_id)
            preflight = self.backend.preflight(
                system, dumps_canonical(single))
            single_requests.append((qa_id, single, single_hash, preflight))
            single_by_qa[qa_id] = (single, single_hash, preflight)
            single_truncations[qa_id] = None
        for row in selection_records:
            if row["qa_id"] not in single_by_qa:
                continue
            preflight = single_by_qa[row["qa_id"]][2]
            row["token_preflight"] = asdict(preflight)
            row["context_truncation"] = single_truncations[row["qa_id"]]
            if not preflight.fits_context:
                row["skip"] = True
                row["skip_reason"] = "context_ineligible"
                row["context_ineligible"] = True

        eligible_qa_ids = tuple(
            qa_id for qa_id in qa_ids
            if single_by_qa[qa_id][2].fits_context)
        selection_summary["eligible_qa_count"] = len(eligible_qa_ids)
        selection_summary["context_ineligible_qa_count"] = sum(
            row["context_ineligible"] for row in selection_records)
        _write_once(os.path.join(output, "selection_audit.json"),
                    selection_summary)

        requests: list[tuple[Mapping[str, Any], str,
                             PromptDeltaRequestPreflight, int, int]] = []
        split_mode = "isolated_per_qa"
        for qa_id in eligible_qa_ids:
            single, single_hash, preflight = single_by_qa[qa_id]
            requests.append((single, single_hash, preflight, 1,
                             self.maximum_deltas_per_qa))

        context_ineligible = [
            (qa_id, request, digest, preflight)
            for qa_id, request, digest, preflight in single_requests
            if not preflight.fits_context]

        provider_identity = _plain(self.backend.configuration_identity)
        catalog_hashes = {
            "history_catalog_hash": sha256_json(payload["history_catalog"]),
            "segment_catalog_hash": sha256_json(payload["segment_catalog"]),
            "qa_records_hash": sha256_json(payload["qa_records"]),
        }
        request_identity = {
            "schema_version": "prompt_delta_proposal_identity_v4",
            "representation_version":
                PROMPT_DELTA_PROPOSAL_REPRESENTATION_VERSION,
            "evidence_scope": PROMPT_DELTA_PROPOSAL_EVIDENCE_SCOPE,
            "split_policy": PROMPT_DELTA_PROPOSAL_SPLIT_POLICY,
            "split_mode": split_mode,
            "normalized_full_request_hash": request_hash,
            "request_hashes": [item[1] for item in requests],
            "catalog_hashes": catalog_hashes,
            "selection_audit_hash": sha256_json(selection_summary),
            "global_inspection_classification_hash": classification_hash,
            "global_inspection_boundary_tolerance_seconds":
                self.global_inspection_boundary_tolerance_seconds,
            "provider_settings": provider_identity,
            "maximum_deltas_per_qa": self.maximum_deltas_per_qa,
            "selection_policy": self.selection_policy,
        }
        request_identity_hash = sha256_json(request_identity)
        request_rows = []
        for index, (request, digest, preflight, minimum, maximum) in enumerate(
                requests, 1):
            qa_scope = request["request_scope"]["qa_ids"]
            path = _write_once(os.path.join(
                output, f"request_{index:03d}.json"), request)
            request_rows.append({
                "request_index": index,
                "request_path": path,
                "request_hash": digest,
                "qa_ids": qa_scope,
                "preflight": asdict(preflight),
                "size_breakdown": _request_size_breakdown(request),
                "minimum_proposals": minimum,
                "maximum_proposals": maximum,
                "context_truncation": (
                    single_truncations.get(qa_scope[0])
                    if len(qa_scope) == 1 else None),
            })
        _write_once(os.path.join(output, "request_manifest.json"), {
            "schema_version": "prompt_delta_proposal_request_manifest_v4",
            "status": ("ready" if requests else
                       "no_eligible_proposal_evidence"),
            "request_identity": request_identity,
            "request_identity_hash": request_identity_hash,
            "full_request_preflight": None,
            "requests": request_rows,
            "planned_proposer_call_count": len(request_rows),
            "context_ineligible_qa_ids": [
                qa_id for qa_id, *_rest in context_ineligible],
            "truncation_filtering_sampling_or_intra_qa_split": any(
                value is not None for value in single_truncations.values()),
        })

        if context_ineligible:
            ineligible = {
                "schema_version": "prompt_delta_context_ineligible_v1",
                "request_identity_hash": request_identity_hash,
                "split_mode": split_mode,
                "offending_requests": [{
                    "qa_ids": [qa_id],
                    "request_hash": digest,
                    "preflight": asdict(preflight),
                    "size_breakdown": _request_size_breakdown(request),
                    "offending_segment_ids": [
                        item["segment_id"]
                        for item in request["segment_catalog"]],
                    "offending_history_refs": [
                        item["history_ref"] for item in
                        request["history_catalog"]["snapshots"]],
                    "skip_reason": "context_ineligible",
                } for qa_id, request, digest, preflight
                    in context_ineligible],
                "truncation_performed": False,
                "provider_call_performed_for_ineligible_qa": False,
            }
            _write_once(os.path.join(output, "context_ineligible.json"),
                        ineligible)

        if not requests:
            _write_once(os.path.join(output, "proposal_plans.json"), {
                "schema_version": "prompt_delta_proposal_plans_v4",
                "status": "no_eligible_proposal_evidence",
                "request_hash": request_hash,
                "request_identity_hash": request_identity_hash,
                "backend_policy_version": self.backend.policy_version,
                "representation_version":
                    PROMPT_DELTA_PROPOSAL_REPRESENTATION_VERSION,
                "split_mode": split_mode,
                "catalog_hashes": catalog_hashes,
                "plans": [],
            })
            return ()

        plan_path = os.path.join(output, "proposal_plans.json")
        if os.path.isfile(plan_path):
            saved = _read_json(plan_path)
            if saved.get("request_identity_hash") != request_identity_hash:
                raise FreshPromptDeltaError(
                    "saved prompt-delta proposal has different inputs")
            return tuple(PromptDeltaExecutionPlan(
                prompt_delta=prompt_delta_from_json(item["prompt_delta"]),
                selected_segment_ids=tuple(item["selected_segment_ids"]),
                selection_policy=item["selection_policy"],
                frame_inspection_classification_hash=item[
                    "frame_inspection_classification_hash"],
                global_inspection_boundary_tolerance_seconds=float(item[
                    "global_inspection_boundary_tolerance_seconds"]))
                for item in saved["plans"])

        qa_by_id = {item["qa_id"]: item for item in payload["qa_records"]}
        segment_ids = {item["segment_id"] for item in payload["segment_catalog"]}
        candidate_rows = []
        for index, (request, digest, _preflight, minimum, maximum) in enumerate(
                requests, 1):
            raw_path = os.path.join(output, f"raw_response_{index:03d}.json")
            if os.path.isfile(raw_path):
                saved_raw = _read_json(raw_path)
                if saved_raw.get("request_hash") != digest or not isinstance(
                        saved_raw.get("raw_response"), str):
                    raise FreshPromptDeltaError(
                        "saved prompt-delta raw response has different inputs")
                raw = saved_raw["raw_response"]
            else:
                try:
                    raw = self.backend.generate(
                        system, dumps_canonical(request),
                        minimum_proposals=minimum, maximum_proposals=maximum)
                except Exception as exc:
                    _write_once(os.path.join(
                        output, f"provider_error_{index:03d}.json"), {
                            "request_hash": digest,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        })
                    raise
                _write_once(raw_path, {
                    "request_hash": digest, "raw_response": raw})
            allowed = tuple(request["request_scope"]["qa_ids"])
            parsed = self._parse_response(
                raw, allowed_qa_ids=allowed,
                minimum_proposals=minimum, maximum_proposals=maximum)
            if any(item["source_qa_ids"] != allowed for item in parsed):
                raise FreshPromptDeltaError(
                    "per-QA proposal must retain exactly its source QA ID")
            candidate_rows.extend(
                (item, digest, index, proposal_index)
                for proposal_index, item in enumerate(parsed, 1))

        plans = []
        for item, candidate_request_hash, request_index, proposal_index in \
                candidate_rows:
            instruction = item["instruction"]
            diagnosis = item["proposer_diagnosis"]
            source_ids = item["source_qa_ids"]
            selected = tuple(dict.fromkeys(
                segment_id for qa_id in source_ids
                for segment_id in qa_by_id[qa_id][
                    "intervention_candidate_segment_refs"]))
            if not selected or set(selected) - segment_ids:
                raise FreshPromptDeltaError(
                    "source QA localized segment scope is empty or invalid")
            delta_id = "prompt_delta_" + sha256_json({
                "video_id": payload["video_id"],
                "parent_meta_prompt_id": payload["parent_meta_prompt_id"],
                "instruction": instruction, "source_qa_ids": source_ids,
                "diagnosis": diagnosis,
                "policy_version": self.backend.policy_version,
                "candidate_request_hash": candidate_request_hash,
                "request_index": request_index,
                "proposal_index": proposal_index,
            })[:20]
            plans.append(PromptDeltaExecutionPlan(
                prompt_delta=PromptDelta(
                    delta_id=delta_id, instruction=instruction,
                    source_qa_ids=source_ids,
                    proposer_diagnosis=diagnosis),
                selected_segment_ids=selected,
                selection_policy=self.selection_policy,
                frame_inspection_classification_hash=
                    classification_hash_by_qa[source_ids[0]],
                global_inspection_boundary_tolerance_seconds=
                    self.global_inspection_boundary_tolerance_seconds))
        _write_once(plan_path, {
            "schema_version": "prompt_delta_proposal_plans_v4",
            "status": "completed",
            "request_hash": request_hash,
            "request_identity_hash": request_identity_hash,
            "backend_policy_version": self.backend.policy_version,
            "representation_version":
                PROMPT_DELTA_PROPOSAL_REPRESENTATION_VERSION,
            "split_mode": split_mode,
            "catalog_hashes": catalog_hashes,
            "plans": plans,
        })
        return tuple(plans)


class PromptDeltaInterventionRunner:
    """Selective frozen-history recaption and all-sibling-QA episode builder."""

    def __init__(
        self, *, segment_captioner: Any,
        sample_loader: Callable[[int], Mapping[str, Any]],
        merge_prompt: str, caption_cache_root: str,
        caption_cache_manifest_path: str,
        composition_separator: str,
        qa_fn: Callable[..., Any] = run_dvd_qa,
        clip_index_fn: Callable[..., list[tuple[str, dict]]] = build_clip_index,
        mixed_view_builder: MixedViewBuilder | None = None,
        dvd_max_iterations: int, gpu: str | None,
        scaffold_contract: Any,
    ) -> None:
        if not composition_separator:
            raise ValueError("composition_separator must be explicit")
        self.segment_captioner = segment_captioner
        self.sample_loader = sample_loader
        self.merge_prompt = merge_prompt
        self.caption_cache_root = caption_cache_root
        self.caption_cache_manifest_path = caption_cache_manifest_path
        self.composition_separator = composition_separator
        self.qa_fn = qa_fn
        self.clip_index_fn = clip_index_fn
        self.mixed = mixed_view_builder or MixedViewBuilder()
        self.dvd_max_iterations = dvd_max_iterations
        self.gpu = gpu
        self.contract = scaffold_contract

    def run(
        self, *, baseline_video_manifest_path: str,
        parent_meta_prompt_id: str, plan: PromptDeltaExecutionPlan,
        output_directory: str,
    ) -> InterventionEpisode:
        if not parent_meta_prompt_id:
            raise ValueError("parent_meta_prompt_id is required")
        baseline = _read_json(baseline_video_manifest_path)
        selection_records = _qa_segment_selection_records(
            baseline_video_manifest_path,
            global_inspection_boundary_tolerance_seconds=
                plan.global_inspection_boundary_tolerance_seconds)
        scope_by_qa = {
            row["qa_id"]: tuple(row["intervention_candidate_segments"])
            for row in selection_records
        }
        selection_by_qa = {row["qa_id"]: row for row in selection_records}
        source_qa_ids = tuple(plan.prompt_delta.source_qa_ids)
        if len(source_qa_ids) != 1 or source_qa_ids[0] not in scope_by_qa:
            raise FreshPromptDeltaError(
                "prompt-delta source QA provenance is missing")
        classification_hash = _source_qa_classification_hash(
            selection_by_qa[source_qa_ids[0]], tolerance_seconds=
                plan.global_inspection_boundary_tolerance_seconds)
        if classification_hash != plan.frame_inspection_classification_hash:
            raise FreshPromptDeltaError(
                "source-QA frame-inspection classification identity changed")
        permitted = {
            segment_id for qa_id in source_qa_ids
            for segment_id in scope_by_qa[qa_id]
        }
        if not plan.selected_segment_ids or \
                set(plan.selected_segment_ids) - permitted:
            raise FreshPromptDeltaError(
                "intervention segment is not localized by a source QA")
        if plan.selection_policy != PROMPT_DELTA_SEGMENT_SELECTION_POLICY:
            raise FreshPromptDeltaError(
                "intervention selection policy is incompatible")
        output = os.path.abspath(output_directory)
        episode_path = os.path.join(output, "intervention_episode.json")
        identity = sha256_text(dumps_canonical({
            "baseline": baseline, "plan": plan,
            "parent_meta_prompt_id": parent_meta_prompt_id,
            "composition_separator": self.composition_separator,
            "captioner": self.segment_captioner.configuration_identity,
            "downstream_qa": dvd_qa_execution_identity(
                self.dvd_max_iterations),
        }))
        if os.path.isfile(episode_path):
            envelope = _read_json(episode_path)
            if envelope.get("input_identity") != identity:
                raise FreshPromptDeltaError(
                    "saved intervention episode has different inputs")
            from surrogate_rollout.optimization.schemas import (
                intervention_episode_from_json,
            )
            return intervention_episode_from_json(envelope["episode"])
        routing = _read_json(baseline["routing_manifest_path"])
        prompts = {item.segment_id: item for item in read_composed_prompts(
            routing["composed_prompts_path"])}
        histories = {item["segment_id"]: item for item in _read_jsonl(
            baseline["frozen_histories_path"])}
        baseline_captions = _read_json(baseline["captions_path"])
        baseline_qas = _read_jsonl(baseline["baseline_qas_path"])
        first = dict(self.sample_loader(int(baseline_qas[0]["provider_index"])))
        clip_by_id = dict(self.clip_index_fn(first, baseline["video_id"]))
        selected = plan.selected_segment_ids
        missing = set(selected) - (set(prompts) & set(histories) & set(clip_by_id))
        if missing:
            raise FreshPromptDeltaError(
                f"selected segments lack frozen baseline state: {sorted(missing)}")
        parsed = {}
        clip_records = []
        for segment_id in selected:
            incumbent = prompts[segment_id]
            text = (incumbent.prompt_text + self.composition_separator +
                    plan.prompt_delta.instruction)
            errors = validate_composed_text(text, self.contract)
            if errors:
                raise FreshPromptDeltaError(
                    f"prompt delta violates caption contract: {errors}")
            composed = replace(
                incumbent, prompt_text=text, prompt_hash=sha256_text(text),
                validation_errors=(), is_valid=True)
            intervention_identity = sha256_json({
                "baseline_identity": identity, "delta_id":
                plan.prompt_delta.delta_id, "segment_id": segment_id,
                "history_hash": histories[segment_id]["history_hash"],
                "composed_prompt_hash": composed.prompt_hash,
            })
            result = self.segment_captioner.caption(
                sample=first, video_id=baseline["video_id"],
                segment_id=segment_id, clip_info=clip_by_id[segment_id],
                composed_prompt=composed, history_snapshot=histories[segment_id],
                merge_prompt=self.merge_prompt,
                cache_root=self.caption_cache_root,
                cache_manifest_path=self.caption_cache_manifest_path,
                intervention_identity_hash=intervention_identity)
            value = dict(result.parsed)
            if not value.get("clip_description"):
                raise FreshPromptDeltaError(
                    f"intervention caption unavailable: {segment_id}")
            parsed[segment_id] = value
            start, end = (float(item) for item in segment_id.split("_", 1))
            clip_records.append(InterventionClipRecord(
                segment_id=segment_id, time_range=[start, end],
                history_snapshot=histories[segment_id],
                base_prompt=incumbent.prompt_text,
                prompt_delta=plan.prompt_delta,
                baseline_caption=baseline_captions[segment_id]["caption"],
                intervention_caption=value["clip_description"]))
        mixed = self.mixed.build(
            baseline_captions_path=baseline["captions_path"],
            candidate_captions=parsed, selected_clip_ids=set(selected),
            work_root=os.path.join(output, "mixed_view"),
            video_id=baseline["video_id"],
            view_name=plan.prompt_delta.delta_id)
        if set(mixed.replaced_clip_ids) != set(selected):
            raise FreshPromptDeltaError("mixed view did not replace every clip")
        mixed_view_identity = {
            "delta_id": plan.prompt_delta.delta_id,
            "captions_hash": mixed.captions_hash,
            "selected_segment_ids": list(selected),
            "replaced_segment_ids": list(mixed.replaced_clip_ids),
        }
        outcomes = []
        qa_records = []
        for baseline_qa in baseline_qas:
            qa_id = baseline_qa["question_id"]
            sample = dict(self.sample_loader(int(baseline_qa["provider_index"])))
            qa_dir = os.path.join(output, "qa", qa_id.replace("/", "-"))
            result = self.qa_fn(
                captions_path=mixed.captions_path, sample=sample,
                run_dir=qa_dir, question_id=qa_id,
                database_path=mixed.database_path,
                max_iterations=self.dvd_max_iterations, gpu=self.gpu)
            if result.errors or result.prediction is None:
                raise FreshPromptDeltaError(
                    f"intervention QA execution failed: {qa_id}: {result.errors}")
            candidate_correct = float(result.score) > 0.0
            trajectory_path = os.path.join(qa_dir, "trajectory.jsonl")
            outcomes.append(QAInterventionOutcome(
                qa_id=qa_id,
                is_source_qa=qa_id in plan.prompt_delta.source_qa_ids,
                baseline_answer=baseline_qa.get("prediction"),
                intervention_answer=result.prediction,
                baseline_correct=bool(baseline_qa["is_correct"]),
                intervention_correct=candidate_correct,
                baseline_trajectory_ref=baseline_qa.get("trajectory_path"),
                intervention_trajectory_ref=trajectory_path))
            qa_records.append({
                "qa_id": qa_id,
                "intervention_answer": result.prediction,
                "baseline_correct": bool(baseline_qa["is_correct"]),
                "intervention_correct": candidate_correct,
                "transition": correctness_transition(
                    bool(baseline_qa["is_correct"]), candidate_correct),
                "intervention_result_path": os.path.join(qa_dir, "result.json"),
                "intervention_trajectory_path": trajectory_path,
            })
        run_ref = _write_once(os.path.join(output, "intervention_manifest.json"), {
            "schema_version": "fresh_prompt_delta_intervention_v1",
            "status": "completed", "input_identity": identity,
            "video_id": baseline["video_id"], "prompt_delta": plan.prompt_delta,
            "selected_segment_ids": selected,
            "selection_policy": plan.selection_policy,
            "source_qa_intervention_candidate_segments": {
                qa_id: scope_by_qa[qa_id] for qa_id in source_qa_ids},
            "frame_inspection_classification_hash": classification_hash,
            "global_inspection_boundary_tolerance_seconds":
                plan.global_inspection_boundary_tolerance_seconds,
            "selection_is_causal_attribution": False,
            "downstream_qa_execution_identity":
                dvd_qa_execution_identity(self.dvd_max_iterations),
            "mixed_captions_path": mixed.captions_path,
            "mixed_view_identity": mixed_view_identity,
            "qa_records": qa_records,
            "legacy_property_codebook_or_router_used": False,
        })
        episode_id = "fresh_episode_" + sha256_json({
            "video_id": baseline["video_id"],
            "delta_id": plan.prompt_delta.delta_id,
            "baseline_run_ref": os.path.abspath(baseline_video_manifest_path),
            "intervention_run_ref": run_ref,
        })[:20]
        episode = InterventionEpisode(
            episode_id=episode_id, video_id=baseline["video_id"],
            parent_meta_prompt_id=parent_meta_prompt_id,
            prompt_delta=plan.prompt_delta, clips=tuple(clip_records),
            qa_outcomes=tuple(outcomes),
            baseline_run_ref=os.path.abspath(baseline_video_manifest_path),
            intervention_run_ref=run_ref,
            mixed_view_identity=mixed_view_identity)
        _write_once(episode_path, {
            "schema_version": "fresh_intervention_episode_envelope_v1",
            "input_identity": identity, "episode": episode})
        return episode
