"""Structural compaction of the tool evidence sent to the feedback generator.

The feedback payload carries every tool result verbatim, and a single
`global_browse_tool` call can return the whole subject registry -- the same blob
again on the next call. That volume is repetition, not information.

Everything here is deterministic and content-blind. Evidence is deduplicated by
exact equality, repeated events are merged, and long strings are cut only on
boundaries the string already has (JSON structure, or timestamp-led lines).
Nothing is selected, ranked, dropped, or rewritten on the basis of what it says:
no keyword matching, no relevance judgement, no model call. Two payloads that
differ only in wording compact to the same shape.

Attribution needs to distinguish "the changed caption text was in the context"
from "the model looked at raw frames" from "a bulk registry mentioned it", so
each event carries an `evidence_source` decided solely by tool identity and the
event's own recorded metadata.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

COMPACT_TOOL_EVIDENCE_VERSION = "compact_tool_evidence_v1"

# Caption text reaches the orchestrator verbatim only through clip_search;
# global_browse returns a bulk registry/summary; frame_inspect goes to the
# vision model with raw frames and never reads a caption.
EVIDENCE_SOURCE_BY_TOOL = {
    "clip_search_tool": "caption_backed",
    "global_browse_tool": "aggregate_registry",
    "frame_inspect_tool": "frame_backed",
}
PROVENANCE_UNKNOWN = "provenance_unknown"
CAPTION_BACKED = "caption_backed"
FRAME_BACKED = "frame_backed"
AGGREGATE_REGISTRY = "aggregate_registry"

# Every class is always reported, so an empty list means "this provenance
# exposed nothing" rather than "this run had no such tool".
ALL_EVIDENCE_SOURCES = (
    CAPTION_BACKED, FRAME_BACKED, AGGREGATE_REGISTRY, PROVENANCE_UNKNOWN)

# A frame_inspect event counts as caption exposure only when the recorded
# metadata states that caption text was supplied to it. These are exact key
# lookups on the event, never an inspection of the evidence text.
CAPTION_INPUT_METADATA_KEYS = (
    "caption_text_provided",
    "captions_provided",
    "caption_input",
    "caption_text_in_prompt",
)

# A line that opens with a segment key ("120_130") or a clock timestamp is a
# boundary the evidence already has; splitting there invents nothing.
_TIMESTAMP_BLOCK = re.compile(
    r"^(?:\d+_\d+|\d{1,2}:\d{2}(?::\d{2})?)\b", re.MULTILINE)


class CompactToolEvidenceError(ValueError):
    """Raised when a trajectory cannot be compacted without guessing."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _structural_units(text: str) -> list[Any]:
    """Split one evidence string on boundaries it already carries.

    JSON decomposes into its top-level members and stays decoded, so the
    payload does not pay for a second round of string escaping; otherwise
    timestamp-led lines start new blocks. A string with neither structure is
    returned whole.
    """
    if not isinstance(text, str):
        raise CompactToolEvidenceError("returned evidence must be text")
    stripped = text.strip()
    if stripped:
        try:
            parsed = json.loads(stripped)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, Mapping) and parsed:
            return [{key: parsed[key]} for key in parsed]
        if isinstance(parsed, list) and parsed:
            return list(parsed)
    starts = [match.start() for match in _TIMESTAMP_BLOCK.finditer(text)]
    if len(starts) > 1:
        bounds = starts + [len(text)]
        blocks = [text[bounds[index]:bounds[index + 1]].strip()
                  for index in range(len(starts))]
        blocks = [block for block in blocks if block]
        if len(blocks) > 1:
            return blocks
    return [text]


def _evidence_source(event: Mapping[str, Any]) -> str:
    tool = event.get("tool")
    source = EVIDENCE_SOURCE_BY_TOOL.get(tool, PROVENANCE_UNKNOWN)
    if source != FRAME_BACKED:
        return source
    args = event.get("args")
    scopes: tuple[Mapping[str, Any], ...] = (event,)
    if isinstance(args, Mapping):
        scopes = (event, args)
    for scope in scopes:
        for key in CAPTION_INPUT_METADATA_KEYS:
            if bool(scope.get(key)):
                return CAPTION_BACKED
    return FRAME_BACKED


def _query(event: Mapping[str, Any]) -> str | None:
    """The event's own query string, taken from its arguments as recorded."""
    args = event.get("args")
    if not isinstance(args, Mapping):
        return None
    for key in ("query", "event_description", "question"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _segment_ids(event: Mapping[str, Any]) -> list[str]:
    values = event.get("returned_segment_ids") or ()
    if not isinstance(values, (list, tuple)):
        raise CompactToolEvidenceError(
            "returned_segment_ids must be an array")
    return [item for item in values if isinstance(item, str) and item]


def _side_segment_sets(
    trajectory: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    """Segments whose captions were returned, and segments frame-inspected."""
    returned: set[str] = set()
    inspected: set[str] = set()
    for event in trajectory.get("tool_events") or ():
        if not isinstance(event, Mapping):
            continue
        ids = set(_segment_ids(event))
        if event.get("tool") == "frame_inspect_tool":
            inspected |= ids
        else:
            returned |= ids
    return returned, inspected


def _changed_ids_by_source(
    trajectory: Mapping[str, Any], changed: frozenset[str],
) -> dict[str, set[str]]:
    """Changed segments each provenance class returned, per side.

    Uses only the event's recorded segment ids and its `evidence_source`; the
    evidence text is never read.
    """
    by_source: dict[str, set[str]] = {
        source: set() for source in ALL_EVIDENCE_SOURCES}
    for event in trajectory.get("tool_events") or ():
        if not isinstance(event, Mapping):
            continue
        source = _evidence_source(event)
        by_source.setdefault(source, set()).update(
            item for item in _segment_ids(event) if item in changed)
    return by_source


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return round(len(left & right) / len(union), 4)


def build_compact_tool_calls(
    trajectory: Mapping[str, Any], *, changed_segment_ids: Iterable[str],
    seen_units: dict[str, dict[str, Any]] | None = None,
    location: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One record per distinct tool event, in the order the events were made.

    `seen_units` lets one registry survive once per episode instead of once per
    QA side: the same blob comes back from every QA of the same video, and a
    later occurrence is replaced by a pointer to where it already appears.
    """
    changed = frozenset(changed_segment_ids)
    events = trajectory.get("tool_events") or ()
    if not isinstance(events, (list, tuple)):
        raise CompactToolEvidenceError("tool_events must be an array")

    records: list[dict[str, Any]] = []
    by_signature: dict[str, int] = {}
    if seen_units is None:
        seen_units = {}
    where = dict(location or {})
    for event_index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise CompactToolEvidenceError(
                f"tool_events[{event_index}] must be an object")
        raw_evidence = event.get("returned_evidence") or ()
        if not isinstance(raw_evidence, (list, tuple)):
            raise CompactToolEvidenceError(
                f"tool_events[{event_index}].returned_evidence must be an array")
        segment_ids = _segment_ids(event)
        signature = _canonical({
            "tool": event.get("tool"), "query": _query(event),
            "segments": segment_ids, "evidence": list(raw_evidence)})
        if signature in by_signature:
            # The identical call, made again: count it, do not restate it.
            records[by_signature[signature]]["repeated_event_occurrences"] += 1
            continue

        units: list[Any] = []
        local_keys: set[str] = set()
        duplicates = 0
        repeats_earlier: list[dict[str, Any]] = []
        for item in raw_evidence:
            for unit in _structural_units(item):
                key = _canonical(unit)
                if key in local_keys:
                    duplicates += 1
                    continue
                earlier = seen_units.get(key)
                if earlier is not None:
                    duplicates += 1
                    if earlier not in repeats_earlier:
                        repeats_earlier.append(earlier)
                    continue
                seen_units[key] = {**where, "event_index": event_index}
                local_keys.add(key)
                units.append(unit)

        by_signature[signature] = len(records)
        records.append({
            "event_index": event_index,
            "tool": event.get("tool"),
            "query": _query(event),
            "evidence_source": _evidence_source(event),
            "returned_segment_ids": segment_ids,
            "changed_segment_ids_returned": [
                item for item in segment_ids if item in changed],
            "returned_evidence_excerpts": units,
            "returned_evidence_item_count": len(raw_evidence),
            "returned_evidence_character_count": sum(
                len(item) for item in raw_evidence if isinstance(item, str)),
            "duplicate_evidence_units_removed": duplicates,
            "evidence_repeated_from_events": repeats_earlier,
            "repeated_event_occurrences": 1,
        })
    return records


def build_trajectory_delta(
    baseline: Mapping[str, Any], intervention: Mapping[str, Any], *,
    changed_segment_ids: Iterable[str],
) -> dict[str, Any]:
    """How the two runs differ in what they consulted, in counts and sets."""
    changed = frozenset(changed_segment_ids)
    base_returned, base_inspected = _side_segment_sets(baseline)
    live_returned, live_inspected = _side_segment_sets(intervention)
    base_by_source = _changed_ids_by_source(baseline, changed)
    live_by_source = _changed_ids_by_source(intervention, changed)
    return {
        "returned_segments_added": len(live_returned - base_returned),
        "returned_segments_removed": len(base_returned - live_returned),
        "returned_segments_jaccard": _jaccard(base_returned, live_returned),
        "frame_inspected_segments_added": len(live_inspected - base_inspected),
        "frame_inspected_segments_removed": len(
            base_inspected - live_inspected),
        "frame_inspected_segments_jaccard": _jaccard(
            base_inspected, live_inspected),
        "changed_segments_returned_in_baseline": sorted(
            base_returned & changed),
        "changed_segments_returned_in_intervention": sorted(
            live_returned & changed),
        "changed_segments_returned_in_both": sorted(
            base_returned & live_returned & changed),
        "changed_segments_returned_only_in_intervention": sorted(
            (live_returned - base_returned) & changed),
        "changed_segments_frame_inspected_in_both": sorted(
            base_inspected & live_inspected & changed),
        # The same overlap split by provenance. Only caption_backed can mean
        # the changed caption text itself was in the context; frame_backed is
        # source-frame evidence, and aggregate_registry / provenance_unknown
        # carry segment provenance without exposing the caption sentence.
        "changed_segments_returned_in_both_by_source": {
            source: sorted(
                base_by_source.get(source, set()) &
                live_by_source.get(source, set()))
            for source in sorted(
                set(ALL_EVIDENCE_SOURCES) | set(base_by_source) |
                set(live_by_source))
        },
    }


def build_compact_qa_evidence(
    baseline: Mapping[str, Any], intervention: Mapping[str, Any], *,
    changed_segment_ids: Iterable[str],
    seen_units: dict[str, dict[str, Any]] | None = None,
    qa_id: str | None = None,
) -> dict[str, Any]:
    """The compact tool evidence for one QA, both runs plus their difference.

    Pass the same `seen_units` for every QA of an episode to keep repeated
    registries down to one copy across the whole payload.
    """
    changed = frozenset(changed_segment_ids)
    identity = {} if qa_id is None else {"qa_id": qa_id}
    return {
        "compact_tool_evidence_version": COMPACT_TOOL_EVIDENCE_VERSION,
        "baseline_tool_calls": build_compact_tool_calls(
            baseline, changed_segment_ids=changed, seen_units=seen_units,
            location={**identity, "side": "baseline"}),
        "intervention_tool_calls": build_compact_tool_calls(
            intervention, changed_segment_ids=changed, seen_units=seen_units,
            location={**identity, "side": "intervention"}),
        "trajectory_delta": build_trajectory_delta(
            baseline, intervention, changed_segment_ids=changed),
    }


def raw_evidence_character_count(trajectories: Sequence[Mapping[str, Any]]) -> int:
    """Characters of returned evidence before compaction, for reporting."""
    total = 0
    for trajectory in trajectories:
        for event in trajectory.get("tool_events") or ():
            if not isinstance(event, Mapping):
                continue
            for item in event.get("returned_evidence") or ():
                if isinstance(item, str):
                    total += len(item)
    return total
