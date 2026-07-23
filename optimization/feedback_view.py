"""The minimal view of an episode that the feedback generator actually reads.

The lean request grew into a provenance report: registry dumps, hundred-entry
segment-id arrays, per-source overlap sets, Jaccard scores. Those exist to let a
reader audit attribution, and they stay in the compact evidence artifact, but
the feedback generator is asked a different question -- what did this prompt
delta do to the captions, and how did the QA set move -- so it receives only
what that question needs.

Four things go in: the tested delta, every changed caption pair, every QA
outcome, and a one-line-per-tool-event trace of what each run consulted.

Every projection here is structural. The answer text is taken from the field
the tool schema already designates as its result; nothing is summarized,
ranked, filtered by relevance, or passed through a model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

EPISODE_FEEDBACK_VIEW_VERSION = "episode_feedback_view_v1_minimal"

# The fields a tool result uses for its own final answer, most specific first.
# `query_related_event` is the DVD browse tool's answer field; the rest are the
# common provider spellings. This is a fixed schema lookup, not a search.
FINAL_ANSWER_FIELDS = (
    "query_related_event", "final_answer", "answer", "result", "text",
)

_TIMESTAMP_BLOCK = re.compile(
    r"^(?:\d+_\d+|\d{1,2}:\d{2}(?::\d{2})?)\b", re.MULTILINE)


class FeedbackViewError(ValueError):
    """Raised when the stored evidence cannot be projected without guessing."""


def _decoded(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except (ValueError, TypeError):
        return None


def _timestamp_blocks(text: str) -> list[str] | None:
    starts = [match.start() for match in _TIMESTAMP_BLOCK.finditer(text)]
    if len(starts) < 2:
        return None
    bounds = starts + [len(text)]
    blocks = [text[bounds[index]:bounds[index + 1]].strip()
              for index in range(len(starts))]
    blocks = [block for block in blocks if block]
    return blocks if len(blocks) > 1 else None


def extract_returned_answer_text(evidence_items: Sequence[str]) -> str:
    """The tool's own answer, or the evidence verbatim when it has none.

    Order of preference, all structural:

    1. the result's designated answer field, if the decoded object has one --
       the rest of that object (a subject registry, say) is then not the
       answer and is left to the compact artifact;
    2. the timestamp blocks the text already carries, in order;
    3. the evidence exactly as stored.
    """
    if isinstance(evidence_items, str):
        evidence_items = [evidence_items]
    units: list[str] = []
    for item in evidence_items:
        if not isinstance(item, str):
            raise FeedbackViewError("returned evidence must be text")
        decoded = _decoded(item)
        projected: str | None = None
        if isinstance(decoded, Mapping):
            for field in FINAL_ANSWER_FIELDS:
                value = decoded.get(field)
                if isinstance(value, str) and value.strip():
                    projected = value
                    break
                if value is not None and not isinstance(value, str):
                    projected = json.dumps(
                        value, sort_keys=True, ensure_ascii=False,
                        separators=(",", ":"))
                    break
        if projected is None:
            blocks = _timestamp_blocks(item)
            projected = "\n".join(blocks) if blocks else item
        if projected not in units:
            units.append(projected)
    return "\n\n".join(units)


def _tools(
    calls: Sequence[Mapping[str, Any]], trajectory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """One line per compact tool event, answered from the stored evidence.

    The compact record split its evidence into structural units, so the answer
    is projected from the event's original `returned_evidence` instead; the
    compact record supplies identity and the already-decided source type.
    """
    events = list(trajectory.get("tool_events") or ())
    tools = []
    for call in calls:
        if not isinstance(call, Mapping):
            raise FeedbackViewError("compact tool call must be an object")
        index = call.get("event_index")
        if not isinstance(index, int) or not 0 <= index < len(events):
            raise FeedbackViewError(
                f"compact tool call points outside the trajectory: {index!r}")
        event = events[index]
        if not isinstance(event, Mapping):
            raise FeedbackViewError("tool event must be an object")
        tools.append({
            "event_index": index,
            "tool": call.get("tool"),
            "query": call.get("query"),
            "returned_answer_text": extract_returned_answer_text(
                event.get("returned_evidence") or ()),
            # Carried through from the compact evidence, never re-derived.
            "source_type": call.get("evidence_source"),
        })
    return tools


def build_feedback_view(
    episode: Any,
    qa_records: Sequence[Mapping[str, Any]],
    compact_evidence_by_qa: Mapping[str, Mapping[str, Any]],
    trajectories_by_qa: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """The whole user payload for one feedback call.

    `episode` supplies the delta and the caption pairs, `qa_records` the stored
    QA outcomes, `compact_evidence_by_qa` the already-classified tool events,
    and `trajectories_by_qa` the stored evidence those events returned. Nothing
    is recomputed from any of them, and none of them is modified.
    """
    changed = [clip for clip in episode.clips
               if clip.baseline_caption != clip.intervention_caption]
    view = {
        "feedback_view_version": EPISODE_FEEDBACK_VIEW_VERSION,
        "episode_id": episode.episode_id,
        "prompt_delta": {
            "delta_id": episode.prompt_delta.delta_id,
            "instruction": episode.prompt_delta.instruction,
        },
        "changed_captions": [{
            "segment_id": clip.segment_id,
            "baseline": clip.baseline_caption,
            "intervention": clip.intervention_caption,
        } for clip in changed],
        "qa_outcomes": [{
            "qa_id": qa["qa_id"],
            "is_source_qa": qa["is_source_qa"],
            "question": qa["question"],
            "choices": list(qa["answer_choices"]),
            "gold_answer": qa["gold_answer"],
            "baseline_answer": qa["baseline_answer"],
            "intervention_answer": qa["intervention_answer"],
            "transition": qa["transition"],
        } for qa in qa_records],
        "reasoning_evidence": [{
            "qa_id": qa["qa_id"],
            "baseline": {"tools": _tools(
                compact_evidence_by_qa[qa["qa_id"]]["baseline_tool_calls"],
                trajectories_by_qa[qa["qa_id"]]["baseline"])},
            "intervention": {"tools": _tools(
                compact_evidence_by_qa[qa["qa_id"]]["intervention_tool_calls"],
                trajectories_by_qa[qa["qa_id"]]["intervention"])},
        } for qa in qa_records],
    }
    return view
