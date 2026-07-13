"""Extract referenced segments from a DVD trajectory + tool sidecar events.

Machine-readable evidence, in CLAUDE.md §12 priority order:

1. `frame_inspect_tool` args (`time_ranges_hhmmss`) — explicit, direct.
2. Vector-DB hits captured by the instrumentation sidecar for
   `clip_search_tool` (captions returned verbatim to the orchestrator) and
   `global_browse_tool` (captions consumed by the tool's internal LLM only).
3. HH:MM:SS timestamps in assistant messages (explicit citations in prose).

Set semantics (all DVD clip keys "{start}_{end}"):
- retrieved_segments:        every vector-DB hit from either retrieval tool.
- returned_segments:         clip_search hits — their caption text entered the
                             orchestrator context verbatim.
- frame_inspected_segments:  clips overlapping frame_inspect time ranges.
- explicitly_cited_segments: frame_inspected + clips whose time range the
                             assistant cited in message text.
- consumed_segments:         segments whose content demonstrably entered some
                             LLM context: returned ∪ frame_inspected ∪
                             global_browse hits (consumed by the sub-LLM).

The final prose answer is deliberately NOT used as evidence when tool metadata
exists; assistant-text timestamps are the lowest-priority source and land only
in explicitly_cited_segments.
"""

from __future__ import annotations

import re

from surrogate_rollout.schemas import ReferenceSets

_HHMMSS = re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})\b")


def hhmmss_to_seconds(text: str) -> float:
    text = text.split(".")[0]
    parts = text.split(":")
    if len(parts) == 2:
        parts = ["0", *parts]
    h, m, s = (int(p) for p in parts)
    return h * 3600 + m * 60 + s


def clips_for_range(start_s: float, end_s: float, clip_keys: list[str]) -> list[str]:
    """Clip keys overlapping [start_s, end_s]; point queries use start==end."""
    out = []
    for k in clip_keys:
        a, b = (float(x) for x in k.split("_"))
        if a <= end_s and b > start_s:
            out.append(k)
        elif a <= start_s < b:  # point exactly at clip start
            out.append(k)
    return out


def clip_for_hit(hit: dict, clip_keys: set[str]) -> str | None:
    """Map a vector-DB hit (exact clip boundaries) back to its clip key."""
    s, e = hit.get("time_start_secs"), hit.get("time_end_secs")
    if s is None or e is None:
        return None
    key = f"{int(s)}_{int(e)}"
    return key if key in clip_keys else None


def extract_references(
    messages: list[dict],
    tool_events: list[dict],
    all_clip_keys: list[str] | set[str],
) -> ReferenceSets:
    clip_list = sorted(all_clip_keys, key=lambda k: float(k.split("_")[0]))
    clip_set = set(clip_list)
    refs = ReferenceSets()

    def note(segment: str, target: str, reason: str, **extra) -> None:
        getattr(refs, target).add(segment)
        refs.evidence.append({"segment": segment, "set": target,
                              "reason": reason, **extra})

    # --- priority 1 + 2: tool events (machine-readable) -------------------- #
    for i, ev in enumerate(tool_events):
        tool = ev.get("tool")
        if tool == "frame_inspect_tool":
            for rng in ev.get("args", {}).get("time_ranges_hhmmss") or []:
                try:
                    s = hhmmss_to_seconds(str(rng[0]))
                    e = hhmmss_to_seconds(str(rng[1]))
                except (ValueError, IndexError):
                    continue
                for k in clips_for_range(s, e, clip_list):
                    note(k, "frame_inspected_segments",
                         "frame_inspect_time_range", event_index=i, range=list(rng))
        elif tool in ("clip_search_tool", "global_browse_tool"):
            for hit in ev.get("hits", []):
                k = clip_for_hit(hit, clip_set)
                if k is None:
                    continue
                note(k, "retrieved_segments", f"{tool}_hit", event_index=i)
                if tool == "clip_search_tool":
                    note(k, "returned_segments", "clip_search_returned_to_orchestrator",
                         event_index=i)

    # --- priority 3: timestamps cited in assistant text --------------------- #
    for mi, m in enumerate(messages or []):
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        stamps = [hhmmss_to_seconds(t.group(0)) for t in _HHMMSS.finditer(content)]
        # consecutive pairs in prose usually denote a range; treat each stamp
        # as a point reference (conservative, no range inference from prose)
        for s in stamps:
            for k in clips_for_range(s, s, clip_list):
                note(k, "explicitly_cited_segments", "assistant_text_timestamp",
                     message_index=mi)

    # frame inspection is an explicit citation too
    for k in refs.frame_inspected_segments:
        refs.explicitly_cited_segments.add(k)

    # --- consumed: content demonstrably entered some LLM context ----------- #
    refs.consumed_segments = (
        refs.returned_segments
        | refs.frame_inspected_segments
        | refs.retrieved_segments  # global_browse hits are consumed by its sub-LLM
    )
    return refs
