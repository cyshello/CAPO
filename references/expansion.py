"""Temporal-neighbor expansion over DVD clip keys ("{start}_{end}").

Clip keys are contiguous CLIP_SECS windows produced by dvd_captioning's
_build_clips; the last clip may be shorter. Expansion works on the ordered
list of all clip keys for the video, so radius-1 means "the clip before and
after each selected clip".
"""

from __future__ import annotations


def sort_clip_keys(clip_keys: list[str] | set[str]) -> list[str]:
    return sorted(clip_keys, key=lambda k: float(k.split("_")[0]))


def expand_references(
    selected: set[str],
    all_clip_keys: list[str] | set[str],
    radius: int = 1,
) -> tuple[set[str], list[dict]]:
    """Return (expanded set, evidence records). Evidence records explain every
    added segment (CLAUDE.md §12: save why each segment was selected)."""
    ordered = sort_clip_keys(all_clip_keys)
    index = {k: i for i, k in enumerate(ordered)}

    unknown = selected - set(index)
    if unknown:
        raise ValueError(f"selected clips not in video clip list: {sorted(unknown)[:5]}")

    expanded = set(selected)
    evidence = [{"segment": k, "reason": "base_selection"} for k in sort_clip_keys(selected)]
    if radius > 0:
        for k in sort_clip_keys(selected):
            i = index[k]
            for j in range(max(0, i - radius), min(len(ordered), i + radius + 1)):
                n = ordered[j]
                if n not in expanded:
                    expanded.add(n)
                    evidence.append({
                        "segment": n,
                        "reason": f"temporal_neighbor(radius={radius})",
                        "of": k,
                    })
    return expanded, evidence
