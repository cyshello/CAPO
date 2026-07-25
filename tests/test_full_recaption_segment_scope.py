"""Which baseline keys the full-recaption runner treats as segments.

A captions file holds the merged subject registry next to the segments, so the
scope walk has to skip it -- float("subject") ended the 2026-07-25 run after
four videos of captioning, the first time a full-recaption reached the
intervention phase. The opposite failure is worse and silent: dropping a real
segment would quietly narrow what gets recaptioned, so an unknown key raises.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _ancestor in (_ROOT, _ROOT.parent):
    if str(_ancestor) not in sys.path:
        sys.path.insert(0, str(_ancestor))

from full_recaption_opt import full_recaption_runner as runner  # noqa: E402


def _baseline(tmp_path, captions) -> str:
    captions_path = tmp_path / "captions.json"
    captions_path.write_text(json.dumps(captions), encoding="utf-8")
    manifest_path = tmp_path / "video_manifest.json"
    manifest_path.write_text(
        json.dumps({"captions_path": str(captions_path)}), encoding="utf-8")
    return str(manifest_path)


def test_registry_keys_are_not_segments(tmp_path):
    manifest = _baseline(tmp_path, {
        "0_10": "first",
        "10_20": "second",
        "subject_registry": {"Judge": "wears a robe"},
        "character_registry": {"Judge": "wears a robe"},
    })
    assert runner._all_baseline_segments(manifest) == ("0_10", "10_20")


def test_segments_are_ordered_by_start_time_not_lexically(tmp_path):
    manifest = _baseline(tmp_path, {
        "100_110": "c", "20_30": "b", "0_10": "a", "1000_1010": "d",
        "subject_registry": {},
    })
    assert runner._all_baseline_segments(manifest) == (
        "0_10", "20_30", "100_110", "1000_1010")


def test_fractional_starts_are_kept(tmp_path):
    manifest = _baseline(tmp_path, {"0.5_10": "a", "10_20.5": "b"})
    assert runner._all_baseline_segments(manifest) == ("0.5_10", "10_20.5")


def test_an_unknown_non_segment_key_raises_rather_than_shrinking_scope(tmp_path):
    manifest = _baseline(tmp_path, {
        "0_10": "first", "clip_description_registry": {"x": "y"},
    })
    with pytest.raises(ValueError, match="clip_description_registry"):
        runner._all_baseline_segments(manifest)


def test_the_whole_video_is_returned(tmp_path):
    captions = {f"{start}_{start + 10}": "c" for start in range(0, 2000, 10)}
    captions["subject_registry"] = {}
    manifest = _baseline(tmp_path, captions)
    segments = runner._all_baseline_segments(manifest)
    assert len(segments) == 200
    assert segments[0] == "0_10" and segments[-1] == "1990_2000"
