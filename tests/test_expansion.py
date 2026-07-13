import pytest

from surrogate_rollout.references.expansion import expand_references, sort_clip_keys

CLIPS = [f"{i * 10}_{(i + 1) * 10}" for i in range(10)]  # 0_10 ... 90_100


def test_sort_clip_keys_numeric_not_lexicographic():
    assert sort_clip_keys(["100_110", "20_30", "0_10"]) == ["0_10", "20_30", "100_110"]


def test_radius_zero_is_identity():
    exp, ev = expand_references({"20_30"}, CLIPS, radius=0)
    assert exp == {"20_30"}
    assert [e["reason"] for e in ev] == ["base_selection"]


def test_radius_one_adds_both_neighbors():
    exp, ev = expand_references({"20_30"}, CLIPS, radius=1)
    assert exp == {"10_20", "20_30", "30_40"}
    reasons = {e["segment"]: e["reason"] for e in ev}
    assert reasons["10_20"].startswith("temporal_neighbor")
    assert reasons["20_30"] == "base_selection"


def test_boundary_clips_clamped():
    exp, _ = expand_references({"0_10"}, CLIPS, radius=1)
    assert exp == {"0_10", "10_20"}
    exp, _ = expand_references({"90_100"}, CLIPS, radius=1)
    assert exp == {"80_90", "90_100"}


def test_radius_two():
    exp, _ = expand_references({"50_60"}, CLIPS, radius=2)
    assert exp == {"30_40", "40_50", "50_60", "60_70", "70_80"}


def test_unknown_clip_raises():
    with pytest.raises(ValueError):
        expand_references({"999_1009"}, CLIPS, radius=1)


def test_all_selected_stays_all():
    exp, _ = expand_references(set(CLIPS), CLIPS, radius=1)
    assert exp == set(CLIPS)
