import json

import pytest

from surrogate_rollout.mixed_views.builder import MixedViewBuilder

CLIPS = [f"{i * 10}_{(i + 1) * 10}" for i in range(5)]


def stub_merge(registries):
    return {"merged_from": len(registries)}


@pytest.fixture
def baseline(tmp_path):
    captions = {k: {"caption": f"baseline caption {k}"} for k in CLIPS}
    captions["subject_registry"] = {"merged_from": 0}
    p = tmp_path / "baseline" / "captions.json"
    p.parent.mkdir()
    p.write_text(json.dumps(captions, indent=4))
    return p


def _candidate(clips):
    return {k: {"clip_description": f"candidate caption {k}",
                "subject_registry": {"clip": k}} for k in clips}


def test_only_selected_clips_replaced(baseline, tmp_path):
    before = baseline.read_bytes()
    selected = {"10_20", "30_40"}
    art = MixedViewBuilder(merge_fn=stub_merge).build(
        str(baseline), _candidate(selected), selected, str(tmp_path / "work"),
        video_id="v1")

    mixed = json.loads(open(art.captions_path).read())
    for k in CLIPS:
        if k in selected:
            assert mixed[k]["caption"] == f"candidate caption {k}"
        else:
            assert mixed[k]["caption"] == f"baseline caption {k}"
    assert art.replaced_clip_ids == sorted(selected)
    # frozen baseline snapshot untouched
    assert baseline.read_bytes() == before


def test_parse_failed_candidate_keeps_baseline(baseline, tmp_path):
    selected = {"10_20", "30_40"}
    candidate = _candidate({"10_20"})
    candidate["30_40"] = {}  # deterministic parse failure cached as {}
    art = MixedViewBuilder(merge_fn=stub_merge).build(
        str(baseline), candidate, selected, str(tmp_path / "work"), video_id="v1")
    mixed = json.loads(open(art.captions_path).read())
    assert mixed["30_40"]["caption"] == "baseline caption 30_40"
    assert art.replaced_clip_ids == ["10_20"]


def test_unknown_selected_clip_raises(baseline, tmp_path):
    with pytest.raises(ValueError):
        MixedViewBuilder(merge_fn=stub_merge).build(
            str(baseline), {}, {"999_1009"}, str(tmp_path / "work"), video_id="v1")


def test_registry_is_re_merged_not_copied(baseline, tmp_path):
    selected = {"10_20"}
    art = MixedViewBuilder(merge_fn=stub_merge).build(
        str(baseline), _candidate(selected), selected, str(tmp_path / "work"),
        video_id="v1")
    mixed = json.loads(open(art.captions_path).read())
    # stub merged over the candidate clip's registry only (no baseline ckpt dir)
    assert mixed["subject_registry"] == {"merged_from": 1}
