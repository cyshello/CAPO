import json

import pytest

from surrogate_rollout.mixed_views.builder import MixedViewBuilder

CLIPS = ["0_10", "10_20", "20_30"]


@pytest.fixture
def baseline(tmp_path):
    captions = {k: {"caption": f"b {k}"} for k in CLIPS}
    captions["subject_registry"] = None
    p = tmp_path / "base" / "captions.json"
    p.parent.mkdir()
    p.write_text(json.dumps(captions, indent=4))
    return p


def _build(baseline, tmp_path, text, name):
    return MixedViewBuilder(merge_fn=lambda r: None).build(
        str(baseline),
        {"10_20": {"clip_description": text}},
        {"10_20"},
        str(tmp_path / "work"),
        video_id="v1",
        view_name=name,
    )


def test_different_contents_different_db_keys(baseline, tmp_path):
    a = _build(baseline, tmp_path, "candidate A", "va")
    b = _build(baseline, tmp_path, "candidate B", "vb")
    assert a.captions_hash != b.captions_hash
    assert a.database_path != b.database_path


def test_identical_contents_same_db_key(baseline, tmp_path):
    a = _build(baseline, tmp_path, "same text", "va")
    b = _build(baseline, tmp_path, "same text", "vb")
    assert a.captions_hash == b.captions_hash
    assert a.database_path == b.database_path


def test_db_path_embeds_content_hash(baseline, tmp_path):
    a = _build(baseline, tmp_path, "candidate A", "va")
    assert a.captions_hash[:16] in a.database_path
