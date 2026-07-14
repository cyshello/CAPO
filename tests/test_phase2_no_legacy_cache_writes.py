import json
import os

import pytest

from surrogate_rollout import config
from surrogate_rollout.cache import caption_cache as cc
from surrogate_rollout.mixed_views.builder import write_captions_json
from surrogate_rollout.schemas import CaptionCacheKey


def _register_readonly(tmp_path, cache_dir):
    manifest = str(tmp_path / "manifest.jsonl")
    cc.register_cache({
        "video_id": "v1", "cache_dir": str(cache_dir),
        "key": {"prompt_hash": "p" * 64},
        "read_only": True, "legacy": True,
    }, manifest)
    return manifest


def test_write_captions_json_refuses_readonly_dir(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy_captions"
    legacy.mkdir()
    manifest = _register_readonly(tmp_path, legacy)
    monkeypatch.setattr(config, "CACHE_MANIFEST_PATH", manifest)

    with pytest.raises(PermissionError):
        write_captions_json({"0_10": {"caption": "x"}}, str(legacy))
    assert not (legacy / "captions.json").exists()


def test_mixed_view_never_touches_baseline_bytes(tmp_path):
    from surrogate_rollout.mixed_views.builder import MixedViewBuilder

    captions = {"0_10": {"caption": "b"}, "subject_registry": None}
    base = tmp_path / "base" / "captions.json"
    base.parent.mkdir()
    base.write_text(json.dumps(captions, indent=4))
    before = base.read_bytes()

    MixedViewBuilder(merge_fn=lambda r: None).build(
        str(base), {"0_10": {"clip_description": "c"}}, {"0_10"},
        str(tmp_path / "work"), video_id="v1")
    assert base.read_bytes() == before


def test_candidate_cache_dir_outside_legacy_workspace():
    key = CaptionCacheKey(video_id="v1", segment_id="*", prompt_hash="p" * 64,
                          caption_model_id="m", decoding_hash="d" * 64,
                          source_hash="s" * 64)
    d = cc.new_candidate_cache_dir(key)
    workspace = os.path.abspath(config.DVD_RUN_WORKSPACE)
    assert not os.path.abspath(d).startswith(workspace + os.sep)
    assert os.path.abspath(d).startswith(os.path.abspath(config.CAPTION_CACHE_ROOT))


def test_assert_writable_blocks_registered_legacy(tmp_path):
    legacy = tmp_path / "captions_tx_fps1_deadbeef"
    legacy.mkdir()
    manifest = _register_readonly(tmp_path, legacy)
    with pytest.raises(PermissionError):
        cc.assert_writable(str(legacy), manifest)
    # unregistered dirs stay writable
    cc.assert_writable(str(tmp_path / "elsewhere"), manifest)
