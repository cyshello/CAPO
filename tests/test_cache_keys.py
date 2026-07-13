import os

import pytest

from surrogate_rollout.cache import caption_cache as cc
from surrogate_rollout.schemas import CaptionCacheKey


def _key(**over):
    base = dict(video_id="v1", segment_id="0_10", prompt_hash="p" * 64,
                caption_model_id="m", decoding_hash="d" * 64, source_hash="s" * 64)
    base.update(over)
    return CaptionCacheKey(**base)


def test_keys_differ_across_every_dimension():
    base = _key()
    assert base != _key(prompt_hash="q" * 64)
    assert base != _key(caption_model_id="other-model")
    assert base != _key(decoding_hash="e" * 64)
    assert base != _key(source_hash="t" * 64)
    assert base != _key(segment_id="10_20")
    assert base == _key()


def test_prompt_hash_full_and_legacy_tag_consistent():
    h = cc.prompt_hash("cap", "merge")
    assert len(h) == 64
    assert cc.prompt_hash("cap", "merge") == h
    assert cc.prompt_hash("cap2", "merge") != h
    # legacy md5 tag matches the vendored DVD scheme
    import hashlib

    assert cc.legacy_md5_tag("cap", "merge") == hashlib.md5(b"cap||merge").hexdigest()[:8]


def test_decoding_and_source_hash_sensitivity(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x" * 100)
    sub = tmp_path / "v.srt"
    sub.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")

    h1 = cc.source_hash(str(video), 1.0, 10, str(sub))
    assert cc.source_hash(str(video), 1.0, 10, str(sub)) == h1
    assert cc.source_hash(str(video), 2.0, 10, str(sub)) != h1     # fps
    assert cc.source_hash(str(video), 1.0, 20, str(sub)) != h1     # clip len
    assert cc.source_hash(str(video), 1.0, 10, None) != h1         # subtitle
    video.write_bytes(b"x" * 101)                                  # content size
    assert cc.source_hash(str(video), 1.0, 10, str(sub)) != h1


def test_manifest_register_and_readonly_guard(tmp_path):
    manifest = str(tmp_path / "manifest.jsonl")
    entry = {
        "video_id": "v1",
        "cache_dir": str(tmp_path / "legacy_cache"),
        "key": {"prompt_hash": "p" * 64},
        "read_only": True,
        "legacy": True,
    }
    cc.register_cache(entry, manifest)
    # idempotent for identical entry
    cc.register_cache(entry, manifest)
    assert len(cc.load_manifest(manifest)) == 1

    # conflicting re-registration aborts
    conflicting = dict(entry, key={"prompt_hash": "q" * 64})
    with pytest.raises(ValueError):
        cc.register_cache(conflicting, manifest)

    # write guard protects read-only cache dirs
    with pytest.raises(PermissionError):
        cc.assert_writable(entry["cache_dir"], manifest)
    cc.assert_writable(str(tmp_path / "other_dir"), manifest)  # unregistered: fine


def test_new_candidate_cache_dir_outside_legacy_workspace(tmp_path):
    key = _key()
    d = cc.new_candidate_cache_dir(key, root=str(tmp_path))
    assert d.startswith(str(tmp_path))
    assert key.video_id in d
    assert key.prompt_hash[:12] in d


def test_captions_content_hash(tmp_path):
    p = tmp_path / "captions.json"
    p.write_text('{"0_10": {"caption": "a"}}')
    h1 = cc.captions_content_hash(str(p))
    p.write_text('{"0_10": {"caption": "b"}}')
    assert cc.captions_content_hash(str(p)) != h1
