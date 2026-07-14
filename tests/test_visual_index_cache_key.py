import numpy as np
import pytest

from surrogate_rollout.retrieval import visual_index as vi


@pytest.fixture
def frames_dir(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(6):
        (d / f"frame_n{i:06d}.jpg").write_bytes(b"jpg" * (i + 1))
    return str(d)


def _key(frames_dir, **over):
    base = dict(video_id="v1", frames_dir=frames_dir,
                model_id="test-model", sample_fps=1.0, clip_secs=10,
                preprocessing_version="vi_v1")
    base.update(over)
    return vi.index_cache_key(**base)


def fake_embed(paths):
    return np.tile(np.arange(4, dtype=np.float32) + 1, (len(paths), 1))


def test_key_sensitive_to_every_field(frames_dir, tmp_path):
    base = _key(frames_dir)
    assert _key(frames_dir) == base
    assert _key(frames_dir, model_id="other") != base
    assert _key(frames_dir, sample_fps=2.0) != base
    assert _key(frames_dir, clip_secs=5) != base
    assert _key(frames_dir, preprocessing_version="vi_v2") != base
    # frame-source change (new frame file) changes the key
    other = tmp_path / "frames2"
    other.mkdir()
    for i in range(7):
        (other / f"frame_n{i:06d}.jpg").write_bytes(b"jpg")
    assert _key(str(other)) != base


def test_save_load_roundtrip_and_miss(frames_dir, tmp_path):
    root = str(tmp_path / "index_root")
    idx = vi.build_visual_index(
        video_id="v1", frames_dir=frames_dir, duration=60.0,
        embed_images_fn=fake_embed, model_id="test-model")
    vi.save_visual_index(idx, root)

    loaded = vi.load_visual_index(idx.key, root)
    assert loaded is not None
    assert loaded.clip_ids == idx.clip_ids
    assert np.allclose(loaded.embeddings, idx.embeddings)

    other_key = dict(idx.key, vision_model_id="different-model")
    assert vi.load_visual_index(other_key, root) is None
