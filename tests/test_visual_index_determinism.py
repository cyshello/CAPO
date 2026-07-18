import numpy as np
import pytest

from surrogate_rollout.retrieval import visual_index as vi


def _fixture_isolated_worker(connection, physical_gpu, model_id, batch_size):
    import os

    connection.send({
        "status": "ready", "pid": os.getpid(),
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": "cuda:0",
    })
    while True:
        message = connection.recv()
        if message["operation"] == "close":
            connection.send({"status": "closed"})
            connection.close()
            return
        count = len(message["values"])
        connection.send({
            "status": "ok",
            "value": np.full((count, 2), batch_size, dtype=np.float32),
        })


@pytest.fixture
def frames_dir(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(25):
        (d / f"frame_n{i:06d}.jpg").write_bytes(bytes([i]) * 10)
    return str(d)


def fake_embed(paths):
    # deterministic per-path embedding (hash of basename)
    import hashlib
    import os

    rows = []
    for p in paths:
        h = hashlib.sha256(os.path.basename(p).encode()).digest()[:8]
        rows.append([b / 255.0 + 0.01 for b in h])
    return np.asarray(rows, dtype=np.float32)


def _build(frames_dir):
    return vi.build_visual_index(
        video_id="v1", frames_dir=frames_dir, duration=25.0,
        embed_images_fn=fake_embed, model_id="test-model")


def test_build_is_deterministic(frames_dir):
    a, b = _build(frames_dir), _build(frames_dir)
    assert a.frame_names == b.frame_names
    assert a.clip_ids == b.clip_ids
    assert np.array_equal(a.embeddings, b.embeddings)
    assert a.key == b.key


def test_embeddings_l2_normalized(frames_dir):
    idx = _build(frames_dir)
    assert np.allclose(np.linalg.norm(idx.embeddings, axis=1), 1.0, atol=1e-5)


def test_frame_to_clip_mapping_matches_dvd_windows(frames_dir):
    # duration 25s, clip_secs 10 -> windows 0_10, 10_20, 20_25 (short tail)
    idx = _build(frames_dir)
    assert idx.all_clip_ids == ["0_10", "10_20", "20_25"]
    # frame i is at t = i * 25 / 25 = i seconds
    assert idx.clip_ids[0] == "0_10"
    assert idx.clip_ids[9] == "0_10"
    assert idx.clip_ids[10] == "10_20"
    assert idx.clip_ids[20] == "20_25"
    assert idx.clip_ids[24] == "20_25"


def test_unloaded_and_cpu_siglip_close_is_idempotent():
    embedder = vi.SiglipEmbedder(device="cpu")
    embedder.close()
    embedder._model = object()
    embedder._processor = object()
    embedder.close()
    embedder.close()

    assert embedder._model is None
    assert embedder._processor is None


def test_dedicated_siglip_process_uses_physical_gpu_as_logical_zero(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    embedder = vi.ProcessIsolatedSiglipEmbedder(
        physical_gpu="4", model_id="fixture-model", batch_size=7,
        worker_target=_fixture_isolated_worker)
    try:
        assert embedder.startup_metadata["cuda_visible_devices"] == "4"
        assert embedder.startup_metadata["logical_device"] == "cuda:0"
        assert embedder.is_alive
        assert np.array_equal(
            embedder.embed_texts(("one", "two")),
            np.full((2, 2), 7, dtype=np.float32))
        assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "5"
    finally:
        embedder.close()
    assert not embedder.is_alive


def test_run_boundary_cleanup_closes_partially_built_runtime_worker(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    embedder = vi.ProcessIsolatedSiglipEmbedder(
        physical_gpu="4", model_id="fixture-model", batch_size=3,
        worker_target=_fixture_isolated_worker)

    records = vi.close_all_isolated_siglip_embedders()

    assert records == ({
        "pid": embedder.worker_pid, "physical_gpu": "4", "released": True},)
    assert not embedder.is_alive
    assert vi.close_all_isolated_siglip_embedders() == ()
