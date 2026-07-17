import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_phase4_bounded_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_phase4_bounded_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_build_retrieval_runner = MODULE._build_retrieval_runner


def test_bounded_smoke_keeps_retrieval_embedder_off_worker_gpus(tmp_path):
    runner = _build_retrieval_runner(
        cache_dir=str(tmp_path), video_duration_seconds=lambda _: 1.0)

    assert runner.retriever.embedder._device == "cpu"
    assert runner.retriever.top_k == 1
    assert runner.visual_index_loader.__closure__ is not None
    captured = {
        cell.cell_contents for cell in runner.visual_index_loader.__closure__
        if hasattr(cell.cell_contents, "embed_texts")
    }
    assert captured == {runner.retriever.embedder}
