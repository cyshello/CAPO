import json
from pathlib import Path

import numpy as np
import pytest

from surrogate_rollout.optimization.property_proposal import CandidatePropertyProposal
from surrogate_rollout.retrieval.property_retrieval import (
    PropertyFrameRetriever,
    PropertyRetrievalBatchRunner,
    PropertyRetrievalConflictError,
    PropertyRetrievalError,
    retrieval_identity,
)
from surrogate_rollout.retrieval.visual_index import VisualIndex


def candidate(candidate_id, text):
    return CandidatePropertyProposal(
        candidate_property_id=candidate_id, property_text=text,
        source_video_id="video-1", source_question_ids=("q1",),
        motivating_failure_types=("fixture_failure",),
        covered_by_existing_property_ids=(), proposal_rationale="fixture",
        proposer_policy_version="fixture_v1")


def index(embeddings=None):
    values = embeddings if embeddings is not None else np.array([
        [0.20, 0.98], [0.98, 0.20],  # 0_10: x max is frame 1
        [0.80, 0.60], [0.70, 0.70],  # 10_20
        [0.00, 1.00], [0.10, 0.99],  # 20_30
    ], dtype=np.float32)
    return VisualIndex(
        video_id="video-1", model_id="fixture-siglip",
        frame_names=[f"frame-{i}.jpg" for i in range(6)],
        clip_ids=["0_10", "0_10", "10_20", "10_20", "20_30", "20_30"],
        timestamps=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        embeddings=np.asarray(values, dtype=np.float32),
        key={
            "video_id": "video-1", "vision_model_id": "fixture-siglip",
            "frame_sampling_config": {"sample_fps": 0.2, "clip_secs": 10},
            "frame_source_hash": "frames", "preprocessing_version": "vi_v1",
        },
    )


class Embedder:
    def __init__(self):
        self.calls = []
        self.image_calls = []

    def embed_texts(self, texts):
        self.calls.append(tuple(texts))
        vectors = {
            "Track object appearance.": [1.0, 0.0],
            "Track action completion.": [0.0, 1.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)

    def embed_images(self, paths):
        self.image_calls.append(tuple(paths))
        return np.asarray([[1.0, 0.0] for _ in paths], dtype=np.float32)


def test_max_frame_pooling_and_top_k_when_video_is_smaller_than_budget(tmp_path):
    embedder = Embedder()
    result = PropertyFrameRetriever(embedder=embedder, top_k=5).retrieve(
        proposal=candidate("cp_appearance", "Track object appearance."),
        visual_index=index(), output_dir=str(tmp_path / "retrieval"))
    assert result.s_sim == ("0_10", "10_20", "20_30")
    first = result.artifact["ranked_segments"][0]
    assert first["segment_id"] == "0_10"
    assert first["max_frame_id"] == "frame-1.jpg"
    assert first["segment_score"] == max(
        item["score"] for item in first["frame_scores"])
    assert len(result.artifact["ranked_segments"]) == 3


def test_filters_to_baseline_caption_universe_before_top_k(tmp_path):
    visual = index(np.array([
        [1.00, 0.00], [0.99, 0.01],  # 0_10: eligible rank 1
        [0.98, 0.02], [0.97, 0.03],  # 10_20: eligible rank 2
        [1.00, 0.00], [1.00, 0.00],  # 20_30: excluded despite top score
    ], dtype=np.float32))
    result = PropertyFrameRetriever(embedder=Embedder(), top_k=2).retrieve(
        proposal=candidate("cp_appearance", "Track object appearance."),
        visual_index=visual, output_dir=str(tmp_path / "filtered"),
        baseline_segment_ids=("0_10", "10_20"))
    assert result.s_sim == ("0_10", "10_20")
    assert [row["segment_id"] for row in result.artifact["ranked_segments"]] == [
        "0_10", "10_20"]
    assert result.artifact["excluded_visual_index_segment_ids"] == ["20_30"]
    assert result.artifact["eligible_index_segment_count"] == 2


def test_baseline_segment_universe_changes_retrieval_identity_and_resume(tmp_path):
    embedder = Embedder()
    proposal = candidate("cp_appearance", "Track object appearance.")
    retriever = PropertyFrameRetriever(embedder=embedder, top_k=1)
    first = retriever.retrieve(
        proposal=proposal, visual_index=index(),
        output_dir=str(tmp_path / "same"),
        baseline_segment_ids=("0_10", "10_20"))
    resumed = retriever.retrieve(
        proposal=proposal, visual_index=index(),
        output_dir=str(tmp_path / "same"),
        baseline_segment_ids=("0_10", "10_20"))
    assert resumed.resumed
    assert first.retrieval_identity["baseline_segment_ids_hash"] == \
        resumed.retrieval_identity["baseline_segment_ids_hash"]
    with pytest.raises(PropertyRetrievalConflictError):
        retriever.retrieve(
            proposal=proposal, visual_index=index(),
            output_dir=str(tmp_path / "same"),
            baseline_segment_ids=("0_10", "20_30"))


def test_invalid_baseline_segment_universe_fails_closed(tmp_path):
    retriever = PropertyFrameRetriever(embedder=Embedder(), top_k=1)
    proposal = candidate("cp_appearance", "Track object appearance.")
    with pytest.raises(PropertyRetrievalError, match="duplicate"):
        retriever.retrieve(
            proposal=proposal, visual_index=index(),
            output_dir=str(tmp_path / "duplicate"),
            baseline_segment_ids=("0_10", "0_10"))
    with pytest.raises(PropertyRetrievalError, match="no segments"):
        retriever.retrieve(
            proposal=proposal, visual_index=index(),
            output_dir=str(tmp_path / "disjoint"),
            baseline_segment_ids=("90_100",))


def test_deterministic_score_ties_use_start_then_stable_segment_id(tmp_path):
    tied = index(np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (6, 1)))
    tied.clip_ids = ["0_5", "0_5", "0_10", "0_10", "20_30", "20_30"]
    result = PropertyFrameRetriever(embedder=Embedder(), top_k=3).retrieve(
        proposal=candidate("cp_tie", "Track object appearance."),
        visual_index=tied, output_dir=str(tmp_path / "ties"))
    assert result.s_sim == ("0_10", "0_5", "20_30")


def test_two_properties_same_video_produce_distinct_s_sim(tmp_path):
    embedder = Embedder()
    retriever = PropertyFrameRetriever(embedder=embedder, top_k=1)
    appearance = retriever.retrieve(
        proposal=candidate("cp_appearance", "Track object appearance."),
        visual_index=index(), output_dir=str(tmp_path / "appearance"))
    completion = retriever.retrieve(
        proposal=candidate("cp_completion", "Track action completion."),
        visual_index=index(), output_dir=str(tmp_path / "completion"))
    assert appearance.s_sim == ("0_10",)
    assert completion.s_sim == ("20_30",)


def test_history_is_absent_from_identity_and_resume_skips_embedding(tmp_path):
    embedder = Embedder()
    proposal = candidate("cp_appearance", "Track object appearance.")
    retriever = PropertyFrameRetriever(embedder=embedder, top_k=1)
    history = tmp_path / "frozen_histories.jsonl"
    history.write_text('{"history_hash":"first"}\n')
    first = retriever.retrieve(
        proposal=proposal, visual_index=index(), output_dir=str(tmp_path / "result"))
    history.write_text('{"history_hash":"changed"}\n')
    second = retriever.retrieve(
        proposal=proposal, visual_index=index(), output_dir=str(tmp_path / "result"))
    assert first.retrieval_identity == second.retrieval_identity
    assert "history" not in json.dumps(first.retrieval_identity).lower()
    assert second.resumed
    assert len(embedder.calls) == 1


def test_property_text_changes_identity_and_conflicts_at_same_path(tmp_path):
    a = candidate("cp_a", "Track object appearance.")
    b = candidate("cp_a", "Track action completion.")
    assert retrieval_identity(a, index(), top_k=1) != retrieval_identity(
        b, index(), top_k=1)
    retriever = PropertyFrameRetriever(embedder=Embedder(), top_k=1)
    retriever.retrieve(
        proposal=a, visual_index=index(), output_dir=str(tmp_path / "same"))
    with pytest.raises(PropertyRetrievalConflictError):
        retriever.retrieve(
            proposal=b, visual_index=index(), output_dir=str(tmp_path / "same"))


def test_stale_visual_index_and_invalid_scores_fail_closed(tmp_path):
    retriever = PropertyFrameRetriever(embedder=Embedder(), top_k=1)
    proposal = candidate("cp_a", "Track object appearance.")
    retriever.retrieve(
        proposal=proposal, visual_index=index(), output_dir=str(tmp_path / "same"))
    changed = index()
    changed.embeddings[0] = [0.5, 0.5]
    with pytest.raises(PropertyRetrievalConflictError):
        retriever.retrieve(
            proposal=proposal, visual_index=changed,
            output_dir=str(tmp_path / "same"))
    invalid = index()
    invalid.embeddings[0, 0] = np.nan
    with pytest.raises(PropertyRetrievalError, match="non-finite"):
        retriever.retrieve(
            proposal=proposal, visual_index=invalid,
            output_dir=str(tmp_path / "invalid"))


def test_batch_runner_persists_each_property_and_zero_batch_skips_index(tmp_path):
    embedder = Embedder()
    load_calls = []

    def loader(sample, video_id):
        load_calls.append((sample["sample_id"], video_id))
        return index()

    runner = PropertyRetrievalBatchRunner(
        retriever=PropertyFrameRetriever(embedder=embedder, top_k=1),
        visual_index_loader=loader)
    sample = {"sample_id": "video-1"}
    proposals = (
        candidate("cp_appearance", "Track object appearance."),
        candidate("cp_completion", "Track action completion."),
    )
    result = runner.run(
        sample=sample, proposals=proposals, output_dir=str(tmp_path / "batch"),
        baseline_segment_ids=("0_10", "20_30"))
    assert len(result.retrieval_artifact_paths) == 2
    assert [json.loads(Path(path).read_text())["s_sim"] for path in
            result.retrieval_artifact_paths] == [["0_10"], ["20_30"]]
    batch_manifest = json.loads(Path(result.manifest_path).read_text())
    assert batch_manifest["segment_universe_policy"] == \
        "valid_baseline_caption_segment_intersection_v2"
    assert batch_manifest["baseline_segment_count"] == 2
    resumed = runner.run(
        sample=sample, proposals=proposals, output_dir=str(tmp_path / "batch"),
        baseline_segment_ids=("0_10", "20_30"))
    assert resumed.resumed
    assert len(embedder.calls) == 2

    zero_loads = []
    zero_runner = PropertyRetrievalBatchRunner(
        retriever=PropertyFrameRetriever(embedder=Embedder()),
        visual_index_loader=lambda *_: zero_loads.append(True))
    zero = zero_runner.run(
        sample=sample, proposals=(), output_dir=str(tmp_path / "zero"))
    assert zero.retrieval_artifact_paths == ()
    assert zero_loads == []


def test_compatible_cached_visual_index_skips_frame_reembedding(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame-0.jpg").write_bytes(b"frame0")
    (frames / "frame-1.jpg").write_bytes(b"frame1")
    embedder = Embedder()
    retriever = PropertyFrameRetriever(embedder=embedder, top_k=1)
    proposal = candidate("cp_appearance", "Track object appearance.")
    for name in ("first", "second"):
        retriever.retrieve_from_frames(
            proposal=proposal, frames_dir=str(frames), duration=20.0,
            output_dir=str(tmp_path / name),
            visual_index_root=str(tmp_path / "visual_cache"),
            model_id="fixture-siglip")
    assert len(embedder.image_calls) == 1
