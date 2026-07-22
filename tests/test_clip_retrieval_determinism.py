import numpy as np

from surrogate_rollout.retrieval.clip_retriever import clip_scores, retrieve_clips
from surrogate_rollout.retrieval.visual_index import VisualIndex


def _index():
    # 3 clips x 2 frames; embeddings chosen so clip scores are unambiguous
    emb = np.array([
        [1.0, 0.0],   # 0_10
        [0.9, 0.1],
        [0.0, 1.0],   # 10_20
        [0.1, 0.9],
        [0.7, 0.7],   # 20_30
        [0.5, 0.5],
    ], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return VisualIndex(
        video_id="v1", model_id="m",
        frame_names=[f"f{i}.jpg" for i in range(6)],
        clip_ids=["0_10", "0_10", "10_20", "10_20", "20_30", "20_30"],
        timestamps=[float(i) * 5 for i in range(6)],
        embeddings=emb, key={})


def _q(vecs):
    q = np.asarray(vecs, dtype=np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_scores_max_over_frames_and_queries():
    scores = clip_scores(_index(), _q([[1.0, 0.0]]))
    assert scores["0_10"] > scores["20_30"] > scores["10_20"]


def test_retrieval_deterministic():
    idx = _index()
    q = _q([[1.0, 0.0], [0.0, 1.0]])
    a = retrieve_clips(idx, q, source="question_clip", top_k=2)
    b = retrieve_clips(idx, q, source="question_clip", top_k=2)
    assert a.clip_ids == b.clip_ids
    assert a.scores == b.scores


def test_tie_breaks_by_clip_start():
    idx = _index()
    # single query equidistant from clips 0_10 and 10_20 -> tie; earlier wins
    q = _q([[1.0, 1.0]])
    result = retrieve_clips(idx, q, source="question_clip", top_k=3)
    assert result.clip_ids[0] == "20_30"  # diagonal clip scores highest
    assert result.clip_ids[1:] == ["0_10", "10_20"]


def test_max_fraction_caps_results():
    idx = _index()
    q = _q([[1.0, 0.0]])
    result = retrieve_clips(idx, q, source="question_clip", top_k=10, max_fraction=0.34)
    assert len(result.clip_ids) == 1  # 3 clips * 0.34 -> 1


def test_sources_tagged():
    result = retrieve_clips(_index(), _q([[1.0, 0.0]]), source="prompt_delta_clip",
                            top_k=1)
    assert result.sources[result.clip_ids[0]] == ["prompt_delta_clip"]
