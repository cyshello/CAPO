"""CLIP/SigLIP clip retrieval over a cached visual index (PHASE2_3 §9).

frame_score = max over queries of cosine similarity (embeddings are already
L2-normalized, so a matrix product); clip_score = max over the clip's frames.
Simple maximum similarity only — no fusion, no reranking. Deterministic: ties
break by clip start time.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from surrogate_rollout import config
from surrogate_rollout.retrieval.visual_index import VisualIndex
from surrogate_rollout.selection.base import SelectionResult


def clip_scores(index: VisualIndex, query_embeddings: np.ndarray) -> dict[str, float]:
    """Max-over-queries, max-over-frames similarity per clip."""
    q = np.asarray(query_embeddings, dtype=np.float32)
    if q.ndim != 2 or q.shape[0] == 0:
        raise ValueError("query_embeddings must be a non-empty (k, dim) matrix")
    sims = index.embeddings @ q.T            # (n_frames, n_queries)
    frame_scores = sims.max(axis=1)          # (n_frames,)
    scores: dict[str, float] = {}
    for clip_id, s in zip(index.clip_ids, frame_scores):
        s = float(s)
        if s > scores.get(clip_id, float("-inf")):
            scores[clip_id] = s
    return scores


def retrieve_clips(
    index: VisualIndex,
    query_embeddings: np.ndarray,
    *,
    source: str,
    top_k: int = config.RETRIEVAL_TOP_K,
    max_fraction: float | None = None,
    queries: Sequence[str] | None = None,
) -> SelectionResult:
    scores = clip_scores(index, query_embeddings)
    total = len(set(index.clip_ids))
    limit = top_k
    if max_fraction is not None:
        limit = min(limit, max(1, int(max_fraction * total)))

    ranked = sorted(scores, key=lambda c: (-scores[c], float(c.split("_")[0])))
    result = SelectionResult()
    for cid in ranked[:limit]:
        result.add(cid, source, scores[cid])
    result.metadata[f"{source}_queries"] = list(queries or [])
    result.metadata[f"{source}_top_k"] = limit
    return result
