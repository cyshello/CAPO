"""Clip-selector interface for selective surrogate rollout (PHASE2_3 §4).

A selector decides which clips get candidate-prompt recaptioning. Every
selected clip records its provenance in `sources` so the sanity check can
attribute agreement to trace evidence vs retrieval vs random baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from surrogate_rollout.evaluation.rollout_evaluator import SelectionBudget

# provenance source tags (budget priority is defined over these; §10)
SOURCE_FRAME_INSPECTION = "trace_frame_inspection"
SOURCE_TRACE_STRICT = "trace_explicit_citation"
SOURCE_TRACE_RETURNED = "trace_returned"
SOURCE_TRACE_RETRIEVED = "trace_retrieved"
SOURCE_PROMPT_DELTA = "prompt_delta_clip"
SOURCE_QUESTION = "question_clip"
SOURCE_NEIGHBOR = "temporal_neighbor"
SOURCE_RANDOM = "random_budget_matched"


@dataclass(frozen=True)
class SelectionContext:
    video_id: str
    qa_id: str
    question: str
    answer_options: tuple[str, ...]
    baseline_prompt: str
    candidate_prompt: str
    baseline_trajectory: tuple[dict, ...]     # messages
    baseline_tool_events: tuple[dict, ...]    # instrumentation sidecar events
    all_clip_ids: tuple[str, ...]             # every clip key, ordered
    reference_policy: str
    budget: SelectionBudget
    visual_index: object | None = None        # retrieval.visual_index.VisualIndex
    random_seed: int = 0
    budget_matched_count: int | None = None


@dataclass
class SelectionResult:
    clip_ids: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    sources: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def add(self, clip_id: str, source: str, score: float | None = None) -> None:
        if clip_id not in self.sources:
            self.clip_ids.append(clip_id)
            self.sources[clip_id] = []
        if source not in self.sources[clip_id]:
            self.sources[clip_id].append(source)
        if score is not None:
            self.scores[clip_id] = max(score, self.scores.get(clip_id, score))


class ClipSelector(Protocol):
    def select(self, context: SelectionContext) -> SelectionResult:
        ...


def merge_results(*results: SelectionResult) -> SelectionResult:
    """Union of selections; provenance and metadata merged, scores keep max."""
    out = SelectionResult()
    for r in results:
        for cid in r.clip_ids:
            for src in r.sources.get(cid, []):
                out.add(cid, src, r.scores.get(cid))
        out.metadata.update(r.metadata)
    return out
