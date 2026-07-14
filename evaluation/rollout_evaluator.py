"""Shared evaluator interface for Phase 2 rollouts (PHASE2_3_SURROGATE.md §1).

Both FullRolloutEvaluator and SelectiveSurrogateRolloutEvaluator consume an
EvaluationRequest and return an EvaluationResult, so switching rollout modes is
a configuration change (CLAUDE.md §10). The GEPA layer never sees mode-specific
logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

from surrogate_rollout.schemas import DVDRunResult

# fallback_type values (PHASE2_3 §11) — recorded, never silent.
FALLBACK_NONE = "none"
FALLBACK_CLIP_ONLY = "clip_only"
FALLBACK_FULL_ROLLOUT = "full_rollout"
FALLBACK_UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SelectionBudget:
    """Recaption budget (PHASE2_3 §10)."""

    max_recaption_fraction: float = 0.5
    max_total_clips: int | None = None
    max_clip_retrieval_fraction: float = 0.25
    neighbor_radius: int = 1

    def max_clips(self, total_clips: int) -> int:
        n = int(self.max_recaption_fraction * total_clips)
        if self.max_total_clips is not None:
            n = min(n, self.max_total_clips)
        return max(n, 1)

    def max_retrieval_clips(self, total_clips: int) -> int:
        return max(1, int(self.max_clip_retrieval_fraction * total_clips))


@dataclass(frozen=True)
class EvaluationRequest:
    """One (QA, candidate prompt) evaluation. `sample` is the provider sample
    dict (video_path / options / extra.videoID / subtitle) needed to run DVD;
    the scalar fields duplicate what metrics and logs need without reparsing
    it."""

    video_id: str
    qa_id: str
    question: str
    answer_options: tuple[str, ...]
    ground_truth: str | None
    sample: dict
    baseline_prompt: str
    candidate_prompt: str
    baseline_captions_path: str
    baseline_trajectory_dir: str | None  # run dir with trajectory.jsonl + tool_events.jsonl
    reference_policy: str
    selection_policy: str
    work_root: str
    budget: SelectionBudget = SelectionBudget()
    merge_prompt: str | None = None      # None -> DVD default
    gpu: str | None = None
    # random_budget_matched: target count; None -> budget.max_clips
    budget_matched_count: int | None = None
    random_seed: int = 0


@dataclass
class EvaluationResult:
    video_id: str
    qa_id: str
    rollout_mode: str                    # "full" | "selective"
    selection_policy: str
    answer: str | None
    parsed_answer: str | None
    is_correct: bool
    score: float
    selected_clip_ids: list[str] = field(default_factory=list)
    recaptioned_clip_ids: list[str] = field(default_factory=list)
    recaption_fraction: float = 0.0
    total_clips: int = 0
    selection_sources: dict[str, list[str]] = field(default_factory=dict)
    fallback_type: str = FALLBACK_NONE
    fallback_reason: str | None = None
    captions_hash: str | None = None
    captions_path: str | None = None
    database_path: str | None = None
    trajectory_path: str | None = None
    caption_call_count: int = 0
    caption_cache_hits: int = 0
    caption_seconds: float = 0.0
    reasoning_seconds: float = 0.0
    latency_seconds: float = 0.0
    errors: list[dict] = field(default_factory=list)
    run_result: DVDRunResult | None = None

    def as_json(self) -> dict:
        d = asdict(self)
        d.pop("run_result", None)  # persisted separately by dvd_qa artifacts
        return d


class RolloutEvaluator(Protocol):
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        ...
