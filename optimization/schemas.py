"""Frozen Stage 4.8 counterfactual-evidence records (Phase 4 section 10.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping


OUTCOME_TRANSITIONS = (
    "correct_to_correct",
    "correct_to_wrong",
    "wrong_to_correct",
    "wrong_to_wrong",
    "unknown",
)


def _freeze_json(value: Any, where: str) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): _freeze_json(item, f"{where}[{key!r}]")
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{where}[]") for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{where} contains non-JSON value {type(value).__name__}")


def _freeze_mapping(record: Any, name: str) -> None:
    value = getattr(record, name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{type(record).__name__}.{name} must be a Mapping")
    object.__setattr__(record, name, _freeze_json(
        value, f"{type(record).__name__}.{name}"))


def _require_str_tuple(value: tuple[str, ...], name: str) -> None:
    if not isinstance(value, tuple) or any(not isinstance(v, str) for v in value):
        raise TypeError(f"{name} must be a tuple of str")


@dataclass(frozen=True)
class CaptionDifference:
    segment_id: str
    baseline_caption: str
    candidate_caption: str
    difference_summary: str | None
    structured_differences: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("CaptionDifference.segment_id must be non-empty")
        if not isinstance(self.baseline_caption, str) or \
                not isinstance(self.candidate_caption, str):
            raise TypeError("CaptionDifference captions must be strings")
        _freeze_mapping(self, "structured_differences")


@dataclass(frozen=True)
class CounterfactualEvidence:
    evidence_id: str
    video_id: str
    question_id: str
    ground_truth: str | None

    baseline_bank_version: str
    baseline_router_version: str
    baseline_scaffold_version: str
    candidate_bank_version: str
    candidate_router_version: str
    candidate_scaffold_version: str

    baseline_selected_prompt_ids: Mapping[str, tuple[str, ...]]
    candidate_selected_prompt_ids: Mapping[str, tuple[str, ...]]
    baseline_composed_prompt_refs: Mapping[str, str]
    candidate_composed_prompt_refs: Mapping[str, str]

    baseline_answer: str | None
    candidate_answer: str | None
    baseline_score: float
    candidate_score: float
    score_delta: float
    outcome_transition: Literal[
        "correct_to_correct",
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_wrong",
        "unknown",
    ]

    segment_ids: tuple[str, ...]
    caption_differences: tuple[CaptionDifference, ...]
    baseline_trajectory_refs: tuple[str, ...]
    candidate_trajectory_refs: tuple[str, ...]
    rollout_artifact_refs: Mapping[str, str]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "evidence_id", "video_id", "question_id",
            "baseline_bank_version", "baseline_router_version",
            "baseline_scaffold_version", "candidate_bank_version",
            "candidate_router_version", "candidate_scaffold_version",
        ):
            if not getattr(self, name):
                raise ValueError(f"CounterfactualEvidence.{name} must be non-empty")
        if self.outcome_transition not in OUTCOME_TRANSITIONS:
            raise ValueError(
                f"invalid outcome_transition {self.outcome_transition!r}")
        if abs(self.score_delta - (
                self.candidate_score - self.baseline_score)) > 1e-12:
            raise ValueError("score_delta must equal candidate_score - baseline_score")
        for name in (
            "segment_ids", "baseline_trajectory_refs",
            "candidate_trajectory_refs",
        ):
            _require_str_tuple(getattr(self, name), name)
        if len(self.segment_ids) != len(set(self.segment_ids)):
            raise ValueError("CounterfactualEvidence.segment_ids contains duplicates")
        if not isinstance(self.caption_differences, tuple) or any(
                not isinstance(item, CaptionDifference)
                for item in self.caption_differences):
            raise TypeError("caption_differences must contain CaptionDifference records")
        for name in (
            "baseline_selected_prompt_ids", "candidate_selected_prompt_ids",
            "baseline_composed_prompt_refs", "candidate_composed_prompt_refs",
            "rollout_artifact_refs", "metadata",
        ):
            _freeze_mapping(self, name)
