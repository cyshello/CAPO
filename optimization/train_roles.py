"""Deterministic train-role derivation and evidence coverage rotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from surrogate_rollout import config


@dataclass(frozen=True)
class TrainVideoRecord:
    video_id: str
    provider_indices: tuple[int, ...]
    question_ids: tuple[str, ...]
    previously_cached: bool

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("TrainVideoRecord.video_id must be non-empty")
        if len(self.provider_indices) != 3 or len(self.question_ids) != 3:
            raise ValueError("each train video must contain exactly three QAs")
        if len(set(self.provider_indices)) != 3 or len(set(self.question_ids)) != 3:
            raise ValueError("train video QA identifiers must be unique")


@dataclass(frozen=True)
class TrainRoleAssignment:
    evidence_videos: tuple[TrainVideoRecord, ...]
    confirmation_videos: tuple[TrainVideoRecord, ...]
    validation_video_ids: tuple[str, ...]
    test_video_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        evidence = {item.video_id for item in self.evidence_videos}
        confirmation = {item.video_id for item in self.confirmation_videos}
        validation = set(self.validation_video_ids)
        test = set(self.test_video_ids)
        if len(evidence) != 8 or len(confirmation) != 2:
            raise ValueError("train roles require eight evidence and two confirmation videos")
        if evidence & confirmation or evidence & validation or evidence & test or \
                confirmation & validation or confirmation & test or validation & test:
            raise ValueError("video-level roles and top-level splits must be disjoint")


@dataclass(frozen=True)
class EvidenceCoverageState:
    rotation_order: tuple[str, ...]
    used_since_confirmation: tuple[str, ...] = ()
    rotation_cursor: int = 0
    coverage_cycle: int = 0
    iterations_since_confirmation: int = 0
    confirmation_due: bool = False

    def __post_init__(self) -> None:
        if len(self.rotation_order) != 8 or len(set(self.rotation_order)) != 8:
            raise ValueError("coverage rotation requires eight unique evidence videos")
        if not 0 <= self.rotation_cursor < len(self.rotation_order):
            raise ValueError("rotation_cursor is outside the evidence pool")
        if not set(self.used_since_confirmation) <= set(self.rotation_order):
            raise ValueError("coverage contains a video outside rotation_order")
        if len(self.used_since_confirmation) != len(set(self.used_since_confirmation)):
            raise ValueError("coverage contains duplicate video IDs")
        complete = len(self.used_since_confirmation) == len(self.rotation_order)
        if self.confirmation_due != complete:
            raise ValueError("confirmation_due must exactly match complete coverage")


def _video_record(value: Mapping[str, Any]) -> TrainVideoRecord:
    return TrainVideoRecord(
        video_id=str(value["video_id"]),
        provider_indices=tuple(int(item) for item in value["provider_indices"]),
        question_ids=tuple(str(item) for item in value["question_ids"]),
        previously_cached=bool(value.get("previously_cached", False)),
    )


def derive_train_roles(
    manifest: Mapping[str, Any],
    *,
    expected_pinned_video_ids: Sequence[str] = config.PREVIOUSLY_CACHED_VIDEOS,
) -> TrainRoleAssignment:
    """Derive the active 8/2 train roles without changing the manifest."""
    splits = manifest.get("splits") or {}
    train = tuple(_video_record(item) for item in splits.get("train") or ())
    validation = tuple(splits.get("validation") or ())
    test = tuple(splits.get("test") or ())
    if len(train) != 10 or len(validation) != 10 or len(test) != 10:
        raise ValueError("frozen split must contain 10 train/validation/test videos")
    for name, values in (("validation", validation), ("test", test)):
        if sum(len(item.get("question_ids") or ()) for item in values) != 30:
            raise ValueError(f"{name} split must contain exactly 30 QAs")
    pinned = tuple(item for item in train if item.previously_cached)
    confirmation = tuple(item for item in train if not item.previously_cached)
    configured_pins = tuple(str(item) for item in
                            manifest.get("pinned_previously_cached") or ())
    if set(configured_pins) != {item.video_id for item in pinned}:
        raise ValueError("manifest pinned list disagrees with train cached flags")
    if set(configured_pins) != set(expected_pinned_video_ids):
        raise ValueError("split manifest pinned videos disagree with configuration")
    return TrainRoleAssignment(
        evidence_videos=pinned,
        confirmation_videos=confirmation,
        validation_video_ids=tuple(str(item["video_id"]) for item in validation),
        test_video_ids=tuple(str(item["video_id"]) for item in test),
    )


def initial_coverage(roles: TrainRoleAssignment) -> EvidenceCoverageState:
    return EvidenceCoverageState(
        rotation_order=tuple(item.video_id for item in roles.evidence_videos))


def select_evidence_batch(
    state: EvidenceCoverageState, batch_size: int = 3,
) -> tuple[tuple[str, ...], EvidenceCoverageState]:
    if batch_size != 3:
        raise ValueError("the active method requires batch_size=3")
    if state.confirmation_due:
        raise RuntimeError("confirmation is due before another evidence iteration")
    order = state.rotation_order
    cyclic = order[state.rotation_cursor:] + order[:state.rotation_cursor]
    used = set(state.used_since_confirmation)
    selected = [video_id for video_id in cyclic if video_id not in used][:batch_size]
    if len(selected) < batch_size:
        fill = [video_id for video_id in cyclic if video_id not in selected]
        selected.extend(fill[:batch_size - len(selected)])
    if len(selected) != batch_size or len(set(selected)) != batch_size:
        raise RuntimeError("failed to select three unique evidence videos")
    covered = used | set(selected)
    covered_ordered = tuple(video_id for video_id in order if video_id in covered)
    cursor = (order.index(selected[-1]) + 1) % len(order)
    next_state = EvidenceCoverageState(
        rotation_order=order,
        used_since_confirmation=covered_ordered,
        rotation_cursor=cursor,
        coverage_cycle=state.coverage_cycle,
        iterations_since_confirmation=state.iterations_since_confirmation + 1,
        confirmation_due=len(covered) == len(order),
    )
    return tuple(selected), next_state


def reset_after_confirmation(state: EvidenceCoverageState) -> EvidenceCoverageState:
    if not state.confirmation_due:
        raise RuntimeError("coverage cannot reset before confirmation is due")
    return EvidenceCoverageState(
        rotation_order=state.rotation_order,
        rotation_cursor=state.rotation_cursor,
        coverage_cycle=state.coverage_cycle + 1,
    )


def records_for_video_ids(
    roles: TrainRoleAssignment, video_ids: Sequence[str],
) -> tuple[TrainVideoRecord, ...]:
    by_id = {item.video_id: item for item in roles.evidence_videos}
    unknown = set(video_ids) - set(by_id)
    if unknown:
        raise ValueError(f"selected videos are outside evidence pool: {sorted(unknown)}")
    return tuple(by_id[video_id] for video_id in video_ids)
