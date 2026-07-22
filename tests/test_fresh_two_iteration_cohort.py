from pathlib import Path

import pytest

from surrogate_rollout.scripts.prepare_fresh_prompt_delta_iteration import (
    _resolve_video_records,
    _video_id_list,
)


class _Provider:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def test_explicit_cohort_resolution_preserves_video_and_qa_order(tmp_path):
    path = tmp_path / "videos.txt"
    path.write_text("v2\nv1\n", encoding="utf-8")
    provider = _Provider([
        {"sample_id": f"s{index}", "extra": {"videoID": video_id}}
        for index, video_id in enumerate(("v1", "v2", "v1", "v2", "v1", "v2"))
    ])
    records = _resolve_video_records(
        video_ids=_video_id_list(path), provider=provider,
        benchmark="videomme", benchmark_split="long")
    assert [item["video_id"] for item in records] == ["v2", "v1"]
    assert records[0]["provider_indices"] == (1, 3, 5)
    assert records[0]["question_ids"] == (
        "videomme/long/1", "videomme/long/3", "videomme/long/5")


def test_explicit_cohort_requires_unique_ids_and_three_qas(tmp_path):
    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text("v1\nv1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty and unique"):
        _video_id_list(duplicate)
    with pytest.raises(ValueError, match="exactly three QAs"):
        _resolve_video_records(
            video_ids=("v1",),
            provider=_Provider([{"extra": {"videoID": "v1"}}]),
            benchmark="videomme", benchmark_split="long")


def test_frozen_ten_and_five_video_cohorts_are_disjoint():
    root = Path(__file__).parents[1]
    evidence = _video_id_list(root / "train_set/10samples.txt")
    confirmation = _video_id_list(
        root / "train_set/confirmation_5samples.txt")
    full_confirmation = _video_id_list(root / "train_set/confirmation.txt")
    assert len(evidence) == 10
    assert len(confirmation) == 5
    assert confirmation == full_confirmation[:5]
    assert set(evidence).isdisjoint(confirmation)


def test_frozen_twenty_and_ten_video_cohorts_are_disjoint():
    root = Path(__file__).parents[1]
    evidence = _video_id_list(root / "train_set/20samples.txt")
    confirmation = _video_id_list(
        root / "train_set/confirmation_10samples.txt")
    full_confirmation = _video_id_list(root / "train_set/confirmation.txt")
    assert len(evidence) == 20
    assert len(confirmation) == 10
    assert confirmation == full_confirmation[:10]
    assert set(evidence).isdisjoint(confirmation)


def test_two_iteration_launcher_uses_configurable_stable_rotation():
    root = Path(__file__).parents[1]
    script = (root / "scripts/run_prompt_delta_two_iteration_10video_pool.sh").read_text()
    assert "PROMPT_DELTA_VIDEOS_PER_ITERATION:-5" in script
    assert "(ordinal - 1) * VIDEOS_PER_ITERATION" in script
    assert 'selected=("${EVIDENCE_IDS[@]:offset:VIDEOS_PER_ITERATION}")' in script
    assert "PROMPT_DELTA_TWO_ITERATION_GPUS" in script
    assert "confirmation_5samples.txt" in script
    evidence = _video_id_list(root / "train_set/10samples.txt")
    batch_size = 5
    first = evidence[:batch_size]
    second = evidence[batch_size:2 * batch_size]
    assert len(first) == len(second) == 5
    assert first + second == evidence
    assert set(first).isdisjoint(second)


def test_single_iteration_launcher_does_not_hardcode_explicit_batch_size():
    root = Path(__file__).parents[1]
    launcher = (root / "scripts/run_fresh_prompt_delta_iteration.sh").read_text()
    preparation = (
        root / "scripts/prepare_fresh_prompt_delta_iteration.py").read_text()
    assert "SELECTED_VIDEO_COUNT * 3" in launcher
    assert "must contain exactly 3 videos" not in launcher
    assert "active repository method requires exactly 3 videos" not in preparation


def test_four_iteration_preset_uses_configurable_five_video_batches():
    root = Path(__file__).parents[1]
    script = (
        root / "scripts/run_prompt_delta_four_iteration_20video_pool.sh"
    ).read_text()
    assert "PROMPT_DELTA_ITERATION_COUNT:-4" in script
    assert "PROMPT_DELTA_VIDEOS_PER_ITERATION:-5" in script
    assert "train_set/20samples.txt" in script
    assert "train_set/confirmation_10samples.txt" in script
    generic = (
        root / "scripts/run_prompt_delta_two_iteration_10video_pool.sh"
    ).read_text()
    assert 'for ordinal in $(seq 1 "$ITERATION_COUNT")' in generic
    assert 'heldout_evaluation:"not_run"' in generic
    assert "completed_paired_confirmation" in generic
