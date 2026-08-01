"""Fanning the full-recaption caption loop across the GPU pool.

The parallel runner is only allowed to change *when* captions are produced. What
the episode contains -- which segments were captioned, with which composed
prompt and intervention identity, in which order -- has to stay byte-identical
to the sequential runner, because that is the evidence the optimizer learns
from. So the assertions here pair every concurrency claim with an equality
claim against the sequential runner over the same fixture.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _ancestor in (_ROOT, _ROOT.parent):
    if str(_ancestor) not in sys.path:
        sys.path.insert(0, str(_ancestor))

from surrogate_rollout.optimization.fresh_prompt_delta_evidence import (  # noqa: E402
    PromptDeltaExecutionPlan,
    PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
    _qa_segment_selection_records,
    _source_qa_classification_hash,
)
from surrogate_rollout.optimization.schemas import PromptDelta  # noqa: E402

from full_recaption_opt.full_recaption_runner import (  # noqa: E402
    FullRecaptionInterventionRunner,
)
from full_recaption_opt.parallel_intervention_runner import (  # noqa: E402
    MAX_IN_FLIGHT_ENVIRONMENT_VARIABLE,
    ParallelFullRecaptionInterventionRunner,
    _resolve_max_in_flight,
)

from test_fresh_prompt_delta_evidence import _baseline  # noqa: E402


class _ConcurrencyProbeCaptioner:
    """Caption stub that records how many calls were ever in flight at once."""

    configuration_identity = {"captioner": "fixture"}

    def __init__(self, hold_seconds: float = 0.05) -> None:
        self.hold_seconds = hold_seconds
        self.segments: list[str] = []
        self.identities: list[str] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def caption(self, **kwargs):
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self.segments.append(kwargs["segment_id"])
            self.identities.append(kwargs["intervention_identity_hash"])
        try:
            # Long enough that a sequential caller cannot overlap by accident
            # and a parallel one cannot avoid overlapping.
            time.sleep(self.hold_seconds)
        finally:
            with self._lock:
                self._in_flight -= 1
        return SimpleNamespace(parsed={
            "clip_description": f"changed {kwargs['segment_id']}"})


class _Mixed:
    def __init__(self) -> None:
        self.builds = 0

    def build(self, **kwargs):
        self.builds += 1
        selected = sorted(kwargs["selected_clip_ids"])
        root = Path(kwargs["work_root"])
        root.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            captions_path=str(root / "mixed-captions.json"),
            captions_hash="mixed-hash",
            database_path=str(root / "mixed-database.json"),
            selected_clip_ids=selected, replaced_clip_ids=selected)


def _qa_fn(**kwargs):
    qa_dir = Path(kwargs["run_dir"])
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "trajectory.jsonl").write_text(
        '{"content": "done", "role": "assistant"}\n', encoding="utf-8")
    return SimpleNamespace(
        errors=(), prediction=("A" if kwargs["question_id"] == "qa1" else "B"),
        score=(1.0 if kwargs["question_id"] == "qa1" else 0.0))


def _plan(manifest: str) -> PromptDeltaExecutionPlan:
    delta = PromptDelta(
        "delta-qa1", "Describe the visible continuity.", ("qa1",),
        "The source trajectory omitted continuity.")
    return PromptDeltaExecutionPlan(
        prompt_delta=delta, selected_segment_ids=("0_10",),
        selection_policy=PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
        frame_inspection_classification_hash=_source_qa_classification_hash(
            next(row for row in _qa_segment_selection_records(
                manifest, global_inspection_boundary_tolerance_seconds=10)
                if row["qa_id"] == "qa1"), tolerance_seconds=10),
        global_inspection_boundary_tolerance_seconds=10)


def _runner(cls, captioner, mixed, tmp_path: Path):
    return cls(
        segment_captioner=captioner, sample_loader=lambda _index: {},
        merge_prompt="merge", caption_cache_root=str(tmp_path / "cache"),
        caption_cache_manifest_path=str(tmp_path / "cache.jsonl"),
        composition_separator="\n", qa_fn=_qa_fn,
        clip_index_fn=lambda _sample, _video: [
            ("0_10", {}), ("10_20", {}), ("20_30", {})],
        mixed_view_builder=mixed, dvd_max_iterations=1, gpu=None,
        scaffold_contract=None)


def _episode_evidence(episode):
    """Everything about an episode except where on disk it was written."""
    return {
        "clips": [(row.segment_id, row.intervention_caption,
                   row.base_prompt, row.baseline_caption)
                  for row in episode.clips],
        "mixed_view_identity": episode.mixed_view_identity,
        "qa": [(row.qa_id, row.is_source_qa, row.baseline_correct,
                row.intervention_correct) for row in episode.qa_outcomes],
        "video_id": episode.video_id,
        "parent_meta_prompt_id": episode.parent_meta_prompt_id,
    }


@pytest.fixture()
def fixture_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "surrogate_rollout.optimization.fresh_prompt_delta_evidence."
        "validate_composed_text", lambda _text, _contract: ())
    return _baseline(tmp_path / "baseline")


def _run(cls, manifest, tmp_path, name, hold_seconds=0.05):
    captioner = _ConcurrencyProbeCaptioner(hold_seconds)
    mixed = _Mixed()
    episode = _runner(cls, captioner, mixed, tmp_path / name).run(
        baseline_video_manifest_path=manifest,
        parent_meta_prompt_id="meta_parent", plan=_plan(manifest),
        output_directory=str(tmp_path / name / "intervention"))
    return episode, captioner, mixed


def test_parallel_episode_matches_the_sequential_episode(
        fixture_baseline, tmp_path, monkeypatch):
    monkeypatch.setenv(MAX_IN_FLIGHT_ENVIRONMENT_VARIABLE, "4")

    sequential, sequential_captioner, _ = _run(
        FullRecaptionInterventionRunner, fixture_baseline, tmp_path, "seq")
    parallel, parallel_captioner, _ = _run(
        ParallelFullRecaptionInterventionRunner, fixture_baseline, tmp_path,
        "par")

    assert _episode_evidence(parallel) == _episode_evidence(sequential)
    # Full recaption: the whole video, in start-time order, not the plan's
    # localized subset -- and the parallel pass must not reorder the episode.
    assert [row.segment_id for row in parallel.clips] == ["0_10", "10_20"]
    # The intervention identity hash is what keys the caption cache; a probe
    # pass that recomputed it differently would silently redo every caption.
    assert sorted(parallel_captioner.identities) == sorted(
        sequential_captioner.identities)


def test_captions_actually_overlap_and_the_sequential_runner_does_not(
        fixture_baseline, tmp_path, monkeypatch):
    monkeypatch.setenv(MAX_IN_FLIGHT_ENVIRONMENT_VARIABLE, "4")

    _, sequential_captioner, _ = _run(
        FullRecaptionInterventionRunner, fixture_baseline, tmp_path, "seq")
    _, parallel_captioner, _ = _run(
        ParallelFullRecaptionInterventionRunner, fixture_baseline, tmp_path,
        "par")

    assert sequential_captioner.max_in_flight == 1
    assert parallel_captioner.max_in_flight == 2  # the fixture's two segments


def test_the_probe_pass_captions_nothing_and_builds_no_mixed_view(
        fixture_baseline, tmp_path, monkeypatch):
    monkeypatch.setenv(MAX_IN_FLIGHT_ENVIRONMENT_VARIABLE, "4")

    _, captioner, mixed = _run(
        ParallelFullRecaptionInterventionRunner, fixture_baseline, tmp_path,
        "par")

    # Two segments, captioned once each: the probe recorded without captioning
    # and the replay was served from the memo, so neither doubled the cost.
    assert captioner.segments == ["0_10", "10_20"]
    assert mixed.builds == 1


def test_a_saved_episode_still_short_circuits_before_any_captioning(
        fixture_baseline, tmp_path, monkeypatch):
    monkeypatch.setenv(MAX_IN_FLIGHT_ENVIRONMENT_VARIABLE, "4")
    manifest = fixture_baseline

    first, _, _ = _run(
        ParallelFullRecaptionInterventionRunner, manifest, tmp_path, "par")

    captioner = _ConcurrencyProbeCaptioner()
    mixed = _Mixed()
    resumed = _runner(
        ParallelFullRecaptionInterventionRunner, captioner, mixed,
        tmp_path / "par").run(
            baseline_video_manifest_path=manifest,
            parent_meta_prompt_id="meta_parent", plan=_plan(manifest),
            output_directory=str(tmp_path / "par" / "intervention"))

    assert _episode_evidence(resumed) == _episode_evidence(first)
    assert captioner.segments == []
    assert mixed.builds == 0


def test_a_caption_failure_still_fails_the_run(
        fixture_baseline, tmp_path, monkeypatch):
    monkeypatch.setenv(MAX_IN_FLIGHT_ENVIRONMENT_VARIABLE, "4")

    class _Failing(_ConcurrencyProbeCaptioner):
        def caption(self, **kwargs):
            if kwargs["segment_id"] == "10_20":
                raise RuntimeError("caption backend died")
            return super().caption(**kwargs)

    runner = _runner(
        ParallelFullRecaptionInterventionRunner, _Failing(), _Mixed(),
        tmp_path / "fail")
    with pytest.raises(RuntimeError, match="caption backend died"):
        runner.run(
            baseline_video_manifest_path=fixture_baseline,
            parent_meta_prompt_id="meta_parent", plan=_plan(fixture_baseline),
            output_directory=str(tmp_path / "fail" / "intervention"))


def test_concurrency_defaults_to_the_gpu_pool_size():
    pooled = SimpleNamespace(
        remote_dispatcher=SimpleNamespace(
            parallel_gpus=("0", "1", "2", "4", "5", "6", "7")))
    assert _resolve_max_in_flight(pooled) == 7


def test_a_captioner_without_a_pool_stays_sequential():
    assert _resolve_max_in_flight(SimpleNamespace()) == 1
    assert _resolve_max_in_flight(
        SimpleNamespace(remote_dispatcher=SimpleNamespace(
            parallel_gpus=()))) == 1


def test_the_environment_override_wins(monkeypatch):
    pooled = SimpleNamespace(
        remote_dispatcher=SimpleNamespace(parallel_gpus=("0", "1")))
    monkeypatch.setenv(MAX_IN_FLIGHT_ENVIRONMENT_VARIABLE, "5")
    assert _resolve_max_in_flight(pooled) == 5
    monkeypatch.setenv(MAX_IN_FLIGHT_ENVIRONMENT_VARIABLE, "0")
    with pytest.raises(ValueError, match=MAX_IN_FLIGHT_ENVIRONMENT_VARIABLE):
        _resolve_max_in_flight(pooled)
