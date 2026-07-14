import json

import pytest

from surrogate_rollout.evaluation.rollout_evaluator import (
    FALLBACK_FULL_ROLLOUT,
    FALLBACK_UNSUPPORTED,
    EvaluationRequest,
    EvaluationResult,
    SelectionBudget,
)
from surrogate_rollout.evaluation.selective_rollout import (
    SelectiveSurrogateRolloutEvaluator,
)

CLIPS = [f"{i * 10}_{(i + 1) * 10}" for i in range(6)]

PROMPT = ("caption TRANSCRIPT_PLACEHOLDER from CLIP_START_TIME to CLIP_END_TIME")


@pytest.fixture
def request_fixture(tmp_path):
    captions = {k: {"caption": f"b {k}"} for k in CLIPS}
    captions["subject_registry"] = None
    base = tmp_path / "base" / "captions.json"
    base.parent.mkdir()
    base.write_text(json.dumps(captions))

    traj = tmp_path / "traj"
    traj.mkdir()
    (traj / "trajectory.jsonl").write_text("")   # browse-only run: no references
    (traj / "tool_events.jsonl").write_text("")

    return EvaluationRequest(
        video_id="v1", qa_id="videomme/long/0", question="q?",
        answer_options=("A", "B"), ground_truth="A",
        sample={"sample_id": "v1", "video_path": str(tmp_path / "v.mp4"),
                "question": "q?", "answer": "A"},
        baseline_prompt=PROMPT + " baseline",
        candidate_prompt=PROMPT + " candidate",
        baseline_captions_path=str(base),
        baseline_trajectory_dir=str(traj),
        reference_policy="returned_and_frame_inspection",
        selection_policy="trace_only",
        work_root=str(tmp_path / "work"),
        budget=SelectionBudget(),
        merge_prompt="merge",
    )


def test_zero_reference_unsupported_recorded(request_fixture):
    ev = SelectiveSurrogateRolloutEvaluator(
        zero_reference_fallback=FALLBACK_UNSUPPORTED)
    result = ev.evaluate(request_fixture)
    assert result.fallback_type == FALLBACK_UNSUPPORTED
    assert result.fallback_reason
    assert result.score == 0.0
    assert result.errors


def test_zero_reference_full_rollout_recorded(request_fixture):
    ev = SelectiveSurrogateRolloutEvaluator(
        zero_reference_fallback=FALLBACK_FULL_ROLLOUT)

    class StubFull:
        def evaluate(self, request):
            return EvaluationResult(
                video_id=request.video_id, qa_id=request.qa_id,
                rollout_mode="full", selection_policy="full_rollout",
                answer="A", parsed_answer="A", is_correct=True, score=1.0)

    ev._full = StubFull()
    result = ev.evaluate(request_fixture)
    assert result.fallback_type == FALLBACK_FULL_ROLLOUT
    assert result.fallback_reason
    assert result.rollout_mode == "selective"      # same result interface
    assert result.selection_policy == "trace_only"
    assert result.score == 1.0


def test_missing_trajectory_dir_is_unsupported(request_fixture, tmp_path):
    from dataclasses import replace

    req = replace(request_fixture, baseline_trajectory_dir=None)
    ev = SelectiveSurrogateRolloutEvaluator(
        zero_reference_fallback=FALLBACK_UNSUPPORTED)
    result = ev.evaluate(req)
    assert result.fallback_type == FALLBACK_UNSUPPORTED


def test_invalid_fallback_config_rejected():
    with pytest.raises(ValueError):
        SelectiveSurrogateRolloutEvaluator(zero_reference_fallback="none")
