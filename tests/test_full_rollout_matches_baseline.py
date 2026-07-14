"""Structural regression: FullRolloutEvaluator run with the baseline prompt
must reproduce the baseline caption view exactly (per-clip texts identical;
subject_registry is allowed to drift, PHASE2_3 §2). Captioning and the DVD
agent are stubbed — the real end-to-end check runs in the Phase 2 sanity
script on GPU."""

import json

import pytest

import surrogate_rollout.evaluation.full_rollout as fr
from surrogate_rollout.captioning.candidate_captions import CandidateCaptionSet
from surrogate_rollout.evaluation.rollout_evaluator import EvaluationRequest, SelectionBudget
from surrogate_rollout.schemas import DVDRunResult, ReferenceSets

CLIPS = ["0_10", "10_20", "20_30"]
PROMPT = "caption TRANSCRIPT_PLACEHOLDER CLIP_START_TIME CLIP_END_TIME baseline"


@pytest.fixture
def baseline_captions(tmp_path):
    captions = {k: {"caption": f"baseline text {k}"} for k in CLIPS}
    captions["subject_registry"] = {"people": ["a"]}
    p = tmp_path / "base" / "captions.json"
    p.parent.mkdir()
    p.write_text(json.dumps(captions, indent=4))
    return p


def _request(tmp_path, baseline_captions):
    return EvaluationRequest(
        video_id="v1", qa_id="videomme/long/0", question="q?",
        answer_options=("A", "B"), ground_truth="A",
        sample={"sample_id": "v1", "video_path": str(tmp_path / "v.mp4"),
                "question": "q?", "answer": "A"},
        baseline_prompt=PROMPT, candidate_prompt=PROMPT,
        baseline_captions_path=str(baseline_captions),
        baseline_trajectory_dir=None,
        reference_policy="returned_and_frame_inspection",
        selection_policy="full_rollout",
        work_root=str(tmp_path / "work"),
        budget=SelectionBudget(), merge_prompt="merge",
    )


def test_baseline_prompt_reproduces_baseline_view(tmp_path, baseline_captions,
                                                  monkeypatch):
    baseline = json.loads(baseline_captions.read_text())

    def stub_caption_clips(*, sample, candidate_prompt, merge_prompt=None,
                           clip_ids=None, cache_root=None):
        assert candidate_prompt == PROMPT
        parsed = {k: {"clip_description": baseline[k]["caption"],
                      "subject_registry": {"clip": k}} for k in CLIPS}
        return CandidateCaptionSet(
            video_id="v1", prompt_hash="a" * 64, ckpt_dir="",
            clips=list(CLIPS), parsed=parsed,
            caption_call_count=0, cache_hits=len(CLIPS))

    captured = {}

    def stub_run_dvd_qa(*, captions_path, sample, run_dir, question_id,
                        database_path=None, gpu=None, **kw):
        captured["captions_path"] = captions_path
        return DVDRunResult(
            question_id=question_id, video_id="v1", prediction="A",
            parsed_answer="A", ground_truth="A", score=1.0, trajectory=[],
            references=ReferenceSets(), total_segments=len(CLIPS),
            token_usage={}, latency_seconds=0.1, caption_cache_tag="t",
            captions_path=captions_path, database_path="db.json")

    monkeypatch.setattr(fr, "caption_clips", stub_caption_clips)
    monkeypatch.setattr(fr, "run_dvd_qa", stub_run_dvd_qa)
    monkeypatch.setattr(fr, "prepare_video_workdir", lambda *a, **k: None)

    evaluator = fr.FullRolloutEvaluator(merge_fn=lambda regs: {"merged": len(regs)})
    result = evaluator.evaluate(_request(tmp_path, baseline_captions))

    produced = json.loads(open(captured["captions_path"]).read())
    for k in CLIPS:  # per-clip texts identical to the captured baseline
        assert produced[k]["caption"] == baseline[k]["caption"]
    assert result.score == 1.0
    assert result.recaption_fraction == 1.0
    assert result.caption_call_count == 0          # cache-served
    assert result.captions_hash


def test_invalid_candidate_prompt_documented_failure(tmp_path, baseline_captions):
    from dataclasses import replace

    req = replace(_request(tmp_path, baseline_captions),
                  candidate_prompt="no placeholders here")
    result = fr.FullRolloutEvaluator(merge_fn=lambda r: None).evaluate(req)
    assert result.score == 0.0
    assert result.errors[0]["type"] == "CandidatePromptError"
