"""Tests for the official-GEPA adapter over the free-form DVD meta-prompt.

The DVD scoring path (`caption_and_score_video`) is monkeypatched so these run
with no GPU, model, or network: the adapter wiring and a real `gepa.optimize`
drive are exercised end-to-end against a deterministic fake scorer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import surrogate_rollout.gepa_meta_prompt.gepa_adapter as adapter_mod
from surrogate_rollout.gepa_meta_prompt.dvd_single_video_evaluator import (
    GepaQAResult,
    GepaVideoInstance,
    GepaVideoScore,
)
from surrogate_rollout.gepa_meta_prompt.gepa_adapter import (
    META_PROMPT_COMPONENT,
    DVDMetaPromptGEPAAdapter,
)

SEED = "SEED META PROMPT"
BETTER = "BETTER META PROMPT"


def _fake_evaluator():
    # _generator() only constructs VLMFreeFormInstructionGenerator, which does
    # not touch the model until .generate(); a bare vlm placeholder suffices.
    return SimpleNamespace(builder=SimpleNamespace(
        router=SimpleNamespace(vlm=object())))


def _instances(n: int) -> list[GepaVideoInstance]:
    return [GepaVideoInstance(
        video_id=f"vid{i}", provider_indices=(3 * i, 3 * i + 1, 3 * i + 2),
        question_ids=(f"vid{i}/q0", f"vid{i}/q1", f"vid{i}/q2"))
        for i in range(n)]


def _score_table(meta_prompt_text: str, video_id: str) -> float:
    return {SEED: 0.2}.get(meta_prompt_text, 0.9)


def _patched_score(*, evaluator, video, generator, bank, router, scaffold,
                   contract, work_root, qa_cache_root) -> GepaVideoScore:
    # The generator carries the meta-prompt text via its template.
    accuracy = _score_table(generator.template, video.video_id)
    qa = tuple(GepaQAResult(
        question_id=q, provider_index=0,
        is_correct=accuracy >= 0.5, prediction="A", parsed_answer="A",
        ground_truth="A" if accuracy >= 0.5 else "B", score=accuracy,
        errors=(), captions_hash="h_" + video.video_id)
        for q in video.question_ids)
    return GepaVideoScore(
        video_id=video.video_id, accuracy=accuracy, evaluated_qa_count=len(qa),
        qa_results=qa, captions_hash="h_" + video.video_id,
        caption_calls=1, caption_cache_hits=0)


class FakeMutator:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict] = []

    def propose(self, *, parent_text, feedback_blocks, instance_ids) -> str:
        self.calls.append({"parent_text": parent_text,
                           "feedback_blocks": list(feedback_blocks),
                           "instance_ids": list(instance_ids)})
        return self._text


@pytest.fixture
def patched_scorer(monkeypatch):
    monkeypatch.setattr(adapter_mod, "caption_and_score_video", _patched_score)


def _adapter(mutator) -> DVDMetaPromptGEPAAdapter:
    return DVDMetaPromptGEPAAdapter(
        evaluator=_fake_evaluator(), bank=object(), router=object(),
        scaffold=object(), contract=object(),
        generator_model_id="fake-caption-model",
        generator_backend_id="fake.backend", generator_max_tokens=64,
        mutator=mutator, work_root="/tmp/gepa_work",
        qa_cache_root="/tmp/gepa_qa")


def test_evaluate_scores_each_video(patched_scorer):
    adapter = _adapter(FakeMutator(BETTER))
    batch = _instances(3)
    result = adapter.evaluate(batch, {META_PROMPT_COMPONENT: SEED},
                              capture_traces=True)
    assert result.scores == [0.2, 0.2, 0.2]
    assert len(result.trajectories) == 3
    assert "accuracy 0.200" in result.trajectories[0]["feedback"]


def test_evaluate_no_traces_when_not_requested(patched_scorer):
    adapter = _adapter(FakeMutator(BETTER))
    result = adapter.evaluate(_instances(2), {META_PROMPT_COMPONENT: BETTER})
    assert result.scores == [0.9, 0.9]
    assert result.trajectories is None


def test_reflective_dataset_and_propose(patched_scorer):
    mutator = FakeMutator(BETTER)
    adapter = _adapter(mutator)
    batch = _instances(2)
    eval_batch = adapter.evaluate(batch, {META_PROMPT_COMPONENT: SEED},
                                  capture_traces=True)
    dataset = adapter.make_reflective_dataset(
        {META_PROMPT_COMPONENT: SEED}, eval_batch, [META_PROMPT_COMPONENT])
    records = dataset[META_PROMPT_COMPONENT]
    assert len(records) == 2
    assert all("Feedback" in r and "Inputs" in r for r in records)
    proposed = adapter._propose_new_texts(
        {META_PROMPT_COMPONENT: SEED}, dataset, [META_PROMPT_COMPONENT])
    assert proposed == {META_PROMPT_COMPONENT: BETTER}
    assert mutator.calls[0]["parent_text"] == SEED


def test_evaluate_failure_is_zero_not_raised(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("captioner exploded")
    monkeypatch.setattr(adapter_mod, "caption_and_score_video", _boom)
    adapter = _adapter(FakeMutator(BETTER))
    result = adapter.evaluate(_instances(1), {META_PROMPT_COMPONENT: SEED},
                              capture_traces=True)
    assert result.scores == [0.0]
    assert "evaluation failed" in result.trajectories[0]["feedback"]


def test_official_gepa_optimize_drives_adapter(patched_scorer, tmp_path):
    import gepa

    adapter = _adapter(FakeMutator(BETTER))
    trainset = _instances(4)
    result = gepa.optimize(
        seed_candidate={META_PROMPT_COMPONENT: SEED},
        trainset=trainset, valset=trainset, adapter=adapter,
        reflection_minibatch_size=2, max_metric_calls=40,
        display_progress_bar=False, seed=0, run_dir=str(tmp_path / "run"))
    # The engine should discover the strictly better meta-prompt.
    assert result.best_candidate[META_PROMPT_COMPONENT] == BETTER
