"""Checkpoint F provider-independent orchestration and atomic state tests."""

import dataclasses
import json

import pytest
from pathlib import Path

from surrogate_rollout.optimization.episode_feedback import (
    DeterministicMockEpisodeFeedbackGenerator,
)
from surrogate_rollout.optimization.meta_prompt_updater import (
    DeterministicMockMetaPromptUpdater,
)
from surrogate_rollout.optimization.prompt_delta_iteration import (
    DeterministicMockMetaPromptConfirmationEvaluator,
    MetaPromptConfirmationCase,
    MetaPromptConfirmationCriterion,
    MetaPromptConfirmationOutcome,
    PromptDeltaIterationOrchestrator,
    build_feedback_grounding,
)
from surrogate_rollout.optimization.feedback_memory import (
    load_parent_feedback_memory_bank,
)
from surrogate_rollout.optimization.schemas import MetaPromptVersion
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from test_episode_feedback import episode_fixture


CANDIDATE = (
    "Inspect current frames and bounded caption history for visible continuity "
    "before generating a concise segment instruction."
)


def parent():
    return MetaPromptVersion(
        meta_prompt_id="meta-parent", parent_meta_prompt_id=None,
        text="Inspect frames and bounded history.",
        created_at="2026-07-20T00:00:00Z", status="parent")


def cases():
    return (
        MetaPromptConfirmationCase(
            "confirmation-1", "holdout-video-1", "holdout-qa-1", "bundle/1"),
        MetaPromptConfirmationCase(
            "confirmation-2", "holdout-video-2", "holdout-qa-2", "bundle/2"),
    )


def criterion(*, delta=0.0, regressions=0):
    return MetaPromptConfirmationCriterion(
        minimum_sample_count=2, minimum_accuracy_delta=delta,
        maximum_correct_to_wrong=regressions,
        require_no_execution_failures=True)


class CountingUpdater:
    def __init__(self, candidate):
        self.inner = DeterministicMockMetaPromptUpdater(
            candidate_meta_prompt=candidate)
        self.calls = 0
        self.grounding = None

    def update(self, parent, feedbacks, *, feedback_grounding=None,
               feedback_memories=None, historical_memories=None,
               current_iteration_id=None,
               historical_memory_character_budget=None):
        self.calls += 1
        self.grounding = feedback_grounding
        self.memories = feedback_memories
        self.historical_memories = historical_memories
        return self.inner.update(
            parent, feedbacks, feedback_grounding=feedback_grounding,
            feedback_memories=feedback_memories,
            historical_memories=historical_memories,
            current_iteration_id=current_iteration_id,
            historical_memory_character_budget=(
                historical_memory_character_budget))


class CountingFeedback:
    def __init__(self):
        self.inner = DeterministicMockEpisodeFeedbackGenerator()
        self.calls = 0

    def generate(self, episode):
        self.calls += 1
        return self.inner.generate(episode)


def outcomes(*, accepted=True, noop_flip=False):
    if accepted:
        first = MetaPromptConfirmationOutcome(
            "confirmation-1", "holdout-video-1", "holdout-qa-1",
            False, True, False)
    else:
        first = MetaPromptConfirmationOutcome(
            "confirmation-1", "holdout-video-1", "holdout-qa-1",
            True, False, False)
    second = MetaPromptConfirmationOutcome(
        "confirmation-2", "holdout-video-2", "holdout-qa-2",
        False, True if noop_flip else False, True)
    return (first, second)


def run(tmp_path, *, candidate=CANDIDATE, accepted=True, noop_flip=False,
        output_name="run", updater=None, feedback=None, evaluator=None):
    updater = updater or CountingUpdater(candidate)
    feedback = feedback or CountingFeedback()
    evaluator = evaluator or DeterministicMockMetaPromptConfirmationEvaluator(
        outcomes(accepted=accepted, noop_flip=noop_flip))
    orchestrator = PromptDeltaIterationOrchestrator(
        feedback_generator=feedback, updater=updater,
        confirmation_evaluator=evaluator)
    result = orchestrator.run(
        iteration_id="iteration-1", parent=parent(),
        update_episodes=(episode_fixture(),), confirmation_cases=cases(),
        criterion=criterion(), model_identity="fixture-caption-qa-model",
        decoding_settings={"temperature": 0.0},
        cache_reset_identity="fresh-paired-cache-v1",
        evaluation_pipeline_identity="fixture-paired-evaluator-v1",
        candidate_created_at="2026-07-20T01:00:00Z",
        output_directory=str(tmp_path / output_name),
        state_directory=str(tmp_path / "state"),
        feedback_memory_bank_directory=str(tmp_path / "memory_bank"),
        initialize_parent_pointer=True)
    return result, updater, feedback, evaluator


def test_no_update_stops_without_confirmation_and_preserves_parent_pointer(tmp_path):
    result, updater, feedback, evaluator = run(tmp_path, candidate=None)
    assert result.status == "no_update"
    assert updater.calls == feedback.calls == 1
    assert evaluator.call_count == 0
    pointer = json.loads((tmp_path / "state/current_meta_prompt.json").read_text())
    assert pointer["active_meta_prompt_id"] == "meta-parent"
    assert not (tmp_path / "run/provisional_meta_prompt.json").exists()
    assert updater.historical_memories == ()
    request = json.loads((tmp_path / "run/updater_result.json").read_text())[
        "request"]
    assert len(request["current_iteration_feedback"]) == 1
    assert request["historical_experience"]["memories"] == []
    assert "compact_memory_text" not in request[
        "current_iteration_feedback"][0]
    assert "feedback_id" not in request["current_iteration_feedback"][0]
    assert "episode_id" not in request["current_iteration_feedback"][0]
    bank = load_parent_feedback_memory_bank(
        str(tmp_path / "memory_bank"), "meta-parent")
    assert len(bank) == 1
    assert bank[0].iteration_id == "iteration-1"


def test_update_writes_provisional_then_confirmation_pass_promotes(tmp_path):
    result, updater, _, evaluator = run(tmp_path, accepted=True)
    assert result.status == "promoted"
    assert updater.calls == evaluator.call_count == 1
    provisional = json.loads(
        (tmp_path / "run/provisional_meta_prompt.json").read_text())
    assert provisional["status"] == "provisional"
    pointer = json.loads((tmp_path / "state/current_meta_prompt.json").read_text())
    assert pointer["active_meta_prompt_id"] == provisional["meta_prompt_id"]
    confirmed = json.loads(open(pointer["artifact_path"]).read())
    assert confirmed["status"] == "confirmed"
    assert confirmed["parent_meta_prompt_id"] == "meta-parent"
    final = json.loads((tmp_path / "run/iteration_result.json").read_text())
    assert final["archived_parent_feedback_memory_bank"] is not None
    assert load_parent_feedback_memory_bank(
        str(tmp_path / "memory_bank"), "meta-parent")
    assert load_parent_feedback_memory_bank(
        str(tmp_path / "memory_bank"), provisional["meta_prompt_id"]) == ()


def test_confirmation_failure_rolls_back_without_pointer_change(tmp_path):
    result, _, _, _ = run(tmp_path, accepted=False)
    assert result.status == "rolled_back"
    pointer = json.loads((tmp_path / "state/current_meta_prompt.json").read_text())
    assert pointer["active_meta_prompt_id"] == "meta-parent"
    rejected = json.loads((tmp_path / "run/rejected_meta_prompt.json").read_text())
    assert rejected["status"] == "rejected"
    assert len(load_parent_feedback_memory_bank(
        str(tmp_path / "memory_bank"), "meta-parent")) == 1


def test_identical_caption_qa_flip_is_uncertain_noop_not_benefit_or_harm(tmp_path):
    result, _, _, _ = run(tmp_path, accepted=True, noop_flip=True)
    decision = json.loads((tmp_path / "run/confirmation/promotion_decision.json").read_text())
    assert decision["uncertain_noop_qa_ids"] == ["holdout-qa-2"]
    assert decision["attributable_correct_to_wrong_qa_ids"] == []
    assert decision["parent_accuracy"] == 0.0
    assert decision["candidate_accuracy"] == 0.5
    assert result.status == "promoted"


def test_feedback_grounding_marks_unchanged_improvement_as_positive_signal():
    episode = episode_fixture()
    unchanged = dataclasses.replace(
        episode,
        clips=tuple(dataclasses.replace(
            clip, intervention_caption=clip.baseline_caption)
            for clip in episode.clips))
    feedback = DeterministicMockEpisodeFeedbackGenerator().generate(unchanged)
    grounding = build_feedback_grounding(feedback, unchanged)
    assert grounding["caption_change_status"] == "unchanged"
    assert grounding["changed_segment_ids"] == []
    assert grounding["qa_flip_attribution"] == \
        "positive_episode_signal_without_caption_change"


def test_next_iteration_uses_previous_parent_memory_not_current_memory(tmp_path):
    first, _, _, _ = run(
        tmp_path, candidate=None, output_name="iteration-one")
    assert first.status == "no_update"
    original = episode_fixture()
    second_episode = dataclasses.replace(
        original, episode_id="episode-002", video_id="video-002",
        prompt_delta=dataclasses.replace(
            original.prompt_delta, delta_id="delta-002"))
    updater = CountingUpdater(None)
    feedback = CountingFeedback()
    result = PromptDeltaIterationOrchestrator(
        feedback_generator=feedback, updater=updater,
        confirmation_evaluator=DeterministicMockMetaPromptConfirmationEvaluator(
            ())).run(
        iteration_id="iteration-2", parent=parent(),
        update_episodes=(second_episode,), confirmation_cases=cases(),
        criterion=criterion(), model_identity="model",
        decoding_settings={"temperature": 0}, cache_reset_identity="reset",
        evaluation_pipeline_identity="pipeline", candidate_created_at="now",
        output_directory=str(tmp_path / "iteration-two"),
        state_directory=str(tmp_path / "state"),
        feedback_memory_bank_directory=str(tmp_path / "memory_bank"),
        initialize_parent_pointer=True)
    assert result.status == "no_update"
    assert len(updater.historical_memories) == 1
    request = json.loads(
        (tmp_path / "iteration-two/updater_result.json").read_text())["request"]
    memories = request["historical_experience"]["memories"]
    assert len(memories) == 1
    assert set(memories[0]) == {"memory_text"}
    assert "iteration-1" not in dumps_canonical(request)
    assert "iteration-2" not in dumps_canonical(request)
    assert len(load_parent_feedback_memory_bank(
        str(tmp_path / "memory_bank"), "meta-parent")) == 2


def test_parent_candidate_pairing_and_update_confirmation_overlap_rejected(tmp_path):
    overlap = (MetaPromptConfirmationCase(
        "overlap", "video-001", "holdout-qa", "bundle/overlap"),)
    orchestrator = PromptDeltaIterationOrchestrator(
        feedback_generator=CountingFeedback(),
        updater=CountingUpdater(CANDIDATE),
        confirmation_evaluator=DeterministicMockMetaPromptConfirmationEvaluator(()))
    with pytest.raises(ValueError, match="overlaps"):
        orchestrator.run(
            iteration_id="iteration-overlap", parent=parent(),
            update_episodes=(episode_fixture(),), confirmation_cases=overlap,
            criterion=MetaPromptConfirmationCriterion(1, 0.0, 0, True),
            model_identity="model", decoding_settings={"temperature": 0},
            cache_reset_identity="reset", evaluation_pipeline_identity="pipeline",
            candidate_created_at="now", output_directory=str(tmp_path / "out"),
            state_directory=str(tmp_path / "state"),
            feedback_memory_bank_directory=str(tmp_path / "memory_bank"),
            initialize_parent_pointer=True)


def test_completed_resume_skips_feedback_updater_and_evaluation(tmp_path):
    result, updater, feedback, evaluator = run(tmp_path)
    assert result.status == "promoted"
    feedback.calls = updater.calls = evaluator.call_count = 0
    resumed, _, _, _ = run(
        tmp_path, updater=updater, feedback=feedback, evaluator=evaluator)
    assert resumed.resumed is True
    assert feedback.calls == updater.calls == evaluator.call_count == 0


def test_resume_after_atomic_promotion_before_final_manifest_reuses_all_stages(
        tmp_path):
    _, updater, feedback, evaluator = run(tmp_path)
    (tmp_path / "run/iteration_result.json").unlink()
    feedback.calls = updater.calls = evaluator.call_count = 0
    resumed, _, _, _ = run(
        tmp_path, updater=updater, feedback=feedback, evaluator=evaluator)
    assert resumed.status == "promoted"
    assert feedback.calls == updater.calls == evaluator.call_count == 0


def test_source_episode_is_not_mutated(tmp_path):
    episode = episode_fixture()
    before = dumps_canonical(episode)
    orchestrator = PromptDeltaIterationOrchestrator(
        feedback_generator=CountingFeedback(), updater=CountingUpdater(None),
        confirmation_evaluator=DeterministicMockMetaPromptConfirmationEvaluator(()))
    orchestrator.run(
        iteration_id="immutable", parent=parent(), update_episodes=(episode,),
        confirmation_cases=cases(), criterion=criterion(),
        model_identity="model", decoding_settings={"temperature": 0},
        cache_reset_identity="reset", evaluation_pipeline_identity="pipeline",
        candidate_created_at="now", output_directory=str(tmp_path / "immutable"),
        state_directory=str(tmp_path / "immutable_state"),
        feedback_memory_bank_directory=str(tmp_path / "memory_bank"),
        initialize_parent_pointer=True)
    assert dumps_canonical(episode) == before


def test_cli_dry_run_executes_complete_mock_path(tmp_path):
    from scripts.run_prompt_delta_iteration import main

    fixtures = Path(__file__).parent / "fixtures/prompt_delta_iteration"
    code = main([
        "--iteration-id", "cli-dry-run",
        "--update-episode", str(fixtures / "update_episode.json"),
        "--confirmation-cases", str(fixtures / "confirmation_cases.json"),
        "--output-dir", str(tmp_path / "output"),
        "--state-dir", str(tmp_path / "state"),
        "--feedback-memory-bank-dir", str(tmp_path / "memory_bank"),
        "--candidate-created-at", "2026-07-20T02:00:00Z",
        "--model-identity", "fixture-model-not-called",
        "--decoding-settings", str(fixtures / "decoding_settings.json"),
        "--cache-reset-identity", "fixture-clean-cache-v1",
        "--evaluation-pipeline-identity", "fixture-paired-pipeline-v1",
        "--minimum-confirmation-samples", "2",
        "--minimum-accuracy-delta", "0.0",
        "--maximum-correct-to-wrong", "0",
        "--require-no-execution-failures", "true",
        "--initialize-parent-pointer", "--dry-run",
        "--mock-candidate-meta-prompt", CANDIDATE,
        "--mock-confirmation-outcomes",
        str(fixtures / "mock_confirmation_outcomes.json"),
    ])
    assert code == 0
    result = json.loads((tmp_path / "output/iteration_result.json").read_text())
    assert result["status"] == "promoted"
    pointer = json.loads((tmp_path / "state/current_meta_prompt.json").read_text())
    assert pointer["active_meta_prompt_id"] == \
        result["active_meta_prompt_id"]
    identity = json.loads((tmp_path / "output/input_identity.json").read_text())
    # the repository-owned parent (optimization/prompts/init_meta_prompt.json)
    assert identity["parent"]["meta_prompt_id"] == \
        "meta_prompt_4e7ca02d27e84339e6e5"
