"""Checkpoint C deterministic episode-feedback boundary tests."""

import dataclasses
import json

import pytest

from surrogate_rollout.optimization.episode_feedback import (
    DETERMINISTIC_CONFIDENCE_MARKER,
    MOCK_EPISODE_FEEDBACK_POLICY_VERSION,
    MOCK_GENERATOR_DIAGNOSIS,
    MOCK_RECOMMENDED_STRATEGY_CHANGE,
    DeterministicMockEpisodeFeedbackGenerator,
    EpisodeFeedbackGenerationError,
    EpisodeFeedbackGenerator,
    evaluate_episode_feedback_eligibility,
)
from surrogate_rollout.optimization.schemas import (
    EpisodeFeedback,
    InterventionClipRecord,
    InterventionEpisode,
    PromptDelta,
    QAInterventionOutcome,
    episode_feedback_from_json,
    validate_episode_feedback,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json


def episode_fixture(*, episode_id: str = "episode-001") -> InterventionEpisode:
    delta = PromptDelta(
        delta_id="delta-001",
        instruction="Describe the visible object transfer.",
        source_qa_ids=("qa-source",),
        proposer_diagnosis="The baseline omitted the transfer.",
    )
    clips = (
        InterventionClipRecord(
            segment_id="0_10",
            time_range={"start_seconds": 0.0, "end_seconds": 10.0},
            history_snapshot={"history": []},
            base_prompt="Base prompt zero.",
            prompt_delta=delta,
            baseline_caption="A person holds an object.",
            intervention_caption="A person passes an object.",
        ),
        InterventionClipRecord(
            segment_id="10_20",
            time_range={"start_seconds": 10.0, "end_seconds": 20.0},
            history_snapshot={"history": [{"segment_id": "0_10"}]},
            base_prompt="Base prompt one.",
            prompt_delta=delta,
            baseline_caption="The person walks away.",
            intervention_caption="The person walks away.",
        ),
    )
    correctness = (
        ("qa-source", True, False, True),
        ("qa-regression", False, True, False),
        ("qa-stable-correct", False, True, True),
        ("qa-stable-wrong", False, False, False),
    )
    outcomes = tuple(QAInterventionOutcome(
        qa_id=qa_id,
        is_source_qa=is_source,
        baseline_answer="baseline answer",
        intervention_answer="intervention answer",
        baseline_correct=baseline,
        intervention_correct=intervention,
        baseline_trajectory_ref=(None if qa_id == "qa-stable-wrong"
                                 else f"baseline/{qa_id}.jsonl"),
        intervention_trajectory_ref=(None if qa_id == "qa-stable-wrong"
                                     else f"intervention/{qa_id}.jsonl"),
    ) for qa_id, is_source, baseline, intervention in correctness)
    return InterventionEpisode(
        episode_id=episode_id,
        video_id="video-001",
        parent_meta_prompt_id="meta-parent-001",
        prompt_delta=delta,
        clips=clips,
        qa_outcomes=outcomes,
        baseline_run_ref="baseline/run.json",
        intervention_run_ref="intervention/run.json",
    )


def generate(episode: InterventionEpisode) -> EpisodeFeedback:
    generator: EpisodeFeedbackGenerator = \
        DeterministicMockEpisodeFeedbackGenerator()
    return generator.generate(episode)


def test_synthetic_episode_generates_valid_feedback_with_exact_counts():
    episode = episode_fixture()
    feedback = generate(episode)

    validate_episode_feedback(feedback, episode)
    assert feedback.episode_id == episode.episode_id
    assert feedback.outcome_summary == (
        "The episode contains 4 QA outcomes: 1 wrong_to_correct, "
        "1 correct_to_wrong, 1 correct_to_correct, and 1 wrong_to_wrong; "
        "1 source QA and 3 sibling QA."
    )
    assert feedback.confidence == DETERMINISTIC_CONFIDENCE_MARKER


def test_changed_caption_observation_excludes_unchanged_clips():
    feedback = generate(episode_fixture())
    caption_observation = feedback.observations[0]
    assert caption_observation.evidence_type == "caption_change"
    assert caption_observation.supporting_segment_ids == ("0_10",)
    assert caption_observation.supporting_qa_ids == ()
    assert "1 clips" in caption_observation.statement


def test_qa_observations_split_every_transition_with_actual_qa_id():
    episode = episode_fixture()
    observations = {
        item.transition_type: item for item in generate(episode).observations
        if item.evidence_type == "qa_transition"
    }
    assert set(observations) == {
        "wrong_to_correct", "correct_to_wrong",
        "correct_to_correct", "wrong_to_wrong",
    }
    assert observations["wrong_to_correct"].supporting_qa_ids == ("qa-source",)
    assert observations["correct_to_wrong"].supporting_qa_ids == (
        "qa-regression",)
    assert observations["correct_to_correct"].supporting_qa_ids == (
        "qa-stable-correct",)
    assert observations["wrong_to_wrong"].supporting_qa_ids == (
        "qa-stable-wrong",)
    assert all(item.supporting_segment_ids == ()
               for item in observations.values())


def test_correct_to_wrong_counterevidence_contains_only_regressed_qas():
    feedback = generate(episode_fixture())
    assert len(feedback.counterevidence) == 1
    item = feedback.counterevidence[0]
    assert item.supporting_segment_ids == ()
    assert item.supporting_qa_ids == ("qa-regression",)
    assert item.evidence_type == "qa_transition"
    assert item.transition_type == "correct_to_wrong"


def test_no_correct_to_wrong_produces_no_placeholder_counterevidence():
    episode = episode_fixture()
    outcomes = tuple(
        dataclasses.replace(item, intervention_correct=True)
        if item.qa_id == "qa-regression" else item
        for item in episode.qa_outcomes
    )
    feedback = generate(dataclasses.replace(episode, qa_outcomes=outcomes))
    assert feedback.counterevidence == ()


def test_feedback_id_uses_only_explicit_stable_identity_and_changes_by_episode():
    first = generate(episode_fixture())
    identity = {
        "episode_id": "episode-001",
        "policy_version": MOCK_EPISODE_FEEDBACK_POLICY_VERSION,
    }
    assert first.feedback_id == f"mock_feedback_{sha256_json(identity)[:20]}"
    assert generate(episode_fixture(episode_id="episode-002")).feedback_id != \
        first.feedback_id


def test_repeated_generation_is_canonical_and_does_not_mutate_input():
    episode = episode_fixture()
    before = dumps_canonical(episode)
    first = generate(episode)
    second = generate(episode)
    assert dumps_canonical(first) == dumps_canonical(second)
    assert dumps_canonical(episode) == before


def test_delta_instruction_mismatch_fails_fast():
    episode = episode_fixture()
    conflicting_delta = dataclasses.replace(
        episode.prompt_delta, instruction="A conflicting clip instruction.")
    clips = (dataclasses.replace(
        episode.clips[0], prompt_delta=conflicting_delta), *episode.clips[1:])
    with pytest.raises(EpisodeFeedbackGenerationError, match="differs"):
        generate(dataclasses.replace(episode, clips=clips))


def test_unavailable_correctness_fails_instead_of_inventing_transition():
    episode = episode_fixture()
    outcomes = (dataclasses.replace(
        episode.qa_outcomes[0], baseline_correct=None), *episode.qa_outcomes[1:])
    with pytest.raises(EpisodeFeedbackGenerationError, match="unavailable"):
        generate(dataclasses.replace(episode, qa_outcomes=outcomes))


def test_mock_text_is_explicitly_nonsemantic_and_has_no_clip_qa_attribution():
    episode = episode_fixture()
    feedback = generate(episode)
    assert feedback.generator_diagnosis == MOCK_GENERATOR_DIAGNOSIS
    assert feedback.recommended_strategy_change == \
        MOCK_RECOMMENDED_STRATEGY_CHANGE
    assert episode.prompt_delta.instruction not in \
        feedback.recommended_strategy_change
    assert not {field.name for field in dataclasses.fields(
        InterventionClipRecord)}.intersection({
            "qa_id", "qa_ids", "qa_transition", "qa_outcome", "qa_outcomes"})
    assert all("caused" not in item.statement.lower()
               for item in (*feedback.observations, *feedback.counterevidence))


def test_episode_feedback_json_round_trip():
    feedback = generate(episode_fixture())
    restored = episode_feedback_from_json(json.loads(dumps_canonical(feedback)))
    assert restored == feedback
    assert dumps_canonical(restored) == dumps_canonical(feedback)


def test_eligibility_does_not_apply_model_output_semantic_rules():
    episode = episode_fixture()
    feedback = generate(episode)
    report = evaluate_episode_feedback_eligibility(feedback, episode)
    assert report.eligible is True
    assert report.reasons == ()
    assert report.feedback_sha256 == sha256_json(
        json.loads(dumps_canonical(feedback)))

    empty = dataclasses.replace(feedback, generator_diagnosis="   ")
    report = evaluate_episode_feedback_eligibility(empty, episode)
    assert report.eligible is True
    assert report.reasons == ()


def test_cross_record_validator_does_not_judge_transition_semantics():
    episode = episode_fixture()
    feedback = generate(episode)
    source = next(item for item in feedback.observations
                  if item.transition_type == "wrong_to_correct")
    without_transition = dataclasses.replace(source, transition_type=None)
    validate_episode_feedback(
        dataclasses.replace(
            feedback,
            observations=tuple(
                dataclasses.replace(item, transition_type="wrong_to_wrong")
                if item is source else item
                for item in feedback.observations)),
        episode)
    assert without_transition.transition_type is None
