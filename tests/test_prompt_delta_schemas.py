"""Checkpoint A contracts for the parallel prompt-delta optimization path.

These tests are schema-only: they perform no persistence writes, model calls,
caption generation, QA inference, or legacy property-codebook mutation.
"""

import dataclasses
import json

import pytest

from surrogate_rollout.optimization.schemas import (
    EpisodeFeedback,
    EpisodeFeedbackEvidence,
    InterventionClipRecord,
    InterventionEpisode,
    MetaPromptVersion,
    PromptDelta,
    QAInterventionOutcome,
    episode_feedback_from_json,
    intervention_clip_record_from_json,
    intervention_episode_from_json,
    meta_prompt_version_from_json,
    prompt_delta_from_json,
    qa_intervention_outcome_from_json,
    validate_episode_feedback,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


def delta() -> PromptDelta:
    return PromptDelta(
        delta_id="delta-001",
        instruction="Describe the visible handoff and preserve temporal order.",
        source_qa_ids=("qa-1",),
        proposer_diagnosis="The baseline caption omitted the handoff.",
    )


def clip() -> InterventionClipRecord:
    # No repository type owns exactly these two meanings.  The contract keeps
    # both existing JSON payload shapes opaque and lossless.
    return InterventionClipRecord(
        segment_id="10_20",
        time_range={"start_seconds": 10.0, "end_seconds": 20.0},
        history_snapshot={
            "schema_version": "frozen_local_caption_history_v1",
            "preceding_captions": [
                {"segment_id": "0_10", "caption": "A person lifts a box."}
            ],
        },
        base_prompt="Describe the current clip.",
        prompt_delta=delta(),
        baseline_caption="A person stands beside a box.",
        intervention_caption="A person hands the box to another person.",
    )


def outcome(*, missing_trajectories: bool = False) -> QAInterventionOutcome:
    return QAInterventionOutcome(
        qa_id="qa-1",
        is_source_qa=True,
        baseline_answer="B",
        intervention_answer="A",
        baseline_correct=False,
        intervention_correct=True,
        baseline_trajectory_ref=(
            None if missing_trajectories else "baseline/qa-1/trajectory.jsonl"),
        intervention_trajectory_ref=(
            None if missing_trajectories
            else "intervention/qa-1/trajectory.jsonl"),
    )


def episode(*, missing_trajectories: bool = False) -> InterventionEpisode:
    return InterventionEpisode(
        episode_id="episode-001",
        video_id="video-001",
        parent_meta_prompt_id="meta-parent",
        prompt_delta=delta(),
        clips=(clip(),),
        qa_outcomes=(outcome(missing_trajectories=missing_trajectories),),
        baseline_run_ref="runs/baseline",
        intervention_run_ref="runs/intervention",
    )


def evidence(**overrides) -> EpisodeFeedbackEvidence:
    values = {
        "statement": "The intervention caption adds the observed handoff.",
        "supporting_segment_ids": ("10_20",),
        "supporting_qa_ids": ("qa-1",),
        "evidence_type": "mixed",
        "transition_type": None,
        # Confidence is intentionally opaque because Checkpoint A defines no
        # scale, threshold, or acceptance policy.
        "confidence": {"reported": "moderate", "source": "generator"},
    }
    values.update(overrides)
    return EpisodeFeedbackEvidence(**values)


def feedback(**overrides) -> EpisodeFeedback:
    values = {
        "feedback_id": "feedback-001",
        "episode_id": "episode-001",
        "outcome_summary": "The source QA changed from wrong to correct.",
        "observations": (evidence(),),
        "counterevidence": (evidence(
            statement="Exact clip-level causal credit is not established.",
            supporting_segment_ids=(),
            evidence_type="qa_transition",
            transition_type="wrong_to_correct",
            confidence="unknown",
        ),),
        "generator_diagnosis": "The generator under-specified interactions.",
        "recommended_strategy_change": (
            "Inspect object ownership changes before writing the clip prompt."),
        "confidence": "generator-reported-moderate",
    }
    values.update(overrides)
    return EpisodeFeedback(**values)


@pytest.mark.parametrize(("record", "loader"), [
    (
        MetaPromptVersion(
            meta_prompt_id="meta-parent", parent_meta_prompt_id=None,
            text="Inspect frames and history.", created_at="2026-07-20T00:00:00Z",
            status="parent"),
        meta_prompt_version_from_json,
    ),
    (delta(), prompt_delta_from_json),
    (clip(), intervention_clip_record_from_json),
    (outcome(), qa_intervention_outcome_from_json),
    (episode(), intervention_episode_from_json),
    (feedback(), episode_feedback_from_json),
])
def test_top_level_record_serialization_round_trip(record, loader):
    encoded = dumps_canonical(record)
    restored = loader(json.loads(encoded))
    assert restored == record
    assert dumps_canonical(restored) == encoded


def test_nested_intervention_episode_round_trip_preserves_opaque_payloads():
    restored = intervention_episode_from_json(
        json.loads(dumps_canonical(episode())))
    restored_clip = restored.clips[0]
    assert restored_clip.prompt_delta == restored.prompt_delta
    assert restored_clip.time_range == {
        "start_seconds": 10.0, "end_seconds": 20.0}
    assert restored_clip.history_snapshot["preceding_captions"][0][
        "segment_id"] == "0_10"


def test_feedback_with_valid_supporting_ids_is_accepted():
    validate_episode_feedback(feedback(), episode())


def test_feedback_rejects_unknown_supporting_segment_id():
    bad = feedback(observations=(evidence(
        supporting_segment_ids=("missing-segment",)),))
    with pytest.raises(ValueError, match="unknown segment IDs"):
        validate_episode_feedback(bad, episode())


def test_feedback_rejects_unknown_supporting_qa_id():
    bad = feedback(counterevidence=(evidence(
        supporting_qa_ids=("missing-qa",)),))
    with pytest.raises(ValueError, match="unknown QA IDs"):
        validate_episode_feedback(bad, episode())


def test_unavailable_trajectory_references_round_trip_as_null():
    original = outcome(missing_trajectories=True)
    payload = json.loads(dumps_canonical(original))
    assert payload["baseline_trajectory_ref"] is None
    assert payload["intervention_trajectory_ref"] is None
    assert qa_intervention_outcome_from_json(payload) == original


def test_invalid_meta_prompt_status_rejected():
    with pytest.raises(ValueError, match="status is invalid"):
        MetaPromptVersion(
            meta_prompt_id="meta-1", parent_meta_prompt_id=None,
            text="text", created_at="now", status="active")


def test_invalid_evidence_type_rejected():
    with pytest.raises(ValueError, match="evidence_type is invalid"):
        evidence(evidence_type="visual_guess")


def test_transition_type_is_structurally_tied_to_qa_transition_evidence():
    with pytest.raises(ValueError, match="supported QA transition"):
        evidence(
            evidence_type="qa_transition", transition_type=None,
            supporting_segment_ids=())
    with pytest.raises(ValueError, match="must be None"):
        evidence(transition_type="wrong_to_correct")


def test_prompt_delta_has_no_legacy_property_fields():
    assert {field.name for field in dataclasses.fields(PromptDelta)} == {
        "delta_id", "instruction", "source_qa_ids", "proposer_diagnosis"}


def test_clip_has_no_qa_transition_attribution_field():
    assert not ({field.name for field in dataclasses.fields(
        InterventionClipRecord)} & {
            "qa_outcome", "qa_outcomes", "qa_transition", "qa_transitions"})
