"""Deterministic Checkpoint C feedback over prompt-delta episodes.

This boundary is intentionally separate from the legacy property/codebook
``FeedbackGenerator``.  It performs no model calls, artifact reads, writes, or
semantic inference; it reports only facts already serialized on an episode.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol

from surrogate_rollout.optimization.schemas import (
    EpisodeFeedback,
    EpisodeFeedbackEvidence,
    InterventionEpisode,
    QAInterventionOutcome,
    validate_episode_feedback,
)
from surrogate_rollout.schemas import sha256_json
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


MOCK_EPISODE_FEEDBACK_POLICY_VERSION = "mock_episode_feedback_v2_grounded"
EPISODE_FEEDBACK_ELIGIBILITY_POLICY_VERSION = \
    "episode_feedback_semantic_eligibility_v1"
DETERMINISTIC_CONFIDENCE_MARKER = "deterministic"
MOCK_GENERATOR_DIAGNOSIS = (
    "Deterministic mock feedback only; no semantic generator diagnosis was "
    "performed."
)
MOCK_RECOMMENDED_STRATEGY_CHANGE = (
    "No strategy change is proposed by the deterministic mock generator."
)

_TRANSITIONS = (
    "wrong_to_correct",
    "correct_to_wrong",
    "correct_to_correct",
    "wrong_to_wrong",
)


class EpisodeFeedbackGenerationError(ValueError):
    """An episode cannot be described without inventing missing facts."""


@dataclass(frozen=True)
class EpisodeFeedbackEligibility:
    """Deterministic, runtime-only updater eligibility decision."""

    schema_version: str
    feedback_id: str
    episode_id: str
    feedback_sha256: str
    eligible: bool
    reasons: tuple[str, ...]


class EpisodeFeedbackGenerator(Protocol):
    def generate(self, episode: InterventionEpisode) -> EpisodeFeedback:
        ...


def classify_qa_transition(outcome: QAInterventionOutcome) -> str:
    baseline = outcome.baseline_correct
    intervention = outcome.intervention_correct
    if not isinstance(baseline, bool) or not isinstance(intervention, bool):
        raise EpisodeFeedbackGenerationError(
            f"QA outcome {outcome.qa_id!r} has unavailable correctness and "
            "cannot be assigned to a deterministic correctness transition")
    if not baseline and intervention:
        return "wrong_to_correct"
    if baseline and not intervention:
        return "correct_to_wrong"
    if baseline and intervention:
        return "correct_to_correct"
    return "wrong_to_wrong"


class DeterministicMockEpisodeFeedbackGenerator:
    """Produce schema-valid, provenance-linked facts without semantic claims."""

    policy_version = MOCK_EPISODE_FEEDBACK_POLICY_VERSION

    def generate(self, episode: InterventionEpisode) -> EpisodeFeedback:
        if not isinstance(episode, InterventionEpisode):
            raise TypeError("episode must be an InterventionEpisode")

        instruction = episode.prompt_delta.instruction
        mismatched_segments = tuple(
            clip.segment_id for clip in episode.clips
            if clip.prompt_delta.instruction != instruction
        )
        if mismatched_segments:
            raise EpisodeFeedbackGenerationError(
                "clip prompt delta instruction differs from the authoritative "
                f"episode prompt delta for segments {mismatched_segments!r}")

        counts = {name: 0 for name in _TRANSITIONS}
        correct_to_wrong_ids = []
        for outcome in episode.qa_outcomes:
            transition = classify_qa_transition(outcome)
            counts[transition] += 1
            if transition == "correct_to_wrong":
                correct_to_wrong_ids.append(outcome.qa_id)

        source_count = sum(item.is_source_qa for item in episode.qa_outcomes)
        sibling_count = len(episode.qa_outcomes) - source_count
        outcome_summary = (
            f"The episode contains {len(episode.qa_outcomes)} QA outcomes: "
            f"{counts['wrong_to_correct']} wrong_to_correct, "
            f"{counts['correct_to_wrong']} correct_to_wrong, "
            f"{counts['correct_to_correct']} correct_to_correct, and "
            f"{counts['wrong_to_wrong']} wrong_to_wrong; "
            f"{source_count} source QA and {sibling_count} sibling QA."
        )

        observations = []
        changed_segment_ids = tuple(
            clip.segment_id for clip in episode.clips
            if clip.baseline_caption != clip.intervention_caption
        )
        if changed_segment_ids:
            observations.append(EpisodeFeedbackEvidence(
                statement=(
                    f"The episode contains {len(changed_segment_ids)} clips "
                    "with different baseline and intervention caption strings."
                ),
                supporting_segment_ids=changed_segment_ids,
                supporting_qa_ids=(),
                evidence_type="caption_change",
                transition_type=None,
                confidence=DETERMINISTIC_CONFIDENCE_MARKER,
            ))

        qa_ids_by_transition = {name: [] for name in _TRANSITIONS}
        for outcome in episode.qa_outcomes:
            qa_ids_by_transition[classify_qa_transition(outcome)].append(
                outcome.qa_id)
        for transition in _TRANSITIONS:
            qa_ids = tuple(qa_ids_by_transition[transition])
            if not qa_ids:
                continue
            observations.append(EpisodeFeedbackEvidence(
                statement=(
                    f"The episode contains {len(qa_ids)} {transition} QA "
                    "outcomes."
                ),
                supporting_segment_ids=(),
                supporting_qa_ids=qa_ids,
                evidence_type="qa_transition",
                transition_type=transition,
                confidence=DETERMINISTIC_CONFIDENCE_MARKER,
            ))

        counterevidence = ()
        if correct_to_wrong_ids:
            counterevidence = (EpisodeFeedbackEvidence(
                statement=(
                    f"The episode contains {len(correct_to_wrong_ids)} "
                    "correct_to_wrong QA outcomes."
                ),
                supporting_segment_ids=(),
                supporting_qa_ids=tuple(correct_to_wrong_ids),
                evidence_type="qa_transition",
                transition_type="correct_to_wrong",
                confidence=DETERMINISTIC_CONFIDENCE_MARKER,
            ),)

        identity = {
            "episode_id": episode.episode_id,
            "policy_version": self.policy_version,
        }
        feedback = EpisodeFeedback(
            feedback_id=f"mock_feedback_{sha256_json(identity)[:20]}",
            episode_id=episode.episode_id,
            outcome_summary=outcome_summary,
            observations=tuple(observations),
            counterevidence=counterevidence,
            generator_diagnosis=MOCK_GENERATOR_DIAGNOSIS,
            recommended_strategy_change=MOCK_RECOMMENDED_STRATEGY_CHANGE,
            confidence=DETERMINISTIC_CONFIDENCE_MARKER,
        )
        validate_episode_feedback(feedback, episode)
        return feedback


def evaluate_episode_feedback_eligibility(
    feedback: EpisodeFeedback,
    episode: InterventionEpisode,
) -> EpisodeFeedbackEligibility:
    """Classify updater eligibility without interpreting natural-language text."""
    if not isinstance(feedback, EpisodeFeedback):
        raise TypeError("feedback must be an EpisodeFeedback")
    if not isinstance(episode, InterventionEpisode):
        raise TypeError("episode must be an InterventionEpisode")

    reasons: list[str] = []
    try:
        validate_episode_feedback(feedback, episode)
    except (TypeError, ValueError) as exc:
        reasons.append(f"schema_or_transition_validation_failed: {exc}")

    for collection_name in ("observations", "counterevidence"):
        for index, item in enumerate(getattr(feedback, collection_name)):
            location = f"{collection_name}[{index}]"
            if not item.supporting_segment_ids and not item.supporting_qa_ids:
                reasons.append(f"{location}: no supporting IDs")
            if item.evidence_type == "caption_change" and (
                    not item.supporting_segment_ids or item.supporting_qa_ids):
                reasons.append(
                    f"{location}: caption_change requires only segment IDs")
            elif item.evidence_type == "qa_transition" and (
                    item.supporting_segment_ids or not item.supporting_qa_ids):
                reasons.append(
                    f"{location}: qa_transition requires only QA IDs")
            elif item.evidence_type in ("trajectory", "mixed") and (
                    not item.supporting_segment_ids or
                    not item.supporting_qa_ids):
                reasons.append(
                    f"{location}: trajectory-linked evidence requires both "
                    "segment and QA IDs")

    if not feedback.generator_diagnosis.strip():
        reasons.append("generator_diagnosis is empty")
    if not feedback.recommended_strategy_change.strip():
        reasons.append("recommended_strategy_change is empty")

    return EpisodeFeedbackEligibility(
        schema_version=EPISODE_FEEDBACK_ELIGIBILITY_POLICY_VERSION,
        feedback_id=feedback.feedback_id,
        episode_id=feedback.episode_id,
        feedback_sha256=sha256_json(json.loads(dumps_canonical(feedback))),
        eligible=not reasons,
        reasons=tuple(reasons),
    )
