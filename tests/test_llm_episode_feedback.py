"""Lean episode-feedback request and parser tests."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from surrogate_rollout.optimization.llm_episode_feedback import (
    EPISODE_FEEDBACK_SYSTEM_INSTRUCTION,
    LEAN_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION,
    EpisodeFeedbackParseError,
    FreshEpisodeFeedbackArtifactResolver,
    LLMEpisodeFeedbackGenerator,
    build_lean_episode_feedback_request,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.optimization.schemas import QAInterventionOutcome
from episode_artifact_fixture import fresh_episode_bundle


POLICY_VERSION = "episode_feedback_request_v6_candidate_mixed_view_sibling_outcomes"


def valid_response(
    episode_id, segment_id="0_10", qa_id="benchmark/train/1",
):
    return {
        "episode_id": episode_id,
        "outcome_summary": "One stored outcome improved.",
        "observations": [{
            "statement": "Observed: Returned evidence links the changed caption to QA use.",
            "supporting_segment_ids": [segment_id],
            "supporting_qa_ids": [qa_id],
            "evidence_type": "trajectory",
            "confidence": "Direct stored evidence.",
        }],
        "counterevidence": [],
        "generator_diagnosis": "A local caption change may have helped.",
        "recommended_strategy_change": "Retain the locally supported detail.",
        "confidence": "Local evidence only.",
        "compact_memory_text": "Visible detail was added.\nThe stored result improved.",
    }


class FakeBackend:
    def __init__(self, response=None):
        self.response = response
        self.calls = []

    def __call__(self, system, user):
        self.calls.append((system, user))
        episode = json.loads(user)["episode"]
        value = self.response or valid_response(episode["episode_id"])
        return dumps_canonical(value)

    def metadata(self):
        return {
            "provider": "fixture", "model": "fixture-model",
            "generation_settings": {"temperature": 0},
            "output_token_limit": 512, "context_limit_tokens": 1_000_000,
        }


def request(tmp_path):
    episode = fresh_episode_bundle(tmp_path)
    value = build_lean_episode_feedback_request(
        episode, artifact_resolver=FreshEpisodeFeedbackArtifactResolver())
    return episode, value


def test_prompt_is_one_repository_owned_lean_file():
    root = Path(__file__).parents[1] / "optimization" / "prompts"
    assert EPISODE_FEEDBACK_SYSTEM_INSTRUCTION == (
        root / "episode_feedback_system_v7.txt").read_text().strip()
    assert not (root / "episode_feedback_system_v6_lean.txt").exists()
    assert not (root / "episode_feedback_system_v5.txt").exists()
    assert not (root / "episode_feedback_model_compact_addendum_v1.txt").exists()


def test_lean_payload_keeps_exact_qa_fields_and_changed_caption_pairs(tmp_path):
    episode, built = request(tmp_path)
    payload = built.model_payload
    assert payload["schema_version"] == LEAN_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION
    body = payload["episode"]
    assert set(body) == {
        "episode_id", "prompt_delta", "qa_transition_summary",
        "changed_captions", "qas"}
    expected_changed = [clip for clip in episode.clips
                        if clip.baseline_caption != clip.intervention_caption]
    assert body["changed_captions"] == [{
        "segment_id": clip.segment_id,
        "baseline_caption": clip.baseline_caption,
        "intervention_caption": clip.intervention_caption,
    } for clip in expected_changed]
    qa = body["qas"][0]
    assert qa["question"] == "Stored question 0?"
    assert qa["answer_choices"] == ["A. first", "B. second", "C. third", "D. fourth"]
    assert qa["gold_answer"] == "A"
    assert qa["baseline_answer"] == "B"
    assert qa["intervention_answer"] == "A"
    assert qa["transition"] == "wrong_to_correct"


def test_lean_feedback_receives_source_and_sibling_all_transition_types(tmp_path):
    episode, _built = request(tmp_path)
    base = FreshEpisodeFeedbackArtifactResolver().resolve_qas(episode)[0]
    states = (
        (False, True, "wrong_to_correct"),
        (True, False, "correct_to_wrong"),
        (True, True, "correct_to_correct"),
        (False, False, "wrong_to_wrong"),
    )
    base_outcome = episode.qa_outcomes[0]
    outcomes = tuple(QAInterventionOutcome(
        qa_id=f"qa-{index}", is_source_qa=index == 0,
        baseline_answer="B", intervention_answer="A",
        baseline_correct=before, intervention_correct=after,
        baseline_trajectory_ref=base_outcome.baseline_trajectory_ref,
        intervention_trajectory_ref=base_outcome.intervention_trajectory_ref,
    ) for index, (before, after, _transition) in enumerate(states))
    episode = replace(episode, episode_id="episode-all-transitions",
                      qa_outcomes=outcomes)

    class _Resolver:
        def resolve_qas(self, _episode):
            return tuple({
                **base,
                "qa_id": outcome.qa_id,
                "is_source_qa": outcome.is_source_qa,
                "baseline_answer": outcome.baseline_answer,
                "intervention_answer": outcome.intervention_answer,
                "baseline_correct": outcome.baseline_correct,
                "intervention_correct": outcome.intervention_correct,
                "transition": state[2],
            } for outcome, state in zip(outcomes, states))

    built = build_lean_episode_feedback_request(
        episode, artifact_resolver=_Resolver())
    qas = built.model_payload["episode"]["qas"]

    assert [row["is_source_qa"] for row in qas] == [True, False, False, False]
    assert [row["transition"] for row in qas] == [
        "wrong_to_correct", "correct_to_wrong", "correct_to_correct",
        "wrong_to_wrong"]


def test_lean_payload_has_no_history_full_clips_or_lossy_marker(tmp_path):
    _, built = request(tmp_path)
    text = dumps_canonical(built.model_payload)
    for forbidden in (
            "history_catalog", "history_item_ids", "base_prompt",
            "referenced_segment_ids", "retrieved_segment_ids", "hits",
            "assistant_steps", "final_response", "[TRUNCATED_TO_CONTEXT]"):
        assert forbidden not in text


def test_tool_calls_keep_only_query_evidence_and_changed_id_intersection(tmp_path):
    _, built = request(tmp_path)
    call = built.model_payload["episode"]["qas"][0]["baseline_tool_calls"][0]
    assert set(call) == {"query", "returned_evidence", "segment_ids"}
    assert call["query"] == "fixture query"
    assert call["returned_evidence"] == ["baseline evidence for 0_10"]
    assert call["segment_ids"] == ["0_10"]


def test_generator_uses_only_lean_request(tmp_path):
    episode = fresh_episode_bundle(tmp_path)
    backend = FakeBackend()
    result = LLMEpisodeFeedbackGenerator(
        response_provider=backend,
        artifact_resolver=FreshEpisodeFeedbackArtifactResolver(),
        policy_version=POLICY_VERSION).generate_with_trace(episode)
    sent = json.loads(backend.calls[0][1])
    assert sent["schema_version"] == LEAN_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION
    assert sent == result.request.model_payload


def test_request_and_raw_response_persist_before_structural_parse_failure(tmp_path):
    episode = fresh_episode_bundle(tmp_path)
    invalid = valid_response(episode.episode_id)
    invalid["observations"][0]["supporting_segment_ids"] = "not-an-array"
    stage = tmp_path / "feedback-stage"
    generator = LLMEpisodeFeedbackGenerator(
        response_provider=FakeBackend(invalid),
        artifact_resolver=FreshEpisodeFeedbackArtifactResolver(),
        policy_version=POLICY_VERSION)
    with pytest.raises(EpisodeFeedbackParseError):
        generator.generate_to_directory(episode, str(stage))
    manifest_path = stage / "request_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert "call_count" not in manifest["backend"]
    manifest["backend"]["call_count"] = 99
    manifest_path.write_text(dumps_canonical(manifest) + "\n")
    with pytest.raises(EpisodeFeedbackParseError):
        generator.generate_to_directory(episode, str(stage))
    assert len(generator.response_provider.calls) == 1
    assert json.loads((stage / "request.json").read_text())["schema_version"] == \
        LEAN_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION
    assert json.loads((stage / "raw_response.json").read_text()) == invalid
    assert json.loads(manifest_path.read_text())["backend"]["call_count"] == 99
