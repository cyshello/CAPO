"""Checkpoint D1 complete request and strict LLM boundary tests."""

import dataclasses
import json

import pytest

from surrogate_rollout.optimization.episode_feedback import (
    EpisodeFeedbackGenerator,
)
from surrogate_rollout.optimization.llm_episode_feedback import (
    EPISODE_FEEDBACK_SYSTEM_INSTRUCTION,
    EpisodeFeedbackBackendConfigurationError,
    EpisodeFeedbackContextOverflowError,
    EpisodeFeedbackParseError,
    LLMEpisodeFeedbackGenerator,
    LegacyEpisodeFeedbackArtifactResolver,
    build_episode_feedback_request,
    parse_episode_feedback_response,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json
from test_legacy_intervention_adapter import convert, legacy_bundle


POLICY_VERSION = "fixture_episode_feedback_policy_v1"


def valid_response(episode_id, *, recommendation="Inspect continuity conditionally."):
    return {
        "episode_id": episode_id,
        "outcome_summary": "Observed: all stored QA outcomes were considered.",
        "observations": [{
            "statement": "Observed: both recaptioned clip strings changed.",
            "supporting_segment_ids": ["0_10", "10_20"],
            "supporting_qa_ids": [],
            "evidence_type": "caption_change",
            "transition_type": None,
            "confidence": "direct string comparison",
        }, {
            "statement": "Observed: the stored source QA changed correctness.",
            "supporting_segment_ids": [],
            "supporting_qa_ids": ["benchmark/train/1"],
            "evidence_type": "qa_transition",
            "transition_type": "wrong_to_correct",
            "confidence": "stored correctness pair",
        }],
        "counterevidence": [],
        "generator_diagnosis": (
            "Hypothesis: inspect continuity evidence; causal credit remains "
            "uncertain across the episode."),
        "recommended_strategy_change": recommendation,
        "confidence": "episode evidence with uncertain attribution",
    }


class FakeBackend:
    def __init__(
        self, response_factory=None, *, context_limit_tokens=None,
        token_count=100,
    ):
        self.response_factory = response_factory or (
            lambda request: valid_response(request["episode"]["episode_id"]))
        self.context_limit_tokens = context_limit_tokens
        self.token_count = token_count
        self.calls = []

    def __call__(self, system_instruction, request):
        payload = json.loads(request)
        self.calls.append((system_instruction, request))
        return dumps_canonical(self.response_factory(payload))

    def count_tokens(self, messages):
        assert tuple(item["role"] for item in messages) == ("system", "user")
        return self.token_count

    def metadata(self):
        return {
            "provider": "deterministic_fake_backend",
            "model": "fixture-model-not-called",
            "generation_settings": {"temperature": 0},
            "output_token_limit": 512,
            "context_limit_tokens": self.context_limit_tokens,
            "call_count": len(self.calls),
        }


def episode_and_request(tmp_path, **request_kwargs):
    episode = convert(legacy_bundle(tmp_path))
    request = build_episode_feedback_request(
        episode,
        artifact_resolver=LegacyEpisodeFeedbackArtifactResolver(),
        **request_kwargs,
    )
    return episode, request


def test_complete_request_preserves_every_clip_and_exact_clip_fields(tmp_path):
    episode, request = episode_and_request(tmp_path)
    clips = request.user_payload["episode"]["clips"]
    assert [item["segment_id"] for item in clips] == [
        item.segment_id for item in episode.clips]
    for source, payload in zip(episode.clips, clips):
        assert payload["time_range"] == json.loads(dumps_canonical(
            source.time_range))
        assert payload["history_snapshot"] == json.loads(dumps_canonical(
            source.history_snapshot))
        assert payload["base_prompt"] == source.base_prompt
        assert payload["applied_prompt_delta_instruction"] == \
            source.prompt_delta.instruction
        assert payload["baseline_caption"] == source.baseline_caption
        assert payload["intervention_caption"] == source.intervention_caption


def test_request_contains_episode_lineage_and_all_sibling_qas(tmp_path):
    episode, request = episode_and_request(tmp_path)
    payload = request.user_payload["episode"]
    assert payload["episode_id"] == episode.episode_id
    assert payload["video_id"] == episode.video_id
    assert payload["parent_meta_prompt_id"] == episode.parent_meta_prompt_id
    assert payload["prompt_delta"] == {
        "delta_id": episode.prompt_delta.delta_id,
        "instruction": episode.prompt_delta.instruction,
        "source_qa_ids": list(episode.prompt_delta.source_qa_ids),
        "proposer_diagnosis": episode.prompt_delta.proposer_diagnosis,
    }
    assert [item["qa_id"] for item in payload["qas"]] == [
        item.qa_id for item in episode.qa_outcomes]
    assert payload["qa_transition_summary"] == {
        "correct_to_correct": ["benchmark/train/2"],
        "wrong_to_wrong": ["benchmark/train/3"],
        "wrong_to_correct": ["benchmark/train/1"],
        "correct_to_wrong": [],
    }


def test_qa_metadata_is_resolved_only_from_saved_baseline_artifact(tmp_path):
    _, request = episode_and_request(tmp_path)
    first = request.user_payload["episode"]["qas"][0]
    assert first["question"] == "Stored question 0?"
    assert first["answer_choices"] == [
        "A. first", "B. second", "C. third"]
    assert first["gold_answer"] == "A"
    assert first["transition"] == "wrong_to_correct"


def test_raw_trajectory_tool_events_and_references_are_lossless(tmp_path):
    _, request = episode_and_request(tmp_path)
    first = request.user_payload["episode"]["qas"][0]
    baseline = first["baseline_trajectory"]
    intervention = first["intervention_trajectory"]
    assert baseline["availability"] == "available"
    assert baseline["content"][3]["content"] == "baseline evidence for 0_10"
    assert intervention["content"][3]["content"] == \
        "intervention evidence for 0_10"
    assert baseline["tool_events"][0]["tool"] == "clip_search_tool"
    assert baseline["referenced_segment_ids"] == ["0_10"]
    assert baseline["retrieved_segment_ids"] == ["0_10"]
    assert baseline["reference_sets"] == {
        "explicitly_cited_segments": ["0_10"],
        "retrieved_segments": ["0_10"],
    }
    assert baseline["reference_evidence"] == [{
        "segment": "0_10", "set": "retrieved_segments",
        "reason": "clip_search_tool_hit", "event_index": 0,
    }]


def test_none_trajectory_is_explicitly_unavailable(tmp_path):
    _, request = episode_and_request(tmp_path)
    unavailable = request.user_payload["episode"]["qas"][1]
    expected = {
        "availability": "unavailable",
        "content": None,
        "tool_events": [],
        "reference_sets": {},
        "reference_evidence": [],
        "referenced_segment_ids": [],
        "retrieved_segment_ids": [],
    }
    assert unavailable["baseline_trajectory"] == expected
    assert unavailable["intervention_trajectory"] == expected


def test_payload_excludes_frames_codebook_and_unrelated_state(tmp_path):
    _, request = episode_and_request(tmp_path)
    payload_text = dumps_canonical(request.user_payload)
    assert '"frames"' not in payload_text
    assert '"codebook"' not in payload_text
    assert '"current_meta_prompt"' not in payload_text
    assert '"updater_history"' not in payload_text


def test_request_serialization_hash_and_size_statistics_are_exact(tmp_path):
    episode, first = episode_and_request(tmp_path)
    _, second = episode_and_request(tmp_path)
    assert first.user_request == second.user_request
    assert first.payload_hash == second.payload_hash
    assert first.payload_hash == sha256_json(first.user_payload)
    stats = first.size_statistics
    assert stats.clip_count == len(episode.clips)
    assert stats.qa_count == len(episode.qa_outcomes)
    assert stats.total_history_item_count == sum(
        len(clip.history_snapshot["history"]) for clip in episode.clips)
    assert stats.serialized_request_character_count == len(
        dumps_canonical(first.messages))
    assert stats.clip_record_character_count == sum(
        len(dumps_canonical(item))
        for item in first.user_payload["episode"]["clips"])
    assert stats.qa_record_character_count == sum(
        len(dumps_canonical(item))
        for item in first.user_payload["episode"]["qas"])
    expected_trajectory_chars = 0
    for qa in first.user_payload["episode"]["qas"]:
        for side in ("baseline_trajectory", "intervention_trajectory"):
            trajectory = qa[side]
            if trajectory["availability"] == "available":
                expected_trajectory_chars += len(dumps_canonical(
                    trajectory["content"]))
                expected_trajectory_chars += len(dumps_canonical(
                    trajectory["tool_events"]))
    assert stats.trajectory_character_count == expected_trajectory_chars
    assert stats.token_count is None
    assert stats.context_limit_checked is False
    assert stats.unresolved_reference_count == 0


def test_system_instruction_contains_required_causal_and_output_limits():
    assert "Do not attribute an episode-level QA outcome to one clip" in \
        EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
    assert "Retrieved or referenced clips are not causal proof" in \
        EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
    assert "Length limits:" in EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
    assert "at most 4 observations" in EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
    assert "at most 3 counterevidence" in EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
    assert "Do not return feedback_id" in EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
    assert "Do not use Markdown fences" in EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
    assert "propose codebook entries" in EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
    assert "Copy QA transition facts exactly from qa_transition_summary" in \
        EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
    assert "Do not infer or recompute QA transitions from trajectories" in \
        EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
    assert "Do not state that all QAs share one transition" in \
        EPISODE_FEEDBACK_SYSTEM_INSTRUCTION


def test_fake_backend_valid_json_returns_feedback_and_full_trace(tmp_path):
    episode = convert(legacy_bundle(tmp_path))
    backend = FakeBackend()
    generator: EpisodeFeedbackGenerator = LLMEpisodeFeedbackGenerator(
        response_provider=backend,
        artifact_resolver=LegacyEpisodeFeedbackArtifactResolver(),
        policy_version=POLICY_VERSION,
    )
    result = generator.generate_with_trace(episode)
    assert result.feedback.episode_id == episode.episode_id
    assert result.raw_response == dumps_canonical(valid_response(
        episode.episode_id))
    assert result.request.messages[0]["content"] == \
        EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
    assert backend.calls[0] == (
        result.request.system_instruction, result.request.user_request)
    assert result.backend_metadata["model"] == "fixture-model-not-called"
    assert generator.generate(episode).episode_id == episode.episode_id


def test_feedback_id_is_deterministic_from_episode_policy_and_request(tmp_path):
    episode = convert(legacy_bundle(tmp_path))
    generator = LLMEpisodeFeedbackGenerator(
        response_provider=FakeBackend(),
        artifact_resolver=LegacyEpisodeFeedbackArtifactResolver(),
        policy_version=POLICY_VERSION,
    )
    first = generator.generate_with_trace(episode)
    second = generator.generate_with_trace(episode)
    identity = {
        "episode_id": episode.episode_id,
        "feedback_policy_version": POLICY_VERSION,
        "request_payload_hash": first.request.payload_hash,
    }
    assert first.feedback.feedback_id == (
        "episode_feedback_" + sha256_json(identity)[:20])
    assert first.feedback.feedback_id == second.feedback.feedback_id
    assert dumps_canonical(first.feedback) == dumps_canonical(second.feedback)


def parse_fixture_response(episode, request, value):
    return parse_episode_feedback_response(
        dumps_canonical(value), episode=episode,
        policy_version=POLICY_VERSION,
        request_payload_hash=request.payload_hash)


def test_wrong_episode_id_is_rejected(tmp_path):
    episode, request = episode_and_request(tmp_path)
    response = valid_response("another-episode")
    with pytest.raises(EpisodeFeedbackParseError, match="episode_id") as caught:
        parse_fixture_response(episode, request, response)
    assert caught.value.raw_response == dumps_canonical(response)


@pytest.mark.parametrize(("field", "bad_id"), (
    ("supporting_segment_ids", "missing-segment"),
    ("supporting_qa_ids", "missing-qa"),
))
def test_unknown_supporting_id_is_rejected(tmp_path, field, bad_id):
    episode, request = episode_and_request(tmp_path)
    response = valid_response(episode.episode_id)
    response["observations"][0][field] = [bad_id]
    with pytest.raises(EpisodeFeedbackParseError, match="unknown"):
        parse_fixture_response(episode, request, response)


def test_evidence_item_with_no_support_is_rejected_at_real_boundary(tmp_path):
    episode, request = episode_and_request(tmp_path)
    response = valid_response(episode.episode_id)
    response["observations"][0]["supporting_segment_ids"] = []
    with pytest.raises(EpisodeFeedbackParseError, match="no supporting IDs"):
        parse_fixture_response(episode, request, response)


@pytest.mark.parametrize("raw", (
    "not-json",
    '```json\n{"episode_id":"x"}\n```',
    '{"episode_id":"x"} trailing prose',
))
def test_malformed_fenced_and_trailing_responses_are_rejected(tmp_path, raw):
    episode, request = episode_and_request(tmp_path)
    with pytest.raises(EpisodeFeedbackParseError) as caught:
        parse_episode_feedback_response(
            raw, episode=episode, policy_version=POLICY_VERSION,
            request_payload_hash=request.payload_hash)
    assert caught.value.raw_response == raw


def test_placeholder_evidence_type_is_rejected(tmp_path):
    episode, request = episode_and_request(tmp_path)
    response = valid_response(episode.episode_id)
    response["observations"][0]["evidence_type"] = "placeholder"
    with pytest.raises(EpisodeFeedbackParseError, match="evidence_type"):
        parse_fixture_response(episode, request, response)


def test_transition_type_must_match_every_stored_supporting_qa(tmp_path):
    episode, request = episode_and_request(tmp_path)
    response = valid_response(episode.episode_id)
    response["observations"][1]["transition_type"] = "correct_to_correct"
    with pytest.raises(EpisodeFeedbackParseError, match="stored QA outcomes"):
        parse_fixture_response(episode, request, response)


def test_transition_type_nullability_follows_evidence_type(tmp_path):
    episode, request = episode_and_request(tmp_path)
    response = valid_response(episode.episode_id)
    response["observations"][0]["transition_type"] = "wrong_to_correct"
    with pytest.raises(EpisodeFeedbackParseError, match="must be null"):
        parse_fixture_response(episode, request, response)

    response = valid_response(episode.episode_id)
    response["observations"][1]["transition_type"] = None
    with pytest.raises(EpisodeFeedbackParseError, match="invalid or null"):
        parse_fixture_response(episode, request, response)


def test_exact_delta_strategy_is_not_silently_repaired(tmp_path):
    episode, request = episode_and_request(tmp_path)
    response = valid_response(
        episode.episode_id,
        recommendation=episode.prompt_delta.instruction)
    parsed = parse_fixture_response(episode, request, response)
    assert parsed.recommended_strategy_change == episode.prompt_delta.instruction


def test_generation_does_not_mutate_episode(tmp_path):
    episode = convert(legacy_bundle(tmp_path))
    before = dumps_canonical(episode)
    LLMEpisodeFeedbackGenerator(
        response_provider=FakeBackend(),
        artifact_resolver=LegacyEpisodeFeedbackArtifactResolver(),
        policy_version=POLICY_VERSION).generate(episode)
    assert dumps_canonical(episode) == before


def test_known_context_overflow_fails_before_backend_without_truncation(tmp_path):
    episode = convert(legacy_bundle(tmp_path))
    backend = FakeBackend(context_limit_tokens=10, token_count=11)
    generator = LLMEpisodeFeedbackGenerator(
        response_provider=backend,
        artifact_resolver=LegacyEpisodeFeedbackArtifactResolver(),
        policy_version=POLICY_VERSION)
    with pytest.raises(EpisodeFeedbackContextOverflowError) as caught:
        generator.generate(episode)
    assert caught.value.observed_tokens == 11
    assert caught.value.configured_limit == 10
    assert caught.value.clip_count == len(episode.clips)
    assert caught.value.qa_count == len(episode.qa_outcomes)
    assert "no truncation" in str(caught.value)
    assert backend.calls == []


def test_unknown_context_limit_applies_no_invented_limit(tmp_path):
    _, request = episode_and_request(tmp_path)
    assert request.size_statistics.context_limit_tokens is None
    assert request.size_statistics.context_limit_checked is False
    assert request.size_statistics.token_count is None


def test_missing_backend_identity_or_generation_config_fails_fast(tmp_path):
    episode = convert(legacy_bundle(tmp_path))

    class IncompleteBackend(FakeBackend):
        def metadata(self):
            return {"provider": "fake", "model": None}

    generator = LLMEpisodeFeedbackGenerator(
        response_provider=IncompleteBackend(),
        artifact_resolver=LegacyEpisodeFeedbackArtifactResolver(),
        policy_version=POLICY_VERSION)
    with pytest.raises(
            EpisodeFeedbackBackendConfigurationError, match="model identity"):
        generator.generate(episode)
