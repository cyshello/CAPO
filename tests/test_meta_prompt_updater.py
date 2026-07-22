"""Checkpoint E1 provider-independent meta-prompt updater tests."""

import dataclasses
import json

import pytest

from surrogate_rollout.optimization.episode_feedback import (
    DeterministicMockEpisodeFeedbackGenerator,
)
from surrogate_rollout.optimization.meta_prompt_updater import (
    EPISODE_HISTORY_META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION,
    GROUNDED_META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION,
    META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION,
    META_PROMPT_UPDATER_SYSTEM_INSTRUCTION,
    DeterministicMockMetaPromptUpdater,
    LLMMetaPromptUpdater,
    MetaPromptUpdaterParseError,
    build_meta_prompt_update_request,
    meta_prompt_update_response_json_schema,
    parse_meta_prompt_update_response,
)
from surrogate_rollout.optimization.feedback_memory import (
    build_episode_feedback_memory_record,
)
from surrogate_rollout.optimization.schemas import (
    MetaPromptUpdateDecision,
    MetaPromptVersion,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json
from test_episode_feedback import episode_fixture


POLICY = "fixture_meta_prompt_updater_v1"
CANDIDATE = (
    "Inspect preceding captions for unresolved entities before generating the "
    "clip instruction. Preserve the primary visible action when requesting "
    "continuity evidence."
)


def parent_fixture():
    return MetaPromptVersion(
        meta_prompt_id="meta-parent-001",
        parent_meta_prompt_id=None,
        text="Inspect the current frames and generate a focused instruction.",
        created_at="2026-07-20T00:00:00Z",
        status="parent",
    )


def feedback_fixtures():
    generator = DeterministicMockEpisodeFeedbackGenerator()
    first = generator.generate(episode_fixture(episode_id="episode-001"))
    second = generator.generate(episode_fixture(episode_id="episode-002"))
    return first, second


def test_current_detailed_and_previous_memory_share_one_updater_request():
    parent = parent_fixture()
    current, previous_feedback = feedback_fixtures()
    previous_feedback = dataclasses.replace(
        previous_feedback,
        generator_diagnosis="Historical diagnosis must not be serialized.")
    previous_episode = episode_fixture(episode_id="episode-002")
    previous = build_episode_feedback_memory_record(
        feedback=previous_feedback, episode=previous_episode,
        iteration_id="iteration-previous",
        parent_meta_prompt_id=parent.meta_prompt_id)
    request = build_meta_prompt_update_request(
        parent, (current,), updater_policy_version=POLICY,
        historical_memories=(previous,),
        current_iteration_id="iteration-current")
    assert request.payload["schema_version"] == \
        EPISODE_HISTORY_META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION
    current_payload = request.payload["current_iteration_feedback"]
    assert len(current_payload) == 1
    assert "generator_diagnosis" not in current_payload[0]
    assert "feedback_id" not in current_payload[0]
    assert "episode_id" not in current_payload[0]
    history = request.payload["historical_experience"]
    assert [item["memory_text"] for item in history["memories"]] == [
        previous.memory_text]
    assert "provenance_index" not in history
    serialized = dumps_canonical(request.payload)
    assert previous_feedback.generator_diagnosis not in serialized


def response(feedback_ids, *, candidate=CANDIDATE):
    return {
        "decision": "update" if candidate is not None else "no_update",
        "candidate_meta_prompt": candidate,
        "change_summary": "Add a conditional continuity inspection step.",
        "rationale": "The ordered feedback supports a minimal procedure change.",
        "supporting_feedback_ids": list(feedback_ids),
    }


class Backend:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def __call__(self, system_instruction, user_request):
        self.calls.append((system_instruction, user_request))
        return dumps_canonical(self.value)

    def metadata(self):
        return {"provider": "fixture_injected", "model": "fixture-updater"}


def test_update_and_no_update_schema_and_nullability():
    schema = meta_prompt_update_response_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "decision", "candidate_meta_prompt", "change_summary", "rationale",
        "supporting_feedback_ids",
    }
    assert schema["properties"]["decision"]["enum"] == [
        "update", "no_update"]
    assert schema["properties"]["candidate_meta_prompt"]["anyOf"] == [
        {"type": "string"}, {"type": "null"}]
    with pytest.raises(ValueError, match="candidate_meta_prompt"):
        MetaPromptUpdateDecision(
            decision="update", candidate_meta_prompt=None,
            change_summary="", rationale="", supporting_feedback_ids=())
    with pytest.raises(ValueError, match="must be None"):
        MetaPromptUpdateDecision(
            decision="no_update", candidate_meta_prompt=CANDIDATE,
            change_summary="", rationale="", supporting_feedback_ids=())


def test_request_preserves_feedback_order_and_contains_no_raw_episode_data():
    parent = parent_fixture()
    feedbacks = feedback_fixtures()
    request = build_meta_prompt_update_request(
        parent, feedbacks, updater_policy_version=POLICY)
    assert request.payload["schema_version"] == \
        EPISODE_HISTORY_META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION
    projected = request.payload["current_iteration_feedback"]
    assert len(projected) == len(feedbacks)
    assert all("feedback_id" not in item and "episode_id" not in item
               for item in projected)
    text = request.user_request
    for excluded in (
            "clips", "trajectories", "baseline_caption", "frames",
            "supporting_segment_ids", "supporting_qa_ids", "transition_type",
            "generator_diagnosis", "feedback_id", "episode_id"):
        assert f'"{excluded}"' not in text
    assert request.payload_hash == sha256_json(request.payload)


def test_same_ordered_input_has_same_request_and_candidate_identity():
    parent = parent_fixture()
    feedbacks = feedback_fixtures()
    updater = DeterministicMockMetaPromptUpdater(
        candidate_meta_prompt=CANDIDATE)
    first = updater.update(parent, feedbacks)
    second = updater.update(parent, feedbacks)
    assert first.request.request_id == second.request.request_id
    assert first.request.payload_hash == second.request.payload_hash
    assert first.candidate_meta_prompt_id == second.candidate_meta_prompt_id
    assert dumps_canonical(first.decision) == dumps_canonical(second.decision)
    reversed_result = updater.update(parent, tuple(reversed(feedbacks)))
    # These fixtures differ only in private IDs, so their ID-free projections
    # are intentionally identical in either order.
    assert reversed_result.request.payload_hash == first.request.payload_hash
    assert reversed_result.candidate_meta_prompt_id == \
        first.candidate_meta_prompt_id


def test_grounded_request_compacts_positive_unchanged_signal_without_mutation():
    parent = parent_fixture()
    feedbacks = feedback_fixtures()
    grounding = tuple({
        "feedback_id": feedback.feedback_id,
        "episode_id": feedback.episode_id,
        "caption_change_status": "unchanged",
        "changed_segment_ids": [],
        "qa_transition_summary": {
            "correct_to_correct": [], "wrong_to_wrong": [],
            "wrong_to_correct": ["opaque-qa"], "correct_to_wrong": []},
        "qa_flip_attribution": "positive_episode_signal_without_caption_change",
    } for feedback in feedbacks)
    before = dumps_canonical(feedbacks)
    request = build_meta_prompt_update_request(
        parent, feedbacks, updater_policy_version=POLICY,
        feedback_grounding=grounding)
    assert request.payload["schema_version"] == \
        EPISODE_HISTORY_META_PROMPT_UPDATE_REQUEST_SCHEMA_VERSION
    projected = request.payload["current_iteration_feedback"]
    assert all(item["caption_change_status"] == "unchanged"
               for item in projected)
    assert all(item["changed_caption_count"] == 0 for item in projected)
    assert all(item["episode_effect"] == "positive" for item in projected)
    assert all(item["qa_transition_counts"]["wrong_to_correct"] == 1
               for item in projected)
    assert "feedback_grounding" not in request.payload
    assert dumps_canonical(feedbacks) == before
    with pytest.raises(ValueError, match="order or feedback IDs"):
        build_meta_prompt_update_request(
            parent, feedbacks, updater_policy_version=POLICY,
            feedback_grounding=tuple(reversed(grounding)))


def test_mock_updater_deterministically_supports_no_update_and_update():
    parent = parent_fixture()
    feedbacks = feedback_fixtures()
    no_update = DeterministicMockMetaPromptUpdater().update(parent, feedbacks)
    assert no_update.decision.decision == "no_update"
    assert no_update.decision.candidate_meta_prompt is None
    assert no_update.candidate_meta_prompt_id is None
    assert no_update.candidate_status is None
    updated = DeterministicMockMetaPromptUpdater(
        candidate_meta_prompt=CANDIDATE).update(parent, feedbacks)
    assert updated.decision.decision == "update"
    assert updated.decision.candidate_meta_prompt == CANDIDATE
    assert updated.candidate_meta_prompt_id.startswith("meta_prompt_")
    assert updated.candidate_status == "provisional"


def test_unknown_supporting_feedback_id_is_rejected():
    parent = parent_fixture()
    feedbacks = feedback_fixtures()
    request = build_meta_prompt_update_request(
        parent, feedbacks, updater_policy_version=POLICY)
    value = response(["unknown-feedback"])
    with pytest.raises(
            MetaPromptUpdaterParseError, match="unknown feedback"):
        parse_meta_prompt_update_response(
            dumps_canonical(value), request=request, feedbacks=feedbacks)


@pytest.mark.parametrize("raw", (
    "not-json",
    dumps_canonical({**response([]), "extra": "not allowed"}),
    dumps_canonical({
        key: value for key, value in response([]).items()
        if key != "rationale"}),
))
def test_malformed_missing_and_extra_field_responses_are_rejected(raw):
    feedbacks = feedback_fixtures()
    request = build_meta_prompt_update_request(
        parent_fixture(), feedbacks, updater_policy_version=POLICY)
    with pytest.raises(MetaPromptUpdaterParseError) as caught:
        parse_meta_prompt_update_response(
            raw, request=request, feedbacks=feedbacks)
    assert caught.value.raw_response == raw


def test_injected_backend_uses_strict_parser_and_returns_provisional_result():
    parent = parent_fixture()
    feedbacks = feedback_fixtures()
    backend = Backend(response([item.feedback_id for item in feedbacks]))
    updater = LLMMetaPromptUpdater(
        backend=backend, updater_policy_version=POLICY)
    result = updater.update(parent, feedbacks)
    assert len(backend.calls) == 1
    assert backend.calls[0][0] == META_PROMPT_UPDATER_SYSTEM_INSTRUCTION
    assert result.raw_response == dumps_canonical(backend.value)
    assert result.candidate_status == "provisional"
    assert result.backend_metadata == {
        "provider": "fixture_injected", "model": "fixture-updater"}


def test_candidate_cannot_embed_episode_segment_or_qa_identifiers():
    parent = parent_fixture()
    feedbacks = feedback_fixtures()
    request = build_meta_prompt_update_request(
        parent, feedbacks, updater_policy_version=POLICY)
    for forbidden in (
            feedbacks[0].episode_id,
            feedbacks[0].observations[0].supporting_segment_ids[0],
            feedbacks[0].observations[1].supporting_qa_ids[0]):
        value = response(
            [feedbacks[0].feedback_id],
            candidate=f"Inspect visible continuity for {forbidden}.")
        with pytest.raises(
                MetaPromptUpdaterParseError, match="provenance-only"):
            parse_meta_prompt_update_response(
                dumps_canonical(value), request=request, feedbacks=feedbacks)


def test_parent_is_not_mutated_and_candidate_id_uses_canonical_inputs():
    parent = parent_fixture()
    feedbacks = feedback_fixtures()
    before = dumps_canonical(parent)
    result = DeterministicMockMetaPromptUpdater(
        candidate_meta_prompt=CANDIDATE).update(parent, feedbacks)
    identity = {
        "parent_meta_prompt_id": parent.meta_prompt_id,
        "updater_policy_version": result.updater_policy_version,
        "request_payload_hash": result.request.payload_hash,
        "candidate_meta_prompt": CANDIDATE,
    }
    assert result.candidate_meta_prompt_id == \
        "meta_prompt_" + sha256_json(identity)[:20]
    assert dumps_canonical(parent) == before
    with pytest.raises(dataclasses.FrozenInstanceError):
        parent.text = "mutated"
