"""Checkpoint D2 exact-token provider-boundary tests; no model is called."""

import hashlib
import json

import pytest

from surrogate_rollout.optimization.llm_episode_feedback import (
    LLMEpisodeFeedbackGenerator,
    FreshEpisodeFeedbackArtifactResolver,
)
from surrogate_rollout.optimization.policies.episode_feedback_provider import (
    EpisodeFeedbackProviderContextOverflowError,
    EpisodeFeedbackProviderNotConfiguredError,
    ExactProviderInputTokenCount,
    OpenAICompatibleEpisodeFeedbackProviderAdapter,
    episode_feedback_response_json_schema,
    prepare_and_measure,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from episode_artifact_fixture import fresh_episode_bundle
from test_llm_episode_feedback import POLICY_VERSION, valid_response


class Transport:
    def __init__(self, response_factory=None):
        self.calls = []
        self.response_factory = response_factory or (
            lambda request: valid_response(json.loads(
                request["messages"][1]["content"])["episode"]["episode_id"]))

    def __call__(self, request):
        self.calls.append(json.loads(dumps_canonical(request)))
        return dumps_canonical(self.response_factory(request))


def exact_character_token_count(messages):
    """Fixture tokenizer: one Unicode code point per content token."""
    system = len(messages[0]["content"])
    user = len(messages[1]["content"])
    return ExactProviderInputTokenCount(
        system_prompt_tokens=system,
        user_payload_tokens=user,
        total_input_tokens=system + user + 7,
    )


def provider(**overrides):
    values = {
        "provider": "fixture_openai_compatible",
        "model_id": "fixture-model-explicit",
        "tokenizer_identity": "fixture_unicode_codepoint_tokenizer_v1",
        "exact_token_counter": exact_character_token_count,
        "context_limit": 1_000_000,
        "maximum_output_tokens": 512,
        "generation_settings": {"temperature": 0},
        "feedback_policy_version": POLICY_VERSION,
        "response_transport": Transport(),
    }
    values.update(overrides)
    return OpenAICompatibleEpisodeFeedbackProviderAdapter(**values)


@pytest.mark.parametrize("missing", (
    "provider", "model_id", "tokenizer_identity", "exact_token_counter",
    "context_limit", "maximum_output_tokens", "generation_settings",
    "feedback_policy_version", "response_transport",
))
def test_every_provider_and_tokenizer_setting_is_required(missing):
    overrides = {missing: None}
    with pytest.raises(
            EpisodeFeedbackProviderNotConfiguredError,
            match="explicit"):
        provider(**overrides)


def test_prepare_and_measure_exact_counts_output_reservation_and_fit(tmp_path):
    episode = fresh_episode_bundle(tmp_path)
    adapter = provider()
    inspection = prepare_and_measure(
        episode, provider_adapter=adapter,
        artifact_resolver=FreshEpisodeFeedbackArtifactResolver())
    messages = inspection.prepared_request.messages
    stats = inspection.token_statistics
    assert stats.system_prompt_tokens == len(messages[0]["content"])
    assert stats.user_payload_tokens == len(messages[1]["content"])
    assert stats.total_input_tokens == (
        stats.system_prompt_tokens + stats.user_payload_tokens + 7)
    assert stats.reserved_output_tokens == 512
    assert stats.context_limit == 1_000_000
    assert stats.remaining_tokens == (
        stats.context_limit - stats.total_input_tokens - 512)
    assert stats.fits_context is True
    assert inspection.fits_context is True
    assert adapter.response_transport.calls == []


def test_exact_provider_body_contains_only_lean_user_payload(tmp_path):
    episode = fresh_episode_bundle(tmp_path)
    inspection = prepare_and_measure(
        episode, provider_adapter=provider(),
        artifact_resolver=FreshEpisodeFeedbackArtifactResolver())
    body = inspection.prepared_request.request_body
    request = inspection.lean_request
    assert body["model"] == "fixture-model-explicit"
    assert body["messages"][1]["content"] == dumps_canonical(
        request.model_payload)
    assert json.loads(body["messages"][1]["content"]) == request.model_payload
    assert "audit_metadata" not in body["messages"][1]["content"]
    assert body["max_tokens"] == 512
    assert body["temperature"] == 0


def test_strict_output_schema_excludes_feedback_id():
    schema = episode_feedback_response_json_schema()
    assert schema["additionalProperties"] is False
    assert "confidence" in schema["required"]
    assert "feedback_id" not in schema["properties"]
    assert "feedback_id" not in schema["required"]
    assert set(schema["properties"]) == {
        "episode_id", "outcome_summary", "observations", "counterevidence",
        "generator_diagnosis", "recommended_strategy_change", "confidence",
        "compact_memory_text",
    }
    assert schema["properties"]["compact_memory_text"]["type"] == [
        "string", "null"]


def test_strict_output_schema_keeps_confidence_open_vocabulary_strings():
    schema = episode_feedback_response_json_schema()
    expected = {
        "type": "string",
        "description": (
            "An open-vocabulary confidence assessment. No fixed scale or "
            "enum is imposed."),
    }
    assert schema["properties"]["confidence"] == expected
    evidence = schema["properties"]["observations"]["items"]
    assert "confidence" in evidence["required"]
    assert evidence["properties"]["confidence"] == expected
    assert "enum" not in schema["properties"]["confidence"]
    assert "enum" not in evidence["properties"]["confidence"]


def test_strict_output_schema_excludes_model_authored_transitions():
    schema = episode_feedback_response_json_schema()
    evidence = schema["properties"]["observations"]["items"]
    assert "transition_type" not in evidence["required"]
    assert "transition_type" not in evidence["properties"]
    assert "qa_transition" not in evidence["properties"]["evidence_type"]["enum"]


def test_every_strict_response_object_rejects_additional_properties():
    def visit(value):
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value.get("additionalProperties") is False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(episode_feedback_response_json_schema())


def test_context_overflow_fails_before_transport_without_truncation(tmp_path):
    episode = fresh_episode_bundle(tmp_path)
    roomy = prepare_and_measure(
        episode, provider_adapter=provider(),
        artifact_resolver=FreshEpisodeFeedbackArtifactResolver())
    total = roomy.token_statistics.total_input_tokens
    transport = Transport()
    adapter = provider(
        context_limit=total + 511, response_transport=transport)
    inspection = prepare_and_measure(
        episode, provider_adapter=adapter,
        artifact_resolver=FreshEpisodeFeedbackArtifactResolver())
    assert inspection.fits_context is False
    assert inspection.token_statistics.remaining_tokens == -1

    generator = LLMEpisodeFeedbackGenerator(
        response_provider=adapter,
        artifact_resolver=FreshEpisodeFeedbackArtifactResolver(),
        policy_version=POLICY_VERSION)
    with pytest.raises(EpisodeFeedbackProviderContextOverflowError):
        generator.generate(episode)
    assert adapter.call_count == 0
    assert transport.calls == []


def test_provider_safety_margin_is_reserved_before_transport():
    adapter = provider(
        context_limit=1000, maximum_output_tokens=100,
        context_safety_margin_tokens=512)
    with pytest.raises(EpisodeFeedbackProviderContextOverflowError):
        adapter.preflight(
            "fixed system", dumps_canonical({"text": "evidence " * 1000}))


def test_lean_request_uses_strict_schema_and_existing_parser(tmp_path):
    episode = fresh_episode_bundle(tmp_path)
    transport = Transport()
    adapter = provider(response_transport=transport)
    stage = tmp_path / "feedback-trace"
    feedback = LLMEpisodeFeedbackGenerator(
        response_provider=adapter,
        artifact_resolver=FreshEpisodeFeedbackArtifactResolver(),
        policy_version=POLICY_VERSION).generate_to_directory(
            episode, str(stage))
    assert feedback.episode_id == episode.episode_id
    assert feedback.observations[0].transition_type is None
    assert adapter.call_count == 1
    assert json.loads((stage / "provider_request.json").read_text()) == \
        transport.calls[0]
    assert json.loads((stage / "request.json").read_text())["schema_version"] == \
        "episode_feedback_request_v6_candidate_mixed_view_sibling_outcomes"
    assert json.loads((stage / "raw_response.json").read_text())["episode_id"] == \
        episode.episode_id
    response_format = transport.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert "feedback_id" not in response_format["json_schema"]["schema"][
        "properties"]


def test_parser_accepts_trajectory_counterevidence_with_only_qa_ids(tmp_path):
    episode = fresh_episode_bundle(tmp_path)

    def sparse_counterevidence(request):
        value = valid_response(json.loads(
            request["messages"][1]["content"])["episode"]["episode_id"])
        value["counterevidence"] = [{
            "statement": "Observed: The QA evidence is not attributable to a segment.",
            "supporting_segment_ids": [],
            "supporting_qa_ids": ["benchmark/train/1"],
            "evidence_type": "trajectory",
            "confidence": "Attribution is incomplete.",
        }]
        return value

    adapter = provider(response_transport=Transport(sparse_counterevidence))
    generator = LLMEpisodeFeedbackGenerator(
        response_provider=adapter,
        artifact_resolver=FreshEpisodeFeedbackArtifactResolver(),
        policy_version=POLICY_VERSION)
    feedback = generator.generate(episode)
    assert feedback.counterevidence[0].supporting_segment_ids == ()
    assert feedback.counterevidence[0].supporting_qa_ids == (
        "benchmark/train/1",)


def test_same_request_has_same_tokens_and_identity(tmp_path):
    episode = fresh_episode_bundle(tmp_path)
    adapter = provider()
    first = prepare_and_measure(
        episode, provider_adapter=adapter,
        artifact_resolver=FreshEpisodeFeedbackArtifactResolver())
    second = prepare_and_measure(
        episode, provider_adapter=adapter,
        artifact_resolver=FreshEpisodeFeedbackArtifactResolver())
    assert first.token_statistics == second.token_statistics
    assert first.prepared_request.request_identity == \
        second.prepared_request.request_identity
    assert first.prepared_request.serialized_request == \
        second.prepared_request.serialized_request


def test_inspection_does_not_mutate_source_artifacts(tmp_path):
    episode = fresh_episode_bundle(tmp_path)
    files = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    episode_before = dumps_canonical(episode)
    adapter = provider()
    prepare_and_measure(
        episode, provider_adapter=adapter,
        artifact_resolver=FreshEpisodeFeedbackArtifactResolver())
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    assert before == after
    assert dumps_canonical(episode) == episode_before
    assert adapter.response_transport.calls == []
