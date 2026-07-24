"""Checkpoint E2 one-call updater and provisional persistence tests."""

import hashlib
import json

import pytest

from surrogate_rollout.optimization.meta_prompt_update_execution import (
    META_PROMPT_UPDATE_STRICT_SCHEMA_NAME,
    MetaPromptUpdateExecutionError,
    OpenAICompatibleMetaPromptUpdaterBackend,
    execute_meta_prompt_update_once,
)
from surrogate_rollout.optimization.policies.episode_feedback_provider import (
    ExactProviderInputTokenCount,
)
from surrogate_rollout.optimization.meta_prompt_updater import (
    MetaPromptUpdaterParseError,
    build_meta_prompt_update_request,
    parse_meta_prompt_update_response,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from test_meta_prompt_updater import (
    CANDIDATE,
    POLICY,
    feedback_fixtures,
    parent_fixture,
    response,
)


CREATED_AT = "2026-07-20T01:02:03Z"


class Transport:
    def __init__(self, content, *, usage=None):
        self.content = content
        self.usage = usage or {"prompt_tokens": 50, "completion_tokens": 20}
        self.calls = []

    def __call__(self, body):
        self.calls.append(json.loads(dumps_canonical(body)))
        return {
            "id": "fixture-response-1",
            "model": "fixture-updater-snapshot",
            "choices": [{"message": {"content": self.content}}],
            "usage": self.usage,
        }


class FailingTransport:
    def __init__(self):
        self.calls = 0

    def __call__(self, body):
        self.calls += 1
        error = RuntimeError("fixture provider failure")
        error.raw_error = "fixture raw provider error"
        raise error


def write_sources(tmp_path):
    parent = parent_fixture()
    feedbacks = feedback_fixtures()
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(dumps_canonical(parent), encoding="utf-8")
    feedback_paths = []
    for index, feedback in enumerate(feedbacks):
        path = tmp_path / f"feedback_{index}.json"
        path.write_text(dumps_canonical(feedback), encoding="utf-8")
        feedback_paths.append(path)
    return parent, feedbacks, parent_path, tuple(feedback_paths)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backend_for(transport):
    return OpenAICompatibleMetaPromptUpdaterBackend(
        provider="fixture_provider",
        model_id="fixture-updater-model",
        maximum_output_tokens=321,
        generation_settings={"temperature": 0.0},
        updater_policy_version=POLICY,
        response_transport=transport,
    )


def test_updater_backend_fits_only_dynamic_json_values_before_transport():
    transport = Transport(dumps_canonical(response([])))

    def counter(messages):
        system = len(messages[0]["content"])
        user = len(messages[1]["content"])
        return ExactProviderInputTokenCount(system, user, system + user + 7)

    backend = OpenAICompatibleMetaPromptUpdaterBackend(
        provider="fixture_provider", model_id="fixture-updater-model",
        maximum_output_tokens=20,
        generation_settings={"temperature": 0.0},
        updater_policy_version=POLICY,
        tokenizer_identity="unicode_codepoints_v1",
        exact_token_counter=counter,
        context_limit=500,
        context_safety_margin_tokens=128,
        response_transport=transport,
    )
    original = dumps_canonical({
        "schema_version": "updater_v1",
        "feedback_id": "feedback_1",
        "diagnosis": "long evidence " * 200,
    })
    backend("fixed system instruction", original)
    sent = transport.calls[0]["messages"]
    assert sent[0]["content"] == "fixed system instruction"
    payload = json.loads(sent[1]["content"])
    assert payload["feedback_id"] == "feedback_1"
    assert "[TRUNCATED_TO_CONTEXT]" in payload["diagnosis"]
    assert backend.last_context_truncation["original_payload_hash"] != \
        backend.last_context_truncation["transmitted_payload_hash"]
    system = len(sent[0]["content"])
    user = len(sent[1]["content"])
    assert system + user + 7 + backend.maximum_output_tokens + 128 <= \
        backend.context_limit


def test_update_executes_once_and_writes_provisional_with_lineage(tmp_path):
    parent, feedbacks, parent_path, feedback_paths = write_sources(tmp_path)
    before = [digest(path) for path in (parent_path, *feedback_paths)]
    transport = Transport(dumps_canonical(response(
        [item.feedback_id for item in feedbacks])))
    backend = backend_for(transport)
    output = tmp_path / "update_run"

    result = execute_meta_prompt_update_once(
        parent_artifact_path=parent_path,
        feedback_artifact_paths=feedback_paths,
        output_directory=output,
        backend=backend,
        updater_policy_version=POLICY,
        candidate_created_at=CREATED_AT,
    )

    assert backend.call_count == len(transport.calls) == 1
    assert result.decision.decision == "update"
    candidate = json.loads((output / "provisional_meta_prompt.json").read_text())
    assert candidate == {
        "meta_prompt_id": result.candidate_meta_prompt_id,
        "parent_meta_prompt_id": parent.meta_prompt_id,
        "text": CANDIDATE,
        "created_at": CREATED_AT,
        "status": "provisional",
    }
    assert not (output / "no_update.json").exists()
    for name in (
        "updater_request.json", "provider_request.json", "raw_response.txt",
        "provider_response.json", "parsed_meta_prompt_update_result.json",
        "usage.json", "input_manifest.json", "run_manifest.json",
    ):
        assert (output / name).is_file()
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "succeeded"
    assert manifest["provider_call_count"] == 1
    assert manifest["ordered_feedback_ids"] == [
        item.feedback_id for item in feedbacks]
    assert [digest(path) for path in (parent_path, *feedback_paths)] == before


def test_no_update_writes_decision_without_candidate(tmp_path):
    parent, feedbacks, parent_path, feedback_paths = write_sources(tmp_path)
    transport = Transport(dumps_canonical(response(
        [item.feedback_id for item in feedbacks], candidate=None)))
    output = tmp_path / "no_update_run"
    result = execute_meta_prompt_update_once(
        parent_artifact_path=parent_path,
        feedback_artifact_paths=feedback_paths,
        output_directory=output,
        backend=backend_for(transport),
        updater_policy_version=POLICY,
        candidate_created_at=CREATED_AT,
    )
    assert result.decision.decision == "no_update"
    assert result.candidate_meta_prompt_id is None
    assert (output / "no_update.json").is_file()
    assert not (output / "provisional_meta_prompt.json").exists()


def test_provider_request_is_strict_and_omits_candidate_identity(tmp_path):
    _, feedbacks, parent_path, feedback_paths = write_sources(tmp_path)
    transport = Transport(dumps_canonical(response(
        [item.feedback_id for item in feedbacks])))
    backend = backend_for(transport)
    request = build_meta_prompt_update_request(
        parent_fixture(), feedbacks, updater_policy_version=POLICY)
    prepared = backend.prepare_messages(
        request.system_instruction, request.user_request)
    response_format = prepared.request_body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == \
        META_PROMPT_UPDATE_STRICT_SCHEMA_NAME
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert "candidate_meta_prompt_id" not in schema["properties"]
    assert "status" not in schema["properties"]


def test_invented_supporting_feedback_ids_do_not_fail_the_update(tmp_path):
    _, feedbacks, parent_path, feedback_paths = write_sources(tmp_path)
    raw = response([feedbacks[0].feedback_id, "feedback-missing"])
    transport = Transport(dumps_canonical(raw))
    backend = backend_for(transport)
    output = tmp_path / "dropped_ids"
    execute_meta_prompt_update_once(
        parent_artifact_path=parent_path,
        feedback_artifact_paths=feedback_paths,
        output_directory=output,
        backend=backend,
        updater_policy_version=POLICY,
        candidate_created_at=CREATED_AT,
    )
    assert backend.call_count == len(transport.calls) == 1
    # the raw response is preserved exactly; only the parsed decision drops the
    # identifier that cites nothing
    assert (output / "raw_response.txt").read_text() == dumps_canonical(raw)
    assert json.loads((output / "run_manifest.json").read_text())["status"] == \
        "succeeded"
    parsed = json.loads(
        (output / "parsed_meta_prompt_update_result.json").read_text())
    assert parsed["decision"]["supporting_feedback_ids"] == [
        feedbacks[0].feedback_id]


def test_provider_failure_is_not_retried_and_raw_error_is_written(tmp_path):
    _, _, parent_path, feedback_paths = write_sources(tmp_path)
    transport = FailingTransport()
    backend = backend_for(transport)
    output = tmp_path / "failed_provider"
    with pytest.raises(MetaPromptUpdateExecutionError, match="provider failure"):
        execute_meta_prompt_update_once(
            parent_artifact_path=parent_path,
            feedback_artifact_paths=feedback_paths,
            output_directory=output,
            backend=backend,
            updater_policy_version=POLICY,
            candidate_created_at=CREATED_AT,
        )
    assert backend.call_count == transport.calls == 1
    assert (output / "raw_error.txt").read_text() == \
        "fixture raw provider error"


def test_existing_output_directory_is_write_once_and_skips_call(tmp_path):
    _, _, parent_path, feedback_paths = write_sources(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "user.txt"
    marker.write_text("keep", encoding="utf-8")
    transport = FailingTransport()
    backend = backend_for(transport)
    with pytest.raises(MetaPromptUpdateExecutionError, match="exists"):
        execute_meta_prompt_update_once(
            parent_artifact_path=parent_path,
            feedback_artifact_paths=feedback_paths,
            output_directory=output,
            backend=backend,
            updater_policy_version=POLICY,
            candidate_created_at=CREATED_AT,
        )
    assert transport.calls == backend.call_count == 0
    assert marker.read_text() == "keep"


def test_candidate_containing_a_known_id_token_is_not_rejected():
    # The provenance-ID screen was removed. A candidate that contains a known
    # supporting-ID token -- whether embedded (no boundary) or as a whole token
    # -- now parses instead of forcing a spurious no_update.
    parent = parent_fixture()
    feedbacks = feedback_fixtures()
    request = build_meta_prompt_update_request(
        parent, feedbacks, updater_policy_version=POLICY)
    known_segment = next(
        segment
        for feedback in feedbacks
        for evidence in (*feedback.observations, *feedback.counterevidence)
        for segment in evidence.supporting_segment_ids)
    for candidate in (
            f"Preserve visible continuity for {known_segment}suffix conditions.",
            f"Preserve visible continuity for {known_segment}."):
        value = response([item.feedback_id for item in feedbacks])
        value["candidate_meta_prompt"] = candidate
        decision, candidate_id, status = parse_meta_prompt_update_response(
            dumps_canonical(value), request=request, feedbacks=feedbacks)
        assert decision.candidate_meta_prompt == candidate
        assert candidate_id and status == "provisional"


def test_runtime_availability_wording_is_left_to_the_prompt():
    """A word search cannot tell a prohibition from a requirement."""
    parent = parent_fixture()
    feedbacks = feedback_fixtures()
    request = build_meta_prompt_update_request(
        parent, feedbacks, updater_policy_version=POLICY)
    value = response([item.feedback_id for item in feedbacks])
    value["candidate_meta_prompt"] = \
        "Do not consult correctness labels or QA answers when writing the " \
        "instruction."
    decision, _, _ = parse_meta_prompt_update_response(
        dumps_canonical(value), request=request, feedbacks=feedbacks)
    assert decision.candidate_meta_prompt == value["candidate_meta_prompt"]

