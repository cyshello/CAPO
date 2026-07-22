import json

import pytest

from surrogate_rollout.optimization.context_budget import (
    CONTEXT_TRUNCATION_POLICY_VERSION,
    PROVIDER_CONTEXT_SAFETY_MARGIN_TOKENS,
    TRUNCATION_MARKER,
    fit_json_payload_to_token_budget,
)


def _tokens(value: str) -> int:
    return len(value)


def test_provider_context_safety_margin_exceeds_observed_accounting_drift():
    assert PROVIDER_CONTEXT_SAFETY_MARGIN_TOKENS == 2048
    assert PROVIDER_CONTEXT_SAFETY_MARGIN_TOKENS > 428


def test_dynamic_json_strings_fit_without_changing_structure_or_ids():
    payload = {
        "schema_version": "request_v1",
        "episode_id": "episode_1",
        "records": [{
            "qa_id": "qa_1",
            "statement": "dynamic evidence " * 100,
            "supporting_segment_ids": ["10_20"],
        }],
    }
    result = fit_json_payload_to_token_budget(
        payload, measure_input_tokens=_tokens, maximum_input_tokens=300)
    restored = json.loads(result.serialized_payload)

    assert len(result.serialized_payload) <= 300
    assert restored["schema_version"] == "request_v1"
    assert restored["episode_id"] == "episode_1"
    assert restored["records"][0]["qa_id"] == "qa_1"
    assert restored["records"][0]["supporting_segment_ids"] == ["10_20"]
    assert TRUNCATION_MARKER in restored["records"][0]["statement"]
    assert result.audit_metadata()["policy_version"] == \
        CONTEXT_TRUNCATION_POLICY_VERSION
    assert "payload" not in result.audit_metadata()


def test_fitting_is_deterministic_and_does_not_mutate_input():
    payload = {"id": "stable", "text": "abc " * 500}
    before = json.dumps(payload, sort_keys=True)
    first = fit_json_payload_to_token_budget(
        payload, measure_input_tokens=_tokens, maximum_input_tokens=200)
    second = fit_json_payload_to_token_budget(
        payload, measure_input_tokens=_tokens, maximum_input_tokens=200)
    assert first == second
    assert json.dumps(payload, sort_keys=True) == before


def test_fixed_structure_that_cannot_fit_fails_closed():
    with pytest.raises(ValueError, match="no truncatable strings"):
        fit_json_payload_to_token_budget(
            {"episode_id": "x" * 500}, measure_input_tokens=_tokens,
            maximum_input_tokens=20)
