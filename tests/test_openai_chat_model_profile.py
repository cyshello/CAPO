import pytest

from surrogate_rollout.optimization.openai_chat_model_profile import (
    ChatModelProfileError,
    adapt_chat_completions_body,
    is_reasoning_chat_model,
    validate_reasoning_effort,
)


def _body(model):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4096,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }


def test_gpt4o_body_is_untouched():
    body = _body("gpt-4o")
    assert adapt_chat_completions_body(body) == body


def test_gpt4o_ignores_reasoning_effort():
    body = _body("gpt-4o-mini")
    adapted = adapt_chat_completions_body(body, reasoning_effort="high")
    assert adapted == body
    assert "reasoning_effort" not in adapted


def test_gpt4o_body_never_carries_reasoning_effort():
    adapted = adapt_chat_completions_body(
        {**_body("gpt-4o"), "reasoning_effort": "high"})
    assert "reasoning_effort" not in adapted


def test_reasoning_model_drops_rejected_sampling_controls():
    adapted = adapt_chat_completions_body(
        {**_body("gpt-5-mini"), "top_p": 1.0})
    assert "temperature" not in adapted
    assert "top_p" not in adapted


def test_reasoning_model_renames_output_budget():
    adapted = adapt_chat_completions_body(_body("gpt-5-mini"))
    assert "max_tokens" not in adapted
    assert adapted["max_completion_tokens"] == 4096


def test_reasoning_effort_is_carried_only_when_requested():
    assert "reasoning_effort" not in adapt_chat_completions_body(
        _body("gpt-5-mini"))
    assert adapt_chat_completions_body(
        _body("gpt-5-mini"), reasoning_effort="minimal",
    )["reasoning_effort"] == "minimal"


def test_response_format_and_messages_survive():
    adapted = adapt_chat_completions_body(
        _body("gpt-5-mini"), reasoning_effort="high")
    assert adapted["response_format"] == {"type": "json_object"}
    assert adapted["messages"] == [{"role": "user", "content": "hi"}]


def test_input_body_is_not_mutated():
    body = _body("gpt-5-mini")
    adapt_chat_completions_body(body, reasoning_effort="low")
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0.0


@pytest.mark.parametrize("model,expected", [
    ("gpt-5-mini", True), ("gpt-5.5", True), ("o3-mini", True),
    ("gpt-4o", False), ("gpt-4o-mini", False),
])
def test_family_detection(model, expected):
    assert is_reasoning_chat_model(model) is expected


def test_unknown_reasoning_effort_is_rejected():
    with pytest.raises(ChatModelProfileError):
        validate_reasoning_effort("extreme")
    with pytest.raises(ChatModelProfileError):
        adapt_chat_completions_body(_body("gpt-5-mini"),
                                    reasoning_effort="extreme")


def test_missing_model_is_rejected():
    with pytest.raises(ChatModelProfileError):
        adapt_chat_completions_body({"messages": []})
