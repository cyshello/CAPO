"""A refused orchestrator request falls to a second model, not to Codex.

gpt-5-mini rejected one video's transcript outright:

    400 invalid_prompt -- "your prompt was flagged as potentially violating
    our usage policy"

Reasoning models apply a stricter prompt filter than gpt-4o, and the trigger is
the video's own content, so it recurs on every retry: a deterministic stop on
that video. The existing fallback went to the Codex shim, which runs on a
ChatGPT account with separate quota -- exhausted at the time -- so the run
ended with an unanswerable QA instead.

The fallback is a second API orchestrator. Codex stays as the last resort.
"""

from pathlib import Path

import pytest

_BACKEND = (Path(__file__).resolve().parents[1] / "vendor" / "dvd_stack" /
            "dvd_backend.py")
_TOOL = {"type": "function", "function": {"name": "clip_search_tool"}}


def _make_router(responder):
    """Load make_router alone, with the transport and Codex shim stubbed."""
    source = _BACKEND.read_text(encoding="utf-8")
    segment = source[source.index("def make_router"):
                     source.index("def install_backend")]
    segment = segment[:segment.rindex("return call") + len("return call")]
    namespace = {
        "_OPENAI_TOOL_RETRIES": 1,
        "_ORIG_CALL": responder,
        "_codex_tool_call": lambda *a, **k: {"content": "CODEX",
                                             "tool_calls": None},
    }
    exec(compile(segment, str(_BACKEND), "exec"), namespace)
    return namespace["make_router"]


def _call(router):
    return router(messages=[{"role": "user", "content": "x"}], endpoints=None,
                  model_name="ignored", tools=[_TOOL])


def _responder(calls, refuse=(), text_only=()):
    def respond(**kwargs):
        model = kwargs["model_name"]
        calls.append(model)
        if model in refuse:
            return None
        if model in text_only:
            return {"content": "no tool call", "tool_calls": None}
        return {"tool_calls": [{"id": "1", "function": {
            "name": "clip_search_tool", "arguments": "{}"}}]}
    return respond


def test_a_refused_request_is_retried_on_the_fallback_model():
    calls = []
    router = _make_router(_responder(calls, refuse={"gpt-5-mini"}))(
        "gpt-5.5", "gpt-5-mini", "key", "key", "openai", "gpt-4o")
    result = _call(router)
    assert calls == ["gpt-5-mini", "gpt-4o"]
    assert result["tool_calls"]
    assert result.get("content") != "CODEX"


def test_the_primary_model_is_used_when_it_answers():
    calls = []
    router = _make_router(_responder(calls))(
        "gpt-5.5", "gpt-5-mini", "key", "key", "openai", "gpt-4o")
    assert _call(router)["tool_calls"]
    assert calls == ["gpt-5-mini"]


def test_codex_is_the_last_resort_not_the_first():
    """Both API models refused: only then does the Codex shim run."""
    calls = []
    router = _make_router(
        _responder(calls, refuse={"gpt-5-mini", "gpt-4o"}))(
            "gpt-5.5", "gpt-5-mini", "key", "key", "openai", "gpt-4o")
    assert _call(router)["content"] == "CODEX"
    assert calls == ["gpt-5-mini", "gpt-4o"]


def test_no_fallback_configured_keeps_the_previous_behaviour():
    calls = []
    router = _make_router(_responder(calls, refuse={"gpt-5-mini"}))(
        "gpt-5.5", "gpt-5-mini", "key", "key", "openai", None)
    assert _call(router)["content"] == "CODEX"
    assert calls == ["gpt-5-mini"]


def test_a_fallback_equal_to_the_primary_is_not_retried():
    calls = []
    router = _make_router(_responder(calls, refuse={"gpt-4o"}))(
        "gpt-5.5", "gpt-4o", "key", "key", "openai", "gpt-4o")
    assert _call(router)["content"] == "CODEX"
    assert calls == ["gpt-4o"]


def test_a_text_only_reply_does_not_trigger_the_fallback():
    """That is the answered-but-toolless case the retry loop already handles."""
    calls = []
    router = _make_router(_responder(calls, text_only={"gpt-5-mini"}))(
        "gpt-5.5", "gpt-5-mini", "key", "key", "openai", "gpt-4o")
    result = _call(router)
    assert result["tool_calls"] is None
    assert set(calls) == {"gpt-5-mini"}


def test_the_fallback_model_is_configurable():
    from surrogate_rollout import config
    assert config.ORCHESTRATOR_TOOL_FALLBACK_MODEL


@pytest.mark.parametrize("fallback", ["gpt-4o", "gpt-4o-mini"])
def test_any_configured_fallback_is_the_one_called(fallback):
    calls = []
    router = _make_router(_responder(calls, refuse={"gpt-5-mini"}))(
        "gpt-5.5", "gpt-5-mini", "key", "key", "openai", fallback)
    _call(router)
    assert calls[-1] == fallback
