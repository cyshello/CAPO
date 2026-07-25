"""The vendored DVD text/tool call must ride out a provider 5xx.

The 2026-07-25 full-recaption run died on OpenAI's HTTP 500 ("The server had an
error processing your request"): its text matches none of the transport markers,
so the decorator printed it and returned None, and `_openai_text` turned that
non-answer into a fatal caption-worker failure. A rejected request must still
fail on the first attempt, where retrying would only hide a defect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "dvd_stack" / "dvd"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

dvd_utils = pytest.importorskip("dvd.utils")

SERVER_ERROR_BODY = (
    '{"error": {"message": "The server had an error processing your request. '
    'Sorry about that!", "type": "server_error"}}'
)


class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _answer():
    return {"choices": [{"message": {"content": " answered "}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}}


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    slept = []
    monkeypatch.setattr(dvd_utils.time, "sleep", slept.append)
    return slept


def _call(monkeypatch, responses):
    """Run one vendored call against a scripted sequence of responses."""
    attempts = []

    def post(url, headers=None, json=None, timeout=None):
        attempts.append(url)
        item = responses[min(len(attempts) - 1, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(dvd_utils.requests, "post", post)
    result = dvd_utils.call_openai_model_with_tools(
        messages=[{"role": "user", "content": "hi"}], endpoints=None,
        model_name="gpt-5-mini", api_key="sk-test")
    return result, attempts


def test_a_server_error_is_retried_and_the_answer_returned(monkeypatch,
                                                          _no_real_sleeping):
    result, attempts = _call(monkeypatch, [
        _Response(500, text=SERVER_ERROR_BODY),
        _Response(200, payload=_answer()),
    ])
    assert len(attempts) == 2
    assert result["content"] == "answered"
    assert _no_real_sleeping  # it waited before resending


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 529])
def test_every_transient_status_is_retried(monkeypatch, status):
    result, attempts = _call(monkeypatch, [
        _Response(status, text="busy"),
        _Response(200, payload=_answer()),
    ])
    assert len(attempts) == 2
    assert result["content"] == "answered"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_rejected_request_fails_on_the_first_attempt(monkeypatch, status):
    # 403 keeps its historical "Forbidden for url" retry only when the provider
    # says so in the body; a bare rejection must not be resent.
    result, attempts = _call(monkeypatch, [_Response(status, text="nope")])
    assert result is None
    assert len(attempts) == 1


def test_retrying_stops_at_the_wall_clock_deadline(monkeypatch):
    monkeypatch.setenv(dvd_utils._RETRY_DEADLINE_VARIABLE, "30")
    clock = iter([0.0] + [float(v) for v in range(1, 200)])
    monkeypatch.setattr(dvd_utils.time, "monotonic", lambda: next(clock))
    result, attempts = _call(monkeypatch, [_Response(500, text=SERVER_ERROR_BODY)])
    assert result is None
    # 30s budget at one simulated second per attempt: it gave up on time
    # instead of after a fixed number of tries.
    assert 25 <= len(attempts) <= 35


def test_the_delay_is_capped_so_a_long_incident_keeps_being_retried(
        monkeypatch, _no_real_sleeping):
    monkeypatch.setenv(dvd_utils._RETRY_DEADLINE_VARIABLE, "600")
    responses = [_Response(500, text=SERVER_ERROR_BODY)] * 12
    responses.append(_Response(200, payload=_answer()))
    result, attempts = _call(monkeypatch, responses)
    assert result["content"] == "answered"
    assert max(_no_real_sleeping) <= dvd_utils._MAXIMUM_RETRY_DELAY_SECONDS


def test_a_resolver_failure_is_still_transient(monkeypatch):
    result, attempts = _call(monkeypatch, [
        dvd_utils.requests.exceptions.ConnectionError(
            "Failed to resolve 'api.openai.com'"),
        _Response(200, payload=_answer()),
    ])
    assert len(attempts) == 2
    assert result["content"] == "answered"
