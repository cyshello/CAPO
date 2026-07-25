"""Transient-transport retry: what is retried, what is not, and for how long.

The failure these cover: on 2026-07-25 a full-recaption run ended 2h35m in
because one generator request could not resolve `api.openai.com`. The retry
budget was three attempts within three seconds, so the blip was fatal.
"""
from __future__ import annotations

import io
import socket
import urllib.error

import pytest
import requests

from surrogate_rollout import network_retry
from surrogate_rollout.prompt_routing.free_form_instruction_generator import (
    FreeFormGenerationError,
)
from surrogate_rollout.prompt_routing.policies.openai_free_form_generator import (
    OpenAIFreeFormInstructionGenerator,
)


class FakeClock:
    """Advances only when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


def dns_failure() -> requests.exceptions.ConnectionError:
    """The exact shape of the failure that ended the 07-25 run."""
    return requests.exceptions.ConnectionError(
        "HTTPSConnectionPool(host='api.openai.com', port=443): Max retries "
        "exceeded with url: /v1/chat/completions (Caused by "
        "NameResolutionError(\"Failed to resolve 'api.openai.com' "
        "([Errno -5] No address associated with hostname)\"))")


def http_error(code: int, *, retry_after: str | None = None
               ) -> urllib.error.HTTPError:
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError(
        "https://api.openai.com/v1/chat/completions", code, "err", headers,
        io.BytesIO(b"{}"))


# ----------------------------- classification ----------------------------- #
@pytest.mark.parametrize("exc", [
    dns_failure(),
    requests.exceptions.Timeout("read timed out"),
    socket.gaierror(-5, "No address associated with hostname"),
    urllib.error.URLError(socket.gaierror(-5, "resolver down")),
    ConnectionResetError("peer reset"),
    http_error(429),
    http_error(500),
    http_error(503),
    network_retry.TransientTransportError("overloaded"),
])
def test_transport_failures_are_transient(exc):
    assert network_retry.is_transient_exception(exc) is True


@pytest.mark.parametrize("exc", [
    http_error(400),
    http_error(401),
    http_error(404),
    http_error(422),
    FreeFormGenerationError("empty completion"),
    ValueError("provider response envelope is not an object"),
    FileNotFoundError("frame_000012.jpg"),
])
def test_rejected_requests_are_not_transient(exc):
    assert network_retry.is_transient_exception(exc) is False


def test_wrapped_cause_is_followed():
    """The DVD/updater transports re-raise `from exc`; classification follows."""
    wrapper = RuntimeError("provider transport failed")
    wrapper.__cause__ = dns_failure()
    assert network_retry.is_transient_exception(wrapper) is True


def test_permanent_error_raised_while_handling_a_transient_one_is_not_transient():
    """`__context__` must not leak a classification onto the next failure."""
    try:
        try:
            raise dns_failure()
        except requests.exceptions.ConnectionError:
            raise FreeFormGenerationError("empty completion")
    except FreeFormGenerationError as exc:
        assert exc.__context__ is not None
        assert network_retry.is_transient_exception(exc) is False


# -------------------------------- retrying -------------------------------- #
def test_retries_until_the_request_succeeds():
    clock = FakeClock()
    attempts = []

    def operation():
        attempts.append(clock.now)
        if len(attempts) < 4:
            raise dns_failure()
        return "instruction"

    result = network_retry.retry_transient(
        operation, deadline_seconds=900, initial_delay_seconds=2,
        maximum_delay_seconds=60, sleep=clock.sleep, monotonic=clock.monotonic)

    assert result == "instruction"
    assert clock.waits == [2, 4, 8]


def test_backoff_is_capped_and_gives_up_at_the_deadline():
    clock = FakeClock()
    calls = []

    def operation():
        calls.append(clock.now)
        raise dns_failure()

    with pytest.raises(requests.exceptions.ConnectionError):
        network_retry.retry_transient(
            operation, deadline_seconds=300, initial_delay_seconds=2,
            maximum_delay_seconds=10, sleep=clock.sleep,
            monotonic=clock.monotonic)

    assert max(clock.waits) == 10
    # Every wait fits inside the deadline, and the last one is the one that
    # would not have: an outage never overruns the budget it was given.
    assert sum(clock.waits) <= 300
    assert sum(clock.waits) + 10 > 300
    assert len(calls) == len(clock.waits) + 1


def test_permanent_failure_is_raised_on_the_first_attempt():
    clock = FakeClock()
    calls = []

    def operation():
        calls.append(1)
        raise FreeFormGenerationError("HTTP 400: bad request")

    with pytest.raises(FreeFormGenerationError):
        network_retry.retry_transient(
            operation, deadline_seconds=900, sleep=clock.sleep,
            monotonic=clock.monotonic)

    assert calls == [1]
    assert clock.waits == []


def test_retry_after_is_honored_but_bounded():
    clock = FakeClock()
    states = [
        network_retry.TransientTransportError("429", retry_after_seconds=45),
        network_retry.TransientTransportError("429", retry_after_seconds=9999),
        None,
    ]

    def operation():
        state = states.pop(0)
        if state is not None:
            raise state
        return "ok"

    assert network_retry.retry_transient(
        operation, deadline_seconds=3600, initial_delay_seconds=2,
        maximum_delay_seconds=60, sleep=clock.sleep,
        monotonic=clock.monotonic) == "ok"
    assert clock.waits == [
        45, network_retry.MAXIMUM_HONORED_RETRY_AFTER_SECONDS]


def test_zero_deadline_disables_retrying():
    clock = FakeClock()
    with pytest.raises(requests.exceptions.ConnectionError):
        network_retry.retry_transient(
            lambda: (_ for _ in ()).throw(dns_failure()),
            deadline_seconds=0, sleep=clock.sleep, monotonic=clock.monotonic)
    assert clock.waits == []


# --------------------------- generator integration ------------------------- #
class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None,
                 *, text: str = "", headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = headers or {}

    def json(self) -> dict:
        return self._payload


def completion(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3}}


def build_generator() -> OpenAIFreeFormInstructionGenerator:
    return OpenAIFreeFormInstructionGenerator(
        model_id="gpt-5-mini", api_key="test-key", max_tokens=4096,
        template_text="write one instruction for this segment")


def test_generator_survives_a_resolver_outage(monkeypatch):
    """The 07-25 crash, replayed: the run continues instead of dying."""
    generator = build_generator()
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        if len(calls) < 3:
            raise dns_failure()
        return FakeResponse(200, completion("describe the whiteboard text"))

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(network_retry.time, "sleep", lambda seconds: None)

    assert generator._complete((), "prompt") == "describe the whiteboard text"
    assert len(calls) == 3


def test_generator_retries_a_429_and_stops_at_a_400(monkeypatch):
    generator = build_generator()
    responses = [
        FakeResponse(429, text="rate limited", headers={"Retry-After": "1"}),
        FakeResponse(200, completion("count the visible people")),
    ]
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: responses.pop(0))
    monkeypatch.setattr(network_retry.time, "sleep", lambda seconds: None)
    assert generator._complete((), "prompt") == "count the visible people"

    attempts = []

    def rejecting_post(url, **kwargs):
        attempts.append(url)
        return FakeResponse(400, text="unsupported image")

    monkeypatch.setattr(requests, "post", rejecting_post)
    with pytest.raises(FreeFormGenerationError):
        generator._complete((), "prompt")
    # The pre-existing empty/rejection budget, not one attempt per retry second.
    assert len(attempts) == generator.retries + 1


def test_unanswered_attempts_are_not_billed(monkeypatch):
    """Token accounting stays per answered request."""
    generator = build_generator()
    recorded = []
    responses = [dns_failure(), dns_failure(),
                 FakeResponse(200, completion("track the red car"))]

    def fake_post(url, **kwargs):
        item = responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(network_retry.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        "surrogate_rollout.prompt_routing.policies.openai_free_form_generator"
        ".token_ledger.record",
        lambda component, model, usage: recorded.append((component, usage)))

    assert generator._complete((), "prompt") == "track the red car"
    assert recorded == [
        ("prompt_generator", {"prompt_tokens": 10, "completion_tokens": 3})]
