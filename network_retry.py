"""Transport-level retry for transient network failures.

Why it exists: on 2026-07-25 the full-recaption run died 2h35m in, mid-baseline,
because one generator request could not resolve `api.openai.com` (`Errno -5`).
The generator's retry policy was three attempts one and two seconds apart, so a
DNS blip lasting longer than three seconds ended the run, and the caption worker
turned that into a fatal `persistent caption worker failure`.

A momentary transport failure carries no information about the request, so
repeating the identical request is not a repair or a reprompt: the retried call
sends the same bytes and, when it succeeds, produces the result the first call
would have. That distinction is what this module encodes -- only transport
failures are retried, and only until a wall-clock deadline. A rejected request
(400, 401, an unparsable body, an empty completion) is returned to the caller
untouched, because repeating it would either fail identically or hide a real
defect.

Retries here do not consume any semantic call budget: callers wrap a single
request in `retry_transient`, so budget accounting sees one invocation whether
the transport needed one attempt or nine.
"""
from __future__ import annotations

import errno
import http.client
import logging
import socket
import ssl
import time
import urllib.error
from typing import Any, Callable, TypeVar

from surrogate_rollout import config

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

# 408 request timeout, 409 conflict, 425 too early, 429 rate limit, and the 5xx
# family: all are states the same request can pass through a moment later. 529
# is the overloaded status the OpenAI and Anthropic edges both return.
TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503,
                                         504, 529})

# Socket-level conditions that a later identical request can survive. Anything
# else raised as an OSError (a missing frame file, a bad path) is a real defect.
_TRANSIENT_ERRNOS = frozenset({
    errno.ECONNABORTED, errno.ECONNREFUSED, errno.ECONNRESET, errno.EHOSTUNREACH,
    errno.ENETDOWN, errno.ENETRESET, errno.ENETUNREACH, errno.ENOTCONN,
    errno.EPIPE, errno.ETIMEDOUT, errno.EAGAIN, errno.EBUSY,
})

# Substrings of resolver/proxy failures that reach us as a bare OSError or as a
# provider SDK's own exception type, with no errno and no recognizable class.
# Matched last, only after the type checks below have all missed.
_TRANSIENT_MESSAGE_MARKERS = (
    "failed to resolve",
    "name or service not known",
    "temporary failure in name resolution",
    "no address associated with hostname",
    "connection aborted",
    "connection reset",
    "connection refused",
    "remote end closed connection",
    "server disconnected",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "temporarily unavailable",
)

# How deep to follow `raise ... from exc` when classifying. Deliberately only
# `__cause__`, never `__context__`: a permanent error raised inside an
# `except TransientTransportError:` block must not inherit its predecessor's
# classification.
_MAXIMUM_CAUSE_DEPTH = 8

# An upper bound on an honored `Retry-After`. A provider asking for an hour is
# not a transport blip, and the run's own deadline should decide instead.
MAXIMUM_HONORED_RETRY_AFTER_SECONDS = 300.0


class TransientTransportError(RuntimeError):
    """A request failed for a reason a later identical request may survive.

    Raised by call sites that classify a provider response themselves, which is
    the only place a transient HTTP status can be told from a permanent one.
    """

    def __init__(self, message: str, *,
                 retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def is_transient_http_status(status: Any) -> bool:
    try:
        return int(status) in TRANSIENT_HTTP_STATUS_CODES
    except (TypeError, ValueError):
        return False


def parse_retry_after(value: Any) -> float | None:
    """Seconds from a `Retry-After` header, when it is given as a delay.

    The HTTP-date form is ignored on purpose: the exponential backoff below is a
    safe fallback, and parsing dates would make the wait depend on clock skew.
    """
    if value is None:
        return None
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _requests_transient_types() -> tuple[type[BaseException], ...]:
    """Transient `requests` exception classes, or `()` when it is absent.

    `requests.exceptions.RequestException` subclasses `OSError`, so these have
    to be matched by class rather than by errno.
    """
    try:
        from requests import exceptions as requests_exceptions
    except Exception:  # noqa: BLE001 - requests is optional for callers
        return ()
    return (
        requests_exceptions.ConnectionError,
        requests_exceptions.Timeout,
        requests_exceptions.ChunkedEncodingError,
        requests_exceptions.ContentDecodingError,
    )


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, TransientTransportError):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        # Checked before URLError, its base class: a 404 is permanent.
        return is_transient_http_status(exc.code)
    if isinstance(exc, urllib.error.URLError):
        # urlopen wraps resolver and socket failures in this.
        return True
    if isinstance(exc, _requests_transient_types()):
        return True
    if isinstance(exc, (socket.gaierror, socket.herror, socket.timeout,
                        TimeoutError, ConnectionError,
                        http.client.RemoteDisconnected,
                        http.client.IncompleteRead,
                        ssl.SSLEOFError, ssl.SSLZeroReturnError)):
        return True
    if isinstance(exc, ssl.SSLError):
        # A handshake that was cut off, not a certificate the peer rejected.
        return not isinstance(exc, ssl.SSLCertVerificationError)
    if isinstance(exc, OSError) and exc.errno in _TRANSIENT_ERRNOS:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MESSAGE_MARKERS)


def is_transient_exception(exc: BaseException) -> bool:
    """Whether `exc`, or something it was explicitly raised from, is transport."""
    current: BaseException | None = exc
    for _ in range(_MAXIMUM_CAUSE_DEPTH):
        if current is None:
            return False
        if _is_transient(current):
            return True
        current = current.__cause__
    return False


def retry_transient(
    operation: Callable[[], T],
    *,
    description: str = "provider request",
    deadline_seconds: float | None = None,
    initial_delay_seconds: float | None = None,
    maximum_delay_seconds: float | None = None,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> T:
    """Call `operation`, repeating it while it fails on transient transport.

    Gives up as soon as the next wait would not fit inside `deadline_seconds`,
    re-raising the provider's own exception so the caller's error text and type
    are the ones a non-retryable failure would have produced. Anything not
    classified as transport propagates on its first occurrence.

    `sleep` and `monotonic` are injectable so tests can exercise the schedule
    without waiting; they are resolved here rather than as signature defaults so
    a patched `time` module still takes effect.
    """
    if sleep is None:
        sleep = time.sleep
    if monotonic is None:
        monotonic = time.monotonic
    if deadline_seconds is None:
        deadline_seconds = config.NETWORK_RETRY_DEADLINE_SECONDS
    if initial_delay_seconds is None:
        initial_delay_seconds = config.NETWORK_RETRY_INITIAL_DELAY_SECONDS
    if maximum_delay_seconds is None:
        maximum_delay_seconds = config.NETWORK_RETRY_MAXIMUM_DELAY_SECONDS

    started = monotonic()
    delay = max(float(initial_delay_seconds), 0.0)
    attempt = 1
    while True:
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - classified immediately below
            if not is_transient_exception(exc):
                raise
            requested = getattr(exc, "retry_after_seconds", None)
            if requested is None and isinstance(exc, urllib.error.HTTPError):
                headers = getattr(exc, "headers", None)
                requested = parse_retry_after(
                    headers.get("Retry-After") if headers else None)
            wait = delay if requested is None else min(
                max(float(requested), delay),
                MAXIMUM_HONORED_RETRY_AFTER_SECONDS)
            elapsed = monotonic() - started
            if elapsed + wait > float(deadline_seconds):
                LOGGER.warning(
                    "%s: giving up after %d transient failure(s) in %.1fs "
                    "(deadline %.0fs): %s: %s", description, attempt, elapsed,
                    float(deadline_seconds), type(exc).__name__, exc)
                raise
            LOGGER.warning(
                "%s: transient failure %d after %.1fs, retrying in %.1fs: "
                "%s: %s", description, attempt, elapsed, wait,
                type(exc).__name__, exc)
            sleep(wait)
            delay = min(max(delay * 2.0, 1.0), float(maximum_delay_seconds))
            attempt += 1
