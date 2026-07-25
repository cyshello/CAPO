"""Write-once reuse of one free-form generator response.

Why it exists: the generated instruction becomes part of the caption cache key
(``prompt_hash`` / ``composed_prompt_hash``), and the generator is a sampled
hosted call. A resumed run re-samples an instruction that differs by a
character, so the segment's caption misses; that caption is also the next
segment's history, so the miss cascades through the rest of the video. Measured
2026-07-25 on the full-recaption run: 587 cached caption segments, zero hits
after a restart, every one of them recomputed.

What the key holds: the rendered generator request hash — video, segment,
generator frames, serialized history, meta-prompt id/version/text, output
budget — plus the provider identity that decides the answer (endpoint, model,
backend, temperature, reasoning effort). A different meta prompt renders a
different request, so a parent and a candidate prompt can never read each
other's entries and the distinction the experiment measures is preserved. What
the key deliberately drops is sampling noise between runs, which is not a
controlled variable.

Not in the key: the instruction parser version. An entry holds the provider's
raw answer, which does not depend on how this repository parses it.

Entries are write-once. Two workers racing on the same request both keep the
first answer that lands, so a cache root can be shared across GPUs.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

from surrogate_rollout import config
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_text

CACHE_SCHEMA_VERSION = "generator_response_cache_v1"
CACHE_DIRECTORY_NAME = "generator_v1"


@dataclass(frozen=True)
class GeneratorResponseCacheKey:
    """Everything that decides the provider's answer to one request."""

    request_hash: str
    base_url: str
    model_id: str
    backend_id: str
    max_tokens: int
    temperature: float
    reasoning_effort: str | None
    schema_version: str = CACHE_SCHEMA_VERSION


def key_hash(key: GeneratorResponseCacheKey) -> str:
    return sha256_text(dumps_canonical(asdict(key)))


def entry_path(root: str, *, video_id: str, segment_id: str,
               key: GeneratorResponseCacheKey) -> str:
    """One file per request identity, under the run's own cache root."""
    safe_segment = str(segment_id).replace("/", "_")
    return os.path.join(root, CACHE_DIRECTORY_NAME, str(video_id),
                        safe_segment, f"{key_hash(key)}.json")


def load(root: str | None, *, video_id: str, segment_id: str,
         key: GeneratorResponseCacheKey) -> str | None:
    """The stored raw response for this exact request, or None.

    A stored key that does not match the requested one is reported as a miss
    rather than an error: the caller then pays for the call it was going to
    pay for anyway, which is always safe.
    """
    if not root:
        return None
    try:
        with open(entry_path(root, video_id=video_id, segment_id=segment_id,
                             key=key), encoding="utf-8") as handle:
            entry = json.load(handle)
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
        return None
    if entry.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    if entry.get("key") != json.loads(dumps_canonical(asdict(key))):
        return None
    raw = entry.get("raw_response")
    return raw if isinstance(raw, str) and raw else None


def store(root: str | None, *, video_id: str, segment_id: str,
          key: GeneratorResponseCacheKey, raw_response: str) -> str | None:
    """Persist one response. An existing entry is never overwritten."""
    if not root or not raw_response:
        return None
    path = entry_path(root, video_id=video_id, segment_id=segment_id, key=key)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "video_id": str(video_id),
        "segment_id": str(segment_id),
        "key": asdict(key),
        "raw_response": raw_response,
        "output_hash": sha256_text(raw_response),
        "stored_at_unix": time.time(),
    }
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(dumps_canonical(payload) + "\n")
    try:
        # link, not replace: whoever gets there first owns the entry.
        os.link(temporary, path)
    except FileExistsError:
        pass
    finally:
        os.unlink(temporary)
    return path


def configure_default_root(cache_root: str | None) -> str | None:
    """Point generator reuse at this run's cache root unless already set.

    Called by a run entry point before any caption worker is spawned; the
    workers inherit the variable. An explicit SR_GENERATOR_CACHE_ROOT wins, so
    an operator can send reuse somewhere else or (with an empty value) switch
    it off for one run.
    """
    if cache_root:
        os.environ.setdefault(
            config.GENERATOR_RESPONSE_CACHE_VARIABLE, str(cache_root))
    return config.generator_response_cache_root()
