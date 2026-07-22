"""Sequential history-aware routing and full-video baseline captioning."""

from __future__ import annotations

import json
import atexit
import contextlib
import multiprocessing
import os
import queue
import re
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from surrogate_rollout import config
from surrogate_rollout.cache.caption_cache import (
    assert_writable,
    build_history_aware_cache_key,
    key_as_dict,
    load_manifest,
    new_history_aware_cache_dir,
    register_cache,
)
from surrogate_rollout.mixed_views.builder import (
    caption_entry_from_parsed,
    default_merge_fn,
    write_captions_json,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.scaffold_applier import (
    composed_prompt_from_json,
    create_scaffold_applier,
)
from surrogate_rollout.prompt_routing.schemas import (
    RoutingDecision,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    SegmentContext,
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_text
from .qwen25_vl import (
    VLLM_MM_CACHE_POLICY_VERSION,
    VLLM_MM_PROCESSOR_CACHE_GB,
    VLLM_PREFIX_CACHING_ENABLED,
)


HISTORY_SCHEMA_VERSION = "frozen_local_caption_history_v1"
# Free-form (static meta-prompt) instruction generators. 'openai' is the
# default: the local 7B captioner cannot hold the "write an instruction, not a
# caption" role (measured 2026-07-22 — instruction and caption came back
# byte-identical on 4 of 5 mid-video segments, and repeated across segments).
FREE_FORM_PROVIDER_OPENAI = "openai"
FREE_FORM_PROVIDERS = (FREE_FORM_PROVIDER_OPENAI,)
DEFAULT_HISTORY_BLOCK_SECONDS = 300.0
DEFAULT_MAX_HISTORY_CAPTIONS = 30
PARALLEL_GROUP_SCHEDULER_VERSION = "history_block_gpu_scheduler_v1"
LOCAL_QWEN_BACKEND_ID = (
    "captioning.qwen25_vl.Qwen25VLCaptioner:"
    + VLLM_MM_CACHE_POLICY_VERSION)
CAPTION_OUTPUT_CONTRACT_VERSION = "caption_registry_json_output_contract_v3"
CAPTION_PARSE_SCHEMA_VERSION = "caption_registry_json_parse_result_v3"
CAPTION_PARSE_NORMALIZATION_VERSION = "caption_registry_json_or_plain_text_v1"
CAPTION_REPETITION_POLICY_VERSION = "caption_repetition_guard_v1"
CAPTION_CACHE_SCHEMA_VERSION = "history_aware_caption_cache_v5"
CAPTION_RETRY_POLICY_VERSION = "caption_registry_json_then_plain_text_retry_v1"
CAPTION_ATTEMPT_SCHEMA_VERSION = "caption_generation_attempt_v2"


def build_caption_output_contract(subject_registry_mode: str) -> str:
    # Retain the argument for call-site compatibility: both former registry
    # modes now use the canonical DVD registry-JSON contract, and an
    # unparsable response falls back to the plain-text contract below.
    if subject_registry_mode not in {"empty", "optional"}:
        raise ValueError("subject_registry_mode must be 'empty' or 'optional'")
    return """

CAPTION_OUTPUT_CONTRACT (this supersedes any earlier output-format wording):
Output template:
{
  "clip_start_time": CLIP_START_TIME,
  "clip_end_time": CLIP_END_TIME,
  "subject_registry": {
    <subject_i>: {
      "name": <fill with short identity if name is unknown>,
      "appearance": <list of appearance descriptions>,
      "identity": <list of identity descriptions>,
      "first_seen": <timestamp>
    },
    ...
  },
  "clip_description": <smooth and detailed natural narration of the video clip>
}
- Return only that JSON object. Do not add markdown fences or prose around it.
- Do not repeat a sentence or phrase to fill the response.
""".rstrip()


def build_plain_text_caption_output_contract() -> str:
    """Description-only fallback contract used after registry JSON fails."""
    return """

CAPTION_OUTPUT_CONTRACT (this supersedes any earlier output-format wording):
- Return only a plain-text visual description of the current clip.
- Do not emit JSON, keys, markdown fences, timestamps, or a subject registry.
- Do not repeat a sentence or phrase to fill the response.
""".rstrip()


CAPTION_OUTPUT_CONTRACT = build_caption_output_contract(
    config.CAPTION_SUBJECT_REGISTRY_MODE)
PLAIN_TEXT_CAPTION_OUTPUT_CONTRACT = build_plain_text_caption_output_contract()


class _LazyLocalQwenBackend:
    """Avoid loading Qwen in the parent before GPU worker processes start."""

    def caption(self, *args: Any, **kwargs: Any) -> str:
        return get_local_qwen_backend().caption(*args, **kwargs)


class _PersistentRawVLMProxy:
    """dvd_backend captioner facade backed by the persistent GPU pool."""

    def __init__(self, dispatcher: Any) -> None:
        self.dispatcher = dispatcher

    def caption(self, images: Any, prompt: str, **kwargs: Any) -> str:
        return self.dispatcher.caption_raw_vlm(
            images=images, prompt=prompt, **kwargs)


def get_local_qwen_backend() -> Any:
    """Return DVD's configured, process-shared Qwen2.5-VL/vLLM captioner."""
    import sys

    for path in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from dvd_backend import get_captioner

    return get_captioner()


def _subtitle_path(sample: Mapping[str, Any]) -> str | None:
    if not config.USE_TRANSCRIPT:
        return None
    return sample.get("extra", {}).get("subtitle_path")


def _segment_times(segment_id: str) -> tuple[float, float]:
    try:
        start, end = (float(value) for value in segment_id.split("_", 1))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"history-aware DVD segment_id must be '<start>_<end>': {segment_id!r}"
        ) from exc
    if start < 0 or end <= start:
        raise ValueError(f"invalid segment interval: {segment_id!r}")
    return start, end


def _hhmmss(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def build_history_snapshot(
    *,
    segment_id: str,
    block_seconds: float,
    preceding: list[dict[str, str]],
    max_history_captions: int,
) -> dict[str, Any]:
    if block_seconds <= 0:
        raise ValueError("history_block_seconds must be positive")
    if max_history_captions < 1:
        raise ValueError("max_history_captions must be positive")
    start, _ = _segment_times(segment_id)
    block_index = int(start // block_seconds)
    block_start = block_index * block_seconds
    bounded = tuple(preceding[-max_history_captions:])
    serialized_value = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "block_index": block_index,
        "block_start_seconds": block_start,
        "block_end_seconds": block_start + block_seconds,
        "max_history_captions": max_history_captions,
        "preceding_captions": bounded,
    }
    serialized = dumps_canonical(serialized_value)
    return {
        "segment_id": segment_id,
        **serialized_value,
        "preceding_segment_ids": tuple(item["segment_id"] for item in bounded),
        "history": bounded,
        "serialized_history": serialized,
        "history_hash": sha256_text(serialized),
        "source": "sequential_history_aware_baseline",
    }


def _sentence_chunks(text: str) -> tuple[str, ...]:
    return tuple(
        chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text)
        if chunk.strip())


def _repetition_error(text: str) -> str | None:
    """Flag deterministic degeneration without rejecting non-empty captions."""
    sentences = _sentence_chunks(text)
    normalized = tuple(
        " ".join(re.findall(r"[\w'-]+", item.casefold()))
        for item in sentences)
    run = 1
    for index in range(1, len(normalized)):
        if normalized[index] and normalized[index] == normalized[index - 1]:
            run += 1
            if run >= 3:
                return "repeated_sentence"
        else:
            run = 1

    tokens = re.findall(r"[\w'-]+", text.casefold())
    ngram_size = 8
    if len(tokens) < 48:
        return None
    starts: dict[tuple[str, ...], list[int]] = {}
    for index in range(len(tokens) - ngram_size + 1):
        starts.setdefault(tuple(tokens[index:index + ngram_size]), []).append(index)
    covered: set[int] = set()
    for positions in starts.values():
        if len(positions) < 4:
            continue
        for start in positions:
            covered.update(range(start, start + ngram_size))
    if len(covered) / len(tokens) >= 0.80:
        return "repeated_ngram_majority"
    return None


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```$", re.DOTALL)


def _strip_code_fence(text: str) -> tuple[str, bool]:
    """Inner text of one surrounding markdown fence, plus whether one existed."""
    match = _FENCE_RE.match(text)
    if match is None:
        return text, False
    return match.group(1).strip(), True


def _parse_registry_json(text: str) -> dict[str, Any] | None:
    """The canonical DVD caption artifact, or None when the text is not one.

    Timestamps are optional here: `clip_start_time` / `clip_end_time` are
    filled from the segment interval downstream, not trusted from the model.
    """
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    description = value.get("clip_description")
    if not isinstance(description, str) or not description.strip():
        return None
    registry = value.get("subject_registry")
    return {
        "clip_description": description.strip(),
        "subject_registry": registry if isinstance(registry, dict) else {},
    }


def _parse_caption_output_with_metadata(
    raw: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The canonical caption artifact from registry JSON, else from plain text.

    Registry JSON is tried first (optionally inside one markdown fence). When
    the response is not a usable registry object the whole stripped text
    becomes `clip_description`, which is what the plain-text fallback contract
    asks the captioner for.
    """
    if not isinstance(raw, str):
        return {}, {
            "schema_version": CAPTION_PARSE_SCHEMA_VERSION,
            "status": "invalid", "error": "invalid_output_type",
            "normalizations": [],
            "repetition_policy_version": CAPTION_REPETITION_POLICY_VERSION,
        }
    text = raw.strip()
    normalizations = ([] if text == raw
                      else ["strip_surrounding_whitespace"])
    base = {
        "schema_version": CAPTION_PARSE_SCHEMA_VERSION,
        "normalizations": normalizations,
        "repetition_policy_version": CAPTION_REPETITION_POLICY_VERSION,
    }
    if not text:
        return {}, {**base, "status": "invalid", "error": "empty_output"}
    unfenced, fenced = _strip_code_fence(text)
    if fenced:
        normalizations = [*normalizations, "strip_markdown_fence"]
        base = {**base, "normalizations": normalizations}
    parsed = _parse_registry_json(unfenced)
    if parsed is not None:
        repetition_warning = _repetition_error(parsed["clip_description"])
        return parsed, {
            **base, "status": "valid", "error": None,
            "parse_path": "registry_json",
            "subject_registry_present": bool(parsed["subject_registry"]),
            "quality_warning": repetition_warning,
            "sentence_count": len(
                _sentence_chunks(parsed["clip_description"])),
        }
    repetition_warning = _repetition_error(text)
    return {"clip_description": text, "subject_registry": {}}, {
        **base, "status": "valid", "error": None,
        "parse_path": "plain_text",
        "subject_registry_present": False,
        "quality_warning": repetition_warning,
        "sentence_count": len(_sentence_chunks(text)),
    }


def _is_registry_json_parse(parsed: Mapping[str, Any],
                            parse_result: Mapping[str, Any]) -> bool:
    """True when the captioner returned a usable registry-JSON object."""
    return bool(parsed) and parse_result.get("parse_path") == "registry_json"


def _parse_caption_output(raw: str) -> dict[str, Any]:
    """Compatibility wrapper for callers which only consume parsed output."""
    return _parse_caption_output_with_metadata(raw)[0]


def routing_decision_from_json(d: Mapping[str, Any]) -> RoutingDecision:
    return RoutingDecision(
        video_id=d["video_id"],
        segment_id=d["segment_id"],
        bank_version=d["bank_version"],
        router_version=d["router_version"],
        selected_prompt_ids=tuple(d.get("selected_prompt_ids") or ()),
        prompt_scores=d.get("prompt_scores") or {},
        matched_rule_ids=tuple(d.get("matched_rule_ids") or ()),
        selection_order=tuple(d.get("selection_order") or ()),
        confidence=d.get("confidence"),
        used_fallback=bool(d.get("used_fallback", False)),
        fallback_reason=d.get("fallback_reason"),
        decision_payload=d.get("decision_payload") or {},
    )


def build_free_form_generator(
    provider: str, *, model_id: str | None = None,
    max_tokens: int | None = None, template_text: str | None = None,
    meta_prompt_id: str | None = None, backend_id: str | None = None,
) -> Any:
    """The configured free-form instruction generator. Unknown providers abort.

    'openai' composes through the replace-body policy shared with the private
    baseline repo (generated text owns the caption prompt's task section, 0.5
    fps half-resolution generator frames). There is no local-Qwen fallback.
    """
    if provider == FREE_FORM_PROVIDER_OPENAI:
        from surrogate_rollout.prompt_routing.policies.openai_free_form_generator import (
            DEFAULT_MODEL_ID,
            OpenAIFreeFormInstructionGenerator,
        )
        from surrogate_rollout.prompt_routing import static_meta_replace_body as smrb

        return OpenAIFreeFormInstructionGenerator(
            model_id=model_id or DEFAULT_MODEL_ID,
            max_tokens=(smrb.DEFAULT_GENERATOR_MAX_TOKENS
                        if max_tokens is None else max_tokens),
            caption_fps=config.SAMPLE_FPS, template_text=template_text,
            meta_prompt_id=meta_prompt_id, backend_id=backend_id)
    raise ValueError(
        f"unknown free-form generator provider {provider!r} "
        f"(supported: {list(FREE_FORM_PROVIDERS)})")


def _history_blocks(
    clip_index: list[tuple[str, dict]], block_seconds: float,
) -> tuple[tuple[int, tuple[tuple[str, dict], ...]], ...]:
    """Partition temporal segments by the exact frozen-history boundary."""
    if block_seconds <= 0:
        raise ValueError("history_block_seconds must be positive")
    grouped: dict[int, list[tuple[str, dict]]] = {}
    for segment_id, clip_info in clip_index:
        block_index = int(_segment_times(segment_id)[0] // block_seconds)
        grouped.setdefault(block_index, []).append((segment_id, clip_info))
    return tuple((index, tuple(grouped[index])) for index in sorted(grouped))


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _seed_manifest_fragment(source: str | None, target: str) -> None:
    """Give a worker the read-only guards without sharing an append target."""
    existing: dict[str, dict[str, Any]] = {}
    for row in load_manifest(source):
        existing[os.path.abspath(row["cache_dir"])] = row
    if os.path.exists(target):
        for row in _read_jsonl(target):
            existing[os.path.abspath(row["cache_dir"])] = row
    payload = "".join(
        json.dumps(existing[key], sort_keys=True, ensure_ascii=False) + "\n"
        for key in sorted(existing)
    )
    os.makedirs(os.path.dirname(target), exist_ok=True)
    _atomic_write_text(target, payload)


def _append_worker_event(path: str | None, value: Mapping[str, Any]) -> None:
    """Append one worker-owned diagnostic event without affecting artifacts."""
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {"unix_time": time.time(), **dict(value)}
    with open(path, "a", encoding="utf-8", buffering=1) as handle:
        handle.write(dumps_canonical(record) + "\n")


def _parallel_history_worker(
    gpu: str,
    command_queue: Any,
    result_queue: Any,
    text_backend: str,
    use_openai_tools: bool,
    routing_mode: str = "free_form_generator",
    worker_log_path: str | None = None,
) -> None:
    """Own one GPU/model instance and serve caption tasks until shutdown."""
    _append_worker_event(worker_log_path, {
        "event": "process_start", "gpu": gpu, "pid": os.getpid(),
        "routing_mode": routing_mode})
    try:
        from surrogate_rollout.evaluation.dvd_qa import ensure_backend
        from surrogate_rollout.prompt_routing.persistence import (
            scaffold_contract_from_json,
            scaffold_policy_from_json,
        )

        ensure_backend(
            gpu, preload_captioner=True,
            text_backend=text_backend,
            use_openai_tools=use_openai_tools,
        )
        _append_worker_event(worker_log_path, {
            "event": "backend_ready", "gpu": gpu, "pid": os.getpid()})
        builder = HistoryAwareBaselineCaptionViewBuilder.from_local_qwen(
            routing_mode=routing_mode)
        while True:
            command = command_queue.get()
            request_id = command["request_id"]
            _append_worker_event(worker_log_path, {
                "event": "request_start", "gpu": gpu,
                "request_id": request_id, "request_type": command["type"]})
            try:
                if command["type"] == "shutdown":
                    engine = getattr(builder.segment_captioner.vlm, "llm", None)
                    engine_core = getattr(
                        getattr(engine, "llm_engine", None), "engine_core", None)
                    cleanup_warning = None
                    if engine_core is not None:
                        try:
                            engine_core.shutdown()
                        except BaseException as exc:
                            cleanup_warning = (
                                f"{type(exc).__name__}: {exc}")
                    result_queue.put({
                        "request_id": request_id, "gpu": gpu,
                        "status": "shutdown_acknowledged",
                        "cleanup_warning": cleanup_warning,
                    })
                    _append_worker_event(worker_log_path, {
                        "event": "process_shutdown", "gpu": gpu,
                        "request_id": request_id})
                    return
                if command["type"] == "build_blocks":
                    common = command["common"]
                    generator_configuration = common.get(
                        "free_form_generator_configuration")
                    if generator_configuration is not None:
                        provider = generator_configuration["provider"]
                        builder.free_form_generator = build_free_form_generator(
                            provider,
                            model_id=generator_configuration["model_id"],
                            max_tokens=generator_configuration["max_tokens"],
                            template_text=generator_configuration[
                                "template_text"],
                            meta_prompt_id=generator_configuration[
                                "meta_prompt_id"],
                            backend_id=generator_configuration["backend_id"])
                    # Keep the captioner's view consistent with the routing
                    # mode this build request actually uses: under the static
                    # meta-prompt condition it sees only the composed prompt.
                    builder.segment_captioner.caption_sees_history = (
                        generator_configuration is None)
                    scaffold_policy = scaffold_policy_from_json(
                        common["scaffold_policy"])
                    scaffold_contract = scaffold_contract_from_json(
                            common["scaffold_contract"])
                    completed = []
                    for block_index, block_clips, block_root, fragment_path in \
                            command["jobs"]:
                        _append_worker_event(worker_log_path, {
                            "event": "block_start", "gpu": gpu,
                            "request_id": request_id,
                            "block_index": block_index,
                            "segment_count": len(block_clips),
                            "block_root": os.path.abspath(block_root)})
                        artifact = builder.build(
                            sample=common["sample"], clip_index=list(block_clips),
                            bank_version=common["bank_version"],
                            router_version=common["router_version"],
                            scaffold_policy=scaffold_policy,
                            scaffold_contract=scaffold_contract,
                            base_prompt_template=common["base_prompt_template"],
                            merge_prompt=common["merge_prompt"],
                            work_root=block_root,
                            history_block_seconds=common["history_block_seconds"],
                            max_history_captions=common["max_history_captions"],
                            candidate_cache_root=common["candidate_cache_root"],
                            cache_manifest_path=fragment_path,
                            caption_run_identity_hash=common[
                                "caption_run_identity_hash"],
                        )
                        completed.append(
                            (block_index, artifact.routing_manifest_path))
                        _append_worker_event(worker_log_path, {
                            "event": "block_complete", "gpu": gpu,
                            "request_id": request_id,
                            "block_index": block_index,
                            "routing_manifest_path":
                                artifact.routing_manifest_path})
                    payload = {"completed": completed}
                elif command["type"] == "caption_segment":
                    payload = command["payload"]
                    result = builder.segment_captioner.caption(
                        sample=payload["sample"], video_id=payload["video_id"],
                        segment_id=payload["segment_id"],
                        clip_info=payload["clip_info"],
                        composed_prompt=composed_prompt_from_json(
                            payload["composed_prompt"]),
                        history_snapshot=payload["history_snapshot"],
                        merge_prompt=payload["merge_prompt"],
                        cache_root=payload["cache_root"],
                        cache_manifest_path=payload["worker_manifest_path"],
                        intervention_identity_hash=payload[
                            "intervention_identity_hash"],
                    )
                    payload = {
                        "caption_result": {
                            "parsed": dict(result.parsed),
                            "cache_key": dict(result.cache_key),
                            "cache_dir": result.cache_dir,
                            "result_path": result.result_path,
                            "cache_hit": result.cache_hit,
                            "caption_seconds": result.caption_seconds,
                            "model_call_count": result.model_call_count,
                            "retry_count": result.retry_count,
                        },
                    }
                elif command["type"] == "raw_vlm":
                    raw_payload = command["payload"]
                    text = builder.segment_captioner.vlm.caption(
                        raw_payload["images"], raw_payload["prompt"],
                        **raw_payload["kwargs"],
                    )
                    payload = {"raw_output": text}
                else:
                    raise ValueError(f"unknown worker command: {command['type']}")
                result_queue.put({
                    "request_id": request_id, "gpu": gpu, **payload,
                })
                _append_worker_event(worker_log_path, {
                    "event": "request_complete", "gpu": gpu,
                    "request_id": request_id,
                    "request_type": command["type"]})
            except BaseException as exc:
                failure = {
                    "request_id": request_id, "gpu": gpu,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                _append_worker_event(worker_log_path, {
                    "event": "request_error", **failure})
                result_queue.put(failure)
    except BaseException as exc:
        failure = {
            "request_id": "startup", "gpu": gpu,
            "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        }
        _append_worker_event(worker_log_path, {
            "event": "startup_error", **failure})
        result_queue.put(failure)


@dataclass(frozen=True)
class HistoryAwareCaptionResult:
    parsed: Mapping[str, Any]
    cache_key: Mapping[str, Any]
    cache_dir: str
    result_path: str
    cache_hit: bool
    caption_seconds: float
    model_call_count: int = 0
    retry_count: int = 0


class HistoryAwareSegmentCaptioner:
    """Per-segment caption adapter with mandatory frozen-history identity."""

    def __init__(self, vlm: Any, *, caption_model_id: str | None = None,
                 backend_id: str | None = None,
                 remote_dispatcher: Any | None = None,
                 caption_sees_history: bool = True) -> None:
        self.vlm = vlm
        self.caption_model_id = caption_model_id or config.CAPTION_MODEL_ID
        self.backend_id = backend_id or (
            f"{type(vlm).__module__}.{type(vlm).__qualname__}")
        self.remote_dispatcher = remote_dispatcher
        # When False the captioner reads ONLY the composed prompt and the
        # segment's own frames — the frozen history reaches it exclusively
        # through the generated instruction. Required by the static
        # meta-prompt condition (the private baseline repo captions that way);
        # the property-bank path keeps the historical default so its existing
        # captions and caches stay valid.
        self.caption_sees_history = bool(caption_sees_history)

    @property
    def configuration_identity(self) -> Mapping[str, Any]:
        return {
            "provider": "local_qwen_vlm",
            "model": self.caption_model_id,
            "backend": self.backend_id,
            "decoding": as_json_dict(config.CAPTION_DECODING),
            "caption_decoding_policy_version":
                config.CAPTION_DECODING_POLICY_VERSION,
            "caption_decoding_hash": config.decoding_hash(),
            "caption_output_contract_version": CAPTION_OUTPUT_CONTRACT_VERSION,
            "caption_output_contract_hash": sha256_text(
                CAPTION_OUTPUT_CONTRACT),
            "caption_parse_schema_version": CAPTION_PARSE_SCHEMA_VERSION,
            "caption_parse_normalization_version":
                CAPTION_PARSE_NORMALIZATION_VERSION,
            "caption_repetition_policy_version":
                CAPTION_REPETITION_POLICY_VERSION,
            "caption_retry_policy_version": CAPTION_RETRY_POLICY_VERSION,
            "caption_parse_max_retries": config.CAPTION_PARSE_MAX_RETRIES,
            "caption_artifact_shape": {"clip_description": "string"},
            "caption_sees_history": self.caption_sees_history,
            "vllm_multimodal_cache_policy_version":
                VLLM_MM_CACHE_POLICY_VERSION,
            "vllm_mm_processor_cache_gb": VLLM_MM_PROCESSOR_CACHE_GB,
            "vllm_enable_prefix_caching": VLLM_PREFIX_CACHING_ENABLED,
            "real_model": True,
        }

    def caption(
        self,
        *,
        sample: Mapping[str, Any],
        video_id: str,
        segment_id: str,
        clip_info: Mapping[str, Any],
        composed_prompt: Any,
        history_snapshot: Mapping[str, Any],
        merge_prompt: str,
        cache_root: str | None = None,
        cache_manifest_path: str | None = None,
        intervention_identity_hash: str | None = None,
    ) -> HistoryAwareCaptionResult:
        if self.remote_dispatcher is not None:
            return self.remote_dispatcher.caption_segment(
                sample=sample, video_id=video_id, segment_id=segment_id,
                clip_info=clip_info, composed_prompt=composed_prompt,
                history_snapshot=history_snapshot, merge_prompt=merge_prompt,
                cache_root=cache_root,
                cache_manifest_path=cache_manifest_path,
                intervention_identity_hash=intervention_identity_hash,
            )
        key = build_history_aware_cache_key(
            video_id=video_id,
            video_path=str(sample["video_path"]),
            caption_prompt=composed_prompt.prompt_text,
            merge_prompt=merge_prompt,
            subtitle_path=_subtitle_path(sample),
            segment_id=segment_id,
            history_hash=str(history_snapshot["history_hash"]),
            composed_prompt_hash=composed_prompt.prompt_hash,
            bank_version=composed_prompt.bank_version,
            router_version=composed_prompt.router_version,
            scaffold_version=composed_prompt.scaffold_version,
            contract_version=composed_prompt.contract_version,
            backend_id=self.backend_id,
            history_config_hash=sha256_text(dumps_canonical({
                "schema_version": HISTORY_SCHEMA_VERSION,
                "block_seconds": history_snapshot["block_end_seconds"]
                - history_snapshot["block_start_seconds"],
                "max_history_captions": history_snapshot["max_history_captions"],
                "boundary_rule": "floor_segment_start_div_block_seconds",
                "caption_output_contract_version":
                    CAPTION_OUTPUT_CONTRACT_VERSION,
                "caption_output_contract_hash":
                    sha256_text(CAPTION_OUTPUT_CONTRACT),
                "caption_decoding_policy_version":
                    config.CAPTION_DECODING_POLICY_VERSION,
                "caption_decoding_hash": config.decoding_hash(),
                "caption_parse_schema_version": CAPTION_PARSE_SCHEMA_VERSION,
                "caption_parse_normalization_version":
                    CAPTION_PARSE_NORMALIZATION_VERSION,
                "caption_repetition_policy_version":
                    CAPTION_REPETITION_POLICY_VERSION,
                "caption_sees_history": self.caption_sees_history,
            })),
            caption_model_id=self.caption_model_id,
            intervention_identity_hash=intervention_identity_hash,
        )
        cache_dir = new_history_aware_cache_dir(key, cache_root)
        result_path = os.path.join(cache_dir, "caption.json")
        assert_writable(cache_dir, cache_manifest_path)
        os.makedirs(cache_dir, exist_ok=True)
        register_cache({
            "video_id": video_id,
            "cache_dir": cache_dir,
            "key": key_as_dict(key),
            "read_only": False,
            "legacy": False,
            "history_dependent": True,
        }, cache_manifest_path)
        retry_result_path = os.path.join(
            cache_dir, "parse_retries", CAPTION_RETRY_POLICY_VERSION,
            "caption.json")
        cached_invalid = None
        if os.path.exists(result_path):
            with open(result_path) as f:
                cached = json.load(f)
            if cached.get("cache_key") != key_as_dict(key):
                raise RuntimeError("history-aware cache file identity mismatch")
            parsed_from_cache = cached.get("parsed") or {}
            cached_policy_matches = (
                cached.get("caption_retry_policy_version") ==
                CAPTION_RETRY_POLICY_VERSION and
                cached.get("caption_parse_normalization_version") ==
                CAPTION_PARSE_NORMALIZATION_VERSION)
            if parsed_from_cache and cached_policy_matches:
                return HistoryAwareCaptionResult(
                    parsed=parsed_from_cache, cache_key=key_as_dict(key),
                    cache_dir=cache_dir, result_path=result_path,
                    cache_hit=True, caption_seconds=0.0,
                    model_call_count=0,
                    retry_count=int(cached.get("retry_count") or 0))
            cached_invalid = cached
            if os.path.exists(retry_result_path):
                with open(retry_result_path) as f:
                    retried = json.load(f)
                if retried.get("cache_key") != key_as_dict(key) or \
                        retried.get("caption_retry_policy_version") != \
                        CAPTION_RETRY_POLICY_VERSION or \
                        retried.get("caption_parse_normalization_version") != \
                        CAPTION_PARSE_NORMALIZATION_VERSION:
                    raise RuntimeError(
                        "history-aware retry cache file identity mismatch")
                return HistoryAwareCaptionResult(
                    parsed=retried.get("parsed") or {},
                    cache_key=key_as_dict(key), cache_dir=cache_dir,
                    result_path=retry_result_path, cache_hit=True,
                    caption_seconds=0.0, model_call_count=0,
                    retry_count=int(retried.get("retry_count") or 0))

        start, end = _segment_times(segment_id)
        transcript = str(clip_info.get("transcript") or "No transcript.")
        body = (composed_prompt.prompt_text
                .replace("TRANSCRIPT_PLACEHOLDER", transcript)
                .replace("CLIP_START_TIME", _hhmmss(start))
                .replace("CLIP_END_TIME", _hhmmss(end)))

        def render(contract: str) -> str:
            text = body + contract
            if self.caption_sees_history:
                text += ("\n\nFROZEN_PRECEDING_CAPTION_HISTORY_JSON:\n"
                         + str(history_snapshot["serialized_history"]))
            return text

        prompt = render(CAPTION_OUTPUT_CONTRACT)
        plain_text_prompt = render(PLAIN_TEXT_CAPTION_OUTPUT_CONTRACT)
        frames = clip_info.get("files") or clip_info.get("frames") or ()
        if isinstance(frames, (str, os.PathLike)):
            frames = (frames,)
        frame_paths = tuple(os.fspath(item) for item in frames)
        attempts = []
        model_call_count = 0
        elapsed = 0.0
        parsed: dict[str, Any] = {}
        raw = ""
        parse_result: dict[str, Any] = {}
        accepted = False
        if cached_invalid is not None:
            raw = str(cached_invalid.get("raw_output") or "")
            parsed, parse_result = _parse_caption_output_with_metadata(raw)
            accepted = _is_registry_json_parse(parsed, parse_result)
            attempts.append({
                "schema_version": CAPTION_ATTEMPT_SCHEMA_VERSION,
                "attempt_index": 1,
                "source": "immutable_cached_initial_attempt",
                "raw_output": raw,
                "parse_result": parse_result,
                "parsed_valid": bool(parsed),
                "parse_accepted": accepted,
                "caption_seconds": 0.0,
            })
            call_indices = (
                () if accepted else
                range(1, config.CAPTION_PARSE_MAX_RETRIES + 1))
        else:
            call_indices = range(0, config.CAPTION_PARSE_MAX_RETRIES + 1)
        for retry_index in call_indices:
            # The first CAPTION_JSON_ATTEMPTS calls ask for the canonical DVD
            # registry JSON; the remaining calls fall back to a
            # description-only plain-text contract, whose output is accepted
            # as written.
            wants_registry_json = model_call_count < config.CAPTION_JSON_ATTEMPTS
            attempt_prompt = prompt if wants_registry_json else plain_text_prompt
            t0 = time.monotonic()
            raw = self.vlm.caption(
                frame_paths, attempt_prompt,
                max_tokens=int(config.CAPTION_DECODING["max_tokens"]),
                temperature=float(config.CAPTION_DECODING["temperature"]),
                top_p=float(config.CAPTION_DECODING["top_p"]),
                repetition_penalty=float(
                    config.CAPTION_DECODING["repetition_penalty"]),
            )
            attempt_seconds = time.monotonic() - t0
            elapsed += attempt_seconds
            model_call_count += 1
            parsed, parse_result = _parse_caption_output_with_metadata(raw)
            accepted = bool(parsed) and (
                not wants_registry_json
                or _is_registry_json_parse(parsed, parse_result))
            attempts.append({
                "schema_version": CAPTION_ATTEMPT_SCHEMA_VERSION,
                "attempt_index": len(attempts) + 1,
                "source": "model_call",
                "retry_index": retry_index,
                "output_contract": (
                    "registry_json" if wants_registry_json else "plain_text"),
                "raw_output": raw,
                "parse_result": parse_result,
                "parsed_valid": bool(parsed),
                "parse_accepted": accepted,
                "caption_seconds": attempt_seconds,
            })
            if accepted:
                break
        retry_count = max(0, len(attempts) - 1)
        payload = {
            "schema_version": CAPTION_CACHE_SCHEMA_VERSION,
            "cache_key": key_as_dict(key),
            "history_hash": history_snapshot["history_hash"],
            "rendered_prompt_hash": sha256_text(prompt),
            "caption_output_contract_version":
                CAPTION_OUTPUT_CONTRACT_VERSION,
            "caption_output_contract_hash": sha256_text(
                CAPTION_OUTPUT_CONTRACT),
            "caption_plain_text_contract_hash": sha256_text(
                PLAIN_TEXT_CAPTION_OUTPUT_CONTRACT),
            "rendered_plain_text_prompt_hash": sha256_text(plain_text_prompt),
            "caption_json_attempts": config.CAPTION_JSON_ATTEMPTS,
            "caption_parse_path": parse_result.get("parse_path"),
            "caption_parse_accepted": accepted,
            "caption_decoding_policy_version":
                config.CAPTION_DECODING_POLICY_VERSION,
            "caption_decoding_hash": config.decoding_hash(),
            "caption_parse_schema_version": CAPTION_PARSE_SCHEMA_VERSION,
            "caption_parse_normalization_version":
                CAPTION_PARSE_NORMALIZATION_VERSION,
            "caption_repetition_policy_version":
                CAPTION_REPETITION_POLICY_VERSION,
            "caption_retry_policy_version": CAPTION_RETRY_POLICY_VERSION,
            "caption_parse_max_retries": config.CAPTION_PARSE_MAX_RETRIES,
            "attempt_count": len(attempts),
            "retry_count": retry_count,
            "retry_exhausted": (
                not accepted
                and retry_count >= config.CAPTION_PARSE_MAX_RETRIES),
            "attempts": attempts,
            "parse_result": parse_result,
            "raw_output": raw,
            "parsed": parsed,
        }
        persisted_result_path = (
            retry_result_path if cached_invalid is not None else result_path)
        _atomic_write_text(persisted_result_path, json.dumps(
            payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
        return HistoryAwareCaptionResult(
            parsed=parsed, cache_key=key_as_dict(key), cache_dir=cache_dir,
            result_path=persisted_result_path, cache_hit=False,
            caption_seconds=elapsed, model_call_count=model_call_count,
            retry_count=retry_count)


@dataclass(frozen=True)
class HistoryAwareBaselineViewArtifact:
    video_id: str
    captions_path: str
    captions_hash: str
    database_path: str
    routed_view_path: str
    routing_manifest_path: str
    frames_path: str
    frozen_histories_path: str
    segment_ids: tuple[str, ...]
    histories: tuple[Mapping[str, Any], ...]
    caption_call_count: int
    caption_cache_hits: int
    router_call_count: int
    resumed_segment_count: int


class HistoryAwareBaselineCaptionViewBuilder:
    """Route, compose, and caption sequentially inside temporal blocks."""

    def __init__(
        self,
        *,
        router: Any,
        segment_captioner: HistoryAwareSegmentCaptioner,
        merge_fn: Callable[[list], object] | None = None,
        captions_writer: Callable[[dict, str], tuple[str, str]] = write_captions_json,
        parallel_gpus: tuple[str, ...] = (),
        parallel_executor: Callable[..., tuple[str, ...]] | None = None,
        worker_text_backend: str = config.DVD_TEXT_BACKEND,
        worker_use_openai_tools: bool = config.DVD_USE_OPENAI_TOOLS,
        free_form_generator: Any | None = None,
        worker_result_timeout_seconds: float = 86400.0,
        worker_health_poll_seconds: float = 1.0,
        worker_log_directory: str | None = None,
    ) -> None:
        normalized_gpus = tuple(str(value).strip() for value in parallel_gpus)
        if any(not value for value in normalized_gpus) or \
                len(normalized_gpus) != len(set(normalized_gpus)):
            raise ValueError("parallel_gpus must contain unique non-empty GPU IDs")
        if worker_result_timeout_seconds <= 0 or worker_health_poll_seconds <= 0:
            raise ValueError("worker timeout and health poll must be positive")
        self.router = router
        self.segment_captioner = segment_captioner
        # Opt-in free-form baseline (ablation): when set, the per-segment
        # adaptive instruction is generated on the fly instead of routed from
        # the prompt bank; everything downstream (scaffold, captioner, caches,
        # DVD evaluation) is shared. None (default) = existing behavior.
        self.free_form_generator = free_form_generator
        self.merge_fn = merge_fn or default_merge_fn
        self.captions_writer = captions_writer
        self.parallel_gpus = normalized_gpus
        self.parallel_executor = parallel_executor
        self.worker_text_backend = worker_text_backend
        self.worker_use_openai_tools = worker_use_openai_tools
        self.worker_result_timeout_seconds = float(worker_result_timeout_seconds)
        self.worker_health_poll_seconds = float(worker_health_poll_seconds)
        self.worker_log_directory = (os.path.abspath(worker_log_directory)
                                     if worker_log_directory else None)
        self._pool_lock = threading.RLock()
        self._pool_context: Any | None = None
        self._pool_result_queue: Any | None = None
        self._pool_result_thread: threading.Thread | None = None
        self._pool_pending: dict[str, queue.Queue] = {}
        self._pool_startup_errors: list[dict[str, Any]] = []
        self._pool_processes: dict[str, Any] = {}
        self._pool_command_queues: dict[str, Any] = {}
        self._pool_request_counter = 0
        self._pool_caption_cursor = 0
        self._pool_affinity = threading.local()
        self._pool_registered_with_atexit = False
        self._parent_dvd_backend: Any | None = None
        self._parent_previous_captioner: Any | None = None

    @classmethod
    def from_local_qwen(
        cls, *, parallel_gpus: tuple[str, ...] = (),
        routing_mode: str = "free_form_generator",
        free_form_provider: str = FREE_FORM_PROVIDER_OPENAI,
        free_form_model_id: str | None = None,
        worker_result_timeout_seconds: float = 86400.0,
        worker_health_poll_seconds: float = 1.0,
        worker_log_directory: str | None = None,
    ):
        if routing_mode != "free_form_generator":
            raise ValueError(
                f"unknown routing_mode {routing_mode!r} "
                "(supported: 'free_form_generator')")
        vlm = (_LazyLocalQwenBackend() if parallel_gpus
               else get_local_qwen_backend())
        free_form_generator = build_free_form_generator(
            free_form_provider, model_id=free_form_model_id)
        # Static meta-prompt condition: the captioner reads only the generated
        # prompt plus the segment's own frames. History reaches it exclusively
        # through the generated instruction, exactly as in the baseline repo.
        builder = cls(
            router=None,
            segment_captioner=HistoryAwareSegmentCaptioner(
                vlm, backend_id=LOCAL_QWEN_BACKEND_ID,
                caption_sees_history=False),
            parallel_gpus=parallel_gpus,
            free_form_generator=free_form_generator,
            worker_result_timeout_seconds=worker_result_timeout_seconds,
            worker_health_poll_seconds=worker_health_poll_seconds,
            worker_log_directory=worker_log_directory,
        )
        if parallel_gpus:
            builder.segment_captioner.remote_dispatcher = builder
        return builder

    def build(
        self,
        *,
        sample: Mapping[str, Any],
        clip_index: list[tuple[str, dict]],
        bank_version: str,
        router_version: str,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        base_prompt_template: str,
        merge_prompt: str,
        work_root: str,
        history_block_seconds: float = DEFAULT_HISTORY_BLOCK_SECONDS,
        max_history_captions: int = DEFAULT_MAX_HISTORY_CAPTIONS,
        candidate_cache_root: str | None = None,
        cache_manifest_path: str | None = None,
        caption_run_identity_hash: str | None = None,
        worker_gpus: tuple[str, ...] | None = None,
    ) -> HistoryAwareBaselineViewArtifact:
        kwargs = {
            "sample": sample, "clip_index": clip_index,
            "bank_version": bank_version,
            "router_version": router_version,
            "scaffold_policy": scaffold_policy,
            "scaffold_contract": scaffold_contract,
            "base_prompt_template": base_prompt_template,
            "merge_prompt": merge_prompt, "work_root": work_root,
            "history_block_seconds": history_block_seconds,
            "max_history_captions": max_history_captions,
            "candidate_cache_root": candidate_cache_root,
            "cache_manifest_path": cache_manifest_path,
            "caption_run_identity_hash": caption_run_identity_hash,
        }
        blocks = _history_blocks(clip_index, history_block_seconds)
        selected_gpus = self.parallel_gpus if worker_gpus is None else tuple(
            str(value) for value in worker_gpus)
        if not set(selected_gpus) <= set(self.parallel_gpus):
            raise ValueError("worker_gpus must be a subset of the persistent pool")
        if selected_gpus and blocks:
            return self._build_parallel(
                blocks=blocks, worker_gpus=selected_gpus, **kwargs)
        return self._build_sequential(**kwargs)

    def _build_sequential(
        self,
        *,
        sample: Mapping[str, Any],
        clip_index: list[tuple[str, dict]],
        bank_version: str,
        router_version: str,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        base_prompt_template: str,
        merge_prompt: str,
        work_root: str,
        history_block_seconds: float = DEFAULT_HISTORY_BLOCK_SECONDS,
        max_history_captions: int = DEFAULT_MAX_HISTORY_CAPTIONS,
        candidate_cache_root: str | None = None,
        cache_manifest_path: str | None = None,
        caption_run_identity_hash: str | None = None,
    ) -> HistoryAwareBaselineViewArtifact:
        video_id = str(sample.get("extra", {}).get("videoID")
                       or sample.get("sample_id"))
        if not video_id:
            raise ValueError("sample has no video ID")
        segment_ids = tuple(key for key, _ in clip_index)
        if not segment_ids or len(segment_ids) != len(set(segment_ids)):
            raise ValueError("clip index must contain unique segments")
        starts = tuple(_segment_times(key)[0] for key in segment_ids)
        if starts != tuple(sorted(starts)):
            raise ValueError("clip index must be in temporal order")

        work_root = os.path.abspath(work_root)
        state_dir = os.path.join(work_root, "segment_state")
        os.makedirs(state_dir, exist_ok=True)
        if self.free_form_generator is None:
            raise ValueError(
                "history-aware baseline requires a free-form generator")
        scaffold = create_scaffold_applier(
            scaffold_policy, base_prompt_template=base_prompt_template)
        histories: list[dict[str, Any]] = []
        decisions = []
        composed_prompts = []
        parsed_by_segment: dict[str, dict] = {}
        cache_rows: list[dict[str, Any]] = []
        block_histories: dict[int, list[dict[str, str]]] = {}
        caption_calls = cache_hits = router_calls = resumed_segments = 0

        for segment_id, clip_info in clip_index:
            start, end = _segment_times(segment_id)
            block_index = int(start // history_block_seconds)
            preceding = block_histories.setdefault(block_index, [])
            history = build_history_snapshot(
                segment_id=segment_id,
                block_seconds=history_block_seconds,
                preceding=preceding,
                max_history_captions=max_history_captions,
            )
            frames = clip_info.get("files") or clip_info.get("frames") or ()
            if isinstance(frames, (str, os.PathLike)):
                frames = (frames,)
            context = SegmentContext(
                video_id=video_id,
                segment_id=segment_id,
                timestamp_start=start,
                timestamp_end=end,
                segment_features={
                    "frame_references": tuple(os.path.abspath(os.fspath(item))
                                              for item in frames),
                },
                history_summary=history["serialized_history"],
                metadata={"history_hash": history["history_hash"]},
            )
            state_identity_payload = {
                "context": context,
                "bank_version": bank_version,
                "router_version": router_version,
                "scaffold_policy": scaffold_policy,
                "scaffold_contract": scaffold_contract,
                "base_prompt_template_hash": sha256_text(base_prompt_template),
                "merge_prompt_hash": sha256_text(merge_prompt),
                "caption_model_id": self.segment_captioner.caption_model_id,
                "caption_backend_id": self.segment_captioner.backend_id,
                "caption_decoding_hash": config.decoding_hash(),
                "caption_decoding_policy_version":
                    config.CAPTION_DECODING_POLICY_VERSION,
                "caption_output_contract_version":
                    CAPTION_OUTPUT_CONTRACT_VERSION,
                "caption_output_contract_hash": sha256_text(
                    CAPTION_OUTPUT_CONTRACT),
                "caption_parse_schema_version": CAPTION_PARSE_SCHEMA_VERSION,
                "caption_parse_normalization_version":
                    CAPTION_PARSE_NORMALIZATION_VERSION,
                "caption_repetition_policy_version":
                    CAPTION_REPETITION_POLICY_VERSION,
                "caption_run_identity_hash": caption_run_identity_hash,
                "history_configuration": {
                    "schema_version": HISTORY_SCHEMA_VERSION,
                    "block_seconds": history_block_seconds,
                    "max_history_captions": max_history_captions,
                    "boundary_rule": "floor_segment_start_div_block_seconds",
                },
            }
            state_identity_payload["free_form_generator_identity"] = dict(
                self.free_form_generator.configuration_identity)
            state_identity = sha256_text(
                dumps_canonical(state_identity_payload))
            state_path = os.path.join(
                state_dir, segment_id.replace("/", "_") + ".json")
            state = None
            if os.path.exists(state_path):
                with open(state_path) as f:
                    candidate = json.load(f)
                result_path = candidate.get("caption_result_path", "")
                if (candidate.get("state_identity") == state_identity
                        and os.path.exists(result_path)):
                    state = candidate

            if state is not None:
                decision = routing_decision_from_json(state["routing_decision"])
                composed = composed_prompt_from_json(state["composed_prompt"])
                with open(state["caption_result_path"]) as f:
                    cached_payload = json.load(f)
                parsed = cached_payload.get("parsed") or {}
                caption_result = {
                    "cache_key": cached_payload["cache_key"],
                    "cache_dir": os.path.dirname(state["caption_result_path"]),
                    "result_path": state["caption_result_path"],
                    "cache_hit": True,
                    "caption_seconds": 0.0,
                    "model_call_count": 0,
                    "retry_count": int(cached_payload.get("retry_count") or 0),
                }
                resumed_segments += 1
            else:
                from surrogate_rollout.prompt_routing.free_form_instruction_generator import (
                    build_free_form_selection,
                )

                decision, selected_entries, exchange = (
                    build_free_form_selection(
                        generator=self.free_form_generator,
                        context=context, clip_info=clip_info,
                        bank_version=bank_version))
                router_calls += 1
                composed = scaffold.apply(
                    context=context,
                    selected_entries=selected_entries,
                    routing_decision=decision,
                    scaffold_policy=scaffold_policy,
                    scaffold_contract=scaffold_contract,
                )
                if not composed.is_valid:
                    raise ValueError(
                        f"invalid composed prompt for {segment_id}: "
                        + "; ".join(composed.validation_errors))
                result = self.segment_captioner.caption(
                    sample=sample,
                    video_id=video_id,
                    segment_id=segment_id,
                    clip_info=clip_info,
                    composed_prompt=composed,
                    history_snapshot=history,
                    merge_prompt=merge_prompt,
                    cache_root=candidate_cache_root,
                    cache_manifest_path=cache_manifest_path,
                    intervention_identity_hash=caption_run_identity_hash,
                )
                parsed = dict(result.parsed)
                caption_calls += result.model_call_count
                cache_hits += int(result.cache_hit)
                caption_result = {
                    "cache_key": result.cache_key,
                    "cache_dir": result.cache_dir,
                    "result_path": result.result_path,
                    "cache_hit": result.cache_hit,
                    "caption_seconds": result.caption_seconds,
                    "model_call_count": result.model_call_count,
                    "retry_count": result.retry_count,
                }
                state_payload = {
                    "schema_version": "history_aware_baseline_segment_v1",
                    "state_identity": state_identity,
                    "history": history,
                    "routing_decision": as_json_dict(decision),
                    "composed_prompt": as_json_dict(composed),
                    "router_exchange": as_json_dict(exchange) if exchange else None,
                    "caption_cache_key": as_json_dict(result.cache_key),
                    "caption_result_path": result.result_path,
                }
                _atomic_write_text(state_path, json.dumps(
                    state_payload, sort_keys=True, ensure_ascii=False,
                    indent=2) + "\n")

            decisions.append(decision)
            composed_prompts.append(composed)
            histories.append(history)
            parsed_by_segment[segment_id] = parsed
            cache_rows.append({"segment_id": segment_id, **caption_result})
            preceding.append({
                "segment_id": segment_id,
                "caption": str(parsed.get("clip_description") or ""),
            })

        captions: dict[str, Any] = {}
        registries = []
        for segment_id in segment_ids:
            parsed = parsed_by_segment[segment_id]
            entry = caption_entry_from_parsed(parsed)
            if entry is not None:
                captions[segment_id] = entry
                if parsed.get("subject_registry"):
                    registries.append(parsed["subject_registry"])
        captions["subject_registry"] = self.merge_fn(registries)
        view_hash = sha256_text(dumps_canonical({
            "video_id": video_id,
            "histories": histories,
            "composed_prompts": composed_prompts,
        }))
        view_dir = os.path.join(work_root, f"captions_history_{view_hash[:12]}")
        captions_path, captions_hash = self.captions_writer(captions, view_dir)

        def write_jsonl(name: str, rows: list[Any]) -> str:
            path = os.path.join(work_root, name)
            _atomic_write_text(path, "".join(
                dumps_canonical(row) + "\n" for row in rows))
            return os.path.abspath(path)

        decisions_path = write_jsonl("routing_decisions.jsonl", decisions)
        composed_path = write_jsonl("composed_prompts.jsonl", composed_prompts)
        histories_path = write_jsonl("frozen_histories.jsonl", histories)
        cache_keys_path = write_jsonl("caption_cache_keys.jsonl", cache_rows)
        frames_path = os.path.join(work_root, "frames.json")
        persisted_frames = {}
        for key, info in clip_index:
            values = info.get("files") or info.get("frames") or ()
            if isinstance(values, (str, os.PathLike)):
                values = (values,)
            persisted_frames[key] = tuple(
                os.path.abspath(os.fspath(item)) for item in values)
        _atomic_write_text(
            frames_path, dumps_canonical(persisted_frames) + "\n")
        routing_manifest_path = os.path.join(work_root, "routing_manifest.json")
        routing_manifest = {
            "schema_version": "history_aware_baseline_routing_v1",
            "video_id": video_id,
            "segment_ids": segment_ids,
            "routing_decisions_path": decisions_path,
            "composed_prompts_path": composed_path,
            "frozen_histories_path": histories_path,
            "caption_cache_keys_path": cache_keys_path,
            "history_block_seconds": history_block_seconds,
            "max_history_captions": max_history_captions,
            "router_calls": router_calls,
            "caption_calls": caption_calls,
            "caption_cache_hits": cache_hits,
            "resumed_segments": resumed_segments,
        }
        _atomic_write_text(routing_manifest_path, json.dumps(
            routing_manifest, sort_keys=True, ensure_ascii=False,
            indent=2) + "\n")
        routed_view_path = os.path.join(view_dir, "routed_view.json")
        _atomic_write_text(routed_view_path, json.dumps({
            **routing_manifest,
            "view_hash": view_hash,
            "captions_path": captions_path,
            "captions_hash": captions_hash,
        }, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
        return HistoryAwareBaselineViewArtifact(
            video_id=video_id,
            captions_path=captions_path,
            captions_hash=captions_hash,
            database_path=os.path.join(
                work_root, f"database_c{captions_hash[:16]}.json"),
            routed_view_path=os.path.abspath(routed_view_path),
            routing_manifest_path=os.path.abspath(routing_manifest_path),
            frames_path=os.path.abspath(frames_path),
            frozen_histories_path=histories_path,
            segment_ids=segment_ids,
            histories=tuple(histories),
            caption_call_count=caption_calls,
            caption_cache_hits=cache_hits,
            router_call_count=router_calls,
            resumed_segment_count=resumed_segments,
        )

    def _build_parallel(
        self,
        *,
        blocks: tuple[tuple[int, tuple[tuple[str, dict], ...]], ...],
        sample: Mapping[str, Any],
        clip_index: list[tuple[str, dict]],
        bank_version: str,
        router_version: str,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        base_prompt_template: str,
        merge_prompt: str,
        work_root: str,
        history_block_seconds: float,
        max_history_captions: int,
        candidate_cache_root: str | None,
        cache_manifest_path: str | None,
        caption_run_identity_hash: str | None,
        worker_gpus: tuple[str, ...],
    ) -> HistoryAwareBaselineViewArtifact:
        work_root = os.path.abspath(work_root)
        parallel_root = os.path.join(work_root, "parallel_history_blocks")
        os.makedirs(parallel_root, exist_ok=True)
        active_gpus = worker_gpus[:min(len(worker_gpus), len(blocks))]
        assigned: dict[str, list[tuple[int, tuple[tuple[str, dict], ...], str, str]]] = {
            gpu: [] for gpu in active_gpus
        }
        fragment_paths: dict[str, str] = {}
        for worker_index, gpu in enumerate(active_gpus):
            fragment = os.path.join(
                parallel_root, f"worker_{worker_index:02d}_cache_manifest.jsonl")
            _seed_manifest_fragment(cache_manifest_path, fragment)
            fragment_paths[gpu] = fragment
        for position, (block_index, block_clips) in enumerate(blocks):
            gpu = active_gpus[position % len(active_gpus)]
            block_root = os.path.join(
                parallel_root, f"block_{block_index:06d}")
            assigned[gpu].append(
                (block_index, block_clips, block_root, fragment_paths[gpu]))

        common = {
            "sample": dict(sample),
            "bank_version": bank_version,
            "router_version": router_version,
            "scaffold_policy": as_json_dict(scaffold_policy),
            "scaffold_contract": as_json_dict(scaffold_contract),
            "base_prompt_template": base_prompt_template,
            "merge_prompt": merge_prompt,
            "history_block_seconds": history_block_seconds,
            "max_history_captions": max_history_captions,
            "candidate_cache_root": candidate_cache_root,
            "caption_run_identity_hash": caption_run_identity_hash,
            "text_backend": self.worker_text_backend,
            "use_openai_tools": self.worker_use_openai_tools,
        }
        if self.free_form_generator is not None:
            # Persistent workers must receive the exact parent/candidate
            # meta-prompt for every build request.  The worker process is
            # intentionally long-lived, so startup-time routing_mode alone is
            # not sufficient when a paired confirmation switches policies.
            common["free_form_generator_configuration"] = {
                "provider": str(
                    self.free_form_generator.configuration_identity.get(
                        "provider", FREE_FORM_PROVIDER_OPENAI)),
                "template_text": self.free_form_generator.template,
                "meta_prompt_id": self.free_form_generator.meta_prompt_id,
                "max_tokens": self.free_form_generator.max_tokens,
                "model_id": self.free_form_generator.model_id,
                "backend_id": self.free_form_generator.backend_id,
            }
        assignments = tuple((gpu, tuple(assigned[gpu])) for gpu in active_gpus)
        if self.parallel_executor is not None:
            manifest_paths = tuple(self.parallel_executor(assignments, common))
        else:
            manifest_paths = self._run_gpu_workers(assignments, common)
        if len(manifest_paths) != len(blocks):
            raise RuntimeError(
                "parallel caption workers did not complete every history block")

        artifact = self._merge_parallel_blocks(
            sample=sample, clip_index=clip_index,
            block_manifest_paths=manifest_paths, work_root=work_root,
            history_block_seconds=history_block_seconds,
            max_history_captions=max_history_captions,
            active_gpus=active_gpus,
        )
        global_existing = {
            os.path.abspath(row["cache_dir"])
            for row in load_manifest(cache_manifest_path)
        }
        for fragment in fragment_paths.values():
            for row in load_manifest(fragment):
                cache_dir = os.path.abspath(row["cache_dir"])
                if cache_dir not in global_existing:
                    register_cache(row, cache_manifest_path)
                    global_existing.add(cache_dir)
        return artifact

    def _run_gpu_workers(
        self,
        assignments: tuple[tuple[str, tuple[Any, ...]], ...],
        common: Mapping[str, Any],
    ) -> tuple[str, ...]:
        request_ids = []
        for gpu, jobs in assignments:
            request_id = self._register_pool_request("blocks")
            request_ids.append(request_id)
            try:
                self._pool_command_queues[gpu].put({
                    "request_id": request_id, "type": "build_blocks",
                    "jobs": jobs, "common": common,
                })
            except BaseException:
                self._discard_pool_request(request_id)
                raise
        results = [self._wait_pool_result(request_id)
                   for request_id in request_ids]
        completed = []
        for item in results:
            completed.extend(item["completed"])
        return tuple(path for _, path in sorted(completed))

    def _next_pool_request_id(self, prefix: str) -> str:
        self._pool_request_counter += 1
        return f"{prefix}-{self._pool_request_counter:08d}"

    def _register_pool_request(self, prefix: str) -> str:
        with self._pool_lock:
            self._ensure_worker_pool()
            if self._pool_startup_errors:
                raise RuntimeError(
                    "persistent caption worker startup failure: "
                    + dumps_canonical(self._pool_startup_errors[0]))
            request_id = self._next_pool_request_id(prefix)
            self._pool_pending[request_id] = queue.Queue(maxsize=1)
            return request_id

    def _discard_pool_request(self, request_id: str) -> None:
        with self._pool_lock:
            self._pool_pending.pop(request_id, None)

    def _pool_process_diagnostics(self) -> dict[str, Mapping[str, Any]]:
        return {str(gpu): {
            "pid": getattr(process, "pid", None),
            "alive": bool(process.is_alive()),
            "exitcode": getattr(process, "exitcode", None),
        } for gpu, process in self._pool_processes.items()}

    def _wait_pool_result(
        self, request_id: str, *, timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        with self._pool_lock:
            waiter = self._pool_pending.get(request_id)
        if waiter is None:
            raise RuntimeError(f"unknown persistent-pool request: {request_id}")
        limit = (self.worker_result_timeout_seconds
                 if timeout_seconds is None else float(timeout_seconds))
        deadline = time.monotonic() + limit
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "persistent caption worker timed out with no truncation "
                        f"or retry: request={request_id}, limit_seconds={limit}, "
                        "processes=" + dumps_canonical(
                            self._pool_process_diagnostics()))
                try:
                    item = waiter.get(timeout=min(
                        self.worker_health_poll_seconds, remaining))
                    break
                except queue.Empty:
                    diagnostics = self._pool_process_diagnostics()
                    dead = {gpu: value for gpu, value in diagnostics.items()
                            if not value["alive"]}
                    if dead:
                        raise RuntimeError(
                            "persistent caption worker died while waiting: "
                            f"request={request_id}, dead=" +
                            dumps_canonical(dead))
        finally:
            self._discard_pool_request(request_id)
        if item.get("request_id") == "startup" or item.get("error"):
            raise RuntimeError(
                "persistent caption worker failure: " + dumps_canonical(item))
        return item

    def _route_pool_results(self) -> None:
        result_queue = self._pool_result_queue
        if result_queue is None:
            return
        while True:
            item = result_queue.get()
            request_id = str(item.get("request_id"))
            if request_id == "__parent_stop__":
                return
            with self._pool_lock:
                if request_id == "startup":
                    self._pool_startup_errors.append(item)
                    waiters = tuple(self._pool_pending.values())
                else:
                    waiter = self._pool_pending.get(request_id)
                    waiters = (waiter,) if waiter is not None else ()
            for target in waiters:
                target.put(item)

    def _ensure_worker_pool(self) -> None:
        if self._pool_processes:
            dead = [gpu for gpu, process in self._pool_processes.items()
                    if not process.is_alive()]
            if dead:
                raise RuntimeError(f"persistent caption workers died: {dead}")
            return
        self._pool_context = multiprocessing.get_context("spawn")
        self._pool_result_queue = self._pool_context.Queue()
        self._pool_startup_errors.clear()
        for gpu in self.parallel_gpus:
            command_queue = self._pool_context.Queue()
            process = self._pool_context.Process(
                target=_parallel_history_worker,
                args=(gpu, command_queue, self._pool_result_queue,
                      self.worker_text_backend, self.worker_use_openai_tools,
                      "free_form_generator",
                      (os.path.join(self.worker_log_directory,
                                    f"worker_gpu_{gpu}.jsonl")
                       if self.worker_log_directory else None)),
                name=f"history-caption-gpu-{gpu}",
            )
            process.start()
            self._pool_command_queues[gpu] = command_queue
            self._pool_processes[gpu] = process
        self._pool_result_thread = threading.Thread(
            target=self._route_pool_results,
            name="history-caption-result-router", daemon=True)
        self._pool_result_thread.start()
        try:
            self._install_parent_vlm_proxy()
        except BaseException:
            self.close_worker_pool()
            raise
        if not self._pool_registered_with_atexit:
            atexit.register(self.close_worker_pool)
            self._pool_registered_with_atexit = True

    def start_worker_pool(self) -> None:
        """Start the run-scoped pool without performing a model request."""
        if not self.parallel_gpus:
            raise RuntimeError("persistent worker pool requires configured GPUs")
        with self._pool_lock:
            self._ensure_worker_pool()

    @contextlib.contextmanager
    def worker_affinity(self, gpu: str):
        """Pin all nested caption/frame-inspection calls to one pool worker."""
        gpu = str(gpu)
        if gpu not in self.parallel_gpus:
            raise ValueError("worker affinity must target a configured GPU")
        previous = getattr(self._pool_affinity, "gpu", None)
        self._pool_affinity.gpu = gpu
        try:
            yield
        finally:
            if previous is None:
                del self._pool_affinity.gpu
            else:
                self._pool_affinity.gpu = previous

    def _select_caption_gpu(self) -> str:
        preferred = getattr(self._pool_affinity, "gpu", None)
        if preferred is not None:
            return str(preferred)
        gpu = self.parallel_gpus[
            self._pool_caption_cursor % len(self.parallel_gpus)]
        self._pool_caption_cursor += 1
        return gpu

    def _install_parent_vlm_proxy(self) -> None:
        import dvd_backend

        existing_captioner = getattr(dvd_backend, "_captioner", None)
        if existing_captioner is not None and not isinstance(
                existing_captioner, _PersistentRawVLMProxy):
            raise RuntimeError(
                "parent Qwen captioner was initialized before persistent pool")
        self._parent_dvd_backend = dvd_backend
        self._parent_previous_captioner = existing_captioner
        dvd_backend._captioner = _PersistentRawVLMProxy(self)

    def _restore_parent_vlm_proxy(self) -> None:
        if self._parent_dvd_backend is not None and isinstance(
                getattr(self._parent_dvd_backend, "_captioner", None),
                _PersistentRawVLMProxy):
            self._parent_dvd_backend._captioner = self._parent_previous_captioner
        self._parent_dvd_backend = None
        self._parent_previous_captioner = None

    def caption_segment(self, **kwargs: Any) -> HistoryAwareCaptionResult:
        """Dispatch selective captioning to the already-loaded GPU pool."""
        with self._pool_lock:
            self._ensure_worker_pool()
            gpu = self._select_caption_gpu()
            global_manifest = kwargs.get("cache_manifest_path")
            manifest_base = global_manifest or config.CACHE_MANIFEST_PATH
            worker_index = self.parallel_gpus.index(gpu)
            worker_manifest = (
                f"{manifest_base}.persistent_worker_{worker_index:02d}.jsonl")
            _seed_manifest_fragment(global_manifest, worker_manifest)
            request_id = self._register_pool_request("caption")
            payload = {
                **kwargs,
                "sample": dict(kwargs["sample"]),
                "clip_info": dict(kwargs["clip_info"]),
                "composed_prompt": as_json_dict(kwargs["composed_prompt"]),
                "history_snapshot": dict(kwargs["history_snapshot"]),
                "worker_manifest_path": worker_manifest,
            }
            payload.pop("cache_manifest_path", None)
        try:
            self._pool_command_queues[gpu].put({
                "request_id": request_id, "type": "caption_segment",
                "payload": payload,
            })
            item = self._wait_pool_result(request_id)
        except BaseException:
            self._discard_pool_request(request_id)
            raise
        with self._pool_lock:
            self._merge_cache_fragment(worker_manifest, global_manifest)
        return HistoryAwareCaptionResult(**item["caption_result"])

    def caption_raw_vlm(
        self, *, images: Any, prompt: str, **kwargs: Any,
    ) -> str:
        """Route DVD frame inspection through an existing Qwen worker."""
        with self._pool_lock:
            self._ensure_worker_pool()
            gpu = self._select_caption_gpu()
            request_id = self._register_pool_request("raw-vlm")
            self._pool_command_queues[gpu].put({
                "request_id": request_id, "type": "raw_vlm",
                "payload": {
                    "images": tuple(os.fspath(item) for item in images),
                    "prompt": prompt, "kwargs": dict(kwargs),
                },
            })
        return str(self._wait_pool_result(request_id)["raw_output"])

    @staticmethod
    def _merge_cache_fragment(fragment: str, target: str | None) -> None:
        existing = {
            os.path.abspath(row["cache_dir"]) for row in load_manifest(target)
        }
        for row in load_manifest(fragment):
            cache_dir = os.path.abspath(row["cache_dir"])
            if cache_dir not in existing:
                register_cache(row, target)
                existing.add(cache_dir)

    def close_worker_pool(self) -> None:
        """Explicit run-boundary shutdown for persistent Qwen workers."""
        with self._pool_lock:
            if not self._pool_processes:
                return
            request_ids = set()
            for gpu, command_queue in self._pool_command_queues.items():
                request_id = self._register_pool_request("shutdown")
                request_ids.add(request_id)
                command_queue.put({
                    "request_id": request_id, "type": "shutdown",
                })
        cleanup_error = None
        try:
            results = [self._wait_pool_result(
                request_id, timeout_seconds=30.0) for request_id in request_ids]
            with self._pool_lock:
                warnings = [
                    f"GPU {item['gpu']}: {item['cleanup_warning']}"
                    for item in results if item.get("cleanup_warning")]
                if warnings:
                    cleanup_error = "; ".join(warnings)
        except BaseException as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._pool_lock:
                processes = tuple(self._pool_processes.values())
                command_queues = tuple(self._pool_command_queues.values())
                result_queue = self._pool_result_queue
                for process in processes:
                    process.join(timeout=30.0)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=10.0)
                if result_queue is not None:
                    result_queue.put({"request_id": "__parent_stop__"})
                if self._pool_result_thread is not None:
                    self._pool_result_thread.join(timeout=5.0)
                    if self._pool_result_thread.is_alive():
                        message = "result-router thread did not stop"
                        cleanup_error = (
                            f"{cleanup_error}; {message}" if cleanup_error else message)
                # multiprocessing.Queue owns non-daemon feeder resources. Explicitly
                # close and join them so interpreter exit never waits on an implicit
                # queue finalizer after the scientific iteration has completed.
                for channel in (*command_queues, result_queue):
                    if channel is None:
                        continue
                    try:
                        channel.close()
                        channel.join_thread()
                    except BaseException as exc:
                        message = f"queue cleanup {type(exc).__name__}: {exc}"
                        cleanup_error = (
                            f"{cleanup_error}; {message}" if cleanup_error else message)
                self._pool_processes.clear()
                self._pool_command_queues.clear()
                self._pool_result_thread = None
                self._pool_result_queue = None
                self._pool_pending.clear()
                self._restore_parent_vlm_proxy()
                if self._pool_registered_with_atexit:
                    atexit.unregister(self.close_worker_pool)
                    self._pool_registered_with_atexit = False
        if cleanup_error:
            print(
                "CAPTION_WORKER_CLEANUP_WARNING " + cleanup_error,
                flush=True,
            )

    def _merge_parallel_blocks(
        self,
        *,
        sample: Mapping[str, Any],
        clip_index: list[tuple[str, dict]],
        block_manifest_paths: tuple[str, ...],
        work_root: str,
        history_block_seconds: float,
        max_history_captions: int,
        active_gpus: tuple[str, ...],
    ) -> HistoryAwareBaselineViewArtifact:
        video_id = str(sample.get("extra", {}).get("videoID")
                       or sample.get("sample_id"))
        segment_ids = tuple(segment_id for segment_id, _ in clip_index)
        by_segment: dict[str, dict[str, Any]] = {}
        totals = {"router_calls": 0, "caption_calls": 0,
                  "caption_cache_hits": 0, "resumed_segments": 0}
        for manifest_path in sorted(
                block_manifest_paths,
                key=lambda path: int(os.path.basename(os.path.dirname(path))
                                     .split("_")[-1])):
            with open(manifest_path) as handle:
                manifest = json.load(handle)
            if manifest.get("video_id") != video_id:
                raise RuntimeError("parallel block belongs to another video")
            decisions = _read_jsonl(manifest["routing_decisions_path"])
            composed = _read_jsonl(manifest["composed_prompts_path"])
            histories = _read_jsonl(manifest["frozen_histories_path"])
            cache_rows = _read_jsonl(manifest["caption_cache_keys_path"])
            lengths = {len(decisions), len(composed), len(histories), len(cache_rows)}
            if lengths != {len(manifest["segment_ids"])}:
                raise RuntimeError("parallel block artifacts have unequal row counts")
            for values in zip(decisions, composed, histories, cache_rows):
                decision, prompt, history, cache_row = values
                segment_id = str(history["segment_id"])
                if segment_id in by_segment or cache_row["segment_id"] != segment_id:
                    raise RuntimeError("parallel block segment identity collision")
                with open(cache_row["result_path"]) as handle:
                    parsed = json.load(handle).get("parsed") or {}
                by_segment[segment_id] = {
                    "decision": decision, "composed": prompt,
                    "history": history, "cache": cache_row, "parsed": parsed,
                }
            for key in totals:
                totals[key] += int(manifest.get(key, 0))
        if set(by_segment) != set(segment_ids) or len(by_segment) != len(segment_ids):
            raise RuntimeError(
                "parallel block segments do not exactly match the source index")

        histories = [by_segment[key]["history"] for key in segment_ids]
        decisions = [by_segment[key]["decision"] for key in segment_ids]
        composed = [by_segment[key]["composed"] for key in segment_ids]
        cache_rows = [by_segment[key]["cache"] for key in segment_ids]
        captions: dict[str, Any] = {}
        registries = []
        for segment_id in segment_ids:
            parsed = by_segment[segment_id]["parsed"]
            entry = caption_entry_from_parsed(parsed)
            if entry is not None:
                captions[segment_id] = entry
                if parsed.get("subject_registry"):
                    registries.append(parsed["subject_registry"])
        captions["subject_registry"] = self.merge_fn(registries)
        view_hash = sha256_text(dumps_canonical({
            "video_id": video_id, "histories": histories,
            "composed_prompts": composed,
        }))
        view_dir = os.path.join(work_root, f"captions_history_{view_hash[:12]}")
        captions_path, captions_hash = self.captions_writer(captions, view_dir)

        def write_jsonl(name: str, rows: list[Any]) -> str:
            path = os.path.join(work_root, name)
            _atomic_write_text(path, "".join(
                dumps_canonical(row) + "\n" for row in rows))
            return os.path.abspath(path)

        decisions_path = write_jsonl("routing_decisions.jsonl", decisions)
        composed_path = write_jsonl("composed_prompts.jsonl", composed)
        histories_path = write_jsonl("frozen_histories.jsonl", histories)
        cache_keys_path = write_jsonl("caption_cache_keys.jsonl", cache_rows)
        frames_path = os.path.join(work_root, "frames.json")
        persisted_frames = {}
        for key, info in clip_index:
            values = info.get("files") or info.get("frames") or ()
            if isinstance(values, (str, os.PathLike)):
                values = (values,)
            persisted_frames[key] = tuple(
                os.path.abspath(os.fspath(item)) for item in values)
        _atomic_write_text(frames_path, dumps_canonical(persisted_frames) + "\n")
        routing_manifest_path = os.path.join(work_root, "routing_manifest.json")
        routing_manifest = {
            "schema_version": "history_aware_baseline_routing_v1",
            "video_id": video_id, "segment_ids": segment_ids,
            "routing_decisions_path": decisions_path,
            "composed_prompts_path": composed_path,
            "frozen_histories_path": histories_path,
            "caption_cache_keys_path": cache_keys_path,
            "history_block_seconds": history_block_seconds,
            "max_history_captions": max_history_captions,
            **totals,
            "parallel_captioning": {
                "scheduler_version": PARALLEL_GROUP_SCHEDULER_VERSION,
                "unit": "history_block", "worker_count": len(active_gpus),
                "gpu_ids": active_gpus, "block_count": len(block_manifest_paths),
                "merge_order": "ascending_segment_start",
            },
        }
        _atomic_write_text(routing_manifest_path, json.dumps(
            routing_manifest, sort_keys=True, ensure_ascii=False,
            indent=2) + "\n")
        routed_view_path = os.path.join(view_dir, "routed_view.json")
        _atomic_write_text(routed_view_path, json.dumps({
            **routing_manifest, "view_hash": view_hash,
            "captions_path": captions_path, "captions_hash": captions_hash,
        }, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
        return HistoryAwareBaselineViewArtifact(
            video_id=video_id, captions_path=captions_path,
            captions_hash=captions_hash,
            database_path=os.path.join(
                work_root, f"database_c{captions_hash[:16]}.json"),
            routed_view_path=os.path.abspath(routed_view_path),
            routing_manifest_path=os.path.abspath(routing_manifest_path),
            frames_path=os.path.abspath(frames_path),
            frozen_histories_path=histories_path, segment_ids=segment_ids,
            histories=tuple(histories),
            caption_call_count=totals["caption_calls"],
            caption_cache_hits=totals["caption_cache_hits"],
            router_call_count=totals["router_calls"],
            resumed_segment_count=totals["resumed_segments"],
        )
