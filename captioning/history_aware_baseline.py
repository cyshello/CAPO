"""Sequential history-aware routing and full-video baseline captioning."""

from __future__ import annotations

import json
import atexit
import contextlib
import multiprocessing
import os
import queue
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
from surrogate_rollout.prompt_routing.router import routing_decision_from_json
from surrogate_rollout.prompt_routing.scaffold_applier import (
    composed_prompt_from_json,
    create_scaffold_applier,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    RouterPolicySnapshot,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    SegmentContext,
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_text


HISTORY_SCHEMA_VERSION = "frozen_local_caption_history_v1"
DEFAULT_HISTORY_BLOCK_SECONDS = 300.0
DEFAULT_MAX_HISTORY_CAPTIONS = 30
PARALLEL_GROUP_SCHEDULER_VERSION = "history_block_gpu_scheduler_v1"
LOCAL_QWEN_BACKEND_ID = "captioning.qwen25_vl.Qwen25VLCaptioner"
CAPTION_OUTPUT_CONTRACT_VERSION = "caption_output_contract_v2"
CAPTION_PARSE_SCHEMA_VERSION = "caption_parse_result_v2"
CAPTION_PARSE_NORMALIZATION_VERSION = (
    "caption_parse_normalization_v2_percent_escape")
CAPTION_CACHE_SCHEMA_VERSION = "history_aware_caption_cache_v4"
CAPTION_RETRY_POLICY_VERSION = "caption_parse_retry_v2_percent_escape"
CAPTION_ATTEMPT_SCHEMA_VERSION = "caption_generation_attempt_v1"


def build_caption_output_contract(subject_registry_mode: str) -> str:
    if subject_registry_mode == "empty":
        registry_contract = (
            '- Set "subject_registry" to an empty object {}. Do not create '
            'registry entries.')
    elif subject_registry_mode == "optional":
        registry_contract = (
            '- "subject_registry" is optional. If reliable subject entries '
            'are unavailable, omit it or return an empty object {} rather '
            'than inventing entries. A valid populated registry is allowed.')
    else:
        raise ValueError("subject_registry_mode must be 'empty' or 'optional'")
    return f"""

CAPTION_OUTPUT_CONTRACT (this supersedes any earlier output-format wording):
- Return exactly one JSON object. Do not wrap it in an array.
- "clip_description" is required and must be a non-empty visual description.
{registry_contract}
""".rstrip()


CAPTION_OUTPUT_CONTRACT = build_caption_output_contract(
    config.CAPTION_SUBJECT_REGISTRY_MODE)


class _LazyLocalQwenBackend:
    """Avoid loading Qwen in the parent before GPU worker processes start."""

    def caption(self, *args: Any, **kwargs: Any) -> str:
        from surrogate_rollout.prompt_routing.policies.history_aware_vlm_router import (
            get_local_qwen_backend,
        )

        return get_local_qwen_backend().caption(*args, **kwargs)


class _PersistentRawVLMProxy:
    """dvd_backend captioner facade backed by the persistent GPU pool."""

    def __init__(self, dispatcher: Any) -> None:
        self.dispatcher = dispatcher

    def caption(self, images: Any, prompt: str, **kwargs: Any) -> str:
        return self.dispatcher.caption_raw_vlm(
            images=images, prompt=prompt, **kwargs)


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


def _parse_caption_output_with_metadata(
    raw: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse the DVD caption contract and record every safe normalization."""
    text = raw.strip()
    normalizations: list[str] = []
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        normalizations.append("strip_markdown_fence")
    normalized = []
    percent_escape_replaced = False
    index = 0
    while index < len(text):
        if text[index] != "\\":
            normalized.append(text[index])
            index += 1
            continue
        end = index
        while end < len(text) and text[end] == "\\":
            end += 1
        slash_count = end - index
        if end < len(text) and text[end] == "%" and slash_count % 2 == 1:
            normalized.append("\\" * (slash_count - 1))
            normalized.append("%")
            percent_escape_replaced = True
            index = end + 1
            continue
        normalized.append("\\" * slash_count)
        index = end
    if percent_escape_replaced:
        text = "".join(normalized)
        normalizations.append("replace_invalid_percent_escape")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            return {}, {
                "schema_version": CAPTION_PARSE_SCHEMA_VERSION,
                "status": "invalid", "error": "invalid_json",
                "normalizations": normalizations,
            }
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}, {
                "schema_version": CAPTION_PARSE_SCHEMA_VERSION,
                "status": "invalid", "error": "invalid_json",
                "normalizations": normalizations,
            }
        normalizations.append("extract_embedded_object")

    original_top_level_type = type(value).__name__
    if isinstance(value, list):
        if len(value) != 1 or not isinstance(value[0], dict):
            return {}, {
                "schema_version": CAPTION_PARSE_SCHEMA_VERSION,
                "status": "invalid", "error": "invalid_top_level_array",
                "original_top_level_type": original_top_level_type,
                "normalizations": normalizations,
            }
        value = value[0]
        normalizations.append("unwrap_singleton_object_array")
    if not isinstance(value, dict):
        return {}, {
            "schema_version": CAPTION_PARSE_SCHEMA_VERSION,
            "status": "invalid", "error": "invalid_top_level_type",
            "original_top_level_type": original_top_level_type,
            "normalizations": normalizations,
        }

    description = value.get("clip_description")
    if not isinstance(description, str) or not description.strip():
        return {}, {
            "schema_version": CAPTION_PARSE_SCHEMA_VERSION,
            "status": "invalid", "error": "missing_clip_description",
            "original_top_level_type": original_top_level_type,
            "normalizations": normalizations,
        }
    parsed = dict(value)
    parsed["clip_description"] = description.strip()
    registry = parsed.get("subject_registry")
    if not isinstance(registry, dict):
        parsed["subject_registry"] = {}
        normalizations.append("default_empty_subject_registry")
    return parsed, {
        "schema_version": CAPTION_PARSE_SCHEMA_VERSION,
        "status": "valid", "error": None,
        "original_top_level_type": original_top_level_type,
        "normalizations": normalizations,
        "subject_registry_populated": bool(parsed["subject_registry"]),
    }


def _parse_caption_output(raw: str) -> dict[str, Any]:
    """Compatibility wrapper for callers which only consume parsed output."""
    return _parse_caption_output_with_metadata(raw)[0]


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


def _parallel_history_worker(
    gpu: str,
    command_queue: Any,
    result_queue: Any,
    text_backend: str,
    use_openai_tools: bool,
    routing_mode: str = "property_bank",
) -> None:
    """Own one GPU/model instance and serve caption tasks until shutdown."""
    try:
        from surrogate_rollout.evaluation.dvd_qa import ensure_backend
        from surrogate_rollout.prompt_routing.persistence import (
            prompt_bank_from_json,
            router_policy_from_json,
            scaffold_contract_from_json,
            scaffold_policy_from_json,
        )

        ensure_backend(
            gpu, preload_captioner=True,
            text_backend=text_backend,
            use_openai_tools=use_openai_tools,
        )
        builder = HistoryAwareBaselineCaptionViewBuilder.from_local_qwen(
            routing_mode=routing_mode)
        while True:
            command = command_queue.get()
            request_id = command["request_id"]
            try:
                if command["type"] == "shutdown":
                    engine = getattr(builder.router.vlm, "llm", None)
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
                    return
                if command["type"] == "build_blocks":
                    common = command["common"]
                    prompt_bank = prompt_bank_from_json(common["prompt_bank"])
                    router_policy = router_policy_from_json(common["router_policy"])
                    scaffold_policy = scaffold_policy_from_json(
                        common["scaffold_policy"])
                    scaffold_contract = scaffold_contract_from_json(
                        common["scaffold_contract"])
                    completed = []
                    for block_index, block_clips, block_root, fragment_path in \
                            command["jobs"]:
                        artifact = builder.build(
                            sample=common["sample"], clip_index=list(block_clips),
                            prompt_bank=prompt_bank, router_policy=router_policy,
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
                    text = builder.router.vlm.caption(
                        raw_payload["images"], raw_payload["prompt"],
                        **raw_payload["kwargs"],
                    )
                    payload = {"raw_output": text}
                else:
                    raise ValueError(f"unknown worker command: {command['type']}")
                result_queue.put({
                    "request_id": request_id, "gpu": gpu, **payload,
                })
            except BaseException as exc:
                result_queue.put({
                    "request_id": request_id, "gpu": gpu,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
    except BaseException as exc:
        result_queue.put({
            "request_id": "startup", "gpu": gpu,
            "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        })


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
                 remote_dispatcher: Any | None = None) -> None:
        self.vlm = vlm
        self.caption_model_id = caption_model_id or config.CAPTION_MODEL_ID
        self.backend_id = backend_id or (
            f"{type(vlm).__module__}.{type(vlm).__qualname__}")
        self.remote_dispatcher = remote_dispatcher

    @property
    def configuration_identity(self) -> Mapping[str, Any]:
        return {
            "provider": "local_qwen_vlm",
            "model": self.caption_model_id,
            "backend": self.backend_id,
            "decoding": as_json_dict(config.CAPTION_DECODING),
            "caption_output_contract_version": CAPTION_OUTPUT_CONTRACT_VERSION,
            "caption_parse_schema_version": CAPTION_PARSE_SCHEMA_VERSION,
            "caption_parse_normalization_version":
                CAPTION_PARSE_NORMALIZATION_VERSION,
            "caption_retry_policy_version": CAPTION_RETRY_POLICY_VERSION,
            "caption_parse_max_retries": config.CAPTION_PARSE_MAX_RETRIES,
            "subject_registry_mode": config.CAPTION_SUBJECT_REGISTRY_MODE,
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
                "caption_parse_schema_version": CAPTION_PARSE_SCHEMA_VERSION,
                "caption_parse_normalization_version":
                    CAPTION_PARSE_NORMALIZATION_VERSION,
                "subject_registry_mode": config.CAPTION_SUBJECT_REGISTRY_MODE,
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
            if cached.get("parsed") or (
                    cached.get("caption_retry_policy_version") ==
                    CAPTION_RETRY_POLICY_VERSION and
                    cached.get("caption_parse_normalization_version") ==
                    CAPTION_PARSE_NORMALIZATION_VERSION):
                return HistoryAwareCaptionResult(
                    parsed=cached.get("parsed") or {}, cache_key=key_as_dict(key),
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
        prompt = (composed_prompt.prompt_text
                  .replace("TRANSCRIPT_PLACEHOLDER", transcript)
                  .replace("CLIP_START_TIME", _hhmmss(start))
                  .replace("CLIP_END_TIME", _hhmmss(end)))
        prompt += CAPTION_OUTPUT_CONTRACT
        prompt += (
            "\n\nFROZEN_PRECEDING_CAPTION_HISTORY_JSON:\n"
            + str(history_snapshot["serialized_history"])
        )
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
        if cached_invalid is not None:
            raw = str(cached_invalid.get("raw_output") or "")
            parsed, parse_result = _parse_caption_output_with_metadata(raw)
            attempts.append({
                "schema_version": CAPTION_ATTEMPT_SCHEMA_VERSION,
                "attempt_index": 1,
                "source": "immutable_cached_initial_attempt",
                "raw_output": raw,
                "parse_result": parse_result,
                "parsed_valid": bool(parsed),
                "caption_seconds": 0.0,
            })
            call_indices = (
                () if parsed else
                range(1, config.CAPTION_PARSE_MAX_RETRIES + 1))
        else:
            call_indices = range(0, config.CAPTION_PARSE_MAX_RETRIES + 1)
        for retry_index in call_indices:
            t0 = time.monotonic()
            raw = self.vlm.caption(
                frame_paths, prompt,
                max_tokens=int(config.CAPTION_DECODING["max_tokens"]),
            )
            attempt_seconds = time.monotonic() - t0
            elapsed += attempt_seconds
            model_call_count += 1
            parsed, parse_result = _parse_caption_output_with_metadata(raw)
            attempts.append({
                "schema_version": CAPTION_ATTEMPT_SCHEMA_VERSION,
                "attempt_index": len(attempts) + 1,
                "source": "model_call",
                "retry_index": retry_index,
                "raw_output": raw,
                "parse_result": parse_result,
                "parsed_valid": bool(parsed),
                "caption_seconds": attempt_seconds,
            })
            if parsed:
                break
        retry_count = max(0, len(attempts) - 1)
        if parsed:
            parsed["clip_description"] = (
                str(parsed["clip_description"])
                + f"\n\nTranscript during this video clip: {transcript}."
            )
        payload = {
            "schema_version": CAPTION_CACHE_SCHEMA_VERSION,
            "cache_key": key_as_dict(key),
            "history_hash": history_snapshot["history_hash"],
            "rendered_prompt_hash": sha256_text(prompt),
            "caption_output_contract_version":
                CAPTION_OUTPUT_CONTRACT_VERSION,
            "caption_parse_normalization_version":
                CAPTION_PARSE_NORMALIZATION_VERSION,
            "caption_retry_policy_version": CAPTION_RETRY_POLICY_VERSION,
            "caption_parse_max_retries": config.CAPTION_PARSE_MAX_RETRIES,
            "attempt_count": len(attempts),
            "retry_count": retry_count,
            "retry_exhausted": (
                not bool(parsed)
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
    ) -> None:
        normalized_gpus = tuple(str(value).strip() for value in parallel_gpus)
        if any(not value for value in normalized_gpus) or \
                len(normalized_gpus) != len(set(normalized_gpus)):
            raise ValueError("parallel_gpus must contain unique non-empty GPU IDs")
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
    def from_local_qwen(cls, *, parallel_gpus: tuple[str, ...] = (),
                        routing_mode: str = "property_bank"):
        from surrogate_rollout.prompt_routing.policies.history_aware_vlm_router import (
            HistoryAwareVLMRouter,
            get_local_qwen_backend,
        )

        if routing_mode not in ("property_bank", "free_form_generator"):
            raise ValueError(
                f"unknown routing_mode {routing_mode!r} "
                "(supported: 'property_bank', 'free_form_generator')")
        vlm = (_LazyLocalQwenBackend() if parallel_gpus
               else get_local_qwen_backend())
        free_form_generator = None
        if routing_mode == "free_form_generator":
            from surrogate_rollout.prompt_routing.free_form_instruction_generator import (
                VLMFreeFormInstructionGenerator,
            )

            free_form_generator = VLMFreeFormInstructionGenerator(
                vlm, backend_id=LOCAL_QWEN_BACKEND_ID)
        builder = cls(
            router=HistoryAwareVLMRouter(vlm, backend_id=LOCAL_QWEN_BACKEND_ID),
            segment_captioner=HistoryAwareSegmentCaptioner(
                vlm, backend_id=LOCAL_QWEN_BACKEND_ID),
            parallel_gpus=parallel_gpus,
            free_form_generator=free_form_generator,
        )
        if parallel_gpus:
            builder.segment_captioner.remote_dispatcher = builder
        return builder

    def build(
        self,
        *,
        sample: Mapping[str, Any],
        clip_index: list[tuple[str, dict]],
        prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
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
            "prompt_bank": prompt_bank, "router_policy": router_policy,
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
        prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
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
        scaffold = create_scaffold_applier(
            scaffold_policy, base_prompt_template=base_prompt_template)
        bank_by_id = {entry.prompt_id: entry for entry in prompt_bank.entries}
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
                "prompt_bank": prompt_bank,
                "router_policy": router_policy,
                "scaffold_policy": scaffold_policy,
                "scaffold_contract": scaffold_contract,
                "base_prompt_template_hash": sha256_text(base_prompt_template),
                "merge_prompt_hash": sha256_text(merge_prompt),
                "caption_model_id": self.segment_captioner.caption_model_id,
                "caption_backend_id": self.segment_captioner.backend_id,
                "caption_decoding_hash": config.decoding_hash(),
                "caption_output_contract_version":
                    CAPTION_OUTPUT_CONTRACT_VERSION,
                "caption_parse_schema_version": CAPTION_PARSE_SCHEMA_VERSION,
                "caption_parse_normalization_version":
                    CAPTION_PARSE_NORMALIZATION_VERSION,
                "subject_registry_mode": config.CAPTION_SUBJECT_REGISTRY_MODE,
                "caption_run_identity_hash": caption_run_identity_hash,
                "history_configuration": {
                    "schema_version": HISTORY_SCHEMA_VERSION,
                    "block_seconds": history_block_seconds,
                    "max_history_captions": max_history_captions,
                    "boundary_rule": "floor_segment_start_div_block_seconds",
                },
            }
            if self.free_form_generator is not None:
                # Key added ONLY in free-form mode so property-bank state
                # identities (and therefore resume behavior) stay byte-stable.
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
                if self.free_form_generator is not None:
                    from surrogate_rollout.prompt_routing.free_form_instruction_generator import (
                        build_free_form_selection,
                    )

                    decision, selected_entries, exchange = (
                        build_free_form_selection(
                            generator=self.free_form_generator,
                            context=context, clip_info=clip_info,
                            prompt_bank=prompt_bank))
                    router_calls += 1
                else:
                    decision = self.router.route(
                        context, prompt_bank, router_policy)
                    router_calls += 1
                    selected_entries = tuple(
                        bank_by_id[value]
                        for value in decision.selected_prompt_ids)
                    exchange = getattr(self.router, "last_exchange", None)
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
        prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
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
            "sample": dict(sample), "prompt_bank": as_json_dict(prompt_bank),
            "router_policy": as_json_dict(router_policy),
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

    def _wait_pool_result(
        self, request_id: str, *, timeout_seconds: float = 86400.0,
    ) -> dict[str, Any]:
        with self._pool_lock:
            waiter = self._pool_pending.get(request_id)
        if waiter is None:
            raise RuntimeError(f"unknown persistent-pool request: {request_id}")
        try:
            item = waiter.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            raise RuntimeError(
                f"persistent caption worker timed out: {request_id}") from exc
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
                      ("free_form_generator"
                       if self.free_form_generator is not None
                       else "property_bank")),
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
                for process in self._pool_processes.values():
                    process.join(timeout=30.0)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=10.0)
                self._pool_processes.clear()
                self._pool_command_queues.clear()
                if self._pool_result_queue is not None:
                    self._pool_result_queue.put({"request_id": "__parent_stop__"})
                if self._pool_result_thread is not None:
                    self._pool_result_thread.join(timeout=5.0)
                self._pool_result_thread = None
                self._pool_result_queue = None
                self._pool_pending.clear()
                self._restore_parent_vlm_proxy()
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
