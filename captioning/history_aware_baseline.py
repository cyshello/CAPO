"""Sequential history-aware routing and full-video baseline captioning."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from surrogate_rollout import config
from surrogate_rollout.cache.caption_cache import (
    assert_writable,
    build_history_aware_cache_key,
    key_as_dict,
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


def _parse_caption_output(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            return {}
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class HistoryAwareCaptionResult:
    parsed: Mapping[str, Any]
    cache_key: Mapping[str, Any]
    cache_dir: str
    result_path: str
    cache_hit: bool
    caption_seconds: float


class HistoryAwareSegmentCaptioner:
    """Per-segment caption adapter with mandatory frozen-history identity."""

    def __init__(self, vlm: Any, *, caption_model_id: str | None = None,
                 backend_id: str | None = None) -> None:
        self.vlm = vlm
        self.caption_model_id = caption_model_id or config.CAPTION_MODEL_ID
        self.backend_id = backend_id or (
            f"{type(vlm).__module__}.{type(vlm).__qualname__}")

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
        if os.path.exists(result_path):
            with open(result_path) as f:
                cached = json.load(f)
            if cached.get("cache_key") != key_as_dict(key):
                raise RuntimeError("history-aware cache file identity mismatch")
            return HistoryAwareCaptionResult(
                parsed=cached.get("parsed") or {}, cache_key=key_as_dict(key),
                cache_dir=cache_dir, result_path=result_path,
                cache_hit=True, caption_seconds=0.0)

        start, end = _segment_times(segment_id)
        transcript = str(clip_info.get("transcript") or "No transcript.")
        prompt = (composed_prompt.prompt_text
                  .replace("TRANSCRIPT_PLACEHOLDER", transcript)
                  .replace("CLIP_START_TIME", _hhmmss(start))
                  .replace("CLIP_END_TIME", _hhmmss(end)))
        prompt += (
            "\n\nFROZEN_PRECEDING_CAPTION_HISTORY_JSON:\n"
            + str(history_snapshot["serialized_history"])
        )
        frames = clip_info.get("files") or clip_info.get("frames") or ()
        if isinstance(frames, (str, os.PathLike)):
            frames = (frames,)
        t0 = time.monotonic()
        raw = self.vlm.caption(
            tuple(os.fspath(item) for item in frames), prompt,
            max_tokens=int(config.CAPTION_DECODING["max_tokens"]),
        )
        elapsed = time.monotonic() - t0
        parsed = _parse_caption_output(raw)
        if parsed:
            parsed["clip_description"] = (
                str(parsed.get("clip_description") or "")
                + f"\n\nTranscript during this video clip: {transcript}."
            )
        payload = {
            "schema_version": "history_aware_caption_cache_v1",
            "cache_key": key_as_dict(key),
            "history_hash": history_snapshot["history_hash"],
            "rendered_prompt_hash": sha256_text(prompt),
            "raw_output": raw,
            "parsed": parsed,
        }
        _atomic_write_text(result_path, json.dumps(
            payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
        return HistoryAwareCaptionResult(
            parsed=parsed, cache_key=key_as_dict(key), cache_dir=cache_dir,
            result_path=result_path, cache_hit=False, caption_seconds=elapsed)


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
    ) -> None:
        self.router = router
        self.segment_captioner = segment_captioner
        self.merge_fn = merge_fn or default_merge_fn
        self.captions_writer = captions_writer

    @classmethod
    def from_local_qwen(cls):
        from surrogate_rollout.prompt_routing.policies.history_aware_vlm_router import (
            HistoryAwareVLMRouter,
            get_local_qwen_backend,
        )

        vlm = get_local_qwen_backend()
        return cls(
            router=HistoryAwareVLMRouter(vlm),
            segment_captioner=HistoryAwareSegmentCaptioner(vlm),
        )

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
            state_identity = sha256_text(dumps_canonical({
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
                "history_configuration": {
                    "schema_version": HISTORY_SCHEMA_VERSION,
                    "block_seconds": history_block_seconds,
                    "max_history_captions": max_history_captions,
                    "boundary_rule": "floor_segment_start_div_block_seconds",
                },
            }))
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
                }
                resumed_segments += 1
            else:
                decision = self.router.route(context, prompt_bank, router_policy)
                router_calls += 1
                selected_entries = tuple(bank_by_id[value]
                                         for value in decision.selected_prompt_ids)
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
                )
                parsed = dict(result.parsed)
                caption_calls += int(not result.cache_hit)
                cache_hits += int(result.cache_hit)
                caption_result = {
                    "cache_key": result.cache_key,
                    "cache_dir": result.cache_dir,
                    "result_path": result.result_path,
                    "cache_hit": result.cache_hit,
                    "caption_seconds": result.caption_seconds,
                }
                exchange = getattr(self.router, "last_exchange", None)
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
