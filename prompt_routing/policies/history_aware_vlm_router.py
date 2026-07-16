"""Local Qwen2.5-VL router for history-aware baseline captioning."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Mapping

from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    RouterPolicySnapshot,
    RoutingDecision,
    SegmentContext,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_text


REQUEST_SCHEMA_VERSION = "history_aware_vlm_router_request_v1"
OUTPUT_SCHEMA_VERSION = "history_aware_vlm_router_output_v2"
DEFAULT_MAX_SELECTED_PROPERTIES = 3


class HistoryAwareRouterError(ValueError):
    """The router request is incomplete or the VLM returned invalid JSON."""


def get_local_qwen_backend() -> Any:
    """Return DVD's configured, process-shared Qwen2.5-VL/vLLM captioner."""
    from surrogate_rollout import config

    for path in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from dvd_backend import get_captioner

    return get_captioner()


@dataclass(frozen=True)
class RouterExchange:
    request: Mapping[str, Any]
    request_hash: str
    raw_output: str
    output_hash: str


def parse_router_output(raw: str) -> tuple[str, ...]:
    """Parse the exact one-key JSON output contract; prose/fences are invalid."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HistoryAwareRouterError("router output is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {"property_ids"}:
        raise HistoryAwareRouterError(
            "router output must be an object with only property_ids")
    selected = value["property_ids"]
    if not isinstance(selected, list) or any(
            not isinstance(item, str) or not item for item in selected):
        raise HistoryAwareRouterError(
            "property_ids must be a JSON array of non-empty strings")
    return tuple(selected)


class HistoryAwareVLMRouter:
    """PromptRouter adapter using the same local multimodal backend as captions."""

    policy_type = "history_aware_vlm"

    def __init__(self, vlm: Any, *, max_selected_properties: int = 3,
                 max_tokens: int = 256) -> None:
        if max_selected_properties < 1:
            raise ValueError("max_selected_properties must be positive")
        self.vlm = vlm
        self.max_selected_properties = max_selected_properties
        self.max_tokens = max_tokens
        self.last_exchange: RouterExchange | None = None

    @classmethod
    def from_local_qwen(cls, **kwargs) -> "HistoryAwareVLMRouter":
        return cls(get_local_qwen_backend(), **kwargs)

    def route(
        self,
        context: SegmentContext,
        prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
    ) -> RoutingDecision:
        if router_policy.policy_type != self.policy_type:
            raise HistoryAwareRouterError(
                f"HistoryAwareVLMRouter requires policy_type={self.policy_type!r}")
        frames = context.segment_features.get("frame_references") or ()
        if isinstance(frames, str):
            frames = (frames,)
        frames = tuple(str(item) for item in frames)
        if not frames:
            raise HistoryAwareRouterError("current segment frames are required")
        if not context.history_summary:
            raise HistoryAwareRouterError("serialized caption history is required")
        try:
            history = json.loads(context.history_summary)
        except json.JSONDecodeError as exc:
            raise HistoryAwareRouterError(
                "history_summary must be serialized JSON") from exc

        active = tuple(entry for entry in prompt_bank.entries
                       if entry.status == "active")
        configured = int(router_policy.configuration.get(
            "max_selected_properties", self.max_selected_properties))
        limit = min(configured, router_policy.max_selected_entries,
                    prompt_bank.max_selected_entries)
        request = {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "current_segment": {
                "video_id": context.video_id,
                "segment_id": context.segment_id,
                "timestamp_start": context.timestamp_start,
                "timestamp_end": context.timestamp_end,
                "frame_references": frames,
            },
            "preceding_caption_history": history,
            "active_codebook": tuple({
                "property_id": entry.prompt_id,
                "name": entry.name,
                "description": entry.description,
                "instruction": entry.prompt_text,
            } for entry in active),
            "max_selected_properties": limit,
            "output_schema": {"property_ids": ["property_id"]},
        }
        supervision = tuple(router_policy.configuration.get(
            "supervision_examples") or ())[-100:]
        if supervision:
            request["router_supervision"] = tuple({
                "property_id": item.get("property_id"),
                "polarity": item.get("polarity"),
                "evidence": item.get("evidence"),
                "context_hash": item.get("context_hash"),
            } for item in supervision)
        prompt = (
            "Select zero or more active caption properties for the current "
            "video segment. Return only strict JSON matching output_schema; "
            "do not use Markdown or add keys.\n" + dumps_canonical(request)
        )
        raw = self.vlm.caption(frames, prompt, max_tokens=int(
            router_policy.configuration.get("max_tokens", self.max_tokens)))
        proposed = parse_router_output(raw)
        known = {entry.prompt_id for entry in active}
        unknown = set(proposed) - known
        if unknown:
            raise HistoryAwareRouterError(
                f"router selected inactive or unknown properties: {sorted(unknown)}")
        deduplicated = set(proposed)
        selected = tuple(entry.prompt_id for entry in active
                         if entry.prompt_id in deduplicated)
        if len(selected) > limit:
            raise HistoryAwareRouterError(
                f"router selected {len(selected)} properties; maximum is {limit}")

        request_hash = sha256_text(dumps_canonical(request))
        output_hash = sha256_text(raw)
        self.last_exchange = RouterExchange(
            request=request, request_hash=request_hash,
            raw_output=raw, output_hash=output_hash)
        return RoutingDecision(
            video_id=context.video_id,
            segment_id=context.segment_id,
            bank_version=prompt_bank.bank_version,
            router_version=router_policy.router_version,
            selected_prompt_ids=selected,
            prompt_scores={value: 1.0 for value in selected},
            selection_order=selected,
            decision_payload={
                "request_schema_version": REQUEST_SCHEMA_VERSION,
                "output_schema_version": OUTPUT_SCHEMA_VERSION,
                "request_hash": request_hash,
                "output_hash": output_hash,
                "raw_output": raw,
                "deduplicated_property_ids": selected,
                "stable_order": "active_codebook_order",
            },
        )
