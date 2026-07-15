"""PromptRouter protocol and routing-decision logging (Stage 4.3, §11.1/§13).

The router selects zero or more prompt-bank entries per segment and returns a
typed RoutingDecision — never unstructured prose. Concrete policies live in
prompt_routing/policies/; an SLM router must implement this same protocol
later without interface changes (§5.2).
"""

from __future__ import annotations

import json
import os
from typing import Protocol

from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    RouterPolicySnapshot,
    RoutingDecision,
    SegmentContext,
    dumps_canonical,
)


class PromptRouter(Protocol):
    def route(
        self,
        context: SegmentContext,
        prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
    ) -> RoutingDecision:
        ...


def routing_decision_from_json(d) -> RoutingDecision:
    return RoutingDecision(
        video_id=d["video_id"], segment_id=d["segment_id"],
        bank_version=d["bank_version"], router_version=d["router_version"],
        selected_prompt_ids=tuple(d.get("selected_prompt_ids") or ()),
        prompt_scores=d.get("prompt_scores") or {},
        matched_rule_ids=tuple(d.get("matched_rule_ids") or ()),
        selection_order=tuple(d.get("selection_order") or ()),
        confidence=d.get("confidence"),
        used_fallback=d.get("used_fallback", False),
        fallback_reason=d.get("fallback_reason"),
        decision_payload=d.get("decision_payload") or {},
    )


def write_routing_decisions(decisions, path: str) -> None:
    """Append-free deterministic JSONL log (one canonical record per line),
    written atomically via a same-directory temp file."""
    from surrogate_rollout.prompt_routing.persistence import _atomic_write_text

    text = "".join(dumps_canonical(d) + "\n" for d in decisions)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, text)


def read_routing_decisions(path: str) -> tuple[RoutingDecision, ...]:
    with open(path) as f:
        return tuple(routing_decision_from_json(json.loads(line))
                     for line in f if line.strip())
