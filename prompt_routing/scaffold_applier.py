"""ScaffoldApplier protocol, composed-prompt validation, and composition
logging (Stage 4.4, PHASE4 §11.2/§12).

The scaffold applier is the INFERENCE-TIME composer (§3.3): it combines the
router-selected prompt-bank entries with the fixed scaffold contract into one
final captioning prompt. It never updates the prompt bank or the scaffold
policy — those belong to the (later) optimization-time proposers and must not
be conflated with this module.

Concrete appliers live in prompt_routing/policies/ and are selected through
configuration (`ScaffoldPolicySnapshot.policy_type`) via
`create_scaffold_applier`, so replacing the implementation (deterministic now,
SLM later) requires no change to callers (§5.3).

Every applier — including future model-backed ones — must funnel its raw
composed text through `finalize_composed_prompt`, which validates against the
hard ScaffoldContract and returns a typed ComposedCaptionPrompt. Contract
violations produce `is_valid=False` with explicit `validation_errors`, never
an exception and never a silent repair (§21 mirrors CLAUDE.md §17 semantics
from captioning/candidate_captions.py: REQUIRED_PLACEHOLDERS are checked as
literal substrings of the prompt text).

Validation conventions (deterministic, documented — production refinements are
tracked as unresolved assumptions, not silently changed here):

- required placeholders and required sections are literal substring markers
  the composed text must contain;
- forbidden patterns are literal substrings the composed text must not
  contain;
- `max_prompt_tokens` is enforced with a whitespace-token proxy
  (`len(text.split())`); the real captioner tokenizer limit stays an open
  configuration question (see PHASE4 progress notes).
"""

from __future__ import annotations

import json
import os
from typing import Protocol, Sequence

from surrogate_rollout.prompt_routing.schemas import (
    ComposedCaptionPrompt,
    CompositionTrace,
    PromptEntry,
    RoutingDecision,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    SegmentContext,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_text


class ScaffoldApplier(Protocol):
    def apply(
        self,
        *,
        context: SegmentContext,
        selected_entries: Sequence[PromptEntry],
        routing_decision: RoutingDecision,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
    ) -> ComposedCaptionPrompt:
        ...


# --------------------------------------------------------------------------- #
#                       contract validation (hard, §12.1)                      #
# --------------------------------------------------------------------------- #
def estimate_prompt_tokens(text: str) -> int:
    """Deterministic whitespace-token proxy used for max_prompt_tokens
    enforcement until a captioner-tokenizer-based count is configured."""
    return len(text.split())


def validate_composed_text(prompt_text: str,
                           contract: ScaffoldContract) -> tuple[str, ...]:
    """Every hard-contract violation in `prompt_text`, empty when valid."""
    errors: list[str] = []
    if not prompt_text:
        errors.append("composed prompt text is empty")
    missing = [ph for ph in contract.required_placeholders
               if ph not in prompt_text]
    if missing:
        errors.append(f"missing required placeholders: {missing}")
    missing_sections = [s for s in contract.required_sections
                        if s not in prompt_text]
    if missing_sections:
        errors.append(f"missing required sections: {missing_sections}")
    present_forbidden = [p for p in contract.forbidden_patterns
                         if p and p in prompt_text]
    if present_forbidden:
        errors.append(f"forbidden patterns present: {present_forbidden}")
    if contract.max_prompt_tokens is not None:
        n = estimate_prompt_tokens(prompt_text)
        if n > contract.max_prompt_tokens:
            errors.append(
                f"prompt length {n} tokens (whitespace proxy) exceeds "
                f"contract max_prompt_tokens={contract.max_prompt_tokens}")
    return tuple(errors)


def finalize_composed_prompt(
    *,
    context: SegmentContext,
    routing_decision: RoutingDecision,
    scaffold_policy: ScaffoldPolicySnapshot,
    scaffold_contract: ScaffoldContract,
    prompt_text: str,
    trace: CompositionTrace,
    raw_policy_output_artifact: str | None = None,
) -> ComposedCaptionPrompt:
    """Validate raw composed text against the hard contract and wrap it in a
    typed ComposedCaptionPrompt. Contract violations become
    `is_valid=False` + `validation_errors`; only structural misuse (context /
    decision mismatch) raises."""
    if (context.video_id != routing_decision.video_id
            or context.segment_id != routing_decision.segment_id):
        raise ValueError(
            f"context ({context.video_id!r}, {context.segment_id!r}) does not "
            f"match routing decision ({routing_decision.video_id!r}, "
            f"{routing_decision.segment_id!r})")
    if trace.selected_prompt_ids != routing_decision.selected_prompt_ids:
        raise ValueError(
            "composition trace selected_prompt_ids "
            f"{trace.selected_prompt_ids} do not match routing decision "
            f"{routing_decision.selected_prompt_ids}")
    if raw_policy_output_artifact is not None:
        trace = CompositionTrace(
            selected_prompt_ids=trace.selected_prompt_ids,
            preserved_prompt_ids=trace.preserved_prompt_ids,
            omitted_prompt_ids=trace.omitted_prompt_ids,
            conflicts_detected=trace.conflicts_detected,
            conflict_resolutions=trace.conflict_resolutions,
            ordering_decisions=trace.ordering_decisions,
            compression_actions=trace.compression_actions,
            raw_policy_output_artifact=raw_policy_output_artifact,
        )
    errors = validate_composed_text(prompt_text, scaffold_contract)
    return ComposedCaptionPrompt(
        video_id=context.video_id,
        segment_id=context.segment_id,
        bank_version=routing_decision.bank_version,
        router_version=routing_decision.router_version,
        scaffold_version=scaffold_policy.scaffold_version,
        contract_version=scaffold_contract.contract_version,
        selected_prompt_ids=routing_decision.selected_prompt_ids,
        prompt_text=prompt_text,
        prompt_hash=sha256_text(prompt_text),
        composition_trace=trace,
        validation_errors=errors,
        is_valid=not errors,
    )


# --------------------------------------------------------------------------- #
#                    applier selection through configuration                   #
# --------------------------------------------------------------------------- #
def create_scaffold_applier(scaffold_policy: ScaffoldPolicySnapshot, *,
                            base_prompt_template: str | None = None):
    """Return the active replace-body applier."""
    from surrogate_rollout.prompt_routing.policies.replace_body_scaffold import (
        ReplaceBodyScaffoldApplier,
    )

    if scaffold_policy.policy_type == "replace_body":
        if not base_prompt_template:
            raise ValueError(
                "policy_type 'replace_body' requires a non-empty "
                "base_prompt_template")
        return ReplaceBodyScaffoldApplier(
            base_prompt_template=base_prompt_template)
    raise ValueError(
        f"unknown scaffold policy_type {scaffold_policy.policy_type!r} "
        "(supported: 'replace_body')")


# --------------------------------------------------------------------------- #
#                      composed-prompt / trace logging                         #
# --------------------------------------------------------------------------- #
def composition_trace_from_json(d) -> CompositionTrace:
    return CompositionTrace(
        selected_prompt_ids=tuple(d.get("selected_prompt_ids") or ()),
        preserved_prompt_ids=tuple(d.get("preserved_prompt_ids") or ()),
        omitted_prompt_ids=tuple(d.get("omitted_prompt_ids") or ()),
        conflicts_detected=tuple(d.get("conflicts_detected") or ()),
        conflict_resolutions=tuple(d.get("conflict_resolutions") or ()),
        ordering_decisions=tuple(d.get("ordering_decisions") or ()),
        compression_actions=tuple(d.get("compression_actions") or ()),
        raw_policy_output_artifact=d.get("raw_policy_output_artifact"),
    )


def composed_prompt_from_json(d) -> ComposedCaptionPrompt:
    return ComposedCaptionPrompt(
        video_id=d["video_id"], segment_id=d["segment_id"],
        bank_version=d["bank_version"], router_version=d["router_version"],
        scaffold_version=d["scaffold_version"],
        contract_version=d["contract_version"],
        selected_prompt_ids=tuple(d.get("selected_prompt_ids") or ()),
        prompt_text=d["prompt_text"], prompt_hash=d["prompt_hash"],
        composition_trace=composition_trace_from_json(d["composition_trace"]),
        validation_errors=tuple(d.get("validation_errors") or ()),
        is_valid=d.get("is_valid", True),
    )


def write_composed_prompts(prompts, path: str) -> None:
    """Deterministic JSONL log (one canonical ComposedCaptionPrompt — trace
    included — per line), written atomically via a same-directory temp file."""
    from surrogate_rollout.prompt_routing.persistence import _atomic_write_text

    text = "".join(dumps_canonical(p) + "\n" for p in prompts)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, text)


def read_composed_prompts(path: str) -> tuple[ComposedCaptionPrompt, ...]:
    with open(path) as f:
        return tuple(composed_prompt_from_json(json.loads(line))
                     for line in f if line.strip())
