"""Offline routing-to-composition orchestration for Phase 4 Stage 4.5.

This module connects only saved ``SegmentContext`` records to the existing
router and scaffold-applier boundaries.  It deliberately has no captioning,
DVD, evidence, feedback, or optimization imports.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from surrogate_rollout.prompt_routing.persistence import (
    _atomic_write_text,
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.policies.rule_based_router import (
    RuleBasedPromptRouter,
)
from surrogate_rollout.prompt_routing.scaffold_applier import (
    create_scaffold_applier,
)
from surrogate_rollout.prompt_routing.schemas import (
    ComposedCaptionPrompt,
    Phase4Config,
    PromptBankSnapshot,
    RouterPolicySnapshot,
    RoutingDecision,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    SegmentContext,
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.prompt_routing.validators import (
    assert_router_compatible,
    validate_prompt_bank,
    validate_scaffold_contract,
    validate_scaffold_policy,
)


class OfflineDryRunError(RuntimeError):
    """A Stage 4.5 boundary returned invalid or incompatible state."""


@dataclass(frozen=True)
class OfflineDryRunResult:
    output_dir: str
    manifest_path: str
    routing_decisions: tuple[RoutingDecision, ...]
    composed_prompts: tuple[ComposedCaptionPrompt, ...]


def segment_context_from_json(data: Mapping[str, Any]) -> SegmentContext:
    return SegmentContext(
        video_id=data["video_id"],
        segment_id=data["segment_id"],
        timestamp_start=data.get("timestamp_start"),
        timestamp_end=data.get("timestamp_end"),
        segment_features=data.get("segment_features") or {},
        history_summary=data.get("history_summary"),
        question=data.get("question"),
        metadata=data.get("metadata") or {},
    )


def load_fixture_bundle(path: str) -> dict[str, Any]:
    """Load the deterministic single-file fixture/input format used by the CLI."""
    with open(path) as f:
        data = json.load(f)
    return {
        "contexts": tuple(segment_context_from_json(d)
                          for d in data.get("segment_contexts") or ()),
        "prompt_bank": prompt_bank_from_json(data["prompt_bank"]),
        "router_policy": router_policy_from_json(data["router_policy"]),
        "scaffold_policy": scaffold_policy_from_json(data["scaffold_policy"]),
        "scaffold_contract": scaffold_contract_from_json(data["scaffold_contract"]),
        "phase4_config": Phase4Config.from_json_dict(data.get("phase4_config") or {}),
        "base_prompt_template": data["base_prompt_template"],
    }


def _validate_inputs(
    contexts: Sequence[SegmentContext],
    prompt_bank: PromptBankSnapshot,
    router_policy: RouterPolicySnapshot,
    scaffold_policy: ScaffoldPolicySnapshot,
    scaffold_contract: ScaffoldContract,
) -> None:
    if not contexts:
        raise OfflineDryRunError("at least one saved SegmentContext is required")
    keys = [(c.video_id, c.segment_id) for c in contexts]
    if len(keys) != len(set(keys)):
        raise OfflineDryRunError("duplicate (video_id, segment_id) contexts")
    errors = (
        *validate_prompt_bank(prompt_bank),
        *validate_scaffold_policy(scaffold_policy),
        *validate_scaffold_contract(scaffold_contract),
    )
    if errors:
        raise OfflineDryRunError("invalid input snapshots: " + "; ".join(errors))
    try:
        assert_router_compatible(router_policy, prompt_bank)
    except ValueError as exc:
        raise OfflineDryRunError(str(exc)) from exc


def _validate_decision(
    decision: Any,
    context: SegmentContext,
    prompt_bank: PromptBankSnapshot,
    router_policy: RouterPolicySnapshot,
) -> RoutingDecision:
    if not isinstance(decision, RoutingDecision):
        raise OfflineDryRunError(
            f"router returned {type(decision).__name__}, expected RoutingDecision")
    expected_key = (context.video_id, context.segment_id)
    if (decision.video_id, decision.segment_id) != expected_key:
        raise OfflineDryRunError(
            "routing decision context mismatch: "
            f"{(decision.video_id, decision.segment_id)!r} != {expected_key!r}")
    if decision.bank_version != prompt_bank.bank_version:
        raise OfflineDryRunError("routing decision bank_version mismatch")
    if decision.router_version != router_policy.router_version:
        raise OfflineDryRunError("routing decision router_version mismatch")
    active = {entry.prompt_id for entry in prompt_bank.entries
              if entry.status == "active"}
    invalid = set(decision.selected_prompt_ids) - active
    if invalid:
        raise OfflineDryRunError(
            f"routing decision selected non-active or unknown entries: {sorted(invalid)}")
    limit = min(prompt_bank.max_selected_entries,
                router_policy.max_selected_entries)
    if len(decision.selected_prompt_ids) > limit:
        raise OfflineDryRunError(
            f"routing decision selected {len(decision.selected_prompt_ids)} entries, "
            f"exceeding limit {limit}")
    return decision


def _validate_composed(
    composed: Any,
    context: SegmentContext,
    decision: RoutingDecision,
    scaffold_policy: ScaffoldPolicySnapshot,
    scaffold_contract: ScaffoldContract,
) -> ComposedCaptionPrompt:
    if not isinstance(composed, ComposedCaptionPrompt):
        raise OfflineDryRunError(
            f"scaffold applier returned {type(composed).__name__}, "
            "expected ComposedCaptionPrompt")
    expected_versions = (
        decision.bank_version,
        decision.router_version,
        scaffold_policy.scaffold_version,
        scaffold_contract.contract_version,
    )
    actual_versions = (
        composed.bank_version,
        composed.router_version,
        composed.scaffold_version,
        composed.contract_version,
    )
    if actual_versions != expected_versions:
        raise OfflineDryRunError(
            f"composed prompt version lineage mismatch: "
            f"{actual_versions!r} != {expected_versions!r}")
    if (composed.video_id, composed.segment_id) != \
            (context.video_id, context.segment_id):
        raise OfflineDryRunError("composed prompt context mismatch")
    if composed.selected_prompt_ids != decision.selected_prompt_ids:
        raise OfflineDryRunError("composed prompt selection mismatch")
    if not composed.is_valid:
        raise OfflineDryRunError(
            "invalid composed prompt rejected: " +
            "; ".join(composed.validation_errors))
    return composed


def _pretty_json(value: Any) -> str:
    return json.dumps(as_json_dict(value), sort_keys=True, ensure_ascii=False,
                      indent=2) + "\n"


def _write_artifact(output_dir: str, relative_path: str, text: str) -> str:
    path = os.path.join(output_dir, relative_path)
    _atomic_write_text(path, text)
    return path


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_offline_dry_run(
    *,
    contexts: Sequence[SegmentContext],
    prompt_bank: PromptBankSnapshot,
    router_policy: RouterPolicySnapshot,
    scaffold_policy: ScaffoldPolicySnapshot,
    scaffold_contract: ScaffoldContract,
    phase4_config: Phase4Config,
    base_prompt_template: str,
    output_dir: str,
    router: Any | None = None,
    scaffold_applier: Any | None = None,
    source_revision: str = "unknown",
) -> OfflineDryRunResult:
    """Route and compose saved contexts, then write complete offline artifacts.

    All results are computed and validated before any artifact is written, so
    an invalid policy output cannot leave a success-looking partial run.
    ``phase4_config`` is logged only; optimization flags never affect this
    inference-only data flow.
    """
    _validate_inputs(contexts, prompt_bank, router_policy,
                     scaffold_policy, scaffold_contract)
    if router is None:
        if router_policy.policy_type != "rule_based":
            raise OfflineDryRunError(
                f"unsupported offline router policy_type "
                f"{router_policy.policy_type!r}")
        router = RuleBasedPromptRouter()
    if scaffold_applier is None:
        scaffold_applier = create_scaffold_applier(
            scaffold_policy, base_prompt_template=base_prompt_template)

    entries_by_id = {entry.prompt_id: entry for entry in prompt_bank.entries}
    decisions: list[RoutingDecision] = []
    composed_prompts: list[ComposedCaptionPrompt] = []
    for context in contexts:
        decision = _validate_decision(
            router.route(context, prompt_bank, router_policy),
            context, prompt_bank, router_policy)
        selected_entries = tuple(entries_by_id[prompt_id]
                                 for prompt_id in decision.selected_prompt_ids)
        composed = _validate_composed(
            scaffold_applier.apply(
                context=context,
                selected_entries=selected_entries,
                routing_decision=decision,
                scaffold_policy=scaffold_policy,
                scaffold_contract=scaffold_contract,
            ),
            context, decision, scaffold_policy, scaffold_contract)
        decisions.append(decision)
        composed_prompts.append(composed)

    output_dir = os.path.abspath(output_dir)
    artifact_text = {
        "run_config.json": _pretty_json(phase4_config),
        "git_commit.txt": source_revision.strip() + "\n",
        "environment.json": _pretty_json({
            "execution_mode": "offline_routing_to_composition",
            "captioning_calls": 0,
            "dvd_reasoning_calls": 0,
            "optimization_stages_run": 0,
        }),
        "input_state/segment_contexts.jsonl": "".join(
            dumps_canonical(c) + "\n" for c in contexts),
        "input_state/prompt_bank.json": _pretty_json(prompt_bank),
        "input_state/router_policy.json": _pretty_json(router_policy),
        "input_state/scaffold_policy.json": _pretty_json(scaffold_policy),
        "input_state/scaffold_contract.json": _pretty_json(scaffold_contract),
        "input_state/base_prompt_template.txt": base_prompt_template,
        "routing/decisions.jsonl": "".join(
            dumps_canonical(d) + "\n" for d in decisions),
        "composition/composed_prompts.jsonl": "".join(
            dumps_canonical(p) + "\n" for p in composed_prompts),
        "composition/traces.jsonl": "".join(
            dumps_canonical(p.composition_trace) + "\n"
            for p in composed_prompts),
    }
    written: dict[str, str] = {}
    for relative_path, text in artifact_text.items():
        written[relative_path] = _write_artifact(
            output_dir, relative_path, text)

    manifest = {
        "schema_version": "phase4_stage4.5_v1",
        "status": "completed",
        "execution_mode": "offline_routing_to_composition",
        "segment_count": len(contexts),
        "fallback_count": sum(d.used_fallback for d in decisions),
        "errors": [],
        "component_versions": {
            "bank": prompt_bank.bank_version,
            "router": router_policy.router_version,
            "scaffold": scaffold_policy.scaffold_version,
            "contract": scaffold_contract.contract_version,
        },
        "optimization_flags": {
            "optimize_prompt_bank": phase4_config.optimization.optimize_prompt_bank,
            "optimize_router": phase4_config.optimization.optimize_router,
            "optimize_scaffold": phase4_config.optimization.optimize_scaffold,
        },
        "calls": {
            "captioning": 0,
            "dvd_reasoning": 0,
            "optimization": 0,
        },
        "artifacts": {
            relative_path: {
                "sha256": _sha256_file(path),
                "size_bytes": os.path.getsize(path),
            }
            for relative_path, path in sorted(written.items())
        },
    }
    manifest_path = _write_artifact(
        output_dir, "manifest.json", _pretty_json(manifest))
    return OfflineDryRunResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        routing_decisions=tuple(decisions),
        composed_prompts=tuple(composed_prompts),
    )
