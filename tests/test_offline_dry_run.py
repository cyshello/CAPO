"""Stage 4.5 offline routing-to-composition dry-run tests."""

import dataclasses
import json
import os
from pathlib import Path

import pytest

from surrogate_rollout.prompt_routing.offline_dry_run import (
    OfflineDryRunError,
    load_fixture_bundle,
    run_offline_dry_run,
)
from surrogate_rollout.prompt_routing.policies.rule_based_router import (
    RuleBasedPromptRouter,
)
from surrogate_rollout.prompt_routing.scaffold_applier import (
    finalize_composed_prompt,
)
from surrogate_rollout.prompt_routing.schemas import (
    CompositionTrace,
    Phase4Config,
    Phase4OptimizationConfig,
    RoutingDecision,
)

FIXTURE = Path(__file__).parents[1] / "prompt_routing" / "fixtures" / \
    "stage4_5_fixture.json"


def bundle():
    return load_fixture_bundle(str(FIXTURE))


def run(tmp_path, **overrides):
    inputs = bundle()
    inputs.update(overrides)
    return run_offline_dry_run(output_dir=str(tmp_path), **inputs)


def artifact_bytes(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def test_successful_offline_run(tmp_path):
    result = run(tmp_path)
    assert [d.selected_prompt_ids for d in result.routing_decisions] == [
        ("preserve_temporal_order", "transcribe_visible_text"),
        ("preserve_temporal_order",),
    ]
    assert all(prompt.is_valid for prompt in result.composed_prompts)
    assert "Preserve the exact temporal order" in \
        result.composed_prompts[0].prompt_text
    environment = json.loads((tmp_path / "environment.json").read_text())
    assert environment == {
        "captioning_calls": 0,
        "dvd_reasoning_calls": 0,
        "execution_mode": "offline_routing_to_composition",
        "optimization_stages_run": 0,
    }


class InvalidDecisionRouter:
    def route(self, context, prompt_bank, router_policy):
        return RoutingDecision(
            video_id=context.video_id,
            segment_id=context.segment_id,
            bank_version=prompt_bank.bank_version,
            router_version=router_policy.router_version,
            selected_prompt_ids=("missing_entry",),
            selection_order=("missing_entry",),
        )


def test_invalid_routing_decision_rejected_before_writes(tmp_path):
    with pytest.raises(OfflineDryRunError, match="non-active or unknown"):
        run(tmp_path, router=InvalidDecisionRouter())
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


class InvalidComposedPromptApplier:
    def apply(self, *, context, selected_entries, routing_decision,
              scaffold_policy, scaffold_contract):
        trace = CompositionTrace(
            selected_prompt_ids=routing_decision.selected_prompt_ids,
            preserved_prompt_ids=routing_decision.selected_prompt_ids,
        )
        return finalize_composed_prompt(
            context=context,
            routing_decision=routing_decision,
            scaffold_policy=scaffold_policy,
            scaffold_contract=scaffold_contract,
            prompt_text="invalid output",
            trace=trace,
        )


def test_invalid_composed_prompt_rejected_before_writes(tmp_path):
    with pytest.raises(OfflineDryRunError, match="invalid composed prompt"):
        run(tmp_path, scaffold_applier=InvalidComposedPromptApplier())
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_reproducible_rerun_is_byte_identical(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    run(first)
    run(second)
    assert artifact_bytes(first) == artifact_bytes(second)


def test_complete_manifest_covers_all_non_manifest_artifacts(tmp_path):
    run(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    actual = {
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(manifest["artifacts"]) == actual
    assert manifest["status"] == "completed"
    assert manifest["segment_count"] == 2
    assert manifest["fallback_count"] == 1
    assert manifest["errors"] == []
    assert manifest["calls"] == {
        "captioning": 0, "dvd_reasoning": 0, "optimization": 0}
    for relative, metadata in manifest["artifacts"].items():
        assert len(metadata["sha256"]) == 64
        assert metadata["size_bytes"] == os.path.getsize(tmp_path / relative)


def test_optimize_scaffold_does_not_change_inference_outputs(tmp_path):
    fixed_config = Phase4Config(optimization=Phase4OptimizationConfig(
        optimize_scaffold=False))
    optimized_config = dataclasses.replace(
        fixed_config,
        optimization=dataclasses.replace(
            fixed_config.optimization, optimize_scaffold=True))
    fixed = run(tmp_path / "fixed", phase4_config=fixed_config)
    optimized = run(tmp_path / "optimized", phase4_config=optimized_config)

    assert fixed.routing_decisions == optimized.routing_decisions
    assert fixed.composed_prompts == optimized.composed_prompts
    fixed_manifest = json.loads(Path(fixed.manifest_path).read_text())
    optimized_manifest = json.loads(Path(optimized.manifest_path).read_text())
    assert fixed_manifest["optimization_flags"]["optimize_scaffold"] is False
    assert optimized_manifest["optimization_flags"]["optimize_scaffold"] is True


def test_fixture_uses_existing_router_boundary():
    inputs = bundle()
    decision = RuleBasedPromptRouter().route(
        inputs["contexts"][0], inputs["prompt_bank"], inputs["router_policy"])
    assert decision.selected_prompt_ids == (
        "preserve_temporal_order", "transcribe_visible_text")
