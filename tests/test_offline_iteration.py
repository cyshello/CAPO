"""Stage 4.12 complete saved-fixture iteration tests."""

import json
from pathlib import Path

import pytest

from surrogate_rollout.optimization.meta_knowledge import meta_knowledge_from_json
from surrogate_rollout.optimization.offline_iteration import (
    PromptRoutingOptimizationLoop,
)
from surrogate_rollout.optimization.policies.saved_fixture_feedback import (
    SavedFixtureFeedbackGenerator,
)
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.schemas import (
    Phase4Config,
    Phase4OptimizationConfig,
    dumps_canonical,
)
from surrogate_rollout.schemas import QAExample
from surrogate_rollout.scripts.run_phase4_offline_iteration import _saved_pair


ROOT = Path(__file__).parents[1]
COMPONENTS = ROOT / "prompt_routing/fixtures/stage4_7_components.json"
ITERATION = ROOT / "optimization/fixtures/stage4_12_iteration.json"
VALIDATION = ROOT / "optimization/fixtures/stage4_11_validation.json"


def _qa(value):
    return QAExample(**{**value, "options": tuple(value.get("options") or ())})


def run_iteration(tmp_path, optimize_scaffold, *, commit=False):
    components = json.loads(COMPONENTS.read_text())
    fixture = json.loads(ITERATION.read_text())
    validation = json.loads(VALIDATION.read_text())
    baseline, candidate = _saved_pair(fixture)
    bank = prompt_bank_from_json(components["prompt_bank"])
    router = router_policy_from_json(components["router_policy"])
    scaffold = scaffold_policy_from_json(components["scaffold_policy"])
    contract = scaffold_contract_from_json(components["scaffold_contract"])
    before = dumps_canonical((bank, router, scaffold, contract))
    result = PromptRoutingOptimizationLoop(
        feedback_generator=SavedFixtureFeedbackGenerator(
            fixture["feedback_overrides"])).run_iteration(
        input_bank=bank, input_router=router, input_scaffold=scaffold,
        scaffold_contract=contract, baseline_results=baseline,
        candidate_results=candidate,
        confirmation_examples=tuple(
            _qa(item) for item in validation["confirmation_examples"]),
        regression_examples=tuple(
            _qa(item) for item in validation["regression_examples"]),
        proposal_lineage=validation["proposal_lineage"],
        saved_scores=validation["saved_scores"],
        initial_meta_knowledge=tuple(meta_knowledge_from_json(item)
                                     for item in validation["review_knowledge"]),
        config=Phase4Config(
            dry_run=True, commit=False,
            optimization=Phase4OptimizationConfig(
                optimize_scaffold=optimize_scaffold)),
        output_dir=str(tmp_path), commit=commit)
    assert dumps_canonical((bank, router, scaffold, contract)) == before
    return result


def read(path):
    return json.loads(Path(path).read_text())


def test_disabled_scaffold_creates_typed_skip_without_candidate(tmp_path):
    result = run_iteration(tmp_path, False)
    proposal = read(result.scaffold_proposal_artifact)
    assert proposal["skipped_reason"] == "disabled_by_config"
    assert proposal["candidate_policy"] is None
    assert proposal["provenance"]["proposal_policy_calls"] == 0
    assert "scaffold" not in result.validation_artifacts
    assert not (tmp_path / "previews/scaffold.json").exists()


def test_enabled_scaffold_creates_supported_validated_preview(tmp_path):
    result = run_iteration(tmp_path, True)
    proposal = read(result.scaffold_proposal_artifact)
    validation = read(result.validation_artifacts["scaffold"])
    assert proposal["skipped_reason"] == "none"
    assert proposal["candidate_policy"] is not None
    assert validation["decision"] == "accept"
    assert (tmp_path / "previews/scaffold.json").exists()


def test_complete_artifact_chain_is_written(tmp_path):
    result = run_iteration(tmp_path, True)
    for path in (
        result.evidence_artifact, result.feedback_artifact,
        result.meta_knowledge_artifact, result.bank_proposal_artifact,
        result.router_proposal_artifact, result.scaffold_proposal_artifact,
        *result.review_artifacts.values(), *result.validation_artifacts.values(),
    ):
        assert Path(path).is_file()
    assert result.status == "dry_run"


def test_bank_and_router_outputs_are_comparable_across_ablation(tmp_path):
    fixed = run_iteration(tmp_path / "fixed", False)
    enabled = run_iteration(tmp_path / "enabled", True)
    for component, proposal_field in (
        ("prompt_bank", "bank_proposal_artifact"),
        ("router", "router_proposal_artifact"),
    ):
        assert read(getattr(fixed, proposal_field)) == \
            read(getattr(enabled, proposal_field))
        assert read(fixed.review_artifacts[component]) == \
            read(enabled.review_artifacts[component])
        assert read(fixed.validation_artifacts[component]) == \
            read(enabled.validation_artifacts[component])


def test_iteration_records_zero_calls_and_no_commits(tmp_path):
    result = run_iteration(tmp_path, True)
    summary = read(tmp_path / "run_summary.json")
    assert set(summary["calls"].values()) == {0}
    assert summary["canonical_state_mutated"] is False
    assert summary["component_updates_committed"] is False
    assert result.committed_bank_version is None
    assert result.committed_router_version is None
    assert result.committed_scaffold_version is None


def test_commit_request_is_rejected_before_artifacts(tmp_path):
    with pytest.raises(ValueError, match="dry-run preview only"):
        run_iteration(tmp_path, False, commit=True)
    assert not tmp_path.exists() or not tuple(tmp_path.iterdir())


def test_fixed_fixture_iteration_is_deterministic(tmp_path):
    first = run_iteration(tmp_path / "first", True)
    second = run_iteration(tmp_path / "second", True)
    assert first.iteration_id == second.iteration_id
    assert read(first.bank_proposal_artifact) == read(second.bank_proposal_artifact)
    assert read(first.router_proposal_artifact) == read(second.router_proposal_artifact)
    assert read(first.scaffold_proposal_artifact) == \
        read(second.scaffold_proposal_artifact)
