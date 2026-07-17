import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from surrogate_rollout.optimization.bounded_smoke import (
    BoundedSmokeConflictError,
    BoundedSmokeRunner,
)
from surrogate_rollout.optimization.train_roles import derive_train_roles, initial_coverage
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.schemas import (
    as_json_dict,
    Phase4Config,
    PostInterventionMode,
)


ROOT = Path(__file__).parents[1]


def _write(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return str(path.resolve())


class FakeBaseline:
    configuration_identity = {"type": "bounded-baseline-fixture"}

    def __init__(self):
        self.proposal_policy = SimpleNamespace(max_proposals=1)
        self.property_retrieval_runner = SimpleNamespace(
            retriever=SimpleNamespace(top_k=1))
        self.work_calls = 0

    def run(self, **kwargs):
        root = Path(kwargs["output_dir"])
        manifest_path = root / "manifest.json"
        video_id = kwargs["bounded_smoke_video_id"]
        complete_path = root / "baseline" / video_id / "video_complete.json"
        if not manifest_path.exists():
            self.work_calls += 1
            proposal = {
                "candidate_property_id": "candidate-1",
                "property_text": "Describe fine-grained object interactions.",
                "source_video_id": video_id,
                "source_question_ids": ["q1"],
                "motivating_failure_types": ["missed_interaction"],
                "covered_by_existing_property_ids": [],
                "proposal_rationale": "Reusable missing detail.",
                "proposer_policy_version": "fixture-proposer-v1",
            }
            proposal_path = _write(root / "proposal.json", {
                "video_id": video_id, "proposals": [proposal]})
            retrieval_path = _write(root / "retrieval.json", {
                "candidate_property_id": "candidate-1", "s_sim": ["0_10"]})
            qa_path = root / "baseline" / video_id / "baseline_qas.jsonl"
            qa_path.parent.mkdir(parents=True, exist_ok=True)
            qa_path.write_text("".join(json.dumps({
                "question_id": f"q{index}", "prediction": "A",
                "parsed_answer": "A", "errors": [],
            }) + "\n" for index in range(3)))
            _write(complete_path, {
                "video_id": video_id,
                "property_proposal_path": proposal_path,
                "property_retrieval_paths": [retrieval_path],
                "baseline_qas_path": str(qa_path.resolve()),
            })
            _write(manifest_path, {"status": "completed", "video_id": video_id})
        return SimpleNamespace(
            selected_video_ids=(video_id,), baseline_qa_count=3,
            video_manifest_paths=(str(complete_path.resolve()),),
            manifest_path=str(manifest_path.resolve()))


class FakeIntervention:
    configuration_identity = {"type": "bounded-intervention-fixture"}

    def __init__(self):
        self.work_calls = 0

    def run(self, **kwargs):
        root = Path(kwargs["output_dir"])
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            self.work_calls += 1
            transitions = _write(root / "candidate-1" / "transitions.json", {
                "transition_counts": {"wrong_to_correct": 1},
                "qas": [{"question_id": f"q{i}"} for i in range(3)],
            })
            result_path = _write(root / "candidate-1" / "result.json", {
                "status": "completed", "transitions_path": transitions})
            _write(manifest_path, {
                "status": "completed", "source_video_id": "0RxMZBLeqRI",
                "results": [{
                    "candidate_property_id": "candidate-1",
                    "result_path": result_path,
                    "transitions_path": transitions,
                }],
            })
        return SimpleNamespace(manifest_path=str(manifest_path.resolve()))


class FakeFeedback:
    configuration_identity = {"type": "bounded-feedback-fixture"}

    def __init__(self):
        self.work_calls = 0

    def run(self, **kwargs):
        root = Path(kwargs["output_dir"])
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            self.work_calls += 1
            result_path = _write(root / "candidate-1" / "result.json", {
                "status": "completed", "candidate_property_id": "candidate-1"})
            _write(manifest_path, {
                "status": "completed", "source_video_id": "0RxMZBLeqRI",
                "properties": [{"result_path": result_path}],
            })
        return SimpleNamespace(manifest_path=str(manifest_path.resolve()))


class FakeUpdateEngine:
    configuration_identity = {"type": "bounded-update-fixture"}
    confirmation_evaluator = None

    def __init__(self):
        self.aggregate_calls = 0
        self.apply_calls = 0

    def aggregate_updates(self, **kwargs):
        assert kwargs["expected_evidence_video_count"] == 1
        self.aggregate_calls += 1
        return {"plan_id": "bounded-plan", "optimize_scaffold": False}

    def apply_update_plan(self, **kwargs):
        self.apply_calls += 1
        return kwargs["prompt_bank"], kwargs["router_policy"]


class FakeMemoryUpdateEngine(FakeUpdateEngine):
    def __init__(self):
        super().__init__()
        self.memory_calls = 0
        self.router_calls = 0
        self.property_memory_runner = SimpleNamespace(
            configuration_identity={"fixture": "memory"})
        self.llm_codebook_updater = SimpleNamespace(
            configuration_identity={"fixture": "codebook"},
            response_provider=SimpleNamespace(call_count=1))
        self.llm_router_updater = SimpleNamespace(
            configuration_identity={"fixture": "router"},
            response_provider=SimpleNamespace(call_count=1))

    def run_memory_conditioned_codebook_checkpoint(self, **kwargs):
        self.memory_calls += 1
        root = Path(kwargs["output_dir"])
        snapshot = _write(root / "property_memory_v1.json", {
            "schema_version": "property_memory_v1", "properties": [],
            "candidates": []})
        manifest = _write(root / "manifest.json", {
            "schema_version": "memory_conditioned_codebook_iteration_v1",
            "status": "completed", "artifacts": {
                "property_memory_snapshot": snapshot}})
        return {"status": "completed", "artifacts": {
            "property_memory_snapshot": snapshot}, "manifest_path": manifest}

    def run_memory_conditioned_router_checkpoint(self, **kwargs):
        self.router_calls += 1
        root = Path(kwargs["output_dir"])
        bank, router = _inputs()["prompt_bank"], _inputs()["router_policy"]
        bank_path = _write(root / "provisional_bank.json", as_json_dict(bank))
        router_path = _write(root / "provisional_router.json", as_json_dict(router))
        pair_path = _write(root / "policy_pair.json", {"status": "committed"})
        manifest = _write(root / "manifest.json", {
            "schema_version": "memory_conditioned_atomic_policy_pair_v1",
            "status": "committed"})
        return {
            "status": "committed", "rendered_router_prompt_hash": "hash-v2",
            "artifacts": {
                "atomic_policy_pair": pair_path,
                "provisional_codebook": bank_path,
                "provisional_router": router_path,
            }, "manifest_path": manifest,
        }


def _inputs():
    components = json.loads((
        ROOT / "prompt_routing/fixtures/stage4_7_components.json").read_text())
    manifest = json.loads((ROOT / "split_manifest.json").read_text())
    return {
        "roles": derive_train_roles(manifest),
        "coverage_state": initial_coverage(derive_train_roles(manifest)),
        "prompt_bank": prompt_bank_from_json(components["prompt_bank"]),
        "router_policy": router_policy_from_json(components["router_policy"]),
        "scaffold_policy": scaffold_policy_from_json(components["scaffold_policy"]),
        "scaffold_contract": scaffold_contract_from_json(
            components["scaffold_contract"]),
    }


def _run(runner, tmp_path, mode, **overrides):
    values = {
        "iteration_id": "bounded-smoke-1", "video_id": "0RxMZBLeqRI",
        **_inputs(),
        "phase4_config": Phase4Config(post_intervention_mode=mode),
        "output_dir": str(tmp_path / "output"),
        "state_dir": str(tmp_path / "state"),
        "cache_dir": str(tmp_path / "cache"),
        "execution_identity": {"seed": 0, "source": "fixture"},
        "baseline_kwargs": {}, "intervention_kwargs": {},
    }
    values.update(overrides)
    return runner.run(**values)


def test_all_modes_progress_without_recomputing_upstream(tmp_path):
    baseline, intervention, feedback, update = (
        FakeBaseline(), FakeIntervention(), FakeFeedback(), FakeUpdateEngine())
    runner = BoundedSmokeRunner(
        baseline_runner=baseline, intervention_runner=intervention,
        feedback_runner=feedback, update_engine=update,
        require_real_models=False)

    qa = _run(runner, tmp_path, PostInterventionMode.QA_ONLY)
    assert qa.feedback_manifest_path is None
    assert (baseline.work_calls, intervention.work_calls, feedback.work_calls) == (1, 1, 0)
    assert update.aggregate_calls == 0

    feedback_result = _run(runner, tmp_path, PostInterventionMode.FEEDBACK_ONLY)
    assert feedback_result.feedback_manifest_path
    assert (baseline.work_calls, intervention.work_calls, feedback.work_calls) == (1, 1, 1)
    assert update.aggregate_calls == 0

    provisional = _run(runner, tmp_path, PostInterventionMode.PROVISIONAL_UPDATE)
    assert provisional.update_plan_path and provisional.provisional_bank_path
    assert (baseline.work_calls, intervention.work_calls, feedback.work_calls) == (1, 1, 1)
    assert (update.aggregate_calls, update.apply_calls) == (1, 1)
    manifest = json.loads(Path(provisional.manifest_path).read_text())
    assert manifest["post_intervention_mode"] == "provisional_update"
    assert manifest["confirmation_enabled"] is False
    assert manifest["canonical_state_mutated"] is False

    resumed = _run(runner, tmp_path, PostInterventionMode.PROVISIONAL_UPDATE)
    assert resumed.resumed is True
    assert (baseline.work_calls, intervention.work_calls, feedback.work_calls) == (1, 1, 1)
    assert (update.aggregate_calls, update.apply_calls) == (1, 1)


def test_memory_conditioned_bounded_update_and_resume(tmp_path, monkeypatch):
    update = FakeMemoryUpdateEngine()
    runner = BoundedSmokeRunner(
        baseline_runner=FakeBaseline(), intervention_runner=FakeIntervention(),
        feedback_runner=FakeFeedback(), update_engine=update,
        require_real_models=False, memory_conditioned_update=True)

    def probe(**kwargs):
        return _write(Path(kwargs["output_dir"]) / "probe.json", {
            "next_router_call_used_rendered_prompt": True})

    monkeypatch.setattr(runner, "_probe_rendered_router_prompt", probe)
    result = _run(runner, tmp_path, PostInterventionMode.PROVISIONAL_UPDATE)
    mode = json.loads((
        tmp_path / "output/mode_manifests/provisional_update.json").read_text())
    assert update.memory_calls == update.router_calls == 1
    assert mode["property_memory_snapshot_path"].endswith(
        "property_memory_v1.json")
    assert mode["rendered_router_prompt_hash"] == "hash-v2"
    assert mode["confirmed_pointer_hash_before"] is None
    assert mode["confirmed_pointer_hash_after"] is None
    assert Path(mode["router_prompt_probe_path"]).exists()
    assert result.provisional_bank_path and result.provisional_router_path

    resumed = _run(runner, tmp_path, PostInterventionMode.PROVISIONAL_UPDATE)
    assert resumed.resumed is True
    assert update.memory_calls == update.router_calls == 1


def test_moving_to_earlier_mode_preserves_later_artifacts(tmp_path):
    runner = BoundedSmokeRunner(
        baseline_runner=FakeBaseline(), intervention_runner=FakeIntervention(),
        feedback_runner=FakeFeedback(), update_engine=FakeUpdateEngine(),
        require_real_models=False)
    _run(runner, tmp_path, PostInterventionMode.PROVISIONAL_UPDATE)
    later = tmp_path / "output/mode_manifests/provisional_update.json"
    before = later.read_bytes()
    _run(runner, tmp_path, PostInterventionMode.QA_ONLY)
    assert later.read_bytes() == before
    assert (tmp_path / "output/mode_manifests/feedback_only.json").exists()


def test_identity_conflicts_and_invalid_configuration_fail_closed(tmp_path):
    runner = BoundedSmokeRunner(
        baseline_runner=FakeBaseline(), intervention_runner=FakeIntervention(),
        feedback_runner=FakeFeedback(), update_engine=FakeUpdateEngine(),
        require_real_models=False)
    _run(runner, tmp_path, PostInterventionMode.QA_ONLY)
    with pytest.raises(BoundedSmokeConflictError, match="base identity"):
        _run(
            runner, tmp_path, PostInterventionMode.FEEDBACK_ONLY,
            execution_identity={"seed": 99, "source": "conflict"})
    with pytest.raises(ValueError):
        Phase4Config(post_intervention_mode="feedback_only")
    assert Phase4Config().post_intervention_mode is \
        PostInterventionMode.PROVISIONAL_UPDATE


def test_directory_and_stage_limit_validation(tmp_path):
    baseline = FakeBaseline()
    baseline.proposal_policy.max_proposals = 2
    runner = BoundedSmokeRunner(
        baseline_runner=baseline, intervention_runner=FakeIntervention(),
        feedback_runner=FakeFeedback(), update_engine=FakeUpdateEngine(),
        require_real_models=False)
    with pytest.raises(Exception, match="max_proposals=1"):
        _run(runner, tmp_path, PostInterventionMode.QA_ONLY)

    baseline.proposal_policy.max_proposals = 1
    with pytest.raises(Exception, match="must be distinct"):
        _run(
            runner, tmp_path, PostInterventionMode.QA_ONLY,
            state_dir=str(tmp_path / "output"))


def test_completed_downstream_artifact_conflicts_fail_closed(tmp_path):
    runner = BoundedSmokeRunner(
        baseline_runner=FakeBaseline(), intervention_runner=FakeIntervention(),
        feedback_runner=FakeFeedback(), update_engine=FakeUpdateEngine(),
        require_real_models=False)
    result = _run(runner, tmp_path, PostInterventionMode.PROVISIONAL_UPDATE)
    Path(result.provisional_bank_path).write_text("{}\n")
    with pytest.raises(BoundedSmokeConflictError, match="missing or changed"):
        _run(runner, tmp_path, PostInterventionMode.PROVISIONAL_UPDATE)
