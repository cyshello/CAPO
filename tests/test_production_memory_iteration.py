import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from surrogate_rollout.optimization.final_iteration import Checkpoint3EOrchestrator
from surrogate_rollout.optimization.production_iteration import (
    MemoryConditionedProductionIterationRunner,
    ProductionIterationConflictError,
    ProductionIterationError,
    build_video_waves,
    resolve_video_selection,
)
from surrogate_rollout.optimization.train_roles import (
    derive_train_roles, initial_coverage, select_evidence_batch,
)
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json, router_policy_from_json,
    scaffold_contract_from_json, scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.schemas import Phase4Config, as_json_dict
from surrogate_rollout.schemas import sha256_json


ROOT = Path(__file__).parents[1]


def _inputs():
    components = json.loads((
        ROOT / "prompt_routing/fixtures/stage4_7_components.json").read_text())
    roles = derive_train_roles(json.loads((ROOT / "split_manifest.json").read_text()))
    return {
        "roles": roles,
        "state": initial_coverage(roles),
        "bank": prompt_bank_from_json(components["prompt_bank"]),
        "router": router_policy_from_json(components["router_policy"]),
        "scaffold": scaffold_policy_from_json(components["scaffold_policy"]),
        "contract": scaffold_contract_from_json(components["scaffold_contract"]),
    }


@pytest.mark.parametrize("count", (1, 3, 5, 8))
def test_configurable_default_and_full_pool_selection(count):
    values = _inputs()
    selected, next_state = resolve_video_selection(
        roles=values["roles"], state=values["state"],
        num_videos=count, video_ids=None)
    assert len(selected) == count
    assert len(set(selected)) == count
    assert next_state.confirmation_due is (count == 8)


def test_default_selection_is_three_and_rotation_is_deterministic():
    values = _inputs()
    first, state = resolve_video_selection(
        roles=values["roles"], state=values["state"],
        num_videos=None, video_ids=None)
    repeated, _ = resolve_video_selection(
        roles=values["roles"], state=values["state"],
        num_videos=None, video_ids=None)
    second, _ = select_evidence_batch(state, batch_size=3)
    assert len(first) == 3
    assert repeated == first
    assert not set(first) & set(second)


def test_explicit_order_is_preserved_and_count_must_agree():
    values = _inputs()
    explicit = tuple(item.video_id for item in values["roles"].evidence_videos[3:6])
    selected, _ = resolve_video_selection(
        roles=values["roles"], state=values["state"],
        num_videos=3, video_ids=explicit)
    assert selected == explicit
    with pytest.raises(ProductionIterationError, match="disagree"):
        resolve_video_selection(
            roles=values["roles"], state=values["state"],
            num_videos=2, video_ids=explicit)


def test_split_leakage_duplicate_and_unknown_video_rejected():
    values = _inputs()
    confirmation = values["roles"].confirmation_videos[0].video_id
    with pytest.raises(ProductionIterationError, match="leak"):
        resolve_video_selection(
            roles=values["roles"], state=values["state"],
            num_videos=1, video_ids=(confirmation,))
    with pytest.raises(ProductionIterationError, match="duplicates"):
        resolve_video_selection(
            roles=values["roles"], state=values["state"],
            num_videos=2, video_ids=("0RxMZBLeqRI", "0RxMZBLeqRI"))
    with pytest.raises(ProductionIterationError, match="outside"):
        resolve_video_selection(
            roles=values["roles"], state=values["state"],
            num_videos=1, video_ids=("missing",))


def test_video_count_and_concurrency_are_independent_with_multiple_waves():
    selected = tuple(f"v{index}" for index in range(8))
    waves = build_video_waves(
        selected, gpus=("4", "5", "6", "7"), max_parallel_videos=4)
    assert [len(wave) for wave in waves] == [4, 4]
    assert tuple(item.video_id for wave in waves for item in wave) == selected
    assert [item.gpu_id for item in waves[1]] == ["4", "5", "6", "7"]
    with pytest.raises(ProductionIterationError, match="worker count"):
        build_video_waves(selected, gpus=("4", "5"), max_parallel_videos=3)
    with pytest.raises(ProductionIterationError, match="unique"):
        build_video_waves(selected, gpus=("4", "4"), max_parallel_videos=1)


def test_gpu_inventory_and_concurrency_validation():
    import scripts.run_phase4_memory_iteration as launcher

    launcher._validate_gpu_configuration(
        gpus=("4", "5"), inventory={"4": 0, "5": 0},
        max_parallel_videos=2, require_free=True)
    launcher._validate_gpu_configuration(
        gpus=("4", "5"), inventory={"3": 0, "4": 0, "5": 0},
        max_parallel_videos=2, require_free=True, embedding_gpu="3")
    assert launcher._embedding_device("3") == \
        "isolated_process:physical_gpu=3:logical_device=cuda:0"
    assert launcher._embedding_device(None) == "cpu"
    with pytest.raises(ProductionIterationError, match="must not overlap"):
        launcher._validate_gpu_configuration(
            gpus=("4", "5"), inventory={"4": 0, "5": 0},
            max_parallel_videos=2, require_free=False, embedding_gpu="4")
    with pytest.raises(ProductionIterationError, match="embedding GPU"):
        launcher._validate_gpu_configuration(
            gpus=("4",), inventory={"4": 0},
            max_parallel_videos=1, require_free=False, embedding_gpu="3")
    with pytest.raises(ProductionIterationError, match="unavailable GPUs"):
        launcher._validate_gpu_configuration(
            gpus=("4", "8"), inventory={"4": 0},
            max_parallel_videos=1, require_free=False)
    with pytest.raises(ProductionIterationError, match="worker count"):
        launcher._validate_gpu_configuration(
            gpus=("4",), inventory={"4": 0},
            max_parallel_videos=2, require_free=False)
    with pytest.raises(ProductionIterationError, match="not free"):
        launcher._validate_gpu_configuration(
            gpus=("4",), inventory={"4": 100},
            max_parallel_videos=1, require_free=True)
    with pytest.raises(ProductionIterationError, match="not free"):
        launcher._validate_gpu_configuration(
            gpus=("4",), inventory={"3": 100, "4": 0},
            max_parallel_videos=1, require_free=True, embedding_gpu="3")


def _write(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return str(path.resolve())


class FakeBaseline:
    policy_version = "fake-production-baseline-v1"

    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        video_id = kwargs["bounded_smoke_video_id"]
        log = kwargs.get("stage_logger")
        if log:
            log(event="START", stage="property_proposal",
                message=f"[{video_id}] property proposal 시작")
        self.calls.append((video_id, kwargs["worker_gpus"]))
        root = Path(kwargs["output_dir"])
        proposal_path = _write(root / "proposal.json", {
            "proposals": [{
                "candidate_property_id": f"candidate-{video_id}",
                "property_text": "Describe reusable visible interactions.",
                "source_video_id": video_id,
                "source_question_ids": [f"{video_id}-q{index}" for index in range(3)],
                "motivating_failure_types": ["missed_interaction"],
                "coverage_hints": [], "proposal_rationale": "fixture",
                "proposer_policy_version": "fixture-v1",
            }]})
        retrieval_path = _write(root / "retrieval.json", {
            "candidate_property_id": f"candidate-{video_id}"})
        video_manifest = _write(root / "video_complete.json", {
            "video_id": video_id, "property_proposal_path": proposal_path,
            "property_retrieval_paths": [retrieval_path]})
        manifest = _write(root / "manifest.json", {"status": "completed"})
        if log:
            log(event="END", stage="property_proposal",
                message=f"[{video_id}] property proposal 끝", proposal_count=1)
            log(event="END", stage="similarity_retrieval",
                message=f"[{video_id}] similarity 측정 끝", retrieval_count=1)
        return SimpleNamespace(
            selected_video_ids=(video_id,), baseline_qa_count=3,
            video_manifest_paths=(video_manifest,), manifest_path=manifest)


class FakeIntervention:
    policy_version = "fake-production-intervention-v1"

    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        video_id = json.loads(Path(
            kwargs["baseline_video_manifest_path"]).read_text())["video_id"]
        self.calls.append(video_id)
        log = kwargs.get("stage_logger")
        if log:
            log(event="START", stage="intervention_recaption",
                message=f"[{video_id}] intervention recaption 시작")
            log(event="END", stage="candidate_qa",
                message=f"[{video_id}] candidate QA 평가 끝")
        path = _write(Path(kwargs["output_dir"]) / "manifest.json", {
            "status": "completed", "source_video_id": video_id, "results": []})
        return SimpleNamespace(manifest_path=path)


class FakeFeedback:
    policy_version = "fake-production-feedback-v1"

    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        video_id = json.loads(Path(
            kwargs["baseline_video_manifest_path"]).read_text())["video_id"]
        self.calls.append(video_id)
        path = _write(Path(kwargs["output_dir"]) / "manifest.json", {
            "status": "completed", "source_video_id": video_id, "properties": []})
        return SimpleNamespace(manifest_path=path)


class FakeProvider:
    def __init__(self):
        self.call_count = 0


class FakeUpdateEngine:
    policy_version = "fake-memory-update-v1"
    confirmation_evaluator = None

    def __init__(self, *, fail_router=False):
        self.property_memory_runner = SimpleNamespace(policy_version="memory-v1")
        self.llm_codebook_updater = SimpleNamespace(
            policy_version="codebook-v1", response_provider=FakeProvider())
        self.llm_router_updater = SimpleNamespace(
            policy_version="router-v1", response_provider=FakeProvider())
        self.codebook_calls = 0
        self.router_calls = 0
        self.fail_router = fail_router

    def run_memory_conditioned_codebook_checkpoint(self, **kwargs):
        self.codebook_calls += 1
        log = kwargs.get("stage_logger")
        if log:
            log(event="START", stage="memory_update",
                message="property_memory_v1 update 시작")
            log(event="END", stage="memory_update",
                message="property_memory_v1 update 끝")
            log(event="START", stage="codebook_update",
                message="LLM codebook update 시작")
        assert len(kwargs["baseline_video_manifest_paths"]) in (1, 3, 5, 8)
        root = Path(kwargs["output_dir"])
        snapshot = _write(root / "property_memory_v1.json", {
            "schema_version": "property_memory_v1", "properties": []})
        _write(root / "manifest.json", {
            "schema_version": "memory_conditioned_codebook_iteration_v1",
            "status": "completed"})
        self.llm_codebook_updater.response_provider.call_count += 1
        if log:
            log(event="END", stage="codebook_update",
                message="LLM codebook update 끝")
        return {"artifacts": {"property_memory_snapshot": snapshot}}

    def run_memory_conditioned_router_checkpoint(self, **kwargs):
        self.router_calls += 1
        log = kwargs.get("stage_logger")
        if log:
            log(event="START", stage="router_update",
                message="LLM router update 시작")
        if self.fail_router:
            raise RuntimeError("router failed")
        root = Path(kwargs["output_dir"])
        bank = _write(root / "pair" / "bank.json", {"bank_version": "candidate"})
        router = _write(root / "pair" / "router.json", {"router_version": "candidate"})
        pair = _write(root / "pair" / "policy_pair.json", {"status": "committed"})
        _write(root / "manifest.json", {"status": "committed"})
        self.llm_router_updater.response_provider.call_count += 1
        if log:
            log(event="END", stage="router_update",
                message="LLM router update 끝")
        return {"status": "committed", "artifacts": {
            "atomic_policy_pair": pair,
            "provisional_codebook": bank,
            "provisional_router": router,
        }}


class FakeHistoryBuilder:
    def __init__(self):
        self.starts = 0

    def start_worker_pool(self):
        self.starts += 1

    def worker_affinity(self, _gpu):
        from contextlib import nullcontext
        return nullcontext()


def _runner(tmp_path, *, fail_router=False, require_real=False):
    baseline, intervention, feedback = FakeBaseline(), FakeIntervention(), FakeFeedback()
    update = FakeUpdateEngine(fail_router=fail_router)
    history = FakeHistoryBuilder()
    runner = MemoryConditionedProductionIterationRunner(
        baseline_runner=baseline, intervention_runner=intervention,
        feedback_runner=feedback, update_engine=update,
        history_builder=history, require_real_models=require_real)
    return runner, baseline, intervention, feedback, update, history


def _run(
    runner, tmp_path, *, count=3, dry=False, iteration_id="iteration-1",
    execution_identity=None,
):
    values = _inputs()
    selected, next_state = resolve_video_selection(
        roles=values["roles"], state=values["state"],
        num_videos=count, video_ids=None)
    return runner.run(
        iteration_id=iteration_id, roles=values["roles"],
        coverage_state=values["state"], next_coverage_state=next_state,
        selected_video_ids=selected, selection_seed=0,
        split_manifest_path=str(ROOT / "split_manifest.json"),
        split_manifest_hash="split-hash", parent_policy_pair_hash="parent-hash",
        prompt_bank=values["bank"], router_policy=values["router"],
        scaffold_policy=values["scaffold"], scaffold_contract=values["contract"],
        phase4_config=Phase4Config(),
        output_dir=str(tmp_path / "output"), state_dir=str(tmp_path / "state"),
        cache_dir=str(tmp_path / "cache"), gpus=("4", "5", "6", "7"),
        max_parallel_videos=min(count, 4),
        execution_identity=execution_identity or {
            "models": "fixture", "prompt_versions": "v1"},
        baseline_kwargs={}, intervention_kwargs={}, dry_run_plan=dry)


def test_latest_memory_codebook_router_path_and_atomic_pair(tmp_path):
    runner, baseline, intervention, feedback, update, history = _runner(tmp_path)
    result = _run(runner, tmp_path, count=5)
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert len(baseline.calls) == len(intervention.calls) == len(feedback.calls) == 5
    assert update.codebook_calls == update.router_calls == 1
    assert history.starts == 1
    assert Path(manifest["artifacts"]["atomic_policy_pair"]).exists()
    assert manifest["num_videos"] == 5
    assert [len(wave) for wave in manifest["waves"]] == [4, 1]


def test_human_readable_iteration_log_covers_requested_stage_boundaries(tmp_path):
    runner, _, _, _, _, _ = _runner(tmp_path)
    result = _run(runner, tmp_path, count=3)
    manifest = json.loads(Path(result.manifest_path).read_text())
    log_path = Path(manifest["operational_log_path"])
    text = log_path.read_text()

    assert "1번 iter 시작" in text
    assert "video 0RxMZBLeqRI,7D-gxaie6UI,GLW9omJfAdk 병렬화 시작" in text
    for phrase in (
        "property proposal 시작", "similarity 측정 끝",
        "intervention recaption 시작", "candidate QA 평가 끝",
        "property_memory_v1 update 시작", "LLM codebook update 끝",
        "LLM router update 끝", "1번 iter 끝",
    ):
        assert phrase in text
    assert all(line.count("[") >= 2 for line in text.splitlines())
    assert "iteration_log" not in manifest["artifacts"]

    resumed = _run(runner, tmp_path, count=3)
    assert resumed.resumed
    assert "완료 artifact 재사용" in log_path.read_text()


def test_exact_resume_identity_includes_order_and_k_without_calls(tmp_path):
    runner, baseline, intervention, feedback, update, history = _runner(tmp_path)
    first = _run(runner, tmp_path, count=3)
    second = _run(runner, tmp_path, count=3)
    assert first.resumed is False and second.resumed is True
    assert len(baseline.calls) == 3
    assert update.codebook_calls == update.router_calls == 1
    assert history.starts == 1
    values = _inputs()
    selected, next_state = resolve_video_selection(
        roles=values["roles"], state=values["state"],
        num_videos=1, video_ids=None)
    with pytest.raises(ProductionIterationConflictError):
        runner.run(
            iteration_id="iteration-1", roles=values["roles"],
            coverage_state=values["state"], next_coverage_state=next_state,
            selected_video_ids=selected, selection_seed=0,
            split_manifest_path=str(ROOT / "split_manifest.json"),
            split_manifest_hash="split-hash", parent_policy_pair_hash="parent-hash",
            prompt_bank=values["bank"], router_policy=values["router"],
            scaffold_policy=values["scaffold"], scaffold_contract=values["contract"],
            phase4_config=Phase4Config(),
            output_dir=str(tmp_path / "output"), state_dir=str(tmp_path / "state"),
            cache_dir=str(tmp_path / "cache"), gpus=("4", "5", "6", "7"),
            max_parallel_videos=1, execution_identity={"models": "fixture"},
            baseline_kwargs={}, intervention_kwargs={})


def test_exact_resume_rejects_changed_prompt_or_model_identity(tmp_path):
    runner, _, _, _, _, _ = _runner(tmp_path)
    _run(runner, tmp_path, count=1, execution_identity={
        "models": "fixture", "prompt_versions": "v1"})
    with pytest.raises(ProductionIterationConflictError):
        _run(runner, tmp_path, count=1, execution_identity={
            "models": "fixture", "prompt_versions": "v2"})


def test_dry_plan_resolves_parent_memory_snapshot_and_hash(tmp_path):
    runner, _, _, _, _, _ = _runner(tmp_path)
    values = _inputs()
    snapshot = tmp_path / "parent_memory/property_memory_v1.json"
    _write(snapshot, {
        "schema_version": "property_memory_v1", "properties": [],
        "candidate_memories": [],
    })
    import hashlib
    snapshot_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    bank_hash = sha256_json(as_json_dict(values["bank"]))
    _write(tmp_path / "state/property_memory/current.json", {
        "schema_version": "property_memory_lineage_pointer_v1",
        "active_bank_version": values["bank"].bank_version,
        "active_bank_hash": bank_hash,
        "snapshot_path": str(snapshot.resolve()),
        "snapshot_hash": snapshot_hash,
    })
    result = _run(runner, tmp_path, count=1, dry=True)
    plan = json.loads(Path(result.plan_path).read_text())
    assert plan["parent_property_memory"]["snapshot_path"] == str(snapshot.resolve())
    assert plan["parent_property_memory"]["snapshot_hash"] == snapshot_hash
    assert plan["parent_policy"]["bank_version"] == values["bank"].bank_version


def test_router_failure_prevents_codebook_only_commit(tmp_path):
    runner, _, _, _, update, _ = _runner(tmp_path, fail_router=True)
    with pytest.raises(RuntimeError, match="router failed"):
        _run(runner, tmp_path, count=1)
    assert update.codebook_calls == update.router_calls == 1
    assert not (tmp_path / "output/manifest.json").exists()
    assert not (tmp_path / "output/memory_router_checkpoint/pair/policy_pair.json").exists()


def test_dry_run_plan_has_no_stage_or_worker_calls(tmp_path):
    runner, baseline, intervention, feedback, update, history = _runner(tmp_path)
    result = _run(runner, tmp_path, count=5, dry=True)
    plan = json.loads(Path(result.plan_path).read_text())
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert plan["num_videos"] == 5
    assert plan["qa_count"] == 15
    assert manifest["model_calls"] == 0
    assert not baseline.calls and not intervention.calls and not feedback.calls
    assert update.codebook_calls == update.router_calls == history.starts == 0


def test_dry_run_plan_records_dedicated_embedding_gpu(tmp_path):
    runner, _, _, _, _, _ = _runner(tmp_path)
    result = _run(runner, tmp_path, count=1, dry=True, execution_identity={
        "models": "fixture",
        "retrieval_device": (
            "isolated_process:physical_gpu=3:logical_device=cuda:0"),
        "embedding_gpu": "3",
    })
    plan = json.loads(Path(result.plan_path).read_text())
    identity = json.loads(Path(
        tmp_path / "output/iteration_identity.json").read_text())["identity"]

    assert plan["embedding_gpu"] == "3"
    assert plan["retrieval_device"] == \
        "isolated_process:physical_gpu=3:logical_device=cuda:0"
    assert identity["execution_identity"]["embedding_gpu"] == "3"


def test_launcher_does_not_import_obsolete_stage_413_path():
    source = (ROOT / "scripts/run_phase4_memory_iteration.py").read_text()
    assert "run_phase4_real_fixed_scaffold_iteration" not in source
    assert "RealFixedScaffoldIterationRunner" not in source


def test_cleanup_helper_closes_workers_for_success_and_failure(tmp_path, monkeypatch):
    import scripts.run_phase4_memory_iteration as launcher

    class Process:
        pid = 123
        alive = True

        def is_alive(self):
            return self.alive

    class Builder:
        def __init__(self):
            self.process = Process()
            self._pool_processes = {"4": self.process}

        def close_worker_pool(self):
            self.process.alive = False
            self._pool_processes = {}

    monkeypatch.setattr(launcher, "_gpu_inventory", lambda: {"4": 0})
    for name in ("success", "failure"):
        builder = Builder()
        launcher._write_cleanup(
            str(tmp_path / name), builder, ("4",),
            calls={"codebook": 0, "router": 0})
        artifact = next((tmp_path / name / "worker_cleanup").glob("*.json"))
        assert json.loads(artifact.read_text())["all_workers_released"] is True
