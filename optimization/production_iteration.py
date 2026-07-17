"""Configurable production orchestration for one memory-conditioned iteration."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from surrogate_rollout.optimization.final_iteration import (
    Checkpoint3EOrchestrator,
    FinalIterationConflictError,
    _proposal_from_json,
    _stage_identity,
    _write_immutable,
)
from surrogate_rollout.optimization.startup_models import (
    build_startup_model_manifest,
    log_startup_models,
)
from surrogate_rollout.optimization.train_roles import (
    EvidenceCoverageState,
    TrainRoleAssignment,
    initial_coverage,
    select_evidence_batch,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import (
    Phase4Config,
    PromptBankSnapshot,
    RouterPolicySnapshot,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_json, sha256_text


PLAN_SCHEMA_VERSION = "memory_production_iteration_plan_v1"
IDENTITY_SCHEMA_VERSION = "memory_production_iteration_identity_v1"
MANIFEST_SCHEMA_VERSION = "memory_production_iteration_manifest_v1"
SELECTION_POINTER_SCHEMA_VERSION = "memory_production_selection_pointer_v1"
SCHEDULER_VERSION = "deterministic_video_wave_scheduler_v1"


class ProductionIterationError(RuntimeError):
    pass


class ProductionIterationConflictError(ProductionIterationError):
    pass


@dataclass(frozen=True)
class VideoWaveAssignment:
    wave_index: int
    position: int
    video_id: str
    gpu_id: str


@dataclass(frozen=True)
class ProductionIterationResult:
    manifest_path: str
    plan_path: str
    selected_video_ids: tuple[str, ...]
    resumed: bool
    dry_run: bool


def _read_json(path: str) -> dict[str, Any]:
    with open(path) as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ProductionIterationError(f"expected JSON object: {path}")
    return value


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parent_memory_summary(
    state_dir: str, prompt_bank: PromptBankSnapshot,
) -> dict[str, Any] | None:
    pointer_path = os.path.join(
        os.path.abspath(state_dir), "property_memory", "current.json")
    if not os.path.exists(pointer_path):
        return None
    pointer = _read_json(pointer_path)
    if pointer.get("schema_version") != "property_memory_lineage_pointer_v1":
        raise ProductionIterationConflictError(
            "incompatible parent property-memory pointer schema")
    snapshot_path = str(pointer.get("snapshot_path") or "")
    if not os.path.exists(snapshot_path) or \
            _file_hash(snapshot_path) != pointer.get("snapshot_hash"):
        raise ProductionIterationConflictError(
            "parent property-memory snapshot path/hash mismatch")
    snapshot = _read_json(snapshot_path)
    if snapshot.get("schema_version") != "property_memory_v1":
        raise ProductionIterationConflictError(
            "incompatible parent property-memory snapshot schema")
    if pointer.get("active_bank_version") != prompt_bank.bank_version or \
            pointer.get("active_bank_hash") != sha256_json(as_json_dict(prompt_bank)):
        raise ProductionIterationConflictError(
            "parent property-memory lineage does not match the active bank")
    return {
        "pointer_path": pointer_path,
        "pointer_hash": _file_hash(pointer_path),
        "snapshot_path": os.path.abspath(snapshot_path),
        "snapshot_hash": _file_hash(snapshot_path),
        "snapshot_schema_version": snapshot["schema_version"],
        "active_bank_version": pointer.get("active_bank_version"),
        "active_bank_hash": pointer.get("active_bank_hash"),
    }


def _write_json(path: str, value: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(
        path,
        json.dumps(as_json_dict(value), sort_keys=True, ensure_ascii=False, indent=2)
        + "\n",
    )
    return os.path.abspath(path)


def _seeded_initial_coverage(
    roles: TrainRoleAssignment, selection_seed: int,
) -> EvidenceCoverageState:
    base = initial_coverage(roles).rotation_order
    offset = selection_seed % len(base)
    return EvidenceCoverageState(rotation_order=base[offset:] + base[:offset])


def _advance_explicit_selection(
    state: EvidenceCoverageState, selected: tuple[str, ...],
) -> EvidenceCoverageState:
    if state.confirmation_due:
        raise ProductionIterationError(
            "confirmation is due; a new evidence iteration cannot start")
    if not selected or len(selected) != len(set(selected)):
        raise ProductionIterationError("selected video IDs must be non-empty and unique")
    if not set(selected) <= set(state.rotation_order):
        raise ProductionIterationError("selected video is outside the evidence pool")
    covered = set(state.used_since_confirmation) | set(selected)
    return EvidenceCoverageState(
        rotation_order=state.rotation_order,
        used_since_confirmation=tuple(
            item for item in state.rotation_order if item in covered),
        rotation_cursor=(state.rotation_order.index(selected[-1]) + 1)
        % len(state.rotation_order),
        coverage_cycle=state.coverage_cycle,
        iterations_since_confirmation=state.iterations_since_confirmation + 1,
        confirmation_due=len(covered) == len(state.rotation_order),
    )


def resolve_video_selection(
    *, roles: TrainRoleAssignment, state: EvidenceCoverageState,
    num_videos: int | None, video_ids: Sequence[str] | None,
) -> tuple[tuple[str, ...], EvidenceCoverageState]:
    evidence = tuple(item.video_id for item in roles.evidence_videos)
    explicit = tuple(str(item).strip() for item in (video_ids or ()) if str(item).strip())
    if explicit:
        if len(explicit) != len(set(explicit)):
            raise ProductionIterationError("--video-ids contains duplicates")
        if num_videos is not None and num_videos != len(explicit):
            raise ProductionIterationError(
                "--num-videos and --video-ids disagree")
        forbidden = (
            set(explicit) & (
                {item.video_id for item in roles.confirmation_videos}
                | set(roles.validation_video_ids) | set(roles.test_video_ids)
            )
        )
        if forbidden:
            raise ProductionIterationError(
                f"selected videos leak outside the evidence pool: {sorted(forbidden)}")
        unknown = set(explicit) - set(evidence)
        if unknown:
            raise ProductionIterationError(
                f"selected videos are outside the evidence pool: {sorted(unknown)}")
        return explicit, _advance_explicit_selection(state, explicit)
    count = 3 if num_videos is None else int(num_videos)
    if not 1 <= count <= len(evidence):
        raise ProductionIterationError(
            f"--num-videos must be between 1 and {len(evidence)}")
    try:
        return select_evidence_batch(state, batch_size=count)
    except (ValueError, RuntimeError) as exc:
        raise ProductionIterationError(str(exc)) from exc


def build_video_waves(
    selected_video_ids: Sequence[str], *, gpus: Sequence[str],
    max_parallel_videos: int,
) -> tuple[tuple[VideoWaveAssignment, ...], ...]:
    gpu_ids = tuple(str(item).strip() for item in gpus if str(item).strip())
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ProductionIterationError("--gpus must contain unique GPU IDs")
    if not 1 <= max_parallel_videos <= len(gpu_ids):
        raise ProductionIterationError(
            "--max-parallel-videos must be between 1 and the worker count")
    selected = tuple(selected_video_ids)
    waves = []
    for wave_index, start in enumerate(range(0, len(selected), max_parallel_videos)):
        rows = []
        for position, video_id in enumerate(
                selected[start:start + max_parallel_videos]):
            rows.append(VideoWaveAssignment(
                wave_index=wave_index, position=position,
                video_id=video_id, gpu_id=gpu_ids[position]))
        waves.append(tuple(rows))
    return tuple(waves)


def _coverage_from_json(value: Mapping[str, Any]) -> EvidenceCoverageState:
    return EvidenceCoverageState(
        rotation_order=tuple(value["rotation_order"]),
        used_since_confirmation=tuple(value.get("used_since_confirmation") or ()),
        rotation_cursor=int(value.get("rotation_cursor", 0)),
        coverage_cycle=int(value.get("coverage_cycle", 0)),
        iterations_since_confirmation=int(value.get("iterations_since_confirmation", 0)),
        confirmation_due=bool(value.get("confirmation_due", False)),
    )


class MemoryConditionedProductionIterationRunner:
    """Run K evidence videos and commit only an atomic provisional pair."""

    policy_version = "memory_conditioned_production_iteration_v1"

    def __init__(
        self, *, baseline_runner: Any, intervention_runner: Any,
        feedback_runner: Any, update_engine: Checkpoint3EOrchestrator,
        history_builder: Any, require_real_models: bool = True,
    ) -> None:
        self.baseline_runner = baseline_runner
        self.intervention_runner = intervention_runner
        self.feedback_runner = feedback_runner
        self.update_engine = update_engine
        self.history_builder = history_builder
        self.require_real_models = require_real_models

    @staticmethod
    def load_selection_state(
        *, roles: TrainRoleAssignment, state_dir: str, selection_seed: int,
    ) -> EvidenceCoverageState:
        pointer_path = os.path.join(
            os.path.abspath(state_dir), "production_selection", "current.json")
        if not os.path.exists(pointer_path):
            return _seeded_initial_coverage(roles, selection_seed)
        pointer = _read_json(pointer_path)
        if pointer.get("schema_version") != SELECTION_POINTER_SCHEMA_VERSION:
            raise ProductionIterationConflictError(
                "incompatible production selection pointer schema")
        if int(pointer.get("selection_seed")) != selection_seed:
            raise ProductionIterationConflictError(
                "selection seed conflicts with saved rotation state")
        return _coverage_from_json(pointer["coverage_state"])

    @staticmethod
    def _validate_directories(output_dir: str, state_dir: str, cache_dir: str) -> None:
        paths = tuple(os.path.abspath(item) for item in (output_dir, state_dir, cache_dir))
        if len(set(paths)) != 3:
            raise ProductionIterationError(
                "output, state, and cache directories must be distinct")
        for index, left in enumerate(paths):
            for right in paths[index + 1:]:
                common = os.path.commonpath((left, right))
                if common in (left, right):
                    raise ProductionIterationError(
                        "output, state, and cache directories must not be nested")

    @staticmethod
    def _run_wave(
        wave: Sequence[VideoWaveAssignment], fn: Callable[[VideoWaveAssignment], Any],
    ) -> dict[str, Any]:
        completed: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = {executor.submit(fn, item): item for item in wave}
            for future in as_completed(futures):
                item = futures[future]
                completed[item.video_id] = future.result()
        return completed

    def _resume(
        self, *, manifest_path: str, identity_hash: str,
        selected_video_ids: tuple[str, ...], plan_path: str,
    ) -> ProductionIterationResult:
        manifest = _read_json(manifest_path)
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or \
                manifest.get("status") != "completed" or \
                manifest.get("identity_hash") != identity_hash or \
                tuple(manifest.get("selected_video_ids") or ()) != selected_video_ids:
            raise ProductionIterationConflictError(
                "completed production iteration identity conflicts")
        artifacts = manifest.get("artifacts") or {}
        hashes = manifest.get("artifact_hashes") or {}
        if set(artifacts) != set(hashes) or any(
                not os.path.exists(path) or _file_hash(path) != hashes[name]
                for name, path in artifacts.items()):
            raise ProductionIterationConflictError(
                "completed production iteration artifact closure mismatch")
        return ProductionIterationResult(
            manifest_path=os.path.abspath(manifest_path),
            plan_path=os.path.abspath(plan_path),
            selected_video_ids=selected_video_ids, resumed=True, dry_run=False)

    def run(
        self, *, iteration_id: str, roles: TrainRoleAssignment,
        coverage_state: EvidenceCoverageState, next_coverage_state: EvidenceCoverageState,
        selected_video_ids: tuple[str, ...], selection_seed: int,
        split_manifest_path: str, split_manifest_hash: str,
        parent_policy_pair_hash: str, prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot, scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract, phase4_config: Phase4Config,
        output_dir: str, state_dir: str, cache_dir: str,
        gpus: tuple[str, ...], max_parallel_videos: int,
        execution_identity: Mapping[str, Any], baseline_kwargs: Mapping[str, Any],
        intervention_kwargs: Mapping[str, Any], dry_run_plan: bool = False,
    ) -> ProductionIterationResult:
        self._validate_directories(output_dir, state_dir, cache_dir)
        output_dir, state_dir, cache_dir = map(os.path.abspath, (
            output_dir, state_dir, cache_dir))
        if not selected_video_ids or len(selected_video_ids) != len(set(selected_video_ids)):
            raise ProductionIterationError("iteration requires unique evidence videos")
        evidence_ids = {item.video_id for item in roles.evidence_videos}
        if set(selected_video_ids) - evidence_ids:
            raise ProductionIterationError("iteration contains a non-evidence video")
        waves = build_video_waves(
            selected_video_ids, gpus=gpus,
            max_parallel_videos=max_parallel_videos)
        qa_count = 3 * len(selected_video_ids)
        confirmed_pointer = os.path.join(state_dir, "confirmed", "current.json")
        confirmed_before = (
            _file_hash(confirmed_pointer) if os.path.exists(confirmed_pointer) else None)
        parent_memory = _parent_memory_summary(state_dir, prompt_bank)
        identity = {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "policy_version": self.policy_version,
            "iteration_id": iteration_id,
            "ordered_video_ids": selected_video_ids,
            "num_videos": len(selected_video_ids),
            "selection_seed": selection_seed,
            "rotation_position": coverage_state.rotation_cursor,
            "coverage_state": coverage_state,
            "next_coverage_state": next_coverage_state,
            "split_manifest_path": os.path.abspath(split_manifest_path),
            "split_manifest_hash": split_manifest_hash,
            "parent_policy_pair_hash": parent_policy_pair_hash,
            "parent_property_memory": parent_memory,
            "gpus": gpus,
            "max_parallel_videos": max_parallel_videos,
            "waves": waves,
            "components": {
                "bank": prompt_bank, "router": router_policy,
                "scaffold": scaffold_policy, "contract": scaffold_contract,
            },
            "stages": {
                "baseline": _stage_identity(self.baseline_runner),
                "intervention": _stage_identity(self.intervention_runner),
                "feedback": _stage_identity(self.feedback_runner),
                "property_memory": _stage_identity(
                    self.update_engine.property_memory_runner),
                "llm_codebook_updater": _stage_identity(
                    self.update_engine.llm_codebook_updater),
                "llm_router_updater": _stage_identity(
                    self.update_engine.llm_router_updater),
            },
            "execution_identity": as_json_dict(execution_identity),
            "directories": {
                "output": output_dir, "state": state_dir, "cache": cache_dir},
            "confirmation_enabled": False,
        }
        identity_hash = sha256_text(dumps_canonical(identity))
        plan = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "status": "planned",
            "iteration_id": iteration_id,
            "selected_video_ids": selected_video_ids,
            "num_videos": len(selected_video_ids),
            "qa_count": qa_count,
            "selection_seed": selection_seed,
            "rotation_position": coverage_state.rotation_cursor,
            "gpu_ids": gpus,
            "max_parallel_videos": max_parallel_videos,
            "scheduler_version": SCHEDULER_VERSION,
            "waves": waves,
            "parent_policy_pair_hash": parent_policy_pair_hash,
            "parent_policy": {
                "bank_version": prompt_bank.bank_version,
                "bank_hash": sha256_json(as_json_dict(prompt_bank)),
                "router_version": router_policy.router_version,
                "router_hash": sha256_json(as_json_dict(router_policy)),
            },
            "parent_property_memory": parent_memory,
            "directories": identity["directories"],
            "expected_stages": (
                "baseline", "intervention", "feedback", "property_memory_v1",
                "llm_codebook_updater", "llm_router_updater",
                "rendered_router_prompt", "atomic_provisional_policy_pair"),
            "resume_identity_hash": identity_hash,
            "dry_run": dry_run_plan,
        }
        identity_path = os.path.join(output_dir, "iteration_identity.json")
        plan_path = os.path.join(output_dir, "iteration_plan.json")
        if os.path.isdir(output_dir) and os.listdir(output_dir) and \
                not os.path.exists(identity_path):
            raise ProductionIterationConflictError(
                "non-empty output directory has no compatible iteration identity")
        for path in (output_dir, state_dir, cache_dir):
            os.makedirs(path, exist_ok=True)
        try:
            _write_immutable(identity_path, {
                "schema_version": IDENTITY_SCHEMA_VERSION,
                "identity_hash": identity_hash, "identity": identity})
            _write_immutable(plan_path, plan)
        except FinalIterationConflictError as exc:
            raise ProductionIterationConflictError(
                "output identity conflicts with an existing run") from exc
        manifest_path = os.path.join(output_dir, "manifest.json")
        startup_models_path = None
        if self.require_real_models and not dry_run_plan:
            startup = build_startup_model_manifest(
                iteration_id=iteration_id,
                baseline_runner=self.baseline_runner,
                intervention_runner=self.intervention_runner,
                feedback_runner=self.feedback_runner,
                confirmation_evaluator=self.update_engine.confirmation_evaluator)
            startup_models_path = log_startup_models(output_dir, startup)
        if os.path.exists(manifest_path):
            manifest = _read_json(manifest_path)
            if dry_run_plan and manifest.get("status") == "dry_run_planned" and \
                    manifest.get("identity_hash") == identity_hash:
                return ProductionIterationResult(
                    manifest_path=os.path.abspath(manifest_path),
                    plan_path=os.path.abspath(plan_path),
                    selected_video_ids=selected_video_ids, resumed=True, dry_run=True)
            return self._resume(
                manifest_path=manifest_path, identity_hash=identity_hash,
                selected_video_ids=selected_video_ids, plan_path=plan_path)
        if dry_run_plan:
            manifest_path = _write_immutable(manifest_path, {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "status": "dry_run_planned", "identity_hash": identity_hash,
                "selected_video_ids": selected_video_ids,
                "num_videos": len(selected_video_ids), "qa_count": qa_count,
                "plan_path": plan_path, "model_calls": 0,
                "confirmed_pointer_hash_before": confirmed_before,
                "confirmed_pointer_hash_after": confirmed_before,
            })
            return ProductionIterationResult(
                manifest_path=manifest_path, plan_path=plan_path,
                selected_video_ids=selected_video_ids, resumed=False, dry_run=True)

        self.history_builder.start_worker_pool()
        cache_manifest_path = os.path.join(cache_dir, "caption_cache_manifest.jsonl")
        baseline_by_video: dict[str, Any] = {}
        intervention_by_video: dict[str, Any] = {}
        feedback_by_video: dict[str, Any] = {}

        def run_baseline(item: VideoWaveAssignment) -> Any:
            with self.history_builder.worker_affinity(item.gpu_id):
                return self.baseline_runner.run(
                    roles=roles, coverage_state=coverage_state,
                    prompt_bank=prompt_bank, router_policy=router_policy,
                    scaffold_policy=scaffold_policy,
                    scaffold_contract=scaffold_contract,
                    phase4_config=phase4_config,
                    output_dir=os.path.join(
                        output_dir, "baseline_videos", item.video_id),
                    bounded_smoke_video_id=item.video_id,
                    caption_cache_root=os.path.join(cache_dir, "captions"),
                    cache_manifest_path=cache_manifest_path,
                    worker_gpus=(item.gpu_id,), **dict(baseline_kwargs))

        for wave in waves:
            baseline_by_video.update(self._run_wave(wave, run_baseline))

        baseline_manifest_paths = []
        video_manifest_paths = []
        proposal_by_video: dict[str, tuple[Any, ...]] = {}
        retrieval_by_video: dict[str, tuple[str, ...]] = {}
        for video_id in selected_video_ids:
            result = baseline_by_video[video_id]
            if tuple(result.selected_video_ids) != (video_id,) or \
                    int(result.baseline_qa_count) != 3:
                raise ProductionIterationError(
                    f"baseline did not complete three QAs for {video_id}")
            baseline_manifest_paths.append(result.manifest_path)
            video_manifest = result.video_manifest_paths[0]
            video_manifest_paths.append(video_manifest)
            record = _read_json(_read_json(video_manifest)["property_proposal_path"])
            proposal_by_video[video_id] = tuple(
                _proposal_from_json(item) for item in record.get("proposals") or ())
            retrieval_by_video[video_id] = tuple(
                _read_json(video_manifest).get("property_retrieval_paths") or ())
        baseline_batch_manifest = _write_immutable(
            os.path.join(output_dir, "baseline_batch_manifest.json"), {
                "schema_version": "memory_production_baseline_batch_v1",
                "status": "completed", "iteration_id": iteration_id,
                "selected_video_ids": selected_video_ids,
                "video_manifest_paths": video_manifest_paths,
                "baseline_manifest_paths": baseline_manifest_paths,
                "qa_count": qa_count, "waves": waves})

        def run_intervention(item: VideoWaveAssignment) -> Any:
            video_id = item.video_id
            with self.history_builder.worker_affinity(item.gpu_id):
                return self.intervention_runner.run(
                    iteration_id=iteration_id,
                    baseline_video_manifest_path=video_manifest_paths[
                        selected_video_ids.index(video_id)],
                    proposals=proposal_by_video[video_id],
                    retrieval_artifact_paths=retrieval_by_video[video_id],
                    prompt_bank=prompt_bank, router_policy=router_policy,
                    scaffold_policy=scaffold_policy,
                    scaffold_contract=scaffold_contract,
                    output_dir=os.path.join(
                        output_dir, "interventions", video_id),
                    **dict(intervention_kwargs))

        for wave in waves:
            intervention_by_video.update(self._run_wave(wave, run_intervention))
        intervention_manifest_paths = tuple(
            intervention_by_video[item].manifest_path for item in selected_video_ids)

        def run_feedback(item: VideoWaveAssignment) -> Any:
            video_id = item.video_id
            return self.feedback_runner.run(
                iteration_id=iteration_id,
                baseline_video_manifest_path=video_manifest_paths[
                    selected_video_ids.index(video_id)],
                proposals=proposal_by_video[video_id],
                retrieval_artifact_paths=retrieval_by_video[video_id],
                intervention_manifest_path=intervention_by_video[video_id].manifest_path,
                prompt_bank=prompt_bank,
                output_dir=os.path.join(output_dir, "feedback", video_id))

        for wave in waves:
            feedback_by_video.update(self._run_wave(wave, run_feedback))
        feedback_manifest_paths = tuple(
            feedback_by_video[item].manifest_path for item in selected_video_ids)

        codebook = self.update_engine.run_memory_conditioned_codebook_checkpoint(
            iteration_id=iteration_id,
            iteration_ordinal=coverage_state.iterations_since_confirmation + 1,
            prompt_bank=prompt_bank,
            baseline_video_manifest_paths=tuple(video_manifest_paths),
            intervention_manifest_paths=intervention_manifest_paths,
            feedback_manifest_paths=feedback_manifest_paths,
            output_dir=os.path.join(output_dir, "memory_codebook_checkpoint"),
            state_dir=state_dir)
        codebook_manifest_path = os.path.join(
            output_dir, "memory_codebook_checkpoint", "manifest.json")
        router = self.update_engine.run_memory_conditioned_router_checkpoint(
            iteration_id=iteration_id, parent_router_policy=router_policy,
            codebook_checkpoint_manifest_path=codebook_manifest_path,
            output_dir=os.path.join(output_dir, "memory_router_checkpoint"),
            state_dir=state_dir)
        if router.get("status") != "committed":
            raise ProductionIterationError(
                "router checkpoint did not atomically commit the policy pair")
        confirmed_after = (
            _file_hash(confirmed_pointer) if os.path.exists(confirmed_pointer) else None)
        if confirmed_before != confirmed_after:
            raise ProductionIterationError(
                "confirmed production pointer changed during provisional iteration")
        artifacts = {
            "iteration_plan": plan_path,
            "iteration_identity": identity_path,
            "baseline_batch_manifest": baseline_batch_manifest,
            "memory_codebook_checkpoint": codebook_manifest_path,
            "memory_router_checkpoint": os.path.join(
                output_dir, "memory_router_checkpoint", "manifest.json"),
            "atomic_policy_pair": router["artifacts"]["atomic_policy_pair"],
            "provisional_codebook": router["artifacts"]["provisional_codebook"],
            "provisional_router": router["artifacts"]["provisional_router"],
            "property_memory_snapshot": codebook["artifacts"][
                "property_memory_snapshot"],
        }
        if startup_models_path is not None:
            artifacts["startup_models"] = startup_models_path
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "completed", "iteration_id": iteration_id,
            "identity_hash": identity_hash,
            "selected_video_ids": selected_video_ids,
            "num_videos": len(selected_video_ids), "qa_count": qa_count,
            "waves": waves, "artifacts": artifacts,
            "artifact_hashes": {name: _file_hash(path)
                                for name, path in artifacts.items()},
            "component_update_provider_calls": {
                "codebook": int(getattr(
                    self.update_engine.llm_codebook_updater.response_provider,
                    "call_count", 0)),
                "router": int(getattr(
                    self.update_engine.llm_router_updater.response_provider,
                    "call_count", 0)),
            },
            "confirmed_pointer_hash_before": confirmed_before,
            "confirmed_pointer_hash_after": confirmed_after,
            "confirmed_production_pointer_mutated": False,
            "confirmation_run": False,
        }
        manifest_path = _write_immutable(manifest_path, manifest)
        selection_pointer = os.path.join(
            state_dir, "production_selection", "current.json")
        _write_json(selection_pointer, {
            "schema_version": SELECTION_POINTER_SCHEMA_VERSION,
            "latest_iteration_id": iteration_id,
            "latest_manifest_path": manifest_path,
            "latest_manifest_hash": _file_hash(manifest_path),
            "selection_seed": selection_seed,
            "coverage_state": next_coverage_state,
        })
        return ProductionIterationResult(
            manifest_path=manifest_path, plan_path=plan_path,
            selected_video_ids=selected_video_ids, resumed=False, dry_run=False)
