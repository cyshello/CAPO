"""Checkpoint 3E artifact-driven final Phase 4 iteration orchestration."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol

from surrogate_rollout.optimization.llm_codebook_updater import (
    CHECKPOINT_MANIFEST_SCHEMA_VERSION as CODEBOOK_UPDATER_MANIFEST_SCHEMA_VERSION,
)
from surrogate_rollout.optimization.iteration_state import (
    ComponentVersions,
    ConfirmedCheckpointState,
    ProvisionalIterationState,
)
from surrogate_rollout.optimization.property_proposal import (
    CandidatePropertyProposal,
    candidate_property_proposal_from_json,
)
from surrogate_rollout.optimization.stage_logging import StageLogger, emit_stage
from surrogate_rollout.optimization.train_roles import (
    EvidenceCoverageState,
    TrainRoleAssignment,
    reset_after_confirmation,
    select_evidence_batch,
)
from surrogate_rollout.prompt_routing.persistence import (
    _atomic_write_text,
    prompt_bank_from_json,
    router_policy_from_json,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    PromptEntry,
    RouterPolicySnapshot,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    as_json_dict,
    dumps_canonical,
    make_component_version,
)
from surrogate_rollout.prompt_routing.validators import (
    validate_prompt_bank,
    validate_router_against_bank,
)
from surrogate_rollout.schemas import sha256_json, sha256_text


CODEBOOK_ACTIONS = ("add", "revise", "merge", "retire", "no_op")
ROUTER_POLARITIES = ("positive", "negative")


class FinalIterationError(RuntimeError):
    pass


class FinalIterationConflictError(FinalIterationError):
    pass


class ConfirmationEvaluator(Protocol):
    def evaluate(
        self, *, confirmation_videos: tuple[Any, ...],
        incumbent_bank: PromptBankSnapshot,
        incumbent_router: RouterPolicySnapshot,
        candidate_bank: PromptBankSnapshot,
        candidate_router: RouterPolicySnapshot,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        output_dir: str,
    ) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class CodebookDecision:
    decision_id: str
    action: str
    candidate_property_ids: tuple[str, ...]
    candidate_lineage: tuple[Mapping[str, str], ...]
    target_property_ids: tuple[str, ...]
    resulting_property_id: str | None
    property_text: str | None
    supporting_feedback_ids: tuple[str, ...]
    supporting_video_ids: tuple[str, ...]
    positive_count: int
    negative_count: int
    rationale: str


@dataclass(frozen=True)
class RouterSupervisionUpdate:
    supervision_id: str
    property_id: str
    polarity: str
    evidence: str
    context_hash: str
    source_feedback_id: str
    source_video_id: str
    source_question_id: str
    attributed_segment_ids: tuple[str, ...]


@dataclass(frozen=True)
class IterationUpdatePlan:
    plan_id: str
    input_bank_version: str
    input_router_version: str
    optimize_scaffold: bool
    codebook_decisions: tuple[CodebookDecision, ...]
    router_supervision: tuple[RouterSupervisionUpdate, ...]
    accepted_feedback_ids: tuple[str, ...]
    rejected_feedback_count: int
    conflict_resolutions: tuple[str, ...]


@dataclass(frozen=True)
class ConfirmationDecision:
    status: str
    accepted: bool
    incumbent_accuracy: float
    candidate_accuracy: float
    correct_to_wrong_question_ids: tuple[str, ...]
    errors: tuple[str, ...]
    criterion: Mapping[str, Any]
    active_bank_version: str
    active_router_version: str
    rollback_to_checkpoint_id: str | None
    artifact_path: str


@dataclass(frozen=True)
class FinalIterationResult:
    iteration_id: str
    status: str
    manifest_path: str
    selected_video_ids: tuple[str, ...]
    baseline_manifest_path: str
    intervention_manifest_paths: tuple[str, ...]
    feedback_manifest_paths: tuple[str, ...]
    update_plan_path: str
    provisional_bank_path: str
    provisional_router_path: str
    provisional_state_path: str
    confirmation_decision_path: str | None
    confirmed_checkpoint_path: str | None
    active_bank_path: str
    active_router_path: str
    active_provisional_reference_path: str
    confirmed_pointer_path: str
    next_coverage_state: EvidenceCoverageState
    resumed: bool = False


@dataclass(frozen=True)
class _ResolvedPolicy:
    prompt_bank: PromptBankSnapshot
    router_policy: RouterPolicySnapshot
    confirmed_prompt_bank: PromptBankSnapshot
    confirmed_router_policy: RouterPolicySnapshot
    confirmed_bank_path: str
    confirmed_router_path: str
    confirmed_checkpoint_path: str
    provisional_lineage: tuple[str, ...]
    active_reference_path: str
    confirmed_pointer_path: str


def _read_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise FinalIterationError(f"expected JSON object: {path}")
    return value


def _write_immutable(path: str, value: Any) -> str:
    text = json.dumps(
        as_json_dict(value), sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    if os.path.exists(path):
        with open(path) as f:
            if f.read() != text:
                raise FinalIterationConflictError(
                    f"immutable artifact exists with different content: {path}")
        return os.path.abspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, text)
    return os.path.abspath(path)


def _write_pointer(path: str, value: Any) -> str:
    text = json.dumps(
        as_json_dict(value), sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, text)
    return os.path.abspath(path)


def _file_hash(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_hash(value: Any) -> str:
    return sha256_json(as_json_dict(value))


def _coverage_hash(value: EvidenceCoverageState) -> str:
    return sha256_text(dumps_canonical(value))


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _next_version(kind: str, current: str, content: Any) -> str:
    match = re.search(r"_v(\d+)", current)
    if not match:
        raise FinalIterationError(f"cannot increment component version {current!r}")
    return make_component_version(
        kind, int(match.group(1)) + 1, sha256_json(as_json_dict(content)))


def _proposal_from_json(value: Mapping[str, Any]) -> CandidatePropertyProposal:
    return candidate_property_proposal_from_json(value)


def _coverage_from_json(value: Mapping[str, Any]) -> EvidenceCoverageState:
    return EvidenceCoverageState(
        rotation_order=tuple(value["rotation_order"]),
        used_since_confirmation=tuple(value.get("used_since_confirmation") or ()),
        rotation_cursor=int(value.get("rotation_cursor", 0)),
        coverage_cycle=int(value.get("coverage_cycle", 0)),
        iterations_since_confirmation=int(
            value.get("iterations_since_confirmation", 0)),
        confirmation_due=bool(value.get("confirmation_due", False)),
    )


def _checkpoint_from_json(value: Mapping[str, Any]) -> ConfirmedCheckpointState:
    return ConfirmedCheckpointState(
        checkpoint_id=value["checkpoint_id"],
        coverage_cycle=int(value["coverage_cycle"]),
        component_versions=ComponentVersions(**value["component_versions"]),
        confirmation_video_ids=tuple(value["confirmation_video_ids"]),
        accepted_provisional_iteration_ids=tuple(
            value["accepted_provisional_iteration_ids"]),
        confirmation_artifact_path=value["confirmation_artifact_path"],
        parent_confirmed_checkpoint_id=value.get(
            "parent_confirmed_checkpoint_id"),
        status=value.get("status", "confirmed"))


def _stage_identity(value: Any) -> Mapping[str, Any]:
    identity = getattr(value, "configuration_identity", None)
    if identity is not None:
        return as_json_dict(identity)
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


class Checkpoint3EOrchestrator:
    policy_version = "checkpoint3e_final_iteration_v2"

    def __init__(
        self, *, baseline_runner: Any, intervention_runner: Any,
        feedback_runner: Any, confirmation_evaluator: ConfirmationEvaluator,
        min_retire_support_videos: int = 2,
        require_real_models: bool = True,
        property_memory_runner: Any | None = None,
        llm_codebook_updater: Any | None = None,
        llm_router_updater: Any | None = None,
    ) -> None:
        if min_retire_support_videos < 2:
            raise ValueError("retirement requires at least two support videos")
        self.baseline_runner = baseline_runner
        self.intervention_runner = intervention_runner
        self.feedback_runner = feedback_runner
        self.confirmation_evaluator = confirmation_evaluator
        self.min_retire_support_videos = min_retire_support_videos
        self.require_real_models = require_real_models
        self.property_memory_runner = property_memory_runner
        self.llm_codebook_updater = llm_codebook_updater
        self.llm_router_updater = llm_router_updater

    @classmethod
    def with_real_confirmation(
        cls, *, baseline_runner: Any, intervention_runner: Any,
        feedback_runner: Any, confirmation_kwargs: Mapping[str, Any],
        min_retire_support_videos: int = 2,
        property_memory_runner: Any | None = None,
        llm_codebook_updater: Any | None = None,
        llm_router_updater: Any | None = None,
    ) -> "Checkpoint3EOrchestrator":
        """Construct the active path with the concrete history-aware evaluator."""
        from surrogate_rollout.optimization.confirmation_evaluator import (
            HistoryAwareDVDConfirmationEvaluator,
        )

        return cls(
            baseline_runner=baseline_runner,
            intervention_runner=intervention_runner,
            feedback_runner=feedback_runner,
            confirmation_evaluator=HistoryAwareDVDConfirmationEvaluator(
                **dict(confirmation_kwargs)),
            min_retire_support_videos=min_retire_support_videos,
            require_real_models=True,
            property_memory_runner=property_memory_runner,
            llm_codebook_updater=llm_codebook_updater,
            llm_router_updater=llm_router_updater)

    def _startup_models(self, *, iteration_id: str, output_dir: str) \
            -> Mapping[str, Any] | None:
        if not self.require_real_models:
            return None
        from surrogate_rollout.optimization.startup_models import (
            build_startup_model_manifest,
            log_startup_models,
        )

        manifest = build_startup_model_manifest(
            iteration_id=iteration_id,
            baseline_runner=self.baseline_runner,
            intervention_runner=self.intervention_runner,
            feedback_runner=self.feedback_runner,
            confirmation_evaluator=self.confirmation_evaluator)
        log_startup_models(output_dir, manifest)
        return manifest

    @staticmethod
    def _state_paths(state_dir: str, coverage_cycle: int) -> Mapping[str, str]:
        confirmed_root = os.path.join(state_dir, "confirmed")
        cycle_root = os.path.join(
            state_dir, "coverage_cycles", f"cycle_{coverage_cycle:04d}")
        return {
            "confirmed_root": confirmed_root,
            "confirmed_pointer": os.path.join(confirmed_root, "current.json"),
            "cycle_root": cycle_root,
            "active_reference": os.path.join(
                cycle_root, "active_provisional.json"),
        }

    @staticmethod
    def _load_bank(path: str, expected_version: str, expected_hash: str) \
            -> PromptBankSnapshot:
        if not os.path.exists(path):
            raise FinalIterationConflictError(f"missing bank snapshot: {path}")
        bank = prompt_bank_from_json(_read_json(path))
        if bank.bank_version != expected_version or _object_hash(bank) != expected_hash:
            raise FinalIterationConflictError("bank snapshot version/hash mismatch")
        errors = validate_prompt_bank(bank)
        if errors:
            raise FinalIterationConflictError("; ".join(errors))
        return bank

    @staticmethod
    def _load_router(
        path: str, expected_version: str, expected_hash: str,
        bank: PromptBankSnapshot,
    ) -> RouterPolicySnapshot:
        if not os.path.exists(path):
            raise FinalIterationConflictError(f"missing router snapshot: {path}")
        router = router_policy_from_json(_read_json(path))
        if router.router_version != expected_version or \
                _object_hash(router) != expected_hash:
            raise FinalIterationConflictError("router snapshot version/hash mismatch")
        errors = validate_router_against_bank(router, bank)
        if errors:
            raise FinalIterationConflictError("; ".join(errors))
        return router

    def _validate_confirmed_pointer(
        self, value: Mapping[str, Any], parent: ConfirmedCheckpointState,
    ) -> tuple[PromptBankSnapshot, RouterPolicySnapshot]:
        if value.get("schema_version") != "checkpoint3e_confirmed_pointer_v1" or \
                value.get("status") != "confirmed":
            raise FinalIterationConflictError("invalid confirmed current pointer")
        if value.get("checkpoint_id") != parent.checkpoint_id or \
                value.get("checkpoint_hash") != _object_hash(parent) or \
                int(value.get("coverage_cycle", -1)) != parent.coverage_cycle:
            raise FinalIterationConflictError(
                "confirmed pointer does not match parent checkpoint")
        bank_ref = value.get("bank") or {}
        router_ref = value.get("router") or {}
        bank = self._load_bank(
            str(bank_ref.get("path")), str(bank_ref.get("version")),
            str(bank_ref.get("hash")))
        router = self._load_router(
            str(router_ref.get("path")), str(router_ref.get("version")),
            str(router_ref.get("hash")), bank)
        if bank.bank_version != parent.component_versions.bank_version or \
                router.router_version != parent.component_versions.router_version:
            raise FinalIterationConflictError(
                "confirmed component versions do not match checkpoint")
        return bank, router

    def _initialize_or_validate_confirmed(
        self, *, state_dir: str, parent: ConfirmedCheckpointState,
        bank: PromptBankSnapshot, router: RouterPolicySnapshot,
    ) -> tuple[PromptBankSnapshot, RouterPolicySnapshot, str, str, str]:
        paths = self._state_paths(state_dir, parent.coverage_cycle)
        pointer_path = paths["confirmed_pointer"]
        if os.path.exists(pointer_path):
            pointer = _read_json(pointer_path)
            confirmed_bank, confirmed_router = self._validate_confirmed_pointer(
                pointer, parent)
            if _object_hash(bank) != _object_hash(confirmed_bank) or \
                    _object_hash(router) != _object_hash(confirmed_router):
                raise FinalIterationConflictError(
                    "explicit new-cycle policy conflicts with confirmed pointer")
            return (
                confirmed_bank, confirmed_router,
                str(pointer["bank"]["path"]), str(pointer["router"]["path"]),
                str(pointer["checkpoint_path"]),
            )

        checkpoint_root = os.path.join(
            paths["confirmed_root"], "checkpoints", parent.checkpoint_id)
        bank_path = _write_immutable(os.path.join(checkpoint_root, "bank.json"), bank)
        router_path = _write_immutable(
            os.path.join(checkpoint_root, "router.json"), router)
        checkpoint_path = _write_immutable(
            os.path.join(checkpoint_root, "checkpoint.json"), parent)
        pointer = self._confirmed_pointer_payload(
            checkpoint=parent, checkpoint_path=checkpoint_path,
            bank=bank, bank_path=bank_path,
            router=router, router_path=router_path)
        _write_pointer(pointer_path, pointer)
        return bank, router, bank_path, router_path, checkpoint_path

    @staticmethod
    def _confirmed_pointer_payload(
        *, checkpoint: ConfirmedCheckpointState, checkpoint_path: str,
        bank: PromptBankSnapshot, bank_path: str,
        router: RouterPolicySnapshot, router_path: str,
    ) -> Mapping[str, Any]:
        return {
            "schema_version": "checkpoint3e_confirmed_pointer_v1",
            "status": "confirmed",
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "checkpoint_hash": _object_hash(checkpoint),
            "coverage_cycle": checkpoint.coverage_cycle,
            "bank": {"path": os.path.abspath(bank_path),
                     "version": bank.bank_version, "hash": _object_hash(bank)},
            "router": {"path": os.path.abspath(router_path),
                       "version": router.router_version,
                       "hash": _object_hash(router)},
        }

    def _resolve_active_policy(
        self, *, state_dir: str, coverage_state: EvidenceCoverageState,
        parent_confirmed: ConfirmedCheckpointState,
        prompt_bank: PromptBankSnapshot | None,
        router_policy: RouterPolicySnapshot | None,
        confirmed_prompt_bank: PromptBankSnapshot | None,
        confirmed_router_policy: RouterPolicySnapshot | None,
    ) -> _ResolvedPolicy:
        if (prompt_bank is None) != (router_policy is None):
            raise FinalIterationConflictError(
                "prompt bank and router must be supplied together")
        paths = self._state_paths(state_dir, coverage_state.coverage_cycle)
        reference_path = paths["active_reference"]
        pointer_path = paths["confirmed_pointer"]
        if os.path.exists(reference_path):
            reference = _read_json(reference_path)
            if reference.get("status") == "active":
                if prompt_bank is not None or router_policy is not None:
                    raise FinalIterationConflictError(
                        "explicit policy cannot overwrite an active provisional cycle")
                if confirmed_prompt_bank is not None or \
                        confirmed_router_policy is not None:
                    raise FinalIterationConflictError(
                        "explicit confirmed snapshots conflict with active cycle")
                if not os.path.exists(pointer_path):
                    raise FinalIterationConflictError(
                        "active provisional reference has no confirmed pointer")
                pointer = _read_json(pointer_path)
                confirmed_bank, confirmed_router = \
                    self._validate_confirmed_pointer(pointer, parent_confirmed)
                self._validate_active_reference(
                    reference, coverage_state=coverage_state,
                    parent_confirmed=parent_confirmed,
                    confirmed_pointer=pointer)
                bank_ref = reference["active_bank"]
                router_ref = reference["active_router"]
                bank = self._load_bank(
                    bank_ref["path"], bank_ref["version"], bank_ref["hash"])
                router = self._load_router(
                    router_ref["path"], router_ref["version"],
                    router_ref["hash"], bank)
                return _ResolvedPolicy(
                    prompt_bank=bank, router_policy=router,
                    confirmed_prompt_bank=confirmed_bank,
                    confirmed_router_policy=confirmed_router,
                    confirmed_bank_path=pointer["bank"]["path"],
                    confirmed_router_path=pointer["router"]["path"],
                    confirmed_checkpoint_path=pointer["checkpoint_path"],
                    provisional_lineage=tuple(reference["provisional_lineage"]),
                    active_reference_path=reference_path,
                    confirmed_pointer_path=pointer_path)
            if reference.get("status") not in ("accepted", "rejected"):
                raise FinalIterationConflictError(
                    "invalid active-provisional reference status")

        if prompt_bank is None or router_policy is None:
            raise FinalIterationConflictError(
                "a new coverage cycle requires explicit confirmed bank/router inputs")
        confirmed_prompt_bank = confirmed_prompt_bank or prompt_bank
        confirmed_router_policy = confirmed_router_policy or router_policy
        if _object_hash(prompt_bank) != _object_hash(confirmed_prompt_bank) or \
                _object_hash(router_policy) != _object_hash(confirmed_router_policy):
            raise FinalIterationConflictError(
                "new cycle must start from the confirmed policy pair")
        confirmed_bank, confirmed_router, bank_path, router_path, checkpoint_path = \
            self._initialize_or_validate_confirmed(
                state_dir=state_dir, parent=parent_confirmed,
                bank=confirmed_prompt_bank, router=confirmed_router_policy)
        return _ResolvedPolicy(
            prompt_bank=confirmed_bank, router_policy=confirmed_router,
            confirmed_prompt_bank=confirmed_bank,
            confirmed_router_policy=confirmed_router,
            confirmed_bank_path=bank_path, confirmed_router_path=router_path,
            confirmed_checkpoint_path=checkpoint_path,
            provisional_lineage=(), active_reference_path=reference_path,
            confirmed_pointer_path=pointer_path)

    @staticmethod
    def _validate_active_reference(
        value: Mapping[str, Any], *, coverage_state: EvidenceCoverageState,
        parent_confirmed: ConfirmedCheckpointState,
        confirmed_pointer: Mapping[str, Any],
    ) -> None:
        if value.get("schema_version") != \
                "checkpoint3e_active_provisional_reference_v1" or \
                value.get("status") != "active":
            raise FinalIterationConflictError("invalid active provisional reference")
        lineage = tuple(value.get("provisional_lineage") or ())
        if not lineage or len(lineage) != len(set(lineage)) or \
                value.get("latest_provisional_iteration_id") != lineage[-1]:
            raise FinalIterationConflictError(
                "active provisional lineage is empty, duplicated, or branched")
        parent = value.get("parent_confirmed") or {}
        expected_parent = {
            "checkpoint_id": parent_confirmed.checkpoint_id,
            "checkpoint_hash": _object_hash(parent_confirmed),
            "bank_hash": confirmed_pointer["bank"]["hash"],
            "router_hash": confirmed_pointer["router"]["hash"],
        }
        if parent != expected_parent:
            raise FinalIterationConflictError(
                "active provisional parent checkpoint/hash mismatch")
        if int(value.get("coverage_cycle_id", -1)) != coverage_state.coverage_cycle or \
                value.get("coverage_state_hash") != _coverage_hash(coverage_state):
            raise FinalIterationConflictError(
                "stale active provisional coverage reference")
        state_ref = value.get("latest_provisional_state") or {}
        state_path = str(state_ref.get("path"))
        if not os.path.exists(state_path) or \
                state_ref.get("hash") != _file_hash(state_path):
            raise FinalIterationConflictError(
                "latest provisional state path/hash mismatch")
        state = _read_json(state_path)
        versions = state.get("provisional_versions") or {}
        if state.get("iteration_id") != lineage[-1] or \
                int(state.get("coverage_cycle", -1)) != coverage_state.coverage_cycle or \
                state.get("parent_confirmed_checkpoint_id") != \
                parent_confirmed.checkpoint_id or \
                tuple(state.get("accumulated_provisional_iteration_ids") or ()) != \
                lineage or \
                state.get("coverage_state_after_hash") != \
                value.get("coverage_state_hash") or \
                versions.get("bank_version") != \
                (value.get("active_bank") or {}).get("version") or \
                versions.get("router_version") != \
                (value.get("active_router") or {}).get("version"):
            raise FinalIterationConflictError(
                "active reference conflicts with latest provisional state")

    def _resume_completed(
        self, *, manifest_target: str, iteration_id: str,
        roles: TrainRoleAssignment, coverage_state: EvidenceCoverageState,
        parent_confirmed: ConfirmedCheckpointState,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        execution_identity: Mapping[str, Any],
        prompt_bank: PromptBankSnapshot | None,
        router_policy: RouterPolicySnapshot | None,
        state_dir: str,
        startup_models: Mapping[str, Any] | None,
    ) -> FinalIterationResult:
        manifest = _read_json(manifest_target)
        if manifest.get("status") != "completed" or \
                manifest.get("iteration_id") != iteration_id:
            raise FinalIterationConflictError(
                "iteration manifest is incomplete or belongs to another iteration")
        self._validate_completed_manifest(manifest)
        resume_contract = {
            "roles": roles, "coverage_state": coverage_state,
            "parent_confirmed": parent_confirmed,
            "scaffold": scaffold_policy, "contract": scaffold_contract,
            "execution_identity": execution_identity,
            "startup_models": startup_models,
            "stages": {
                "baseline": _stage_identity(self.baseline_runner),
                "intervention": _stage_identity(self.intervention_runner),
                "feedback": _stage_identity(self.feedback_runner),
                "confirmation": _stage_identity(self.confirmation_evaluator),
            },
            "min_retire_support_videos": self.min_retire_support_videos,
            "state_dir": state_dir,
        }
        if manifest.get("resume_contract_hash") != \
                sha256_text(dumps_canonical(resume_contract)):
            raise FinalIterationConflictError("completed resume contract mismatch")
        explicit_hashes = manifest.get("input_policy_hashes") or {}
        if prompt_bank is not None and (
                router_policy is None or
                _object_hash(prompt_bank) != explicit_hashes.get("bank") or
                _object_hash(router_policy) != explicit_hashes.get("router")):
            raise FinalIterationConflictError(
                "explicit policy conflicts with completed iteration input")
        reference = _read_json(manifest["active_provisional_reference_path"])
        if reference.get("latest_provisional_iteration_id") != iteration_id or \
                tuple(reference.get("provisional_lineage") or ()) != \
                tuple(manifest["provisional_lineage"]):
            raise FinalIterationConflictError(
                "completed iteration no longer matches cycle-local reference")
        pointer = _read_json(manifest["confirmed_pointer_path"])
        if manifest["iteration_status"] == "provisional":
            self._validate_active_reference(
                reference,
                coverage_state=_coverage_from_json(manifest["next_coverage_state"]),
                parent_confirmed=parent_confirmed,
                confirmed_pointer=pointer)
        elif reference.get("status") not in ("accepted", "rejected"):
            raise FinalIterationConflictError(
                "confirmed iteration requires a closed provisional reference")
        return self._result_from_manifest(manifest_target, manifest, resumed=True)

    def run(
        self, *, iteration_id: str, roles: TrainRoleAssignment,
        coverage_state: EvidenceCoverageState,
        parent_confirmed: ConfirmedCheckpointState,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        output_dir: str,
        state_dir: str,
        execution_identity: Mapping[str, Any],
        baseline_kwargs: Mapping[str, Any],
        intervention_kwargs: Mapping[str, Any],
        prompt_bank: PromptBankSnapshot | None = None,
        router_policy: RouterPolicySnapshot | None = None,
        confirmed_prompt_bank: PromptBankSnapshot | None = None,
        confirmed_router_policy: RouterPolicySnapshot | None = None,
        pending_provisional_iteration_ids: tuple[str, ...] = (),
    ) -> FinalIterationResult:
        output_dir = os.path.abspath(output_dir)
        state_dir = os.path.abspath(state_dir)
        startup_models = self._startup_models(
            iteration_id=iteration_id, output_dir=output_dir)
        manifest_target = os.path.join(output_dir, "manifest.json")
        if os.path.exists(manifest_target):
            return self._resume_completed(
                manifest_target=manifest_target, iteration_id=iteration_id,
                roles=roles, coverage_state=coverage_state,
                parent_confirmed=parent_confirmed,
                scaffold_policy=scaffold_policy,
                scaffold_contract=scaffold_contract,
                execution_identity=execution_identity,
                prompt_bank=prompt_bank, router_policy=router_policy,
                state_dir=state_dir, startup_models=startup_models)

        resolved = self._resolve_active_policy(
            state_dir=state_dir, coverage_state=coverage_state,
            parent_confirmed=parent_confirmed,
            prompt_bank=prompt_bank, router_policy=router_policy,
            confirmed_prompt_bank=confirmed_prompt_bank,
            confirmed_router_policy=confirmed_router_policy)
        prompt_bank = resolved.prompt_bank
        router_policy = resolved.router_policy
        confirmed_prompt_bank = resolved.confirmed_prompt_bank
        confirmed_router_policy = resolved.confirmed_router_policy
        if pending_provisional_iteration_ids and \
                pending_provisional_iteration_ids != resolved.provisional_lineage:
            raise FinalIterationConflictError(
                "caller provisional lineage conflicts with active reference")
        pending_provisional_iteration_ids = resolved.provisional_lineage
        if parent_confirmed.component_versions.scaffold_version != \
                scaffold_policy.scaffold_version or \
                parent_confirmed.component_versions.contract_version != \
                scaffold_contract.contract_version:
            raise FinalIterationError("fixed scaffold/contract lineage mismatch")
        selected_ids, expected_coverage = select_evidence_batch(coverage_state)
        evidence_ids = {item.video_id for item in roles.evidence_videos}
        forbidden = (set(selected_ids) & (
            set(roles.validation_video_ids) | set(roles.test_video_ids) |
            {item.video_id for item in roles.confirmation_videos}))
        if set(selected_ids) - evidence_ids or forbidden:
            raise FinalIterationError("selected videos are outside the evidence pool")
        identity = {
            "policy_version": self.policy_version,
            "iteration_id": iteration_id,
            "roles": roles,
            "coverage_state": coverage_state,
            "parent_confirmed": parent_confirmed,
            "components": {
                "bank": prompt_bank, "router": router_policy,
                "confirmed_bank": confirmed_prompt_bank,
                "confirmed_router": confirmed_router_policy,
                "scaffold": scaffold_policy, "contract": scaffold_contract,
            },
            "execution_identity": execution_identity,
            "startup_models": startup_models,
            "stages": {
                "baseline": _stage_identity(self.baseline_runner),
                "intervention": _stage_identity(self.intervention_runner),
                "feedback": _stage_identity(self.feedback_runner),
                "confirmation": _stage_identity(self.confirmation_evaluator),
            },
            "min_retire_support_videos": self.min_retire_support_videos,
            "optimize_scaffold": False,
            "pending_provisional_iteration_ids": pending_provisional_iteration_ids,
            "state_dir": state_dir,
        }
        fingerprint = sha256_text(dumps_canonical(identity))

        baseline_result = self.baseline_runner.run(
            roles=roles, coverage_state=coverage_state,
            prompt_bank=prompt_bank, router_policy=router_policy,
            scaffold_policy=scaffold_policy, scaffold_contract=scaffold_contract,
            output_dir=os.path.join(output_dir, "baseline_stage"),
            parent_confirmed_checkpoint_id=parent_confirmed.checkpoint_id,
            **dict(baseline_kwargs))
        if tuple(baseline_result.selected_video_ids) != selected_ids or \
                baseline_result.next_coverage_state != expected_coverage:
            raise FinalIterationError(
                "baseline stage selection or coverage result is inconsistent")

        video_manifests = []
        for path in baseline_result.video_manifest_paths:
            value = _read_json(path)
            value["manifest_path"] = os.path.abspath(path)
            video_manifests.append(value)
        video_manifests = tuple(video_manifests)
        if {item["video_id"] for item in video_manifests} != set(selected_ids):
            raise FinalIterationError("baseline video manifests do not match selection")
        interventions = []
        feedback_results = []
        all_proposals = []
        for video_manifest in sorted(video_manifests, key=lambda item: item["video_id"]):
            proposal_record = _read_json(video_manifest["property_proposal_path"])
            proposals = tuple(_proposal_from_json(item)
                              for item in proposal_record.get("proposals") or ())
            all_proposals.extend(proposals)
            retrieval_paths = tuple(video_manifest.get("property_retrieval_paths") or ())
            if len(retrieval_paths) != len(proposals):
                raise FinalIterationError(
                    "every proposal must have one retrieval artifact")
            intervention = self.intervention_runner.run(
                iteration_id=iteration_id,
                baseline_video_manifest_path=video_manifest["manifest_path"],
                proposals=proposals, retrieval_artifact_paths=retrieval_paths,
                prompt_bank=prompt_bank, router_policy=router_policy,
                scaffold_policy=scaffold_policy,
                scaffold_contract=scaffold_contract,
                output_dir=os.path.join(
                    output_dir, "interventions", video_manifest["video_id"]),
                **dict(intervention_kwargs))
            interventions.append(intervention)
            feedback = self.feedback_runner.run(
                iteration_id=iteration_id,
                baseline_video_manifest_path=video_manifest["manifest_path"],
                proposals=proposals, retrieval_artifact_paths=retrieval_paths,
                intervention_manifest_path=intervention.manifest_path,
                prompt_bank=prompt_bank,
                output_dir=os.path.join(
                    output_dir, "feedback", video_manifest["video_id"]),
            )
            feedback_results.append(feedback)

        plan = self.aggregate_updates(
            prompt_bank=prompt_bank,
            router_policy=router_policy,
            feedback_manifest_paths=tuple(
                item.manifest_path for item in feedback_results))
        update_dir = os.path.join(output_dir, "next_state")
        plan_path = _write_immutable(
            os.path.join(update_dir, "update_plan.json"), plan)
        candidate_bank, candidate_router = self.apply_update_plan(
            prompt_bank=prompt_bank, router_policy=router_policy, plan=plan)
        bank_path = _write_immutable(
            os.path.join(update_dir, "provisional_bank.json"), candidate_bank)
        router_path = _write_immutable(
            os.path.join(update_dir, "provisional_router.json"), candidate_router)
        _write_immutable(os.path.join(update_dir, "fixed_scaffold_reference.json"), {
            "optimize_scaffold": False,
            "scaffold_version": scaffold_policy.scaffold_version,
            "contract_version": scaffold_contract.contract_version,
            "scaffold_hash": sha256_json(as_json_dict(scaffold_policy)),
            "contract_hash": sha256_json(as_json_dict(scaffold_contract)),
        })
        provisional = ProvisionalIterationState(
            iteration_id=iteration_id,
            coverage_cycle=coverage_state.coverage_cycle,
            selected_video_ids=selected_ids,
            parent_confirmed_checkpoint_id=parent_confirmed.checkpoint_id,
            input_versions=ComponentVersions(
                prompt_bank.bank_version, router_policy.router_version,
                scaffold_policy.scaffold_version,
                scaffold_contract.contract_version),
            provisional_versions=ComponentVersions(
                candidate_bank.bank_version, candidate_router.router_version,
                scaffold_policy.scaffold_version,
                scaffold_contract.contract_version),
            baseline_manifest_path=baseline_result.manifest_path,
            property_proposal_paths=tuple(
                item["property_proposal_path"] for item in video_manifests),
            property_retrieval_paths=tuple(
                path for item in video_manifests
                for path in item.get("property_retrieval_paths") or ()),
            accumulated_provisional_iteration_ids=(
                *pending_provisional_iteration_ids, iteration_id),
            coverage_state_before_hash=sha256_text(dumps_canonical(coverage_state)),
            coverage_state_after_hash=sha256_text(
                dumps_canonical(expected_coverage)))
        provisional_path = _write_immutable(
            os.path.join(update_dir, "provisional_state.json"), provisional)

        confirmation_path = None
        checkpoint_path = None
        active_bank_path = bank_path
        active_router_path = router_path
        next_coverage = expected_coverage
        final_status = "provisional"
        if expected_coverage.confirmation_due:
            decision, checkpoint_path, active_bank_path, active_router_path = \
                self._run_confirmation(
                    parent_confirmed=parent_confirmed, roles=roles,
                    incumbent_bank=confirmed_prompt_bank,
                    incumbent_router=confirmed_router_policy,
                    candidate_bank=candidate_bank,
                    candidate_router=candidate_router,
                    scaffold_policy=scaffold_policy,
                    scaffold_contract=scaffold_contract,
                    iteration_id=iteration_id,
                    coverage_cycle=coverage_state.coverage_cycle,
                    pending_provisional_iteration_ids=(
                        *pending_provisional_iteration_ids, iteration_id),
                    output_dir=os.path.join(output_dir, "confirmation"),
                    provisional_bank_path=bank_path,
                    provisional_router_path=router_path)
            confirmation_path = decision.artifact_path
            next_coverage = reset_after_confirmation(expected_coverage)
            final_status = "confirmed" if decision.accepted else "rolled_back"
            if not decision.accepted:
                checkpoint_path = resolved.confirmed_checkpoint_path
                active_bank_path = resolved.confirmed_bank_path
                active_router_path = resolved.confirmed_router_path

        lineage = (*pending_provisional_iteration_ids, iteration_id)
        resume_contract = {
            "roles": roles, "coverage_state": coverage_state,
            "parent_confirmed": parent_confirmed,
            "scaffold": scaffold_policy, "contract": scaffold_contract,
            "execution_identity": execution_identity,
            "startup_models": startup_models,
            "stages": {
                "baseline": _stage_identity(self.baseline_runner),
                "intervention": _stage_identity(self.intervention_runner),
                "feedback": _stage_identity(self.feedback_runner),
                "confirmation": _stage_identity(self.confirmation_evaluator),
            },
            "min_retire_support_videos": self.min_retire_support_videos,
            "state_dir": state_dir,
        }

        manifest = {
            "schema_version": "checkpoint3e_iteration_manifest_v1",
            "status": "completed", "iteration_status": final_status,
            "iteration_id": iteration_id, "input_fingerprint": fingerprint,
            "selected_video_ids": selected_ids,
            "baseline_manifest_path": baseline_result.manifest_path,
            "intervention_manifest_paths": tuple(
                item.manifest_path for item in interventions),
            "feedback_manifest_paths": tuple(
                item.manifest_path for item in feedback_results),
            "update_plan_path": plan_path,
            "provisional_bank_path": bank_path,
            "provisional_router_path": router_path,
            "provisional_state_path": provisional_path,
            "confirmation_decision_path": confirmation_path,
            "confirmed_checkpoint_path": checkpoint_path,
            "active_bank_path": active_bank_path,
            "active_router_path": active_router_path,
            "active_provisional_reference_path":
                resolved.active_reference_path,
            "confirmed_pointer_path": resolved.confirmed_pointer_path,
            "provisional_lineage": lineage,
            "input_policy_versions": {
                "bank": prompt_bank.bank_version,
                "router": router_policy.router_version,
            },
            "input_policy_hashes": {
                "bank": _object_hash(prompt_bank),
                "router": _object_hash(router_policy),
            },
            "resume_contract_hash": sha256_text(
                dumps_canonical(resume_contract)),
            "artifact_hashes": {
                "update_plan": _file_hash(plan_path),
                "provisional_bank": _file_hash(bank_path),
                "provisional_router": _file_hash(router_path),
                "provisional_state": _file_hash(provisional_path),
            },
            "next_coverage_state": next_coverage,
            "fixed_scaffold_version": scaffold_policy.scaffold_version,
            "fixed_contract_version": scaffold_contract.contract_version,
            "optimize_scaffold": False,
            "canonical_component_state_mutated": False,
        }
        manifest_path = _write_immutable(manifest_target, manifest)
        reference = self._provisional_reference_payload(
            status=("active" if final_status == "provisional" else
                    ("accepted" if final_status == "confirmed" else "rejected")),
            coverage_cycle=coverage_state.coverage_cycle,
            parent_confirmed=parent_confirmed,
            confirmed_bank=confirmed_prompt_bank,
            confirmed_router=confirmed_router_policy,
            iteration_id=iteration_id, lineage=lineage,
            active_bank=candidate_bank, active_bank_path=bank_path,
            active_router=candidate_router, active_router_path=router_path,
            provisional_state_path=provisional_path,
            coverage_state=(expected_coverage if final_status == "provisional"
                            else next_coverage),
            confirmation_decision_path=confirmation_path)
        _write_pointer(resolved.active_reference_path, reference)
        if final_status != "provisional":
            checkpoint = _checkpoint_from_json(_read_json(checkpoint_path))
            confirmed_bank = (candidate_bank if final_status == "confirmed"
                              else confirmed_prompt_bank)
            confirmed_router = (candidate_router if final_status == "confirmed"
                                else confirmed_router_policy)
            confirmed_bank_path = (bank_path if final_status == "confirmed"
                                   else resolved.confirmed_bank_path)
            confirmed_router_path = (router_path if final_status == "confirmed"
                                     else resolved.confirmed_router_path)
            _write_pointer(
                resolved.confirmed_pointer_path,
                self._confirmed_pointer_payload(
                    checkpoint=checkpoint, checkpoint_path=checkpoint_path,
                    bank=confirmed_bank, bank_path=confirmed_bank_path,
                    router=confirmed_router, router_path=confirmed_router_path))
        return self._result_from_manifest(manifest_path, _read_json(manifest_path))

    @staticmethod
    def _provisional_reference_payload(
        *, status: str, coverage_cycle: int,
        parent_confirmed: ConfirmedCheckpointState,
        confirmed_bank: PromptBankSnapshot,
        confirmed_router: RouterPolicySnapshot,
        iteration_id: str, lineage: tuple[str, ...],
        active_bank: PromptBankSnapshot, active_bank_path: str,
        active_router: RouterPolicySnapshot, active_router_path: str,
        provisional_state_path: str,
        coverage_state: EvidenceCoverageState,
        confirmation_decision_path: str | None,
    ) -> Mapping[str, Any]:
        if status not in ("active", "accepted", "rejected"):
            raise FinalIterationError("invalid provisional reference status")
        return {
            "schema_version": "checkpoint3e_active_provisional_reference_v1",
            "status": status,
            "coverage_cycle_id": coverage_cycle,
            "parent_confirmed": {
                "checkpoint_id": parent_confirmed.checkpoint_id,
                "checkpoint_hash": _object_hash(parent_confirmed),
                "bank_hash": _object_hash(confirmed_bank),
                "router_hash": _object_hash(confirmed_router),
            },
            "latest_provisional_iteration_id": iteration_id,
            "active_bank": {
                "path": os.path.abspath(active_bank_path),
                "version": active_bank.bank_version,
                "hash": _object_hash(active_bank),
            },
            "active_router": {
                "path": os.path.abspath(active_router_path),
                "version": active_router.router_version,
                "hash": _object_hash(active_router),
            },
            "latest_provisional_state": {
                "path": os.path.abspath(provisional_state_path),
                "hash": _file_hash(provisional_state_path),
            },
            "provisional_lineage": lineage,
            "coverage_state_hash": _coverage_hash(coverage_state),
            "confirmation_decision_path": confirmation_decision_path,
        }

    def aggregate_updates(
        self, *, prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
        feedback_manifest_paths: tuple[str, ...],
        expected_evidence_video_count: int = 3,
    ) -> IterationUpdatePlan:
        if expected_evidence_video_count < 1:
            raise ValueError("expected evidence video count must be positive")
        properties = []
        accepted_rows = []
        rejected_count = 0
        feedback_video_ids = set()
        for manifest_path in feedback_manifest_paths:
            manifest = _read_json(manifest_path)
            if manifest.get("status") != "completed":
                raise FinalIterationError("feedback manifest is not completed")
            feedback_video_ids.add(str(manifest["source_video_id"]))
            for item in manifest.get("properties") or ():
                result = _read_json(item["result_path"])
                properties.append(result)
                for qa in result.get("qa_feedback") or ():
                    if qa.get("status") == "accepted":
                        accepted_rows.append({
                            **qa,
                            "candidate_property_id": result["candidate_property_id"],
                            "property_text": result["property_text"],
                            "source_video_id": result["source_video_id"],
                        })
                    elif qa.get("status") == "rejected":
                        rejected_count += 1
        if len(feedback_video_ids) != expected_evidence_video_count:
            raise FinalIterationError(
                "iteration aggregation received an unexpected evidence-video count")
        active = {entry.prompt_id: entry for entry in prompt_bank.entries
                  if entry.status == "active"}
        for row in accepted_rows:
            if row.get("transition") not in (
                    "wrong_to_correct", "correct_to_wrong"):
                raise FinalIterationError(
                    "accepted update feedback must be a correctness flip")
            if row.get("recommendation") not in (
                    "add", "revise", "merge", "retire", "router_positive",
                    "router_negative", "no_op"):
                raise FinalIterationError(
                    "accepted update feedback has an invalid recommendation")
            if not row.get("feedback_id") or not row.get("evidence"):
                raise FinalIterationError(
                    "accepted update feedback lacks evidence identity")
            unknown_coverage = set(row.get("covered_by_property_ids") or ()) - \
                set(active)
            if unknown_coverage:
                raise FinalIterationError(
                    "accepted feedback cites unknown codebook coverage: "
                    f"{sorted(unknown_coverage)}")
        active_by_text = {_normalized_text(entry.prompt_text): entry.prompt_id
                          for entry in active.values()}
        by_candidate = {
            (item["source_video_id"], item["candidate_property_id"]): item
            for item in properties
        }
        if len(by_candidate) != len(properties):
            raise FinalIterationError("duplicate candidate lineage in feedback")
        rows_by_candidate = {
            key: tuple(row for row in accepted_rows
                       if (row["source_video_id"],
                           row["candidate_property_id"]) == key)
            for key in by_candidate
        }
        conflicts = []
        decisions = []
        candidate_targets: dict[str, tuple[str, ...]] = {}

        retire_support: dict[str, set[str]] = {}
        for row in accepted_rows:
            if row.get("recommendation") != "retire" or \
                    row.get("transition") != "correct_to_wrong":
                continue
            for target in row.get("covered_by_property_ids") or ():
                retire_support.setdefault(target, set()).add(row["source_video_id"])

        text_groups: dict[str, list[tuple[str, str]]] = {}
        for candidate_key, item in by_candidate.items():
            text_groups.setdefault(_normalized_text(item["property_text"]), []).append(
                candidate_key)
        handled = set()
        retire_claimed: set[str] = set()
        for text_key, candidate_ids in sorted(text_groups.items()):
            group = tuple(sorted(candidate_ids))
            if any(item in handled for item in group):
                continue
            handled.update(group)
            group_rows = tuple(row for candidate_key in group
                               for row in rows_by_candidate[candidate_key])
            feedback_ids = tuple(sorted({str(row["feedback_id"])
                                         for row in group_rows}))
            videos = tuple(sorted({str(row["source_video_id"])
                                   for row in group_rows}))
            positive = sum(row.get("transition") == "wrong_to_correct"
                           for row in group_rows)
            negative = sum(row.get("transition") == "correct_to_wrong"
                           for row in group_rows)
            covered = tuple(entry.prompt_id for entry in prompt_bank.entries
                            if any(entry.prompt_id in (
                                row.get("covered_by_property_ids") or ())
                                for row in group_rows))
            exact = active_by_text.get(text_key)
            if exact and exact not in covered:
                covered = (*covered, exact)
            recommendations = {row.get("recommendation") for row in group_rows}
            action = "no_op"
            targets = covered
            result_id = None
            rationale = "insufficient accepted evidence"
            property_text = by_candidate[group[0]]["property_text"]
            qualified_retire = tuple(target for target in covered
                                     if len(retire_support.get(target, ())) >=
                                     self.min_retire_support_videos
                                     and target not in retire_claimed)
            if qualified_retire:
                action = "retire"
                targets = qualified_retire
                retire_claimed.update(qualified_retire)
                retire_rows = tuple(
                    row for row in accepted_rows
                    if row.get("recommendation") == "retire"
                    and row.get("transition") == "correct_to_wrong"
                    and set(row.get("covered_by_property_ids") or ())
                    & set(qualified_retire))
                feedback_ids = tuple(sorted({str(row["feedback_id"])
                                             for row in retire_rows}))
                videos = tuple(sorted({str(row["source_video_id"])
                                       for row in retire_rows}))
                negative = len(retire_rows)
                rationale = "repeated harmful evidence across distinct videos"
            elif any(len(retire_support.get(target, ())) >=
                     self.min_retire_support_videos for target in covered):
                rationale = "retirement evidence consolidated into another decision"
            elif "retire" in recommendations:
                rationale = "retirement deferred: insufficient distinct-video support"
            elif "merge" in recommendations and len(covered) >= 2 and positive > 0:
                action = "merge"
                result_id = "pe_merged_" + sha256_text(text_key)[:12]
                rationale = "accepted evidence supports merging covered entries"
            elif "revise" in recommendations and len(covered) == 1:
                action = "revise"
                result_id = covered[0]
                rationale = "accepted evidence supports revising one covered entry"
            elif covered:
                rationale = "active codebook coverage prevents duplicate addition"
            elif positive > negative and "add" in recommendations:
                action = "add"
                result_id = "pe_learned_" + sha256_text(text_key)[:12]
                rationale = "accepted positive evidence supports a new reusable entry"
            elif len(group) > 1:
                conflicts.append(
                    f"duplicate candidate group {group} retained as no_op")
            if action in ("revise", "merge", "retire") and any(
                    target not in active for target in targets):
                action, result_id = "no_op", None
                rationale = "mutation target is not active"
            if action == "add" and (result_id in active or text_key in active_by_text):
                action, result_id = "no_op", None
                rationale = "active codebook already covers exact property text"
            decision_identity = {
                "action": action, "candidates": group, "targets": targets,
                "result": result_id, "feedback": feedback_ids,
                "positive": positive, "negative": negative,
                "rationale": rationale,
            }
            decisions.append(CodebookDecision(
                decision_id="codebook_decision_" + sha256_json(decision_identity)[:20],
                action=action,
                candidate_property_ids=tuple(item[1] for item in group),
                candidate_lineage=tuple({
                    "source_video_id": item[0], "candidate_property_id": item[1],
                } for item in group),
                target_property_ids=targets,
                resulting_property_id=result_id,
                property_text=(property_text if action != "no_op" else None),
                supporting_feedback_ids=feedback_ids,
                supporting_video_ids=videos,
                positive_count=positive, negative_count=negative,
                rationale=rationale))
            mapped = ((result_id,) if result_id else targets)
            for candidate_key in group:
                candidate_targets[candidate_key] = tuple(mapped)

        target_mutations: dict[str, list[CodebookDecision]] = {}
        for decision in decisions:
            if decision.action in ("revise", "merge", "retire"):
                for target in decision.target_property_ids:
                    target_mutations.setdefault(target, []).append(decision)
        conflicting_ids = {decision.decision_id
                           for values in target_mutations.values() if len(values) > 1
                           for decision in values}
        if conflicting_ids:
            resolved = []
            for decision in decisions:
                if decision.decision_id not in conflicting_ids:
                    resolved.append(decision)
                    continue
                conflicts.append(
                    f"conflicting codebook mutations forced no_op: {decision.decision_id}")
                resolved.append(replace(
                    decision, action="no_op", resulting_property_id=None,
                    property_text=None,
                    rationale="conflicting mutations on the same active entry"))
                for lineage in decision.candidate_lineage:
                    candidate_targets[(
                        lineage["source_video_id"],
                        lineage["candidate_property_id"])] = \
                        decision.target_property_ids
            decisions = resolved

        supervision = []
        for row in sorted(accepted_rows, key=lambda item: (
                item["source_video_id"], item["question_id"],
                item["candidate_property_id"], item["feedback_id"])):
            polarity = ("positive" if row["transition"] == "wrong_to_correct"
                        else "negative")
            retired_ids = {
                target for decision in decisions if decision.action == "retire"
                for target in decision.target_property_ids
            }
            for property_id in candidate_targets.get((
                    row["source_video_id"], row["candidate_property_id"]), ()):
                if property_id in retired_ids:
                    continue
                context_hash = sha256_json({
                    "evidence": row["evidence"],
                    "segments": row.get("s_feedback") or (),
                    "polarity": polarity,
                })
                identity = {
                    "property_id": property_id, "polarity": polarity,
                    "feedback_id": row["feedback_id"],
                    "context_hash": context_hash,
                }
                supervision.append(RouterSupervisionUpdate(
                    supervision_id="router_supervision_" + sha256_json(identity)[:20],
                    property_id=property_id, polarity=polarity,
                    evidence=row["evidence"], context_hash=context_hash,
                    source_feedback_id=row["feedback_id"],
                    source_video_id=row["source_video_id"],
                    source_question_id=row["question_id"],
                    attributed_segment_ids=tuple(row.get("s_feedback") or ())))
        accepted_ids = tuple(sorted({str(row["feedback_id"])
                                     for row in accepted_rows}))
        plan_identity = {
            "bank": prompt_bank.bank_version,
            "decisions": decisions, "supervision": supervision,
            "accepted": accepted_ids, "rejected_count": rejected_count,
            "conflicts": conflicts,
        }
        return IterationUpdatePlan(
            plan_id="iteration_update_" + sha256_text(
                dumps_canonical(plan_identity))[:24],
            input_bank_version=prompt_bank.bank_version,
            input_router_version=router_policy.router_version,
            optimize_scaffold=False,
            codebook_decisions=tuple(decisions),
            router_supervision=tuple(supervision),
            accepted_feedback_ids=accepted_ids,
            rejected_feedback_count=rejected_count,
            conflict_resolutions=tuple(conflicts))

    def run_memory_conditioned_codebook_checkpoint(
        self, *, iteration_id: str, iteration_ordinal: int,
        prompt_bank: PromptBankSnapshot,
        baseline_video_manifest_paths: tuple[str, ...],
        intervention_manifest_paths: tuple[str, ...],
        feedback_manifest_paths: tuple[str, ...],
        output_dir: str, state_dir: str,
        stage_logger: StageLogger | None = None,
    ) -> Mapping[str, Any]:
        """Run the post-feedback memory/codebook checkpoint without router commit.

        This boundary is intentionally separate from ``run``: the existing
        production iteration commits a bank/router pair, while this checkpoint
        may only materialize a candidate codebook for the deferred router stage.
        """
        if self.property_memory_runner is None or self.llm_codebook_updater is None:
            raise FinalIterationError(
                "memory-conditioned codebook checkpoint requires both stage runners")
        output_dir = os.path.abspath(output_dir)
        state_dir = os.path.abspath(state_dir)
        manifest_target = os.path.join(output_dir, "manifest.json")
        pointer_path = os.path.join(state_dir, "property_memory", "current.json")
        bank_hash = _object_hash(prompt_bank)
        input_contract = {
            "iteration_id": iteration_id, "iteration_ordinal": iteration_ordinal,
            "input_bank_version": prompt_bank.bank_version,
            "input_bank_hash": bank_hash,
            "baseline_manifests": tuple({"path": os.path.abspath(path),
                                          "hash": _file_hash(path)}
                                         for path in baseline_video_manifest_paths),
            "intervention_manifests": tuple({"path": os.path.abspath(path),
                                              "hash": _file_hash(path)}
                                             for path in intervention_manifest_paths),
            "feedback_manifests": tuple({"path": os.path.abspath(path),
                                          "hash": _file_hash(path)}
                                         for path in feedback_manifest_paths),
            "stages": {
                "property_memory": _stage_identity(self.property_memory_runner),
                "llm_codebook_updater": _stage_identity(self.llm_codebook_updater),
            },
            "state_dir": state_dir,
        }
        input_contract_hash = sha256_json(input_contract)
        if os.path.exists(manifest_target):
            manifest = _read_json(manifest_target)
            if manifest.get("schema_version") != \
                    "memory_conditioned_codebook_iteration_v1" or \
                    manifest.get("status") != "completed" or \
                    manifest.get("iteration_id") != iteration_id or \
                    manifest.get("input_bank_hash") != bank_hash or \
                    manifest.get("input_contract_hash") != input_contract_hash:
                raise FinalIterationConflictError(
                    "incompatible completed memory-codebook checkpoint")
            artifacts = manifest.get("artifacts") or {}
            hashes = manifest.get("artifact_hashes") or {}
            if set(artifacts) != set(hashes) or any(
                    not os.path.exists(path) or _file_hash(path) != hashes[name]
                    for name, path in artifacts.items()):
                raise FinalIterationConflictError(
                    "memory-codebook checkpoint artifact mismatch")
            memory_manifest = _read_json(artifacts["property_memory_manifest"])
            for ref in memory_manifest.get("raw_artifact_refs") or ():
                if not os.path.exists(str(ref.get("path") or "")) or \
                        ref.get("sha256") != _file_hash(ref["path"]):
                    raise FinalIterationConflictError(
                        "memory-codebook raw artifact closure mismatch")
            if not os.path.exists(pointer_path):
                raise FinalIterationConflictError("property-memory lineage pointer is missing")
            pointer = _read_json(pointer_path)
            if pointer.get("schema_version") != "property_memory_lineage_pointer_v1" or \
                    pointer.get("latest_iteration_id") != iteration_id or \
                    pointer.get("snapshot_hash") != _file_hash(
                        pointer.get("snapshot_path", "")):
                raise FinalIterationConflictError(
                    "property-memory lineage pointer conflicts with completed checkpoint")
            emit_stage(
                stage_logger, event="RESUME", stage="memory_update",
                message="property_memory_v1 update artifact 재사용",
                snapshot_path=artifacts["property_memory_snapshot"])
            emit_stage(
                stage_logger, event="RESUME", stage="codebook_update",
                message="LLM codebook update artifact 재사용",
                updater_manifest=artifacts["llm_codebook_updater_manifest"])
            return {**manifest, "resumed": True}
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            raise FinalIterationConflictError(
                "incomplete memory-codebook checkpoint cannot be resumed")

        parent_memory_path = None
        if os.path.exists(pointer_path):
            pointer = _read_json(pointer_path)
            if pointer.get("schema_version") != "property_memory_lineage_pointer_v1" or \
                    pointer.get("active_bank_version") != prompt_bank.bank_version or \
                    pointer.get("active_bank_hash") != bank_hash:
                raise FinalIterationConflictError(
                    "property-memory parent does not match active policy lineage")
            parent_memory_path = str(pointer.get("snapshot_path") or "")
            if not os.path.exists(parent_memory_path) or \
                    pointer.get("snapshot_hash") != _file_hash(parent_memory_path):
                raise FinalIterationConflictError(
                    "property-memory parent path/hash mismatch")

        emit_stage(
            stage_logger, event="START", stage="memory_update",
            message="property_memory_v1 update 시작",
            iteration_id=iteration_id,
            parent_memory_path=parent_memory_path)
        try:
            memory_result = self.property_memory_runner.run(
                iteration_id=iteration_id, iteration_ordinal=iteration_ordinal,
                prompt_bank=prompt_bank,
                baseline_video_manifest_paths=baseline_video_manifest_paths,
                intervention_manifest_paths=intervention_manifest_paths,
                feedback_manifest_paths=feedback_manifest_paths,
                output_dir=os.path.join(output_dir, "compact_property_memory"),
                parent_memory_path=parent_memory_path)
        except Exception as exc:
            emit_stage(
                stage_logger, event="ERROR", stage="memory_update",
                message="property_memory_v1 update 실패",
                error_type=type(exc).__name__, error=str(exc))
            raise
        memory_path = memory_result["property_memory_snapshot_path"]
        memory_snapshot = _read_json(memory_path)
        emit_stage(
            stage_logger, event="END", stage="memory_update",
            message="property_memory_v1 update 끝",
            snapshot_path=memory_path,
            active_property_count=len(memory_snapshot.get("properties") or ()),
            candidate_memory_count=len(memory_snapshot.get("candidates") or ()))
        emit_stage(
            stage_logger, event="START", stage="codebook_update",
            message="LLM codebook update 시작",
            iteration_id=iteration_id, memory_snapshot_path=memory_path)
        try:
            updater_result = self.llm_codebook_updater.run(
                iteration_id=iteration_id, prompt_bank=prompt_bank,
                property_memory_snapshot_path=memory_path,
                intervention_summary_path=memory_result["effect_summary_path"],
                output_dir=os.path.join(output_dir, "llm_codebook_updater"))
        except Exception as exc:
            emit_stage(
                stage_logger, event="ERROR", stage="codebook_update",
                message="LLM codebook update 실패",
                error_type=type(exc).__name__, error=str(exc))
            raise
        applied_codebook_plan = _read_json(
            updater_result["artifacts"]["applied_plan"])
        codebook_validation = _read_json(
            updater_result["artifacts"]["validation_report"])
        emit_stage(
            stage_logger, event="END", stage="codebook_update",
            message="LLM codebook update 끝",
            applied_actions=applied_codebook_plan.get("actions") or (),
            rejected_action_count=len(
                codebook_validation.get("rejected_actions") or ()),
            candidate_codebook=updater_result["artifacts"]["candidate_codebook"])
        updater_manifest_path = os.path.join(
            output_dir, "llm_codebook_updater", "manifest.json")
        artifacts = {
            "property_memory_manifest": os.path.join(
                output_dir, "compact_property_memory", "manifest.json"),
            "property_memory_snapshot": memory_path,
            "intervention_summaries": memory_result["effect_summary_path"],
            "llm_codebook_updater_manifest": updater_manifest_path,
            "candidate_codebook": updater_result["artifacts"]["candidate_codebook"],
            "property_id_mapping": updater_result["artifacts"]["property_id_mapping"],
            "promoted_property_memory": updater_result["artifacts"][
                "promoted_property_memory"],
        }
        manifest = {
            "schema_version": "memory_conditioned_codebook_iteration_v1",
            "status": "completed", "iteration_id": iteration_id,
            "iteration_ordinal": iteration_ordinal,
            "input_bank_version": prompt_bank.bank_version,
            "input_bank_hash": bank_hash,
            "input_contract_hash": input_contract_hash,
            "parent_property_memory_path": parent_memory_path,
            "stages": {
                "property_memory": _stage_identity(self.property_memory_runner),
                "llm_codebook_updater": _stage_identity(self.llm_codebook_updater),
            },
            "artifacts": artifacts,
            "artifact_hashes": {name: _file_hash(path)
                                for name, path in artifacts.items()},
            "router_updated": False,
            "production_pointer_mutated": False,
            "resumed": False,
        }
        manifest_path = _write_immutable(manifest_target, manifest)
        _write_pointer(pointer_path, {
            "schema_version": "property_memory_lineage_pointer_v1",
            "latest_iteration_id": iteration_id,
            "active_bank_version": prompt_bank.bank_version,
            "active_bank_hash": bank_hash,
            "snapshot_path": os.path.abspath(memory_path),
            "snapshot_hash": _file_hash(memory_path),
            "checkpoint_manifest_path": manifest_path,
            "checkpoint_manifest_hash": _file_hash(manifest_path),
        })
        return _read_json(manifest_path)

    def run_memory_conditioned_router_checkpoint(
        self, *, iteration_id: str,
        parent_router_policy: RouterPolicySnapshot,
        codebook_checkpoint_manifest_path: str,
        output_dir: str, state_dir: str,
        stage_logger: StageLogger | None = None,
    ) -> Mapping[str, Any]:
        """Finish the memory-conditioned update as one provisional policy pair.

        This uses a checkpoint-local pointer. It does not touch coverage-cycle,
        active-provisional, or confirmed production pointers.
        """
        if self.llm_router_updater is None:
            raise FinalIterationError(
                "memory-conditioned router checkpoint requires its stage runner")
        output_dir = os.path.abspath(output_dir)
        state_dir = os.path.abspath(state_dir)
        manifest_target = os.path.join(output_dir, "manifest.json")
        failure_target = os.path.join(output_dir, "failure.json")
        pointer_path = os.path.join(
            state_dir, "memory_conditioned_provisional", "current.json")
        codebook_manifest_path = os.path.abspath(codebook_checkpoint_manifest_path)
        codebook_checkpoint = _read_json(codebook_manifest_path)
        if codebook_checkpoint.get("schema_version") != \
                "memory_conditioned_codebook_iteration_v1" or \
                codebook_checkpoint.get("status") != "completed" or \
                codebook_checkpoint.get("iteration_id") != iteration_id:
            raise FinalIterationConflictError(
                "incompatible memory-conditioned codebook checkpoint")
        checkpoint_artifacts = codebook_checkpoint.get("artifacts") or {}
        checkpoint_hashes = codebook_checkpoint.get("artifact_hashes") or {}
        if set(checkpoint_artifacts) != set(checkpoint_hashes) or any(
                not os.path.exists(path) or _file_hash(path) != checkpoint_hashes[name]
                for name, path in checkpoint_artifacts.items()):
            raise FinalIterationConflictError("codebook-checkpoint artifact mismatch")
        updater_manifest_path = checkpoint_artifacts["llm_codebook_updater_manifest"]
        codebook_updater_manifest = _read_json(updater_manifest_path)
        if codebook_updater_manifest.get("schema_version") != \
                CODEBOOK_UPDATER_MANIFEST_SCHEMA_VERSION or \
                codebook_updater_manifest.get("status") != "completed":
            raise FinalIterationConflictError("incompatible codebook-updater artifacts")
        updater_artifacts = codebook_updater_manifest.get("artifacts") or {}
        candidate_codebook_path = checkpoint_artifacts["candidate_codebook"]
        mapping_path = checkpoint_artifacts["property_id_mapping"]
        promoted_memory_path = checkpoint_artifacts["promoted_property_memory"]
        required = ("applied_plan", "validation_report")
        if any(name not in updater_artifacts for name in required):
            raise FinalIterationConflictError("incomplete codebook-updater audit closure")
        input_contract = {
            "schema_version": "memory_router_atomic_input_v1",
            "iteration_id": iteration_id,
            "parent_router": as_json_dict(parent_router_policy),
            "codebook_checkpoint_path": codebook_manifest_path,
            "codebook_checkpoint_hash": _file_hash(codebook_manifest_path),
            "candidate_codebook_hash": _file_hash(candidate_codebook_path),
            "property_id_mapping_hash": _file_hash(mapping_path),
            "property_memory_hash": _file_hash(promoted_memory_path),
            "codebook_applied_plan_hash": _file_hash(updater_artifacts["applied_plan"]),
            "codebook_validation_hash": _file_hash(
                updater_artifacts["validation_report"]),
            "router_updater": _stage_identity(self.llm_router_updater),
            "state_dir": state_dir,
        }
        fingerprint = sha256_json(input_contract)
        if os.path.exists(manifest_target):
            manifest = _read_json(manifest_target)
            if manifest.get("schema_version") != \
                    "memory_conditioned_atomic_policy_pair_v1" or \
                    manifest.get("status") != "committed" or \
                    manifest.get("input_fingerprint") != fingerprint:
                raise FinalIterationConflictError(
                    "incompatible completed atomic policy-pair checkpoint")
            artifacts, hashes = manifest.get("artifacts") or {}, \
                manifest.get("artifact_hashes") or {}
            if set(artifacts) != set(hashes) or any(
                    not os.path.exists(path) or _file_hash(path) != hashes[name]
                    for name, path in artifacts.items()):
                raise FinalIterationConflictError("atomic policy-pair artifact mismatch")
            if not os.path.exists(pointer_path):
                raise FinalIterationConflictError("atomic provisional-pair pointer missing")
            pointer = _read_json(pointer_path)
            if pointer.get("schema_version") != \
                    "memory_conditioned_provisional_pair_pointer_v1" or \
                    pointer.get("pair_manifest_hash") != _file_hash(manifest_target):
                raise FinalIterationConflictError("atomic provisional-pair pointer mismatch")
            candidate_bank = prompt_bank_from_json(
                _read_json(artifacts["provisional_codebook"]))
            candidate_router = router_policy_from_json(
                _read_json(artifacts["provisional_router"]))
            from surrogate_rollout.prompt_routing.structured_router_policy import (
                validate_rendered_prompt_configuration,
            )
            validate_rendered_prompt_configuration(candidate_router, candidate_bank)
            emit_stage(
                stage_logger, event="RESUME", stage="router_update",
                message="LLM router update 및 atomic pair artifact 재사용",
                rendered_router_prompt_hash=manifest.get(
                    "rendered_router_prompt_hash"),
                atomic_policy_pair=artifacts.get("atomic_policy_pair"))
            return {**manifest, "resumed": True}
        if os.path.exists(failure_target) or (
                os.path.isdir(output_dir) and os.listdir(output_dir)):
            raise FinalIterationConflictError(
                "incomplete or failed atomic policy-pair checkpoint cannot be resumed")

        try:
            emit_stage(
                stage_logger, event="START", stage="router_update",
                message="LLM router update 시작",
                iteration_id=iteration_id,
                candidate_codebook_path=candidate_codebook_path)
            router_result = self.llm_router_updater.run(
                iteration_id=iteration_id,
                parent_router_policy=parent_router_policy,
                candidate_codebook_path=candidate_codebook_path,
                property_id_mapping_path=mapping_path,
                property_memory_snapshot_path=promoted_memory_path,
                codebook_applied_plan_path=updater_artifacts["applied_plan"],
                codebook_validation_report_path=updater_artifacts[
                    "validation_report"],
                output_dir=os.path.join(output_dir, "llm_router_updater"))
            candidate_wrapper = _read_json(candidate_codebook_path)
            candidate_bank = prompt_bank_from_json(candidate_wrapper["candidate_bank"])
            candidate_router = router_policy_from_json(_read_json(
                router_result["artifacts"]["candidate_router_policy"]))
            errors = validate_router_against_bank(candidate_router, candidate_bank)
            if errors:
                raise FinalIterationError("; ".join(errors))
            from surrogate_rollout.prompt_routing.structured_router_policy import (
                validate_rendered_prompt_configuration,
            )
            validate_rendered_prompt_configuration(candidate_router, candidate_bank)
            active_bank_ids = {entry.prompt_id for entry in candidate_bank.entries
                               if entry.status == "active"}
            structured = _read_json(
                router_result["artifacts"]["structured_router_policy"])
            structured_ids = {row["property_id"] for row in structured["properties"]}
            if structured_ids != active_bank_ids:
                raise FinalIterationError("candidate codebook/router ID-space mismatch")

            pair_root = os.path.join(output_dir, "provisional_policy_pair")
            bank_path = _write_immutable(
                os.path.join(pair_root, "provisional_codebook.json"), candidate_bank)
            router_path = _write_immutable(
                os.path.join(pair_root, "provisional_router.json"), candidate_router)
            pair_artifacts = {
                "provisional_codebook": bank_path,
                "provisional_router": router_path,
                "property_memory_snapshot": promoted_memory_path,
                "candidate_codebook_source": candidate_codebook_path,
                "property_id_mapping": mapping_path,
                "structured_router_policy": router_result["artifacts"][
                    "structured_router_policy"],
                "rendered_router_prompt": router_result["artifacts"][
                    "rendered_router_prompt"],
                "codebook_applied_plan": updater_artifacts["applied_plan"],
                "codebook_validation_report": updater_artifacts[
                    "validation_report"],
                "router_applied_plan": router_result["artifacts"]["applied_plan"],
                "router_validation_report": router_result["artifacts"][
                    "validation_report"],
                "router_updater_execution": router_result["artifacts"][
                    "execution"],
                "codebook_updater_manifest": updater_manifest_path,
                "router_updater_manifest": os.path.join(
                    output_dir, "llm_router_updater", "manifest.json"),
            }
            mapping = _read_json(mapping_path)
            rendered = _read_json(pair_artifacts["rendered_router_prompt"])
            pair = {
                "schema_version": "atomic_provisional_policy_pair_v1",
                "status": "committed", "iteration_id": iteration_id,
                "parent_policy_pair": {
                    "input_bank_version": codebook_checkpoint["input_bank_version"],
                    "input_bank_hash": codebook_checkpoint["input_bank_hash"],
                    "router_version": parent_router_policy.router_version,
                    "router_hash": _object_hash(parent_router_policy),
                },
                "candidate_policy_pair": {
                    "bank_version": candidate_bank.bank_version,
                    "bank_hash": _object_hash(candidate_bank),
                    "router_version": candidate_router.router_version,
                    "router_hash": _object_hash(candidate_router),
                    "rendered_router_prompt_hash": rendered["prompt_sha256"],
                    "structured_router_policy_schema_version": structured[
                        "schema_version"],
                },
                "old_to_new_property_ids": mapping[
                    "old_to_new_property_ids"],
                "candidate_promotions": mapping["candidate_promotions"],
                "router_updater_execution_mode": router_result.get(
                    "execution_mode"),
                "router_updater_provider_called": router_result.get(
                    "provider_called"),
                "artifacts": pair_artifacts,
                "artifact_hashes": {name: _file_hash(path)
                                    for name, path in pair_artifacts.items()},
                "confirmed_production_pointer_mutated": False,
                "confirmation_run": False,
            }
            pair_path = _write_immutable(
                os.path.join(pair_root, "policy_pair.json"), pair)
            artifacts = {**pair_artifacts, "atomic_policy_pair": pair_path}
            manifest = {
                "schema_version": "memory_conditioned_atomic_policy_pair_v1",
                "status": "committed", "iteration_id": iteration_id,
                "input_fingerprint": fingerprint,
                "artifacts": artifacts,
                "artifact_hashes": {name: _file_hash(path)
                                    for name, path in artifacts.items()},
                "candidate_bank_version": candidate_bank.bank_version,
                "candidate_router_version": candidate_router.router_version,
                "rendered_router_prompt_hash": rendered["prompt_sha256"],
                "router_updater_execution_mode": router_result.get(
                    "execution_mode"),
                "router_updater_provider_called": router_result.get(
                    "provider_called"),
                "confirmed_production_pointer_mutated": False,
                "confirmation_run": False, "resumed": False,
            }
            manifest_path = _write_immutable(manifest_target, manifest)
            _write_pointer(pointer_path, {
                "schema_version": "memory_conditioned_provisional_pair_pointer_v1",
                "iteration_id": iteration_id,
                "pair_manifest_path": manifest_path,
                "pair_manifest_hash": _file_hash(manifest_path),
                "policy_pair_path": pair_path,
                "policy_pair_hash": _file_hash(pair_path),
                "bank_path": bank_path, "bank_hash": _file_hash(bank_path),
                "router_path": router_path, "router_hash": _file_hash(router_path),
                "rendered_router_prompt_hash": rendered["prompt_sha256"],
                "property_memory_path": promoted_memory_path,
                "property_memory_hash": _file_hash(promoted_memory_path),
            })
            _write_pointer(os.path.join(
                    state_dir, "property_memory", "current.json"), {
                "schema_version": "property_memory_lineage_pointer_v1",
                "latest_iteration_id": iteration_id,
                "active_bank_version": candidate_bank.bank_version,
                "active_bank_hash": _object_hash(candidate_bank),
                "snapshot_path": os.path.abspath(promoted_memory_path),
                "snapshot_hash": _file_hash(promoted_memory_path),
                "checkpoint_manifest_path": manifest_path,
                "checkpoint_manifest_hash": _file_hash(manifest_path),
            })
            emit_stage(
                stage_logger, event="END", stage="router_update",
                message="LLM router update 및 atomic provisional pair 저장 끝",
                rendered_router_prompt_hash=rendered["prompt_sha256"],
                atomic_policy_pair=pair_path,
                candidate_bank_version=candidate_bank.bank_version,
                candidate_router_version=candidate_router.router_version,
                router_updater_execution_mode=router_result.get(
                    "execution_mode"),
                router_updater_provider_called=router_result.get(
                    "provider_called"))
            return _read_json(manifest_path)
        except Exception as exc:
            emit_stage(
                stage_logger, event="ERROR", stage="router_update",
                message="LLM router update 실패; parent pair 유지",
                error_type=type(exc).__name__, error=str(exc))
            _write_immutable(failure_target, {
                "schema_version": "memory_conditioned_atomic_policy_pair_failure_v1",
                "status": "failed", "iteration_id": iteration_id,
                "input_fingerprint": fingerprint,
                "error_type": type(exc).__name__, "error": str(exc),
                "rollback": {
                    "router_version": parent_router_policy.router_version,
                    "codebook_checkpoint_committed_as_pair": False,
                    "provisional_pair_pointer_written": False,
                },
            })
            raise

    def apply_update_plan(
        self, *, prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
        plan: IterationUpdatePlan,
    ) -> tuple[PromptBankSnapshot, RouterPolicySnapshot]:
        if plan.input_bank_version != prompt_bank.bank_version or \
                plan.input_router_version != router_policy.router_version or \
                plan.optimize_scaffold:
            raise FinalIterationError("stale update plan or scaffold optimization")
        entries = list(prompt_bank.entries)
        by_id = {entry.prompt_id: index for index, entry in enumerate(entries)}
        replacement: dict[str, str | None] = {}
        bank_changed = False
        for decision in plan.codebook_decisions:
            if decision.action not in CODEBOOK_ACTIONS:
                raise FinalIterationError("invalid codebook action")
            if decision.action == "no_op":
                continue
            bank_changed = True
            if decision.action == "add":
                if decision.resulting_property_id in by_id:
                    raise FinalIterationError("duplicate added property ID")
                entries.append(PromptEntry(
                    prompt_id=str(decision.resulting_property_id),
                    prompt_text=str(decision.property_text),
                    prompt_hash=sha256_text(str(decision.property_text)),
                    name="Learned reusable caption property",
                    description="Accepted from interventional feedback.",
                    tags=("learned",),
                    created_by="checkpoint3e",
                    provenance={
                        "decision_id": decision.decision_id,
                        "feedback_ids": decision.supporting_feedback_ids,
                        "video_ids": decision.supporting_video_ids,
                    }))
                by_id[str(decision.resulting_property_id)] = len(entries) - 1
            elif decision.action == "revise":
                target = decision.target_property_ids[0]
                current = entries[by_id[target]]
                entries[by_id[target]] = replace(
                    current, prompt_text=str(decision.property_text),
                    prompt_hash=sha256_text(str(decision.property_text)),
                    parent_prompt_ids=(target,), created_by="checkpoint3e",
                    provenance={"decision_id": decision.decision_id,
                                "feedback_ids": decision.supporting_feedback_ids})
            elif decision.action == "retire":
                for target in decision.target_property_ids:
                    entries[by_id[target]] = replace(
                        entries[by_id[target]], status="retired",
                        provenance={"decision_id": decision.decision_id,
                                    "feedback_ids": decision.supporting_feedback_ids})
                    replacement[target] = None
            elif decision.action == "merge":
                result_id = str(decision.resulting_property_id)
                if result_id in by_id:
                    raise FinalIterationError("duplicate merged property ID")
                entries.append(PromptEntry(
                    prompt_id=result_id, prompt_text=str(decision.property_text),
                    prompt_hash=sha256_text(str(decision.property_text)),
                    name="Merged reusable caption property",
                    description="Merged from accepted interventional evidence.",
                    tags=("learned", "merged"),
                    parent_prompt_ids=decision.target_property_ids,
                    created_by="checkpoint3e",
                    provenance={"decision_id": decision.decision_id,
                                "feedback_ids": decision.supporting_feedback_ids}))
                by_id[result_id] = len(entries) - 1
                for target in decision.target_property_ids:
                    entries[by_id[target]] = replace(
                        entries[by_id[target]], status="retired")
                    replacement[target] = result_id
        default_ids = tuple(dict.fromkeys(
            replacement.get(item, item) for item in prompt_bank.default_entry_ids
            if replacement.get(item, item) is not None))
        if bank_changed:
            bank_payload = {
                "parent": prompt_bank.bank_version, "entries": entries,
                "defaults": default_ids, "plan": plan.plan_id,
            }
            bank = replace(
                prompt_bank,
                bank_version=_next_version(
                    "bank", prompt_bank.bank_version, bank_payload),
                parent_bank_version=prompt_bank.bank_version,
                entries=tuple(entries), default_entry_ids=default_ids,
                created_by="checkpoint3e",
                provenance={"plan_id": plan.plan_id,
                            "feedback_ids": plan.accepted_feedback_ids,
                            "provisional": True})
        else:
            bank = prompt_bank
        errors = validate_prompt_bank(bank)
        if errors:
            raise FinalIterationError("; ".join(errors))

        active_after = {entry.prompt_id for entry in bank.entries
                        if entry.status == "active"}
        invalid_supervision = {
            item.property_id for item in plan.router_supervision
            if item.property_id not in active_after
        }
        if invalid_supervision:
            raise FinalIterationError(
                "router supervision targets inactive properties: "
                f"{sorted(invalid_supervision)}")

        rules = []
        for rule in router_policy.rules:
            targets = tuple(dict.fromkeys(
                replacement.get(item, item) for item in rule.target_prompt_ids
                if replacement.get(item, item) is not None))
            rules.append(replace(rule, target_prompt_ids=(targets or rule.target_prompt_ids),
                                 enabled=bool(targets) and rule.enabled))
        fallback = tuple(dict.fromkeys(
            replacement.get(item, item) for item in router_policy.fallback_prompt_ids
            if replacement.get(item, item) is not None))
        previous_examples = tuple(
            router_policy.configuration.get("supervision_examples") or ())
        new_examples = tuple({
            "supervision_id": item.supervision_id,
            "property_id": item.property_id,
            "polarity": item.polarity,
            "evidence": item.evidence,
            "context_hash": item.context_hash,
            "attributed_segment_hash": sha256_json(item.attributed_segment_ids),
        } for item in plan.router_supervision)
        router_changed = bool(new_examples or replacement)
        if router_changed:
            configuration = {
                **as_json_dict(router_policy.configuration),
                "supervision_schema_version": "router_supervision_v1",
                "supervision_examples": (*previous_examples, *new_examples),
            }
            router_payload = {
                "parent": router_policy.router_version, "rules": rules,
                "fallback": fallback, "configuration": configuration,
                "plan": plan.plan_id,
            }
            router = replace(
                router_policy,
                router_version=_next_version(
                    "router", router_policy.router_version, router_payload),
                parent_router_version=router_policy.router_version,
                rules=tuple(rules), fallback_prompt_ids=fallback,
                configuration=configuration,
                provenance={"plan_id": plan.plan_id,
                            "feedback_ids": plan.accepted_feedback_ids,
                            "provisional": True})
        else:
            router = router_policy
        errors = validate_router_against_bank(router, bank)
        if errors:
            raise FinalIterationError("; ".join(errors))
        return bank, router

    def _run_confirmation(
        self, *, parent_confirmed: ConfirmedCheckpointState,
        roles: TrainRoleAssignment, incumbent_bank: PromptBankSnapshot,
        incumbent_router: RouterPolicySnapshot,
        candidate_bank: PromptBankSnapshot,
        candidate_router: RouterPolicySnapshot,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        iteration_id: str, coverage_cycle: int, output_dir: str,
        pending_provisional_iteration_ids: tuple[str, ...],
        provisional_bank_path: str, provisional_router_path: str,
    ) -> tuple[ConfirmationDecision, str, str, str]:
        raw = self.confirmation_evaluator.evaluate(
            confirmation_videos=roles.confirmation_videos,
            incumbent_bank=incumbent_bank, incumbent_router=incumbent_router,
            candidate_bank=candidate_bank, candidate_router=candidate_router,
            scaffold_policy=scaffold_policy, scaffold_contract=scaffold_contract,
            output_dir=output_dir)
        raw_path = _write_immutable(
            os.path.join(output_dir, "evaluation.json"), raw)
        expected = {(video.video_id, question_id)
                    for video in roles.confirmation_videos
                    for question_id in video.question_ids}
        rows = tuple(raw.get("qas") or ())
        observed = {(str(row.get("video_id")), str(row.get("question_id")))
                    for row in rows}
        errors = tuple(str(row.get("error")) for row in rows if row.get("error"))
        if len(rows) != 6 or observed != expected:
            errors = (*errors, "confirmation must contain exactly six holdout QAs")
        incumbent_accuracy = (sum(float(row.get("incumbent_score", 0.0))
                                  for row in rows) / 6 if len(rows) == 6 else 0.0)
        candidate_accuracy = (sum(float(row.get("candidate_score", 0.0))
                                  for row in rows) / 6 if len(rows) == 6 else 0.0)
        regressions = tuple(sorted(str(row["question_id"]) for row in rows
                                   if float(row.get("incumbent_score", 0.0)) > 0.0
                                   and float(row.get("candidate_score", 0.0)) <= 0.0))
        accepted = not errors and not regressions and \
            candidate_accuracy >= incumbent_accuracy
        active_bank = candidate_bank if accepted else incumbent_bank
        active_router = candidate_router if accepted else incumbent_router
        active_bank_path = (provisional_bank_path if accepted else _write_immutable(
            os.path.join(output_dir, "rollback_bank.json"), incumbent_bank))
        active_router_path = (provisional_router_path if accepted else _write_immutable(
            os.path.join(output_dir, "rollback_router.json"), incumbent_router))
        decision_target = os.path.join(output_dir, "decision.json")
        decision_payload = {
            "schema_version": "coverage_confirmation_decision_v1",
            "status": "accepted" if accepted else "rejected",
            "accepted": accepted,
            "incumbent_accuracy": incumbent_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "correct_to_wrong_question_ids": regressions,
            "errors": errors,
            "criterion": {
                "required_video_count": 2, "required_qa_count": 6,
                "minimum_accuracy_delta": 0.0,
                "maximum_correct_to_wrong": 0,
                "all_evaluations_error_free": True,
                "confirmation_generates_feedback": False,
            },
            "incumbent_versions": {
                "bank": incumbent_bank.bank_version,
                "router": incumbent_router.router_version,
            },
            "incumbent_hashes": {
                "bank": sha256_json(as_json_dict(incumbent_bank)),
                "router": sha256_json(as_json_dict(incumbent_router)),
            },
            "candidate_versions": {
                "bank": candidate_bank.bank_version,
                "router": candidate_router.router_version,
            },
            "candidate_hashes": {
                "bank": sha256_json(as_json_dict(candidate_bank)),
                "router": sha256_json(as_json_dict(candidate_router)),
            },
            "active_versions": {
                "bank": active_bank.bank_version,
                "router": active_router.router_version,
            },
            "rollback_to_checkpoint_id": (
                None if accepted else parent_confirmed.checkpoint_id),
            "rollback_action": (
                "none" if accepted else "restore_exact_parent_confirmed_pair"),
            "evaluation_artifact_path": raw_path,
            "optimize_scaffold": False,
            "scaffold_version": scaffold_policy.scaffold_version,
            "contract_version": scaffold_contract.contract_version,
        }
        decision_path = _write_immutable(decision_target, decision_payload)
        if accepted:
            checkpoint_id = "confirmed_" + sha256_json({
                "parent": parent_confirmed.checkpoint_id,
                "iteration": iteration_id,
                "bank": candidate_bank.bank_version,
                "router": candidate_router.router_version,
                "decision": _read_json(decision_path),
            })[:20]
            checkpoint = ConfirmedCheckpointState(
                checkpoint_id=checkpoint_id,
                coverage_cycle=coverage_cycle + 1,
                component_versions=ComponentVersions(
                    candidate_bank.bank_version, candidate_router.router_version,
                    scaffold_policy.scaffold_version,
                    scaffold_contract.contract_version),
                confirmation_video_ids=tuple(
                    item.video_id for item in roles.confirmation_videos),
                accepted_provisional_iteration_ids=(
                    *parent_confirmed.accepted_provisional_iteration_ids,
                    *pending_provisional_iteration_ids),
                confirmation_artifact_path=decision_path,
                parent_confirmed_checkpoint_id=parent_confirmed.checkpoint_id)
        else:
            checkpoint = parent_confirmed
        checkpoint_path = _write_immutable(
            os.path.join(output_dir, "active_confirmed_checkpoint.json"), checkpoint)
        decision = ConfirmationDecision(
            status=decision_payload["status"], accepted=accepted,
            incumbent_accuracy=incumbent_accuracy,
            candidate_accuracy=candidate_accuracy,
            correct_to_wrong_question_ids=regressions, errors=errors,
            criterion=decision_payload["criterion"],
            active_bank_version=active_bank.bank_version,
            active_router_version=active_router.router_version,
            rollback_to_checkpoint_id=decision_payload["rollback_to_checkpoint_id"],
            artifact_path=decision_path)
        return decision, checkpoint_path, active_bank_path, active_router_path

    @staticmethod
    def _validate_completed_manifest(value: Mapping[str, Any]) -> None:
        required_paths = (
            value.get("baseline_manifest_path"),
            value.get("update_plan_path"),
            value.get("provisional_bank_path"),
            value.get("provisional_router_path"),
            value.get("provisional_state_path"),
            value.get("active_bank_path"), value.get("active_router_path"),
            *(value.get("intervention_manifest_paths") or ()),
            *(value.get("feedback_manifest_paths") or ()),
        )
        optional_paths = (
            value.get("confirmation_decision_path"),
            value.get("confirmed_checkpoint_path"),
        )
        state_paths = (
            value.get("active_provisional_reference_path"),
            value.get("confirmed_pointer_path"),
        )
        missing = [path for path in (*required_paths, *optional_paths, *state_paths)
                   if path and not os.path.exists(path)]
        if any(path is None for path in (*required_paths, *state_paths)) or missing:
            raise FinalIterationConflictError(
                f"completed iteration references missing artifacts: {missing}")
        hashes = value.get("artifact_hashes") or {}
        hash_paths = {
            "update_plan": value["update_plan_path"],
            "provisional_bank": value["provisional_bank_path"],
            "provisional_router": value["provisional_router_path"],
            "provisional_state": value["provisional_state_path"],
        }
        if set(hashes) != set(hash_paths) or any(
                hashes[name] != _file_hash(path)
                for name, path in hash_paths.items()):
            raise FinalIterationConflictError(
                "completed iteration artifact hash mismatch")

    @staticmethod
    def _result_from_manifest(path: str, value: Mapping[str, Any],
                              resumed: bool = False) -> FinalIterationResult:
        return FinalIterationResult(
            iteration_id=value["iteration_id"],
            status=value["iteration_status"], manifest_path=os.path.abspath(path),
            selected_video_ids=tuple(value["selected_video_ids"]),
            baseline_manifest_path=value["baseline_manifest_path"],
            intervention_manifest_paths=tuple(
                value["intervention_manifest_paths"]),
            feedback_manifest_paths=tuple(value["feedback_manifest_paths"]),
            update_plan_path=value["update_plan_path"],
            provisional_bank_path=value["provisional_bank_path"],
            provisional_router_path=value["provisional_router_path"],
            provisional_state_path=value["provisional_state_path"],
            confirmation_decision_path=value.get("confirmation_decision_path"),
            confirmed_checkpoint_path=value.get("confirmed_checkpoint_path"),
            active_bank_path=value["active_bank_path"],
            active_router_path=value["active_router_path"],
            active_provisional_reference_path=
                value["active_provisional_reference_path"],
            confirmed_pointer_path=value["confirmed_pointer_path"],
            next_coverage_state=_coverage_from_json(value["next_coverage_state"]),
            resumed=resumed)
