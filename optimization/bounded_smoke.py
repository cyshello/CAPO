"""One-video progressive smoke orchestration isolated from production state."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

from surrogate_rollout.optimization.final_iteration import (
    Checkpoint3EOrchestrator,
    FinalIterationConflictError,
    _write_immutable,
)
from surrogate_rollout.optimization.baseline_phase import (
    qa_execution_failure_reasons,
)
from surrogate_rollout.optimization.property_proposal import CandidatePropertyProposal
from surrogate_rollout.optimization.startup_models import (
    build_startup_model_manifest,
    log_startup_models,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import (
    Phase4Config,
    PostInterventionMode,
    PromptBankSnapshot,
    RouterPolicySnapshot,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    as_json_dict,
    dumps_canonical,
    SegmentContext,
)
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
)
from surrogate_rollout.schemas import sha256_json, sha256_text


MODE_ORDER = {
    PostInterventionMode.QA_ONLY: 0,
    PostInterventionMode.FEEDBACK_ONLY: 1,
    PostInterventionMode.PROVISIONAL_UPDATE: 2,
}


class BoundedSmokeError(RuntimeError):
    pass


class BoundedSmokeConflictError(BoundedSmokeError):
    pass


@dataclass(frozen=True)
class BoundedSmokeResult:
    video_id: str
    post_intervention_mode: PostInterventionMode
    manifest_path: str
    baseline_manifest_path: str
    intervention_manifest_path: str
    feedback_manifest_path: str | None
    update_plan_path: str | None
    provisional_bank_path: str | None
    provisional_router_path: str | None
    resumed: bool


def _read_json(path: str) -> dict[str, Any]:
    with open(path) as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise BoundedSmokeError(f"expected JSON object: {path}")
    return value


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: str) -> tuple[dict[str, Any], ...]:
    with open(path) as handle:
        return tuple(json.loads(line) for line in handle if line.strip())


def _completed_mode_has_invalid_baseline_qa(mode_manifest: str) -> bool:
    mode = _read_json(mode_manifest)
    if "baseline_manifest_path" not in mode:
        qa_manifest = os.path.join(os.path.dirname(mode_manifest), "qa_only.json")
        if not os.path.exists(qa_manifest):
            return False
        mode = _read_json(qa_manifest)
    baseline = _read_json(mode["baseline_manifest_path"])
    for video_path in baseline.get("video_manifest_paths") or ():
        video = _read_json(video_path)
        rows = _read_jsonl(video["baseline_qas_path"])
        if len(rows) != 3 or any(qa_execution_failure_reasons(row) for row in rows):
            return True
    return False


def _archive_invalid_qa_attempt(output_dir: str, video_id: str) -> str:
    """Preserve invalid downstream artifacts while retaining caption blocks."""
    archive_parent = os.path.join(output_dir, "invalid_attempts")
    attempt = 1
    while os.path.exists(os.path.join(
            archive_parent, f"qa_execution_failure_{attempt:03d}")):
        attempt += 1
    archive = os.path.join(
        archive_parent, f"qa_execution_failure_{attempt:03d}")
    relative_targets = (
        os.path.join("baseline", "baseline", video_id, "qa"),
        os.path.join("baseline", "baseline", video_id, "baseline_qas.jsonl"),
        os.path.join("baseline", "baseline", video_id, "video_complete.json"),
        os.path.join("baseline", "property_proposals"),
        os.path.join("baseline", "property_retrieval"),
        os.path.join("baseline", "iteration_state"),
        os.path.join("baseline", "manifest.json"),
        "intervention", "feedback", "mode_manifests", "run_manifest.json",
    )
    moved = []
    for relative in relative_targets:
        source = os.path.join(output_dir, relative)
        if not os.path.exists(source):
            continue
        target = os.path.join(archive, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.replace(source, target)
        moved.append(relative)
    os.makedirs(archive, exist_ok=True)
    _atomic_write_text(os.path.join(archive, "archive_manifest.json"), json.dumps({
        "schema_version": "bounded_smoke_invalid_qa_archive_v1",
        "reason": "qa_execution_failure_not_captioning_incorrectness",
        "video_id": video_id, "moved_paths": moved,
        "caption_artifacts_preserved_in_place": True,
    }, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
    return os.path.abspath(archive)


def _proposal_from_json(value: Mapping[str, Any]) -> CandidatePropertyProposal:
    return CandidatePropertyProposal(
        candidate_property_id=value["candidate_property_id"],
        property_text=value["property_text"],
        source_video_id=value["source_video_id"],
        source_question_ids=tuple(value["source_question_ids"]),
        motivating_failure_types=tuple(value["motivating_failure_types"]),
        coverage_hints=tuple(
            value.get("coverage_hints")
            or value.get("possible_coverage_by_existing_property_ids")
            or value.get("covered_by_existing_property_ids") or ()),
        proposal_rationale=value["proposal_rationale"],
        proposer_policy_version=value["proposer_policy_version"],
    )


def _stage_identity(value: Any) -> Mapping[str, Any]:
    identity = getattr(value, "configuration_identity", None)
    if identity is not None:
        return as_json_dict(identity)
    result = {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
    if hasattr(value, "policy_version"):
        result["policy_version"] = str(value.policy_version)
    if hasattr(value, "min_retire_support_videos"):
        result["min_retire_support_videos"] = int(
            value.min_retire_support_videos)
    proposal_policy = getattr(value, "proposal_policy", None)
    if proposal_policy is not None:
        result["proposal_policy"] = _stage_identity(proposal_policy)
    retrieval = getattr(value, "property_retrieval_runner", None)
    if retrieval is not None:
        result["property_retrieval"] = _stage_identity(retrieval)
    builder = getattr(value, "history_aware_builder", None)
    if builder is not None:
        result["history_router"] = _stage_identity(builder.router)
        result["history_captioner"] = _stage_identity(builder.segment_captioner)
    captioner = getattr(value, "segment_captioner", None)
    if captioner is not None:
        result["segment_captioner"] = _stage_identity(captioner)
    provider = getattr(value, "response_provider", None)
    if provider is not None:
        metadata = getattr(provider, "metadata", None)
        result["response_provider"] = (
            as_json_dict(metadata()) if callable(metadata) else
            {"type": f"{type(provider).__module__}.{type(provider).__qualname__}"})
        result["response_provider"].pop("call_count", None)
    if hasattr(value, "bounds"):
        result["bounds"] = as_json_dict(value.bounds)
    if hasattr(value, "frame_transform"):
        result["frame_transform"] = as_json_dict(value.frame_transform)
    for name in ("property_memory_runner", "llm_codebook_updater",
                 "llm_router_updater"):
        stage = getattr(value, name, None)
        if stage is not None:
            result[name] = _stage_identity(stage)
    for name in ("caption_cache_root", "caption_cache_manifest_path"):
        configured_path = getattr(value, name, None)
        if configured_path is not None:
            result[name] = os.path.abspath(configured_path)
    return result


class BoundedSmokeRunner:
    """Progress a one-video smoke through a configured post-intervention mode."""

    policy_version = "phase4_one_video_bounded_smoke_v1"

    def __init__(
        self, *, baseline_runner: Any, intervention_runner: Any,
        feedback_runner: Any, update_engine: Checkpoint3EOrchestrator,
        require_real_models: bool = True,
        memory_conditioned_update: bool = False,
    ) -> None:
        self.baseline_runner = baseline_runner
        self.intervention_runner = intervention_runner
        self.feedback_runner = feedback_runner
        self.update_engine = update_engine
        self.require_real_models = require_real_models
        self.memory_conditioned_update = memory_conditioned_update

    @staticmethod
    def _validate_directories(output_dir: str, state_dir: str, cache_dir: str) -> None:
        paths = tuple(os.path.abspath(item) for item in (
            output_dir, state_dir, cache_dir))
        if len(set(paths)) != 3:
            raise BoundedSmokeError(
                "bounded smoke output, state, and cache directories must be distinct")
        for index, left in enumerate(paths):
            for right in paths[index + 1:]:
                if os.path.commonpath((left, right)) in (left, right):
                    raise BoundedSmokeError(
                        "bounded smoke directories must not contain one another")

    def _validate_configuration(self, config: Phase4Config) -> None:
        if not isinstance(config.post_intervention_mode, PostInterventionMode):
            raise BoundedSmokeError("invalid post_intervention_mode")
        if not config.dry_run or config.commit:
            raise BoundedSmokeError("bounded smoke must be dry-run and non-committing")
        if config.optimize_scaffold:
            raise BoundedSmokeError("bounded smoke keeps the scaffold fixed")
        proposal_policy = getattr(self.baseline_runner, "proposal_policy", None)
        if getattr(proposal_policy, "max_proposals", None) != 1:
            raise BoundedSmokeError("bounded smoke requires max_proposals=1")
        retrieval = getattr(self.baseline_runner, "property_retrieval_runner", None)
        retriever = getattr(retrieval, "retriever", None)
        if getattr(retriever, "top_k", None) != 1:
            raise BoundedSmokeError("bounded smoke requires property_retrieval_top_k=1")
        if self.memory_conditioned_update and \
                config.post_intervention_mode is PostInterventionMode.PROVISIONAL_UPDATE and \
                any(getattr(self.update_engine, name, None) is None for name in (
                    "property_memory_runner", "llm_codebook_updater",
                    "llm_router_updater")):
            raise BoundedSmokeError(
                "provisional smoke requires all memory-conditioned update stages")

    def _probe_rendered_router_prompt(
        self, *, video_manifest_path: str, candidate_bank_path: str,
        candidate_router_path: str, output_dir: str,
    ) -> str:
        """Make one real post-commit router call through the active GPU pool."""
        video = _read_json(video_manifest_path)
        frames = _read_json(video["frames_path"])
        histories = _read_jsonl(video["frozen_histories_path"])
        history = next((row for row in histories
                        if frames.get(str(row.get("segment_id") or ""))), None)
        if history is None:
            raise BoundedSmokeError("router probe lacks a framed frozen history")
        segment_id = str(history["segment_id"])
        try:
            start, end = (float(item) for item in segment_id.split("_", 1))
        except (TypeError, ValueError) as exc:
            raise BoundedSmokeError("router probe segment ID is invalid") from exc
        bank = prompt_bank_from_json(_read_json(candidate_bank_path))
        router = router_policy_from_json(_read_json(candidate_router_path))
        rendered_hash = str(router.configuration.get(
            "rendered_router_prompt_hash") or "")
        if not rendered_hash:
            raise BoundedSmokeError("candidate router lacks rendered-prompt hash")
        builder = getattr(self.baseline_runner, "history_aware_builder", None)
        if builder is None:
            raise BoundedSmokeError("router probe requires history-aware builder")
        decision = builder.router.route(
            SegmentContext(
                video_id=str(video["video_id"]), segment_id=segment_id,
                timestamp_start=start, timestamp_end=end,
                segment_features={"frame_references": tuple(frames[segment_id])},
                history_summary=str(history["serialized_history"])),
            bank, router)
        exchange = builder.router.last_exchange
        if exchange is None or \
                exchange.request.get("router_prompt_identity", {}).get(
                    "rendered_prompt_hash") != rendered_hash or \
                decision.decision_payload.get(
                    "rendered_router_prompt_hash") != rendered_hash:
            raise BoundedSmokeError(
                "post-commit router call did not consume rendered prompt")
        return _write_immutable(os.path.join(
            output_dir, "router_prompt_consumption_probe.json"), {
                "schema_version": "bounded_smoke_router_prompt_probe_v1",
                "status": "completed", "video_id": video["video_id"],
                "segment_id": segment_id,
                "candidate_bank_version": bank.bank_version,
                "candidate_router_version": router.router_version,
                "rendered_router_prompt_hash": rendered_hash,
                "request": exchange.request,
                "request_hash": exchange.request_hash,
                "raw_output": exchange.raw_output,
                "output_hash": exchange.output_hash,
                "routing_decision": decision,
                "next_router_call_used_rendered_prompt": True,
            })

    @staticmethod
    def _write_run_pointer(
        *, output_dir: str, state_dir: str, cache_dir: str,
        iteration_id: str, video_id: str, mode: PostInterventionMode,
        base_fingerprint: str, requested_manifest: str,
    ) -> str:
        mode_root = os.path.dirname(requested_manifest)
        completed_modes = tuple(
            candidate.value for candidate in PostInterventionMode
            if os.path.exists(os.path.join(mode_root, f"{candidate.value}.json")))
        pointer = {
            "schema_version": "bounded_smoke_run_manifest_v1",
            "status": "completed", "iteration_id": iteration_id,
            "video_id": video_id, "post_intervention_mode": mode.value,
            "base_fingerprint": base_fingerprint,
            "completed_modes": completed_modes,
            "requested_mode_manifest_path": os.path.abspath(requested_manifest),
            "output_dir": output_dir, "state_dir": state_dir,
            "cache_dir": cache_dir, "confirmation_enabled": False,
            "canonical_state_mutated": False,
        }
        manifest_path = os.path.join(output_dir, "run_manifest.json")
        _atomic_write_text(manifest_path, json.dumps(
            pointer, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
        return os.path.abspath(manifest_path)

    def run(
        self, *, iteration_id: str, video_id: str, roles: Any,
        coverage_state: Any, phase4_config: Phase4Config,
        prompt_bank: PromptBankSnapshot, router_policy: RouterPolicySnapshot,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        output_dir: str, state_dir: str, cache_dir: str,
        execution_identity: Mapping[str, Any],
        baseline_kwargs: Mapping[str, Any],
        intervention_kwargs: Mapping[str, Any],
    ) -> BoundedSmokeResult:
        self._validate_configuration(phase4_config)
        self._validate_directories(output_dir, state_dir, cache_dir)
        output_dir, state_dir, cache_dir = map(os.path.abspath, (
            output_dir, state_dir, cache_dir))
        if not iteration_id or not video_id:
            raise BoundedSmokeError("iteration and video IDs must be non-empty")
        evidence_ids = {item.video_id for item in roles.evidence_videos}
        if video_id not in evidence_ids:
            raise BoundedSmokeError("bounded smoke video is outside the evidence pool")

        mode = phase4_config.post_intervention_mode
        mode_root = os.path.join(output_dir, "mode_manifests")
        requested_manifest = os.path.join(mode_root, f"{mode.value}.json")
        requested_mode_completed_before_run = os.path.exists(requested_manifest)
        base_identity = {
            "policy_version": self.policy_version,
            "iteration_id": iteration_id,
            "video_id": video_id,
            "components": {
                "prompt_bank": prompt_bank,
                "router_policy": router_policy,
                "scaffold_policy": scaffold_policy,
                "scaffold_contract": scaffold_contract,
            },
            "execution_identity": execution_identity,
            "phase4_config": {
                key: value for key, value in as_json_dict(phase4_config).items()
                if key != "post_intervention_mode"
            },
            "stage_identities": {
                "baseline": _stage_identity(self.baseline_runner),
                "intervention": _stage_identity(self.intervention_runner),
                "feedback": _stage_identity(self.feedback_runner),
                "update": _stage_identity(self.update_engine),
            },
            "limits": {
                "evidence_videos": 1, "qas_per_video": 3,
                "max_proposals": 1, "retrieval_top_k": 1,
                "max_candidate_interventions": 1,
            },
            "directories": {
                "output": output_dir, "state": state_dir, "cache": cache_dir,
            },
            "confirmation_enabled": False,
            **({"memory_conditioned_update": True}
               if self.memory_conditioned_update else {}),
        }
        base_fingerprint = sha256_text(dumps_canonical(base_identity))
        state_identity_path = os.path.join(state_dir, "bounded_smoke_identity.json")
        try:
            _write_immutable(state_identity_path, {
                "schema_version": "bounded_smoke_identity_v1",
                "base_fingerprint": base_fingerprint,
                "identity": base_identity,
            })
        except FinalIterationConflictError as exc:
            raise BoundedSmokeConflictError(
                "bounded smoke base identity conflicts with saved state") from exc

        if self.require_real_models:
            startup = build_startup_model_manifest(
                iteration_id=iteration_id,
                baseline_runner=self.baseline_runner,
                intervention_runner=self.intervention_runner,
                feedback_runner=self.feedback_runner,
                confirmation_evaluator=self.update_engine.confirmation_evaluator)
            log_startup_models(output_dir, startup)

        if requested_mode_completed_before_run and \
                _completed_mode_has_invalid_baseline_qa(requested_manifest):
            archive = _archive_invalid_qa_attempt(output_dir, video_id)
            print("INVALID_QA_ATTEMPT_ARCHIVED " + archive, flush=True)
            requested_mode_completed_before_run = False
            requested_manifest = os.path.join(
                output_dir, "mode_manifests", f"{mode.value}.json")

        if requested_mode_completed_before_run:
            requested = _read_json(requested_manifest)
            if requested.get("base_fingerprint") != base_fingerprint or \
                    requested.get("post_intervention_mode") != mode.value:
                raise BoundedSmokeConflictError(
                    "completed mode manifest conflicts with the requested run")
            qa = _read_json(os.path.join(mode_root, "qa_only.json"))
            for path_key, hash_key in (
                ("baseline_manifest_path", "baseline_manifest_hash"),
                ("intervention_manifest_path", "intervention_manifest_hash"),
            ):
                path = qa.get(path_key)
                if not path or not os.path.exists(path) or \
                        _file_hash(path) != qa.get(hash_key):
                    raise BoundedSmokeConflictError(
                        "completed QA upstream artifact is missing or changed")
            feedback_path = None
            plan_path = bank_path = router_path = None
            if MODE_ORDER[mode] >= MODE_ORDER[PostInterventionMode.FEEDBACK_ONLY]:
                feedback_mode = _read_json(os.path.join(
                    mode_root, "feedback_only.json"))
                feedback_path = feedback_mode["feedback_manifest_path"]
                if not os.path.exists(feedback_path) or \
                        _file_hash(feedback_path) != \
                        feedback_mode["feedback_manifest_hash"]:
                    raise BoundedSmokeConflictError(
                        "completed feedback artifact is missing or changed")
            if mode is PostInterventionMode.PROVISIONAL_UPDATE:
                plan_path = requested["update_plan_path"]
                bank_path = requested["provisional_bank_path"]
                router_path = requested["provisional_router_path"]
                artifact_paths = {
                    "update_plan_hash": plan_path,
                    "provisional_bank_hash": bank_path,
                    "provisional_router_hash": router_path,
                }
                if any(not os.path.exists(path) or _file_hash(path) !=
                       requested.get(hash_key)
                       for hash_key, path in artifact_paths.items()):
                    raise BoundedSmokeConflictError(
                        "completed provisional update artifact is missing or changed")
                if self.memory_conditioned_update:
                    for path_key, hash_key in (
                        ("memory_codebook_checkpoint_manifest_path",
                         "memory_codebook_checkpoint_manifest_hash"),
                        ("memory_router_checkpoint_manifest_path",
                         "memory_router_checkpoint_manifest_hash"),
                        ("router_prompt_probe_path", "router_prompt_probe_hash"),
                    ):
                        path = requested.get(path_key)
                        if not path or not os.path.exists(path) or \
                                _file_hash(path) != requested.get(hash_key):
                            raise BoundedSmokeConflictError(
                                "completed memory-update artifact is missing or changed")
            manifest_path = self._write_run_pointer(
                output_dir=output_dir, state_dir=state_dir, cache_dir=cache_dir,
                iteration_id=iteration_id, video_id=video_id, mode=mode,
                base_fingerprint=base_fingerprint,
                requested_manifest=requested_manifest)
            return BoundedSmokeResult(
                video_id=video_id, post_intervention_mode=mode,
                manifest_path=manifest_path,
                baseline_manifest_path=qa["baseline_manifest_path"],
                intervention_manifest_path=qa["intervention_manifest_path"],
                feedback_manifest_path=feedback_path,
                update_plan_path=plan_path,
                provisional_bank_path=bank_path,
                provisional_router_path=router_path,
                resumed=True)

        cache_manifest_path = os.path.join(cache_dir, "caption_cache_manifest.jsonl")
        baseline_result = self.baseline_runner.run(
            roles=roles, coverage_state=coverage_state,
            prompt_bank=prompt_bank, router_policy=router_policy,
            scaffold_policy=scaffold_policy, scaffold_contract=scaffold_contract,
            phase4_config=phase4_config,
            output_dir=os.path.join(output_dir, "baseline"),
            bounded_smoke_video_id=video_id,
            caption_cache_root=os.path.join(cache_dir, "captions"),
            cache_manifest_path=cache_manifest_path,
            **dict(baseline_kwargs))
        if tuple(baseline_result.selected_video_ids) != (video_id,) or \
                int(baseline_result.baseline_qa_count) != 3 or \
                len(baseline_result.video_manifest_paths) != 1:
            raise BoundedSmokeError("bounded baseline did not produce one video/three QAs")
        video_manifest_path = baseline_result.video_manifest_paths[0]
        video_manifest = _read_json(video_manifest_path)
        proposal_record = _read_json(video_manifest["property_proposal_path"])
        proposals = tuple(_proposal_from_json(item)
                          for item in proposal_record.get("proposals") or ())
        retrieval_paths = tuple(video_manifest.get("property_retrieval_paths") or ())
        if len(proposals) > 1 or len(retrieval_paths) != len(proposals):
            raise BoundedSmokeError(
                "bounded baseline exceeded proposal/intervention limits")

        intervention = self.intervention_runner.run(
            iteration_id=iteration_id,
            baseline_video_manifest_path=video_manifest_path,
            proposals=proposals, retrieval_artifact_paths=retrieval_paths,
            prompt_bank=prompt_bank, router_policy=router_policy,
            scaffold_policy=scaffold_policy, scaffold_contract=scaffold_contract,
            output_dir=os.path.join(output_dir, "intervention"),
            **dict(intervention_kwargs))
        intervention_manifest = _read_json(intervention.manifest_path)
        if len(intervention_manifest.get("results") or ()) > 1:
            raise BoundedSmokeError("bounded smoke produced multiple interventions")

        qa_manifest_path = _write_immutable(os.path.join(mode_root, "qa_only.json"), {
            "schema_version": "bounded_smoke_mode_manifest_v1",
            "post_intervention_mode": PostInterventionMode.QA_ONLY.value,
            "base_fingerprint": base_fingerprint,
            "baseline_manifest_path": baseline_result.manifest_path,
            "baseline_manifest_hash": _file_hash(baseline_result.manifest_path),
            "intervention_manifest_path": intervention.manifest_path,
            "intervention_manifest_hash": _file_hash(intervention.manifest_path),
            "transition_artifact_paths": tuple(
                item.get("transitions_path") for item in
                intervention_manifest.get("results") or ()
                if item.get("transitions_path")),
            "confirmation_calls": 0,
        })

        feedback_path = None
        feedback_result = None
        if MODE_ORDER[mode] >= MODE_ORDER[PostInterventionMode.FEEDBACK_ONLY]:
            feedback_result = self.feedback_runner.run(
                iteration_id=iteration_id,
                baseline_video_manifest_path=video_manifest_path,
                proposals=proposals, retrieval_artifact_paths=retrieval_paths,
                intervention_manifest_path=intervention.manifest_path,
                prompt_bank=prompt_bank,
                output_dir=os.path.join(output_dir, "feedback"))
            feedback_path = feedback_result.manifest_path
            _write_immutable(os.path.join(mode_root, "feedback_only.json"), {
                "schema_version": "bounded_smoke_mode_manifest_v1",
                "post_intervention_mode": PostInterventionMode.FEEDBACK_ONLY.value,
                "base_fingerprint": base_fingerprint,
                "qa_manifest_path": qa_manifest_path,
                "qa_manifest_hash": _file_hash(qa_manifest_path),
                "feedback_manifest_path": feedback_path,
                "feedback_manifest_hash": _file_hash(feedback_path),
                "confirmation_calls": 0,
            })

        plan_path = bank_path = router_path = None
        if mode is PostInterventionMode.PROVISIONAL_UPDATE:
            if feedback_result is None:
                raise BoundedSmokeError("provisional update requires completed feedback")
            if not self.memory_conditioned_update:
                plan = self.update_engine.aggregate_updates(
                    prompt_bank=prompt_bank, router_policy=router_policy,
                    feedback_manifest_paths=(feedback_result.manifest_path,),
                    expected_evidence_video_count=1)
                candidate_bank, candidate_router = self.update_engine.apply_update_plan(
                    prompt_bank=prompt_bank, router_policy=router_policy, plan=plan)
                update_root = os.path.join(state_dir, "provisional_update")
                plan_path = _write_immutable(
                    os.path.join(update_root, "update_plan.json"), plan)
                bank_path = _write_immutable(
                    os.path.join(update_root, "provisional_bank.json"), candidate_bank)
                router_path = _write_immutable(
                    os.path.join(update_root, "provisional_router.json"), candidate_router)
                _write_immutable(os.path.join(
                    mode_root, "provisional_update.json"), {
                        "schema_version": "bounded_smoke_mode_manifest_v1",
                        "post_intervention_mode": mode.value,
                        "base_fingerprint": base_fingerprint,
                        "feedback_manifest_path": feedback_path,
                        "feedback_manifest_hash": _file_hash(feedback_path),
                        "update_identity": sha256_json({
                            "stage": mode.value, "plan": plan,
                            "input_bank": prompt_bank.bank_version,
                            "input_router": router_policy.router_version,
                        }),
                        "update_plan_path": plan_path,
                        "update_plan_hash": _file_hash(plan_path),
                        "provisional_bank_path": bank_path,
                        "provisional_bank_hash": _file_hash(bank_path),
                        "provisional_router_path": router_path,
                        "provisional_router_hash": _file_hash(router_path),
                        "canonical_state_mutated": False,
                        "confirmation_calls": 0,
                    })
                manifest_path = self._write_run_pointer(
                    output_dir=output_dir, state_dir=state_dir,
                    cache_dir=cache_dir, iteration_id=iteration_id,
                    video_id=video_id, mode=mode,
                    base_fingerprint=base_fingerprint,
                    requested_manifest=requested_manifest)
                return BoundedSmokeResult(
                    video_id=video_id, post_intervention_mode=mode,
                    manifest_path=manifest_path,
                    baseline_manifest_path=baseline_result.manifest_path,
                    intervention_manifest_path=intervention.manifest_path,
                    feedback_manifest_path=feedback_path,
                    update_plan_path=plan_path,
                    provisional_bank_path=bank_path,
                    provisional_router_path=router_path,
                    resumed=requested_mode_completed_before_run)
            confirmed_pointer = os.path.join(state_dir, "confirmed", "current.json")
            confirmed_before = (_file_hash(confirmed_pointer)
                                if os.path.exists(confirmed_pointer) else None)
            codebook_checkpoint = self.update_engine.\
                run_memory_conditioned_codebook_checkpoint(
                    iteration_id=iteration_id, iteration_ordinal=1,
                    prompt_bank=prompt_bank,
                    baseline_video_manifest_paths=tuple(
                        baseline_result.video_manifest_paths),
                    intervention_manifest_paths=(intervention.manifest_path,),
                    feedback_manifest_paths=(feedback_result.manifest_path,),
                    output_dir=os.path.join(
                        output_dir, "memory_codebook_checkpoint"),
                    state_dir=state_dir)
            codebook_manifest_path = os.path.join(
                output_dir, "memory_codebook_checkpoint", "manifest.json")
            router_checkpoint = self.update_engine.\
                run_memory_conditioned_router_checkpoint(
                    iteration_id=iteration_id,
                    parent_router_policy=router_policy,
                    codebook_checkpoint_manifest_path=codebook_manifest_path,
                    output_dir=os.path.join(
                        output_dir, "memory_router_checkpoint"),
                    state_dir=state_dir)
            pair_artifacts = router_checkpoint["artifacts"]
            plan_path = pair_artifacts["atomic_policy_pair"]
            bank_path = pair_artifacts["provisional_codebook"]
            router_path = pair_artifacts["provisional_router"]
            probe_path = self._probe_rendered_router_prompt(
                video_manifest_path=video_manifest_path,
                candidate_bank_path=bank_path,
                candidate_router_path=router_path,
                output_dir=os.path.join(output_dir, "memory_router_checkpoint"))
            confirmed_after = (_file_hash(confirmed_pointer)
                               if os.path.exists(confirmed_pointer) else None)
            if confirmed_before != confirmed_after:
                raise BoundedSmokeError("confirmed pointer changed during bounded smoke")
            provider_calls = {}
            for name in ("llm_codebook_updater", "llm_router_updater"):
                stage = getattr(self.update_engine, name)
                provider = getattr(stage, "response_provider", None)
                provider_calls[name] = int(getattr(provider, "call_count", 0))
            _write_immutable(os.path.join(mode_root, "provisional_update.json"), {
                "schema_version": "bounded_smoke_mode_manifest_v1",
                "post_intervention_mode": mode.value,
                "base_fingerprint": base_fingerprint,
                "feedback_manifest_path": feedback_path,
                "feedback_manifest_hash": _file_hash(feedback_path),
                "update_identity": sha256_json({
                    "stage": mode.value,
                    "codebook_checkpoint": codebook_checkpoint,
                    "router_checkpoint": router_checkpoint,
                    "input_bank": prompt_bank.bank_version,
                    "input_router": router_policy.router_version,
                }),
                "update_plan_path": plan_path,
                "update_plan_hash": _file_hash(plan_path),
                "provisional_bank_path": bank_path,
                "provisional_bank_hash": _file_hash(bank_path),
                "provisional_router_path": router_path,
                "provisional_router_hash": _file_hash(router_path),
                "memory_codebook_checkpoint_manifest_path":
                    codebook_manifest_path,
                "memory_codebook_checkpoint_manifest_hash":
                    _file_hash(codebook_manifest_path),
                "memory_router_checkpoint_manifest_path": os.path.join(
                    output_dir, "memory_router_checkpoint", "manifest.json"),
                "memory_router_checkpoint_manifest_hash": _file_hash(os.path.join(
                    output_dir, "memory_router_checkpoint", "manifest.json")),
                "property_memory_snapshot_path": codebook_checkpoint["artifacts"][
                    "property_memory_snapshot"],
                "property_memory_snapshot_hash": _file_hash(
                    codebook_checkpoint["artifacts"]["property_memory_snapshot"]),
                "router_prompt_probe_path": probe_path,
                "router_prompt_probe_hash": _file_hash(probe_path),
                "rendered_router_prompt_hash": router_checkpoint[
                    "rendered_router_prompt_hash"],
                "component_update_provider_calls": provider_calls,
                "confirmed_pointer_path": confirmed_pointer,
                "confirmed_pointer_hash_before": confirmed_before,
                "confirmed_pointer_hash_after": confirmed_after,
                "canonical_state_mutated": False,
                "confirmation_calls": 0,
            })

        manifest_path = self._write_run_pointer(
            output_dir=output_dir, state_dir=state_dir, cache_dir=cache_dir,
            iteration_id=iteration_id, video_id=video_id, mode=mode,
            base_fingerprint=base_fingerprint,
            requested_manifest=requested_manifest)
        return BoundedSmokeResult(
            video_id=video_id, post_intervention_mode=mode,
            manifest_path=manifest_path,
            baseline_manifest_path=baseline_result.manifest_path,
            intervention_manifest_path=intervention.manifest_path,
            feedback_manifest_path=feedback_path,
            update_plan_path=plan_path,
            provisional_bank_path=bank_path,
            provisional_router_path=router_path,
            resumed=requested_mode_completed_before_run,
        )
