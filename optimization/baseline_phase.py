"""Reusable three-video incumbent baseline phase for Checkpoint 2."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from surrogate_rollout.captioning.candidate_captions import build_clip_index
from surrogate_rollout.captioning.history_aware_baseline import (
    DEFAULT_HISTORY_BLOCK_SECONDS,
    DEFAULT_MAX_HISTORY_CAPTIONS,
    HistoryAwareBaselineCaptionViewBuilder,
)
from surrogate_rollout.evaluation.dvd_qa import run_dvd_qa
from surrogate_rollout.optimization.iteration_state import (
    ComponentVersions,
    ProvisionalIterationState,
)
from surrogate_rollout.optimization.property_proposal import (
    NoOpPropertyProposalPolicy,
    PropertyProposalPolicy,
    VideoPropertyProposalRecord,
    VideoProposalContext,
)
from surrogate_rollout.optimization.train_roles import (
    EvidenceCoverageState,
    TrainRoleAssignment,
    records_for_video_ids,
    select_evidence_batch,
)
from surrogate_rollout.prompt_routing.offline_dry_run import run_offline_dry_run
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.routed_caption_view import RoutedCaptionViewBuilder
from surrogate_rollout.prompt_routing.schemas import (
    Phase4Config,
    PromptBankSnapshot,
    RouterPolicySnapshot,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    SegmentContext,
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.schemas import QAExample, sha256_text


def _write(path: str, value: object) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_write_text(path, dumps_canonical(value) + "\n")
    return os.path.abspath(path)


def _read_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


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


def _video_id(sample: Mapping[str, Any]) -> str:
    return str(sample.get("extra", {}).get("videoID") or sample["sample_id"])


def _caption_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("caption") or "")
    return value if isinstance(value, str) else ""


def _frame_references(clip_index: list[tuple[str, dict]]) -> dict[str, tuple[str, ...]]:
    output = {}
    for segment_id, info in clip_index:
        files = info.get("files") or info.get("frames") or ()
        if isinstance(files, (str, os.PathLike)):
            files = (files,)
        output[segment_id] = tuple(os.path.abspath(os.fspath(item)) for item in files)
    return output


def _history_snapshots(
    segment_ids: tuple[str, ...], captions: Mapping[str, Any], block_size: int,
) -> tuple[dict[str, Any], ...]:
    if block_size < 1:
        raise ValueError("history_block_size must be positive")
    output = []
    for index, segment_id in enumerate(segment_ids):
        block_start = (index // block_size) * block_size
        preceding_ids = segment_ids[block_start:index]
        preceding = tuple({"segment_id": value,
                           "caption": _caption_text(captions.get(value))}
                          for value in preceding_ids)
        identity = {"block_start_index": block_start, "preceding": preceding}
        output.append({
            "segment_id": segment_id,
            "block_index": index // block_size,
            "preceding_segment_ids": preceding_ids,
            "history": preceding,
            "history_hash": sha256_text(dumps_canonical(identity)),
            "source": "frozen_incumbent_caption_view",
        })
    return tuple(output)


def _reference_sets(result: Any) -> dict[str, tuple[str, ...]]:
    refs = result.references
    names = ("retrieved_segments", "returned_segments",
             "frame_inspected_segments", "explicitly_cited_segments",
             "consumed_segments")
    return {name: tuple(sorted(getattr(refs, name, ()) or ())) for name in names}


def _used_segments(reference_sets: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    consumed = reference_sets.get("consumed_segments") or ()
    if consumed:
        return tuple(consumed)
    fallback = set(reference_sets.get("frame_inspected_segments") or ())
    fallback.update(reference_sets.get("explicitly_cited_segments") or ())
    fallback.update(reference_sets.get("returned_segments") or ())
    return tuple(sorted(fallback))


def qa_execution_failure_reasons(value: Mapping[str, Any]) -> tuple[str, ...]:
    """Separate evaluator/runtime failure from a valid incorrect prediction."""
    reasons = []
    if value.get("errors"):
        reasons.append("runtime_or_parse_errors")
    prediction = value.get("prediction")
    if prediction is None or not str(prediction).strip():
        reasons.append("null_prediction")
    parsed = value.get("parsed_answer")
    if parsed is None or not str(parsed).strip():
        reasons.append("parse_failure")
    return tuple(reasons)


@dataclass(frozen=True)
class BaselinePhaseResult:
    run_id: str
    output_dir: str
    selected_video_ids: tuple[str, ...]
    baseline_qa_count: int
    video_manifest_paths: tuple[str, ...]
    property_proposal_paths: tuple[str, ...]
    provisional_state_path: str
    manifest_path: str
    next_coverage_state: EvidenceCoverageState
    property_retrieval_paths: tuple[str, ...] = ()
    resumed: bool = False


class BaselinePhaseRunner:
    """Materialize each selected incumbent video once, then run all its QAs."""

    def __init__(
        self,
        *,
        routing_fn: Callable[..., Any] = run_offline_dry_run,
        caption_view_builder: Any | None = None,
        qa_fn: Callable[..., Any] = run_dvd_qa,
        clip_index_fn: Callable[..., list[tuple[str, dict]]] = build_clip_index,
        proposal_policy: PropertyProposalPolicy | None = None,
        history_aware_builder: Any | None = None,
        property_retrieval_runner: Any | None = None,
    ) -> None:
        self.routing_fn = routing_fn
        self.caption_view_builder = caption_view_builder or RoutedCaptionViewBuilder()
        self.qa_fn = qa_fn
        self.clip_index_fn = clip_index_fn
        self.proposal_policy = proposal_policy or NoOpPropertyProposalPolicy()
        self.history_aware_builder = history_aware_builder
        self.property_retrieval_runner = property_retrieval_runner

    def _history_builder(self) -> Any:
        if self.history_aware_builder is None:
            self.history_aware_builder = (
                HistoryAwareBaselineCaptionViewBuilder.from_local_qwen())
        return self.history_aware_builder

    def run(
        self,
        *,
        roles: TrainRoleAssignment,
        coverage_state: EvidenceCoverageState,
        sample_loader: Callable[[int], Mapping[str, Any]],
        prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        phase4_config: Phase4Config,
        base_prompt_template: str,
        merge_prompt: str,
        output_dir: str,
        parent_confirmed_checkpoint_id: str,
        history_block_size: int | None = None,
        history_block_seconds: float = DEFAULT_HISTORY_BLOCK_SECONDS,
        max_history_captions: int = DEFAULT_MAX_HISTORY_CAPTIONS,
        source_revision: str = "unknown",
        gpu: str | None = None,
        dvd_max_iterations: int = 10,
        bounded_smoke_video_id: str | None = None,
        caption_cache_root: str | None = None,
        cache_manifest_path: str | None = None,
        worker_gpus: tuple[str, ...] | None = None,
    ) -> BaselinePhaseResult:
        if bounded_smoke_video_id is None:
            selected_ids, next_coverage = select_evidence_batch(coverage_state)
            execution_scope = "production_three_video_iteration"
        else:
            evidence_ids = {item.video_id for item in roles.evidence_videos}
            if bounded_smoke_video_id not in evidence_ids:
                raise ValueError("bounded smoke video must belong to the evidence pool")
            selected_ids = (bounded_smoke_video_id,)
            next_coverage = coverage_state
            execution_scope = "bounded_smoke_one_video"
        selected = records_for_video_ids(roles, selected_ids)
        output_dir = os.path.abspath(output_dir)
        versions = ComponentVersions(
            bank_version=prompt_bank.bank_version,
            router_version=router_policy.router_version,
            scaffold_version=scaffold_policy.scaffold_version,
            contract_version=scaffold_contract.contract_version,
        )
        input_identity = {
            "selected_video_ids": selected_ids,
            "coverage_state": coverage_state,
            "components": {
                "prompt_bank": prompt_bank, "router_policy": router_policy,
                "scaffold_policy": scaffold_policy,
                "scaffold_contract": scaffold_contract,
            },
            "phase4_config": {
                key: value for key, value in as_json_dict(phase4_config).items()
                if key != "post_intervention_mode"
            },
            "base_prompt_template_hash": sha256_text(base_prompt_template),
            "merge_prompt_hash": sha256_text(merge_prompt),
            "source_revision": source_revision,
            "proposal_policy_version": self.proposal_policy.policy_version,
            "parent_confirmed_checkpoint_id": parent_confirmed_checkpoint_id,
            "property_retrieval_policy_version": getattr(
                self.property_retrieval_runner, "policy_version", None),
        }
        if bounded_smoke_video_id is not None:
            input_identity["execution_scope"] = execution_scope
        if caption_cache_root is not None or cache_manifest_path is not None:
            input_identity["caption_cache_root"] = (
                os.path.abspath(caption_cache_root) if caption_cache_root else None)
            input_identity["cache_manifest_path"] = (
                os.path.abspath(cache_manifest_path) if cache_manifest_path else None)
        if worker_gpus is not None:
            input_identity["worker_gpus"] = tuple(worker_gpus)
        if router_policy.policy_type == "history_aware_vlm":
            input_identity["history_policy"] = {
                "mode": "sequential_vlm",
                "history_block_seconds": history_block_seconds,
                "max_history_captions": max_history_captions,
            }
        else:
            # Preserve the exact Checkpoint 2 fingerprint for old completed runs.
            input_identity["history_block_size"] = history_block_size
        if hasattr(self.proposal_policy, "configuration_identity"):
            input_identity["proposal_policy_configuration"] = (
                self.proposal_policy.configuration_identity)
        if self.property_retrieval_runner is not None and hasattr(
                self.property_retrieval_runner, "configuration_identity"):
            input_identity["property_retrieval_configuration"] = (
                self.property_retrieval_runner.configuration_identity)
        input_fingerprint = sha256_text(dumps_canonical(input_identity))
        run_id = "baseline_" + input_fingerprint[:24]
        manifest_path = os.path.join(output_dir, "manifest.json")
        if os.path.exists(manifest_path):
            manifest = _read_json(manifest_path)
            if manifest.get("input_fingerprint") != input_fingerprint:
                raise RuntimeError("baseline output exists for different frozen inputs")
            if manifest.get("status") != "completed":
                raise RuntimeError("baseline manifest exists but is not completed")
            return BaselinePhaseResult(
                run_id=manifest["run_id"], output_dir=output_dir,
                selected_video_ids=tuple(manifest["selected_video_ids"]),
                baseline_qa_count=int(manifest["baseline_qa_count"]),
                video_manifest_paths=tuple(manifest["video_manifest_paths"]),
                property_proposal_paths=tuple(manifest["property_proposal_paths"]),
                provisional_state_path=manifest["provisional_state_path"],
                manifest_path=manifest_path,
                next_coverage_state=_coverage_from_json(
                    manifest["next_coverage_state"]),
                property_retrieval_paths=tuple(
                    manifest.get("property_retrieval_paths") or ()),
                resumed=True)

        _write(os.path.join(output_dir, "policy_snapshot", "prompt_bank.json"),
               prompt_bank)
        _write(os.path.join(output_dir, "policy_snapshot", "router_policy.json"),
               router_policy)
        _write(os.path.join(output_dir, "policy_snapshot", "scaffold_policy.json"),
               scaffold_policy)
        _write(os.path.join(output_dir, "policy_snapshot", "scaffold_contract.json"),
               scaffold_contract)
        _write(os.path.join(output_dir, "coverage", "before.json"), coverage_state)
        _write(os.path.join(output_dir, "coverage", "after.json"), next_coverage)

        video_manifests = []
        proposal_paths = []
        total_qas = 0
        total_router_calls = 0
        retrieval_paths: list[str] = []
        for record in selected:
            video_dir = os.path.join(output_dir, "baseline", record.video_id)
            complete_path = os.path.join(video_dir, "video_complete.json")
            video_fingerprint = sha256_text(dumps_canonical({
                "run": input_fingerprint, "video": record,
            }))
            if os.path.exists(complete_path):
                complete = _read_json(complete_path)
                if complete.get("video_fingerprint") != video_fingerprint:
                    raise RuntimeError(
                        f"completed video has stale inputs: {record.video_id}")
                required = (
                    "routing_manifest_path", "caption_view_path", "captions_path",
                    "frames_path", "frozen_histories_path", "baseline_qas_path",
                    "property_proposal_path",
                )
                if self.property_retrieval_runner is not None:
                    required += ("property_retrieval_manifest_path",)
                missing = [name for name in required
                           if not os.path.exists(complete.get(name, ""))]
                missing.extend(
                    f"property_retrieval:{path}"
                    for path in complete.get("property_retrieval_paths") or ()
                    if not os.path.exists(path))
                if missing:
                    raise RuntimeError(
                        f"completed video is missing artifacts: {record.video_id}: "
                        f"{missing}")
                video_manifests.append(os.path.abspath(complete_path))
                proposal_paths.append(complete["property_proposal_path"])
                total_qas += int(complete["qa_count"])
                total_router_calls += int(complete.get("real_vlm_router_calls", 0))
                retrieval_paths.extend(
                    complete.get("property_retrieval_paths") or ())
                continue

            samples = tuple(sample_loader(index) for index in record.provider_indices)
            if any(_video_id(sample) != record.video_id for sample in samples):
                raise RuntimeError(f"provider video changed for {record.video_id}")
            clip_index = self.clip_index_fn(samples[0], record.video_id)
            segment_ids = tuple(segment_id for segment_id, _ in clip_index)
            frames = _frame_references(clip_index)
            if router_policy.policy_type == "history_aware_vlm":
                caption_view = self._history_builder().build(
                    sample=dict(samples[0]),
                    clip_index=clip_index,
                    prompt_bank=prompt_bank,
                    router_policy=router_policy,
                    scaffold_policy=scaffold_policy,
                    scaffold_contract=scaffold_contract,
                    base_prompt_template=base_prompt_template,
                    merge_prompt=merge_prompt,
                    work_root=os.path.join(video_dir, "history_aware_baseline"),
                    history_block_seconds=history_block_seconds,
                    max_history_captions=max_history_captions,
                    candidate_cache_root=caption_cache_root,
                    cache_manifest_path=cache_manifest_path,
                    worker_gpus=worker_gpus,
                )
                routing_manifest_path = caption_view.routing_manifest_path
                frames_path = caption_view.frames_path
                histories_path = caption_view.frozen_histories_path
                histories = caption_view.histories
            else:
                if history_block_size is None:
                    history_block_size = max_history_captions
                contexts = tuple(SegmentContext(
                    video_id=record.video_id,
                    segment_id=segment_id,
                    segment_features={"frame_references": frames[segment_id]},
                    metadata={"checkpoint": 2,
                              "history_router_input": "deferred"},
                ) for segment_id in segment_ids)
                routing = self.routing_fn(
                    contexts=contexts, prompt_bank=prompt_bank,
                    router_policy=router_policy,
                    scaffold_policy=scaffold_policy,
                    scaffold_contract=scaffold_contract,
                    phase4_config=phase4_config,
                    base_prompt_template=base_prompt_template,
                    output_dir=os.path.join(video_dir, "routing"),
                    source_revision=source_revision)
                prompt_map = {item.segment_id: item
                              for item in routing.composed_prompts}
                caption_view = self.caption_view_builder.build(
                    sample=dict(samples[0]),
                    segment_composed_prompt_map=prompt_map,
                    work_root=os.path.join(video_dir, "caption_view"),
                    merge_prompt=merge_prompt)
                with open(caption_view.captions_path) as f:
                    legacy_captions = json.load(f)
                histories = _history_snapshots(
                    tuple(caption_view.segment_ids), legacy_captions,
                    history_block_size)
                frames_path = _write(
                    os.path.join(video_dir, "frames.json"), frames)
                histories_path = os.path.join(
                    video_dir, "frozen_histories.jsonl")
                _atomic_write_text(histories_path, "".join(
                    dumps_canonical(item) + "\n" for item in histories))
                routing_manifest_path = routing.manifest_path

            with open(caption_view.captions_path) as f:
                captions = json.load(f)

            qa_rows = []
            for question_id, provider_index, sample in zip(
                    record.question_ids, record.provider_indices, samples):
                example = QAExample(
                    video_id=record.video_id, question_id=question_id,
                    question=str(sample["question"]),
                    ground_truth=str(sample.get("answer") or ""),
                    provider_index=provider_index,
                    options=tuple(sample.get("options") or ()),
                    subtitle_available=bool(
                        sample.get("extra", {}).get("subtitle_path")),
                )
                qa_dir = os.path.join(
                    video_dir, "qa", question_id.replace("/", "-"))
                result = self.qa_fn(
                    captions_path=caption_view.captions_path,
                    sample=dict(sample), run_dir=qa_dir,
                    question_id=question_id,
                    database_path=caption_view.database_path,
                    max_iterations=dvd_max_iterations, gpu=gpu)
                references = _reference_sets(result)
                row = {
                    "video_id": record.video_id,
                    "question_id": question_id,
                    "provider_index": provider_index,
                    "question": example.question,
                    "options": example.options,
                    "ground_truth": result.ground_truth,
                    "prediction": result.prediction,
                    "parsed_answer": result.parsed_answer,
                    "score": float(result.score),
                    "is_correct": float(result.score) > 0.0,
                    "used_segments": _used_segments(references),
                    "reference_sets": references,
                    "trajectory_path": os.path.join(qa_dir, "trajectory.jsonl"),
                    "references_path": os.path.join(qa_dir, "references.json"),
                    "result_path": os.path.join(qa_dir, "result.json"),
                    "errors": tuple(result.errors),
                    "latency_seconds": float(result.latency_seconds),
                }
                _write(os.path.join(qa_dir, "baseline_record.json"), row)
                qa_rows.append(row)
            qa_path = os.path.join(video_dir, "baseline_qas.jsonl")
            _atomic_write_text(qa_path, "".join(
                dumps_canonical(item) + "\n" for item in qa_rows))

            qa_failures = tuple({
                "question_id": item["question_id"],
                "reasons": qa_execution_failure_reasons(item),
                "errors": item["errors"],
            } for item in qa_rows if qa_execution_failure_reasons(item))
            if qa_failures:
                failure_path = _write(os.path.join(
                    video_dir, "baseline_qa_failures.json"), {
                        "schema_version": "baseline_qa_execution_failure_v1",
                        "status": "failed_before_property_proposal",
                        "video_id": record.video_id,
                        "qa_failures": qa_failures,
                        "property_proposer_called": False,
                    })
                raise RuntimeError(
                    "baseline QA execution failed before property proposal: "
                    + failure_path)

            proposal_context = VideoProposalContext(
                video_id=record.video_id, baseline_run_id=run_id,
                baseline_qa_results=tuple(MappingProxyType(item) for item in qa_rows),
                captions=MappingProxyType(captions),
                frame_references=MappingProxyType(frames),
                frozen_histories=tuple(MappingProxyType(item) for item in histories),
                prompt_bank=prompt_bank,
                proposal_artifact_dir=os.path.join(
                    output_dir, "property_proposals", record.video_id,
                    "model_artifacts"))
            proposals = tuple(self.proposal_policy.propose(proposal_context))
            proposal_record = VideoPropertyProposalRecord(
                video_id=record.video_id, baseline_run_id=run_id,
                baseline_qa_ids=record.question_ids, proposals=proposals,
                proposer_policy_version=self.proposal_policy.policy_version)
            proposal_path = _write(os.path.join(
                output_dir, "property_proposals", f"{record.video_id}.json"),
                proposal_record)
            video_retrieval_paths: tuple[str, ...] = ()
            retrieval_manifest_path: str | None = None
            if self.property_retrieval_runner is not None:
                retrieval_result = self.property_retrieval_runner.run(
                    sample=dict(samples[0]), proposals=proposals,
                    output_dir=os.path.join(
                        output_dir, "property_retrieval", record.video_id))
                video_retrieval_paths = retrieval_result.retrieval_artifact_paths
                retrieval_manifest_path = retrieval_result.manifest_path
            complete = {
                "video_id": record.video_id,
                "video_fingerprint": video_fingerprint,
                "qa_count": len(qa_rows),
                "routing_manifest_path": routing_manifest_path,
                "caption_view_path": caption_view.routed_view_path,
                "captions_path": caption_view.captions_path,
                "frames_path": frames_path,
                "frozen_histories_path": os.path.abspath(histories_path),
                "baseline_qas_path": os.path.abspath(qa_path),
                "property_proposal_path": proposal_path,
                "property_retrieval_paths": video_retrieval_paths,
                "property_retrieval_manifest_path": retrieval_manifest_path,
                "caption_materializations": 1,
                "history_mode": ("sequential_vlm" if
                                 router_policy.policy_type == "history_aware_vlm"
                                 else "legacy_posthoc"),
                "real_vlm_router_calls": int(getattr(
                    caption_view, "router_call_count", 0)),
                "component_versions": as_json_dict(versions),
            }
            _write(complete_path, complete)
            video_manifests.append(os.path.abspath(complete_path))
            proposal_paths.append(proposal_path)
            retrieval_paths.extend(video_retrieval_paths)
            total_qas += len(qa_rows)
            total_router_calls += int(getattr(
                caption_view, "router_call_count", 0))

        expected_qas = 3 * len(selected_ids)
        if total_qas != expected_qas:
            raise RuntimeError(
                f"baseline phase requires {expected_qas} QAs, got {total_qas}")
        before_hash = sha256_text(dumps_canonical(coverage_state))
        after_hash = sha256_text(dumps_canonical(next_coverage))
        if bounded_smoke_video_id is None:
            provisional = ProvisionalIterationState(
                iteration_id=run_id,
                coverage_cycle=coverage_state.coverage_cycle,
                selected_video_ids=selected_ids,
                parent_confirmed_checkpoint_id=parent_confirmed_checkpoint_id,
                input_versions=versions, provisional_versions=versions,
                baseline_manifest_path=manifest_path,
                property_proposal_paths=tuple(proposal_paths),
                coverage_state_before_hash=before_hash,
                coverage_state_after_hash=after_hash,
                property_retrieval_paths=tuple(retrieval_paths))
        else:
            provisional = {
                "schema_version": "bounded_smoke_baseline_state_v1",
                "status": "baseline_completed",
                "run_id": run_id,
                "selected_video_id": bounded_smoke_video_id,
                "parent_input_id": parent_confirmed_checkpoint_id,
                "component_versions": versions,
                "baseline_manifest_path": os.path.abspath(manifest_path),
                "property_proposal_paths": tuple(proposal_paths),
                "property_retrieval_paths": tuple(retrieval_paths),
                "production_coverage_state_mutated": False,
                "coverage_state_hash": before_hash,
            }
        state_name = ("provisional.json" if bounded_smoke_video_id is None
                      else "bounded_baseline_state.json")
        provisional_path = _write(os.path.join(
            output_dir, "iteration_state", state_name), provisional)
        manifest = {
            "schema_version": (
                "phase4_checkpoint3a_bounded_smoke_baseline_v1"
                if bounded_smoke_video_id is not None else
                "phase4_checkpoint3a_baseline_v1"
                if router_policy.policy_type == "history_aware_vlm"
                else "phase4_checkpoint2_baseline_v1"),
            "status": "completed", "run_id": run_id,
            "input_fingerprint": input_fingerprint,
            "selected_video_ids": selected_ids,
            "baseline_qa_count": total_qas,
            "video_manifest_paths": tuple(video_manifests),
            "property_proposal_paths": tuple(proposal_paths),
            "property_retrieval_paths": tuple(retrieval_paths),
            "provisional_state_path": provisional_path,
            "next_coverage_state": next_coverage,
            "calls": {
                "video_caption_materializations": len(selected_ids),
                "dvd_reasoning": expected_qas,
                "property_proposal_policy": len(selected_ids),
                "real_vlm_router": total_router_calls,
                "interventional_feedback_models": 0,
                "confirmation_evaluations": 0,
            },
        }
        _write(manifest_path, manifest)
        return BaselinePhaseResult(
            run_id=run_id, output_dir=output_dir,
            selected_video_ids=selected_ids, baseline_qa_count=total_qas,
            video_manifest_paths=tuple(video_manifests),
            property_proposal_paths=tuple(proposal_paths),
            property_retrieval_paths=tuple(retrieval_paths),
            provisional_state_path=provisional_path,
            manifest_path=os.path.abspath(manifest_path),
            next_coverage_state=next_coverage)
