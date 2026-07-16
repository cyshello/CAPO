"""Independent property-wise selective interventions for Checkpoint 3C."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from surrogate_rollout import config
from surrogate_rollout.captioning.candidate_captions import build_clip_index
from surrogate_rollout.evaluation.dvd_qa import run_dvd_qa
from surrogate_rollout.mixed_views.builder import MixedViewBuilder
from surrogate_rollout.optimization.property_proposal import CandidatePropertyProposal
from surrogate_rollout.prompt_routing.intervention_composer import (
    InterventionCompositionError,
    compose_forced_property,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.scaffold_applier import read_composed_prompts
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    RouterPolicySnapshot,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_json, sha256_text


TRANSITION_LABELS = (
    "wrong_to_correct", "correct_to_wrong",
    "correct_to_correct", "wrong_to_wrong",
)


class PropertyInterventionError(RuntimeError):
    pass


class PropertyInterventionConflictError(PropertyInterventionError):
    pass


@dataclass(frozen=True)
class PropertyInterventionWorkItem:
    iteration_id: str
    source_video_id: str
    candidate_property_id: str
    property_text_hash: str
    retrieval_artifact_path: str
    output_dir: str


@dataclass(frozen=True)
class PropertyInterventionResult:
    candidate_property_id: str
    source_video_id: str
    status: str
    result_path: str | None
    mixed_captions_path: str | None = None
    transitions_path: str | None = None
    failure_type: str | None = None
    failure_reason: str | None = None
    resumed: bool = False


@dataclass(frozen=True)
class PropertyInterventionBatchResult:
    video_id: str
    manifest_path: str
    results: tuple[PropertyInterventionResult, ...]


def correctness_transition(before: bool, after: bool) -> str:
    if not before and after:
        return "wrong_to_correct"
    if before and not after:
        return "correct_to_wrong"
    if before and after:
        return "correct_to_correct"
    return "wrong_to_wrong"


def _read_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: str, value: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, json.dumps(
        as_json_dict(value), sort_keys=True, ensure_ascii=False, indent=2) + "\n")
    return os.path.abspath(path)


def _write_jsonl(path: str, rows: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, "".join(dumps_canonical(row) + "\n" for row in rows))
    return os.path.abspath(path)


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_id(sample: Mapping[str, Any]) -> str:
    return str(sample.get("extra", {}).get("videoID") or sample.get("sample_id") or "")


def _reference_sets(result: Any) -> dict[str, tuple[str, ...]]:
    refs = getattr(result, "references", None)
    names = ("retrieved_segments", "returned_segments",
             "frame_inspected_segments", "explicitly_cited_segments",
             "consumed_segments")
    if refs is None:
        return {name: () for name in names}
    return {name: tuple(sorted(getattr(refs, name, ()) or ())) for name in names}


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_"
                   for char in value)


class PropertyInterventionBatchRunner:
    policy_version = "property_selective_intervention_v1"

    def __init__(
        self,
        *,
        segment_captioner: Any,
        qa_fn: Callable[..., Any] = run_dvd_qa,
        clip_index_fn: Callable[..., list[tuple[str, dict]]] = build_clip_index,
        mixed_view_builder: MixedViewBuilder | None = None,
        caption_cache_root: str | None = None,
        caption_cache_manifest_path: str | None = None,
    ) -> None:
        self.segment_captioner = segment_captioner
        self.qa_fn = qa_fn
        self.clip_index_fn = clip_index_fn
        self.mixed_view_builder = mixed_view_builder or MixedViewBuilder()
        self.caption_cache_root = caption_cache_root
        self.caption_cache_manifest_path = caption_cache_manifest_path

    def run(
        self,
        *,
        iteration_id: str,
        baseline_video_manifest_path: str,
        proposals: tuple[CandidatePropertyProposal, ...],
        retrieval_artifact_paths: tuple[str, ...],
        sample_loader: Callable[[int], Mapping[str, Any]],
        prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        base_prompt_template: str,
        merge_prompt: str,
        output_dir: str,
        gpu: str | None = None,
        dvd_max_iterations: int = 10,
    ) -> PropertyInterventionBatchResult:
        baseline = _read_json(baseline_video_manifest_path)
        video_id = str(baseline["video_id"])
        if any(item.source_video_id != video_id for item in proposals):
            raise PropertyInterventionError("proposal source-video lineage mismatch")
        if len({item.candidate_property_id for item in proposals}) != len(proposals):
            raise PropertyInterventionError("duplicate candidate property work items")
        retrievals = {}
        for path in retrieval_artifact_paths:
            artifact = _read_json(path)
            candidate_id = artifact.get("candidate_property_id")
            if candidate_id in retrievals:
                raise PropertyInterventionError("duplicate retrieval candidate ID")
            retrievals[candidate_id] = (os.path.abspath(path), artifact)
        if set(retrievals) != {item.candidate_property_id for item in proposals}:
            raise PropertyInterventionError(
                "proposal and retrieval candidate sets do not match")

        routing_manifest = _read_json(baseline["routing_manifest_path"])
        composed_path = routing_manifest["composed_prompts_path"]
        cache_keys_path = routing_manifest["caption_cache_keys_path"]
        parent_identity = sha256_json({
            "video_fingerprint": baseline["video_fingerprint"],
            "component_versions": baseline["component_versions"],
            "captions_hash": _file_hash(baseline["captions_path"]),
            "histories_hash": _file_hash(baseline["frozen_histories_path"]),
            "composed_prompts_hash": _file_hash(composed_path),
            "caption_cache_keys_hash": _file_hash(cache_keys_path),
            "baseline_qas_hash": _file_hash(baseline["baseline_qas_path"]),
        })
        output_dir = os.path.abspath(output_dir)
        work_items = tuple(self._work_item(
            iteration_id=iteration_id, video_id=video_id, proposal=item,
            retrieval_path=retrievals[item.candidate_property_id][0],
            output_dir=output_dir) for item in proposals)
        caption_configuration = self._caption_configuration()
        qa_configuration = self._qa_configuration(dvd_max_iterations)
        batch_identity = {
            "policy_version": self.policy_version,
            "iteration_id": iteration_id,
            "source_video_id": video_id,
            "parent_baseline_identity": parent_identity,
            "work_items": as_json_dict(work_items),
            "retrieval_fingerprints": tuple(
                retrievals[item.candidate_property_id][1].get("input_fingerprint")
                for item in proposals),
            "prompt_bank_version": prompt_bank.bank_version,
            "router_version": router_policy.router_version,
            "scaffold_version": scaffold_policy.scaffold_version,
            "contract_version": scaffold_contract.contract_version,
            "base_prompt_hash": sha256_text(base_prompt_template),
            "merge_prompt_hash": sha256_text(merge_prompt),
            "caption_configuration": caption_configuration,
            "qa_configuration": qa_configuration,
        }
        batch_fingerprint = sha256_json(batch_identity)
        manifest_target = os.path.join(output_dir, "manifest.json")
        if os.path.exists(manifest_target):
            existing_manifest = _read_json(manifest_target)
            if existing_manifest.get("input_fingerprint") != batch_fingerprint:
                raise PropertyInterventionConflictError(
                    "intervention batch output exists for different frozen inputs")
        results = []
        for work_item, proposal in zip(work_items, proposals):
            try:
                result = self._run_one(
                    work_item=work_item, proposal=proposal,
                    retrieval=retrievals[proposal.candidate_property_id][1],
                    baseline=baseline, routing_manifest=routing_manifest,
                    parent_baseline_identity=parent_identity,
                    sample_loader=sample_loader, prompt_bank=prompt_bank,
                    router_policy=router_policy,
                    scaffold_policy=scaffold_policy,
                    scaffold_contract=scaffold_contract,
                    base_prompt_template=base_prompt_template,
                    merge_prompt=merge_prompt, gpu=gpu,
                    dvd_max_iterations=dvd_max_iterations)
            except Exception as exc:
                result = PropertyInterventionResult(
                    candidate_property_id=proposal.candidate_property_id,
                    source_video_id=video_id, status="failed", result_path=None,
                    failure_type=type(exc).__name__, failure_reason=str(exc))
            results.append(result)

        manifest_path = _write_json(os.path.join(output_dir, "manifest.json"), {
            "schema_version": "property_intervention_batch_v1",
            "status": "completed_with_failures" if any(
                item.status == "failed" for item in results) else "completed",
            "iteration_id": iteration_id,
            "source_video_id": video_id,
            "parent_baseline_identity": parent_identity,
            "input_fingerprint": batch_fingerprint,
            "work_items": work_items,
            "results": results,
        })
        return PropertyInterventionBatchResult(
            video_id=video_id, manifest_path=manifest_path,
            results=tuple(results))

    def _work_item(self, *, iteration_id: str, video_id: str,
                   proposal: CandidatePropertyProposal, retrieval_path: str,
                   output_dir: str) -> PropertyInterventionWorkItem:
        text_hash = sha256_text(proposal.property_text)
        directory = os.path.join(
            output_dir, video_id,
            f"{_safe_id(proposal.candidate_property_id)}_p{text_hash[:12]}")
        return PropertyInterventionWorkItem(
            iteration_id=iteration_id, source_video_id=video_id,
            candidate_property_id=proposal.candidate_property_id,
            property_text_hash=text_hash,
            retrieval_artifact_path=retrieval_path,
            output_dir=directory)

    def _run_one(
        self,
        *,
        work_item: PropertyInterventionWorkItem,
        proposal: CandidatePropertyProposal,
        retrieval: dict[str, Any],
        baseline: dict[str, Any],
        routing_manifest: dict[str, Any],
        parent_baseline_identity: str,
        sample_loader: Callable[[int], Mapping[str, Any]],
        prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        base_prompt_template: str,
        merge_prompt: str,
        gpu: str | None,
        dvd_max_iterations: int,
    ) -> PropertyInterventionResult:
        if retrieval.get("source_video_id") != work_item.source_video_id or \
                retrieval.get("candidate_property_id") != proposal.candidate_property_id:
            raise PropertyInterventionError("retrieval lineage mismatch")
        if retrieval.get("property_text") != proposal.property_text:
            raise PropertyInterventionError("retrieval property text mismatch")
        selected = tuple(retrieval.get("s_sim") or ())
        if not selected or len(selected) != len(set(selected)):
            raise PropertyInterventionError("S_sim must be non-empty and unique")

        incumbent_prompts = {
            item.segment_id: item for item in read_composed_prompts(
                routing_manifest["composed_prompts_path"])}
        histories = {item["segment_id"]: item for item in _read_jsonl(
            baseline["frozen_histories_path"])}
        unknown = ((set(selected) - set(incumbent_prompts)) |
                   (set(selected) - set(histories)))
        if unknown:
            raise PropertyInterventionError(
                f"selected segments missing baseline state: {sorted(unknown)}")
        frozen = []
        composed = []
        composition_error = None
        try:
            for segment_id in selected:
                history = histories[segment_id]
                if sha256_text(history["serialized_history"]) != history["history_hash"]:
                    raise PropertyInterventionError(
                        f"frozen history hash mismatch for {segment_id}")
                frozen.append(history)
                composed.append(compose_forced_property(
                    incumbent=incumbent_prompts[segment_id], candidate=proposal,
                    prompt_bank=prompt_bank, router_policy=router_policy,
                    scaffold_policy=scaffold_policy,
                    scaffold_contract=scaffold_contract,
                    base_prompt_template=base_prompt_template))
        except (InterventionCompositionError, PropertyInterventionError) as exc:
            composition_error = f"{type(exc).__name__}: {exc}"

        caption_configuration = self._caption_configuration()
        qa_configuration = self._qa_configuration(dvd_max_iterations)
        work_identity = {
            "policy_version": self.policy_version,
            "iteration_id": work_item.iteration_id,
            "source_video_id": work_item.source_video_id,
            "candidate_property_id": proposal.candidate_property_id,
            "property_text_hash": work_item.property_text_hash,
            "parent_baseline_identity": parent_baseline_identity,
            "retrieval_identity": retrieval.get("input_fingerprint")
            or sha256_json(retrieval.get("retrieval_identity") or {}),
            "selected_segment_ids": selected,
            "frozen_history_hashes": tuple(item["history_hash"] for item in frozen),
            "composed_prompt_hashes": tuple(item.prompt_hash for item in composed),
            "composition_error": composition_error,
            "caption_configuration": caption_configuration,
            "qa_configuration": qa_configuration,
            "base_prompt_hash": sha256_text(base_prompt_template),
            "merge_prompt_hash": sha256_text(merge_prompt),
            "component_versions": baseline["component_versions"],
        }
        fingerprint = sha256_json(work_identity)
        result_path = os.path.join(work_item.output_dir, "result.json")
        if os.path.exists(result_path):
            existing = _read_json(result_path)
            if existing.get("input_fingerprint") != fingerprint:
                raise PropertyInterventionConflictError(
                    "candidate output exists for different frozen inputs")
            if existing["status"] == "completed":
                required = (existing.get("mixed_captions_path"),
                            existing.get("transitions_path"))
                if not all(path and os.path.exists(path) for path in required):
                    raise PropertyInterventionError(
                        "completed candidate is missing required artifacts")
            return PropertyInterventionResult(
                candidate_property_id=proposal.candidate_property_id,
                source_video_id=work_item.source_video_id,
                status=existing["status"], result_path=os.path.abspath(result_path),
                mixed_captions_path=existing.get("mixed_captions_path"),
                transitions_path=existing.get("transitions_path"),
                failure_type=existing.get("failure_type"),
                failure_reason=existing.get("failure_reason"), resumed=True)

        os.makedirs(work_item.output_dir, exist_ok=True)
        _write_json(os.path.join(work_item.output_dir, "work_item.json"), {
            "work_item": work_item, "input_fingerprint": fingerprint,
            "identity": work_identity,
        })
        if composition_error:
            return self._persist_failure(
                work_item, fingerprint, "composition_failed", composition_error)
        _write_jsonl(os.path.join(
            work_item.output_dir, "composed_prompts.jsonl"), composed)
        _write_jsonl(os.path.join(
            work_item.output_dir, "frozen_histories.jsonl"), frozen)

        baseline_qas = _read_jsonl(baseline["baseline_qas_path"])
        if len(baseline_qas) != 3:
            return self._persist_failure(
                work_item, fingerprint, "invalid_baseline",
                "source video does not have exactly three baseline QAs")
        first_sample = dict(sample_loader(int(baseline_qas[0]["provider_index"])))
        if _video_id(first_sample) != work_item.source_video_id:
            return self._persist_failure(
                work_item, fingerprint, "source_video_mismatch",
                "sample loader returned a different source video")
        clip_index = self.clip_index_fn(first_sample, work_item.source_video_id)
        clip_by_id = dict(clip_index)
        if set(selected) - set(clip_by_id):
            return self._persist_failure(
                work_item, fingerprint, "invalid_selection",
                "S_sim contains a segment outside the source video")

        parsed = {}
        cache_rows = []
        composed_by_id = {item.segment_id: item for item in composed}
        history_by_id = {item["segment_id"]: item for item in frozen}
        try:
            for segment_id in selected:
                intervention_identity = sha256_json({
                    "candidate_property_id": proposal.candidate_property_id,
                    "property_text_hash": work_item.property_text_hash,
                    "parent_baseline_identity": parent_baseline_identity,
                    "retrieval_identity": work_identity["retrieval_identity"],
                    "composed_prompt_hash": composed_by_id[segment_id].prompt_hash,
                    "history_hash": history_by_id[segment_id]["history_hash"],
                    "caption_configuration": caption_configuration,
                })
                result = self.segment_captioner.caption(
                    sample=first_sample, video_id=work_item.source_video_id,
                    segment_id=segment_id, clip_info=clip_by_id[segment_id],
                    composed_prompt=composed_by_id[segment_id],
                    history_snapshot=history_by_id[segment_id],
                    merge_prompt=merge_prompt,
                    cache_root=self.caption_cache_root,
                    cache_manifest_path=self.caption_cache_manifest_path,
                    intervention_identity_hash=intervention_identity)
                value = dict(result.parsed)
                if not value.get("clip_description"):
                    raise PropertyInterventionError(
                        f"selected segment {segment_id} returned no valid caption")
                if result.cache_key.get("intervention_identity_hash") != \
                        intervention_identity:
                    raise PropertyInterventionError(
                        f"selected segment {segment_id} cache identity mismatch")
                parsed[segment_id] = value
                cache_rows.append({
                    "segment_id": segment_id,
                    "history_hash": history_by_id[segment_id]["history_hash"],
                    "composed_prompt_hash": composed_by_id[segment_id].prompt_hash,
                    "intervention_identity_hash": intervention_identity,
                    "cache_key": result.cache_key,
                    "cache_dir": result.cache_dir,
                    "cache_hit": result.cache_hit,
                    "caption_seconds": result.caption_seconds,
                })
        except Exception as exc:
            return self._persist_failure(
                work_item, fingerprint, "selected_segment_caption_failed",
                f"{type(exc).__name__}: {exc}")
        _write_jsonl(os.path.join(
            work_item.output_dir, "caption_cache_keys.jsonl"), cache_rows)

        try:
            registries = self._baseline_registries(
                routing_manifest["caption_cache_keys_path"])
            mixed = self.mixed_view_builder.build(
                baseline_captions_path=baseline["captions_path"],
                candidate_captions=parsed,
                selected_clip_ids=set(selected),
                work_root=os.path.join(work_item.output_dir, "mixed_view"),
                video_id=work_item.source_video_id,
                baseline_registries=registries,
                view_name=(f"candidate_{_safe_id(proposal.candidate_property_id)}_"
                           f"p{work_item.property_text_hash[:12]}"))
            if set(mixed.replaced_clip_ids) != set(selected):
                raise PropertyInterventionError(
                    "mixed view did not replace every selected segment")
        except Exception as exc:
            return self._persist_failure(
                work_item, fingerprint, "mixed_view_failed",
                f"{type(exc).__name__}: {exc}")

        transition_rows = []
        try:
            for baseline_qa in baseline_qas:
                provider_index = int(baseline_qa["provider_index"])
                sample = dict(sample_loader(provider_index))
                if _video_id(sample) != work_item.source_video_id:
                    raise PropertyInterventionError(
                        "QA sample source-video lineage mismatch")
                question_id = str(baseline_qa["question_id"])
                qa_dir = os.path.join(
                    work_item.output_dir, "qa", _safe_id(question_id))
                qa_identity = sha256_json({
                    "work_fingerprint": fingerprint,
                    "captions_hash": mixed.captions_hash,
                    "question_id": question_id,
                    "provider_index": provider_index,
                })
                record_path = os.path.join(qa_dir, "intervention_record.json")
                if os.path.exists(record_path):
                    row = _read_json(record_path)
                    if row.get("qa_identity") != qa_identity:
                        raise PropertyInterventionConflictError(
                            "QA artifact exists for different candidate inputs")
                else:
                    result = self.qa_fn(
                        captions_path=mixed.captions_path, sample=sample,
                        run_dir=qa_dir, question_id=question_id,
                        database_path=mixed.database_path,
                        max_iterations=dvd_max_iterations, gpu=gpu)
                    candidate_correct = float(result.score) > 0.0
                    row = {
                        "qa_identity": qa_identity,
                        "candidate_property_id": proposal.candidate_property_id,
                        "source_video_id": work_item.source_video_id,
                        "question_id": question_id,
                        "provider_index": provider_index,
                        "baseline_prediction": baseline_qa.get("prediction"),
                        "candidate_prediction": result.prediction,
                        "ground_truth": result.ground_truth,
                        "baseline_correct": bool(baseline_qa["is_correct"]),
                        "candidate_correct": candidate_correct,
                        "transition": correctness_transition(
                            bool(baseline_qa["is_correct"]), candidate_correct),
                        "reference_sets": _reference_sets(result),
                        "trajectory_path": os.path.join(qa_dir, "trajectory.jsonl"),
                        "result_path": os.path.join(qa_dir, "result.json"),
                        "errors": tuple(result.errors),
                        "latency_seconds": float(result.latency_seconds),
                    }
                    _write_json(record_path, row)
                transition_rows.append(row)
        except Exception as exc:
            return self._persist_failure(
                work_item, fingerprint, "qa_rerun_failed",
                f"{type(exc).__name__}: {exc}")
        if len(transition_rows) != 3:
            return self._persist_failure(
                work_item, fingerprint, "qa_rerun_failed",
                "candidate did not produce exactly three QA results")
        transition_counts = {
            label: sum(row["transition"] == label for row in transition_rows)
            for label in TRANSITION_LABELS
        }
        transitions_path = _write_json(os.path.join(
            work_item.output_dir, "transitions.json"), {
            "candidate_property_id": proposal.candidate_property_id,
            "source_video_id": work_item.source_video_id,
            "transition_counts": transition_counts,
            "qas": transition_rows,
        })
        result_payload = {
            "schema_version": "property_intervention_result_v1",
            "status": "completed", "input_fingerprint": fingerprint,
            "candidate_property_id": proposal.candidate_property_id,
            "source_video_id": work_item.source_video_id,
            "parent_baseline_identity": parent_baseline_identity,
            "retrieval_identity": work_identity["retrieval_identity"],
            "mixed_captions_path": mixed.captions_path,
            "mixed_captions_hash": mixed.captions_hash,
            "selected_segment_ids": selected,
            "transitions_path": transitions_path,
            "transition_counts": transition_counts,
        }
        result_path = _write_json(result_path, result_payload)
        return PropertyInterventionResult(
            candidate_property_id=proposal.candidate_property_id,
            source_video_id=work_item.source_video_id, status="completed",
            result_path=result_path, mixed_captions_path=mixed.captions_path,
            transitions_path=transitions_path)

    def _persist_failure(self, work_item: PropertyInterventionWorkItem,
                         fingerprint: str, failure_type: str,
                         reason: str) -> PropertyInterventionResult:
        path = _write_json(os.path.join(work_item.output_dir, "result.json"), {
            "schema_version": "property_intervention_result_v1",
            "status": "failed", "input_fingerprint": fingerprint,
            "candidate_property_id": work_item.candidate_property_id,
            "source_video_id": work_item.source_video_id,
            "failure_type": failure_type, "failure_reason": reason,
        })
        return PropertyInterventionResult(
            candidate_property_id=work_item.candidate_property_id,
            source_video_id=work_item.source_video_id, status="failed",
            result_path=path, failure_type=failure_type,
            failure_reason=reason)

    @staticmethod
    def _baseline_registries(cache_keys_path: str) -> dict[str, object]:
        registries = {}
        for row in _read_jsonl(cache_keys_path):
            result_path = row.get("result_path") or os.path.join(
                row.get("cache_dir", ""), "caption.json")
            if not result_path or not os.path.exists(result_path):
                raise PropertyInterventionError(
                    f"missing baseline caption cache for {row.get('segment_id')}")
            payload = _read_json(result_path)
            registry = (payload.get("parsed") or {}).get("subject_registry")
            if registry:
                registries[str(row["segment_id"])] = registry
        return registries

    def _caption_configuration(self) -> dict[str, str]:
        return {
            "model_id": self.segment_captioner.caption_model_id,
            "backend_id": self.segment_captioner.backend_id,
            "decoding_hash": config.decoding_hash(),
        }

    def _qa_configuration(self, dvd_max_iterations: int) -> dict[str, Any]:
        module = getattr(self.qa_fn, "__module__", type(self.qa_fn).__module__)
        name = getattr(self.qa_fn, "__qualname__", type(self.qa_fn).__qualname__)
        return {
            "runner": f"{module}.{name}",
            "dvd_max_iterations": dvd_max_iterations,
        }
