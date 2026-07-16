"""Exactly one real prompt-routing optimization iteration for Stages 4.13-4.14."""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from surrogate_rollout.captioning.candidate_captions import build_clip_index
from surrogate_rollout.evaluation.dvd_qa import ensure_backend, run_dvd_qa
from surrogate_rollout.optimization.candidate_preview import (
    CandidatePreviewBuilder,
    CandidatePreviewError,
)
from surrogate_rollout.optimization.failure_attributor import (
    DeterministicFailureAttributor,
)
from surrogate_rollout.optimization.feedback_generator import (
    FeedbackGenerator,
    attach_attributions,
    write_feedback_batch,
)
from surrogate_rollout.optimization.meta_knowledge import MetaKnowledgeStore
from surrogate_rollout.optimization.prompt_bank_update_proposer import (
    DeterministicPromptBankUpdateProposer,
)
from surrogate_rollout.optimization.router_update_proposer import (
    DeterministicRouterUpdateProposer,
)
from surrogate_rollout.optimization.scaffold_update_proposer import (
    DeterministicScaffoldUpdateProposer,
)
from surrogate_rollout.optimization.schemas import (
    CounterfactualEvidence,
    NoChangeEvaluationResult,
    OptimizationIterationResult,
)
from surrogate_rollout.optimization.update_reviewer import (
    DeterministicUpdateReviewer,
)
from surrogate_rollout.optimization.update_validator import (
    DeterministicUpdateValidator,
)
from surrogate_rollout.prompt_routing.offline_dry_run import run_offline_dry_run
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.routed_caption_view import (
    RoutedCaptionViewBuilder,
)
from surrogate_rollout.prompt_routing.schemas import (
    Phase4Config,
    PromptBankSnapshot,
    RouterPolicySnapshot,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    SegmentContext,
    dumps_canonical,
)
from surrogate_rollout.schemas import QAExample, sha256_text


def _write(path: str, value: object) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_write_text(path, dumps_canonical(value) + "\n")
    return path


@dataclass(frozen=True)
class RealEvaluationResult:
    incumbent_scores: Mapping[str, float]
    candidate_scores: Mapping[str, float]
    incumbent_errors: Mapping[str, str]
    candidate_errors: Mapping[str, str]
    incumbent_artifact: str
    candidate_artifact: str
    comparison_artifact: str
    timing: Mapping[str, float]
    no_change: NoChangeEvaluationResult | None = None


def _prompt_hashes(offline_by_video: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    return {
        video_id: tuple(item.prompt_hash for item in result.composed_prompts)
        for video_id, result in sorted(offline_by_video.items())
    }


def evaluate_routed_states(
    *,
    incumbent_bank: PromptBankSnapshot,
    incumbent_router: RouterPolicySnapshot,
    candidate_bank: PromptBankSnapshot,
    candidate_router: RouterPolicySnapshot,
    incumbent_scaffold: ScaffoldPolicySnapshot,
    candidate_scaffold: ScaffoldPolicySnapshot,
    incumbent_contract: ScaffoldContract,
    candidate_contract: ScaffoldContract,
    qa_samples: Sequence[tuple[QAExample, dict]],
    base_prompt_template: str,
    merge_prompt: str,
    phase4_config: Phase4Config,
    output_dir: str,
    source_revision: str,
    gpu: str,
    dvd_max_iterations: int,
    rollout_mode: str = "full",
    max_selected_clips: int = 32,
    candidate_cache_root: str | None = None,
    text_backend: str = "openai",
    use_openai_tools: bool = True,
) -> RealEvaluationResult:
    """Evaluate two component states through existing routed/DVD services."""
    if rollout_mode not in ("full", "selective"):
        raise ValueError("rollout_mode must be full or selective")
    if max_selected_clips < 1:
        raise ValueError("max_selected_clips must be positive")
    states = {
        "incumbent": (
            incumbent_bank, incumbent_router, incumbent_scaffold,
            incumbent_contract),
        "candidate": (
            candidate_bank, candidate_router, candidate_scaffold,
            candidate_contract),
    }
    by_video: dict[str, list[tuple[QAExample, dict]]] = {}
    for example, sample in qa_samples:
        by_video.setdefault(example.video_id, []).append((example, sample))

    offline_results: dict[str, dict[str, Any]] = {}
    for state_name, (bank, router, scaffold, contract) in states.items():
        offline_results[state_name] = {}
        for video_id, video_qas in by_video.items():
            sample = video_qas[0][1]
            clip_keys = tuple(key for key, _ in build_clip_index(sample, video_id))
            contexts = tuple(SegmentContext(
                video_id=video_id, segment_id=segment_id,
                segment_features={"smoke_route": "multi_entry"},
                metadata={"stage": "4.14", "state": state_name},
            ) for segment_id in clip_keys)
            offline_results[state_name][video_id] = run_offline_dry_run(
                contexts=contexts, prompt_bank=bank, router_policy=router,
                scaffold_policy=scaffold, scaffold_contract=contract,
                phase4_config=phase4_config,
                base_prompt_template=base_prompt_template,
                output_dir=os.path.join(
                    output_dir, state_name, "routing", video_id),
                source_revision=source_revision)

    incumbent_hashes = _prompt_hashes(offline_results["incumbent"])
    candidate_hashes = _prompt_hashes(offline_results["candidate"])
    component_equivalence = {
        "prompt_bank": dumps_canonical(incumbent_bank) ==
        dumps_canonical(candidate_bank),
        "router": dumps_canonical(incumbent_router) ==
        dumps_canonical(candidate_router),
        "scaffold": dumps_canonical(incumbent_scaffold) ==
        dumps_canonical(candidate_scaffold),
        "contract": dumps_canonical(incumbent_contract) ==
        dumps_canonical(candidate_contract),
    }
    equivalent = all(component_equivalence.values()) and \
        incumbent_hashes == candidate_hashes
    no_change = None
    if equivalent:
        no_change = NoChangeEvaluationResult(
            status="no_change",
            reason="equivalent_components_and_composed_prompts",
            component_equivalence=component_equivalence,
            incumbent_composed_prompt_hashes=incumbent_hashes,
            candidate_composed_prompt_hashes=candidate_hashes,
            candidate_dvd_executed=False,
            eligible_for_validation=False,
            eligible_for_regression_evidence=False,
        )

    ensure_backend(
        gpu, preload_captioner=True, text_backend=text_backend,
        use_openai_tools=use_openai_tools)
    score_maps = {}
    error_maps = {}
    artifacts = {}
    timing = {}
    incumbent_views = {}
    incumbent_qa_dirs = {}
    for state_name, (bank, router, scaffold, contract) in states.items():
        if state_name == "candidate" and no_change is not None:
            score_maps[state_name] = {}
            error_maps[state_name] = {}
            artifacts[state_name] = _write(os.path.join(
                output_dir, "candidate", "no_change.json"), no_change)
            timing[state_name] = 0.0
            continue
        started = time.monotonic()
        state_dir = os.path.join(output_dir, state_name)
        rows = []
        for video_id, video_qas in by_video.items():
            sample = video_qas[0][1]
            routing_dir = os.path.join(state_dir, "routing", video_id)
            offline = offline_results[state_name][video_id]
            prompt_map = {
                item.segment_id: item for item in offline.composed_prompts}
            all_segment_ids = tuple(prompt_map)
            caption_view = None
            if state_name == "incumbent" or rollout_mode == "full":
                caption_view = RoutedCaptionViewBuilder().build(
                    sample=sample,
                    segment_composed_prompt_map=prompt_map,
                    work_root=os.path.join(state_dir, "caption_view"),
                    merge_prompt=merge_prompt,
                    candidate_cache_root=candidate_cache_root)
                if state_name == "incumbent":
                    incumbent_views[video_id] = caption_view
            for example, qa_sample in video_qas:
                qa_dir = os.path.join(
                    state_dir, "qa", example.question_id.replace("/", "-"))
                selection = None
                fallback_type = "none"
                fallback_reason = None
                if state_name == "candidate" and rollout_mode == "selective":
                    reference_path = os.path.join(
                        incumbent_qa_dirs[example.question_id], "references.json")
                    with open(reference_path) as f:
                        references = json.load(f)
                    referenced = set(
                        references.get("consumed_segments") or
                        references.get("frame_inspected_segments") or
                        references.get("returned_segments") or ())
                    ordered = [value for value in incumbent_views[video_id].segment_ids
                               if value in referenced]
                    if not ordered:
                        ordered = list(incumbent_views[video_id].segment_ids)
                        fallback_type = "full_rollout"
                        fallback_reason = "no incumbent trace references"
                    selection = set(ordered[:max_selected_clips])
                    caption_view = RoutedCaptionViewBuilder().build_selective(
                        sample=sample,
                        segment_composed_prompt_map=prompt_map,
                        selected_segment_ids=selection,
                        baseline_view=incumbent_views[video_id],
                        work_root=os.path.join(state_dir, "caption_view"),
                        merge_prompt=merge_prompt,
                        candidate_cache_root=candidate_cache_root,
                        view_name=example.question_id.replace("/", "-"))
                result = run_dvd_qa(
                    captions_path=caption_view.captions_path,
                    sample=qa_sample, run_dir=qa_dir,
                    question_id=example.question_id,
                    database_path=caption_view.database_path,
                    max_iterations=dvd_max_iterations, gpu=gpu)
                if state_name == "incumbent":
                    incumbent_qa_dirs[example.question_id] = qa_dir
                rows.append({
                    "question_id": example.question_id,
                    "video_id": example.video_id,
                    "score": result.score,
                    "prediction": result.prediction,
                    "parsed_answer": result.parsed_answer,
                    "ground_truth": result.ground_truth,
                    "errors": result.errors,
                    "token_usage": result.token_usage,
                    "reasoning_seconds": result.latency_seconds,
                    "captions_path": result.captions_path,
                    "database_path": result.database_path,
                    "trajectory_path": os.path.join(qa_dir, "trajectory.jsonl"),
                    "routing_decisions_path": os.path.join(
                        routing_dir, "routing", "decisions.jsonl"),
                    "composition_traces_path": os.path.join(
                        routing_dir, "composition", "traces.jsonl"),
                    "composed_prompts_path": os.path.join(
                        routing_dir, "composition", "composed_prompts.jsonl"),
                    "routed_view_path": caption_view.routed_view_path,
                    "caption_call_count": caption_view.caption_call_count,
                    "caption_cache_hits": caption_view.caption_cache_hits,
                    "caption_seconds": caption_view.caption_seconds,
                    "rollout_mode": (rollout_mode if state_name == "candidate"
                                     else "full"),
                    "selection_policy": ("trace_reference_budget"
                                         if selection is not None else
                                         "full_rollout"),
                    "selected_clip_ids": (tuple(sorted(selection))
                                          if selection is not None else
                                          all_segment_ids),
                    "recaption_fraction": (
                        len(selection) / len(all_segment_ids)
                        if selection is not None else 1.0),
                    "fallback_type": fallback_type,
                    "fallback_reason": fallback_reason,
                    "component_versions": {
                        "bank": bank.bank_version,
                        "router": router.router_version,
                        "scaffold": scaffold.scaffold_version,
                        "contract": contract.contract_version,
                    },
                })
        result_path = os.path.join(state_dir, "results.jsonl")
        _atomic_write_text(result_path, "".join(
            dumps_canonical(row) + "\n" for row in rows))
        score_maps[state_name] = {
            row["question_id"]: float(row["score"]) for row in rows}
        error_maps[state_name] = {
            row["question_id"]: dumps_canonical(row["errors"])
            for row in rows if row["errors"]}
        artifacts[state_name] = result_path
        timing[state_name] = time.monotonic() - started

    comparison_path = _write(os.path.join(output_dir, "comparison.json"), {
        "incumbent_scores": score_maps["incumbent"],
        "candidate_scores": (score_maps["candidate"]
                             if no_change is None else None),
        "score_deltas": ({
            key: score_maps["candidate"][key] - score_maps["incumbent"][key]
            for key in score_maps["incumbent"]} if no_change is None else None),
        "incumbent_scaffold_version": incumbent_scaffold.scaffold_version,
        "candidate_scaffold_version": candidate_scaffold.scaffold_version,
        "contract_version": incumbent_contract.contract_version,
        "candidate_rollout_mode": rollout_mode,
        "no_change": no_change,
    })
    return RealEvaluationResult(
        incumbent_scores=score_maps["incumbent"],
        candidate_scores=score_maps["candidate"],
        incumbent_errors=error_maps["incumbent"],
        candidate_errors=error_maps["candidate"],
        incumbent_artifact=artifacts["incumbent"],
        candidate_artifact=artifacts["candidate"],
        comparison_artifact=comparison_path, timing=timing,
        no_change=no_change)


class RealFixedScaffoldIterationRunner:
    def __init__(
        self,
        *,
        feedback_generator: FeedbackGenerator,
        evaluation_fn: Callable[..., RealEvaluationResult] = evaluate_routed_states,
        optimize_scaffold: bool = False,
        stage: str = "4.13",
    ) -> None:
        self.feedback_generator = feedback_generator
        self.evaluation_fn = evaluation_fn
        self.optimize_scaffold = optimize_scaffold
        self.stage = stage

    def run(
        self,
        *,
        input_bank: PromptBankSnapshot,
        input_router: RouterPolicySnapshot,
        input_scaffold: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        evidence: Sequence[CounterfactualEvidence],
        confirmation_examples: Sequence[QAExample],
        regression_examples: Sequence[QAExample],
        qa_samples: Sequence[tuple[QAExample, dict]],
        base_prompt_template: str,
        merge_prompt: str,
        config: Phase4Config,
        output_dir: str,
        source_revision: str,
        frozen_manifest_path: str,
        verify_frozen: Callable[[], None],
        gpu: str,
        dvd_max_iterations: int,
    ) -> OptimizationIterationResult:
        if config.commit or config.dry_run is False:
            raise ValueError(f"Stage {self.stage} is preview-only")
        if config.optimization.max_iterations != 1:
            raise ValueError(f"Stage {self.stage} requires max_iterations=1")
        if not config.optimization.optimize_prompt_bank or \
                not config.optimization.optimize_router or \
                config.optimize_scaffold != self.optimize_scaffold:
            raise ValueError(
                f"Stage {self.stage} requires bank/router on and "
                f"optimize_scaffold={self.optimize_scaffold}")
        verify_frozen()

        feedback = self.feedback_generator.generate(evidence, ())
        attributions = DeterministicFailureAttributor().attribute(
            evidence, feedback)
        feedback = attach_attributions(feedback, attributions)
        feedback_path = write_feedback_batch(
            feedback, os.path.join(output_dir, "feedback", "parsed_feedback.json"))
        attribution_path = _write(os.path.join(
            output_dir, "attribution", "failure_attributions.json"),
            attributions)
        store = MetaKnowledgeStore(os.path.join(
            output_dir, "meta_knowledge", "meta_knowledge.json"))
        store.ingest(evidence, feedback, attributions)
        store.save()
        knowledge = store.list_items()

        proposals = {
            "prompt_bank": DeterministicPromptBankUpdateProposer(
                max_operations=config.optimization.max_new_entries_per_iteration,
            ).propose(
                prompt_bank=input_bank, feedback=feedback,
                meta_knowledge=knowledge),
            "router": DeterministicRouterUpdateProposer(
                max_operations=config.optimization.max_router_operations_per_iteration,
            ).propose(
                prompt_bank=input_bank, router_policy=input_router,
                feedback=feedback, meta_knowledge=knowledge),
            "scaffold": DeterministicScaffoldUpdateProposer().propose(
                scaffold_policy=input_scaffold,
                scaffold_contract=scaffold_contract, feedback=feedback,
                meta_knowledge=knowledge, enabled=config.optimize_scaffold),
        }
        proposal_paths = {key: _write(os.path.join(
            output_dir, "proposals", f"{key}.json"), value)
            for key, value in proposals.items()}
        reviewer = DeterministicUpdateReviewer()
        reviews = {key: reviewer.review(
            proposal=value, component=key, meta_knowledge=knowledge)
            for key, value in proposals.items()}
        review_paths = {key: _write(os.path.join(
            output_dir, "reviews", f"{key}.json"), value)
            for key, value in reviews.items()}

        preview_builder = CandidatePreviewBuilder()
        previews = {}
        preview_errors = {}
        if reviews["prompt_bank"].decision == "accept_for_validation":
            try:
                previews["prompt_bank"] = preview_builder.build_prompt_bank(
                    input_bank, proposals["prompt_bank"])
            except CandidatePreviewError as exc:
                preview_errors["prompt_bank"] = str(exc)
        candidate_bank = (previews["prompt_bank"].candidate_state
                          if "prompt_bank" in previews else input_bank)
        if reviews["router"].decision == "accept_for_validation":
            try:
                previews["router"] = preview_builder.build_router(
                    input_router, proposals["router"], prompt_bank=candidate_bank)
            except CandidatePreviewError as exc:
                preview_errors["router"] = str(exc)
        candidate_router = (previews["router"].candidate_state
                            if "router" in previews else input_router)
        if reviews["scaffold"].decision == "accept_for_validation":
            try:
                previews["scaffold"] = preview_builder.build_scaffold(
                    input_scaffold, proposals["scaffold"],
                    scaffold_contract=scaffold_contract)
            except CandidatePreviewError as exc:
                preview_errors["scaffold"] = str(exc)
        candidate_scaffold = (previews["scaffold"].candidate_state
                              if "scaffold" in previews else input_scaffold)
        preview_paths = {key: _write(os.path.join(
            output_dir, "output_state", f"preview_{key}.json"), value)
            for key, value in previews.items()}
        preview_error_path = None
        if preview_errors:
            preview_error_path = _write(os.path.join(
                output_dir, "output_state", "preview_errors.json"),
                preview_errors)

        verify_frozen()
        evaluation = self.evaluation_fn(
            incumbent_bank=input_bank, incumbent_router=input_router,
            candidate_bank=candidate_bank, candidate_router=candidate_router,
            incumbent_scaffold=input_scaffold,
            candidate_scaffold=candidate_scaffold,
            incumbent_contract=scaffold_contract,
            candidate_contract=scaffold_contract, qa_samples=qa_samples,
            base_prompt_template=base_prompt_template,
            merge_prompt=merge_prompt, phase4_config=config,
            output_dir=os.path.join(output_dir, "evaluation"),
            source_revision=source_revision, gpu=gpu,
            dvd_max_iterations=dvd_max_iterations)

        evidence_by_id = {item.evidence_id: item for item in evidence}
        feedback_by_id = {item.feedback_id: item for item in feedback.items}
        validation_paths = {}
        validations = {}
        versions = {
            "prompt_bank": input_bank.bank_version,
            "router": input_router.router_version,
            "scaffold": input_scaffold.scaffold_version,
        }
        validation_previews = (() if evaluation.no_change is not None
                               else previews.items())
        for component, preview in validation_previews:
            source_feedback_ids = (
                tuple(value for op in proposals[component].operations
                      for value in op.source_feedback_ids)
                if component == "prompt_bank"
                else proposals[component].source_feedback_ids)
            source_evidence_ids = tuple(sorted({
                evidence_id for feedback_id in source_feedback_ids
                for evidence_id in feedback_by_id[feedback_id].evidence_ids}))
            source_videos = tuple(sorted({
                evidence_by_id[value].video_id for value in source_evidence_ids}))
            validations[component] = DeterministicUpdateValidator().validate(
                incumbent_state={
                    "version": versions[component],
                    "contract_version": scaffold_contract.contract_version,
                    "scores": evaluation.incumbent_scores},
                candidate_state={
                    "version": preview.candidate_version,
                    "contract_version": scaffold_contract.contract_version,
                    "proposal_example_ids": source_evidence_ids,
                    "proposal_video_ids": source_videos,
                    "support_video_ids": (source_videos
                                          if component == "scaffold" else ()),
                    "scores": evaluation.candidate_scores,
                    "errors": evaluation.candidate_errors},
                component=component,
                confirmation_examples=confirmation_examples,
                regression_examples=regression_examples)
            validation_paths[component] = _write(os.path.join(
                output_dir, "validation", f"{component}.json"),
                validations[component])

        accepted = {component for component, value in validations.items()
                    if value.decision == "accept"}
        next_bank = (candidate_bank if "prompt_bank" in accepted else input_bank)
        next_router = (candidate_router if "router" in accepted else input_router)
        next_scaffold = (candidate_scaffold
                         if "scaffold" in accepted else input_scaffold)
        next_state_path = _write(os.path.join(
            output_dir, "output_state", "next_components.json"), {
                "prompt_bank": next_bank,
                "router_policy": next_router,
                "scaffold_policy": next_scaffold,
                "scaffold_contract": scaffold_contract,
                "accepted_components": tuple(sorted(accepted)),
                "canonical_state_mutated": False,
            })

        verify_frozen()
        identity = {
            "source_revision": source_revision,
            "input_versions": (
                input_bank.bank_version, input_router.router_version,
                input_scaffold.scaffold_version),
            "evidence_ids": tuple(item.evidence_id for item in evidence),
            "proposal_ids": tuple(value.proposal_id
                                  for value in proposals.values()),
            "evaluation": evaluation.comparison_artifact,
        }
        result = OptimizationIterationResult(
            iteration_id="iteration_" + sha256_text(
                dumps_canonical(identity))[:24],
            input_bank_version=input_bank.bank_version,
            input_router_version=input_router.router_version,
            input_scaffold_version=input_scaffold.scaffold_version,
            optimize_prompt_bank=True, optimize_router=True,
            optimize_scaffold=config.optimize_scaffold,
            evidence_artifact=os.path.join(
                output_dir, "frozen_batches", "evidence.jsonl"),
            feedback_artifact=feedback_path,
            meta_knowledge_artifact=store.path,
            bank_proposal_artifact=proposal_paths["prompt_bank"],
            router_proposal_artifact=proposal_paths["router"],
            scaffold_proposal_artifact=proposal_paths["scaffold"],
            review_artifacts=review_paths,
            validation_artifacts=validation_paths,
            committed_bank_version=None, committed_router_version=None,
            committed_scaffold_version=None, status="dry_run", errors=())
        _write(os.path.join(output_dir, "iteration_result.json"), result)
        _write(os.path.join(output_dir, "run_summary.json"), {
            "stage": self.stage, "max_iterations": 1,
            "optimize_scaffold": config.optimize_scaffold,
            "fixed_scaffold_version": input_scaffold.scaffold_version,
            "frozen_manifest": frozen_manifest_path,
            "feedback_policy": feedback.feedback_policy,
            "feedback_policy_version": feedback.feedback_policy_version,
            "review_decisions": {
                key: value.decision for key, value in reviews.items()},
            "validation_decisions": {
                key: value.decision for key, value in validations.items()},
            "preview_artifacts": preview_paths,
            "preview_error_artifact": preview_error_path,
            "attribution_artifact": attribution_path,
            "evaluation_artifacts": {
                "incumbent": evaluation.incumbent_artifact,
                "candidate": evaluation.candidate_artifact,
                "comparison": evaluation.comparison_artifact},
            "evaluation_timing": evaluation.timing,
            "no_change": evaluation.no_change,
            "candidate_validation_eligible": evaluation.no_change is None,
            "candidate_regression_evidence_eligible": evaluation.no_change is None,
            "canonical_commits": 0,
            "next_state_artifact": next_state_path,
            "accepted_components": tuple(sorted(accepted)),
            "scaffold_candidate_created": (
                proposals["scaffold"].candidate_policy is not None),
            "scaffold_skipped_reason": proposals["scaffold"].skipped_reason,
        })
        return result
