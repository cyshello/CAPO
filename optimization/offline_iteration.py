"""One-iteration saved-fixture optimization orchestration for Stage 4.12."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Mapping, Sequence

from surrogate_rollout.optimization.candidate_preview import CandidatePreviewBuilder
from surrogate_rollout.optimization.evidence_builder import (
    CounterfactualEvidenceBuilder,
    SavedEvaluationArtifact,
    write_evidence_artifacts,
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
    MetaKnowledgeItem,
    OptimizationIterationResult,
)
from surrogate_rollout.optimization.update_reviewer import (
    DeterministicUpdateReviewer,
)
from surrogate_rollout.optimization.update_validator import (
    DeterministicUpdateValidator,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import (
    Phase4Config,
    PromptBankSnapshot,
    RouterPolicySnapshot,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    dumps_canonical,
)
from surrogate_rollout.schemas import QAExample, sha256_text


def _write(path: str, value: object) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_write_text(path, dumps_canonical(value) + "\n")
    return path


def _retain_saved_scaffold_facts(evidence):
    """Promote only an explicit structured saved-fixture fact for attribution."""
    output = []
    for item in evidence:
        candidate_metadata = item.metadata.get("candidate_metadata") or {}
        explicit = candidate_metadata.get("scaffold_failure_evidence")
        if explicit:
            output.append(replace(item, metadata={
                **dict(item.metadata),
                "scaffold_failure_evidence": explicit,
            }))
        else:
            output.append(item)
    return tuple(output)


class PromptRoutingOptimizationLoop:
    """Policy-free one-iteration coordinator over saved fixture inputs."""

    def __init__(self, *, feedback_generator: FeedbackGenerator) -> None:
        self.feedback_generator = feedback_generator

    def run_iteration(
        self,
        *,
        input_bank: PromptBankSnapshot,
        input_router: RouterPolicySnapshot,
        input_scaffold: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        baseline_results: Sequence[SavedEvaluationArtifact],
        candidate_results: Sequence[SavedEvaluationArtifact],
        confirmation_examples: Sequence[QAExample],
        regression_examples: Sequence[QAExample],
        proposal_lineage: Mapping[str, Mapping[str, Sequence[str]]],
        saved_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
        initial_meta_knowledge: Sequence[MetaKnowledgeItem],
        config: Phase4Config,
        output_dir: str,
        commit: bool,
    ) -> OptimizationIterationResult:
        if commit or config.commit or not config.dry_run:
            raise ValueError("Stage 4.12 supports dry-run preview only")
        frozen_inputs = dumps_canonical({
            "bank": input_bank, "router": input_router,
            "scaffold": input_scaffold, "contract": scaffold_contract,
        })

        evidence = _retain_saved_scaffold_facts(
            CounterfactualEvidenceBuilder().build(
                baseline_results, candidate_results))
        evidence_path, _ = write_evidence_artifacts(
            evidence, os.path.join(output_dir, "evidence"),
            source={"kind": "saved_fixture", "evaluation_calls": 0})

        initial_feedback = self.feedback_generator.generate(
            evidence, initial_meta_knowledge)
        attributions = DeterministicFailureAttributor().attribute(
            evidence, initial_feedback)
        feedback = attach_attributions(initial_feedback, attributions)
        feedback_path = write_feedback_batch(
            feedback, os.path.join(output_dir, "feedback", "feedback.json"))
        attribution_path = _write(os.path.join(
            output_dir, "attribution", "attributions.json"), attributions)

        knowledge_path = os.path.join(
            output_dir, "meta_knowledge", "meta_knowledge.json")
        store = MetaKnowledgeStore(
            knowledge_path, min_support_for_confirmed=1,
            min_distinct_videos_for_confirmed=1,
            min_distinct_videos_for_global_scaffold=3)
        for item in initial_meta_knowledge:
            store.upsert(item)
        store.ingest(evidence, feedback, attributions)
        store.save()
        knowledge = store.list_items()

        proposals: dict[str, Any] = {}
        if config.optimization.optimize_prompt_bank:
            proposals["prompt_bank"] = \
                DeterministicPromptBankUpdateProposer().propose(
                    prompt_bank=input_bank, feedback=feedback,
                    meta_knowledge=knowledge)
        if config.optimization.optimize_router:
            proposals["router"] = DeterministicRouterUpdateProposer().propose(
                prompt_bank=input_bank, router_policy=input_router,
                feedback=feedback, meta_knowledge=knowledge)
        proposals["scaffold"] = DeterministicScaffoldUpdateProposer().propose(
            scaffold_policy=input_scaffold, scaffold_contract=scaffold_contract,
            feedback=feedback, meta_knowledge=knowledge,
            enabled=config.optimize_scaffold)

        proposal_paths = {
            component: _write(os.path.join(
                output_dir, "proposals", f"{component}.json"), proposal)
            for component, proposal in proposals.items()
        }
        reviewer = DeterministicUpdateReviewer()
        reviews = {
            component: reviewer.review(
                proposal=proposal, component=component,
                meta_knowledge=knowledge)
            for component, proposal in proposals.items()
        }
        review_paths = {
            component: _write(os.path.join(
                output_dir, "reviews", f"{component}.json"), review)
            for component, review in reviews.items()
        }

        builder = CandidatePreviewBuilder()
        previews = {}
        if reviews.get("prompt_bank") and \
                reviews["prompt_bank"].decision == "accept_for_validation":
            previews["prompt_bank"] = builder.build_prompt_bank(
                input_bank, proposals["prompt_bank"])
        if reviews.get("router") and \
                reviews["router"].decision == "accept_for_validation":
            previews["router"] = builder.build_router(
                input_router, proposals["router"], prompt_bank=input_bank)
        if reviews["scaffold"].decision == "accept_for_validation":
            previews["scaffold"] = builder.build_scaffold(
                input_scaffold, proposals["scaffold"],
                scaffold_contract=scaffold_contract)
        for component, preview in previews.items():
            _write(os.path.join(
                output_dir, "previews", f"{component}.json"), preview)

        versions = {
            "prompt_bank": input_bank.bank_version,
            "router": input_router.router_version,
            "scaffold": input_scaffold.scaffold_version,
        }
        validator = DeterministicUpdateValidator()
        validations = {}
        for component, preview in previews.items():
            lineage = proposal_lineage[component]
            component_scores = saved_scores[component]
            incumbent_state = {
                "version": versions[component],
                "contract_version": scaffold_contract.contract_version,
                "scores": component_scores["incumbent"],
            }
            candidate_state = {
                "version": preview.candidate_version,
                "contract_version": scaffold_contract.contract_version,
                "proposal_example_ids": lineage["example_ids"],
                "proposal_video_ids": lineage["video_ids"],
                "support_video_ids": (lineage["video_ids"]
                                      if component == "scaffold" else ()),
                "scores": component_scores["candidate"], "errors": {},
            }
            validations[component] = validator.validate(
                incumbent_state=incumbent_state,
                candidate_state=candidate_state, component=component,
                confirmation_examples=confirmation_examples,
                regression_examples=regression_examples)
        validation_paths = {
            component: _write(os.path.join(
                output_dir, "validations", f"{component}.json"), validation)
            for component, validation in validations.items()
        }

        identity = {
            "versions": versions,
            "optimize": {
                "prompt_bank": config.optimization.optimize_prompt_bank,
                "router": config.optimization.optimize_router,
                "scaffold": config.optimize_scaffold,
            },
            "evidence_ids": tuple(item.evidence_id for item in evidence),
            "proposal_ids": tuple(sorted(
                proposal.proposal_id for proposal in proposals.values())),
            "validation_ids": tuple(sorted(
                result.validation_id for result in validations.values())),
        }
        result = OptimizationIterationResult(
            iteration_id="iteration_" + sha256_text(
                dumps_canonical(identity))[:24],
            input_bank_version=input_bank.bank_version,
            input_router_version=input_router.router_version,
            input_scaffold_version=input_scaffold.scaffold_version,
            optimize_prompt_bank=config.optimization.optimize_prompt_bank,
            optimize_router=config.optimization.optimize_router,
            optimize_scaffold=config.optimize_scaffold,
            evidence_artifact=evidence_path,
            feedback_artifact=feedback_path,
            meta_knowledge_artifact=knowledge_path,
            bank_proposal_artifact=proposal_paths.get("prompt_bank"),
            router_proposal_artifact=proposal_paths.get("router"),
            scaffold_proposal_artifact=proposal_paths["scaffold"],
            review_artifacts=review_paths,
            validation_artifacts=validation_paths,
            committed_bank_version=None, committed_router_version=None,
            committed_scaffold_version=None, status="dry_run", errors=())
        _write(os.path.join(output_dir, "iteration_result.json"), result)
        _write(os.path.join(output_dir, "run_summary.json"), {
            "schema_version": "phase4_stage4.12_v1",
            "iteration_id": result.iteration_id,
            "optimize_scaffold": config.optimize_scaffold,
            "scaffold_skipped_reason": proposals["scaffold"].skipped_reason,
            "scaffold_candidate_created": (
                proposals["scaffold"].candidate_policy is not None),
            "review_decisions": {
                key: value.decision for key, value in reviews.items()},
            "validation_decisions": {
                key: value.decision for key, value in validations.items()},
            "attribution_artifact": attribution_path,
            "canonical_state_mutated": False,
            "component_updates_committed": False,
            "calls": {
                "captioning": 0, "dvd_reasoning": 0, "feedback_models": 0,
                "proposal_models": 0, "evaluation_models": 0,
                "optimization_models": 0, "component_commits": 0,
            },
        })
        if dumps_canonical({
                "bank": input_bank, "router": input_router,
                "scaffold": input_scaffold,
                "contract": scaffold_contract}) != frozen_inputs:
            raise RuntimeError("offline iteration mutated incumbent state")
        return result
