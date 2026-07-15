"""Run one complete Stage 4.12 iteration from repository fixtures only.

From the repository parent:

    python -m surrogate_rollout.scripts.run_phase4_offline_iteration \
      --optimize-scaffold false --output-dir surrogate_rollout/runs/stage4_12_fixed

Use ``--optimize-scaffold true`` for the scaffold-enabled condition. Both
commands are dry-run only and make no captioning, DVD, model, or commit calls.
"""

from __future__ import annotations

import argparse
import json
import os

from surrogate_rollout.optimization.evidence_builder import SavedEvaluationArtifact
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
)
from surrogate_rollout.schemas import QAExample


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_COMPONENTS = os.path.join(
    REPO_ROOT, "prompt_routing", "fixtures", "stage4_7_components.json")
DEFAULT_ITERATION = os.path.join(
    REPO_ROOT, "optimization", "fixtures", "stage4_12_iteration.json")
DEFAULT_VALIDATION = os.path.join(
    REPO_ROOT, "optimization", "fixtures", "stage4_11_validation.json")


def _saved_pair(fixture: dict) -> tuple[tuple, tuple]:
    baseline = []
    candidate = []
    for item in fixture["examples"]:
        common = {
            "source_format": "stage4_12_saved_fixture",
            "video_id": item["video_id"], "question_id": item["question_id"],
            "ground_truth": "A", "bank_version": "bank_v0002",
            "router_version": "router_v0002",
            "scaffold_version": "scaffold_v0002",
            "config_fingerprint": fixture["config_fingerprint"],
        }
        baseline.append(SavedEvaluationArtifact(
            **common, artifact_id="baseline:" + item["question_id"],
            condition_name="baseline", answer="B", score=0.0,
            selected_prompt_ids={
                "0_10": tuple(item["baseline_selected_prompt_ids"])},
            composition_traces={
                "0_10": {"selected_prompt_ids": item[
                    "baseline_selected_prompt_ids"]}},
            composed_prompt_refs={"0_10": "baseline_prompt_hash"},
            metadata={"fixture_component": item["component"]}))
        candidate.append(SavedEvaluationArtifact(
            **common, artifact_id="candidate:" + item["question_id"],
            condition_name="candidate", answer="A", score=1.0,
            selected_prompt_ids={
                "0_10": tuple(item["candidate_selected_prompt_ids"])},
            composition_traces={
                "0_10": {"selected_prompt_ids": item[
                    "candidate_selected_prompt_ids"]}},
            composed_prompt_refs={"0_10": "candidate_prompt_hash"},
            metadata={
                "fixture_component": item["component"],
                "scaffold_failure_evidence": (
                    "selected_instruction_omitted"
                    if item["component"] == "scaffold" else None),
            }))
    return tuple(baseline), tuple(candidate)


def _qa(value: dict) -> QAExample:
    return QAExample(**{**value, "options": tuple(value.get("options") or ())})


def _bool(value: str) -> bool:
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete saved-fixture Stage 4.12 dry run")
    parser.add_argument("--optimize-scaffold", required=True, type=_bool)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--components", default=DEFAULT_COMPONENTS)
    parser.add_argument("--iteration-fixture", default=DEFAULT_ITERATION)
    parser.add_argument("--validation-fixture", default=DEFAULT_VALIDATION)
    args = parser.parse_args()

    with open(args.components) as f:
        components = json.load(f)
    with open(args.iteration_fixture) as f:
        fixture = json.load(f)
    with open(args.validation_fixture) as f:
        validation = json.load(f)
    baseline, candidate = _saved_pair(fixture)
    config = Phase4Config(
        dry_run=True, commit=False,
        optimization=Phase4OptimizationConfig(
            optimize_prompt_bank=True, optimize_router=True,
            optimize_scaffold=args.optimize_scaffold, max_iterations=1))
    loop = PromptRoutingOptimizationLoop(
        feedback_generator=SavedFixtureFeedbackGenerator(
            fixture["feedback_overrides"]))
    loop.run_iteration(
        input_bank=prompt_bank_from_json(components["prompt_bank"]),
        input_router=router_policy_from_json(components["router_policy"]),
        input_scaffold=scaffold_policy_from_json(
            components["scaffold_policy"]),
        scaffold_contract=scaffold_contract_from_json(
            components["scaffold_contract"]),
        baseline_results=baseline, candidate_results=candidate,
        confirmation_examples=tuple(
            _qa(item) for item in validation["confirmation_examples"]),
        regression_examples=tuple(
            _qa(item) for item in validation["regression_examples"]),
        proposal_lineage=validation["proposal_lineage"],
        saved_scores=validation["saved_scores"],
        initial_meta_knowledge=tuple(meta_knowledge_from_json(item)
                                     for item in validation["review_knowledge"]),
        config=config, output_dir=args.output_dir, commit=False)
    print(os.path.join(args.output_dir, "iteration_result.json"))


if __name__ == "__main__":
    main()
