"""Write the deterministic rollback-safe Stage 4.11 checkpoint artifacts."""

from __future__ import annotations

import argparse
import json
import os

from surrogate_rollout.optimization.candidate_preview import CandidatePreviewBuilder
from surrogate_rollout.optimization.meta_knowledge import meta_knowledge_from_json
from surrogate_rollout.optimization.prompt_bank_update_proposer import (
    DeterministicPromptBankUpdateProposer,
)
from surrogate_rollout.optimization.router_update_proposer import (
    DeterministicRouterUpdateProposer,
)
from surrogate_rollout.optimization.scaffold_update_proposer import (
    DeterministicScaffoldUpdateProposer,
)
from surrogate_rollout.optimization.update_reviewer import (
    DeterministicUpdateReviewer,
)
from surrogate_rollout.optimization.update_validator import (
    DeterministicUpdateValidator,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import QAExample
from surrogate_rollout.scripts.run_phase4_proposal_checkpoint import _load_inputs


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_COMPONENTS = os.path.join(
    REPO_ROOT, "prompt_routing", "fixtures", "stage4_7_components.json")
DEFAULT_FEEDBACK = os.path.join(
    REPO_ROOT, "optimization", "fixtures", "stage4_10_feedback.json")
DEFAULT_VALIDATION = os.path.join(
    REPO_ROOT, "optimization", "fixtures", "stage4_11_validation.json")


def _write(path: str, value: object) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_write_text(path, dumps_canonical(value) + "\n")
    return path


def _qa(value: dict) -> QAExample:
    return QAExample(**{**value, "options": tuple(value.get("options") or ())})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4.11 deterministic review/validation checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--components", default=DEFAULT_COMPONENTS)
    parser.add_argument("--feedback-fixture", default=DEFAULT_FEEDBACK)
    parser.add_argument("--validation-fixture", default=DEFAULT_VALIDATION)
    args = parser.parse_args()

    inputs = _load_inputs(args.components, args.feedback_fixture)
    with open(args.validation_fixture) as f:
        fixture = json.load(f)
    review_knowledge = tuple(meta_knowledge_from_json(item)
                             for item in fixture["review_knowledge"])
    all_knowledge = (*inputs["meta_knowledge"], *review_knowledge)
    confirmation = tuple(_qa(item) for item in fixture["confirmation_examples"])
    regression = tuple(_qa(item) for item in fixture["regression_examples"])
    frozen_before = dumps_canonical(inputs)

    proposals = {
        "prompt_bank": DeterministicPromptBankUpdateProposer().propose(
            prompt_bank=inputs["prompt_bank"], feedback=inputs["feedback"],
            meta_knowledge=all_knowledge),
        "router": DeterministicRouterUpdateProposer().propose(
            prompt_bank=inputs["prompt_bank"],
            router_policy=inputs["router_policy"], feedback=inputs["feedback"],
            meta_knowledge=all_knowledge),
        "scaffold": DeterministicScaffoldUpdateProposer().propose(
            scaffold_policy=inputs["scaffold_policy"],
            scaffold_contract=inputs["scaffold_contract"],
            feedback=inputs["feedback"], meta_knowledge=all_knowledge,
            enabled=True),
    }
    reviewer = DeterministicUpdateReviewer()
    reviews = {component: reviewer.review(
        proposal=proposal, component=component, meta_knowledge=review_knowledge)
        for component, proposal in proposals.items()}
    if any(review.decision != "accept_for_validation"
           for review in reviews.values()):
        raise RuntimeError("synthetic proposals did not pass review")

    builder = CandidatePreviewBuilder()
    previews = {
        "prompt_bank": builder.build_prompt_bank(
            inputs["prompt_bank"], proposals["prompt_bank"]),
        "router": builder.build_router(
            inputs["router_policy"], proposals["router"],
            prompt_bank=inputs["prompt_bank"]),
        "scaffold": builder.build_scaffold(
            inputs["scaffold_policy"], proposals["scaffold"],
            scaffold_contract=inputs["scaffold_contract"]),
    }
    incumbent_versions = {
        "prompt_bank": inputs["prompt_bank"].bank_version,
        "router": inputs["router_policy"].router_version,
        "scaffold": inputs["scaffold_policy"].scaffold_version,
    }
    validator = DeterministicUpdateValidator()
    validations = {}
    artifacts = {}
    for component in ("prompt_bank", "router", "scaffold"):
        lineage = fixture["proposal_lineage"][component]
        scores = fixture["saved_scores"][component]
        incumbent_state = {
            "version": incumbent_versions[component],
            "contract_version": inputs["scaffold_contract"].contract_version,
            "scores": scores["incumbent"],
        }
        candidate_state = {
            "version": previews[component].candidate_version,
            "contract_version": inputs["scaffold_contract"].contract_version,
            "proposal_example_ids": lineage["example_ids"],
            "proposal_video_ids": lineage["video_ids"],
            "support_video_ids": (lineage["video_ids"]
                                  if component == "scaffold" else ()),
            "scores": scores["candidate"], "errors": {},
        }
        validations[component] = validator.validate(
            incumbent_state=incumbent_state, candidate_state=candidate_state,
            component=component, confirmation_examples=confirmation,
            regression_examples=regression)
        artifacts[component] = {
            "review": _write(os.path.join(
                args.output_dir, "reviews", f"{component}.json"),
                reviews[component]),
            "candidate_preview": _write(os.path.join(
                args.output_dir, "previews", f"{component}.json"),
                previews[component]),
            "validation": _write(os.path.join(
                args.output_dir, "validations", f"{component}.json"),
                validations[component]),
        }

    if dumps_canonical(inputs) != frozen_before:
        raise RuntimeError("checkpoint mutated incumbent component state")
    proposal_videos = {
        video for lineage in fixture["proposal_lineage"].values()
        for video in lineage["video_ids"]}
    confirmation_videos = {item.video_id for item in confirmation}
    regression_videos = {item.video_id for item in regression}
    summary = {
        "schema_version": "phase4_stage4.11_v1",
        "preview_only": True,
        "canonical_state_mutated": False,
        "component_updates_committed": False,
        "input_versions": {
            "bank": inputs["prompt_bank"].bank_version,
            "router": inputs["router_policy"].router_version,
            "scaffold": inputs["scaffold_policy"].scaffold_version,
            "contract": inputs["scaffold_contract"].contract_version,
        },
        "proposal_video_ids": sorted(proposal_videos),
        "confirmation_video_ids": sorted(confirmation_videos),
        "regression_video_ids": sorted(regression_videos),
        "video_splits_disjoint": not (
            proposal_videos & confirmation_videos
            or proposal_videos & regression_videos
            or confirmation_videos & regression_videos),
        "review_decisions": {
            key: value.decision for key, value in reviews.items()},
        "validation_decisions": {
            key: value.decision for key, value in validations.items()},
        "artifacts": artifacts,
        "configuration": {
            "optimize_prompt_bank": True, "optimize_router": True,
            "optimize_scaffold": True, "dry_run": True, "commit": False,
        },
        "calls": {
            "captioning": 0, "dvd_reasoning": 0, "feedback_models": 0,
            "proposal_models": 0, "evaluation_models": 0,
            "optimization": 0, "component_commits": 0,
        },
    }
    summary_path = os.path.join(args.output_dir, "stage4_11_summary.json")
    _atomic_write_text(summary_path, json.dumps(
        summary, sort_keys=True, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
