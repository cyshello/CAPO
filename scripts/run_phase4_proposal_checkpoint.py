"""Write deterministic preview-only Stage 4.10 proposal artifacts."""

from __future__ import annotations

import argparse
import json
import os

from surrogate_rollout.optimization.feedback_generator import (
    feedback_batch_from_json,
)
from surrogate_rollout.optimization.meta_knowledge import (
    meta_knowledge_from_json,
)
from surrogate_rollout.optimization.prompt_bank_update_proposer import (
    DeterministicPromptBankUpdateProposer,
)
from surrogate_rollout.optimization.router_update_proposer import (
    DeterministicRouterUpdateProposer,
)
from surrogate_rollout.optimization.scaffold_update_proposer import (
    DeterministicScaffoldUpdateProposer,
)
from surrogate_rollout.prompt_routing.persistence import (
    _atomic_write_text,
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_COMPONENTS = os.path.join(
    REPO_ROOT, "prompt_routing", "fixtures", "stage4_7_components.json")
DEFAULT_FEEDBACK = os.path.join(
    REPO_ROOT, "optimization", "fixtures", "stage4_10_feedback.json")


def _load_inputs(component_path: str, feedback_path: str) -> dict:
    with open(component_path) as f:
        components = json.load(f)
    with open(feedback_path) as f:
        feedback_data = json.load(f)
    return {
        "prompt_bank": prompt_bank_from_json(components["prompt_bank"]),
        "router_policy": router_policy_from_json(components["router_policy"]),
        "scaffold_policy": scaffold_policy_from_json(
            components["scaffold_policy"]),
        "scaffold_contract": scaffold_contract_from_json(
            components["scaffold_contract"]),
        "feedback": feedback_batch_from_json(feedback_data["feedback"]),
        "meta_knowledge": tuple(meta_knowledge_from_json(item)
                                for item in feedback_data["meta_knowledge"]),
    }


def _write(path: str, value: object) -> str:
    _atomic_write_text(path, dumps_canonical(value) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4.10 deterministic preview proposal checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--components", default=DEFAULT_COMPONENTS)
    parser.add_argument("--feedback-fixture", default=DEFAULT_FEEDBACK)
    args = parser.parse_args()

    inputs = _load_inputs(args.components, args.feedback_fixture)
    frozen_before = dumps_canonical(inputs)
    artifacts = {}
    condition_summaries = {}
    for condition, scaffold_enabled in (
        ("scaffold_disabled", False), ("scaffold_enabled", True),
    ):
        proposal_dir = os.path.join(args.output_dir, condition, "proposals")
        os.makedirs(proposal_dir, exist_ok=True)
        bank = DeterministicPromptBankUpdateProposer().propose(
            prompt_bank=inputs["prompt_bank"], feedback=inputs["feedback"],
            meta_knowledge=inputs["meta_knowledge"])
        router = DeterministicRouterUpdateProposer().propose(
            prompt_bank=inputs["prompt_bank"],
            router_policy=inputs["router_policy"],
            feedback=inputs["feedback"],
            meta_knowledge=inputs["meta_knowledge"])
        scaffold = DeterministicScaffoldUpdateProposer().propose(
            scaffold_policy=inputs["scaffold_policy"],
            scaffold_contract=inputs["scaffold_contract"],
            feedback=inputs["feedback"],
            meta_knowledge=inputs["meta_knowledge"], enabled=scaffold_enabled)
        artifacts[condition] = {
            "prompt_bank": _write(os.path.join(
                proposal_dir, "prompt_bank_proposal.json"), bank),
            "router": _write(os.path.join(
                proposal_dir, "router_proposal.json"), router),
            "scaffold": _write(os.path.join(
                proposal_dir, "scaffold_proposal.json"), scaffold),
        }
        condition_summaries[condition] = {
            "optimize_prompt_bank": True,
            "optimize_router": True,
            "optimize_scaffold": scaffold_enabled,
            "dry_run": True,
            "commit": False,
            "bank_valid": bank.is_valid,
            "router_valid": router.is_valid,
            "scaffold_valid": scaffold.is_valid,
            "scaffold_skipped_reason": scaffold.skipped_reason,
            "scaffold_candidate_created": scaffold.candidate_policy is not None,
            "scaffold_proposal_policy_calls": scaffold.provenance[
                "proposal_policy_calls"],
            "model_calls": 0,
        }

    if dumps_canonical(inputs) != frozen_before:
        raise RuntimeError("proposal generation mutated input state")
    summary_path = os.path.join(args.output_dir, "stage4_10_summary.json")
    _atomic_write_text(summary_path, json.dumps({
        "schema_version": "phase4_stage4.10_v1",
        "preview_only": True,
        "canonical_state_mutated": False,
        "input_versions": {
            "bank": inputs["prompt_bank"].bank_version,
            "router": inputs["router_policy"].router_version,
            "scaffold": inputs["scaffold_policy"].scaffold_version,
            "contract": inputs["scaffold_contract"].contract_version,
        },
        "conditions": condition_summaries,
        "artifacts": artifacts,
        "calls": {
            "captioning": 0,
            "dvd_reasoning": 0,
            "feedback_models": 0,
            "proposal_models": 0,
            "optimization": 0,
        },
    }, sort_keys=True, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
