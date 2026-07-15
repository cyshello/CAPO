"""Run the Stage 4.9 deterministic feedback checkpoint offline."""

from __future__ import annotations

import argparse
import json
import os

from surrogate_rollout.optimization.evidence_builder import read_evidence_jsonl
from surrogate_rollout.optimization.failure_attributor import (
    DeterministicFailureAttributor,
)
from surrogate_rollout.optimization.feedback_generator import (
    attach_attributions,
    write_feedback_batch,
)
from surrogate_rollout.optimization.meta_knowledge import MetaKnowledgeStore
from surrogate_rollout.optimization.policies.mock_feedback import (
    MockFeedbackGenerator,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline deterministic Stage 4.9 feedback checkpoint")
    parser.add_argument("--evidence-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    evidence = read_evidence_jsonl(args.evidence_jsonl)
    feedback_dir = os.path.join(args.output_dir, "feedback")
    attribution_dir = os.path.join(args.output_dir, "attribution")
    knowledge_dir = os.path.join(args.output_dir, "meta_knowledge")
    for path in (feedback_dir, attribution_dir, knowledge_dir):
        os.makedirs(path, exist_ok=True)

    request_path = os.path.join(feedback_dir, "request.json")
    _atomic_write_text(request_path, json.dumps({
        "feedback_policy": "deterministic_mock",
        "input_evidence_ids": [item.evidence_id for item in evidence],
        "real_model_enabled": False,
    }, sort_keys=True, indent=2) + "\n")

    initial_feedback = MockFeedbackGenerator().generate(evidence, ())
    attributions = DeterministicFailureAttributor().attribute(
        evidence, initial_feedback)
    feedback = attach_attributions(initial_feedback, attributions)
    feedback_path = write_feedback_batch(
        feedback, os.path.join(feedback_dir, "parsed_feedback.json"))

    attribution_path = os.path.join(
        attribution_dir, "failure_attributions.jsonl")
    _atomic_write_text(
        attribution_path,
        "".join(dumps_canonical(item) + "\n" for item in attributions),
    )

    knowledge_path = os.path.join(knowledge_dir, "meta_knowledge.json")
    store = MetaKnowledgeStore(knowledge_path)
    store.ingest(evidence, feedback, attributions)
    store.save()

    summary_path = os.path.join(args.output_dir, "stage4_9_summary.json")
    _atomic_write_text(summary_path, json.dumps({
        "schema_version": "phase4_stage4.9_v1",
        "evidence_count": len(evidence),
        "feedback_item_count": len(feedback.items),
        "attribution_count": len(attributions),
        "attribution_targets": [item.targets for item in attributions],
        "knowledge_count": len(store.list_items()),
        "knowledge_statuses": [item.status for item in store.list_items()],
        "real_model_enabled": False,
        "calls": {
            "captioning": 0,
            "dvd_reasoning": 0,
            "feedback_models": 0,
            "optimization": 0,
        },
        "artifacts": {
            "request": request_path,
            "feedback": feedback_path,
            "attributions": attribution_path,
            "meta_knowledge": knowledge_path,
        },
    }, sort_keys=True, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
