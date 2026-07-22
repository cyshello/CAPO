#!/usr/bin/env python3
"""Run one Checkpoint F prompt-delta iteration with injected components."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from surrogate_rollout.optimization.episode_feedback import (
    DeterministicMockEpisodeFeedbackGenerator,
)
from surrogate_rollout.optimization.meta_prompt_updater import (
    DeterministicMockMetaPromptUpdater,
)
from surrogate_rollout.optimization.meta_prompt_defaults import (
    resolve_meta_prompt_artifact_path,
)
from surrogate_rollout.optimization.prompt_delta_iteration import (
    DeterministicMockMetaPromptConfirmationEvaluator,
    MetaPromptConfirmationCase,
    MetaPromptConfirmationCriterion,
    MetaPromptConfirmationOutcome,
    PromptDeltaIterationOrchestrator,
)
from surrogate_rollout.optimization.schemas import (
    intervention_episode_from_json,
    meta_prompt_version_from_json,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


def _object(path: str):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _array(path: str):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return value


def _factory(value: str, args: argparse.Namespace):
    if ":" not in value:
        raise ValueError("component factory must use module:callable")
    module_name, name = value.split(":", 1)
    factory = getattr(importlib.import_module(module_name), name)
    components = factory(args)
    if not isinstance(components, tuple) or len(components) != 3:
        raise TypeError(
            "component factory must return (feedback_generator, updater, "
            "confirmation_evaluator)")
    return components


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Collect grounded feedback, update one meta-prompt, run paired "
            "confirmation, and atomically promote or roll back. Real "
            "components are injected by an explicit module:callable factory."))
    parser.add_argument("--iteration-id", required=True)
    parser.add_argument(
        "--parent-meta-prompt",
        help=("MetaPromptVersion JSON. Omit to use "
              "optimization/prompts/init_meta_prompt.json."))
    parser.add_argument(
        "--update-episode", required=True, action="append",
        help="Ordered InterventionEpisode JSON; repeat for each update episode.")
    parser.add_argument("--confirmation-cases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument(
        "--feedback-memory-bank-dir", required=True,
        help=("Cumulative compact feedback memory bank directory; immutable "
              "snapshots are appended idempotently across iterations."))
    parser.add_argument(
        "--historical-memory-character-budget", type=int,
        help=("Optional explicit canonical-character budget for prior-iteration "
              "memory text. Omit to include every prior memory for the parent."))
    parser.add_argument("--candidate-created-at", required=True)
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--decoding-settings", required=True)
    parser.add_argument("--cache-reset-identity", required=True)
    parser.add_argument("--evaluation-pipeline-identity", required=True)
    parser.add_argument("--worker-result-timeout-seconds", type=float)
    parser.add_argument(
        "--worker-gpus",
        help=("Explicit comma-separated execution-only GPU override passed "
              "to the real confirmation component factory."))
    parser.add_argument("--minimum-confirmation-samples", required=True, type=int)
    parser.add_argument("--minimum-accuracy-delta", required=True, type=float)
    parser.add_argument("--maximum-correct-to-wrong", required=True, type=int)
    parser.add_argument(
        "--require-no-execution-failures", required=True,
        choices=("true", "false"))
    parser.add_argument("--initialize-parent-pointer", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--component-factory")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--component-config",
        help=("Explicit JSON configuration consumed by the selected real "
              "component factory; never used in --dry-run mode."))
    parser.add_argument("--mock-candidate-meta-prompt")
    parser.add_argument("--mock-confirmation-outcomes")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        parent_path = resolve_meta_prompt_artifact_path(
            args.parent_meta_prompt)
        parent = meta_prompt_version_from_json(_object(str(parent_path)))
        episodes = tuple(intervention_episode_from_json(_object(path))
                         for path in args.update_episode)
        cases = tuple(MetaPromptConfirmationCase(**item)
                      for item in _array(args.confirmation_cases))
        decoding = _object(args.decoding_settings)
        if args.dry_run:
            if args.mock_candidate_meta_prompt is not None:
                if not args.mock_confirmation_outcomes:
                    raise ValueError(
                        "dry-run update requires --mock-confirmation-outcomes")
                outcomes = tuple(MetaPromptConfirmationOutcome(**item)
                                 for item in _array(
                                     args.mock_confirmation_outcomes))
            else:
                outcomes = ()
            components = (
                DeterministicMockEpisodeFeedbackGenerator(),
                DeterministicMockMetaPromptUpdater(
                    candidate_meta_prompt=args.mock_candidate_meta_prompt),
                DeterministicMockMetaPromptConfirmationEvaluator(outcomes),
            )
        else:
            if not args.component_config:
                raise ValueError(
                    "real component factory requires --component-config")
            components = _factory(args.component_factory, args)
        result = PromptDeltaIterationOrchestrator(
            feedback_generator=components[0], updater=components[1],
            confirmation_evaluator=components[2]).run(
                iteration_id=args.iteration_id, parent=parent,
                update_episodes=episodes, confirmation_cases=cases,
                criterion=MetaPromptConfirmationCriterion(
                    minimum_sample_count=args.minimum_confirmation_samples,
                    minimum_accuracy_delta=args.minimum_accuracy_delta,
                    maximum_correct_to_wrong=args.maximum_correct_to_wrong,
                    require_no_execution_failures=(
                        args.require_no_execution_failures == "true")),
                model_identity=args.model_identity,
                decoding_settings=decoding,
                cache_reset_identity=args.cache_reset_identity,
                evaluation_pipeline_identity=(
                    args.evaluation_pipeline_identity),
                candidate_created_at=args.candidate_created_at,
                output_directory=args.output_dir,
                state_directory=args.state_dir,
                feedback_memory_bank_directory=args.feedback_memory_bank_dir,
                historical_memory_character_budget=(
                    args.historical_memory_character_budget),
                initialize_parent_pointer=args.initialize_parent_pointer)
    except Exception as exc:
        print(f"prompt-delta iteration failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    print(dumps_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
