#!/usr/bin/env python
"""Re-run feedback and the meta-prompt update for one completed evidence run.

Operator-run only: this performs paid provider calls.

It reads the episodes an evidence run already produced and writes a fresh
feedback + updater stage into its own output directory. It never touches the
evidence run, the experiment state pointer, or an existing iteration output, so
a re-run under a new prompt/model cannot disturb what is already recorded. No
promotion and no held-out measurement happen here.
"""

from __future__ import annotations

import argparse
import json

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from surrogate_rollout.optimization.checkpoint_g_factory import (  # noqa: E402
    build_checkpoint_g_components,
)
from surrogate_rollout.optimization.episode_feedback import (  # noqa: E402
    evaluate_episode_feedback_eligibility,
)
from surrogate_rollout.optimization.schemas import (  # noqa: E402
    intervention_episode_from_json,
    meta_prompt_version_from_json,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical  # noqa: E402


def _object(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_canonical(value) + "\n", encoding="utf-8")


def candidate_record(update) -> dict | None:
    """The candidate the updater produced, read off the decision.

    `MetaPromptUpdateResult` carries the candidate's id and status; the text
    itself lives on the decision. Reaching for a `candidate` attribute on the
    result raised AttributeError after the result had already been written,
    losing only this file -- but losing it silently at the very end of a paid
    run.
    """
    text = update.decision.candidate_meta_prompt
    if text is None:
        return None
    return {
        "candidate_meta_prompt_id": update.candidate_meta_prompt_id,
        "candidate_status": update.candidate_status,
        "text": text,
        "line_count": len(text.splitlines()),
    }


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-manifest", required=True)
    parser.add_argument("--parent-meta-prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--component-config", default=None)
    parser.add_argument(
        "--feedback-model-id", default=None,
        help="Overrides feedback.model_id from the resolved component config.")
    parser.add_argument("--updater-model-id", default=None)
    parser.add_argument("--feedback-maximum-output-tokens", type=int,
                        default=None)
    parser.add_argument("--updater-maximum-output-tokens", type=int,
                        default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _args(argv)
    manifest = _object(args.evidence_manifest)
    component_config_path = (
        args.component_config or manifest["resolved_component_config_path"])
    configuration = _object(component_config_path)

    for section, model_id, budget in (
            ("feedback", args.feedback_model_id,
             args.feedback_maximum_output_tokens),
            ("updater", args.updater_model_id,
             args.updater_maximum_output_tokens)):
        if model_id is not None:
            configuration[section]["model_id"] = model_id
        if budget is not None:
            configuration[section]["maximum_output_tokens"] = budget

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    resolved_path = output / "resolved_component_config.json"
    _write(resolved_path, configuration)

    generator, updater, _confirmation = build_checkpoint_g_components(
        argparse.Namespace(component_config=str(resolved_path)))

    parent = meta_prompt_version_from_json(_object(args.parent_meta_prompt))
    episode_paths = list(manifest["episode_paths"])
    print(f"episodes: {len(episode_paths)}", flush=True)

    feedbacks = []
    grounding = []
    for index, episode_path in enumerate(episode_paths):
        episode = intervention_episode_from_json(_object(episode_path))
        stage = output / "feedback" / f"{index:03d}_{episode.episode_id}"
        stage.mkdir(parents=True, exist_ok=True)
        feedback = generator.generate_to_directory(episode, str(stage))
        eligibility = evaluate_episode_feedback_eligibility(feedback, episode)
        _write(stage / "eligibility.json", eligibility)
        feedbacks.append(feedback)
        grounding.append(_object(str(stage / "grounding.json"))
                         if (stage / "grounding.json").exists() else {})
        print(f"  [{index + 1}/{len(episode_paths)}] {episode.episode_id} "
              f"attribution={feedback.attribution_status} "
              f"eligible={eligibility.eligible}", flush=True)

    update = updater.update(
        parent, tuple(feedbacks),
        feedback_grounding=tuple(grounding) if any(grounding) else None,
        historical_memories=(),
        current_iteration_id=f"feedback_update_only_{output.name}")
    _write(output / "updater_result.json", {
        "schema_version": "feedback_and_update_only_v1",
        "request": update.request.payload,
        "request_payload_hash": update.request.payload_hash,
        "decision": update.decision,
        "candidate_meta_prompt_id": update.candidate_meta_prompt_id,
        "candidate_status": update.candidate_status,
        "backend_metadata": update.backend_metadata,
        "raw_response": update.raw_response,
    })
    candidate = candidate_record(update)
    if candidate is not None:
        _write(output / "candidate_meta_prompt.json", candidate)
    print(f"decision: {update.decision.decision}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
