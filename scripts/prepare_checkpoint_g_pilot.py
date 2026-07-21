#!/usr/bin/env python3
"""Prepare immutable Checkpoint G inputs without model/API execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from surrogate_rollout import config
from surrogate_rollout.optimization.legacy_intervention_adapter import (
    legacy_property_intervention_to_episode,
)
from surrogate_rollout.optimization.train_roles import derive_train_roles
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


def _args(argv=None):
    parser = argparse.ArgumentParser(
        description=("Materialize read-only legacy episodes, the active "
                     "two-video confirmation manifest, and explicit real "
                     "Checkpoint G component config. No model is called."))
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--parent-meta-prompt-id", required=True)
    parser.add_argument("--intervention-result", action="append", required=True)
    parser.add_argument("--baseline-manifest", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-endpoint", required=True)
    parser.add_argument("--api-key-environment-variable", required=True)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--feedback-model-id", required=True)
    parser.add_argument("--feedback-context-limit", required=True, type=int)
    parser.add_argument("--feedback-maximum-output-tokens", required=True, type=int)
    parser.add_argument("--feedback-temperature", required=True, type=float)
    parser.add_argument("--feedback-policy-version", required=True)
    parser.add_argument("--updater-model-id", required=True)
    parser.add_argument("--updater-maximum-output-tokens", required=True, type=int)
    parser.add_argument("--updater-temperature", required=True, type=float)
    parser.add_argument("--updater-policy-version", required=True)
    parser.add_argument("--worker-gpus", required=True)
    parser.add_argument("--prompt-generator-backend-id", required=True)
    parser.add_argument("--prompt-generator-max-tokens", required=True, type=int)
    parser.add_argument("--scaffold-components", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cache-manifest-path", required=True)
    parser.add_argument("--history-block-seconds", required=True, type=float)
    parser.add_argument("--max-history-captions", required=True, type=int)
    parser.add_argument("--dvd-max-iterations", required=True, type=int)
    parser.add_argument("--paired-model-identity", required=True)
    parser.add_argument("--cache-reset-identity", required=True)
    parser.add_argument("--evaluation-pipeline-identity", required=True)
    return parser.parse_args(argv)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    _atomic_write_text(str(path), dumps_canonical(value) + "\n")


def main(argv=None) -> int:
    args = _args(argv)
    if len(args.intervention_result) != len(args.baseline_manifest):
        raise ValueError("intervention and baseline arguments must pair in order")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    split = Path(args.split_manifest).resolve()
    components = Path(args.scaffold_components).resolve()
    primary_sources = (split, components, *(Path(v).resolve()
                       for v in args.intervention_result), *(Path(v).resolve()
                       for v in args.baseline_manifest))
    for path in primary_sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    artifact_roots = {
        Path(value).resolve().parent for value in
        (*args.intervention_result, *args.baseline_manifest)
    }
    sources = tuple(sorted({split, components, *(
        path for root in artifact_roots for path in root.rglob("*")
        if path.is_file())}, key=str))
    source_before = {str(path): _hash(path) for path in sources}
    roles = derive_train_roles(json.loads(split.read_text(encoding="utf-8")))
    episodes_dir = output / "update_episodes"
    episodes_dir.mkdir()
    episode_paths = []
    update_video_ids = set()
    for index, (intervention, baseline) in enumerate(zip(
            args.intervention_result, args.baseline_manifest), 1):
        episode = legacy_property_intervention_to_episode(
            intervention_result_path=str(Path(intervention).resolve()),
            baseline_video_manifest_path=str(Path(baseline).resolve()),
            parent_meta_prompt_id=args.parent_meta_prompt_id)
        if episode.video_id in update_video_ids:
            raise ValueError("pilot update episodes must use unique videos")
        update_video_ids.add(episode.video_id)
        path = episodes_dir / f"{index:02d}_{episode.video_id}_{episode.episode_id}.json"
        _write(path, episode)
        episode_paths.append(str(path))
    confirmation_ids = {item.video_id for item in roles.confirmation_videos}
    overlap = update_video_ids & confirmation_ids
    if overlap:
        raise ValueError(f"update/confirmation video overlap: {sorted(overlap)}")
    cases = [{
        "case_id": f"confirmation_{video.video_id}_{qa_id}",
        "video_id": video.video_id, "qa_id": qa_id,
        "input_ref": f"{split}#provider_index={provider_index}",
    } for video in roles.confirmation_videos
      for qa_id, provider_index in zip(video.question_ids,
                                        video.provider_indices)]
    cases_path = output / "confirmation_cases.json"
    _write(cases_path, cases)
    decoding_path = output / "paired_decoding_settings.json"
    paired_decoding = {
        "captioner": config.CAPTION_DECODING,
        "prompt_generator": {
            "max_tokens": args.prompt_generator_max_tokens,
            "model_id": config.CAPTION_MODEL_ID,
            "backend_id": args.prompt_generator_backend_id,
        },
        "dvd": {
            "orchestrator_tool_model": config.ORCHESTRATOR_TOOL_MODEL,
            "text_fallback_model": config.TEXT_FALLBACK_MODEL,
            "max_iterations": args.dvd_max_iterations,
            "text_backend": config.DVD_TEXT_BACKEND,
            "use_openai_tools": config.DVD_USE_OPENAI_TOOLS,
        },
    }
    _write(decoding_path, paired_decoding)
    component_config = {
        "schema_version": "checkpoint_g_component_config_v1",
        "provider": {
            "name": "openai_api", "api_endpoint": args.api_endpoint,
            "api_key_environment_variable": args.api_key_environment_variable,
            "timeout_seconds": args.timeout_seconds,
        },
        "feedback": {
            "model_id": args.feedback_model_id,
            "context_limit": args.feedback_context_limit,
            "maximum_output_tokens": args.feedback_maximum_output_tokens,
            "generation_settings": {"temperature": args.feedback_temperature},
            "policy_version": args.feedback_policy_version,
            "maximum_calls": len(episode_paths),
        },
        "updater": {
            "model_id": args.updater_model_id,
            "maximum_output_tokens": args.updater_maximum_output_tokens,
            "generation_settings": {"temperature": args.updater_temperature},
            "policy_version": args.updater_policy_version,
        },
        "runtime": {
            "benchmark": config.BENCHMARK,
            "benchmark_split": config.BENCHMARK_SPLIT,
            "captioner_model_id": config.CAPTION_MODEL_ID,
            "prompt_generator_model_id": config.CAPTION_MODEL_ID,
            "prompt_generator_backend_id": args.prompt_generator_backend_id,
            "prompt_generator_max_tokens": args.prompt_generator_max_tokens,
            "dvd_orchestrator_tool_model": config.ORCHESTRATOR_TOOL_MODEL,
            "dvd_text_fallback_model": config.TEXT_FALLBACK_MODEL,
            "dvd_text_backend": config.DVD_TEXT_BACKEND,
            "dvd_use_openai_tools": config.DVD_USE_OPENAI_TOOLS,
            "use_transcript": config.USE_TRANSCRIPT,
            "sample_fps": config.SAMPLE_FPS,
            "clip_seconds": config.CLIP_SECS,
            "caption_decoding": config.CAPTION_DECODING,
            "worker_gpus": [v.strip() for v in args.worker_gpus.split(",") if v.strip()],
            "scaffold_components_path": str(components),
            "sample_source_identity": _hash(split),
            "cache_root": str(Path(args.cache_root).resolve()),
            "cache_manifest_path": str(Path(args.cache_manifest_path).resolve()),
            "history_block_seconds": args.history_block_seconds,
            "max_history_captions": args.max_history_captions,
            "dvd_max_iterations": args.dvd_max_iterations,
            "paired_model_identity": args.paired_model_identity,
            "paired_decoding_settings": paired_decoding,
            "cache_reset_identity": args.cache_reset_identity,
            "evaluation_pipeline_identity": args.evaluation_pipeline_identity,
        },
        "confirmation_videos": [{
            "video_id": item.video_id,
            "provider_indices": item.provider_indices,
            "question_ids": item.question_ids,
        } for item in roles.confirmation_videos],
    }
    config_path = output / "component_config.json"
    _write(config_path, component_config)
    source_after = {str(path): _hash(path) for path in sources}
    if source_before != source_after:
        raise RuntimeError("source artifact changed during preparation")
    _write(output / "manifest.json", {
        "schema_version": "checkpoint_g_pilot_inputs_v1",
        "status": "prepared", "update_episode_paths": episode_paths,
        "confirmation_cases_path": str(cases_path),
        "component_config_path": str(config_path),
        "paired_decoding_settings_path": str(decoding_path),
        "confirmation_video_ids": tuple(item.video_id
                                        for item in roles.confirmation_videos),
        "source_hashes_before": source_before,
        "source_hashes_after": source_after,
        "model_or_api_calls": 0,
    })
    print(dumps_canonical({
        "output_directory": str(output), "update_episode_paths": episode_paths,
        "confirmation_cases_path": str(cases_path),
        "component_config_path": str(config_path),
        "paired_decoding_settings_path": str(decoding_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
