#!/usr/bin/env python3
"""Prepare immutable inputs for one fresh prompt-delta production iteration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from surrogate_rollout import config
from surrogate_rollout.optimization.fresh_prompt_delta_evidence import (
    DEFAULT_PROMPT_DELTA_PROPOSAL_TARGET_POLICY,
    PROMPT_DELTA_FRAME_INSPECTION_CLASSIFICATION_POLICY,
    PROMPT_DELTA_PROPOSAL_TARGET_POLICIES,
    PROMPT_DELTA_PROPOSAL_EVIDENCE_SCOPE,
    PROMPT_DELTA_PROPOSAL_REPRESENTATION_VERSION,
    PROMPT_DELTA_PROPOSAL_SPLIT_POLICY,
)
from surrogate_rollout.optimization.meta_prompt_defaults import (
    INITIAL_META_PROMPT_PATH,
    resolve_meta_prompt_artifact_path,
)
from surrogate_rollout.optimization.schemas import meta_prompt_version_from_json
from surrogate_rollout.optimization.train_roles import derive_train_roles
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    _atomic_write_text(str(path), dumps_canonical(value) + "\n")


def _video_id_list(path: Path) -> tuple[str, ...]:
    values = tuple(line.strip() for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"video list must be non-empty and unique: {path}")
    return values


def _resolve_video_records(
    *, video_ids: tuple[str, ...], provider, benchmark: str,
    benchmark_split: str,
) -> tuple[dict, ...]:
    wanted = set(video_ids)
    grouped: dict[str, list[tuple[int, dict]]] = {item: [] for item in video_ids}
    for index in range(len(provider)):
        sample = dict(provider[index])
        video_id = str(sample.get("extra", {}).get("videoID") or
                       sample.get("sample_id") or "")
        if video_id in wanted:
            grouped[video_id].append((index, sample))
    missing = tuple(video_id for video_id in video_ids if not grouped[video_id])
    if missing:
        raise ValueError(f"cohort videos are absent from provider: {missing}")
    records = []
    for video_id in video_ids:
        rows = grouped[video_id]
        if len(rows) != 3:
            raise ValueError(
                f"cohort video requires exactly three QAs: {video_id}")
        indices = tuple(index for index, _ in rows)
        records.append({
            "video_id": video_id,
            "provider_indices": indices,
            "question_ids": tuple(
                f"{benchmark}/{benchmark_split}/{index}" for index in indices),
            "previously_cached": False,
        })
    return tuple(records)


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parent-meta-prompt",
        help=("MetaPromptVersion JSON. Omit to use "
              "optimization/prompts/init_meta_prompt.json."))
    p.add_argument(
        "--active-pointer",
        help="Optional pointer that must match the selected parent artifact.")
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--scaffold-components", required=True)
    p.add_argument("--video-id", action="append", required=True)
    p.add_argument("--previous-update-video-id", action="append", default=[])
    p.add_argument("--evidence-video-list")
    p.add_argument("--confirmation-video-list")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--api-endpoint", required=True)
    p.add_argument("--api-key-environment-variable", required=True)
    p.add_argument("--timeout-seconds", required=True, type=int)
    p.add_argument("--proposer-model-id", required=True)
    p.add_argument("--proposer-context-limit", required=True, type=int)
    p.add_argument("--proposer-maximum-output-tokens", required=True, type=int)
    p.add_argument("--proposer-temperature", required=True, type=float)
    p.add_argument("--proposer-policy-version", required=True)
    p.add_argument("--maximum-deltas-per-qa", required=True, type=int)
    p.add_argument("--selection-policy", required=True)
    p.add_argument(
        "--proposal-target-policy",
        choices=PROMPT_DELTA_PROPOSAL_TARGET_POLICIES,
        default=DEFAULT_PROMPT_DELTA_PROPOSAL_TARGET_POLICY,
        help=("Which baseline QAs may receive a delta proposal. Every QA of "
              "the video is still evaluated in each intervention episode."))
    p.add_argument(
        "--global-inspection-boundary-tolerance-seconds",
        required=True, type=float)
    p.add_argument("--feedback-model-id", required=True)
    p.add_argument("--feedback-context-limit", required=True, type=int)
    p.add_argument("--feedback-maximum-output-tokens", required=True, type=int)
    p.add_argument("--feedback-temperature", required=True, type=float)
    p.add_argument("--feedback-policy-version", required=True)
    p.add_argument("--updater-model-id", required=True)
    p.add_argument("--updater-context-limit", required=True, type=int)
    p.add_argument("--updater-maximum-output-tokens", required=True, type=int)
    p.add_argument("--updater-temperature", required=True, type=float)
    p.add_argument("--updater-policy-version", required=True)
    p.add_argument("--worker-gpus", required=True)
    p.add_argument("--worker-result-timeout-seconds", required=True, type=float)
    p.add_argument("--prompt-generator-model-id", required=True)
    p.add_argument("--prompt-generator-backend-id", required=True)
    p.add_argument("--prompt-generator-max-tokens", required=True, type=int)
    p.add_argument("--history-block-seconds", required=True, type=float)
    p.add_argument("--max-history-captions", required=True, type=int)
    p.add_argument("--dvd-max-iterations", required=True, type=int)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--cache-manifest-path", required=True)
    p.add_argument("--paired-model-identity", required=True)
    p.add_argument("--cache-reset-identity", required=True)
    p.add_argument("--evaluation-pipeline-identity", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = _args(argv)
    if a.global_inspection_boundary_tolerance_seconds < 0:
        raise ValueError(
            "global inspection boundary tolerance must be non-negative")
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    parent_path = resolve_meta_prompt_artifact_path(a.parent_meta_prompt)
    pointer_path = Path(a.active_pointer).resolve() if a.active_pointer else None
    split_path = Path(a.split_manifest).resolve()
    components_path = Path(a.scaffold_components).resolve()
    evidence_list_path = (Path(a.evidence_video_list).resolve()
                          if a.evidence_video_list else None)
    confirmation_list_path = (Path(a.confirmation_video_list).resolve()
                              if a.confirmation_video_list else None)
    if (evidence_list_path is None) != (confirmation_list_path is None):
        raise ValueError(
            "evidence and confirmation video lists must be supplied together")
    sources = tuple(path for path in (
        parent_path, pointer_path, split_path, components_path,
        evidence_list_path, confirmation_list_path)
        if path is not None)
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    before = {str(path): _sha(path) for path in sources}
    parent = meta_prompt_version_from_json(json.loads(parent_path.read_text()))
    if pointer_path is not None:
        pointer = json.loads(pointer_path.read_text())
        if pointer.get("active_meta_prompt_id") != parent.meta_prompt_id or \
                pointer.get("artifact_sha256") != _sha(parent_path):
            raise ValueError("parent artifact does not match active pointer")
    roles = derive_train_roles(json.loads(split_path.read_text()))
    source_components = json.loads(components_path.read_text())
    if source_components.get("scaffold_policy", {}).get("policy_type") != \
            "replace_body":
        raise ValueError(
            "fresh prompt-delta requires the replace_body scaffold policy")
    scaffold_only_path = out / "scaffold_components.json"
    _write(scaffold_only_path, {
        "schema_version": "prompt_delta_scaffold_components_v1",
        "scaffold_policy": source_components["scaffold_policy"],
        "scaffold_contract": source_components["scaffold_contract"],
        "source_path": str(components_path),
        "source_sha256": _sha(components_path),
        "legacy_property_codebook_or_router_included": False,
    })
    if evidence_list_path is None:
        evidence_records = tuple({
            "video_id": item.video_id,
            "provider_indices": item.provider_indices,
            "question_ids": item.question_ids,
            "previously_cached": item.previously_cached,
        } for item in roles.evidence_videos)
        confirmation_records = tuple({
            "video_id": item.video_id,
            "provider_indices": item.provider_indices,
            "question_ids": item.question_ids,
            "previously_cached": item.previously_cached,
        } for item in roles.confirmation_videos)
        cohort_policy = "active_split_manifest_8_evidence_2_confirmation"
    else:
        for path in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
            if path not in sys.path:
                sys.path.insert(0, path)
        from data_provider import get_provider
        provider = get_provider(config.BENCHMARK, split=config.BENCHMARK_SPLIT)
        evidence_ids = _video_id_list(evidence_list_path)
        confirmation_ids = _video_id_list(confirmation_list_path)
        if set(evidence_ids) & set(confirmation_ids):
            raise ValueError("evidence and confirmation cohorts overlap")
        evidence_records = _resolve_video_records(
            video_ids=evidence_ids, provider=provider,
            benchmark=config.BENCHMARK,
            benchmark_split=config.BENCHMARK_SPLIT)
        confirmation_records = _resolve_video_records(
            video_ids=confirmation_ids, provider=provider,
            benchmark=config.BENCHMARK,
            benchmark_split=config.BENCHMARK_SPLIT)
        cohort_policy = "explicit_video_lists_v1"
    evidence = tuple(item["video_id"] for item in evidence_records)
    confirmation = tuple(item["video_id"] for item in confirmation_records)
    selected = tuple(a.video_id)
    previous = tuple(a.previous_update_video_id)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("select a non-empty unique update-video batch")
    if set(selected) - set(evidence) or set(previous) - set(evidence):
        raise ValueError("update video is outside frozen evidence pool")
    if set(selected) & set(previous):
        raise ValueError("selected video was already used as update evidence")
    if set(selected) & set(confirmation):
        raise ValueError("update and confirmation videos overlap")
    selected_roles = tuple(
        video for video in evidence_records if video["video_id"] in set(selected))
    proposer_call_budget = sum(len(video["question_ids"])
                               for video in selected_roles)
    if a.maximum_deltas_per_qa not in (1, 2):
        raise ValueError("maximum deltas per QA must be 1 or 2")
    cases = [{
        "case_id": f"confirmation_{video['video_id']}_{qa_id}",
        "video_id": video["video_id"], "qa_id": qa_id,
        "input_ref": f"{split_path}#provider_index={index}",
    } for video in confirmation_records
      for qa_id, index in zip(video["question_ids"],
                              video["provider_indices"])]
    _write(out / "confirmation_cases.json", cases)
    paired = {
        "captioner": config.CAPTION_DECODING,
        "prompt_generator": {
            "model_id": a.prompt_generator_model_id,
            "backend_id": a.prompt_generator_backend_id,
            "max_tokens": a.prompt_generator_max_tokens,
        },
        "dvd": {
            "orchestrator_tool_model": config.ORCHESTRATOR_TOOL_MODEL,
            "text_fallback_model": config.TEXT_FALLBACK_MODEL,
            "text_backend": config.DVD_TEXT_BACKEND,
            "use_openai_tools": config.DVD_USE_OPENAI_TOOLS,
            "max_iterations": a.dvd_max_iterations,
            "frame_inspect_tool_contract_version":
                config.DVD_FRAME_INSPECT_TOOL_CONTRACT_VERSION,
            "frame_inspect_corrective_retry_limit":
                config.DVD_FRAME_INSPECT_CORRECTIVE_RETRY_LIMIT,
        },
    }
    _write(out / "paired_decoding_settings.json", paired)
    component = {
        "schema_version": "fresh_prompt_delta_component_config_v1",
        "provider": {"name": "openai_api", "api_endpoint": a.api_endpoint,
                     "api_key_environment_variable": a.api_key_environment_variable,
                     "timeout_seconds": a.timeout_seconds},
        "prompt_delta_proposer": {
            "model_id": a.proposer_model_id,
            "context_limit": a.proposer_context_limit,
            "maximum_output_tokens": a.proposer_maximum_output_tokens,
            "generation_settings": {"temperature": a.proposer_temperature},
            "policy_version": a.proposer_policy_version,
            "maximum_calls": proposer_call_budget,
            "maximum_deltas_per_qa": a.maximum_deltas_per_qa,
            "selection_policy": a.selection_policy,
            "proposal_target_policy": a.proposal_target_policy,
            "frame_inspection_classification_policy":
                PROMPT_DELTA_FRAME_INSPECTION_CLASSIFICATION_POLICY,
            "global_inspection_boundary_tolerance_seconds":
                a.global_inspection_boundary_tolerance_seconds,
            "representation_version":
                PROMPT_DELTA_PROPOSAL_REPRESENTATION_VERSION,
            "evidence_scope": PROMPT_DELTA_PROPOSAL_EVIDENCE_SCOPE,
            "split_policy": PROMPT_DELTA_PROPOSAL_SPLIT_POLICY,
        },
        "feedback": {
            "model_id": a.feedback_model_id,
            "context_limit": a.feedback_context_limit,
            "maximum_output_tokens": a.feedback_maximum_output_tokens,
            "generation_settings": {"temperature": a.feedback_temperature},
            "policy_version": a.feedback_policy_version,
        },
        "updater": {
            "model_id": a.updater_model_id,
            "context_limit": a.updater_context_limit,
            "maximum_output_tokens": a.updater_maximum_output_tokens,
            "generation_settings": {"temperature": a.updater_temperature},
            "policy_version": a.updater_policy_version,
        },
        "runtime": {
            "benchmark": config.BENCHMARK,
            "benchmark_split": config.BENCHMARK_SPLIT,
            "captioner_model_id": config.CAPTION_MODEL_ID,
            "prompt_generator_model_id": a.prompt_generator_model_id,
            "prompt_generator_backend_id": a.prompt_generator_backend_id,
            "prompt_generator_max_tokens": a.prompt_generator_max_tokens,
            "dvd_orchestrator_tool_model": config.ORCHESTRATOR_TOOL_MODEL,
            "dvd_text_fallback_model": config.TEXT_FALLBACK_MODEL,
            "dvd_text_backend": config.DVD_TEXT_BACKEND,
            "dvd_use_openai_tools": config.DVD_USE_OPENAI_TOOLS,
            "use_transcript": config.USE_TRANSCRIPT,
            "sample_fps": config.SAMPLE_FPS, "clip_seconds": config.CLIP_SECS,
            "caption_decoding": config.CAPTION_DECODING,
            "worker_gpus": [x.strip() for x in a.worker_gpus.split(",") if x.strip()],
            "worker_result_timeout_seconds": a.worker_result_timeout_seconds,
            "scaffold_components_path": str(scaffold_only_path),
            "sample_source_identity": _sha(split_path),
            "cache_root": str(Path(a.cache_root).resolve()),
            "cache_manifest_path": str(Path(a.cache_manifest_path).resolve()),
            "history_block_seconds": a.history_block_seconds,
            "max_history_captions": a.max_history_captions,
            "dvd_max_iterations": a.dvd_max_iterations,
            "dvd_frame_inspect_tool_contract_version":
                config.DVD_FRAME_INSPECT_TOOL_CONTRACT_VERSION,
            "dvd_frame_inspect_corrective_retry_limit":
                config.DVD_FRAME_INSPECT_CORRECTIVE_RETRY_LIMIT,
            "paired_model_identity": a.paired_model_identity,
            "paired_decoding_settings": paired,
            "cache_reset_identity": a.cache_reset_identity,
            "evaluation_pipeline_identity": a.evaluation_pipeline_identity,
        },
        "confirmation_videos": [{
            "video_id": item["video_id"],
            "provider_indices": item["provider_indices"],
            "question_ids": item["question_ids"],
        } for item in confirmation_records],
    }
    _write(out / "component_config.json", component)
    after = {str(path): _sha(path) for path in sources}
    if before != after:
        raise RuntimeError("source changed during preparation")
    _write(out / "manifest.json", {
        "schema_version": "fresh_prompt_delta_iteration_inputs_v1",
        "status": "prepared", "parent_meta_prompt_id": parent.meta_prompt_id,
        "parent_meta_prompt_path": str(parent_path),
        "initial_meta_prompt_default_used": a.parent_meta_prompt is None,
        "initial_meta_prompt_default_path": str(INITIAL_META_PROMPT_PATH),
        "selected_video_ids": selected,
        "selected_video_records": tuple(
            {item["video_id"]: item for item in evidence_records}[video_id]
            for video_id in selected),
        "previous_update_video_ids": previous,
        "confirmation_video_ids": confirmation,
        "evidence_cohort_video_ids": evidence,
        "cohort_policy": cohort_policy,
        "evidence_video_list_path": (str(evidence_list_path)
                                     if evidence_list_path else None),
        "confirmation_video_list_path": (str(confirmation_list_path)
                                         if confirmation_list_path else None),
        "component_config_path": str(out / "component_config.json"),
        "confirmation_cases_path": str(out / "confirmation_cases.json"),
        "paired_decoding_settings_path": str(out / "paired_decoding_settings.json"),
        "source_hashes_before": before, "source_hashes_after": after,
        "model_or_api_calls": 0,
        "legacy_property_codebook_or_router_used": False,
    })
    print(dumps_canonical({"output_directory": str(out),
                           "selected_video_ids": selected}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
