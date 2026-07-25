#!/usr/bin/env python3
"""Run fresh free-form baseline, prompt-delta proposals, and interventions.

This launcher deliberately stops before episode feedback/updating.  Its output
manifest supplies newly generated InterventionEpisode paths to the existing
Checkpoint F/G orchestrator.  It never reads or writes a legacy property bank
or router state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from surrogate_rollout import config
from surrogate_rollout.captioning.history_aware_baseline import (
    HistoryAwareBaselineCaptionViewBuilder,
    FREE_FORM_PROVIDER_OPENAI,
    build_free_form_generator,
)
from surrogate_rollout.optimization.baseline_phase import (
    BaselinePhaseRunner,
    load_completed_baseline_for_read_only_resume,
)
from surrogate_rollout.prompt_routing import generator_response_cache
from surrogate_rollout.optimization.checkpoint_g_factory import (
    _BoundedOpenAITransport, _exact_counter,
)
from surrogate_rollout.optimization.fresh_prompt_delta_evidence import (
    LLMPromptDeltaProposer, OpenAICompatiblePromptDeltaProposalBackend,
    DEFAULT_PROMPT_DELTA_PROPOSAL_TARGET_POLICY,
    PROMPT_DELTA_FRAME_INSPECTION_CLASSIFICATION_POLICY,
    PROMPT_DELTA_PROPOSAL_TARGET_POLICIES,
    PROMPT_DELTA_INTERVENTION_EXECUTION_NAMESPACE,
    PROMPT_DELTA_PROPOSAL_EVIDENCE_SCOPE,
    PROMPT_DELTA_POLICY_OUTPUT_NAMESPACE,
    PROMPT_DELTA_PROPOSAL_REPRESENTATION_VERSION,
    PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
    PROMPT_DELTA_PROPOSAL_SPLIT_POLICY,
    PromptDeltaInterventionRunner,
)
from surrogate_rollout.optimization.schemas import (
    intervention_episode_from_json, meta_prompt_version_from_json,
)
from surrogate_rollout.optimization.meta_prompt_defaults import (
    resolve_meta_prompt_artifact_path,
)
from surrogate_rollout.optimization.train_roles import (
    EvidenceCoverageState, TrainVideoRecord, derive_train_roles,
)
from surrogate_rollout.prompt_routing.persistence import (
    _atomic_write_text, scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.schemas import (
    Phase4Config, Phase4OptimizationConfig, PostInterventionMode,
    dumps_canonical,
)


def _object(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.resolve()): _sha(path) for path in sorted(root.rglob("*"))
            if path.is_file()}


def _write_once(path: Path, value) -> str:
    text = dumps_canonical(value) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"immutable artifact conflict: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(str(path), text)
    return str(path.resolve())


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-inputs", required=True)
    p.add_argument(
        "--parent-meta-prompt",
        help=("MetaPromptVersion JSON. Omit to use "
              "optimization/prompts/init_meta_prompt.json."))
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--baseline-resume-dir",
        help=("Optional completed baseline root to validate and reuse read-only. "
              "A missing or incomplete external baseline fails closed."))
    p.add_argument("--source-revision", required=True)
    p.add_argument("--worker-result-timeout-seconds", required=True, type=float)
    p.add_argument(
        "--worker-gpus",
        help=("Explicit comma-separated execution-only GPU override. The "
              "prepared input remains immutable; omitted means use its "
              "recorded runtime.worker_gpus."))
    p.add_argument("--proposer-policy-version-override")
    p.add_argument("--selection-policy-override")
    p.add_argument(
        "--proposal-target-policy-override",
        choices=PROMPT_DELTA_PROPOSAL_TARGET_POLICIES)
    p.add_argument(
        "--global-inspection-boundary-tolerance-seconds-override", type=float)
    p.add_argument("--proposer-maximum-calls-override", type=int)
    p.add_argument("--maximum-deltas-per-qa-override", type=int)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = _args(argv)
    prepared = Path(a.prepared_inputs).resolve()
    input_manifest = _object(prepared / "manifest.json")
    component = _object(prepared / "component_config.json")
    parent_path = resolve_meta_prompt_artifact_path(a.parent_meta_prompt)
    split_path = Path(a.split_manifest).resolve()
    parent = meta_prompt_version_from_json(_object(parent_path))
    if parent.meta_prompt_id != input_manifest["parent_meta_prompt_id"]:
        raise ValueError("prepared input parent differs")
    selected = tuple(input_manifest["selected_video_ids"])
    explicit_selected_records = tuple(TrainVideoRecord(
        video_id=str(item["video_id"]),
        provider_indices=tuple(int(value) for value in item["provider_indices"]),
        question_ids=tuple(str(value) for value in item["question_ids"]),
        previously_cached=bool(item.get("previously_cached", False)),
    ) for item in input_manifest.get("selected_video_records") or ())
    if explicit_selected_records and tuple(
            item.video_id for item in explicit_selected_records) != selected:
        raise ValueError("prepared selected-video records are out of order")
    previous = tuple(input_manifest["previous_update_video_ids"])
    roles = derive_train_roles(_object(split_path))
    output = Path(a.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "fresh_evidence_manifest.json"
    if final_path.is_file():
        saved = _object(final_path)
        if tuple(saved["selected_video_ids"]) != selected:
            raise RuntimeError("saved fresh evidence has different videos")
        for path in saved["episode_paths"]:
            intervention_episode_from_json(_object(Path(path)))
        print(dumps_canonical(saved))
        return 0

    runtime = component["runtime"]
    # Generator answers are reused from this run's own cache root, so a restart
    # regenerates the identical instruction and the caption cache below it
    # still resolves. Set before any caption worker is spawned; the workers
    # inherit it.
    generator_response_cache.configure_default_root(runtime.get("cache_root"))
    provider_cfg = component["provider"]
    proposer_cfg = component["prompt_delta_proposer"]
    expected = {
        "captioner": runtime["captioner_model_id"],
        "prompt_generator": runtime["prompt_generator_model_id"],
        "dvd_tool": runtime["dvd_orchestrator_tool_model"],
        "dvd_fallback": runtime["dvd_text_fallback_model"],
        "proposer": proposer_cfg["model_id"],
    }
    for name, identity in expected.items():
        if not identity or any(word in str(identity).lower()
                               for word in ("mock", "fixture", "stub")):
            raise ValueError(f"{name} is not a reviewed real model identity")
    if runtime["prompt_generator_model_id"] != \
            config.PROMPT_GENERATOR_MODEL_ID or \
            runtime["prompt_generator_backend_id"] != \
            config.PROMPT_GENERATOR_BACKEND_ID:
        raise ValueError(
            "fresh prompt-delta requires the active OpenAI prompt generator")
    api_key = os.environ.get(provider_cfg["api_key_environment_variable"])
    if not api_key:
        raise ValueError("configured provider API key environment variable is absent")
    sources = (parent_path, split_path, prepared / "manifest.json",
               prepared / "component_config.json")
    source_before = {str(path): _sha(path) for path in sources}
    components = _object(Path(runtime["scaffold_components_path"]))
    scaffold = scaffold_policy_from_json(components["scaffold_policy"])
    contract = scaffold_contract_from_json(components["scaffold_contract"])
    if scaffold.policy_type != "replace_body":
        raise ValueError("fresh prompt-delta requires replace_body composition")
    for path in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from data_provider import get_provider
    from dvd_prompt import get_prompts
    from surrogate_rollout.evaluation.dvd_qa import ensure_backend

    if a.worker_gpus is None:
        gpus = tuple(str(value) for value in runtime["worker_gpus"])
    else:
        gpus = tuple(value.strip() for value in a.worker_gpus.split(",")
                     if value.strip())
    if not gpus:
        raise ValueError("worker_gpus is empty")
    if len(set(gpus)) != len(gpus):
        raise ValueError("worker_gpus must be unique")
    ensure_backend(
        gpus[0], preload_captioner=False, preload_embedder=True,
        text_backend=runtime["dvd_text_backend"],
        use_openai_tools=runtime["dvd_use_openai_tools"])
    provider_data = get_provider(runtime["benchmark"],
                                 split=runtime["benchmark_split"])
    prompts = get_prompts()
    builder = HistoryAwareBaselineCaptionViewBuilder.from_local_qwen(
        parallel_gpus=gpus, routing_mode="free_form_generator",
        free_form_provider=FREE_FORM_PROVIDER_OPENAI,
        free_form_model_id=runtime["prompt_generator_model_id"],
        worker_result_timeout_seconds=a.worker_result_timeout_seconds,
        worker_log_directory=str(output / "worker_logs"))
    builder.free_form_generator = build_free_form_generator(
        FREE_FORM_PROVIDER_OPENAI,
        max_tokens=int(runtime["prompt_generator_max_tokens"]),
        template_text=parent.text, meta_prompt_id=parent.meta_prompt_id,
        model_id=runtime["prompt_generator_model_id"],
        backend_id=runtime["prompt_generator_backend_id"])
    bank_version = "bank_v9999"
    router_version = "router_v9999"
    coverage = EvidenceCoverageState(
        rotation_order=tuple(item.video_id for item in roles.evidence_videos),
        used_since_confirmation=tuple(
            item.video_id for item in roles.evidence_videos
            if item.video_id in set(previous)),
        rotation_cursor=0, coverage_cycle=0,
        iterations_since_confirmation=1, confirmation_due=False)
    phase = Phase4Config(
        seed=0, dry_run=False, commit=False,
        post_intervention_mode=PostInterventionMode.PROVISIONAL_UPDATE,
        optimization=Phase4OptimizationConfig(
            optimize_prompt_bank=False, optimize_router=False,
            optimize_scaffold=False, max_iterations=1))
    proposer_policy_version = (
        a.proposer_policy_version_override or proposer_cfg["policy_version"])
    selection_policy = (
        a.selection_policy_override or proposer_cfg["selection_policy"])
    proposal_target_policy = (
        a.proposal_target_policy_override
        or proposer_cfg.get("proposal_target_policy")
        or DEFAULT_PROMPT_DELTA_PROPOSAL_TARGET_POLICY)
    global_inspection_tolerance = (
        a.global_inspection_boundary_tolerance_seconds_override
        if a.global_inspection_boundary_tolerance_seconds_override is not None
        else proposer_cfg.get("global_inspection_boundary_tolerance_seconds"))
    if global_inspection_tolerance is None or global_inspection_tolerance < 0:
        raise ValueError(
            "global inspection boundary tolerance must be configured")
    proposer_maximum_calls = (
        a.proposer_maximum_calls_override or proposer_cfg["maximum_calls"])
    maximum_deltas_per_qa = (
        a.maximum_deltas_per_qa_override or
        proposer_cfg["maximum_deltas_per_qa"])
    if proposer_maximum_calls <= 0 or maximum_deltas_per_qa not in (1, 2):
        raise ValueError("effective proposer call/delta limits must be positive")
    for key, expected_value in (
            ("representation_version",
             PROMPT_DELTA_PROPOSAL_REPRESENTATION_VERSION),
            ("evidence_scope", PROMPT_DELTA_PROPOSAL_EVIDENCE_SCOPE),
            ("split_policy", PROMPT_DELTA_PROPOSAL_SPLIT_POLICY)):
        configured = proposer_cfg.get(key)
        if configured is not None and configured != expected_value and \
                a.proposer_policy_version_override is None:
            raise ValueError(f"configured proposer {key} is unsupported")
    configured_classification = proposer_cfg.get(
        "frame_inspection_classification_policy")
    if configured_classification is not None and configured_classification != \
            PROMPT_DELTA_FRAME_INSPECTION_CLASSIFICATION_POLICY:
        raise ValueError(
            "configured frame inspection classification policy is unsupported")
    transport = _BoundedOpenAITransport(
        endpoint=provider_cfg["api_endpoint"], api_key=api_key,
        timeout_seconds=int(provider_cfg["timeout_seconds"]),
        maximum_calls=int(proposer_maximum_calls))
    proposer_counter, proposer_tokenizer_identity = _exact_counter(
        proposer_cfg["model_id"])
    proposer = LLMPromptDeltaProposer(
        backend=OpenAICompatiblePromptDeltaProposalBackend(
            provider=provider_cfg["name"],
            provider_endpoint=provider_cfg["api_endpoint"],
            model_id=proposer_cfg["model_id"],
            context_limit=int(proposer_cfg["context_limit"]),
            maximum_output_tokens=int(proposer_cfg["maximum_output_tokens"]),
            generation_settings=proposer_cfg["generation_settings"],
            policy_version=proposer_policy_version,
            exact_token_counter=proposer_counter,
            response_transport=transport.request,
            tokenizer_identity=proposer_tokenizer_identity,
            maximum_calls=proposer_maximum_calls),
        maximum_deltas_per_qa=int(maximum_deltas_per_qa),
        selection_policy=selection_policy,
        proposal_target_policy=proposal_target_policy,
        global_inspection_boundary_tolerance_seconds=float(
            global_inspection_tolerance))
    baseline_root = (Path(a.baseline_resume_dir).resolve()
                     if a.baseline_resume_dir else output / "baseline")
    external_baseline = a.baseline_resume_dir is not None
    try:
        baseline = load_completed_baseline_for_read_only_resume(
            str(baseline_root), selected_video_ids=selected)
        baseline_resume_mode = (
            ("validated_external_completed_immutable_baseline"
             if external_baseline else
             "validated_completed_immutable_baseline")
            if baseline is not None else "fresh_strict_dvd_execution")
        if baseline is None:
            if external_baseline:
                raise RuntimeError(
                    "external baseline resume root is incomplete or incompatible")
            baseline = BaselinePhaseRunner(history_aware_builder=builder).run(
                roles=roles, coverage_state=coverage,
                sample_loader=provider_data.__getitem__,
                bank_version=bank_version,
                router_version=router_version,
                scaffold_policy=scaffold,
                scaffold_contract=contract, phase4_config=phase,
                base_prompt_template=prompts.caption_prompt,
                merge_prompt=prompts.merge_prompt,
                output_dir=str(baseline_root),
                parent_confirmed_checkpoint_id=parent.meta_prompt_id,
                history_block_seconds=float(runtime["history_block_seconds"]),
                max_history_captions=int(runtime["max_history_captions"]),
                source_revision=a.source_revision, gpu=gpus[0],
                dvd_max_iterations=int(runtime["dvd_max_iterations"]),
                caption_cache_root=str(
                    Path(runtime["cache_root"]) / "evidence"),
                cache_manifest_path=runtime["cache_manifest_path"],
                worker_gpus=gpus,
                explicit_selected_records=(explicit_selected_records or None))
        if tuple(baseline.selected_video_ids) != selected:
            raise RuntimeError(
                f"coverage selected {baseline.selected_video_ids}, expected {selected}")
        baseline_source_hashes_before = _tree_hashes(baseline_root)
        intervention = PromptDeltaInterventionRunner(
            segment_captioner=builder.segment_captioner,
            sample_loader=provider_data.__getitem__,
            merge_prompt=prompts.merge_prompt,
            caption_cache_root=str(Path(runtime["cache_root"]) / "interventions"),
            caption_cache_manifest_path=runtime["cache_manifest_path"],
            composition_separator=(
                "\n\nTemporary intervention correction:\n"),
            dvd_max_iterations=int(runtime["dvd_max_iterations"]),
            gpu=gpus[0], scaffold_contract=contract)
        episode_paths = []
        selection_audits = []
        intervention_segment_keys = set()
        candidate_count = 0
        for video_manifest in baseline.video_manifest_paths:
            video_id = _object(Path(video_manifest))["video_id"]
            proposal_output = (output / "proposals" /
                               PROMPT_DELTA_POLICY_OUTPUT_NAMESPACE / video_id)
            plans = proposer.propose(
                video_manifest, parent_meta_prompt_id=parent.meta_prompt_id,
                parent_meta_prompt_text=parent.text,
                output_directory=str(proposal_output))
            selection_audits.append(_object(
                proposal_output / "selection_audit.json"))
            candidate_count += len(plans)
            for plan in plans:
                intervention_segment_keys.update(
                    (video_id, segment_id)
                    for segment_id in plan.selected_segment_ids)
                episode = intervention.run(
                    baseline_video_manifest_path=video_manifest,
                    parent_meta_prompt_id=parent.meta_prompt_id, plan=plan,
                    output_directory=str(
                        output / "interventions" /
                        PROMPT_DELTA_POLICY_OUTPUT_NAMESPACE / video_id /
                        plan.prompt_delta.delta_id /
                        PROMPT_DELTA_INTERVENTION_EXECUTION_NAMESPACE))
                episode_paths.append(_write_once(
                    output / "episodes" /
                    PROMPT_DELTA_POLICY_OUTPUT_NAMESPACE /
                    PROMPT_DELTA_INTERVENTION_EXECUTION_NAMESPACE /
                    f"{len(episode_paths)+1:03d}_{episode.episode_id}.json", episode))
        if candidate_count and len(episode_paths) < 2:
            raise RuntimeError(
                "fewer than two fresh intervention episodes; updater is not eligible")
    finally:
        builder.close_worker_pool()
    resolved_component = json.loads(dumps_canonical(component))
    resolved_component["prompt_delta_proposer"].update({
        "policy_version": proposer_policy_version,
        "maximum_calls": proposer_maximum_calls,
        "maximum_deltas_per_qa": maximum_deltas_per_qa,
        "selection_policy": selection_policy,
        "proposal_target_policy": proposal_target_policy,
        "frame_inspection_classification_policy":
            PROMPT_DELTA_FRAME_INSPECTION_CLASSIFICATION_POLICY,
        "global_inspection_boundary_tolerance_seconds":
            global_inspection_tolerance,
        "representation_version":
            PROMPT_DELTA_PROPOSAL_REPRESENTATION_VERSION,
        "evidence_scope": PROMPT_DELTA_PROPOSAL_EVIDENCE_SCOPE,
        "split_policy": PROMPT_DELTA_PROPOSAL_SPLIT_POLICY,
        "tokenizer_identity": proposer_tokenizer_identity,
    })
    resolved_component["runtime"].update({
        "dvd_frame_inspect_tool_contract_version":
            config.DVD_FRAME_INSPECT_TOOL_CONTRACT_VERSION,
        "dvd_frame_inspect_corrective_retry_limit":
            config.DVD_FRAME_INSPECT_CORRECTIVE_RETRY_LIMIT,
    })
    resolved_component_path = _write_once(
        output / "resolved_component_config.json", resolved_component)
    source_after = {str(path): _sha(path) for path in sources}
    if source_before != source_after:
        raise RuntimeError("source artifact changed during fresh evidence run")
    baseline_source_hashes_after = _tree_hashes(baseline_root)
    if baseline_source_hashes_before != baseline_source_hashes_after:
        raise RuntimeError(
            "completed baseline artifacts changed during proposal/intervention")
    selection_aggregate = {
        "total_qa_count": sum(row["total_qa_count"]
                              for row in selection_audits),
        "eligible_qa_count": sum(
            row["eligible_qa_count"]
            for row in selection_audits),
        "qa_without_localized_evidence": sum(
            row["qa_without_localized_evidence"]
            for row in selection_audits),
        "context_ineligible_qa_count": sum(
            row["context_ineligible_qa_count"]
            for row in selection_audits),
        "excluded_global_inspection_segment_count": len({
            (video_id, segment_id)
            for video_id, audit in zip(selected, selection_audits)
            for record in audit["qa_records"]
            for segment_id in record["global_frame_inspected_segments"]}),
        "total_unique_intervention_candidate_segments": len({
            (video_id, segment_id)
            for video_id, audit in zip(selected, selection_audits)
            for record in audit["qa_records"]
            for segment_id in record["intervention_candidate_segments"]}),
        "proposer_call_count": transport.call_count,
        "candidates_generated": candidate_count,
        "intervention_segments_selected": len(intervention_segment_keys),
    }
    run_status = ("completed" if episode_paths else
                  "no_eligible_proposal_evidence")
    manifest = {
        "schema_version": "fresh_prompt_delta_evidence_run_v1",
        "status": run_status, "parent_meta_prompt_id": parent.meta_prompt_id,
        "selected_video_ids": selected,
        "baseline_manifest_path": baseline.manifest_path,
        "baseline_resume_root": str(baseline_root),
        "baseline_resume_mode": baseline_resume_mode,
        "intervention_execution_namespace":
            PROMPT_DELTA_INTERVENTION_EXECUTION_NAMESPACE,
        "baseline_video_manifest_paths": baseline.video_manifest_paths,
        "episode_paths": episode_paths,
        "model_identities": expected,
        "proposer_provider_call_count": transport.call_count,
        "selection_audit_paths": [str(
            (output / "proposals" / PROMPT_DELTA_POLICY_OUTPUT_NAMESPACE /
             video_id / "selection_audit.json").resolve())
            for video_id in selected],
        "segment_selection_aggregate": selection_aggregate,
        "proposer_configuration": {
            "policy_version": proposer_policy_version,
            "maximum_calls": proposer_maximum_calls,
            "maximum_deltas_per_qa": maximum_deltas_per_qa,
            "selection_policy": selection_policy,
        "proposal_target_policy": proposal_target_policy,
            "frame_inspection_classification_policy":
                PROMPT_DELTA_FRAME_INSPECTION_CLASSIFICATION_POLICY,
            "global_inspection_boundary_tolerance_seconds":
                global_inspection_tolerance,
            "representation_version":
                PROMPT_DELTA_PROPOSAL_REPRESENTATION_VERSION,
            "evidence_scope": PROMPT_DELTA_PROPOSAL_EVIDENCE_SCOPE,
            "split_policy": PROMPT_DELTA_PROPOSAL_SPLIT_POLICY,
            "tokenizer_identity": proposer_tokenizer_identity,
        },
        "resolved_component_config_path": resolved_component_path,
        "resolved_worker_gpus": gpus,
        "worker_result_timeout_seconds": a.worker_result_timeout_seconds,
        "worker_log_directory": str(output / "worker_logs"),
        "source_hashes_before": source_before,
        "source_hashes_after": source_after,
        "baseline_source_hashes_before": baseline_source_hashes_before,
        "baseline_source_hashes_after": baseline_source_hashes_after,
        "legacy_property_codebook_or_router_used": False,
    }
    _write_once(final_path, manifest)
    print(dumps_canonical(manifest))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        try:
            marker = sys.argv.index("--output-dir")
            failure_root = Path(sys.argv[marker + 1]).resolve() / "failures"
            failure_root.mkdir(parents=True, exist_ok=True)
            failure_path = failure_root / f"failure_{time.time_ns()}.json"
            _atomic_write_text(str(failure_path), dumps_canonical({
                "schema_version": "fresh_prompt_delta_failure_v1",
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
                "retry_repair_or_fallback_performed": False,
            }) + "\n")
        except Exception:
            pass
        print(f"fresh evidence failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
