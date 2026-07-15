"""Run exactly one real Stage 4.13 or Stage 4.14 iteration.

The command freezes all inputs before the single feedback-model call, then
uses the existing routed caption-view and DVD reasoning path for one small
incumbent/candidate evaluation. It never writes canonical component state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import replace

from surrogate_rollout import config
from surrogate_rollout.optimization.evidence_builder import (
    CounterfactualEvidenceBuilder,
    load_phase2_3_condition,
    load_stage4_7_pair,
    read_evidence_jsonl,
    write_evidence_jsonl,
)
from surrogate_rollout.optimization.policies.codex_feedback import (
    CodexStructuredFeedbackProvider,
)
from surrogate_rollout.optimization.policies.llm_feedback import (
    LLMFeedbackGenerator,
)
from surrogate_rollout.optimization.real_fixed_scaffold_iteration import (
    RealFixedScaffoldIterationRunner,
)
from surrogate_rollout.prompt_routing.persistence import (
    _atomic_write_text,
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.schemas import (
    Phase4Config,
    Phase4OptimizationConfig,
    dumps_canonical,
)
from surrogate_rollout.schemas import QAExample


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_COMPONENTS = os.path.join(
    REPO_ROOT, "prompt_routing", "fixtures", "stage4_7_components.json")
DEFAULT_STAGE4_7 = os.path.join(
    REPO_ROOT, "runs", "phase4_stage4_7_smoke_1f24dc1")
DEFAULT_PHASE2_RESULTS = os.path.join(
    REPO_ROOT, "runs", "phase2_sanity_20260714_170825", "results.jsonl")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze(path: str, value: object) -> str:
    _atomic_write_text(path, dumps_canonical(value) + "\n")
    return path


def _qa(question_id: str, sample: dict, provider_index: int) -> QAExample:
    return QAExample(
        video_id=sample.get("extra", {}).get("videoID") or sample["sample_id"],
        question_id=question_id, question=sample["question"],
        ground_truth=sample["answer"], provider_index=provider_index,
        options=tuple(sample.get("options") or ()),
        subtitle_available=bool(sample.get("extra", {}).get("subtitle_path")),
    )


def _qa_from_json(value: dict) -> QAExample:
    return QAExample(
        video_id=value["video_id"], question_id=value["question_id"],
        question=value["question"], ground_truth=value.get("ground_truth"),
        provider_index=int(value["provider_index"]),
        options=tuple(value.get("options") or ()),
        subtitle_available=bool(value.get("subtitle_available", False)),
    )


def _prepare_evidence(stage4_7_root: str, phase2_results: str):
    baseline_47, candidate_47 = load_stage4_7_pair(stage4_7_root)
    baseline_23 = load_phase2_3_condition(
        phase2_results, condition_name="phase2_baseline",
        filters={"candidate_name": "__baseline__", "rollout_mode": "full"})
    candidate_23 = load_phase2_3_condition(
        phase2_results, condition_name="phase2_perceptual",
        filters={"candidate_name": "Perceptual only", "rollout_mode": "full"})
    selected_questions = {"videomme/long/11", "videomme/long/14"}
    baseline = tuple(item for item in baseline_23
                     if item.question_id in selected_questions) + baseline_47
    candidate = tuple(item for item in candidate_23
                      if item.question_id in selected_questions) + candidate_47
    evidence = CounterfactualEvidenceBuilder().build(baseline, candidate)
    return tuple(replace(item, caption_differences=item.caption_differences[:20])
                 for item in evidence)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One real Stage 4.13/4.14 prompt-routing iteration")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--feedback-model", default=config.TEXT_FALLBACK_MODEL)
    parser.add_argument("--dvd-max-iterations", type=int, default=10)
    parser.add_argument("--stage", choices=("4.13", "4.14"), default="4.13")
    parser.add_argument("--optimize-scaffold", action="store_true")
    parser.add_argument("--frozen-input-run")
    parser.add_argument("--components", default=DEFAULT_COMPONENTS)
    parser.add_argument("--stage4-7-run", default=DEFAULT_STAGE4_7)
    parser.add_argument("--phase2-results", default=DEFAULT_PHASE2_RESULTS)
    args = parser.parse_args()
    output_dir = os.path.abspath(args.output_dir)
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise SystemExit(f"output directory must be new or empty: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if args.stage == "4.13" and args.optimize_scaffold:
        raise SystemExit("Stage 4.13 requires optimize_scaffold=false")
    if args.stage == "4.14" and not args.optimize_scaffold:
        raise SystemExit("Stage 4.14 requires --optimize-scaffold")
    if args.stage == "4.14" and not args.frozen_input_run:
        raise SystemExit("Stage 4.14 requires --frozen-input-run")

    source_manifest = None
    if args.frozen_input_run:
        source_root = os.path.abspath(args.frozen_input_run)
        source_manifest_path = os.path.join(source_root, "frozen_manifest.json")
        with open(source_manifest_path) as f:
            source_manifest = json.load(f)
        for value in source_manifest["files"].values():
            if _sha256_file(value["path"]) != value["sha256"]:
                raise SystemExit(f"source frozen input changed: {value['path']}")
        def read_source(name):
            with open(source_manifest["files"][name]["path"]) as source:
                return json.load(source)
        component_data = {
            "prompt_bank": read_source("prompt_bank"),
            "router_policy": read_source("router_policy"),
            "scaffold_policy": read_source("scaffold_policy"),
            "scaffold_contract": read_source("scaffold_contract"),
        }
        evidence = read_evidence_jsonl(
            source_manifest["files"]["evidence"]["path"])
        confirmation = tuple(_qa_from_json(item)
                             for item in read_source("confirmation"))
        regression = tuple(_qa_from_json(item)
                           for item in read_source("regression"))
        base_prompt_template = read_source("base_prompt")["text"]
        merge_prompt = read_source("merge_prompt")["text"]
    else:
        with open(args.components) as f:
            component_data = json.load(f)
    bank = prompt_bank_from_json(component_data["prompt_bank"])
    router = router_policy_from_json(component_data["router_policy"])
    scaffold = scaffold_policy_from_json(component_data["scaffold_policy"])
    contract = scaffold_contract_from_json(component_data["scaffold_contract"])
    if source_manifest is None:
        evidence = _prepare_evidence(args.stage4_7_run, args.phase2_results)

    import sys
    for path in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from data_provider import get_provider
    from dvd_prompt import get_prompts

    provider = get_provider(config.BENCHMARK, split="short")
    if source_manifest is None:
        confirmation_sample = provider[3]
        regression_sample = provider[4]
        confirmation = (_qa("videomme/short/3", confirmation_sample, 3),)
        regression = (_qa("videomme/short/4", regression_sample, 4),)
        prompts = get_prompts()
        base_prompt_template = prompts.caption_prompt
        merge_prompt = prompts.merge_prompt
    else:
        confirmation_sample = provider[confirmation[0].provider_index]
        regression_sample = provider[regression[0].provider_index]
    config_record = Phase4Config(
        seed=0, dry_run=True, commit=False,
        optimization=Phase4OptimizationConfig(
            optimize_prompt_bank=True, optimize_router=True,
            optimize_scaffold=args.optimize_scaffold, max_iterations=1))

    frozen_dir = os.path.join(output_dir, "frozen_batches")
    input_dir = os.path.join(output_dir, "input_state")
    os.makedirs(frozen_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    frozen_paths = {
        "prompt_bank": _freeze(os.path.join(input_dir, "prompt_bank.json"), bank),
        "router_policy": _freeze(os.path.join(input_dir, "router_policy.json"), router),
        "scaffold_policy": _freeze(os.path.join(input_dir, "scaffold_policy.json"), scaffold),
        "scaffold_contract": _freeze(os.path.join(input_dir, "scaffold_contract.json"), contract),
        "configuration": _freeze(os.path.join(input_dir, "phase4_config.json"), config_record),
        "base_prompt": _freeze(os.path.join(input_dir, "base_prompt.json"), {
            "text": base_prompt_template}),
        "merge_prompt": _freeze(os.path.join(input_dir, "merge_prompt.json"), {
            "text": merge_prompt}),
        "confirmation": _freeze(os.path.join(
            frozen_dir, "confirmation.json"), confirmation),
        "regression": _freeze(os.path.join(
            frozen_dir, "regression.json"), regression),
    }
    evidence_path = os.path.join(frozen_dir, "evidence.jsonl")
    write_evidence_jsonl(evidence, evidence_path)
    frozen_paths["evidence"] = evidence_path
    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    implementation_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], check=True,
        capture_output=True, cwd=REPO_ROOT).stdout
    manifest_path = os.path.join(output_dir, "frozen_manifest.json")
    manifest = {
        "schema_version": f"phase4_stage{args.stage}_frozen_v1",
        "source_revision": source_revision,
        "implementation_diff_sha256": hashlib.sha256(
            implementation_diff).hexdigest(),
        "source_artifacts": {
            "stage4_7_run": os.path.abspath(args.stage4_7_run),
            "phase2_results": os.path.abspath(args.phase2_results),
            "matched_frozen_manifest": (
                os.path.join(os.path.abspath(args.frozen_input_run),
                             "frozen_manifest.json")
                if args.frozen_input_run else None),
        },
        "component_versions": {
            "bank": bank.bank_version, "router": router.router_version,
            "scaffold": scaffold.scaffold_version,
            "contract": contract.contract_version},
        "evidence_ids": [item.evidence_id for item in evidence],
        "evidence_video_ids": sorted({item.video_id for item in evidence}),
        "confirmation_ids": [item.question_id for item in confirmation],
        "confirmation_video_ids": [item.video_id for item in confirmation],
        "regression_ids": [item.question_id for item in regression],
        "regression_video_ids": [item.video_id for item in regression],
        "files": {key: {"path": path, "sha256": _sha256_file(path)}
                  for key, path in frozen_paths.items()},
    }
    if source_manifest is not None:
        matched_names = (
            "prompt_bank", "router_policy", "scaffold_policy",
            "scaffold_contract", "base_prompt", "merge_prompt", "evidence",
            "confirmation", "regression",
        )
        manifest["matched_stage4_13_sha256"] = {
            name: source_manifest["files"][name]["sha256"]
            for name in matched_names}
        mismatched = [name for name in matched_names
                      if manifest["files"][name]["sha256"] !=
                      source_manifest["files"][name]["sha256"]]
        if mismatched:
            raise RuntimeError(
                f"Stage 4.14 inputs do not match Stage 4.13: {mismatched}")
    _atomic_write_text(manifest_path, json.dumps(
        manifest, sort_keys=True, indent=2) + "\n")

    def verify_frozen() -> None:
        for value in manifest["files"].values():
            if _sha256_file(value["path"]) != value["sha256"]:
                raise RuntimeError(f"frozen input changed: {value['path']}")

    verify_frozen()
    feedback_provider = CodexStructuredFeedbackProvider(
        model=args.feedback_model,
        allow_scaffold_updates=args.optimize_scaffold)
    feedback_generator = LLMFeedbackGenerator(
        artifact_dir=os.path.join(output_dir, "feedback"), enabled=True,
        response_provider=feedback_provider)
    runner = RealFixedScaffoldIterationRunner(
        feedback_generator=feedback_generator,
        optimize_scaffold=args.optimize_scaffold, stage=args.stage)
    result = runner.run(
        input_bank=bank, input_router=router, input_scaffold=scaffold,
        scaffold_contract=contract, evidence=evidence,
        confirmation_examples=confirmation, regression_examples=regression,
        qa_samples=((confirmation[0], confirmation_sample),
                    (regression[0], regression_sample)),
        base_prompt_template=base_prompt_template,
        merge_prompt=merge_prompt, config=config_record,
        output_dir=output_dir, source_revision=source_revision,
        frozen_manifest_path=manifest_path, verify_frozen=verify_frozen,
        gpu=args.gpu, dvd_max_iterations=args.dvd_max_iterations)
    verify_frozen()
    _atomic_write_text(os.path.join(
        output_dir, "feedback", "provider_metadata.json"), json.dumps(
            feedback_provider.metadata(), sort_keys=True, indent=2) + "\n")
    if source_manifest is not None:
        source_root = os.path.dirname(os.path.abspath(
            source_manifest["files"]["evidence"]["path"]))
        source_root = os.path.dirname(source_root)
        with open(os.path.join(source_root, "run_summary.json")) as f:
            stage4_13_summary = json.load(f)
        with open(os.path.join(source_root, "evaluation", "comparison.json")) as f:
            stage4_13_comparison = json.load(f)
        with open(os.path.join(output_dir, "run_summary.json")) as f:
            stage4_14_summary = json.load(f)
        with open(os.path.join(output_dir, "evaluation", "comparison.json")) as f:
            stage4_14_comparison = json.load(f)
        matched_summary = {
            "schema_version": "phase4_stage4.13_4.14_matched_smoke_v1",
            "matched_input_sha256": manifest["matched_stage4_13_sha256"],
            "stage4_13": {
                "optimize_scaffold": False,
                "run_summary": os.path.join(source_root, "run_summary.json"),
                "review_decisions": stage4_13_summary["review_decisions"],
                "comparison": stage4_13_comparison,
                "equivalence_guard_available_at_execution": False,
                "repeated_candidate_result_eligible_under_current_guard": False,
            },
            "stage4_14": {
                "optimize_scaffold": True,
                "run_summary": os.path.join(output_dir, "run_summary.json"),
                "review_decisions": stage4_14_summary["review_decisions"],
                "comparison": stage4_14_comparison,
                "equivalence_guard_available_at_execution": True,
                "candidate_validation_eligible": stage4_14_summary[
                    "candidate_validation_eligible"],
                "candidate_regression_evidence_eligible": stage4_14_summary[
                    "candidate_regression_evidence_eligible"],
            },
        }
        _atomic_write_text(os.path.join(
            output_dir, "matched_stage4_13_4_14_summary.json"), json.dumps(
                matched_summary, sort_keys=True, indent=2) + "\n")
    print(os.path.join(output_dir, "iteration_result.json"))
    print(result.iteration_id)


if __name__ == "__main__":
    main()
