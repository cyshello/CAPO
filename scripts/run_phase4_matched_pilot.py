"""Prepare, run, and summarize the bounded two-iteration Phase 4 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from surrogate_rollout import config
from surrogate_rollout.optimization.evidence_builder import (
    CounterfactualEvidenceBuilder,
    SavedEvaluationArtifact,
    write_evidence_jsonl,
)
from surrogate_rollout.optimization.feedback_generator import FeedbackGenerator
from surrogate_rollout.optimization.policies.codex_feedback import (
    CodexStructuredFeedbackProvider,
)
from surrogate_rollout.optimization.policies.llm_feedback import LLMFeedbackGenerator
from surrogate_rollout.optimization.real_fixed_scaffold_iteration import (
    RealFixedScaffoldIterationRunner,
    evaluate_routed_states,
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


REPO_ROOT = Path(__file__).parents[1]
COMPONENT_FIXTURE = REPO_ROOT / "prompt_routing/fixtures/stage4_7_components.json"
SPLIT_MANIFEST = REPO_ROOT / "split_manifest.json"
ROLES = {
    "proposal": (
        ("wCkQ138sg6M", 39, 165),
        ("GLW9omJfAdk", 33, 179),
        ("0RxMZBLeqRI", 9, 184),
    ),
    "confirmation": (("xKiRmesHWIA", 12, 190),),
    "regression": (("pU_yyadYgG8", 90, 214),),
    "final_evaluation": (("w0Wmc8C0Eq0", 3, 282),),
}
CONDITIONS = {
    "fixed_scaffold": False,
    "optimized_scaffold": True,
}


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(str(path), dumps_canonical(value) + "\n")
    return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _qa(sample: dict, index: int) -> QAExample:
    video_id = sample.get("extra", {}).get("videoID") or sample["sample_id"]
    return QAExample(
        video_id=video_id,
        question_id=f"videomme/long/{index}",
        question=sample["question"],
        ground_truth=sample.get("answer"),
        provider_index=index,
        options=tuple(sample.get("options") or ()),
        subtitle_available=bool(sample.get("extra", {}).get("subtitle_path")),
    )


def _components(data: dict):
    return (
        prompt_bank_from_json(data["prompt_bank"]),
        router_policy_from_json(data["router_policy"]),
        scaffold_policy_from_json(data["scaffold_policy"]),
        scaffold_contract_from_json(data["scaffold_contract"]),
    )


def prepare(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"pilot root must be new or empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    with SPLIT_MANIFEST.open() as f:
        split = json.load(f)
    test_ids = {item["video_id"] for item in split["splits"]["test"]}
    role_ids = [video_id for values in ROLES.values()
                for video_id, _, _ in values]
    if len(role_ids) != len(set(role_ids)) or set(role_ids) & test_ids:
        raise RuntimeError("pilot roles overlap or touch the held-out test split")
    train_ids = {item["video_id"] for item in split["splits"]["train"]}
    if not set(role_ids) <= train_ids:
        raise RuntimeError("pilot uses a video outside the frozen train split")

    import sys
    for path in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from data_provider import get_provider
    from dvd_prompt import get_prompts

    provider = get_provider("videomme", split="long")
    batches = {}
    for role, values in ROLES.items():
        records = []
        for expected_video, index, clip_count in values:
            example = _qa(provider[index], index)
            if example.video_id != expected_video:
                raise RuntimeError(f"provider index {index} video changed")
            records.append({
                "role": role, "video_id": expected_video,
                "provider_index": index, "question_id": example.question_id,
                "question": example.question, "options": example.options,
                "ground_truth": example.ground_truth,
                "subtitle_available": example.subtitle_available,
                "clip_count": clip_count, "split": "train",
            })
        batches[role] = records

    with COMPONENT_FIXTURE.open() as f:
        components = json.load(f)
    prompts = get_prompts()
    plan = {
        "experiment": "phase4_bounded_matched_multi_iteration_pilot",
        "paper_scale": False,
        "conditions": {
            key: {"optimize_prompt_bank": True, "optimize_router": True,
                  "optimize_scaffold": value}
            for key, value in CONDITIONS.items()},
        "max_iterations": 2,
        "max_operations_per_component_per_iteration": 1,
        "feedback_calls_per_condition_iteration": 1,
        "external_model_max_retries": 1,
        "proposal_validation_rollout": "selective_trace_reference_budget",
        "proposal_validation_max_selected_clips": 32,
        "final_evaluation_rollout": "full",
        "seed": 0,
        "feedback_model": "gpt-5.5",
        "dvd_max_iterations": 10,
        "canonical_production_commits": False,
        "scaffold_min_distinct_support_videos": 3,
    }
    frozen = root / "frozen"
    files = {
        "plan": Path(_write(frozen / "experiment_plan.json", plan)),
        "batches": Path(_write(frozen / "video_batches.json", batches)),
        "components": Path(_write(frozen / "starting_components.json", components)),
        "prompts": Path(_write(frozen / "prompts.json", {
            "base_prompt_template": prompts.caption_prompt,
            "merge_prompt": prompts.merge_prompt,
        })),
        "split_manifest": frozen / "split_manifest.json",
    }
    shutil.copyfile(SPLIT_MANIFEST, files["split_manifest"])
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        check=True, capture_output=True, text=True).stdout.strip()
    manifest = {
        "schema_version": "phase4_matched_pilot_frozen_v1",
        "source_revision": revision,
        "roles": {role: {
            "video_ids": [item["video_id"] for item in records],
            "question_ids": [item["question_id"] for item in records],
            "provider_indices": [item["provider_index"] for item in records],
        } for role, records in batches.items()},
        "held_out_test_video_ids": sorted(test_ids),
        "files": {name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                  for name, path in files.items()},
    }
    _write(root / "frozen_manifest.json", manifest)
    print(root / "frozen_manifest.json", flush=True)


def _verify_manifest(root: Path) -> dict:
    with (root / "frozen_manifest.json").open() as f:
        manifest = json.load(f)
    for value in manifest["files"].values():
        if _sha256(Path(value["path"])) != value["sha256"]:
            raise RuntimeError(f"frozen file changed: {value['path']}")
    return manifest


def _load_inputs(root: Path):
    manifest = _verify_manifest(root)
    paths = {key: value["path"] for key, value in manifest["files"].items()}
    with open(paths["components"]) as f:
        components = json.load(f)
    with open(paths["batches"]) as f:
        batches = json.load(f)
    with open(paths["prompts"]) as f:
        prompts = json.load(f)
    return manifest, batches, prompts, _components(components)


class RetryFeedbackGenerator(FeedbackGenerator):
    def __init__(self, artifact_dir: Path, *, allow_scaffold: bool) -> None:
        self.artifact_dir = artifact_dir
        self.allow_scaffold = allow_scaffold

    def generate(self, evidence, meta_knowledge):
        attempts = []
        for attempt in range(2):
            provider = CodexStructuredFeedbackProvider(
                model="gpt-5.5", allow_scaffold_updates=self.allow_scaffold)
            directory = self.artifact_dir / f"attempt_{attempt + 1}"
            generator = LLMFeedbackGenerator(
                artifact_dir=str(directory), enabled=True,
                response_provider=provider)
            try:
                batch = generator.generate(evidence, meta_knowledge)
                attempts.append({"attempt": attempt + 1, "status": "success",
                                 "provider": provider.metadata()})
                _write(self.artifact_dir / "attempts.json", attempts)
                return batch
            except Exception as exc:
                attempts.append({"attempt": attempt + 1, "status": "error",
                                 "error_type": type(exc).__name__,
                                 "error": str(exc),
                                 "provider": provider.metadata()})
                _write(self.artifact_dir / "attempts.json", attempts)
                if attempt == 1:
                    raise
                time.sleep(2)
        raise AssertionError("unreachable")


def _normalized(row: dict, condition: str) -> SavedEvaluationArtifact:
    decisions = _read_jsonl(row["routing_decisions_path"])
    composed = _read_jsonl(row["composed_prompts_path"])
    selected = {item["segment_id"]: tuple(item["selected_prompt_ids"])
                for item in decisions}
    prompt_refs = {item["segment_id"]: item["prompt_hash"] for item in composed}
    traces = {item["segment_id"]: item["composition_trace"] for item in composed}
    explicit_scaffold_failure = any(
        trace.get("omitted_prompt_ids") or trace.get("conflicts_detected")
        for trace in traces.values())
    return SavedEvaluationArtifact(
        artifact_id=f"pilot:{condition}:{row['question_id']}",
        condition_name=condition, source_format="phase4_matched_pilot",
        video_id=row["video_id"], question_id=row["question_id"],
        ground_truth=row.get("ground_truth"),
        answer=row.get("parsed_answer") or row.get("prediction"),
        score=float(row["score"]),
        bank_version=row["component_versions"]["bank"],
        router_version=row["component_versions"]["router"],
        scaffold_version=row["component_versions"]["scaffold"],
        selected_prompt_ids=selected, composed_prompt_refs=prompt_refs,
        composition_traces=traces, captions_path=row.get("captions_path"),
        trajectory_refs=((row["trajectory_path"],)
                         if row.get("trajectory_path") else ()),
        rollout_artifact_refs={
            "results": row.get("results_path", ""),
            "routing_decisions": row["routing_decisions_path"],
            "composed_prompts": row["composed_prompts_path"],
            "composition_traces": row["composition_traces_path"],
            "routed_view": row["routed_view_path"],
        },
        config_fingerprint="phase4_matched_pilot_v1",
        used_fallback=bool(row.get("fallback_type") not in (None, "none")),
        fallback_segments=(),
        metadata={
            "missing_trace_sides": (),
            "scaffold_failure_evidence": explicit_scaffold_failure,
            "rollout_mode": row.get("rollout_mode", "full"),
            "selection_policy": row.get("selection_policy", "full_rollout"),
        })


def _build_evidence(evaluation, proposal_ids: set[str], output: Path):
    if evaluation.no_change is not None:
        return None
    baseline_rows = _read_jsonl(evaluation.incumbent_artifact)
    candidate_rows = _read_jsonl(evaluation.candidate_artifact)
    baseline = tuple(_normalized(row, "incumbent") for row in baseline_rows
                     if row["question_id"] in proposal_ids)
    candidate = tuple(_normalized(row, "candidate") for row in candidate_rows
                      if row["question_id"] in proposal_ids)
    evidence = CounterfactualEvidenceBuilder().build(baseline, candidate)
    write_evidence_jsonl(evidence, str(output))
    return evidence


def _samples(batches: dict):
    import sys
    for path in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from data_provider import get_provider

    provider = get_provider("videomme", split="long")
    output = {}
    for role, records in batches.items():
        values = []
        for record in records:
            sample = provider[record["provider_index"]]
            example = _qa(sample, record["provider_index"])
            values.append((example, sample))
        output[role] = tuple(values)
    return output


def _load_next(path: str):
    with open(path) as f:
        data = json.load(f)
    return _components(data)


def run_condition(root: Path, condition: str, gpu: str) -> None:
    manifest, batches, prompts, starting = _load_inputs(root)
    optimize_scaffold = CONDITIONS[condition]
    samples = _samples(batches)
    proposal_qas = samples["proposal"]
    validation_qas = (*samples["proposal"], *samples["confirmation"],
                      *samples["regression"])
    proposal_ids = {item.question_id for item, _ in proposal_qas}
    condition_dir = root / "conditions" / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    _write(condition_dir / "process.json", {
        "pid": os.getpid(), "gpu": gpu, "condition": condition,
        "started_at": time.time(), "source_manifest": str(
            (root / "frozen_manifest.json").resolve())})
    bank, router, scaffold, contract = starting
    fallback_router = replace(
        router, router_version="router_v9000",
        parent_router_version=router.router_version, rules=(),
        fallback_prompt_ids=bank.default_entry_ids,
        provenance={"pilot_bootstrap": "fallback_default"})
    phase_config = Phase4Config(
        seed=0, dry_run=True, commit=False,
        optimization=Phase4OptimizationConfig(
            optimize_prompt_bank=True, optimize_router=True,
            optimize_scaffold=optimize_scaffold, max_iterations=1,
            max_new_entries_per_iteration=1,
            max_router_operations_per_iteration=1,
            min_feedback_confidence=0.7))
    cache_root = condition_dir / "caption_cache"
    bootstrap = evaluate_routed_states(
        incumbent_bank=bank, incumbent_router=fallback_router,
        candidate_bank=bank, candidate_router=router,
        incumbent_scaffold=scaffold, candidate_scaffold=scaffold,
        incumbent_contract=contract, candidate_contract=contract,
        qa_samples=proposal_qas,
        base_prompt_template=prompts["base_prompt_template"],
        merge_prompt=prompts["merge_prompt"], phase4_config=phase_config,
        output_dir=str(condition_dir / "bootstrap_evidence_evaluation"),
        source_revision=manifest["source_revision"], gpu=gpu,
        dvd_max_iterations=10, rollout_mode="full",
        candidate_cache_root=str(cache_root))
    evidence = _build_evidence(
        bootstrap, proposal_ids, condition_dir / "bootstrap_evidence.jsonl")
    if not evidence or len({item.video_id for item in evidence}) != 3:
        raise RuntimeError("bootstrap did not produce three routed evidence videos")

    state_history = []
    for iteration in range(1, 3):
        iteration_dir = condition_dir / f"iteration_{iteration:02d}"
        write_evidence_jsonl(evidence, str(iteration_dir / "frozen_evidence.jsonl"))
        feedback = RetryFeedbackGenerator(
            iteration_dir / "feedback_model",
            allow_scaffold=optimize_scaffold)

        def selective_evaluation(**kwargs):
            return evaluate_routed_states(
                **kwargs, rollout_mode="selective", max_selected_clips=32,
                candidate_cache_root=str(cache_root))

        runner = RealFixedScaffoldIterationRunner(
            feedback_generator=feedback, evaluation_fn=selective_evaluation,
            optimize_scaffold=optimize_scaffold, stage="matched_pilot")
        runner.run(
            input_bank=bank, input_router=router, input_scaffold=scaffold,
            scaffold_contract=contract, evidence=evidence,
            confirmation_examples=tuple(item for item, _ in samples["confirmation"]),
            regression_examples=tuple(item for item, _ in samples["regression"]),
            qa_samples=validation_qas,
            base_prompt_template=prompts["base_prompt_template"],
            merge_prompt=prompts["merge_prompt"], config=phase_config,
            output_dir=str(iteration_dir),
            source_revision=manifest["source_revision"],
            frozen_manifest_path=str(root / "frozen_manifest.json"),
            verify_frozen=lambda: _verify_manifest(root), gpu=gpu,
            dvd_max_iterations=10)
        with (iteration_dir / "run_summary.json").open() as f:
            summary = json.load(f)
        bank, router, scaffold, contract = _load_next(
            summary["next_state_artifact"])
        state_history.append({
            "iteration": iteration,
            "bank_version": bank.bank_version,
            "router_version": router.router_version,
            "scaffold_version": scaffold.scaffold_version,
            "accepted_components": summary["accepted_components"],
            "review_decisions": summary["review_decisions"],
            "validation_decisions": summary["validation_decisions"],
            "no_change": summary["no_change"] is not None,
        })
        _write(condition_dir / "condition_checkpoint.json", {
            "condition": condition, "completed_iterations": iteration,
            "state_history": state_history,
            "latest_state_artifact": summary["next_state_artifact"],
            "canonical_commits": 0,
        })
        evaluation = type("Evaluation", (), {
            "incumbent_artifact": summary["evaluation_artifacts"]["incumbent"],
            "candidate_artifact": summary["evaluation_artifacts"]["candidate"],
            "no_change": summary["no_change"],
        })()
        next_evidence = _build_evidence(
            evaluation, proposal_ids, iteration_dir / "next_evidence.jsonl")
        if next_evidence:
            evidence = next_evidence

    start_bank, start_router, start_scaffold, start_contract = starting
    final = evaluate_routed_states(
        incumbent_bank=start_bank, incumbent_router=start_router,
        candidate_bank=bank, candidate_router=router,
        incumbent_scaffold=start_scaffold, candidate_scaffold=scaffold,
        incumbent_contract=start_contract, candidate_contract=contract,
        qa_samples=samples["final_evaluation"],
        base_prompt_template=prompts["base_prompt_template"],
        merge_prompt=prompts["merge_prompt"], phase4_config=phase_config,
        output_dir=str(condition_dir / "final_full_evaluation"),
        source_revision=manifest["source_revision"], gpu=gpu,
        dvd_max_iterations=10, rollout_mode="full",
        candidate_cache_root=str(cache_root))
    _write(condition_dir / "condition_summary.json", {
        "condition": condition, "optimize_scaffold": optimize_scaffold,
        "completed_iterations": 2, "state_history": state_history,
        "final_component_versions": {
            "bank": bank.bank_version, "router": router.router_version,
            "scaffold": scaffold.scaffold_version,
            "contract": contract.contract_version},
        "final_evaluation": {
            "incumbent_scores": final.incumbent_scores,
            "candidate_scores": (final.candidate_scores
                                 if final.no_change is None else None),
            "no_change": final.no_change,
            "comparison_artifact": final.comparison_artifact,
            "timing": final.timing},
        "canonical_commits": 0,
    })
    print(condition_dir / "condition_summary.json", flush=True)


def summarize(root: Path, wait: bool) -> None:
    paths = {condition: root / "conditions" / condition / "condition_summary.json"
             for condition in CONDITIONS}
    while wait and not all(path.exists() for path in paths.values()):
        time.sleep(30)
    if not all(path.exists() for path in paths.values()):
        raise SystemExit("condition summaries are not complete")
    data = {condition: json.loads(path.read_text())
            for condition, path in paths.items()}
    rows = []
    for condition, value in data.items():
        final = value["final_evaluation"]
        scores = final["candidate_scores"] or final["incumbent_scores"]
        rows.append({
            "condition": condition,
            "optimize_scaffold": value["optimize_scaffold"],
            "bank_version": value["final_component_versions"]["bank"],
            "router_version": value["final_component_versions"]["router"],
            "scaffold_version": value["final_component_versions"]["scaffold"],
            "final_score": next(iter(scores.values())),
            "final_no_change": final["no_change"] is not None,
        })
    _write(root / "matched_final_summary.json", {"conditions": rows})
    lines = [
        "| Condition | Scaffold optimization | Bank | Router | Scaffold | Score | No change |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {str(row['optimize_scaffold']).lower()} | "
            f"{row['bank_version']} | {row['router_version']} | "
            f"{row['scaffold_version']} | {row['final_score']} | "
            f"{str(row['final_no_change']).lower()} |")
    _atomic_write_text(str(root / "matched_final_summary.md"), "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--root", required=True)
    run_parser = sub.add_parser("run-condition")
    run_parser.add_argument("--root", required=True)
    run_parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    run_parser.add_argument("--gpu", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--root", required=True)
    summary_parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command == "prepare":
        prepare(root)
    elif args.command == "run-condition":
        try:
            run_condition(root, args.condition, args.gpu)
        except BaseException as exc:
            _write(root / "conditions" / args.condition / "condition_error.json", {
                "condition": args.condition, "error_type": type(exc).__name__,
                "error": str(exc), "time": time.time()})
            raise
    else:
        summarize(root, args.wait)


if __name__ == "__main__":
    main()
