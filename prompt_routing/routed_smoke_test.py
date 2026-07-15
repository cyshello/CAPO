"""Stage 4.7 routed evaluator integration smoke test.

Runs two caption-composition conditions with one frozen component set:

``fallback_default``
    No rule matches, so the frozen router selects its configured fallback.
``routed_multi_entry``
    The same router matches one explicit smoke trait and selects multiple
    entries from the same bank.

Both caption views are evaluated through ``evaluation.dvd_qa.run_dvd_qa``.
This module records inference and evaluation artifacts only; it contains no
evidence, feedback, proposal, update, or optimization behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from surrogate_rollout import config
from surrogate_rollout.captioning.candidate_captions import build_clip_index
from surrogate_rollout.evaluation.dvd_qa import ensure_backend, run_dvd_qa
from surrogate_rollout.prompt_routing.offline_dry_run import (
    run_offline_dry_run,
)
from surrogate_rollout.prompt_routing.persistence import (
    _atomic_write_text,
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.routed_caption_view import (
    RoutedCaptionViewArtifact,
    RoutedCaptionViewBuilder,
)
from surrogate_rollout.prompt_routing.schemas import (
    Phase4Config,
    PromptBankSnapshot,
    RouterPolicySnapshot,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    SegmentContext,
    as_json_dict,
)
from surrogate_rollout.schemas import DVDRunResult, sha256_text


FALLBACK_CONDITION = "fallback_default"
ROUTED_CONDITION = "routed_multi_entry"
CONDITIONS = (FALLBACK_CONDITION, ROUTED_CONDITION)


class RoutedSmokeTestError(RuntimeError):
    """Invalid smoke configuration or failure to form the two conditions."""


@dataclass(frozen=True)
class SmokeConditionResult:
    condition: str
    routing_composition_dir: str
    caption_view: RoutedCaptionViewArtifact
    qa_results: tuple[Mapping[str, Any], ...]
    cost: Mapping[str, Any]
    timing: Mapping[str, float]
    summary_path: str


@dataclass(frozen=True)
class RoutedSmokeTestResult:
    work_root: str
    manifest_path: str
    comparison_path: str
    conditions: tuple[SmokeConditionResult, ...]


def load_smoke_components(path: str) -> dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    return {
        "prompt_bank": prompt_bank_from_json(data["prompt_bank"]),
        "router_policy": router_policy_from_json(data["router_policy"]),
        "scaffold_policy": scaffold_policy_from_json(data["scaffold_policy"]),
        "scaffold_contract": scaffold_contract_from_json(
            data["scaffold_contract"]),
        "phase4_config": Phase4Config.from_json_dict(
            data.get("phase4_config") or {}),
    }


def _pretty(value: Any) -> str:
    return json.dumps(as_json_dict(value), sort_keys=True, ensure_ascii=False,
                      indent=2, default=str) + "\n"


def _write(path: str, value: Any, *, raw: bool = False) -> None:
    text = str(value) if raw else _pretty(value)
    _atomic_write_text(path, text)


def _video_id(sample: Mapping[str, Any]) -> str | None:
    return sample.get("extra", {}).get("videoID") or sample.get("sample_id")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qa_record(question_id: str, result: DVDRunResult, run_dir: str) -> dict:
    return {
        "question_id": question_id,
        "prediction": result.prediction,
        "parsed_answer": result.parsed_answer,
        "ground_truth": result.ground_truth,
        "score": result.score,
        "token_usage": result.token_usage,
        "reasoning_seconds": result.latency_seconds,
        "total_segments": result.total_segments,
        "database_path": result.database_path,
        "run_dir": run_dir,
        "errors": result.errors,
    }


class RoutedEvaluatorSmokeRunner:
    """Orchestrate the two-condition Stage 4.7 integration smoke."""

    def __init__(
        self,
        *,
        backend_fn: Callable[..., None] = ensure_backend,
        qa_fn: Callable[..., DVDRunResult] = run_dvd_qa,
        clip_index_fn: Callable[..., list[tuple[str, dict]]] = build_clip_index,
        caption_view_builder: RoutedCaptionViewBuilder | None = None,
    ) -> None:
        self.backend_fn = backend_fn
        self.qa_fn = qa_fn
        self.clip_index_fn = clip_index_fn
        self.caption_view_builder = caption_view_builder or \
            RoutedCaptionViewBuilder()

    def run(
        self,
        *,
        qa_samples: Sequence[tuple[str, dict]],
        prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
        scaffold_policy: ScaffoldPolicySnapshot,
        scaffold_contract: ScaffoldContract,
        phase4_config: Phase4Config,
        base_prompt_template: str,
        merge_prompt: str,
        work_root: str,
        gpu: str | None = None,
        candidate_cache_root: str | None = None,
        source_revision: str = "unknown",
        max_segments: int = 32,
        max_iterations: int = config.MAX_ITERATIONS,
    ) -> RoutedSmokeTestResult:
        qa_samples = tuple(qa_samples)
        if not 1 <= len(qa_samples) <= 3:
            raise RoutedSmokeTestError(
                f"Stage 4.7 requires one to three QAs, got {len(qa_samples)}")
        sample = qa_samples[0][1]
        video_id = _video_id(sample)
        if not video_id:
            raise RoutedSmokeTestError("QA sample has no video ID")
        for question_id, qa_sample in qa_samples:
            if not question_id:
                raise RoutedSmokeTestError("question IDs must be non-empty")
            if _video_id(qa_sample) != video_id:
                raise RoutedSmokeTestError(
                    "all Stage 4.7 QAs must belong to one video")

        work_root = os.path.abspath(work_root)
        if os.path.isdir(work_root) and os.listdir(work_root):
            raise RoutedSmokeTestError(
                f"smoke work_root must be new or empty: {work_root}")
        os.makedirs(work_root, exist_ok=True)
        input_dir = os.path.join(work_root, "input_state")
        os.makedirs(input_dir, exist_ok=True)
        _write(os.path.join(input_dir, "prompt_bank.json"), prompt_bank)
        _write(os.path.join(input_dir, "router_policy.json"), router_policy)
        _write(os.path.join(input_dir, "scaffold_policy.json"), scaffold_policy)
        _write(os.path.join(input_dir, "scaffold_contract.json"), scaffold_contract)
        _write(os.path.join(input_dir, "base_prompt_template.txt"),
               base_prompt_template, raw=True)
        _write(os.path.join(input_dir, "merge_prompt.txt"), merge_prompt, raw=True)
        _write(os.path.join(work_root, "git_commit.txt"),
               source_revision.strip() + "\n", raw=True)
        _write(os.path.join(work_root, "run_config.json"), {
            "stage": "4.7",
            "video_id": video_id,
            "question_ids": [question_id for question_id, _ in qa_samples],
            "conditions": list(CONDITIONS),
            "component_versions": {
                "bank": prompt_bank.bank_version,
                "router": router_policy.router_version,
                "scaffold": scaffold_policy.scaffold_version,
                "contract": scaffold_contract.contract_version,
            },
            "optimization_flags": {
                "optimize_prompt_bank": phase4_config.optimization.optimize_prompt_bank,
                "optimize_router": phase4_config.optimization.optimize_router,
                "optimize_scaffold": phase4_config.optimization.optimize_scaffold,
            },
            "caption_model": config.CAPTION_MODEL_ID,
            "reasoning_model": config.TEXT_FALLBACK_MODEL,
            "tool_calling_model": config.ORCHESTRATOR_TOOL_MODEL,
            "caption_decoding": config.CAPTION_DECODING,
            "max_iterations": max_iterations,
            "gpu": gpu,
            "base_prompt_hash": sha256_text(base_prompt_template),
            "merge_prompt_hash": sha256_text(merge_prompt),
        })

        # vLLM must be initialized before any later database thread pools.
        self.backend_fn(gpu, preload_captioner=True)
        clip_keys = tuple(key for key, _ in self.clip_index_fn(sample, video_id))
        if not clip_keys or len(clip_keys) > max_segments:
            raise RoutedSmokeTestError(
                f"Stage 4.7 requires a small video with 1..{max_segments} "
                f"segments, got {len(clip_keys)}")

        condition_results: list[SmokeConditionResult] = []
        for condition in CONDITIONS:
            condition_start = time.monotonic()
            condition_dir = os.path.join(work_root, "conditions", condition)
            routing_dir = os.path.join(condition_dir, "routing_composition")
            trait = "multi_entry" if condition == ROUTED_CONDITION \
                else "fallback_default"
            contexts = tuple(SegmentContext(
                video_id=video_id,
                segment_id=segment_id,
                segment_features={"smoke_route": trait},
                metadata={"stage": "4.7", "condition": condition},
            ) for segment_id in clip_keys)
            offline = run_offline_dry_run(
                contexts=contexts,
                prompt_bank=prompt_bank,
                router_policy=router_policy,
                scaffold_policy=scaffold_policy,
                scaffold_contract=scaffold_contract,
                phase4_config=phase4_config,
                base_prompt_template=base_prompt_template,
                output_dir=routing_dir,
                source_revision=source_revision,
            )
            if condition == FALLBACK_CONDITION:
                if not all(d.used_fallback and len(d.selected_prompt_ids) == 1
                           for d in offline.routing_decisions):
                    raise RoutedSmokeTestError(
                        "fallback_default did not produce one fallback entry "
                        "for every segment")
            elif not all(not d.used_fallback and len(d.selected_prompt_ids) > 1
                         for d in offline.routing_decisions):
                raise RoutedSmokeTestError(
                    "routed_multi_entry did not produce multiple routed entries "
                    "for every segment")

            prompt_map = {p.segment_id: p for p in offline.composed_prompts}
            caption_view = self.caption_view_builder.build(
                sample=sample,
                segment_composed_prompt_map=prompt_map,
                work_root=os.path.join(condition_dir, "caption_view"),
                merge_prompt=merge_prompt,
                candidate_cache_root=candidate_cache_root,
            )
            qa_records: list[dict] = []
            for question_id, qa_sample in qa_samples:
                qa_dir = os.path.join(
                    condition_dir, "qa", question_id.replace("/", "-"))
                qa_result = self.qa_fn(
                    captions_path=caption_view.captions_path,
                    sample=qa_sample,
                    run_dir=qa_dir,
                    question_id=question_id,
                    database_path=caption_view.database_path,
                    max_iterations=max_iterations,
                    gpu=gpu,
                )
                qa_records.append(_qa_record(question_id, qa_result, qa_dir))

            wall_seconds = time.monotonic() - condition_start
            cost = {
                "caption_call_count": caption_view.caption_call_count,
                "caption_cache_hits": caption_view.caption_cache_hits,
                "token_usage_by_question": {
                    record["question_id"]: record["token_usage"]
                    for record in qa_records
                },
                "estimated_monetary_cost_usd": None,
                "cost_note": (
                    "Backend pricing is not stable/exposed; computational cost "
                    "is retained as caption calls, cache hits, token usage, and "
                    "timing."
                ),
            }
            timing = {
                "caption_seconds": caption_view.caption_seconds,
                "reasoning_seconds": sum(
                    record["reasoning_seconds"] for record in qa_records),
                "condition_wall_seconds": wall_seconds,
            }
            summary_path = os.path.join(condition_dir, "condition_summary.json")
            summary = {
                "condition": condition,
                "component_versions": {
                    "bank": prompt_bank.bank_version,
                    "router": router_policy.router_version,
                    "scaffold": scaffold_policy.scaffold_version,
                    "contract": scaffold_contract.contract_version,
                },
                "routing_decisions_path": os.path.join(
                    routing_dir, "routing", "decisions.jsonl"),
                "composition_traces_path": os.path.join(
                    routing_dir, "composition", "traces.jsonl"),
                "composed_prompts_path": os.path.join(
                    routing_dir, "composition", "composed_prompts.jsonl"),
                "routed_caption_view_path": caption_view.routed_view_path,
                "composed_prompt_hashes": sorted({
                    p.prompt_hash for p in offline.composed_prompts}),
                "caption_cache_keys": [
                    as_json_dict(group.caption_cache_key)
                    for group in caption_view.prompt_groups
                ],
                "qa_results": qa_records,
                "cost": cost,
                "timing": timing,
            }
            _write(summary_path, summary)
            condition_results.append(SmokeConditionResult(
                condition=condition,
                routing_composition_dir=routing_dir,
                caption_view=caption_view,
                qa_results=tuple(qa_records),
                cost=cost,
                timing=timing,
                summary_path=summary_path,
            ))

        by_condition = {result.condition: result for result in condition_results}
        comparison_rows = []
        for question_id, _ in qa_samples:
            fallback = next(r for r in by_condition[FALLBACK_CONDITION].qa_results
                            if r["question_id"] == question_id)
            routed = next(r for r in by_condition[ROUTED_CONDITION].qa_results
                          if r["question_id"] == question_id)
            comparison_rows.append({
                "question_id": question_id,
                "ground_truth": fallback["ground_truth"],
                FALLBACK_CONDITION: fallback,
                ROUTED_CONDITION: routed,
                "score_delta": routed["score"] - fallback["score"],
            })
        comparison = {
            "stage": "4.7",
            "integration_smoke_only": True,
            "video_id": video_id,
            "frozen_component_versions": {
                "bank": prompt_bank.bank_version,
                "router": router_policy.router_version,
                "scaffold": scaffold_policy.scaffold_version,
                "contract": scaffold_contract.contract_version,
            },
            "conditions": {
                result.condition: {
                    "summary_path": result.summary_path,
                    "cost": result.cost,
                    "timing": result.timing,
                }
                for result in condition_results
            },
            "comparisons": comparison_rows,
        }
        comparison_path = os.path.join(work_root, "comparison.json")
        _write(comparison_path, comparison)

        artifacts: dict[str, dict[str, Any]] = {}
        for directory, _, filenames in os.walk(work_root):
            for filename in sorted(filenames):
                path = os.path.join(directory, filename)
                relative = os.path.relpath(path, work_root)
                if relative == "manifest.json":
                    continue
                artifacts[relative] = {
                    "sha256": _sha256_file(path),
                    "size_bytes": os.path.getsize(path),
                }
        manifest_path = os.path.join(work_root, "manifest.json")
        _write(manifest_path, {
            "schema_version": "phase4_stage4.7_v1",
            "status": "completed",
            "integration_smoke_only": True,
            "video_id": video_id,
            "segment_count": len(clip_keys),
            "question_count": len(qa_samples),
            "conditions": list(CONDITIONS),
            "comparison_path": comparison_path,
            "artifacts": artifacts,
            "later_stages_run": [],
        })
        return RoutedSmokeTestResult(
            work_root=work_root,
            manifest_path=manifest_path,
            comparison_path=comparison_path,
            conditions=tuple(condition_results),
        )
