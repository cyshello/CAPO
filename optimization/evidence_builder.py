"""Offline counterfactual-evidence construction for Phase 4 Stage 4.8.

Only saved JSON/JSONL artifacts are read. This module has no captioning,
reasoning, feedback-model, or optimization-loop imports.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from surrogate_rollout.optimization.schemas import (
    CaptionDifference,
    CounterfactualEvidence,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import as_json_dict, dumps_canonical
from surrogate_rollout.schemas import sha256_text


LEGACY_COMPONENT_VERSION = "legacy_phase2_3"


class EvidenceBuildError(RuntimeError):
    """Saved artifacts cannot be paired or violate evidence invariants."""


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType({str(k): freeze(v) for k, v in item.items()})
        if isinstance(item, (list, tuple)):
            return tuple(freeze(v) for v in item)
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f"saved artifact contains non-JSON value {type(item).__name__}")

    return freeze(value)


@dataclass(frozen=True)
class SavedEvaluationArtifact:
    """Format-neutral input record for saved Phase 2–4 evaluations."""

    artifact_id: str
    condition_name: str
    source_format: str
    video_id: str
    question_id: str
    ground_truth: str | None
    answer: str | None
    score: float
    bank_version: str
    router_version: str
    scaffold_version: str
    selected_prompt_ids: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    composed_prompt_refs: Mapping[str, str] = field(default_factory=dict)
    composition_traces: Mapping[str, Any] = field(default_factory=dict)
    captions_path: str | None = None
    trajectory_refs: tuple[str, ...] = ()
    rollout_artifact_refs: Mapping[str, str] = field(default_factory=dict)
    config_fingerprint: str = ""
    used_fallback: bool = False
    fallback_segments: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "artifact_id", "condition_name", "source_format", "video_id",
            "question_id", "bank_version", "router_version",
            "scaffold_version", "config_fingerprint",
        ):
            if not getattr(self, name):
                raise ValueError(f"SavedEvaluationArtifact.{name} must be non-empty")
        for name in (
            "selected_prompt_ids", "composed_prompt_refs", "composition_traces",
            "rollout_artifact_refs", "metadata",
        ):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name)))
        for name in ("trajectory_refs", "fallback_segments"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(v, str) for v in value):
                raise TypeError(f"SavedEvaluationArtifact.{name} must be tuple[str, ...]")


def saved_evaluation_from_json(data: Mapping[str, Any]) -> SavedEvaluationArtifact:
    return SavedEvaluationArtifact(
        artifact_id=data["artifact_id"],
        condition_name=data["condition_name"],
        source_format=data["source_format"],
        video_id=data["video_id"],
        question_id=data["question_id"],
        ground_truth=data.get("ground_truth"),
        answer=data.get("answer"),
        score=float(data["score"]),
        bank_version=data["bank_version"],
        router_version=data["router_version"],
        scaffold_version=data["scaffold_version"],
        selected_prompt_ids={
            key: tuple(value) for key, value in
            (data.get("selected_prompt_ids") or {}).items()
        },
        composed_prompt_refs=data.get("composed_prompt_refs") or {},
        composition_traces=data.get("composition_traces") or {},
        captions_path=data.get("captions_path"),
        trajectory_refs=tuple(data.get("trajectory_refs") or ()),
        rollout_artifact_refs=data.get("rollout_artifact_refs") or {},
        config_fingerprint=data["config_fingerprint"],
        used_fallback=bool(data.get("used_fallback", False)),
        fallback_segments=tuple(data.get("fallback_segments") or ()),
        metadata=data.get("metadata") or {},
    )


def write_normalized_evaluations(
    records: Sequence[SavedEvaluationArtifact], path: str,
) -> str:
    text = "".join(dumps_canonical(record) + "\n" for record in records)
    _atomic_write_text(path, text)
    return path


def load_normalized_evaluations(path: str) -> tuple[SavedEvaluationArtifact, ...]:
    with open(path) as f:
        return tuple(saved_evaluation_from_json(json.loads(line))
                     for line in f if line.strip())


def _read_jsonl(path: str) -> tuple[dict, ...]:
    with open(path) as f:
        return tuple(json.loads(line) for line in f if line.strip())


def _config_fingerprint(config_data: Mapping[str, Any]) -> str:
    return sha256_text(dumps_canonical(config_data))


def _resolve_saved_path(path: str | None, root: str) -> str | None:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root, path))


def load_stage4_7_pair(
    run_root: str,
) -> tuple[tuple[SavedEvaluationArtifact, ...],
           tuple[SavedEvaluationArtifact, ...]]:
    """Normalize a saved Stage 4.7 fallback/routed run directory."""
    run_root = os.path.abspath(run_root)
    with open(os.path.join(run_root, "run_config.json")) as f:
        run_config = json.load(f)
    fingerprint = _config_fingerprint(run_config)
    video_id = run_config["video_id"]
    normalized: dict[str, tuple[SavedEvaluationArtifact, ...]] = {}
    for condition in ("fallback_default", "routed_multi_entry"):
        condition_dir = os.path.join(run_root, "conditions", condition)
        summary_path = os.path.join(condition_dir, "condition_summary.json")
        with open(summary_path) as f:
            summary = json.load(f)
        routing_path = _resolve_saved_path(summary["routing_decisions_path"], run_root)
        composed_path = _resolve_saved_path(summary["composed_prompts_path"], run_root)
        trace_path = _resolve_saved_path(summary.get("composition_traces_path"), run_root)
        view_path = _resolve_saved_path(summary["routed_caption_view_path"], run_root)
        decisions = _read_jsonl(routing_path)
        composed = _read_jsonl(composed_path)
        traces = _read_jsonl(trace_path) if trace_path and os.path.exists(trace_path) else ()
        with open(view_path) as f:
            view = json.load(f)

        selected = {item["segment_id"]: tuple(item["selected_prompt_ids"])
                    for item in decisions}
        composed_refs = {item["segment_id"]: item["prompt_hash"]
                         for item in composed}
        trace_map = {
            item["segment_id"]: item["composition_trace"] for item in composed
            if item.get("composition_trace") is not None
        }
        fallback_segments = tuple(
            item["segment_id"] for item in decisions if item.get("used_fallback"))
        versions = summary["component_versions"]
        records = []
        for qa in summary["qa_results"]:
            run_dir = _resolve_saved_path(qa.get("run_dir"), run_root)
            refs = {
                "condition_summary": summary_path,
                "routing_decisions": routing_path,
                "composed_prompts": composed_path,
                "caption_view": view_path,
                "captions": view["captions_path"],
            }
            if trace_path:
                refs["composition_traces"] = trace_path
            for name in ("result.json", "references.json", "tool_events.jsonl",
                         "llm_calls.jsonl"):
                path = os.path.join(run_dir, name)
                if os.path.exists(path):
                    refs[name.removesuffix(".json").removesuffix(".jsonl")] = path
            trajectory = os.path.join(run_dir, "trajectory.jsonl")
            records.append(SavedEvaluationArtifact(
                artifact_id=(f"stage4_7:{condition}:{video_id}:"
                             f"{qa['question_id']}"),
                condition_name=condition,
                source_format="stage4_7",
                video_id=video_id,
                question_id=qa["question_id"],
                ground_truth=qa.get("ground_truth"),
                answer=qa.get("parsed_answer") or qa.get("prediction"),
                score=float(qa["score"]),
                bank_version=versions["bank"],
                router_version=versions["router"],
                scaffold_version=versions["scaffold"],
                selected_prompt_ids=selected,
                composed_prompt_refs=composed_refs,
                composition_traces=trace_map,
                captions_path=view["captions_path"],
                trajectory_refs=((trajectory,) if os.path.exists(trajectory) else ()),
                rollout_artifact_refs=refs,
                config_fingerprint=fingerprint,
                used_fallback=bool(fallback_segments),
                fallback_segments=fallback_segments,
                metadata={
                    "condition_cost": summary.get("cost") or {},
                    "condition_timing": summary.get("timing") or {},
                    "caption_cache_keys": summary.get("caption_cache_keys") or [],
                    "trace_file_record_count": len(traces),
                },
            ))
        normalized[condition] = tuple(records)
    return normalized["fallback_default"], normalized["routed_multi_entry"]


def load_phase2_3_condition(
    results_path: str,
    *,
    condition_name: str,
    filters: Mapping[str, Any],
) -> tuple[SavedEvaluationArtifact, ...]:
    """Normalize selected rows from a Phase 2–3 results.jsonl artifact."""
    results_path = os.path.abspath(results_path)
    run_root = os.path.dirname(results_path)
    config_path = os.path.join(run_root, "run_config.json")
    with open(config_path) as f:
        fingerprint = _config_fingerprint(json.load(f))
    rows = [row for row in _read_jsonl(results_path)
            if all(row.get(key) == value for key, value in filters.items())]
    if not rows:
        raise EvidenceBuildError(
            f"no Phase 2–3 rows matched {dict(filters)!r} in {results_path}")
    records = []
    for row in rows:
        trajectory = row.get("trajectory_path")
        refs = {
            key: row[key] for key in (
                "captions_path", "database_path", "trajectory_path")
            if row.get(key)
        }
        records.append(SavedEvaluationArtifact(
            artifact_id=(f"phase2_3:{condition_name}:{row['video_id']}:"
                         f"{row['qa_id']}:{row.get('candidate_name', '')}:"
                         f"{row.get('rollout_mode', '')}:"
                         f"{row.get('selection_policy', '')}"),
            condition_name=condition_name,
            source_format="phase2_3",
            video_id=row["video_id"],
            question_id=row["qa_id"],
            ground_truth=row.get("ground_truth"),
            answer=row.get("parsed_answer") or row.get("answer"),
            score=float(row["score"]),
            bank_version=LEGACY_COMPONENT_VERSION,
            router_version=LEGACY_COMPONENT_VERSION,
            scaffold_version=LEGACY_COMPONENT_VERSION,
            captions_path=row.get("captions_path"),
            trajectory_refs=((trajectory,) if trajectory else ()),
            rollout_artifact_refs=refs,
            config_fingerprint=fingerprint,
            used_fallback=row.get("fallback_type", "none") != "none",
            metadata={
                "candidate_name": row.get("candidate_name"),
                "rollout_mode": row.get("rollout_mode"),
                "selection_policy": row.get("selection_policy"),
                "selected_clip_ids": row.get("selected_clip_ids") or [],
                "recaptioned_clip_ids": row.get("recaptioned_clip_ids") or [],
                "selection_sources": row.get("selection_sources") or {},
                "fallback_type": row.get("fallback_type"),
                "fallback_reason": row.get("fallback_reason"),
            },
        ))
    return tuple(records)


def _segment_key(segment_id: str) -> tuple:
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)", segment_id)
    if match:
        return (0, float(match.group(1)), float(match.group(2)), segment_id)
    return (1, segment_id)


def _caption_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping) and "caption" in value:
        return str(value["caption"])
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _load_caption_map(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    if not os.path.exists(path):
        raise EvidenceBuildError(f"saved captions artifact does not exist: {path}")
    with open(path) as f:
        data = json.load(f)
    return {
        key: _caption_text(value) for key, value in data.items()
        if key not in ("subject_registry", "character_registry")
    }


def _caption_differences(
    baseline: Mapping[str, str], candidate: Mapping[str, str],
) -> tuple[CaptionDifference, ...]:
    differences = []
    for segment_id in sorted(set(baseline) | set(candidate), key=_segment_key):
        before = baseline.get(segment_id, "")
        after = candidate.get(segment_id, "")
        if before == after:
            continue
        if segment_id not in baseline:
            summary = "caption_added"
        elif segment_id not in candidate:
            summary = "caption_removed"
        else:
            summary = "caption_changed"
        differences.append(CaptionDifference(
            segment_id=segment_id,
            baseline_caption=before,
            candidate_caption=after,
            difference_summary=summary,
            structured_differences={
                "baseline_present": segment_id in baseline,
                "candidate_present": segment_id in candidate,
                "baseline_length": len(before),
                "candidate_length": len(after),
                "length_delta": len(after) - len(before),
            },
        ))
    return tuple(differences)


def _transition(baseline: SavedEvaluationArtifact,
                candidate: SavedEvaluationArtifact) -> str:
    if baseline.answer is None or candidate.answer is None:
        return "unknown"
    before = baseline.score > 0.0
    after = candidate.score > 0.0
    return {
        (True, True): "correct_to_correct",
        (True, False): "correct_to_wrong",
        (False, True): "wrong_to_correct",
        (False, False): "wrong_to_wrong",
    }[(before, after)]


def _mapping_differences(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    output = {}
    for key in sorted(set(baseline) | set(candidate), key=_segment_key):
        before = as_json_dict(baseline.get(key))
        after = as_json_dict(candidate.get(key))
        if before != after:
            output[key] = {"baseline": before, "candidate": after}
    return output


class CounterfactualEvidenceBuilder:
    def build(
        self,
        baseline_records: Sequence[SavedEvaluationArtifact],
        candidate_records: Sequence[SavedEvaluationArtifact],
    ) -> tuple[CounterfactualEvidence, ...]:
        baseline = self._index(baseline_records, "baseline")
        candidate = self._index(candidate_records, "candidate")
        if set(baseline) != set(candidate):
            missing_candidate = sorted(set(baseline) - set(candidate))
            missing_baseline = sorted(set(candidate) - set(baseline))
            raise EvidenceBuildError(
                "saved evaluation pairing mismatch; "
                f"missing_candidate={missing_candidate}, "
                f"missing_baseline={missing_baseline}")
        evidence = []
        for key in sorted(baseline):
            before = baseline[key]
            after = candidate[key]
            evidence.append(self._build_one(before, after))
        return tuple(evidence)

    @staticmethod
    def _index(
        records: Sequence[SavedEvaluationArtifact], side: str,
    ) -> dict[tuple[str, str], SavedEvaluationArtifact]:
        output = {}
        for record in records:
            if not isinstance(record, SavedEvaluationArtifact):
                raise TypeError(f"{side} records must be SavedEvaluationArtifact")
            key = (record.video_id, record.question_id)
            if key in output:
                raise EvidenceBuildError(f"duplicate {side} match key {key!r}")
            output[key] = record
        if not output:
            raise EvidenceBuildError(f"{side} records must not be empty")
        return output

    def _build_one(
        self, baseline: SavedEvaluationArtifact,
        candidate: SavedEvaluationArtifact,
    ) -> CounterfactualEvidence:
        if baseline.config_fingerprint != candidate.config_fingerprint:
            raise EvidenceBuildError(
                f"configuration mismatch for {(baseline.video_id, baseline.question_id)!r}: "
                f"{baseline.config_fingerprint} != {candidate.config_fingerprint}")
        if baseline.ground_truth is not None and candidate.ground_truth is not None \
                and baseline.ground_truth != candidate.ground_truth:
            raise EvidenceBuildError(
                f"ground-truth mismatch for {baseline.question_id!r}")
        baseline_captions = _load_caption_map(baseline.captions_path)
        candidate_captions = _load_caption_map(candidate.captions_path)
        caption_differences = _caption_differences(
            baseline_captions, candidate_captions)
        segment_ids = tuple(sorted(
            set(baseline_captions) | set(candidate_captions)
            | set(baseline.selected_prompt_ids) | set(candidate.selected_prompt_ids)
            | set(baseline.composed_prompt_refs) | set(candidate.composed_prompt_refs),
            key=_segment_key,
        ))
        selected_differences = _mapping_differences(
            baseline.selected_prompt_ids, candidate.selected_prompt_ids)
        composed_differences = _mapping_differences(
            baseline.composed_prompt_refs, candidate.composed_prompt_refs)
        missing_trace_sides = []
        if not baseline.composition_traces:
            missing_trace_sides.append("baseline")
        if not candidate.composition_traces:
            missing_trace_sides.append("candidate")
        metadata = {
            "baseline_artifact_id": baseline.artifact_id,
            "candidate_artifact_id": candidate.artifact_id,
            "baseline_condition": baseline.condition_name,
            "candidate_condition": candidate.condition_name,
            "baseline_source_format": baseline.source_format,
            "candidate_source_format": candidate.source_format,
            "config_fingerprint": baseline.config_fingerprint,
            "baseline_used_fallback": baseline.used_fallback,
            "candidate_used_fallback": candidate.used_fallback,
            "baseline_fallback_segments": baseline.fallback_segments,
            "candidate_fallback_segments": candidate.fallback_segments,
            "baseline_composition_traces": baseline.composition_traces,
            "candidate_composition_traces": candidate.composition_traces,
            "missing_trace_sides": missing_trace_sides,
            "selected_entry_differences": selected_differences,
            "composed_prompt_differences": composed_differences,
            "baseline_metadata": baseline.metadata,
            "candidate_metadata": candidate.metadata,
        }
        identity_payload = {
            "video_id": baseline.video_id,
            "question_id": baseline.question_id,
            "baseline_artifact_id": baseline.artifact_id,
            "candidate_artifact_id": candidate.artifact_id,
            "config_fingerprint": baseline.config_fingerprint,
            "versions": {
                "baseline": [baseline.bank_version, baseline.router_version,
                             baseline.scaffold_version],
                "candidate": [candidate.bank_version, candidate.router_version,
                              candidate.scaffold_version],
            },
            "selected": [baseline.selected_prompt_ids,
                         candidate.selected_prompt_ids],
            "composed": [baseline.composed_prompt_refs,
                         candidate.composed_prompt_refs],
            "answers": [baseline.answer, candidate.answer],
            "scores": [baseline.score, candidate.score],
            "caption_differences": caption_differences,
        }
        evidence_id = "evidence_" + sha256_text(
            dumps_canonical(identity_payload))[:24]
        rollout_refs = {
            **{f"baseline.{key}": value
               for key, value in baseline.rollout_artifact_refs.items()},
            **{f"candidate.{key}": value
               for key, value in candidate.rollout_artifact_refs.items()},
        }
        return CounterfactualEvidence(
            evidence_id=evidence_id,
            video_id=baseline.video_id,
            question_id=baseline.question_id,
            ground_truth=baseline.ground_truth or candidate.ground_truth,
            baseline_bank_version=baseline.bank_version,
            baseline_router_version=baseline.router_version,
            baseline_scaffold_version=baseline.scaffold_version,
            candidate_bank_version=candidate.bank_version,
            candidate_router_version=candidate.router_version,
            candidate_scaffold_version=candidate.scaffold_version,
            baseline_selected_prompt_ids=baseline.selected_prompt_ids,
            candidate_selected_prompt_ids=candidate.selected_prompt_ids,
            baseline_composed_prompt_refs=baseline.composed_prompt_refs,
            candidate_composed_prompt_refs=candidate.composed_prompt_refs,
            baseline_answer=baseline.answer,
            candidate_answer=candidate.answer,
            baseline_score=baseline.score,
            candidate_score=candidate.score,
            score_delta=candidate.score - baseline.score,
            outcome_transition=_transition(baseline, candidate),
            segment_ids=segment_ids,
            caption_differences=caption_differences,
            baseline_trajectory_refs=baseline.trajectory_refs,
            candidate_trajectory_refs=candidate.trajectory_refs,
            rollout_artifact_refs=rollout_refs,
            metadata=metadata,
        )


def caption_difference_from_json(data: Mapping[str, Any]) -> CaptionDifference:
    return CaptionDifference(
        segment_id=data["segment_id"],
        baseline_caption=data["baseline_caption"],
        candidate_caption=data["candidate_caption"],
        difference_summary=data.get("difference_summary"),
        structured_differences=data.get("structured_differences") or {},
    )


def evidence_from_json(data: Mapping[str, Any]) -> CounterfactualEvidence:
    return CounterfactualEvidence(
        evidence_id=data["evidence_id"],
        video_id=data["video_id"],
        question_id=data["question_id"],
        ground_truth=data.get("ground_truth"),
        baseline_bank_version=data["baseline_bank_version"],
        baseline_router_version=data["baseline_router_version"],
        baseline_scaffold_version=data["baseline_scaffold_version"],
        candidate_bank_version=data["candidate_bank_version"],
        candidate_router_version=data["candidate_router_version"],
        candidate_scaffold_version=data["candidate_scaffold_version"],
        baseline_selected_prompt_ids={
            key: tuple(value) for key, value in
            data["baseline_selected_prompt_ids"].items()
        },
        candidate_selected_prompt_ids={
            key: tuple(value) for key, value in
            data["candidate_selected_prompt_ids"].items()
        },
        baseline_composed_prompt_refs=data["baseline_composed_prompt_refs"],
        candidate_composed_prompt_refs=data["candidate_composed_prompt_refs"],
        baseline_answer=data.get("baseline_answer"),
        candidate_answer=data.get("candidate_answer"),
        baseline_score=float(data["baseline_score"]),
        candidate_score=float(data["candidate_score"]),
        score_delta=float(data["score_delta"]),
        outcome_transition=data["outcome_transition"],
        segment_ids=tuple(data.get("segment_ids") or ()),
        caption_differences=tuple(caption_difference_from_json(item)
                                  for item in data.get("caption_differences") or ()),
        baseline_trajectory_refs=tuple(data.get("baseline_trajectory_refs") or ()),
        candidate_trajectory_refs=tuple(data.get("candidate_trajectory_refs") or ()),
        rollout_artifact_refs=data.get("rollout_artifact_refs") or {},
        metadata=data.get("metadata") or {},
    )


def write_evidence_jsonl(
    evidence: Sequence[CounterfactualEvidence], path: str,
) -> str:
    text = "".join(dumps_canonical(item) + "\n" for item in evidence)
    _atomic_write_text(path, text)
    return path


def read_evidence_jsonl(path: str) -> tuple[CounterfactualEvidence, ...]:
    with open(path) as f:
        return tuple(evidence_from_json(json.loads(line))
                     for line in f if line.strip())


def write_evidence_artifacts(
    evidence: Sequence[CounterfactualEvidence],
    output_dir: str,
    *,
    source: Mapping[str, Any],
) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = write_evidence_jsonl(
        evidence, os.path.join(output_dir, "counterfactual_evidence.jsonl"))
    transitions = {
        transition: sum(item.outcome_transition == transition for item in evidence)
        for transition in (
            "correct_to_correct", "correct_to_wrong", "wrong_to_correct",
            "wrong_to_wrong", "unknown")
    }
    summary_path = os.path.join(output_dir, "evidence_summary.json")
    _atomic_write_text(summary_path, json.dumps({
        "schema_version": "phase4_stage4.8_v1",
        "evidence_count": len(evidence),
        "transition_counts": transitions,
        "beneficial_count": sum(item.score_delta > 0 for item in evidence),
        "harmful_count": sum(item.score_delta < 0 for item in evidence),
        "neutral_count": sum(item.score_delta == 0 for item in evidence),
        "evidence_ids": [item.evidence_id for item in evidence],
        "source": as_json_dict(source),
        "calls": {"captioning": 0, "dvd_reasoning": 0,
                  "feedback_models": 0, "optimization": 0},
    }, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
    return jsonl_path, summary_path
