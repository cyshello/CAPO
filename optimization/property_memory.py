"""Checkpoint 1: compact, bounded, property-centric evidence memory.

This module is deliberately observational.  It reads immutable Phase 4 raw
artifacts and writes compact summaries and memory snapshots; it does not
propose, retrieve, intervene, generate feedback, or mutate router/codebook
updates.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_json


COMPACT_SUMMARY_SCHEMA_VERSION = "property_compact_summary_v1"
PROPERTY_MEMORY_SCHEMA_VERSION = "property_memory_v1"
PROPERTY_MEMORY_MANIFEST_SCHEMA_VERSION = "property_memory_manifest_v1"
SELECTION_POLICY_VERSION = "property_memory_selection_v1"
TRANSITIONS = (
    "wrong_to_correct", "correct_to_wrong",
    "correct_to_correct", "wrong_to_wrong",
)


class PropertyMemoryError(RuntimeError):
    pass


class PropertyMemoryConflictError(PropertyMemoryError):
    pass


@dataclass(frozen=True)
class PropertyMemoryBounds:
    strong_positive: int = 3
    weak_positive: int = 2
    harmful: int = 3
    no_effect: int = 2
    routing_positive: int = 2
    routing_negative: int = 2

    def __post_init__(self) -> None:
        for name, value in as_json_dict(self).items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"PropertyMemoryBounds.{name} must be non-negative")


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path) as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PropertyMemoryError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise PropertyMemoryError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: str) -> tuple[dict[str, Any], ...]:
    try:
        with open(path) as handle:
            rows = tuple(json.loads(line) for line in handle if line.strip())
    except (OSError, json.JSONDecodeError) as exc:
        raise PropertyMemoryError(f"cannot read JSONL artifact: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise PropertyMemoryError(f"expected JSON objects in: {path}")
    return rows


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PropertyMemoryError(f"missing source artifact: {path}") from exc
    return digest.hexdigest()


def _artifact_ref(path: str, role: str) -> dict[str, str]:
    absolute = os.path.abspath(path)
    return {"role": role, "path": absolute, "sha256": _file_hash(absolute)}


def _collect_artifact_refs(value: Any) -> tuple[dict[str, str], ...]:
    output = []
    if isinstance(value, Mapping):
        if set(("role", "path", "sha256")) <= set(value):
            output.append({key: str(value[key]) for key in ("role", "path", "sha256")})
        else:
            for item in value.values():
                output.extend(_collect_artifact_refs(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            output.extend(_collect_artifact_refs(item))
    return tuple(output)


def _write_immutable(path: str, value: Any) -> str:
    text = json.dumps(
        as_json_dict(value), sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    if os.path.exists(path):
        with open(path) as handle:
            if handle.read() != text:
                raise PropertyMemoryConflictError(
                    f"immutable memory artifact conflicts: {path}")
        return os.path.abspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, text)
    return os.path.abspath(path)


def _write_jsonl_immutable(path: str, rows: tuple[Mapping[str, Any], ...]) -> str:
    text = "".join(dumps_canonical(row) + "\n" for row in rows)
    if os.path.exists(path):
        with open(path) as handle:
            if handle.read() != text:
                raise PropertyMemoryConflictError(
                    f"immutable compact-summary artifact conflicts: {path}")
        return os.path.abspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, text)
    return os.path.abspath(path)


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _caption_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("caption") or value.get("clip_description") or "")
    return str(value or "")


def _used_segments(row: Mapping[str, Any]) -> tuple[str, ...]:
    direct = row.get("used_segments")
    if direct:
        return tuple(dict.fromkeys(str(item) for item in direct))
    refs = row.get("reference_sets") or {}
    consumed = refs.get("consumed_segments") or ()
    if consumed:
        return tuple(dict.fromkeys(str(item) for item in consumed))
    fallback = set(refs.get("frame_inspected_segments") or ())
    fallback.update(refs.get("explicitly_cited_segments") or ())
    fallback.update(refs.get("returned_segments") or ())
    return tuple(sorted(str(item) for item in fallback))


_STOPWORDS = frozenset({
    "a", "an", "and", "as", "be", "brief", "caption", "captioning", "clearly",
    "describe", "details", "events", "exact", "for", "in", "instruction",
    "meaningful", "of", "only", "or", "precisely", "preserve", "property",
    "reusable", "the", "to", "when", "with", "visually", "visible",
})


def _tokens(value: str) -> frozenset[str]:
    return frozenset(
        token for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 2 and token not in _STOPWORDS)


def _reasoning_text(path: str | None) -> str:
    if not path or not os.path.exists(path):
        return ""
    return " ".join(dumps_canonical(row) for row in _read_jsonl(path))


def _credit_for_property(
    *, property_text: str, routed_segments: tuple[str, ...],
    used_segments: tuple[str, ...], captions: Mapping[str, Any], reasoning: str,
) -> tuple[str, tuple[str, ...], str]:
    """Conservative deterministic evidence-chain check.

    A route is never sufficient.  A segment must also be used, its caption must
    contain a concept from the property instruction, and the reasoning must
    reuse caption information.  Exact caption reuse or two content-token links
    is strong; a single content-token link is weak.
    """
    supporting = []
    strongest = "none"
    property_tokens = _tokens(property_text)
    reasoning_tokens = _tokens(reasoning)
    for segment_id in routed_segments:
        if segment_id not in used_segments:
            continue
        caption = _caption_text(captions.get(segment_id)).strip()
        caption_tokens = _tokens(caption)
        if not caption or not (property_tokens & caption_tokens):
            continue
        reasoning_overlap = caption_tokens & reasoning_tokens
        normalized_caption = " ".join(caption.casefold().split())
        normalized_reasoning = " ".join(reasoning.casefold().split())
        exact_use = len(normalized_caption) >= 12 and normalized_caption in normalized_reasoning
        if exact_use or len(reasoning_overlap) >= 2:
            strongest = "strong"
            supporting.append(segment_id)
        elif reasoning_overlap:
            if strongest != "strong":
                strongest = "weak"
            supporting.append(segment_id)
    supporting_ids = tuple(dict.fromkeys(supporting))
    if strongest == "strong":
        summary = "Property-related caption evidence was clearly reused by the reasoning for the correct answer."
    elif strongest == "weak":
        summary = "Property-related caption evidence was relevant to the reasoning, but decisive use was uncertain."
    else:
        summary = "The property was routed, but no complete caption-to-reasoning credit chain was established."
    return strongest, supporting_ids, summary


def _effect_from_counts(counts: Mapping[str, Any]) -> str:
    positive = int(counts.get("wrong_to_correct", 0))
    negative = int(counts.get("correct_to_wrong", 0))
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "no_effect"


def _effect_summary(effect: str, counts: Mapping[str, int]) -> str:
    return (
        f"The intervention had {effect.replace('_', ' ')} effect across three QAs: "
        f"{counts['wrong_to_correct']} improved, {counts['correct_to_wrong']} regressed, "
        f"{counts['correct_to_correct']} stayed correct, and "
        f"{counts['wrong_to_wrong']} stayed incorrect."
    )


def _example_id(payload: Mapping[str, Any]) -> str:
    return "memory_example_" + sha256_json(payload)[:24]


def _quality(example: Mapping[str, Any], category: str) -> int:
    if category == "positive":
        return 2 if example.get("credit") == "strong" else 1
    if category == "negative":
        return int((example.get("transition_counts") or {}).get(
            "correct_to_wrong", 0))
    if category == "no_effect":
        # A controlled intervention with no transition is stronger no-effect
        # evidence than mere routed-property co-occurrence without credit.
        return 2 if example.get("kind") == "intervention_effect" else 1
    return 1


def _select_examples(
    examples: tuple[Mapping[str, Any], ...], *, limit: int, category: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    unique = {str(item["example_id"]): dict(item) for item in examples}
    pool = list(unique.values())
    selected: list[dict[str, Any]] = []
    videos: set[str] = set()
    signatures: set[str] = set()
    while pool and len(selected) < limit:
        ranked = sorted(pool, key=lambda item: (
            -_quality(item, category),
            0 if str(item.get("video_id") or "") not in videos else 1,
            0 if str(item.get("representative_signature") or "") not in signatures else 1,
            -int(item.get("iteration_ordinal", 0)),
            str(item["example_id"]),
        ))
        chosen = ranked[0]
        selected.append(chosen)
        videos.add(str(chosen.get("video_id") or ""))
        signatures.add(str(chosen.get("representative_signature") or ""))
        pool.remove(chosen)
    retained_ids = {str(item["example_id"]) for item in selected}
    audit = tuple({
        "example_id": str(item["example_id"]),
        "category": category,
        "retained": str(item["example_id"]) in retained_ids,
        "reason": (
            "retained by strength, distinct-video, representative-diversity, then recency priority"
            if str(item["example_id"]) in retained_ids else
            f"evicted by bounded {category} limit={limit}; raw compact summary remains immutable"
        ),
    } for item in sorted(unique.values(), key=lambda value: str(value["example_id"])))
    return tuple(selected), audit


def _blank_memory(entry: Any) -> dict[str, Any]:
    origin_kind = "seed_or_legacy" if entry.created_by in ("", "seed", "fixture") else "lineage"
    why = (
        "Legacy or seed property; no creation artifact is available."
        if origin_kind == "seed_or_legacy" else
        f"Property originated from {entry.created_by} codebook lineage."
    )
    return {
        "schema_version": PROPERTY_MEMORY_SCHEMA_VERSION,
        "property_id": entry.prompt_id,
        "property_text": entry.prompt_text,
        "why_created": why,
        "intended_behavior": entry.prompt_text,
        "origin": {
            "kind": origin_kind,
            "created_by": entry.created_by or "unknown_legacy",
            "provenance": as_json_dict(entry.provenance),
            "artifact_refs": [],
        },
        "positive_examples": [], "negative_examples": [],
        "no_effect_examples": [],
        "routing_examples": {"positive": [], "negative": []},
        "aliases": [], "revision_history": [],
        "latest_decision": {
            "action": "seed_initialize" if origin_kind == "seed_or_legacy" else "lineage_initialize",
            "reason": why, "artifact_refs": [],
        },
    }


class CompactPropertyMemoryRunner:
    """Build versioned summaries and bounded memory from completed raw artifacts."""

    policy_version = "compact_property_memory_checkpoint1_v1"

    def __init__(self, *, bounds: PropertyMemoryBounds | None = None) -> None:
        self.bounds = bounds or PropertyMemoryBounds()
        self.configuration_identity = {
            "policy_version": self.policy_version,
            "summary_schema_version": COMPACT_SUMMARY_SCHEMA_VERSION,
            "memory_schema_version": PROPERTY_MEMORY_SCHEMA_VERSION,
            "selection_policy_version": SELECTION_POLICY_VERSION,
            "bounds": as_json_dict(self.bounds),
        }

    def run(
        self, *, iteration_id: str, iteration_ordinal: int,
        prompt_bank: PromptBankSnapshot,
        baseline_video_manifest_paths: tuple[str, ...],
        intervention_manifest_paths: tuple[str, ...], output_dir: str,
        feedback_manifest_paths: tuple[str, ...] = (),
        parent_memory_path: str | None = None,
        update_plan: Mapping[str, Any] | Any | None = None,
        update_plan_path: str | None = None,
        resulting_prompt_bank: PromptBankSnapshot | None = None,
    ) -> Mapping[str, Any]:
        if iteration_ordinal < 0:
            raise ValueError("iteration_ordinal must be non-negative")
        output_dir = os.path.abspath(output_dir)
        manifest_path = os.path.join(output_dir, "manifest.json")
        source_paths = tuple(os.path.abspath(path) for path in (
            *baseline_video_manifest_paths, *intervention_manifest_paths,
            *feedback_manifest_paths))
        source_hashes = tuple({"path": path, "sha256": _file_hash(path)}
                              for path in source_paths)
        parent = self._load_parent(parent_memory_path)
        plan = as_json_dict(update_plan) if update_plan is not None else None
        plan_ref = _artifact_ref(update_plan_path, "iteration_update_plan") \
            if update_plan_path else None
        identity = {
            "configuration": self.configuration_identity,
            "iteration_id": iteration_id,
            "iteration_ordinal": iteration_ordinal,
            "input_bank_version": prompt_bank.bank_version,
            "input_bank_hash": sha256_json(as_json_dict(prompt_bank)),
            "resulting_bank_hash": (sha256_json(as_json_dict(resulting_prompt_bank))
                                    if resulting_prompt_bank else None),
            "parent_memory": ({"path": os.path.abspath(parent_memory_path),
                               "sha256": _file_hash(parent_memory_path)}
                              if parent_memory_path else None),
            "raw_sources": source_hashes,
            "update_plan": plan,
            "update_plan_ref": plan_ref,
        }
        fingerprint = sha256_json(identity)
        if os.path.exists(manifest_path):
            existing = _read_json(manifest_path)
            if existing.get("schema_version") != PROPERTY_MEMORY_MANIFEST_SCHEMA_VERSION:
                raise PropertyMemoryConflictError("incompatible completed memory schema")
            if existing.get("status") != "completed" or \
                    existing.get("input_fingerprint") != fingerprint:
                raise PropertyMemoryConflictError("memory resume identity mismatch")
            self._validate_completed(existing)
            return {**existing, "resumed": True}
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            raise PropertyMemoryConflictError(
                "incomplete memory directory exists without a compatible manifest")

        credits, proposals = self._credit_summaries(
            baseline_video_manifest_paths, prompt_bank, iteration_id,
            iteration_ordinal)
        effects = self._intervention_summaries(
            intervention_manifest_paths, iteration_id, iteration_ordinal)
        feedback_refs = self._feedback_artifact_refs(feedback_manifest_paths)
        summaries_dir = os.path.join(output_dir, "compact_summaries")
        credit_path = _write_jsonl_immutable(
            os.path.join(summaries_dir, "correct_qa_property_credit.jsonl"), credits)
        effect_path = _write_jsonl_immutable(
            os.path.join(summaries_dir, "intervention_effects.jsonl"), effects)
        summary_refs = (
            _artifact_ref(credit_path, "correct_qa_property_credit_summaries"),
            _artifact_ref(effect_path, "intervention_effect_summaries"),
        )
        memories, candidates, audit = self._build_memory(
            prompt_bank=prompt_bank, resulting_prompt_bank=resulting_prompt_bank,
            parent=parent, credits=credits, effects=effects, proposals=proposals,
            plan=plan, plan_ref=plan_ref, feedback_refs=feedback_refs,
            iteration_id=iteration_id,
            iteration_ordinal=iteration_ordinal)
        memory_dir = os.path.join(output_dir, "property_memory")
        property_paths = []
        for property_id, memory in sorted(memories.items()):
            property_paths.append(_write_immutable(
                os.path.join(memory_dir, "properties", f"{_safe_id(property_id)}.json"),
                memory))
        candidate_paths = []
        for candidate_key, memory in sorted(candidates.items()):
            candidate_paths.append(_write_immutable(
                os.path.join(memory_dir, "candidates", f"{_safe_id(candidate_key)}.json"),
                memory))
        audit_path = _write_immutable(
            os.path.join(memory_dir, "selection_audit.json"), {
                "schema_version": PROPERTY_MEMORY_SCHEMA_VERSION,
                "selection_policy_version": SELECTION_POLICY_VERSION,
                "iteration_id": iteration_id, "decisions": audit,
            })
        snapshot = {
            "schema_version": PROPERTY_MEMORY_SCHEMA_VERSION,
            "summary_schema_version": COMPACT_SUMMARY_SCHEMA_VERSION,
            "selection_policy_version": SELECTION_POLICY_VERSION,
            "iteration_id": iteration_id, "iteration_ordinal": iteration_ordinal,
            "parent_memory_path": (os.path.abspath(parent_memory_path)
                                   if parent_memory_path else None),
            "input_bank_version": prompt_bank.bank_version,
            "active_bank_version": (resulting_prompt_bank or prompt_bank).bank_version,
            "bounds": as_json_dict(self.bounds),
            "properties": tuple(memories[key] for key in sorted(memories)),
            "candidates": tuple(candidates[key] for key in sorted(candidates)),
            "summary_artifact_refs": summary_refs,
            "selection_audit_ref": _artifact_ref(audit_path, "selection_audit"),
        }
        snapshot_path = _write_immutable(
            os.path.join(memory_dir, "snapshot.json"), snapshot)
        raw_refs_by_path = {
            item["path"]: item for item in (
                *({"role": "input_manifest", **item} for item in source_hashes),
                *(ref for summary in (*credits, *effects)
                  for ref in summary.get("artifact_refs") or ()),
                *_collect_artifact_refs(memories),
                *_collect_artifact_refs(candidates),
            )
        }
        manifest = {
            "schema_version": PROPERTY_MEMORY_MANIFEST_SCHEMA_VERSION,
            "status": "completed", "input_fingerprint": fingerprint,
            "iteration_id": iteration_id,
            "compact_summary_schema_version": COMPACT_SUMMARY_SCHEMA_VERSION,
            "property_memory_schema_version": PROPERTY_MEMORY_SCHEMA_VERSION,
            "selection_policy_version": SELECTION_POLICY_VERSION,
            "credit_summary_path": credit_path, "effect_summary_path": effect_path,
            "property_memory_snapshot_path": snapshot_path,
            "property_memory_paths": tuple(property_paths),
            "candidate_memory_paths": tuple(candidate_paths),
            "selection_audit_path": audit_path,
            "artifact_hashes": {
                "credit_summaries": _file_hash(credit_path),
                "effect_summaries": _file_hash(effect_path),
                "memory_snapshot": _file_hash(snapshot_path),
                "selection_audit": _file_hash(audit_path),
            },
            "raw_source_artifacts": source_hashes,
            "raw_artifact_refs": tuple(
                raw_refs_by_path[path] for path in sorted(raw_refs_by_path)),
            "resumed": False,
        }
        _write_immutable(manifest_path, manifest)
        return _read_json(manifest_path)

    def _load_parent(self, path: str | None) -> dict[str, Any] | None:
        if path is None:
            return None
        value = _read_json(path)
        if value.get("schema_version") != PROPERTY_MEMORY_SCHEMA_VERSION or \
                value.get("selection_policy_version") != SELECTION_POLICY_VERSION:
            raise PropertyMemoryConflictError("incompatible parent property-memory schema")
        if not isinstance(value.get("properties"), list) or \
                not isinstance(value.get("candidates"), list):
            raise PropertyMemoryConflictError("incomplete parent property-memory snapshot")
        return value

    @staticmethod
    def _validate_completed(manifest: Mapping[str, Any]) -> None:
        for key in (
            "credit_summary_path", "effect_summary_path",
            "property_memory_snapshot_path", "selection_audit_path",
        ):
            if not os.path.exists(str(manifest.get(key) or "")):
                raise PropertyMemoryConflictError(
                    f"completed memory manifest is missing {key}")
        hashes = manifest.get("artifact_hashes") or {}
        checks = {
            "credit_summaries": manifest["credit_summary_path"],
            "effect_summaries": manifest["effect_summary_path"],
            "memory_snapshot": manifest["property_memory_snapshot_path"],
            "selection_audit": manifest["selection_audit_path"],
        }
        for name, path in checks.items():
            if hashes.get(name) != _file_hash(path):
                raise PropertyMemoryConflictError(
                    f"completed memory artifact hash mismatch: {name}")
        raw_refs = manifest.get("raw_artifact_refs")
        if not isinstance(raw_refs, list):
            raise PropertyMemoryConflictError(
                "completed memory manifest lacks raw artifact hash closure")
        for ref in raw_refs:
            if not isinstance(ref, dict) or not ref.get("path") or \
                    ref.get("sha256") != _file_hash(str(ref["path"])):
                raise PropertyMemoryConflictError(
                    "completed memory raw artifact hash mismatch")

    def _credit_summaries(
        self, manifest_paths: tuple[str, ...], prompt_bank: PromptBankSnapshot,
        iteration_id: str, iteration_ordinal: int,
    ) -> tuple[tuple[dict[str, Any], ...], dict[tuple[str, str], dict[str, Any]]]:
        active = {entry.prompt_id: entry for entry in prompt_bank.entries
                  if entry.status == "active"}
        summaries = []
        proposals: dict[tuple[str, str], dict[str, Any]] = {}
        for manifest_path in sorted(manifest_paths):
            manifest = _read_json(manifest_path)
            video_id = str(manifest.get("video_id") or "")
            proposal_path = manifest.get("property_proposal_path")
            if proposal_path and os.path.exists(proposal_path):
                for proposal in _read_json(proposal_path).get("proposals") or ():
                    proposals[(video_id, str(proposal["candidate_property_id"]))] = {
                        **proposal, "artifact_ref": _artifact_ref(
                            proposal_path, "property_proposal")}
            for retrieval_path in manifest.get("property_retrieval_paths") or ():
                retrieval = _read_json(retrieval_path)
                key = (video_id, str(retrieval.get("candidate_property_id") or ""))
                if key in proposals:
                    proposals[key]["retrieval_artifact_ref"] = _artifact_ref(
                        retrieval_path, "property_retrieval")
            required = ("routing_manifest_path", "captions_path", "baseline_qas_path")
            if any(not manifest.get(key) for key in required):
                continue
            routing_manifest = _read_json(manifest["routing_manifest_path"])
            route_path = (routing_manifest.get("composed_prompts_path") or
                          routing_manifest.get("routing_decisions_path"))
            if not route_path:
                raise PropertyMemoryError("routing manifest lacks routed-property rows")
            routed: dict[str, tuple[str, ...]] = {}
            for row in _read_jsonl(route_path):
                segment_id = str(row.get("segment_id") or "")
                ids = row.get("selected_prompt_ids") or row.get("property_ids") or ()
                routed[segment_id] = tuple(str(item) for item in ids)
            captions = _read_json(manifest["captions_path"])
            for qa in _read_jsonl(manifest["baseline_qas_path"]):
                if not qa.get("is_correct") or qa.get("errors") or \
                        qa.get("prediction") is None:
                    continue
                question_id = str(qa.get("question_id") or "")
                used = _used_segments(qa)
                reasoning = _reasoning_text(qa.get("trajectory_path"))
                video_routed_ids = tuple(dict.fromkeys(
                    property_id for ids in routed.values() for property_id in ids
                    if property_id in active))
                for property_id in video_routed_ids:
                    routed_segments = tuple(segment_id for segment_id, ids in routed.items()
                                            if property_id in ids)
                    credit, supporting, summary = _credit_for_property(
                        property_text=active[property_id].prompt_text,
                        routed_segments=routed_segments, used_segments=used,
                        captions=captions, reasoning=reasoning)
                    refs = [
                        _artifact_ref(manifest_path, "baseline_video_manifest"),
                        _artifact_ref(manifest["routing_manifest_path"], "routing_manifest"),
                        _artifact_ref(route_path, "routed_properties"),
                        _artifact_ref(manifest["captions_path"], "baseline_captions"),
                        _artifact_ref(manifest["baseline_qas_path"], "baseline_qa_results"),
                    ]
                    if qa.get("trajectory_path") and os.path.exists(qa["trajectory_path"]):
                        refs.append(_artifact_ref(qa["trajectory_path"], "qa_reasoning"))
                    if qa.get("result_path") and os.path.exists(qa["result_path"]):
                        refs.append(_artifact_ref(qa["result_path"], "qa_result"))
                    if qa.get("references_path") and os.path.exists(qa["references_path"]):
                        refs.append(_artifact_ref(
                            qa["references_path"], "qa_used_segment_references"))
                    identity = {
                        "schema_version": COMPACT_SUMMARY_SCHEMA_VERSION,
                        "kind": "correct_qa_property_credit", "iteration_id": iteration_id,
                        "property_id": property_id, "video_id": video_id,
                        "question_id": question_id, "credit": credit,
                        "supporting_segment_ids": supporting,
                    }
                    summaries.append({
                        **identity, "summary_id": "credit_" + sha256_json(identity)[:24],
                        "one_sentence_summary": summary,
                        "artifact_refs": tuple(refs),
                        "iteration_ordinal": iteration_ordinal,
                    })
        return (tuple(sorted(summaries, key=lambda item: (
            item["video_id"], item["question_id"], item["property_id"]))), proposals)

    def _intervention_summaries(
        self, manifest_paths: tuple[str, ...], iteration_id: str,
        iteration_ordinal: int,
    ) -> tuple[dict[str, Any], ...]:
        summaries = []
        for manifest_path in sorted(manifest_paths):
            manifest = _read_json(manifest_path)
            for item in manifest.get("results") or ():
                result_path = item.get("result_path")
                if not result_path:
                    raise PropertyMemoryError("intervention result lacks result_path")
                result = _read_json(result_path)
                if result.get("status") != "completed":
                    continue
                transitions_path = result.get("transitions_path")
                if not transitions_path:
                    raise PropertyMemoryError("completed intervention lacks transitions")
                transitions = _read_json(transitions_path)
                raw_counts = transitions.get("transition_counts") or {}
                counts = {label: int(raw_counts.get(label, 0)) for label in TRANSITIONS}
                if sum(counts.values()) != len(transitions.get("qas") or ()):
                    raise PropertyMemoryError("intervention transition counts conflict with QAs")
                effect = _effect_from_counts(counts)
                identity = {
                    "schema_version": COMPACT_SUMMARY_SCHEMA_VERSION,
                    "kind": "intervention_effect", "iteration_id": iteration_id,
                    "candidate_property_id": str(result["candidate_property_id"]),
                    "video_id": str(result["source_video_id"]),
                    "effect": effect, "transition_counts": counts,
                }
                summaries.append({
                    **identity, "summary_id": "effect_" + sha256_json(identity)[:24],
                    "one_sentence_summary": _effect_summary(effect, counts),
                    "artifact_refs": (
                        _artifact_ref(manifest_path, "intervention_manifest"),
                        _artifact_ref(result_path, "intervention_result"),
                        _artifact_ref(transitions_path, "qa_transitions"),
                    ),
                    "iteration_ordinal": iteration_ordinal,
                })
        return tuple(sorted(summaries, key=lambda item: (
            item["video_id"], item["candidate_property_id"])))

    def _feedback_artifact_refs(
        self, manifest_paths: tuple[str, ...],
    ) -> dict[str, tuple[dict[str, str], ...]]:
        output: dict[str, tuple[dict[str, str], ...]] = {}
        for manifest_path in sorted(manifest_paths):
            manifest = _read_json(manifest_path)
            manifest_ref = _artifact_ref(manifest_path, "feedback_manifest")
            for item in manifest.get("properties") or ():
                result_path = item.get("result_path")
                if not result_path:
                    continue
                result = _read_json(result_path)
                result_ref = _artifact_ref(result_path, "property_feedback_result")
                for qa in result.get("qa_feedback") or ():
                    feedback_id = qa.get("feedback_id")
                    if not feedback_id:
                        continue
                    refs = [manifest_ref, result_ref]
                    for field, role in (
                        ("request_path", "feedback_request"),
                        ("raw_output_path", "feedback_raw_output"),
                        ("parsed_output_path", "feedback_parsed_output"),
                        ("rejection_path", "feedback_rejection"),
                    ):
                        path = qa.get(field)
                        if path and os.path.exists(path):
                            refs.append(_artifact_ref(path, role))
                    output[str(feedback_id)] = tuple(refs)
        return output

    def _build_memory(
        self, *, prompt_bank: PromptBankSnapshot,
        resulting_prompt_bank: PromptBankSnapshot | None,
        parent: Mapping[str, Any] | None,
        credits: tuple[Mapping[str, Any], ...],
        effects: tuple[Mapping[str, Any], ...],
        proposals: Mapping[tuple[str, str], Mapping[str, Any]],
        plan: Mapping[str, Any] | None, plan_ref: Mapping[str, str] | None,
        feedback_refs: Mapping[str, tuple[Mapping[str, str], ...]],
        iteration_id: str, iteration_ordinal: int,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], tuple[dict[str, Any], ...]]:
        final_bank = resulting_prompt_bank or prompt_bank
        final_active = {entry.prompt_id: entry for entry in final_bank.entries
                        if entry.status == "active"}
        memories = {
            str(item["property_id"]): json.loads(json.dumps(item))
            for item in (parent or {}).get("properties", ())
            if str(item.get("property_id")) in final_active
        }
        for property_id, entry in final_active.items():
            memories.setdefault(property_id, _blank_memory(entry))
            memories[property_id]["property_text"] = entry.prompt_text
            memories[property_id]["intended_behavior"] = entry.prompt_text
        candidates = {
            f"{item['source_video_id']}::{item['candidate_property_id']}":
                json.loads(json.dumps(item))
            for item in (parent or {}).get("candidates", ())
            if item.get("status") != "promoted"
        }
        new_examples: dict[str, dict[str, list[dict[str, Any]]]] = {
            property_id: {"positive": [], "negative": [], "no_effect": [],
                          "routing_positive": [], "routing_negative": []}
            for property_id in memories
        }
        for property_id, memory in memories.items():
            new_examples[property_id]["positive"].extend(memory.get("positive_examples") or ())
            new_examples[property_id]["negative"].extend(memory.get("negative_examples") or ())
            new_examples[property_id]["no_effect"].extend(memory.get("no_effect_examples") or ())
            new_examples[property_id]["routing_positive"].extend(
                (memory.get("routing_examples") or {}).get("positive") or ())
            new_examples[property_id]["routing_negative"].extend(
                (memory.get("routing_examples") or {}).get("negative") or ())
        for credit in credits:
            property_id = str(credit["property_id"])
            if property_id not in new_examples:
                continue
            example = {
                "example_id": _example_id(credit), "kind": "correct_qa_credit",
                "credit": credit["credit"], "video_id": credit["video_id"],
                "question_id": credit["question_id"],
                "summary": credit["one_sentence_summary"],
                "supporting_segment_ids": credit["supporting_segment_ids"],
                "artifact_refs": credit["artifact_refs"],
                "iteration_id": iteration_id, "iteration_ordinal": iteration_ordinal,
                "representative_signature": sha256_json({
                    "credit": credit["credit"],
                    "segments": credit["supporting_segment_ids"],
                }),
            }
            category = "positive" if credit["credit"] in ("strong", "weak") else "no_effect"
            new_examples[property_id][category].append(example)

        effect_by_key = {(str(item["video_id"]), str(item["candidate_property_id"])): item
                         for item in effects}
        for key, effect in effect_by_key.items():
            proposal = proposals.get(key, {})
            candidate_key = f"{key[0]}::{key[1]}"
            existing = candidates.get(candidate_key, {})
            candidate_examples = list(existing.get("intervention_examples") or ())
            candidate_example = {
                "example_id": _example_id(effect), "effect": effect["effect"],
                "video_id": effect["video_id"],
                "transition_counts": effect["transition_counts"],
                "summary": effect["one_sentence_summary"],
                "artifact_refs": effect["artifact_refs"],
                "iteration_id": iteration_id, "iteration_ordinal": iteration_ordinal,
                "representative_signature": sha256_json(effect["transition_counts"]),
            }
            candidate_examples.append(candidate_example)
            candidates[candidate_key] = {
                "schema_version": PROPERTY_MEMORY_SCHEMA_VERSION,
                "candidate_property_id": key[1], "source_video_id": key[0],
                "property_text": proposal.get("property_text") or key[1],
                "why_proposed": proposal.get("proposal_rationale") or
                    "Proposal rationale is available only in the referenced raw artifact.",
                "intended_reusable_behavior": proposal.get("property_text") or key[1],
                "related_active_property_ids": tuple(
                    proposal.get("coverage_hints") or
                    proposal.get("covered_by_existing_property_ids") or ()),
                "intervention_examples": tuple(candidate_examples),
                "artifact_refs": tuple(filter(None, (
                    proposal.get("artifact_ref"),
                    proposal.get("retrieval_artifact_ref"),
                    *effect["artifact_refs"]))),
                "status": "candidate", "promoted_to_property_id": None,
            }

        decisions = (plan or {}).get("codebook_decisions") or ()
        lineage_to_decision = {}
        for decision in decisions:
            for lineage in decision.get("candidate_lineage") or ():
                lineage_to_decision[(str(lineage["source_video_id"]),
                                     str(lineage["candidate_property_id"]))] = decision
        for key, effect in effect_by_key.items():
            decision = lineage_to_decision.get(key)
            if not decision:
                continue
            decision_feedback_refs = tuple(
                ref for feedback_id in decision.get("supporting_feedback_ids") or ()
                for ref in feedback_refs.get(str(feedback_id), ()))
            refs = tuple(filter(None, (
                plan_ref, *decision_feedback_refs, *effect["artifact_refs"])))
            targets = tuple(str(item) for item in decision.get("target_property_ids") or ())
            original_targets = targets
            result_id = decision.get("resulting_property_id")
            action = str(decision.get("action") or "no_op")
            candidate_key = f"{key[0]}::{key[1]}"
            if action in ("add", "merge") and result_id in memories:
                targets = (str(result_id),)
                candidate = candidates[candidate_key]
                memories[str(result_id)]["why_created"] = candidate["why_proposed"]
                memories[str(result_id)]["origin"] = {
                    "kind": "promoted_candidate", "created_by": "checkpoint3e",
                    "candidate_property_id": key[1], "source_video_id": key[0],
                    "artifact_refs": refs,
                }
                candidate["status"] = "promoted"
                candidate["promoted_to_property_id"] = str(result_id)
            example = {
                "example_id": _example_id({"effect": effect, "targets": targets}),
                "kind": "intervention_effect", "effect": effect["effect"],
                "video_id": effect["video_id"],
                "transition_counts": effect["transition_counts"],
                "summary": effect["one_sentence_summary"], "artifact_refs": refs,
                "iteration_id": iteration_id, "iteration_ordinal": iteration_ordinal,
                "representative_signature": sha256_json(effect["transition_counts"]),
            }
            for target in targets:
                if target not in new_examples:
                    continue
                if effect["effect"] in ("positive", "mixed"):
                    new_examples[target]["positive"].append(example)
                if effect["effect"] in ("negative", "mixed"):
                    new_examples[target]["negative"].append(example)
                if effect["effect"] == "no_effect":
                    new_examples[target]["no_effect"].append(example)
                memories[target]["latest_decision"] = {
                    "action": action, "reason": str(decision.get("rationale") or ""),
                    "artifact_refs": refs,
                }
                if action == "merge":
                    memories[target]["aliases"] = sorted(set(
                        (*memories[target].get("aliases", ()),
                         *original_targets, key[1])))
                memories[target]["revision_history"] = [
                    *memories[target].get("revision_history", ()), {
                        "iteration_id": iteration_id, "action": action,
                        "candidate_property_id": key[1], "artifact_refs": refs,
                    }]

        for supervision in (plan or {}).get("router_supervision") or ():
            property_id = str(supervision["property_id"])
            if property_id not in new_examples:
                continue
            polarity = str(supervision["polarity"])
            routing = {
                "example_id": _example_id(supervision), "kind": "router_supervision",
                "polarity": polarity,
                "video_id": str(supervision.get("source_video_id") or ""),
                "question_id": str(supervision.get("source_question_id") or ""),
                "summary": str(supervision.get("evidence") or ""),
                "supporting_segment_ids": tuple(
                    supervision.get("attributed_segment_ids") or ()),
                "artifact_refs": tuple(filter(None, (
                    plan_ref,
                    *feedback_refs.get(str(
                        supervision.get("source_feedback_id") or ""), ()),
                ))),
                "iteration_id": iteration_id, "iteration_ordinal": iteration_ordinal,
                "representative_signature": str(supervision.get("context_hash") or ""),
            }
            new_examples[property_id][f"routing_{polarity}"].append(routing)

        audit = []
        for property_id, memory in sorted(memories.items()):
            strong = tuple(item for item in new_examples[property_id]["positive"]
                           if item.get("credit") == "strong" or item.get("kind") == "intervention_effect")
            weak = tuple(item for item in new_examples[property_id]["positive"]
                         if item.get("credit") == "weak")
            selected_strong, audit_strong = _select_examples(
                strong, limit=self.bounds.strong_positive, category="positive")
            selected_weak, audit_weak = _select_examples(
                weak, limit=self.bounds.weak_positive, category="positive")
            negative, audit_negative = _select_examples(
                tuple(new_examples[property_id]["negative"]),
                limit=self.bounds.harmful, category="negative")
            no_effect, audit_no_effect = _select_examples(
                tuple(new_examples[property_id]["no_effect"]),
                limit=self.bounds.no_effect, category="no_effect")
            route_pos, audit_route_pos = _select_examples(
                tuple(new_examples[property_id]["routing_positive"]),
                limit=self.bounds.routing_positive, category="routing_positive")
            route_neg, audit_route_neg = _select_examples(
                tuple(new_examples[property_id]["routing_negative"]),
                limit=self.bounds.routing_negative, category="routing_negative")
            memory["positive_examples"] = [*selected_strong, *selected_weak]
            memory["negative_examples"] = list(negative)
            memory["no_effect_examples"] = list(no_effect)
            memory["routing_examples"] = {
                "positive": list(route_pos), "negative": list(route_neg)}
            audit.extend({"property_id": property_id, **item} for item in (
                *audit_strong, *audit_weak, *audit_negative, *audit_no_effect,
                *audit_route_pos, *audit_route_neg))
        for candidate_key, candidate in sorted(candidates.items()):
            prior = tuple(candidate.pop("intervention_examples", ()))
            all_examples = tuple({str(item["example_id"]): dict(item) for item in (
                *candidate.get("positive_examples", ()),
                *candidate.get("negative_examples", ()),
                *candidate.get("mixed_examples", ()),
                *candidate.get("no_effect_examples", ()), *prior,
            )}.values())
            categories = {
                "positive": ("positive_examples", self.bounds.strong_positive),
                "negative": ("negative_examples", self.bounds.harmful),
                "mixed": ("mixed_examples", min(
                    self.bounds.strong_positive, self.bounds.harmful)),
                "no_effect": ("no_effect_examples", self.bounds.no_effect),
            }
            for effect, (field, limit) in categories.items():
                selected, decisions = _select_examples(
                    tuple(item for item in all_examples if item.get("effect") == effect),
                    limit=limit,
                    category=("negative" if effect == "negative" else
                              "no_effect" if effect == "no_effect" else "positive"))
                candidate[field] = list(selected)
                audit.extend({"candidate_key": candidate_key, **item}
                             for item in decisions)
        return memories, candidates, tuple(audit)
