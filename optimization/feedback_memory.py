"""Compact, cumulative updater memory derived from immutable episode feedback.

Full :class:`EpisodeFeedback` remains the audit artifact.  This module creates
short grounded records, stores immutable bank snapshots, and builds a compact
updater projection without rewriting the source episode or feedback files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from surrogate_rollout.optimization.episode_feedback import (
    classify_qa_transition,
)
from surrogate_rollout.optimization.schemas import (
    CompactFeedbackMemory,
    CompactFeedbackMemoryRecord,
    CompactFeedbackProvenance,
    EpisodeFeedback,
    EpisodeFeedbackMemoryRecord,
    InterventionEpisode,
    compact_feedback_memory_record_from_json,
    episode_feedback_memory_record_from_json,
    validate_episode_feedback,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json


COMPACT_FEEDBACK_MEMORY_SCHEMA_VERSION = "compact_feedback_memory_v1"
COMPACT_FEEDBACK_BANK_SCHEMA_VERSION = "compact_feedback_memory_bank_v1"
COMPACT_FEEDBACK_PROJECTION_SCHEMA_VERSION = \
    "compact_feedback_updater_projection_v1"
COMPACT_FEEDBACK_MEMORY_POLICY_VERSION = \
    "deterministic_grounded_compact_feedback_memory_v1"
EPISODE_MEMORY_BANK_SCHEMA_VERSION = "episode_feedback_memory_bank_v2"
EPISODE_MEMORY_RECORD_POLICY_VERSION = "provider_authored_episode_memory_v1"

_WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
_CAUSAL_RE = re.compile(
    r"\b(?:caused|led\s+to|corrected|resulted\s+in)\b",
    flags=re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"(?:```|^\s*[-*#>]\s|\n)", flags=re.MULTILINE)
_EFFECT_ORDER = (
    "wrong_to_correct", "correct_to_wrong", "correct_to_correct",
    "wrong_to_wrong",
)
_ATTRIBUTION_ORDER = ("direct", "indirect", "unresolved", "no_op", "invalid")


class CompactFeedbackMemoryError(ValueError):
    pass


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name))
                for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_compact_feedback_text(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CompactFeedbackMemoryError(f"{field_name} must be non-empty")
    if len(_WORD_RE.findall(value)) > 30:
        raise CompactFeedbackMemoryError(f"{field_name} exceeds 30 words")
    if _MARKDOWN_RE.search(value):
        raise CompactFeedbackMemoryError(
            f"{field_name} must be one plain-text sentence")
    if len(re.findall(r"[.!?](?=\s|$)", value)) > 1:
        raise CompactFeedbackMemoryError(
            f"{field_name} must contain at most one sentence")
    if _CAUSAL_RE.search(value):
        raise CompactFeedbackMemoryError(
            f"{field_name} contains prohibited causal wording")


def validate_compact_feedback_memory_record(
    record: CompactFeedbackMemoryRecord,
) -> None:
    if not isinstance(record, CompactFeedbackMemoryRecord):
        raise TypeError("record must be CompactFeedbackMemoryRecord")
    validate_compact_feedback_text(
        record.memory.runtime_condition, field_name="runtime_condition")
    validate_compact_feedback_text(
        record.memory.description_change, field_name="description_change")
    identifiers = (
        record.provenance.parent_meta_prompt_id,
        record.provenance.iteration_id,
        record.provenance.episode_id,
        record.provenance.video_id,
        record.provenance.feedback_id,
        *record.provenance.qa_ids,
        *record.provenance.segment_ids,
    )
    for identifier in identifiers:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])"
        if any(re.search(pattern, value) for value in (
                record.memory.runtime_condition,
                record.memory.description_change)):
            raise CompactFeedbackMemoryError(
                "compact semantic fields contain provenance identifier")


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _memory_id(
    memory: CompactFeedbackMemory,
    provenance: CompactFeedbackProvenance,
) -> str:
    content = {
        "runtime_condition": _normalized_text(memory.runtime_condition),
        "description_change": _normalized_text(memory.description_change),
        "effect": memory.effect,
        "attribution": memory.attribution,
        "episode_id": provenance.episode_id,
        "qa_ids": list(provenance.qa_ids),
    }
    return "feedback_memory_" + sha256_json(content)[:20]


def _read_references(trajectory_ref: str | None) -> tuple[dict[str, Any] | None,
                                                          dict[str, Any]]:
    if trajectory_ref is None:
        return None, {"trajectory_ref": None, "reason": "unavailable"}
    reference_path = os.path.join(os.path.dirname(trajectory_ref),
                                  "references.json")
    audit = {"trajectory_ref": trajectory_ref,
             "references_path": reference_path}
    try:
        value = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("references artifact is not an object")
        audit.update({
            "trajectory_sha256": _file_sha256(trajectory_ref),
            "references_sha256": _file_sha256(reference_path),
        })
        return value, audit
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        audit["reason"] = f"{type(exc).__name__}: {exc}"
        return None, audit


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _strings(item)


def _segment_set(value: Mapping[str, Any], name: str) -> set[str]:
    raw = value.get(name, [])
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise CompactFeedbackMemoryError(
            f"trajectory references {name} must be a string array")
    return set(raw)


def _grounded_attribution(
    episode: InterventionEpisode,
    qa_ids: tuple[str, ...],
    changed_segments: tuple[str, ...],
) -> tuple[str, dict[str, Any]]:
    if not changed_segments:
        return "no_op", {"reason": "target captions are unchanged"}
    changed = set(changed_segments)
    clip_by_id = {clip.segment_id: clip for clip in episode.clips}
    direct = False
    indirect = False
    invalid = False
    audits = []
    outcome_by_id = {outcome.qa_id: outcome for outcome in episode.qa_outcomes}
    for qa_id in qa_ids:
        outcome = outcome_by_id.get(qa_id)
        if outcome is None:
            invalid = True
            audits.append({"qa_id": qa_id, "reason": "missing QA outcome"})
            continue
        baseline, baseline_audit = _read_references(
            outcome.baseline_trajectory_ref)
        intervention, intervention_audit = _read_references(
            outcome.intervention_trajectory_ref)
        audits.append({"qa_id": qa_id, "baseline": baseline_audit,
                       "intervention": intervention_audit})
        if baseline is None or intervention is None:
            invalid = True
            continue
        try:
            returned = _segment_set(intervention, "returned_segments")
            retrieved_before = _segment_set(baseline, "retrieved_segments")
            retrieved_after = _segment_set(intervention, "retrieved_segments")
        except CompactFeedbackMemoryError:
            invalid = True
            continue
        evidence_strings = set(_strings(intervention.get("evidence", [])))
        for segment_id in changed & returned:
            caption = clip_by_id[segment_id].intervention_caption
            if caption in evidence_strings:
                direct = True
        if retrieved_before != retrieved_after and changed & (
                retrieved_before | retrieved_after):
            indirect = True
    if direct:
        attribution = "direct"
    elif indirect:
        attribution = "indirect"
    elif invalid:
        attribution = "invalid"
    else:
        attribution = "unresolved"
    return attribution, {"trajectory_audit": audits,
                         "direct_caption_evidence_match": direct,
                         "retrieval_structure_changed": indirect}


def _contains_exact_identifier(value: str, identifiers: Iterable[str]) -> bool:
    return any(
        identifier and re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])",
            value)
        for identifier in identifiers)


def _safe_sentence(
    value: str,
    *,
    identifier_replacements: Mapping[str, str] | None = None,
) -> str | None:
    candidate = " ".join(value.strip().split())
    candidate = re.sub(r"^(?:Observed|Hypothesis):\s*", "", candidate,
                       flags=re.IGNORECASE)
    replacements = identifier_replacements or {}
    for identifier in sorted(replacements, key=len, reverse=True):
        replacement = replacements[identifier]
        escaped = re.escape(identifier)
        boundary = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
        if replacement == "the grounded target segment":
            candidate = re.sub(
                rf"\bsegment(?:\s+ID)?\s+{boundary}", replacement,
                candidate, flags=re.IGNORECASE)
        elif replacement == "the supported QA outcome":
            candidate = re.sub(
                rf"\bQA(?:\s+ID)?\s+{boundary}", replacement,
                candidate, flags=re.IGNORECASE)
        candidate = re.sub(boundary, replacement, candidate)
    candidate = " ".join(candidate.split())
    if _contains_exact_identifier(candidate, replacements):
        return None
    try:
        validate_compact_feedback_text(candidate, field_name="candidate")
    except CompactFeedbackMemoryError:
        return None
    return candidate


def _runtime_condition(
    feedback: EpisodeFeedback,
    identifier_replacements: Mapping[str, str],
) -> str:
    candidate = None
    if not _contains_exact_identifier(
            feedback.recommended_strategy_change, identifier_replacements):
        candidate = _safe_sentence(feedback.recommended_strategy_change)
    if candidate and re.match(
            r"^(?:if|when|while|during|where)\b", candidate,
            flags=re.IGNORECASE):
        return candidate
    return (
        "The current frames or bounded history match the recorded "
        "intervention context.")


def _description_change(
    feedback: EpisodeFeedback,
    changed_segments: tuple[str, ...],
    identifier_replacements: Mapping[str, str],
) -> str:
    if not changed_segments:
        return "The target descriptions remain unchanged from the baseline."
    changed = set(changed_segments)
    for evidence in (*feedback.observations, *feedback.counterevidence):
        if evidence.evidence_type not in ("caption_change", "mixed"):
            continue
        if not evidence.supporting_segment_ids or not \
                set(evidence.supporting_segment_ids) <= changed:
            continue
        candidate = _safe_sentence(
            evidence.statement,
            identifier_replacements=identifier_replacements)
        if candidate:
            return candidate
    return "The intervention changes descriptions in the grounded target segments."


def build_compact_feedback_memories(
    *,
    feedback: EpisodeFeedback,
    episode: InterventionEpisode,
    iteration_id: str,
    feedback_artifact_ref: str | None = None,
    episode_artifact_ref: str | None = None,
    grounding_artifact_ref: str | None = None,
) -> tuple[CompactFeedbackMemoryRecord, ...]:
    """Build one short record per deterministic QA-transition group."""
    validate_episode_feedback(feedback, episode)
    if feedback.episode_id != episode.episode_id:
        raise CompactFeedbackMemoryError("feedback and episode IDs differ")
    groups: dict[str, list[str]] = {name: [] for name in _EFFECT_ORDER}
    for outcome in episode.qa_outcomes:
        groups[classify_qa_transition(outcome)].append(outcome.qa_id)
    changed = tuple(clip.segment_id for clip in episode.clips
                    if clip.baseline_caption != clip.intervention_caption)
    identifier_replacements = {
        episode.parent_meta_prompt_id: "the parent meta-prompt",
        iteration_id: "the current iteration",
        episode.episode_id: "the intervention episode",
        episode.video_id: "the source video",
        feedback.feedback_id: "the feedback record",
        **{outcome.qa_id: "the supported QA outcome"
           for outcome in episode.qa_outcomes},
        **{clip.segment_id: "the grounded target segment"
           for clip in episode.clips},
    }
    condition = _runtime_condition(feedback, identifier_replacements)
    description = _description_change(
        feedback, changed, identifier_replacements)
    source_hashes = {}
    for name, path in (
            ("feedback", feedback_artifact_ref),
            ("episode", episode_artifact_ref),
            ("grounding", grounding_artifact_ref)):
        if path is not None:
            source_hashes[name] = _file_sha256(path)
    records = []
    for effect in _EFFECT_ORDER:
        effect_qa_ids = tuple(groups[effect])
        if not effect_qa_ids:
            continue
        grounded_by_attribution: dict[str, list[tuple[str, Any]]] = {
            name: [] for name in _ATTRIBUTION_ORDER}
        for qa_id in effect_qa_ids:
            attribution, audit = _grounded_attribution(
                episode, (qa_id,), changed)
            grounded_by_attribution[attribution].append((qa_id, audit))
        for attribution in _ATTRIBUTION_ORDER:
            grouped = grounded_by_attribution[attribution]
            if not grouped:
                continue
            qa_ids = tuple(item[0] for item in grouped)
            memory = CompactFeedbackMemory(
                runtime_condition=condition,
                description_change=description,
                effect=effect,
                attribution=attribution,
            )
            provenance = CompactFeedbackProvenance(
                parent_meta_prompt_id=episode.parent_meta_prompt_id,
                iteration_id=iteration_id,
                episode_id=episode.episode_id,
                video_id=episode.video_id,
                qa_ids=qa_ids,
                segment_ids=changed,
                feedback_id=feedback.feedback_id,
            )
            conflicts = []
            if not changed and any(item.supporting_segment_ids for item in (
                    *feedback.observations, *feedback.counterevidence)):
                conflicts.append(
                    "feedback cites segment change while captions match")
            feedback_prose = " ".join((
                feedback.outcome_summary, feedback.generator_diagnosis,
                feedback.recommended_strategy_change))
            if attribution != "direct" and _CAUSAL_RE.search(feedback_prose):
                conflicts.append(
                    "feedback uses causal wording without direct trajectory grounding")
            metadata = {
                "schema_version": COMPACT_FEEDBACK_MEMORY_SCHEMA_VERSION,
                "policy_version": COMPACT_FEEDBACK_MEMORY_POLICY_VERSION,
                "source_artifact_refs": {
                    "feedback": feedback_artifact_ref,
                    "episode": episode_artifact_ref,
                    "grounding": grounding_artifact_ref,
                },
                "source_artifact_sha256": source_hashes,
                "grounding_sha256": (
                    source_hashes.get("grounding") or sha256_json({
                        "episode_id": episode.episode_id,
                        "changed_segment_ids": list(changed),
                        "effect": effect,
                        "qa_ids": list(qa_ids),
                        "attribution": attribution,
                    })),
                "natural_language_grounding_conflicts": conflicts,
                "attribution_audit": {
                    "per_qa": [item[1] for item in grouped]},
            }
            record = CompactFeedbackMemoryRecord(
                memory_id=_memory_id(memory, provenance), memory=memory,
                provenance=provenance, metadata=metadata)
            validate_compact_feedback_memory_record(record)
            records.append(record)
    if not records:
        memory = CompactFeedbackMemory(
            condition, description, "no_effect",
            "no_op" if not changed else "unresolved")
        provenance = CompactFeedbackProvenance(
            episode.parent_meta_prompt_id, iteration_id, episode.episode_id,
            episode.video_id, tuple(outcome.qa_id for outcome in episode.qa_outcomes),
            changed, feedback.feedback_id)
        record = CompactFeedbackMemoryRecord(
            _memory_id(memory, provenance), memory, provenance,
            {"schema_version": COMPACT_FEEDBACK_MEMORY_SCHEMA_VERSION,
             "policy_version": COMPACT_FEEDBACK_MEMORY_POLICY_VERSION,
             "source_artifact_refs": {}, "source_artifact_sha256": {},
             "grounding_sha256": sha256_json({"episode": episode.episode_id}),
             "natural_language_grounding_conflicts": []})
        validate_compact_feedback_memory_record(record)
        records.append(record)
    return tuple(records)


def _atomic_text(path: str, text: str) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_compact_feedback_memory_bank(
    bank_directory: str,
) -> tuple[CompactFeedbackMemoryRecord, ...]:
    pointer = os.path.join(os.path.abspath(bank_directory), "current.json")
    if not os.path.exists(pointer):
        return ()
    data = json.loads(Path(pointer).read_text(encoding="utf-8"))
    version_path = data["artifact_path"]
    if _file_sha256(version_path) != data["artifact_sha256"]:
        raise CompactFeedbackMemoryError("memory-bank pointer hash mismatch")
    bank = json.loads(Path(version_path).read_text(encoding="utf-8"))
    return tuple(compact_feedback_memory_record_from_json(item)
                 for item in bank["records"])


def append_compact_feedback_memory_bank(
    bank_directory: str,
    records: Sequence[CompactFeedbackMemoryRecord],
) -> Mapping[str, Any]:
    existing = list(load_compact_feedback_memory_bank(bank_directory))
    by_id = {item.memory_id: item for item in existing}
    added = []
    for record in records:
        validate_compact_feedback_memory_record(record)
        prior = by_id.get(record.memory_id)
        if prior is not None:
            if dumps_canonical(prior) != dumps_canonical(record):
                raise CompactFeedbackMemoryError(
                    f"memory ID collision: {record.memory_id}")
            continue
        by_id[record.memory_id] = record
        existing.append(record)
        added.append(record.memory_id)
    payload = {
        "schema_version": COMPACT_FEEDBACK_BANK_SCHEMA_VERSION,
        "records": existing,
    }
    bank_hash = sha256_json(_plain(payload))
    root = os.path.abspath(bank_directory)
    version_path = os.path.join(root, "versions", f"bank_{bank_hash}.json")
    os.makedirs(os.path.dirname(version_path), exist_ok=True)
    serialized = dumps_canonical(payload) + "\n"
    if os.path.exists(version_path):
        if Path(version_path).read_text(encoding="utf-8") != serialized:
            raise CompactFeedbackMemoryError("memory-bank snapshot collision")
    else:
        with open(version_path, "x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    pointer = {
        "schema_version": "compact_feedback_memory_bank_pointer_v1",
        "bank_sha256": bank_hash,
        "artifact_path": version_path,
        "artifact_sha256": _file_sha256(version_path),
        "record_count": len(existing),
    }
    _atomic_text(os.path.join(root, "current.json"),
                 dumps_canonical(pointer) + "\n")
    return {**pointer, "added_memory_ids": added}


def build_compact_feedback_updater_projection(
    records: Sequence[CompactFeedbackMemoryRecord],
) -> Mapping[str, Any]:
    """Keep high-value grounded records and aggregate uncertain lower tiers."""
    ordered = tuple(records)
    if not ordered:
        raise CompactFeedbackMemoryError("updater memory bank is empty")
    for record in ordered:
        validate_compact_feedback_memory_record(record)
    priorities = {
        ("direct", "wrong_to_correct"): 0,
        ("direct", "correct_to_wrong"): 1,
    }
    individual = [item for item in ordered
                  if item.memory.attribution in ("direct", "indirect")]
    individual.sort(key=lambda item: (
        priorities.get((item.memory.attribution, item.memory.effect), 2),
        ordered.index(item)))
    aggregates = []
    for attribution in ("unresolved", "no_op", "invalid"):
        selected = [item for item in ordered
                    if item.memory.attribution == attribution]
        if not selected:
            continue
        counts = Counter(item.memory.effect for item in selected)
        aggregates.append({
            "attribution": attribution,
            "count": len(selected),
            "effects": {
                "wrong_to_correct": counts["wrong_to_correct"],
                "correct_to_wrong": counts["correct_to_wrong"],
                "unchanged": (
                    counts["correct_to_correct"] + counts["wrong_to_wrong"] +
                    counts["no_effect"]),
                "mixed": counts["mixed"],
            },
        })
    return json.loads(dumps_canonical({
        "schema_version": COMPACT_FEEDBACK_PROJECTION_SCHEMA_VERSION,
        "memories": [{
            "memory_id": item.memory_id,
            "memory": item.memory,
            "feedback_id": item.provenance.feedback_id,
        } for item in individual],
        "aggregates": aggregates,
        "provenance_index": [{
            "memory_id": item.memory_id,
            "feedback_id": item.provenance.feedback_id,
            "episode_id": item.provenance.episode_id,
            "full_feedback_available_by_provenance": bool(
                item.metadata.get("source_artifact_refs", {}).get("feedback")),
        } for item in ordered],
        "full_feedback_omitted": True,
        "record_count": len(ordered),
    }))


def load_representative_full_feedback(
    record: CompactFeedbackMemoryRecord,
) -> Mapping[str, Any]:
    """Explicit audit lookup; never called while building updater requests."""
    refs = record.metadata.get("source_artifact_refs", {})
    hashes = record.metadata.get("source_artifact_sha256", {})
    path = refs.get("feedback") if isinstance(refs, Mapping) else None
    if not isinstance(path, str) or not path:
        raise CompactFeedbackMemoryError("full feedback provenance is absent")
    if hashes.get("feedback") != _file_sha256(path):
        raise CompactFeedbackMemoryError("full feedback provenance hash mismatch")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("feedback_id") != \
            record.provenance.feedback_id:
        raise CompactFeedbackMemoryError("full feedback provenance ID mismatch")
    return value


# The v1 deterministic semantic-memory functions above remain available only
# for reading and migrating legacy artifacts. New iterations use the simple
# provider-authored, one-record-per-episode path below.

def build_episode_feedback_memory_record(
    *,
    feedback: EpisodeFeedback,
    episode: InterventionEpisode,
    iteration_id: str,
    parent_meta_prompt_id: str | None = None,
    feedback_artifact_ref: str | None = None,
) -> EpisodeFeedbackMemoryRecord | None:
    """Wrap provider-authored memory without interpreting or rewriting it."""
    validate_episode_feedback(feedback, episode)
    if feedback.episode_id != episode.episode_id:
        raise CompactFeedbackMemoryError("feedback and episode IDs differ")
    if feedback.compact_memory_text is None:
        return None
    scoped_parent_id = parent_meta_prompt_id or episode.parent_meta_prompt_id
    identity = {
        "parent_meta_prompt_id": scoped_parent_id,
        "episode_id": episode.episode_id,
        "candidate_id": episode.prompt_delta.delta_id,
    }
    metadata: dict[str, Any] = {
        "schema_version": "episode_feedback_memory_record_v1",
        "policy_version": EPISODE_MEMORY_RECORD_POLICY_VERSION,
    }
    if feedback_artifact_ref is not None:
        path = os.path.abspath(feedback_artifact_ref)
        metadata["feedback_artifact_ref"] = path
        metadata["feedback_artifact_sha256"] = _file_sha256(path)
    return EpisodeFeedbackMemoryRecord(
        memory_id="episode_memory_" + sha256_json(identity)[:20],
        parent_meta_prompt_id=scoped_parent_id,
        iteration_id=iteration_id,
        episode_id=episode.episode_id,
        candidate_id=episode.prompt_delta.delta_id,
        feedback_id=feedback.feedback_id,
        memory_text=feedback.compact_memory_text,
        metadata=metadata,
    )


def _parent_bank_directory(root: str, parent_meta_prompt_id: str) -> str:
    if not isinstance(parent_meta_prompt_id, str) or not parent_meta_prompt_id:
        raise CompactFeedbackMemoryError("parent_meta_prompt_id is required")
    key = sha256_json({"parent_meta_prompt_id": parent_meta_prompt_id})[:20]
    return os.path.join(os.path.abspath(root), "parents", f"parent_{key}")


def load_parent_feedback_memory_bank(
    bank_directory: str,
    parent_meta_prompt_id: str,
) -> tuple[EpisodeFeedbackMemoryRecord, ...]:
    root = _parent_bank_directory(bank_directory, parent_meta_prompt_id)
    pointer_path = os.path.join(root, "current.json")
    if not os.path.exists(pointer_path):
        return ()
    pointer = json.loads(Path(pointer_path).read_text(encoding="utf-8"))
    if pointer.get("parent_meta_prompt_id") != parent_meta_prompt_id:
        raise CompactFeedbackMemoryError("parent memory-bank identity mismatch")
    artifact = pointer["artifact_path"]
    if _file_sha256(artifact) != pointer["artifact_sha256"]:
        raise CompactFeedbackMemoryError("parent memory-bank pointer hash mismatch")
    payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
    if payload.get("parent_meta_prompt_id") != parent_meta_prompt_id:
        raise CompactFeedbackMemoryError("parent memory-bank snapshot mismatch")
    return tuple(episode_feedback_memory_record_from_json(item)
                 for item in payload["records"])


def append_parent_feedback_memory_bank(
    bank_directory: str,
    parent_meta_prompt_id: str,
    records: Sequence[EpisodeFeedbackMemoryRecord],
) -> Mapping[str, Any]:
    existing = list(load_parent_feedback_memory_bank(
        bank_directory, parent_meta_prompt_id))
    by_identity = {
        (item.episode_id, item.candidate_id): item for item in existing}
    added: list[str] = []
    for record in records:
        if not isinstance(record, EpisodeFeedbackMemoryRecord):
            raise TypeError("records must contain EpisodeFeedbackMemoryRecord")
        if record.parent_meta_prompt_id != parent_meta_prompt_id:
            raise CompactFeedbackMemoryError(
                "memory record belongs to another parent meta-prompt")
        key = (record.episode_id, record.candidate_id)
        prior = by_identity.get(key)
        if prior is not None:
            if dumps_canonical(prior) != dumps_canonical(record):
                raise CompactFeedbackMemoryError(
                    "episode/candidate memory identity collision")
            continue
        by_identity[key] = record
        existing.append(record)
        added.append(record.memory_id)
    payload = {
        "schema_version": EPISODE_MEMORY_BANK_SCHEMA_VERSION,
        "parent_meta_prompt_id": parent_meta_prompt_id,
        "records": existing,
    }
    bank_hash = sha256_json(_plain(payload))
    root = _parent_bank_directory(bank_directory, parent_meta_prompt_id)
    version_path = os.path.join(root, "versions", f"bank_{bank_hash}.json")
    os.makedirs(os.path.dirname(version_path), exist_ok=True)
    serialized = dumps_canonical(payload) + "\n"
    if os.path.exists(version_path):
        if Path(version_path).read_text(encoding="utf-8") != serialized:
            raise CompactFeedbackMemoryError("parent bank snapshot collision")
    else:
        with open(version_path, "x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    pointer = {
        "schema_version": "episode_feedback_memory_bank_pointer_v2",
        "parent_meta_prompt_id": parent_meta_prompt_id,
        "bank_sha256": bank_hash,
        "artifact_path": version_path,
        "artifact_sha256": _file_sha256(version_path),
        "record_count": len(existing),
    }
    _atomic_text(os.path.join(root, "current.json"),
                 dumps_canonical(pointer) + "\n")
    return {**pointer, "added_memory_ids": added}


def initialize_parent_feedback_memory_bank(
    bank_directory: str,
    parent_meta_prompt_id: str,
) -> Mapping[str, Any]:
    """Create an empty scoped bank, or return its existing pointer."""
    root = _parent_bank_directory(bank_directory, parent_meta_prompt_id)
    pointer_path = os.path.join(root, "current.json")
    if os.path.exists(pointer_path):
        return json.loads(Path(pointer_path).read_text(encoding="utf-8"))
    return append_parent_feedback_memory_bank(
        bank_directory, parent_meta_prompt_id, ())


def archive_parent_feedback_memory_bank(
    bank_directory: str,
    parent_meta_prompt_id: str,
    *,
    promoted_meta_prompt_id: str,
) -> Mapping[str, Any]:
    """Write an immutable archive marker without changing the old snapshot."""
    root = _parent_bank_directory(bank_directory, parent_meta_prompt_id)
    pointer = initialize_parent_feedback_memory_bank(
        bank_directory, parent_meta_prompt_id)
    archive = {
        "schema_version": "episode_feedback_memory_bank_archive_v1",
        "parent_meta_prompt_id": parent_meta_prompt_id,
        "promoted_meta_prompt_id": promoted_meta_prompt_id,
        "bank_pointer": pointer,
    }
    path = os.path.join(root, f"archived_for_{promoted_meta_prompt_id}.json")
    serialized = dumps_canonical(archive) + "\n"
    if os.path.exists(path):
        if Path(path).read_text(encoding="utf-8") != serialized:
            raise CompactFeedbackMemoryError("parent bank archive collision")
    else:
        _atomic_text(path, serialized)
    return {**archive, "artifact_path": path,
            "artifact_sha256": _file_sha256(path)}


def select_historical_feedback_memories(
    records: Sequence[EpisodeFeedbackMemoryRecord],
    *,
    current_iteration_id: str,
    maximum_serialized_characters: int | None = None,
) -> tuple[tuple[Mapping[str, str], ...], Mapping[str, Any]]:
    """Select a recent-first stable prefix without rewriting memory text."""
    if maximum_serialized_characters is not None and (
            not isinstance(maximum_serialized_characters, int) or
            isinstance(maximum_serialized_characters, bool) or
            maximum_serialized_characters <= 0):
        raise ValueError("maximum_serialized_characters must be positive or None")
    eligible = [item for item in records
                if item.iteration_id != current_iteration_id]
    iteration_order: list[str] = []
    grouped: dict[str, list[EpisodeFeedbackMemoryRecord]] = {}
    for item in eligible:
        if item.iteration_id not in grouped:
            iteration_order.append(item.iteration_id)
            grouped[item.iteration_id] = []
        grouped[item.iteration_id].append(item)
    recent_first = [item for iteration in reversed(iteration_order)
                    for item in grouped[iteration]]
    selected: list[Mapping[str, str]] = []
    included: list[str] = []
    excluded: list[str] = []
    used = 0
    budget_exhausted = False
    for item in recent_first:
        projected = {
            "iteration_id": item.iteration_id,
            "episode_id": item.episode_id,
            "memory_id": item.memory_id,
            "memory_text": item.memory_text,
        }
        size = len(dumps_canonical(projected))
        if budget_exhausted or (
                maximum_serialized_characters is not None and
                used + size > maximum_serialized_characters):
            budget_exhausted = True
            excluded.append(item.memory_id)
            continue
        selected.append(projected)
        included.append(item.memory_id)
        used += size
    manifest = {
        "schema_version": "historical_episode_memory_selection_v1",
        "current_iteration_id": current_iteration_id,
        "selection_policy": "recent_iteration_then_stable_episode_order_v1",
        "maximum_serialized_characters": maximum_serialized_characters,
        "selected_serialized_characters": used,
        "selected_memory_ids": included,
        "excluded_memory_ids": excluded,
    }
    return tuple(selected), json.loads(dumps_canonical(manifest))
