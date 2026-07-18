"""Memory-conditioned LLM codebook planning with deterministic application."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from surrogate_rollout.prompt_routing.persistence import (
    _atomic_write_text,
    prompt_bank_from_json,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    PromptEntry,
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.prompt_routing.validators import validate_prompt_bank
from surrogate_rollout.schemas import sha256_json, sha256_text


SYSTEM_PROMPT_VERSION = "memory_codebook_updater_prompt_v1"
UPDATER_REQUEST_SCHEMA_VERSION = "memory_codebook_updater_request_v2"
UPDATER_RESPONSE_SCHEMA_VERSION = "memory_codebook_updater_response_v1"
VALIDATION_REPORT_SCHEMA_VERSION = "memory_codebook_validation_report_v1"
APPLIED_PLAN_SCHEMA_VERSION = "memory_codebook_applied_plan_v1"
CANDIDATE_CODEBOOK_SCHEMA_VERSION = "memory_candidate_codebook_v1"
ID_MAPPING_SCHEMA_VERSION = "property_id_mapping_v1"
CHECKPOINT_MANIFEST_SCHEMA_VERSION = "memory_codebook_checkpoint_manifest_v2"
UPDATER_INPUT_IDENTITY_SCHEMA_VERSION = "memory_codebook_updater_input_identity_v1"
UPDATER_ATTEMPTS_SCHEMA_VERSION = "memory_codebook_updater_attempts_v1"
ACTION_NAMES = ("add", "revise", "merge", "preserve", "retire", "no_op")
ACTION_FIELDS = frozenset({
    "action_id", "action", "target_property_ids", "candidate_ids",
    "proposed_property_id", "proposed_property_text", "reasoning",
    "supporting_memory_example_ids", "supporting_evidence_ids",
    "behaviors_to_preserve", "confidence",
})
PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "prompts", "codebook_updater_v1.txt")


class LLMCodebookUpdaterError(RuntimeError):
    pass


class LLMCodebookUpdaterConflictError(LLMCodebookUpdaterError):
    pass


class LLMCodebookUpdaterParseError(LLMCodebookUpdaterError):
    pass


@dataclass(frozen=True)
class CodebookUpdaterBounds:
    max_request_chars: int = 240000
    max_actions: int = 32
    max_reasoning_chars: int = 600
    max_behavior_chars: int = 300
    min_retire_support_groups: int = 2
    max_parse_attempts: int = 3

    def __post_init__(self) -> None:
        for name, value in as_json_dict(self).items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"CodebookUpdaterBounds.{name} must be positive")


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path) as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMCodebookUpdaterError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise LLMCodebookUpdaterError(f"expected JSON object: {path}")
    return value


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LLMCodebookUpdaterError(f"missing artifact: {path}") from exc
    return digest.hexdigest()


def _write_immutable(path: str, value: Any) -> str:
    text = json.dumps(
        as_json_dict(value), sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    if os.path.exists(path):
        with open(path) as handle:
            if handle.read() != text:
                raise LLMCodebookUpdaterConflictError(
                    f"immutable updater artifact conflicts: {path}")
        return os.path.abspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, text)
    return os.path.abspath(path)


def _write_text_immutable(path: str, text: str) -> str:
    if os.path.exists(path):
        with open(path) as handle:
            if handle.read() != text:
                raise LLMCodebookUpdaterConflictError(
                    f"immutable updater text conflicts: {path}")
        return os.path.abspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, text)
    return os.path.abspath(path)


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _tokens(value: str) -> frozenset[str]:
    stop = {"a", "an", "and", "the", "to", "of", "or", "when", "only",
            "describe", "preserve", "capture", "include", "visible", "visually"}
    return frozenset(token for token in re.findall(r"[a-z0-9]+", value.casefold())
                     if len(token) > 2 and token not in stop)


def _text_rejection(text: str) -> str | None:
    normalized = " ".join(str(text or "").split())
    if len(normalized) < 12 or len(normalized) > 300:
        return "property text must contain 12-300 normalized characters"
    lowered = normalized.casefold()
    forbidden = (
        r"\bthis video\b", r"\bthe question\b", r"\bground[ -]?truth\b",
        r"\bcorrect answer\b", r"\banswer (?:is|option)\b",
        r"\boption\s+[a-z0-9]\b", r"\b\d{1,2}:\d{2}(?::\d{2})?\b",
        r"\b\d+_\d+\b", r"\bvideo[-_ ]?id\b", r"\bquestion[-_ ]?id\b",
)


def _non_visual_knowledge_reason(text: str) -> str | None:
    """Updater-only semantic guard; proposal generation intentionally defers it."""
    lowered = text.casefold()
    direct = (
        r"\bnon[- ]visual (?:knowledge|information|context|facts?)\b",
        r"\bexternal (?:knowledge|information|context|facts?|sources?)\b",
        r"\bbackground (?:knowledge|information|context|facts?|research)\b",
        r"\bhistorical (?:knowledge|context|background|facts?|research)\b",
        r"\b(?:world|domain|prior) knowledge\b",
        r"\bbeyond (?:what is|the content) (?:visible|shown|depicted)\b",
        r"\boutside (?:the|of the) (?:frames?|video|visual evidence)\b",
    )
    if any(re.search(pattern, lowered) for pattern in direct):
        return "requires non-visual or external/background/historical knowledge"
    return None
    if any(re.search(pattern, lowered) for pattern in forbidden):
        return "property text contains instance-specific lineage or answer language"
    if re.search(r"\b(?:describe|capture|record|track|transcribe)\s+(?:the|a|an)\s+"
                 r"[a-z][a-z-]*['’]s\b", lowered):
        return "property text contains a domain-specific possessive subject"
    knowledge = _non_visual_knowledge_reason(normalized)
    if knowledge:
        return knowledge
    return None


def _provider_identity(provider: Callable[..., str]) -> Mapping[str, Any]:
    metadata = getattr(provider, "metadata", None)
    if callable(metadata):
        return as_json_dict(metadata())
    return {"type": f"{type(provider).__module__}.{type(provider).__qualname__}",
            "model": getattr(provider, "model", None)}


def parse_updater_plan(raw: str, *, max_actions: int) -> tuple[dict[str, Any], ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMCodebookUpdaterParseError("updater response is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "actions"} or \
            value.get("schema_version") != UPDATER_RESPONSE_SCHEMA_VERSION:
        raise LLMCodebookUpdaterParseError("invalid updater response envelope")
    actions = value.get("actions")
    if not isinstance(actions, list) or len(actions) > max_actions:
        raise LLMCodebookUpdaterParseError("actions must be a bounded array")
    output = []
    seen = set()
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or set(action) != ACTION_FIELDS:
            raise LLMCodebookUpdaterParseError(
                f"action {index} does not match the strict action schema")
        if not isinstance(action["action_id"], str) or not action["action_id"] or \
                action["action_id"] in seen:
            raise LLMCodebookUpdaterParseError("action IDs must be unique strings")
        seen.add(action["action_id"])
        if action["action"] not in ACTION_NAMES:
            raise LLMCodebookUpdaterParseError("unknown updater action")
        for field in ("target_property_ids", "candidate_ids",
                      "supporting_memory_example_ids", "supporting_evidence_ids",
                      "behaviors_to_preserve"):
            items = action[field]
            if not isinstance(items, list) or any(
                    not isinstance(item, str) or not item for item in items) or \
                    len(items) != len(set(items)):
                raise LLMCodebookUpdaterParseError(
                    f"{field} must be a unique string array")
        for field in ("proposed_property_id", "proposed_property_text"):
            if action[field] is not None and (
                    not isinstance(action[field], str) or not action[field]):
                raise LLMCodebookUpdaterParseError(f"{field} must be null or string")
        if not isinstance(action["reasoning"], str) or not action["reasoning"]:
            raise LLMCodebookUpdaterParseError("reasoning must be a non-empty string")
        confidence = action["confidence"]
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or \
                not 0.0 <= float(confidence) <= 1.0:
            raise LLMCodebookUpdaterParseError("confidence must be in [0,1]")
        output.append(action)
    return tuple(output)


class MemoryConditionedLLMCodebookUpdater:
    policy_version = "memory_conditioned_llm_codebook_updater_v3_strict_retry"

    def __init__(
        self, *, response_provider: Callable[[str, str], str],
        bounds: CodebookUpdaterBounds | None = None,
        system_prompt_path: str = PROMPT_PATH,
    ) -> None:
        self.response_provider = response_provider
        self.bounds = bounds or CodebookUpdaterBounds()
        self.system_prompt_path = os.path.abspath(system_prompt_path)
        with open(self.system_prompt_path) as handle:
            self.system_prompt = handle.read()
        self.configuration_identity = {
            "policy_version": self.policy_version,
            "request_schema_version": UPDATER_REQUEST_SCHEMA_VERSION,
            "response_schema_version": UPDATER_RESPONSE_SCHEMA_VERSION,
            "validation_schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
            "candidate_codebook_schema_version": CANDIDATE_CODEBOOK_SCHEMA_VERSION,
            "system_prompt_version": SYSTEM_PROMPT_VERSION,
            "system_prompt_sha256": sha256_text(self.system_prompt),
            "bounds": as_json_dict(self.bounds),
            "provider": _provider_identity(response_provider),
        }

    def run(
        self, *, iteration_id: str, prompt_bank: PromptBankSnapshot,
        property_memory_snapshot_path: str,
        intervention_summary_path: str, output_dir: str,
    ) -> Mapping[str, Any]:
        output_dir = os.path.abspath(output_dir)
        manifest_path = os.path.join(output_dir, "manifest.json")
        memory = _read_json(property_memory_snapshot_path)
        if memory.get("schema_version") != "property_memory_v1":
            raise LLMCodebookUpdaterConflictError("incompatible property-memory schema")
        effects = tuple(
            json.loads(line) for line in open(intervention_summary_path) if line.strip())
        request = self._build_request(
            iteration_id=iteration_id, prompt_bank=prompt_bank,
            memory=memory, effects=effects)
        identity = {
            "configuration": self.configuration_identity,
            "iteration_id": iteration_id,
            "bank_hash": sha256_json(as_json_dict(prompt_bank)),
            "memory_path": os.path.abspath(property_memory_snapshot_path),
            "memory_hash": _file_hash(property_memory_snapshot_path),
            "effect_path": os.path.abspath(intervention_summary_path),
            "effect_hash": _file_hash(intervention_summary_path),
            "request": request,
        }
        fingerprint = sha256_json(identity)
        if os.path.exists(manifest_path):
            existing = _read_json(manifest_path)
            if existing.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA_VERSION or \
                    existing.get("input_fingerprint") != fingerprint or \
                    existing.get("status") != "completed":
                raise LLMCodebookUpdaterConflictError(
                    "completed updater artifact uses incompatible semantics")
            self._validate_resume(existing)
            return {**existing, "resumed": True}
        identity_path = os.path.join(output_dir, "input_identity.json")
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            if not os.path.exists(identity_path):
                raise LLMCodebookUpdaterConflictError(
                    "incomplete updater directory predates strict retry semantics")
            existing_identity = _read_json(identity_path)
            if existing_identity.get("schema_version") != \
                    UPDATER_INPUT_IDENTITY_SCHEMA_VERSION or \
                    existing_identity.get("input_fingerprint") != fingerprint:
                raise LLMCodebookUpdaterConflictError(
                    "incomplete updater input identity mismatch")
        identity_path = _write_immutable(identity_path, {
            "schema_version": UPDATER_INPUT_IDENTITY_SCHEMA_VERSION,
            "input_fingerprint": fingerprint,
            "identity": identity,
        })
        prompt_copy = _write_text_immutable(
            os.path.join(output_dir, "system_prompt.txt"), self.system_prompt)
        request_path = _write_immutable(os.path.join(output_dir, "request.json"), request)
        raw, actions, attempts_path = self._load_or_generate_plan(
            output_dir=output_dir, request=request)
        raw_path = _write_text_immutable(os.path.join(output_dir, "raw_response.txt"), raw)
        parsed_path = _write_immutable(os.path.join(output_dir, "parsed_plan.json"), {
            "schema_version": UPDATER_RESPONSE_SCHEMA_VERSION, "actions": actions})
        validated, rejected = self._validate_actions(
            actions=actions, prompt_bank=prompt_bank, memory=memory, effects=effects)
        validation_path = _write_immutable(
            os.path.join(output_dir, "validation_report.json"), {
                "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
                "accepted_action_ids": tuple(item["action_id"] for item in validated),
                "rejected_actions": rejected,
            })
        rejected_path = _write_immutable(
            os.path.join(output_dir, "rejected_actions.json"), {
                "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
                "actions": rejected})
        candidate_bank, mapping, promotions, summaries = self._apply_actions(
            prompt_bank=prompt_bank, actions=validated, memory=memory,
            iteration_id=iteration_id)
        applied_path = _write_immutable(os.path.join(output_dir, "applied_plan.json"), {
            "schema_version": APPLIED_PLAN_SCHEMA_VERSION,
            "actions": validated, **summaries})
        bank_path = _write_immutable(os.path.join(output_dir, "candidate_codebook.json"), {
            "schema_version": CANDIDATE_CODEBOOK_SCHEMA_VERSION,
            "input_bank_version": prompt_bank.bank_version,
            "candidate_bank": candidate_bank,
            "candidate_codebook_hash": sha256_json(as_json_dict(candidate_bank)),
            "lineage": {"iteration_id": iteration_id,
                        "updater_policy_version": self.policy_version,
                        "memory_snapshot_hash": _file_hash(property_memory_snapshot_path)},
        })
        mapping_path = _write_immutable(os.path.join(output_dir, "property_id_mapping.json"), {
            "schema_version": ID_MAPPING_SCHEMA_VERSION,
            "old_to_new_property_ids": mapping,
            "candidate_promotions": promotions,
        })
        promoted_memory_path = _write_immutable(
            os.path.join(output_dir, "promoted_property_memory.json"),
            self._promote_memory(memory, candidate_bank, promotions, validated))
        artifacts = {
            "input_identity": identity_path,
            "system_prompt": prompt_copy, "request": request_path,
            "response_attempts": attempts_path,
            "raw_response": raw_path, "parsed_plan": parsed_path,
            "validation_report": validation_path, "rejected_actions": rejected_path,
            "applied_plan": applied_path, "candidate_codebook": bank_path,
            "property_id_mapping": mapping_path,
            "promoted_property_memory": promoted_memory_path,
        }
        manifest = {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            "status": "completed", "input_fingerprint": fingerprint,
            "iteration_id": iteration_id, "system_prompt_version": SYSTEM_PROMPT_VERSION,
            "artifacts": artifacts,
            "artifact_hashes": {name: _file_hash(path)
                                for name, path in artifacts.items()},
            "accepted_action_count": len(validated),
            "rejected_action_count": len(rejected), "resumed": False,
        }
        _write_immutable(manifest_path, manifest)
        return _read_json(manifest_path)

    def _load_or_generate_plan(
        self, *, output_dir: str, request: Mapping[str, Any],
    ) -> tuple[str, tuple[dict[str, Any], ...], str]:
        attempts = []
        previous_error = None
        request_text = dumps_canonical(request)
        last_error: LLMCodebookUpdaterParseError | None = None
        for attempt_index in range(1, self.bounds.max_parse_attempts + 1):
            attempt_dir = os.path.join(
                output_dir, "attempts", f"attempt_{attempt_index:03d}")
            raw_path = os.path.join(attempt_dir, "raw_response.txt")
            if os.path.exists(raw_path):
                with open(raw_path) as handle:
                    raw = handle.read()
            else:
                repair = "" if previous_error is None else (
                    "\n\nThe previous response failed strict parsing: " +
                    previous_error +
                    "\nReturn a complete object matching the required schema exactly.")
                raw = self.response_provider(
                    self.system_prompt + repair, request_text)
                if not isinstance(raw, str):
                    raise LLMCodebookUpdaterError(
                        "updater provider must return text")
                _write_text_immutable(raw_path, raw)
            try:
                actions = parse_updater_plan(
                    raw, max_actions=self.bounds.max_actions)
            except LLMCodebookUpdaterParseError as exc:
                last_error = exc
                previous_error = str(exc)
                status = "rejected"
                error = {"type": type(exc).__name__, "message": str(exc)}
            else:
                status, error = "accepted", None
            result_path = _write_immutable(
                os.path.join(attempt_dir, "result.json"), {
                    "schema_version": UPDATER_ATTEMPTS_SCHEMA_VERSION,
                    "attempt_index": attempt_index,
                    "status": status,
                    "raw_source": "provider_response",
                    "raw_response_sha256": sha256_text(raw),
                    "parse_error": error,
                })
            attempts.append({
                "attempt_index": attempt_index, "status": status,
                "raw_response_path": os.path.abspath(raw_path),
                "result_path": result_path,
            })
            if status == "accepted":
                attempts_path = _write_immutable(
                    os.path.join(output_dir, "response_attempts.json"), {
                        "schema_version": UPDATER_ATTEMPTS_SCHEMA_VERSION,
                        "max_parse_attempts": self.bounds.max_parse_attempts,
                        "accepted_attempt_index": attempt_index,
                        "attempts": attempts,
                    })
                return raw, actions, attempts_path
        assert last_error is not None
        _write_immutable(os.path.join(output_dir, "response_attempts.json"), {
            "schema_version": UPDATER_ATTEMPTS_SCHEMA_VERSION,
            "max_parse_attempts": self.bounds.max_parse_attempts,
            "accepted_attempt_index": None, "attempts": attempts,
        })
        raise last_error

    def _build_request(
        self, *, iteration_id: str, prompt_bank: PromptBankSnapshot,
        memory: Mapping[str, Any], effects: tuple[Mapping[str, Any], ...],
    ) -> dict[str, Any]:
        properties = tuple(memory.get("properties") or ())
        candidates = tuple(memory.get("candidates") or ())
        recent = []
        for item in properties:
            decision = item.get("latest_decision") or {}
            if decision.get("action") in ("no_op", "preserve"):
                recent.append({"property_id": item["property_id"], **decision})
        for item in candidates:
            if item.get("no_effect_examples"):
                recent.append({"candidate_id": item["candidate_property_id"],
                               "reason": "bounded no-effect intervention memory"})
        request = {
            "schema_version": UPDATER_REQUEST_SCHEMA_VERSION,
            "system_prompt_version": SYSTEM_PROMPT_VERSION,
            "iteration_id": iteration_id,
            "current_codebook": tuple({
                "property_id": entry.prompt_id, "property_text": entry.prompt_text,
                "name": entry.name, "description": entry.description,
                "status": entry.status,
                "applicability_traits": as_json_dict(entry.applicability_traits),
            } for entry in prompt_bank.entries),
            "active_property_memories": properties,
            "candidate_memories": candidates,
            "current_intervention_summaries": effects,
            "recent_rejection_or_no_op_summaries": tuple(recent[-12:]),
            "output_schema": {
                "schema_version": UPDATER_RESPONSE_SCHEMA_VERSION,
                "actions": [{
                    "action_id": "unique string", "action": " | ".join(ACTION_NAMES),
                    "target_property_ids": ["active property ID"],
                    "candidate_ids": ["candidate ID"],
                    "proposed_property_id": "string or null",
                    "proposed_property_text": "string or null",
                    "reasoning": "concise bounded reasoning",
                    "supporting_memory_example_ids": ["existing example_id"],
                    "supporting_evidence_ids": ["existing summary_id"],
                    "behaviors_to_preserve": ["documented intended behavior"],
                    "confidence": 0.0,
                }],
            },
        }
        if len(dumps_canonical(request)) > self.bounds.max_request_chars:
            raise LLMCodebookUpdaterError("bounded updater request exceeds limit")
        return request

    def _validate_actions(
        self, *, actions: tuple[dict[str, Any], ...],
        prompt_bank: PromptBankSnapshot, memory: Mapping[str, Any],
        effects: tuple[Mapping[str, Any], ...],
    ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        active = {entry.prompt_id: entry for entry in prompt_bank.entries
                  if entry.status == "active"}
        properties = {item["property_id"]: item for item in memory.get("properties") or ()}
        candidate_rows = tuple(memory.get("candidates") or ())
        candidates = {item["candidate_property_id"]: item for item in candidate_rows}
        if len(candidates) != len(candidate_rows):
            raise LLMCodebookUpdaterError(
                "candidate IDs must be unique within one updater request")
        example_ids = set()
        for item in (*properties.values(), *candidates.values()):
            for field in ("positive_examples", "negative_examples", "mixed_examples",
                          "no_effect_examples"):
                example_ids.update(example.get("example_id")
                                   for example in item.get(field) or ())
            routing = item.get("routing_examples") or {}
            for values in routing.values():
                example_ids.update(example.get("example_id") for example in values)
        evidence_ids = {item.get("summary_id") for item in effects}
        accepted, rejected = [], []
        claimed_targets, claimed_candidates, proposed_ids = set(), set(), set()
        for action in actions:
            reasons = []
            name = action["action"]
            targets = tuple(action["target_property_ids"])
            candidate_ids = tuple(action["candidate_ids"])
            if set(targets) - set(active):
                reasons.append("references unknown or inactive target property")
            if set(candidate_ids) - set(candidates):
                reasons.append("references unknown candidate")
            if set(action["supporting_memory_example_ids"]) - example_ids:
                reasons.append("references unknown memory example")
            if set(action["supporting_evidence_ids"]) - evidence_ids:
                reasons.append("references unknown compact evidence")
            if len(action["reasoning"]) > self.bounds.max_reasoning_chars:
                reasons.append("reasoning exceeds configured bound")
            if any(len(item) > self.bounds.max_behavior_chars
                   for item in action["behaviors_to_preserve"]):
                reasons.append("preserved behavior exceeds configured bound")
            proposed_id = action["proposed_property_id"]
            proposed_text = action["proposed_property_text"]
            if proposed_id and not re.fullmatch(r"pe_[a-z0-9_]+", proposed_id):
                reasons.append("proposed property ID must use stable pe_snake_case form")
            if name == "add":
                if len(candidate_ids) != 1 or targets or not proposed_id or not proposed_text:
                    reasons.append("add requires one candidate, no target, and proposed ID/text")
                elif proposed_id in active or proposed_id in proposed_ids:
                    reasons.append("added property ID collides")
                else:
                    candidate = candidates[candidate_ids[0]]
                    completed = tuple(candidate.get("positive_examples") or ()) + tuple(
                        candidate.get("negative_examples") or ()) + tuple(
                        candidate.get("mixed_examples") or ()) + tuple(
                        candidate.get("no_effect_examples") or ())
                    if not completed:
                        reasons.append("add candidate lacks completed intervention memory")
                    if not candidate.get("positive_examples"):
                        reasons.append("add lacks positive intervention support")
                    elif not ({item["example_id"] for item in
                               candidate.get("positive_examples") or ()} & set(
                                   action["supporting_memory_example_ids"])):
                        reasons.append("add does not cite its positive candidate memory")
            elif name == "revise":
                if len(targets) != 1 or not proposed_text or proposed_id != targets[0]:
                    reasons.append("revise must preserve one existing property identity")
                elif not self._preserves_positive_behavior(
                        properties.get(targets[0], {}), action, proposed_text):
                    reasons.append("revise does not preserve documented positive behavior")
            elif name == "merge":
                if len(targets) < 2 or not proposed_text or proposed_id not in targets:
                    reasons.append("merge requires two active sources and one canonical source ID")
                elif not all(self._preserves_positive_behavior(
                        properties.get(target, {}), action, proposed_text)
                             for target in targets):
                    reasons.append("merge does not preserve source positive behavior")
            elif name == "retire":
                if len(targets) != 1 or candidate_ids or proposed_id or proposed_text:
                    reasons.append("retire requires exactly one active target")
                elif self._harmful_support(properties.get(targets[0], {})) < \
                        self.bounds.min_retire_support_groups:
                    reasons.append("retire lacks harmful support across distinct videos or iterations")
            elif name in ("preserve", "no_op"):
                if proposed_id or proposed_text:
                    reasons.append(f"{name} cannot propose property ID or text")
            if name in ("add", "revise", "merge", "retire") and not (
                    action["supporting_memory_example_ids"] or
                    action["supporting_evidence_ids"]):
                reasons.append("mutating action must cite bounded support")
            if proposed_text:
                text_reason = _text_rejection(proposed_text)
                if text_reason:
                    reasons.append(text_reason)
                normalized_text = _normalized(proposed_text)
                duplicate = next((property_id for property_id, entry in active.items()
                                  if _normalized(entry.prompt_text) == normalized_text
                                  and property_id not in targets), None)
                if duplicate:
                    reasons.append("proposed text duplicates another active property")
            mutating = name in ("add", "revise", "merge", "retire")
            if mutating and (set(targets) & claimed_targets or
                             set(candidate_ids) & claimed_candidates):
                reasons.append("action conflicts with another accepted mutation")
            if reasons:
                rejected.append({"action_id": action["action_id"],
                                 "action": name, "reasons": tuple(reasons)})
                continue
            accepted.append(action)
            if mutating:
                claimed_targets.update(targets)
                claimed_candidates.update(candidate_ids)
                if proposed_id:
                    proposed_ids.add(proposed_id)
        return tuple(accepted), tuple(rejected)

    @staticmethod
    def _preserves_positive_behavior(
        memory: Mapping[str, Any], action: Mapping[str, Any], proposed_text: str,
    ) -> bool:
        positive = tuple(memory.get("positive_examples") or ())
        intended = str(memory.get("intended_behavior") or "")
        behaviors = tuple(action["behaviors_to_preserve"])
        if positive and not set(example["example_id"] for example in positive) <= set(
                action["supporting_memory_example_ids"]):
            return False
        if intended and _normalized(intended) not in {_normalized(item) for item in behaviors}:
            return False
        return not intended or bool(_tokens(intended) & _tokens(proposed_text))

    @staticmethod
    def _harmful_support(memory: Mapping[str, Any]) -> int:
        groups = {(str(item.get("video_id") or ""),
                   int(item.get("iteration_ordinal", 0)))
                  for item in memory.get("negative_examples") or ()}
        return len(groups)

    def _apply_actions(
        self, *, prompt_bank: PromptBankSnapshot,
        actions: tuple[Mapping[str, Any], ...], memory: Mapping[str, Any],
        iteration_id: str,
    ) -> tuple[PromptBankSnapshot, dict[str, str | None], dict[str, str], dict[str, Any]]:
        entries = list(prompt_bank.entries)
        by_id = {entry.prompt_id: index for index, entry in enumerate(entries)}
        mapping: dict[str, str | None] = {
            entry.prompt_id: entry.prompt_id for entry in prompt_bank.entries}
        promotions = {}
        candidates = {item["candidate_property_id"]: item
                      for item in memory.get("candidates") or ()}
        summaries = {name: [] for name in (
            "added", "revised", "merged", "preserved", "retired", "no_op")}
        summary_field = {
            "add": "added", "revise": "revised", "merge": "merged",
            "preserve": "preserved", "retire": "retired", "no_op": "no_op",
        }
        changed = False
        for action in actions:
            name = action["action"]
            summaries[summary_field[name]].append(action["action_id"])
            if name in ("preserve", "no_op"):
                continue
            changed = True
            proposed_text = action["proposed_property_text"]
            if name == "add":
                property_id = action["proposed_property_id"]
                candidate = candidates[action["candidate_ids"][0]]
                entries.append(PromptEntry(
                    prompt_id=property_id, prompt_text=proposed_text,
                    prompt_hash=sha256_text(proposed_text),
                    name="Memory-conditioned reusable property",
                    description="Validated from bounded property memory.",
                    tags=("learned", "memory_conditioned"), created_by=self.policy_version,
                    applicability_traits=candidate.get("applicability") or {},
                    provenance={"iteration_id": iteration_id,
                                "action_id": action["action_id"],
                                "candidate_ids": action["candidate_ids"],
                                "proposal_failure_analysis": candidate.get(
                                    "failure_analysis")}))
                promotions[action["candidate_ids"][0]] = property_id
            elif name == "revise":
                target = action["target_property_ids"][0]
                current = entries[by_id[target]]
                history = tuple(current.provenance.get("revision_history") or ())
                entries[by_id[target]] = replace(
                    current, prompt_text=proposed_text,
                    prompt_hash=sha256_text(proposed_text),
                    parent_prompt_ids=tuple(dict.fromkeys((*current.parent_prompt_ids, target))),
                    created_by=self.policy_version,
                    provenance={**as_json_dict(current.provenance),
                                "revision_history": (*history, {
                                    "iteration_id": iteration_id,
                                    "action_id": action["action_id"],
                                    "previous_text": current.prompt_text})})
            elif name == "merge":
                canonical = action["proposed_property_id"]
                sources = tuple(action["target_property_ids"])
                current = entries[by_id[canonical]]
                aliases = tuple(dict.fromkeys((
                    *(current.provenance.get("aliases") or ()),
                    *(item for source in sources for item in (
                        source, entries[by_id[source]].prompt_text)),
                )))
                entries[by_id[canonical]] = replace(
                    current, prompt_text=proposed_text,
                    prompt_hash=sha256_text(proposed_text),
                    parent_prompt_ids=tuple(dict.fromkeys(
                        (*current.parent_prompt_ids, *sources))),
                    created_by=self.policy_version,
                    provenance={**as_json_dict(current.provenance),
                                "aliases": aliases, "merge_action_id": action["action_id"]})
                for source in sources:
                    mapping[source] = canonical
                    if source != canonical:
                        entries[by_id[source]] = replace(
                            entries[by_id[source]], status="retired")
                for candidate_id in action["candidate_ids"]:
                    promotions[candidate_id] = canonical
            elif name == "retire":
                target = action["target_property_ids"][0]
                entries[by_id[target]] = replace(entries[by_id[target]], status="retired")
                mapping[target] = None
        if changed:
            match = re.search(r"_v(\d+)", prompt_bank.bank_version)
            number = int(match.group(1)) + 1 if match else 1
            payload_hash = sha256_json({"parent": prompt_bank.bank_version,
                                        "entries": as_json_dict(entries),
                                        "actions": as_json_dict(actions)})
            bank = replace(
                prompt_bank, bank_version=f"bank_v{number:04d}_{payload_hash[:8]}",
                parent_bank_version=prompt_bank.bank_version, entries=tuple(entries),
                default_entry_ids=tuple(dict.fromkeys(
                    mapping.get(item, item) for item in prompt_bank.default_entry_ids
                    if mapping.get(item, item) is not None)),
                created_by=self.policy_version,
                provenance={"iteration_id": iteration_id,
                            "validated_action_ids": tuple(
                                action["action_id"] for action in actions),
                            "candidate_only": True})
        else:
            bank = prompt_bank
        errors = validate_prompt_bank(bank)
        if errors:
            raise LLMCodebookUpdaterError("; ".join(errors))
        return bank, mapping, promotions, summaries

    @staticmethod
    def _promote_memory(
        memory: Mapping[str, Any], candidate_bank: PromptBankSnapshot,
        promotions: Mapping[str, str], actions: tuple[Mapping[str, Any], ...],
    ) -> Mapping[str, Any]:
        value = json.loads(json.dumps(memory))
        value["schema_version"] = "property_memory_v1"
        value["active_bank_version"] = candidate_bank.bank_version
        by_candidate = {item["candidate_property_id"]: item
                        for item in value.get("candidates") or ()}
        by_property = {item["property_id"]: item
                       for item in value.get("properties") or ()}
        action_by_candidate = {
            candidate_id: action for action in actions
            for candidate_id in action.get("candidate_ids") or ()}
        for candidate_id, property_id in promotions.items():
            candidate = by_candidate[candidate_id]
            candidate["status"] = "promoted"
            candidate["promoted_to_property_id"] = property_id
            if property_id not in by_property:
                entry = candidate_bank.entry(property_id)
                by_property[property_id] = {
                    "schema_version": "property_memory_v1", "property_id": property_id,
                    "property_text": entry.prompt_text,
                    "applicability": candidate.get("applicability"),
                    "failure_analysis": candidate.get("failure_analysis"),
                    "why_created": candidate.get("why_proposed"),
                    "intended_behavior": entry.prompt_text,
                    "origin": {"kind": "promoted_candidate",
                               "candidate_property_id": candidate_id,
                               "artifact_refs": candidate.get("artifact_refs") or []},
                    "positive_examples": candidate.get("positive_examples") or [],
                    "negative_examples": candidate.get("negative_examples") or [],
                    "no_effect_examples": candidate.get("no_effect_examples") or [],
                    "routing_examples": {"positive": [], "negative": []},
                    "aliases": [], "revision_history": [],
                    "latest_decision": {"action": "add", "reason": "validated LLM action",
                                        "artifact_refs": []},
                }
            else:
                target = by_property[property_id]
                for field, limit in (("positive_examples", 5),
                                     ("negative_examples", 3),
                                     ("no_effect_examples", 2)):
                    combined = [*target.get(field, ()), *candidate.get(field, ())]
                    unique = {item["example_id"]: item for item in combined}
                    selected = []
                    videos = set()
                    for item in sorted(unique.values(), key=lambda row: (
                            0 if str(row.get("video_id") or "") not in videos else 1,
                            -int(row.get("iteration_ordinal", 0)), row["example_id"])):
                        if len(selected) >= limit:
                            break
                        selected.append(item)
                        videos.add(str(item.get("video_id") or ""))
                    target[field] = selected
                target["revision_history"] = [*target.get("revision_history", ()), {
                    "action": action_by_candidate[candidate_id]["action"],
                    "action_id": action_by_candidate[candidate_id]["action_id"],
                    "candidate_property_id": candidate_id,
                }]
        for action in actions:
            if action["action"] == "revise":
                target = by_property[action["target_property_ids"][0]]
                target["property_text"] = action["proposed_property_text"]
                target["intended_behavior"] = action["proposed_property_text"]
                target["revision_history"] = [*target.get("revision_history", ()), {
                    "action": "revise", "action_id": action["action_id"],
                    "previous_intended_behavior": action["behaviors_to_preserve"],
                }]
            elif action["action"] == "merge":
                canonical = action["proposed_property_id"]
                target = by_property[canonical]
                target["property_text"] = action["proposed_property_text"]
                target["intended_behavior"] = action["proposed_property_text"]
                target["aliases"] = sorted(set((
                    *target.get("aliases", ()),
                    *(item for source in action["target_property_ids"] for item in (
                        source, by_property[source]["property_text"])),
                )))
                for source in action["target_property_ids"]:
                    if source == canonical:
                        continue
                    source_memory = by_property[source]
                    for field, limit in (("positive_examples", 5),
                                         ("negative_examples", 3),
                                         ("no_effect_examples", 2)):
                        combined = [*target.get(field, ()),
                                    *source_memory.get(field, ())]
                        target[field] = list({item["example_id"]: item
                                              for item in combined}.values())[:limit]
                    target_routing = target.setdefault(
                        "routing_examples", {"positive": [], "negative": []})
                    source_routing = source_memory.get("routing_examples") or {}
                    for polarity in ("positive", "negative"):
                        combined = [*target_routing.get(polarity, ()),
                                    *source_routing.get(polarity, ())]
                        target_routing[polarity] = list({
                            item["example_id"]: item for item in combined
                        }.values())[:2]
                    del by_property[source]
            elif action["action"] == "retire":
                by_property.pop(action["target_property_ids"][0], None)
        value["properties"] = [by_property[key] for key in sorted(by_property)]
        value["candidates"] = [by_candidate[key] for key in sorted(by_candidate)]
        value["promotion_action_ids"] = [action["action_id"] for action in actions
                                         if action["action"] in ("add", "merge")]
        return value

    @staticmethod
    def _validate_resume(manifest: Mapping[str, Any]) -> None:
        artifacts = manifest.get("artifacts") or {}
        hashes = manifest.get("artifact_hashes") or {}
        if set(artifacts) != set(hashes) or not artifacts:
            raise LLMCodebookUpdaterConflictError("incomplete updater manifest")
        for name, path in artifacts.items():
            if not os.path.exists(path) or hashes[name] != _file_hash(path):
                raise LLMCodebookUpdaterConflictError(
                    f"updater artifact hash mismatch: {name}")
        candidate = _read_json(artifacts["candidate_codebook"])
        if candidate.get("schema_version") != CANDIDATE_CODEBOOK_SCHEMA_VERSION:
            raise LLMCodebookUpdaterConflictError("incompatible candidate-codebook schema")
