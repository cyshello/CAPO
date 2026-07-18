"""Memory-conditioned LLM router-policy planning with deterministic validation."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from surrogate_rollout.prompt_routing.persistence import (
    _atomic_write_text,
    prompt_bank_from_json,
    router_policy_from_json,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    RouterPolicySnapshot,
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.prompt_routing.structured_router_policy import (
    NEGATIVE_EXAMPLE_LIMIT,
    POSITIVE_EXAMPLE_LIMIT,
    RENDERED_ROUTER_PROMPT_SCHEMA_VERSION,
    ROUTER_PROMPT_RENDERER_VERSION,
    STRUCTURED_ROUTER_POLICY_SCHEMA_VERSION,
    bootstrap_structured_router_policy,
    install_structured_router_policy,
    remap_structured_router_policy,
    render_router_prompt,
    rendered_prompt_artifact,
    validate_rendered_prompt_configuration,
    validate_structured_router_policy,
)
from surrogate_rollout.prompt_routing.validators import validate_router_against_bank
from surrogate_rollout.schemas import sha256_json, sha256_text


SYSTEM_PROMPT_VERSION = "memory_router_updater_prompt_v1"
ROUTER_UPDATER_REQUEST_SCHEMA_VERSION = "memory_router_updater_request_v1"
ROUTER_UPDATER_RESPONSE_SCHEMA_VERSION = "memory_router_updater_response_v1"
ROUTER_UPDATER_PLAN_SCHEMA_VERSION = "memory_router_updater_plan_v1"
ROUTER_VALIDATION_SCHEMA_VERSION = "memory_router_validation_report_v1"
ROUTER_APPLIED_PLAN_SCHEMA_VERSION = "memory_router_applied_plan_v1"
ROUTER_UPDATER_MANIFEST_SCHEMA_VERSION = "memory_router_updater_manifest_v2"
ROUTER_EXECUTION_SCHEMA_VERSION = "memory_router_updater_execution_v2"
ROUTER_EXECUTION_POLICY_VERSION = "no_routing_evidence_empty_plan_v1"
ROUTER_INPUT_IDENTITY_SCHEMA_VERSION = "memory_router_updater_input_identity_v1"
ROUTER_ATTEMPTS_SCHEMA_VERSION = "memory_router_updater_attempts_v1"
ACTION_TYPES = (
    "set_selection_guidance", "set_avoidance_guidance",
    "add_positive_example", "add_negative_example", "preserve", "no_op",
)
ACTION_FIELDS = frozenset({
    "action_id", "target_property_id", "action_type", "proposed_value",
    "reason", "supporting_memory_example_ids", "supporting_evidence_ids",
    "confidence",
})
PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "prompts", "router_updater_v1.txt")


class LLMRouterUpdaterError(RuntimeError):
    pass


class LLMRouterUpdaterConflictError(LLMRouterUpdaterError):
    pass


class LLMRouterUpdaterParseError(LLMRouterUpdaterError):
    pass


@dataclass(frozen=True)
class RouterUpdaterBounds:
    max_request_chars: int = 180000
    max_actions: int = 48
    max_guidance_chars: int = 500
    max_example_chars: int = 300
    max_parse_attempts: int = 3

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or value < 1
               for value in as_json_dict(self).values()):
            raise ValueError("RouterUpdaterBounds values must be positive")


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path) as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMRouterUpdaterError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise LLMRouterUpdaterError(f"expected JSON object: {path}")
    return value


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LLMRouterUpdaterError(f"missing artifact: {path}") from exc
    return digest.hexdigest()


def _write_immutable(path: str, value: Any) -> str:
    text = json.dumps(
        as_json_dict(value), sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    if os.path.exists(path):
        with open(path) as handle:
            if handle.read() != text:
                raise LLMRouterUpdaterConflictError(
                    f"immutable router-updater artifact conflicts: {path}")
        return os.path.abspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, text)
    return os.path.abspath(path)


def _write_text_immutable(path: str, text: str) -> str:
    if os.path.exists(path):
        with open(path) as handle:
            if handle.read() != text:
                raise LLMRouterUpdaterConflictError(
                    f"immutable router-updater text conflicts: {path}")
        return os.path.abspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_text(path, text)
    return os.path.abspath(path)


def _provider_identity(provider: Callable[..., str]) -> Mapping[str, Any]:
    metadata = getattr(provider, "metadata", None)
    if callable(metadata):
        return as_json_dict(metadata())
    return {"type": f"{type(provider).__module__}.{type(provider).__qualname__}",
            "model": getattr(provider, "model", None)}


def parse_router_updater_plan(
    raw: str, *, max_actions: int,
) -> tuple[dict[str, Any], ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMRouterUpdaterParseError("router updater response is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "actions"} or \
            value.get("schema_version") != ROUTER_UPDATER_RESPONSE_SCHEMA_VERSION:
        raise LLMRouterUpdaterParseError("invalid router updater response envelope")
    actions = value["actions"]
    if not isinstance(actions, list) or len(actions) > max_actions:
        raise LLMRouterUpdaterParseError("router actions must be a bounded array")
    output, seen = [], set()
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or set(action) != ACTION_FIELDS:
            raise LLMRouterUpdaterParseError(
                f"router action {index} does not match the strict schema")
        if not isinstance(action["action_id"], str) or not action["action_id"] or \
                action["action_id"] in seen:
            raise LLMRouterUpdaterParseError("router action IDs must be unique strings")
        seen.add(action["action_id"])
        if action["action_type"] not in ACTION_TYPES:
            raise LLMRouterUpdaterParseError("unknown router action type")
        if not isinstance(action["target_property_id"], str) or not \
                action["target_property_id"]:
            raise LLMRouterUpdaterParseError("target_property_id must be a string")
        if action["proposed_value"] is not None and (
                not isinstance(action["proposed_value"], str) or
                not action["proposed_value"]):
            raise LLMRouterUpdaterParseError("proposed_value must be null or string")
        if not isinstance(action["reason"], str) or not action["reason"]:
            raise LLMRouterUpdaterParseError("reason must be a non-empty string")
        for field in ("supporting_memory_example_ids", "supporting_evidence_ids"):
            items = action[field]
            if not isinstance(items, list) or any(
                    not isinstance(item, str) or not item for item in items) or \
                    len(items) != len(set(items)):
                raise LLMRouterUpdaterParseError(f"{field} must be a unique string array")
        confidence = action["confidence"]
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or \
                not 0.0 <= float(confidence) <= 1.0:
            raise LLMRouterUpdaterParseError("confidence must be in [0,1]")
        output.append(action)
    return tuple(output)


_LEAKAGE_PATTERNS = (
    r"\bquestion\b", r"\banswer\b", r"\bgold(?:en)? label\b",
    r"\bground[ -]?truth\b", r"\bprediction\b", r"\breasoning\b",
    r"\bquestion[-_ ]?id\b", r"\bvideo[-_ ]?id\b", r"\bsegment[-_ ]?id\b",
)


def _leaks_downstream_data(value: str) -> bool:
    lowered = value.casefold()
    return any(re.search(pattern, lowered) for pattern in _LEAKAGE_PATTERNS)


class MemoryConditionedLLMRouterUpdater:
    policy_version = "memory_conditioned_llm_router_updater_v3_strict_retry"

    def __init__(
        self, *, response_provider: Callable[[str, str], str],
        bounds: RouterUpdaterBounds | None = None,
        system_prompt_path: str = PROMPT_PATH,
    ) -> None:
        self.response_provider = response_provider
        self.bounds = bounds or RouterUpdaterBounds()
        self.system_prompt_path = os.path.abspath(system_prompt_path)
        with open(self.system_prompt_path) as handle:
            self.system_prompt = handle.read()
        self.configuration_identity = {
            "policy_version": self.policy_version,
            "request_schema_version": ROUTER_UPDATER_REQUEST_SCHEMA_VERSION,
            "response_schema_version": ROUTER_UPDATER_RESPONSE_SCHEMA_VERSION,
            "plan_schema_version": ROUTER_UPDATER_PLAN_SCHEMA_VERSION,
            "validation_schema_version": ROUTER_VALIDATION_SCHEMA_VERSION,
            "structured_policy_schema_version":
                STRUCTURED_ROUTER_POLICY_SCHEMA_VERSION,
            "rendered_prompt_schema_version": RENDERED_ROUTER_PROMPT_SCHEMA_VERSION,
            "renderer_version": ROUTER_PROMPT_RENDERER_VERSION,
            "system_prompt_version": SYSTEM_PROMPT_VERSION,
            "system_prompt_sha256": sha256_text(self.system_prompt),
            "execution_policy_version": ROUTER_EXECUTION_POLICY_VERSION,
            "bounds": as_json_dict(self.bounds),
            "provider": _provider_identity(response_provider),
        }

    def run(
        self, *, iteration_id: str, parent_router_policy: RouterPolicySnapshot,
        candidate_codebook_path: str, property_id_mapping_path: str,
        property_memory_snapshot_path: str, codebook_applied_plan_path: str,
        codebook_validation_report_path: str, output_dir: str,
    ) -> Mapping[str, Any]:
        output_dir = os.path.abspath(output_dir)
        manifest_path = os.path.join(output_dir, "manifest.json")
        wrapper = _read_json(candidate_codebook_path)
        if wrapper.get("schema_version") != "memory_candidate_codebook_v1":
            raise LLMRouterUpdaterConflictError("incompatible candidate-codebook schema")
        candidate_bank = prompt_bank_from_json(wrapper["candidate_bank"])
        if wrapper.get("candidate_codebook_hash") != sha256_json(
                as_json_dict(candidate_bank)):
            raise LLMRouterUpdaterConflictError("candidate-codebook hash mismatch")
        mapping_artifact = _read_json(property_id_mapping_path)
        if mapping_artifact.get("schema_version") != "property_id_mapping_v1":
            raise LLMRouterUpdaterConflictError("incompatible property-ID mapping schema")
        mapping = mapping_artifact.get("old_to_new_property_ids") or {}
        promotions = mapping_artifact.get("candidate_promotions") or {}
        candidate_entry_ids = {entry.prompt_id for entry in candidate_bank.entries}
        active_candidate_ids = {entry.prompt_id for entry in candidate_bank.entries
                                if entry.status == "active"}
        if not isinstance(mapping, dict) or not mapping or any(
                not isinstance(key, str) or
                (value is not None and not isinstance(value, str))
                for key, value in mapping.items()):
            raise LLMRouterUpdaterConflictError("invalid complete property-ID mapping")
        if set(mapping) - candidate_entry_ids or \
                {value for value in mapping.values() if value is not None} - \
                active_candidate_ids or set(promotions.values()) - active_candidate_ids:
            raise LLMRouterUpdaterConflictError("stale property ID in mapping")
        reached = {value for value in mapping.values() if value is not None} | \
            set(promotions.values())
        if reached != active_candidate_ids:
            raise LLMRouterUpdaterConflictError(
                "candidate codebook/property-ID mapping is incomplete")
        memory = _read_json(property_memory_snapshot_path)
        if memory.get("schema_version") != "property_memory_v1":
            raise LLMRouterUpdaterConflictError("incompatible property-memory schema")
        codebook_plan = _read_json(codebook_applied_plan_path)
        codebook_validation = _read_json(codebook_validation_report_path)
        if parent_router_policy.configuration.get("structured_router_policy"):
            parent_structured = as_json_dict(
                parent_router_policy.configuration["structured_router_policy"])
            if not {row["property_id"] for row in parent_structured["properties"]} \
                    <= set(mapping):
                raise LLMRouterUpdaterConflictError(
                    "property-ID mapping omits parent structured-policy IDs")
            parent_prompt = render_router_prompt(parent_structured)
            if parent_router_policy.configuration.get(
                    "rendered_router_prompt") != parent_prompt or \
                    parent_router_policy.configuration.get(
                        "rendered_router_prompt_hash") != sha256_text(parent_prompt):
                raise LLMRouterUpdaterConflictError(
                    "parent rendered-router prompt hash mismatch")
        else:
            # Legacy parents have no editable guidance, so bootstrapping from the
            # surviving old-ID rows is lossless. The complete mapping is applied
            # immediately below before candidate validation.
            parent_bank = prompt_bank_from_json({
                **wrapper["candidate_bank"],
                "bank_version": wrapper["input_bank_version"],
                "parent_bank_version": None,
                "entries": [{**entry, "status": "active"}
                            for entry in wrapper["candidate_bank"]["entries"]
                            if entry["prompt_id"] in mapping and (
                                entry["status"] == "active" or
                                mapping[entry["prompt_id"]] != entry["prompt_id"])],
                "default_entry_ids": [item for item in
                                      wrapper["candidate_bank"].get("default_entry_ids", [])
                                      if item in mapping and mapping[item] == item],
            })
            parent_structured = bootstrap_structured_router_policy(
                parent_bank, parent_router_policy)
        remapped = remap_structured_router_policy(
            parent_structured, mapping, candidate_bank)
        evidence, examples = self._routing_evidence(
            memory, mapping_artifact=mapping_artifact,
            codebook_plan=codebook_plan)
        request = self._build_request(
            iteration_id=iteration_id, candidate_bank=candidate_bank,
            mapping_artifact=mapping_artifact, parent_structured=parent_structured,
            remapped_structured=remapped, evidence=evidence,
            codebook_plan=codebook_plan)
        identity = {
            "configuration": self.configuration_identity,
            "iteration_id": iteration_id,
            "parent_router": as_json_dict(parent_router_policy),
            "inputs": {path: _file_hash(path) for path in (
                candidate_codebook_path, property_id_mapping_path,
                property_memory_snapshot_path, codebook_applied_plan_path,
                codebook_validation_report_path)},
            "request": request,
        }
        fingerprint = sha256_json(identity)
        if os.path.exists(manifest_path):
            manifest = _read_json(manifest_path)
            if manifest.get("schema_version") != ROUTER_UPDATER_MANIFEST_SCHEMA_VERSION or \
                    manifest.get("status") != "completed" or \
                    manifest.get("input_fingerprint") != fingerprint:
                raise LLMRouterUpdaterConflictError(
                    "completed router updater uses incompatible semantics")
            self._validate_resume(manifest, candidate_bank)
            return {**manifest, "resumed": True}
        identity_path = os.path.join(output_dir, "input_identity.json")
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            if not os.path.exists(identity_path):
                raise LLMRouterUpdaterConflictError(
                    "incomplete router-updater directory predates strict retry semantics")
            existing_identity = _read_json(identity_path)
            if existing_identity.get("schema_version") != \
                    ROUTER_INPUT_IDENTITY_SCHEMA_VERSION or \
                    existing_identity.get("input_fingerprint") != fingerprint:
                raise LLMRouterUpdaterConflictError(
                    "incomplete router-updater input identity mismatch")
        identity_path = _write_immutable(identity_path, {
            "schema_version": ROUTER_INPUT_IDENTITY_SCHEMA_VERSION,
            "input_fingerprint": fingerprint,
            "identity": identity,
        })

        parent_path = _write_immutable(
            os.path.join(output_dir, "parent_structured_router_policy.json"),
            parent_structured)
        prompt_path = _write_text_immutable(
            os.path.join(output_dir, "system_prompt.txt"), self.system_prompt)
        request_path = _write_immutable(os.path.join(output_dir, "request.json"), request)
        if evidence:
            execution_mode = "llm_provider"
            provider_called = True
            deterministic_reason = None
            raw, actions, attempts_path = self._load_or_generate_plan(
                output_dir=output_dir, request=request)
        else:
            execution_mode = "deterministic_empty_plan"
            provider_called = False
            deterministic_reason = "no_routing_evidence"
            raw = dumps_canonical({
                "schema_version": ROUTER_UPDATER_RESPONSE_SCHEMA_VERSION,
                "actions": (),
            })
            actions = parse_router_updater_plan(
                raw, max_actions=self.bounds.max_actions)
            attempts_path = _write_immutable(
                os.path.join(output_dir, "response_attempts.json"), {
                    "schema_version": ROUTER_ATTEMPTS_SCHEMA_VERSION,
                    "max_parse_attempts": self.bounds.max_parse_attempts,
                    "accepted_attempt_index": 0,
                    "attempts": [],
                    "deterministic_reason": deterministic_reason,
                })
        execution_path = _write_immutable(
            os.path.join(output_dir, "execution.json"), {
                "schema_version": ROUTER_EXECUTION_SCHEMA_VERSION,
                "execution_policy_version": ROUTER_EXECUTION_POLICY_VERSION,
                "mode": execution_mode,
                "provider_called": provider_called,
                "routing_evidence_count": len(evidence),
                "deterministic_reason": deterministic_reason,
            })
        raw_path = _write_text_immutable(os.path.join(output_dir, "raw_response.txt"), raw)
        parsed_path = _write_immutable(os.path.join(output_dir, "parsed_plan.json"), {
            "schema_version": ROUTER_UPDATER_PLAN_SCHEMA_VERSION,
            "response_schema_version": ROUTER_UPDATER_RESPONSE_SCHEMA_VERSION,
            "actions": actions})
        accepted, rejected = self._validate_actions(
            actions, candidate_bank=candidate_bank, examples=examples,
            evidence=evidence)
        validation_path = _write_immutable(
            os.path.join(output_dir, "validation_report.json"), {
                "schema_version": ROUTER_VALIDATION_SCHEMA_VERSION,
                "accepted_action_ids": tuple(item["action_id"] for item in accepted),
                "rejected_actions": rejected,
                "execution_mode": execution_mode,
                "fixed_protocol_unchanged": True})
        rejected_path = _write_immutable(
            os.path.join(output_dir, "rejected_actions.json"), {
                "schema_version": ROUTER_VALIDATION_SCHEMA_VERSION,
                "actions": rejected})
        updated = self._apply_actions(remapped, accepted)
        validate_structured_router_policy(updated, candidate_bank)
        applied_path = _write_immutable(os.path.join(output_dir, "applied_plan.json"), {
            "schema_version": ROUTER_APPLIED_PLAN_SCHEMA_VERSION,
            "actions": accepted,
            "deterministic_id_remapping": mapping,
            "retired_guidance_removed": tuple(
                key for key, value in mapping.items() if value is None)})
        structured_path = _write_immutable(
            os.path.join(output_dir, "structured_router_policy.json"), updated)
        rendered = rendered_prompt_artifact(updated)
        rendered_path = _write_immutable(
            os.path.join(output_dir, "rendered_router_prompt.json"), rendered)
        prompt_text_path = _write_text_immutable(
            os.path.join(output_dir, "rendered_router_prompt.txt"), rendered["prompt"])
        parent_prompt = rendered_prompt_artifact(parent_structured)["prompt"]
        prompt_diff = "".join(difflib.unified_diff(
            parent_prompt.splitlines(True), rendered["prompt"].splitlines(True),
            fromfile="parent_router_prompt", tofile="candidate_router_prompt"))
        diff_path = _write_text_immutable(
            os.path.join(output_dir, "router_prompt.diff"), prompt_diff)
        candidate_router = install_structured_router_policy(
            parent_router_policy, candidate_bank, updated,
            lineage={
                "iteration_id": iteration_id,
                "router_updater_policy_version": self.policy_version,
                "old_to_new_property_ids": mapping,
                "rendered_prompt_hash": rendered["prompt_sha256"],
            })
        errors = validate_router_against_bank(candidate_router, candidate_bank)
        if errors:
            raise LLMRouterUpdaterError("; ".join(errors))
        validate_rendered_prompt_configuration(candidate_router, candidate_bank)
        router_path = _write_immutable(
            os.path.join(output_dir, "candidate_router_policy.json"), candidate_router)
        artifacts = {
            "parent_structured_router_policy": parent_path,
            "input_identity": identity_path,
            "system_prompt": prompt_path, "request": request_path,
            "response_attempts": attempts_path,
            "execution": execution_path,
            "raw_response": raw_path, "parsed_plan": parsed_path,
            "validation_report": validation_path,
            "rejected_actions": rejected_path, "applied_plan": applied_path,
            "structured_router_policy": structured_path,
            "rendered_router_prompt": rendered_path,
            "rendered_router_prompt_text": prompt_text_path,
            "prompt_diff": diff_path, "candidate_router_policy": router_path,
        }
        manifest = {
            "schema_version": ROUTER_UPDATER_MANIFEST_SCHEMA_VERSION,
            "status": "completed", "iteration_id": iteration_id,
            "input_fingerprint": fingerprint,
            "system_prompt_version": SYSTEM_PROMPT_VERSION,
            "structured_policy_version": STRUCTURED_ROUTER_POLICY_SCHEMA_VERSION,
            "rendered_prompt_hash": rendered["prompt_sha256"],
            "candidate_router_version": candidate_router.router_version,
            "execution_mode": execution_mode,
            "provider_called": provider_called,
            "deterministic_reason": deterministic_reason,
            "artifacts": artifacts,
            "artifact_hashes": {name: _file_hash(path)
                                for name, path in artifacts.items()},
            "accepted_action_count": len(accepted),
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
        last_error: LLMRouterUpdaterParseError | None = None
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
                    raise LLMRouterUpdaterError(
                        "router updater provider must return text")
                _write_text_immutable(raw_path, raw)
            try:
                actions = parse_router_updater_plan(
                    raw, max_actions=self.bounds.max_actions)
            except LLMRouterUpdaterParseError as exc:
                last_error = exc
                previous_error = str(exc)
                status = "rejected"
                error = {"type": type(exc).__name__, "message": str(exc)}
            else:
                status, error = "accepted", None
            result_path = _write_immutable(
                os.path.join(attempt_dir, "result.json"), {
                    "schema_version": ROUTER_ATTEMPTS_SCHEMA_VERSION,
                    "attempt_index": attempt_index, "status": status,
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
                        "schema_version": ROUTER_ATTEMPTS_SCHEMA_VERSION,
                        "max_parse_attempts": self.bounds.max_parse_attempts,
                        "accepted_attempt_index": attempt_index,
                        "attempts": attempts,
                    })
                return raw, actions, attempts_path
        assert last_error is not None
        _write_immutable(os.path.join(output_dir, "response_attempts.json"), {
            "schema_version": ROUTER_ATTEMPTS_SCHEMA_VERSION,
            "max_parse_attempts": self.bounds.max_parse_attempts,
            "accepted_attempt_index": None, "attempts": attempts,
        })
        raise last_error

    @staticmethod
    def _routing_evidence(
        memory: Mapping[str, Any], *, mapping_artifact: Mapping[str, Any],
        codebook_plan: Mapping[str, Any],
    ) -> tuple[
            tuple[dict[str, Any], ...], dict[str, dict[str, Any]]]:
        rows, by_id = [], {}
        for prop in memory.get("properties") or ():
            for polarity, values in (prop.get("routing_examples") or {}).items():
                for value in values or ():
                    example = as_json_dict(value)
                    example_id = str(example.get("example_id") or "")
                    if not example_id:
                        continue
                    row = {"evidence_id": example_id,
                           "property_id": prop["property_id"],
                           "polarity": polarity, "example": example}
                    rows.append(row)
                    by_id[example_id] = row
        candidate_targets = dict(mapping_artifact.get("candidate_promotions") or {})
        for action in codebook_plan.get("actions") or ():
            name = action.get("action")
            if name == "revise":
                target = (action.get("target_property_ids") or [None])[0]
            elif name == "merge":
                target = action.get("proposed_property_id")
            else:
                target = None
            if target:
                for candidate_id in action.get("candidate_ids") or ():
                    candidate_targets[candidate_id] = target
        for candidate in memory.get("candidates") or ():
            candidate_id = candidate.get("candidate_property_id")
            target = candidate_targets.get(candidate_id) or \
                candidate.get("promoted_to_property_id")
            if not target:
                continue
            for polarity, field in (("positive", "positive_examples"),
                                    ("negative", "negative_examples")):
                for value in candidate.get(field) or ():
                    example = as_json_dict(value)
                    example_id = str(example.get("example_id") or "")
                    if not example_id or example_id in by_id:
                        continue
                    row = {
                        "evidence_id": example_id, "property_id": target,
                        "polarity": polarity, "example": example,
                        "source_kind": "candidate_intervention_effect",
                    }
                    rows.append(row)
                    by_id[example_id] = row
        return tuple(rows), by_id

    def _build_request(
        self, *, iteration_id: str, candidate_bank: PromptBankSnapshot,
        mapping_artifact: Mapping[str, Any], parent_structured: Mapping[str, Any],
        remapped_structured: Mapping[str, Any],
        evidence: tuple[Mapping[str, Any], ...], codebook_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        request = {
            "schema_version": ROUTER_UPDATER_REQUEST_SCHEMA_VERSION,
            "system_prompt_version": SYSTEM_PROMPT_VERSION,
            "iteration_id": iteration_id,
            "validated_candidate_codebook": tuple({
                "property_id": entry.prompt_id, "property_text": entry.prompt_text,
                "status": entry.status,
            } for entry in candidate_bank.entries if entry.status == "active"),
            "property_id_mapping": mapping_artifact,
            "current_structured_router_policy": parent_structured,
            "remapped_structured_router_policy": remapped_structured,
            "routing_memory_examples": evidence,
            "current_compact_routing_evidence": evidence,
            "validated_codebook_actions": tuple(codebook_plan.get("actions") or ()),
            "fixed_protocol": remapped_structured["protocol"],
            "output_schema": {
                "schema_version": ROUTER_UPDATER_RESPONSE_SCHEMA_VERSION,
                "actions": [{
                    "action_id": "unique string",
                    "target_property_id": "active candidate-codebook property ID",
                    "action_type": " | ".join(ACTION_TYPES),
                    "proposed_value": "guidance/example string or null",
                    "reason": "concise reason",
                    "supporting_memory_example_ids": ["existing example_id"],
                    "supporting_evidence_ids": ["existing evidence_id"],
                    "confidence": 0.0,
                }],
            },
        }
        if len(dumps_canonical(request)) > self.bounds.max_request_chars:
            raise LLMRouterUpdaterError("bounded router-updater request exceeds limit")
        return request

    def _validate_actions(
        self, actions: tuple[dict[str, Any], ...], *,
        candidate_bank: PromptBankSnapshot,
        examples: Mapping[str, Mapping[str, Any]],
        evidence: tuple[Mapping[str, Any], ...],
    ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        active = {entry.prompt_id for entry in candidate_bank.entries
                  if entry.status == "active"}
        evidence_ids = {row["evidence_id"] for row in evidence}
        accepted, rejected, claimed = [], [], set()
        example_polarities: dict[tuple[str, str], str] = {}
        for action in actions:
            reasons = []
            action_type = action["action_type"]
            target = action["target_property_id"]
            value = action["proposed_value"]
            memory_ids = set(action["supporting_memory_example_ids"])
            cited_evidence = set(action["supporting_evidence_ids"])
            if target not in active:
                reasons.append("target property is stale or absent from candidate codebook")
            if memory_ids - set(examples):
                reasons.append("references unknown routing-memory example")
            if cited_evidence - evidence_ids:
                reasons.append("references unknown routing evidence")
            if action_type in ("preserve", "no_op"):
                if value is not None:
                    reasons.append(f"{action_type} requires null proposed_value")
            elif not value:
                reasons.append("mutating router action requires proposed_value")
            if value:
                limit = (self.bounds.max_example_chars if "example" in action_type
                         else self.bounds.max_guidance_chars)
                if len(value) > limit:
                    reasons.append("proposed routing text exceeds configured bound")
                if _leaks_downstream_data(value):
                    reasons.append("routing text leaks downstream QA information")
            required_polarity = None
            if action_type in ("set_selection_guidance", "add_positive_example"):
                required_polarity = "positive"
            elif action_type in ("set_avoidance_guidance", "add_negative_example"):
                required_polarity = "negative"
            if required_polarity:
                if not memory_ids and not cited_evidence:
                    reasons.append("router mutation lacks demonstrated effect support")
                cited = tuple(examples[item] for item in memory_ids if item in examples)
                if not cited or any(row["polarity"] != required_polarity or
                                    row["property_id"] != target for row in cited):
                    reasons.append(
                        f"router action lacks supported {required_polarity} effect")
            conflict_key = (target, action_type)
            if action_type.startswith("set_") and conflict_key in claimed:
                reasons.append("router actions conflict on one guidance field")
            if action_type in ("add_positive_example", "add_negative_example") and value:
                polarity = "positive" if action_type == "add_positive_example" \
                    else "negative"
                example_key = (target, value.casefold().strip())
                if example_key in example_polarities and \
                        example_polarities[example_key] != polarity:
                    reasons.append("same routing example has conflicting polarity")
            if reasons:
                rejected.append({"action_id": action["action_id"],
                                 "action_type": action_type,
                                 "reasons": tuple(reasons)})
                continue
            accepted.append(action)
            if action_type.startswith("set_"):
                claimed.add(conflict_key)
            if action_type in ("add_positive_example", "add_negative_example"):
                example_polarities[(target, value.casefold().strip())] = (
                    "positive" if action_type == "add_positive_example" else "negative")
        return tuple(accepted), tuple(rejected)

    @staticmethod
    def _apply_actions(
        policy: Mapping[str, Any], actions: tuple[Mapping[str, Any], ...],
    ) -> dict[str, Any]:
        value = as_json_dict(policy)
        by_id = {row["property_id"]: row for row in value["properties"]}
        for action in actions:
            row = by_id[action["target_property_id"]]
            kind, proposed = action["action_type"], action["proposed_value"]
            if kind == "set_selection_guidance":
                row["selection_guidance"] = proposed
            elif kind == "set_avoidance_guidance":
                row["avoidance_guidance"] = proposed
            elif kind == "add_positive_example":
                row["positive_examples"] = list(dict.fromkeys((
                    *row["positive_examples"], proposed)))[-POSITIVE_EXAMPLE_LIMIT:]
            elif kind == "add_negative_example":
                row["negative_examples"] = list(dict.fromkeys((
                    *row["negative_examples"], proposed)))[-NEGATIVE_EXAMPLE_LIMIT:]
        return value

    @staticmethod
    def _validate_resume(
        manifest: Mapping[str, Any], candidate_bank: PromptBankSnapshot,
    ) -> None:
        artifacts, hashes = manifest.get("artifacts") or {}, \
            manifest.get("artifact_hashes") or {}
        if set(artifacts) != set(hashes) or any(
                not os.path.exists(path) or _file_hash(path) != hashes[name]
                for name, path in artifacts.items()):
            raise LLMRouterUpdaterConflictError("router-updater artifact mismatch")
        router = router_policy_from_json(_read_json(artifacts["candidate_router_policy"]))
        validate_rendered_prompt_configuration(router, candidate_bank)
        rendered = _read_json(artifacts["rendered_router_prompt"])
        if rendered.get("schema_version") != RENDERED_ROUTER_PROMPT_SCHEMA_VERSION or \
                rendered.get("prompt_sha256") != manifest.get("rendered_prompt_hash"):
            raise LLMRouterUpdaterConflictError("rendered prompt schema/hash mismatch")
