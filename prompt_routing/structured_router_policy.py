"""Versioned structured guidance and deterministic prompt rendering for Phase 4."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    RouterPolicySnapshot,
    RoutingRule,
    as_json_dict,
    dumps_canonical,
    make_component_version,
)
from surrogate_rollout.schemas import sha256_json, sha256_text


STRUCTURED_ROUTER_POLICY_SCHEMA_VERSION = "structured_router_policy_v1"
ROUTER_PROMPT_RENDERER_VERSION = "history_aware_router_prompt_renderer_v1"
RENDERED_ROUTER_PROMPT_SCHEMA_VERSION = "rendered_router_prompt_v1"
ROUTER_PROTOCOL_VERSION = "history_aware_router_protocol_v1"
POSITIVE_EXAMPLE_LIMIT = 2
NEGATIVE_EXAMPLE_LIMIT = 2

FIXED_ROUTER_PROTOCOL = {
    "version": ROUTER_PROTOCOL_VERSION,
    "query_independent": True,
    "inputs": ["current_segment_frames", "bounded_preceding_caption_history"],
    "active_property_ids_only": True,
    "strict_json_output": True,
    "output_field": "property_ids",
    "fallback": "fail_closed",
    "parser": "strict_one_key_json",
}


class StructuredRouterPolicyError(ValueError):
    pass


def _property_row(entry: Any) -> dict[str, Any]:
    aliases = tuple(entry.provenance.get("aliases") or ())
    return {
        "property_id": entry.prompt_id,
        "selection_guidance": "",
        "avoidance_guidance": "",
        "positive_examples": (),
        "negative_examples": (),
        "aliases": aliases,
        "remapped_from_property_ids": (),
    }


def bootstrap_structured_router_policy(
    prompt_bank: PromptBankSnapshot,
    router_policy: RouterPolicySnapshot,
) -> dict[str, Any]:
    existing = router_policy.configuration.get("structured_router_policy")
    if existing:
        value = as_json_dict(existing)
        validate_structured_router_policy(value, prompt_bank)
        return value
    active = tuple(entry for entry in prompt_bank.entries if entry.status == "active")
    return {
        "schema_version": STRUCTURED_ROUTER_POLICY_SCHEMA_VERSION,
        "protocol": {**FIXED_ROUTER_PROTOCOL,
                     "max_selected_properties": min(
                         router_policy.max_selected_entries,
                         prompt_bank.max_selected_entries,
                         int(router_policy.configuration.get(
                             "max_selected_properties",
                             router_policy.max_selected_entries)))},
        "properties": tuple(_property_row(entry) for entry in active),
    }


def validate_structured_router_policy(
    policy: Mapping[str, Any], prompt_bank: PromptBankSnapshot,
) -> None:
    if policy.get("schema_version") != STRUCTURED_ROUTER_POLICY_SCHEMA_VERSION:
        raise StructuredRouterPolicyError("incompatible structured router-policy schema")
    protocol = policy.get("protocol")
    if not isinstance(protocol, Mapping):
        raise StructuredRouterPolicyError("structured router policy lacks protocol")
    expected = dict(FIXED_ROUTER_PROTOCOL)
    for key, value in expected.items():
        if as_json_dict(protocol.get(key)) != value:
            raise StructuredRouterPolicyError(f"fixed router protocol changed: {key}")
    limit = protocol.get("max_selected_properties")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or \
            limit > prompt_bank.max_selected_entries:
        raise StructuredRouterPolicyError("invalid structured router selection limit")
    rows = policy.get("properties")
    if not isinstance(rows, (list, tuple)):
        raise StructuredRouterPolicyError("structured router properties must be an array")
    active = {entry.prompt_id for entry in prompt_bank.entries if entry.status == "active"}
    ids = []
    required = {
        "property_id", "selection_guidance", "avoidance_guidance",
        "positive_examples", "negative_examples", "aliases",
        "remapped_from_property_ids",
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise StructuredRouterPolicyError("invalid structured property guidance row")
        ids.append(row["property_id"])
        if any(not isinstance(row[field], str)
               for field in ("property_id", "selection_guidance", "avoidance_guidance")):
            raise StructuredRouterPolicyError("guidance fields must be strings")
        for field, bound in (("positive_examples", POSITIVE_EXAMPLE_LIMIT),
                             ("negative_examples", NEGATIVE_EXAMPLE_LIMIT)):
            values = row[field]
            if not isinstance(values, (list, tuple)) or len(values) > bound or any(
                    not isinstance(item, str) or not item for item in values):
                raise StructuredRouterPolicyError(f"invalid bounded {field}")
        for field in ("aliases", "remapped_from_property_ids"):
            values = row[field]
            if not isinstance(values, (list, tuple)) or any(
                    not isinstance(item, str) or not item for item in values):
                raise StructuredRouterPolicyError(f"invalid {field}")
    if len(ids) != len(set(ids)) or set(ids) != active:
        raise StructuredRouterPolicyError("structured policy/candidate codebook ID mismatch")


def remap_structured_router_policy(
    parent: Mapping[str, Any], mapping: Mapping[str, str | None],
    candidate_bank: PromptBankSnapshot,
) -> dict[str, Any]:
    """Apply the complete Checkpoint-2 ID mapping before LLM policy actions."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in parent.get("properties") or ():
        old_id = str(row["property_id"])
        new_id = mapping.get(old_id, old_id)
        if new_id is not None:
            grouped.setdefault(new_id, []).append(row)
    active_entries = tuple(
        entry for entry in candidate_bank.entries if entry.status == "active")
    rows = []
    for entry in active_entries:
        sources = grouped.get(entry.prompt_id, [])
        selection = next((str(row.get("selection_guidance") or "")
                          for row in sources if row.get("selection_guidance")), "")
        avoidance = next((str(row.get("avoidance_guidance") or "")
                          for row in sources if row.get("avoidance_guidance")), "")
        positive = tuple(dict.fromkeys(
            str(item) for row in sources for item in row.get("positive_examples") or ()))
        negative = tuple(dict.fromkeys(
            str(item) for row in sources for item in row.get("negative_examples") or ()))
        source_ids = tuple(dict.fromkeys(
            str(row["property_id"]) for row in sources
            if row["property_id"] != entry.prompt_id))
        aliases = tuple(dict.fromkeys((
            *(str(item) for row in sources for item in row.get("aliases") or ()),
            *(str(item) for item in entry.provenance.get("aliases") or ()),
            *source_ids,
        )))
        rows.append({
            "property_id": entry.prompt_id,
            "selection_guidance": selection,
            "avoidance_guidance": avoidance,
            "positive_examples": positive[:POSITIVE_EXAMPLE_LIMIT],
            "negative_examples": negative[:NEGATIVE_EXAMPLE_LIMIT],
            "aliases": aliases,
            "remapped_from_property_ids": source_ids,
        })
    value = {
        "schema_version": STRUCTURED_ROUTER_POLICY_SCHEMA_VERSION,
        "protocol": as_json_dict(parent["protocol"]),
        "properties": tuple(rows),
    }
    validate_structured_router_policy(value, candidate_bank)
    return value


def render_router_prompt(policy: Mapping[str, Any]) -> str:
    protocol = policy["protocol"]
    guidance = tuple({
        "property_id": row["property_id"],
        "selection_guidance": row["selection_guidance"],
        "avoidance_guidance": row["avoidance_guidance"],
        "positive_examples": tuple(row["positive_examples"]),
        "negative_examples": tuple(row["negative_examples"]),
        "aliases": tuple(row["aliases"]),
    } for row in policy["properties"])
    return (
        "Select zero or more active caption properties for the current video "
        "segment using only its current frames and bounded preceding caption "
        "history. Routing is query-independent. Select only IDs present in "
        "active_codebook and select at most "
        f"{protocol['max_selected_properties']}. Return only strict JSON "
        "matching output_schema; do not use Markdown or add keys. If no "
        "property applies, return an empty property_ids array.\n"
        "Structured per-property routing guidance:\n" + dumps_canonical(guidance)
    )


def rendered_prompt_artifact(policy: Mapping[str, Any]) -> dict[str, Any]:
    prompt = render_router_prompt(policy)
    return {
        "schema_version": RENDERED_ROUTER_PROMPT_SCHEMA_VERSION,
        "renderer_version": ROUTER_PROMPT_RENDERER_VERSION,
        "structured_policy_schema_version": STRUCTURED_ROUTER_POLICY_SCHEMA_VERSION,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
    }


def install_structured_router_policy(
    parent_router: RouterPolicySnapshot,
    candidate_bank: PromptBankSnapshot,
    structured_policy: Mapping[str, Any],
    *, lineage: Mapping[str, Any],
) -> RouterPolicySnapshot:
    validate_structured_router_policy(structured_policy, candidate_bank)
    rendered = rendered_prompt_artifact(structured_policy)
    mapping = lineage.get("old_to_new_property_ids") or {}
    active = {entry.prompt_id for entry in candidate_bank.entries
              if entry.status == "active"}
    rules = []
    for rule in parent_router.rules:
        targets = tuple(dict.fromkeys(
            mapping.get(item, item) for item in rule.target_prompt_ids
            if mapping.get(item, item) in active))
        if targets:
            rules.append(replace(rule, target_prompt_ids=targets))
    fallback = tuple(dict.fromkeys(
        mapping.get(item, item) for item in parent_router.fallback_prompt_ids
        if mapping.get(item, item) in active))
    parent_configuration = as_json_dict(parent_router.configuration)
    # Legacy supervision remains in immutable parent artifacts, but the new
    # policy consumes the structured guidance rendered below rather than
    # continuing a second, unversioned inference-time channel.
    parent_configuration.pop("supervision_examples", None)
    configuration = {
        **parent_configuration,
        "max_selected_properties": structured_policy["protocol"][
            "max_selected_properties"],
        "structured_router_policy": as_json_dict(structured_policy),
        "structured_router_policy_schema_version":
            STRUCTURED_ROUTER_POLICY_SCHEMA_VERSION,
        "rendered_router_prompt": rendered["prompt"],
        "rendered_router_prompt_hash": rendered["prompt_sha256"],
        "rendered_router_prompt_schema_version":
            RENDERED_ROUTER_PROMPT_SCHEMA_VERSION,
        "router_prompt_renderer_version": ROUTER_PROMPT_RENDERER_VERSION,
    }
    payload = {
        "parent": parent_router.router_version,
        "candidate_bank": candidate_bank.bank_version,
        "configuration": configuration,
        "rules": as_json_dict(tuple(rules)),
        "fallback": fallback,
        "lineage": lineage,
    }
    version = make_component_version(
        "router", int(parent_router.router_version.split("_v", 1)[1].split("_", 1)[0]) + 1,
        sha256_json(payload))
    return replace(
        parent_router, router_version=version,
        parent_router_version=parent_router.router_version,
        rules=tuple(rules), fallback_prompt_ids=fallback,
        max_selected_entries=min(parent_router.max_selected_entries,
                                 candidate_bank.max_selected_entries),
        configuration=configuration,
        provenance={**as_json_dict(parent_router.provenance), **as_json_dict(lineage),
                    "provisional": True})


def validate_rendered_prompt_configuration(
    router_policy: RouterPolicySnapshot, prompt_bank: PromptBankSnapshot,
) -> None:
    config = router_policy.configuration
    keys = {
        "structured_router_policy", "structured_router_policy_schema_version",
        "rendered_router_prompt", "rendered_router_prompt_hash",
        "rendered_router_prompt_schema_version", "router_prompt_renderer_version",
    }
    present = keys & set(config)
    if not present:
        return
    if present != keys:
        raise StructuredRouterPolicyError("incomplete rendered router prompt configuration")
    if config["structured_router_policy_schema_version"] != \
            STRUCTURED_ROUTER_POLICY_SCHEMA_VERSION or \
            config["rendered_router_prompt_schema_version"] != \
            RENDERED_ROUTER_PROMPT_SCHEMA_VERSION or \
            config["router_prompt_renderer_version"] != ROUTER_PROMPT_RENDERER_VERSION:
        raise StructuredRouterPolicyError("incompatible rendered router prompt semantics")
    structured = as_json_dict(config["structured_router_policy"])
    validate_structured_router_policy(structured, prompt_bank)
    expected = render_router_prompt(structured)
    if config["rendered_router_prompt"] != expected or \
            config["rendered_router_prompt_hash"] != sha256_text(expected):
        raise StructuredRouterPolicyError("rendered router prompt hash/content mismatch")
