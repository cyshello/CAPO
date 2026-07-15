"""Foundational Phase 4 typed records (PHASE4_PROMPT_ROUTING.md §10, Stage 4.1).

Routing-owned records only: prompt bank, router policy, segment context,
routing decisions, scaffold contract/policy, composition traces, composed
prompts, component-version identifiers, and the typed Phase 4 configuration.
Optimization-owned records (evidence, feedback, proposals, reviews) belong to
`optimization/schemas.py` in a later stage (§8: no duplicated records across
the two schema modules).

Design constraints enforced here:

- Records are frozen dataclasses; mapping fields are wrapped in
  MappingProxyType at construction so neither attributes nor mappings can be
  mutated in place (§21.2/§21.9).
- Hashes and version identifiers are validated at construction so no record
  can exist with an inconsistent prompt_hash or a version string of the wrong
  component kind (§21.3).
- `as_json_dict` / `dumps_canonical` give deterministic serialization
  (sorted keys) for artifact writing and round-trip tests (§21.4).
- Caption-cache identity stays separate from component provenance: the
  composed prompt text/hash alone determines the (later) caption-cache key,
  while ComposedCaptionPrompt separately retains bank/router/scaffold/contract
  versions and the selected prompt IDs. Different component versions may share
  a cache entry when they compose byte-identical prompt text; their provenance
  stays distinguishable in these records.

No persistence, routing logic, composition, captioning, or model calls here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

from surrogate_rollout.schemas import sha256_text

# --------------------------------------------------------------------------- #
#                        component-version identifiers                         #
# --------------------------------------------------------------------------- #
# Format: "<kind>_v<NNNN>" with an optional content suffix "_<hex8..16>",
# e.g. "bank_v0001", "scaffold_v0002_ab12cd34". The numeric part orders
# versions; the optional suffix ties a version to content without replacing
# the lineage number.

COMPONENT_KINDS = ("bank", "router", "scaffold", "contract")

_VERSION_RE = {
    kind: re.compile(rf"^{kind}_v\d{{4,}}(_[0-9a-f]{{8,16}})?$")
    for kind in COMPONENT_KINDS
}

# Existing repo clip key, e.g. "0_10" (PHASE4 segment == Phase 0-3 clip).
_SEGMENT_ID_RE = re.compile(r"^\d+_\d+$")


def make_component_version(kind: str, number: int, content_hash: str | None = None) -> str:
    """Deterministic version identifier for one component snapshot."""
    if kind not in COMPONENT_KINDS:
        raise ValueError(f"unknown component kind: {kind!r} (want one of {COMPONENT_KINDS})")
    if number < 0:
        raise ValueError(f"component version number must be >= 0, got {number}")
    suffix = ""
    if content_hash:
        h = content_hash[:16].lower()
        if not re.fullmatch(r"[0-9a-f]{8,16}", h):
            raise ValueError(f"content_hash must be >=8 hex chars, got {content_hash!r}")
        suffix = f"_{h[:8]}"
    return f"{kind}_v{number:04d}{suffix}"


def validate_component_version(kind: str, version: str | None, *,
                               field_name: str, allow_none: bool = False) -> None:
    if version is None:
        if allow_none:
            return
        raise ValueError(f"{field_name} must not be None")
    if kind not in COMPONENT_KINDS:
        raise ValueError(f"unknown component kind: {kind!r}")
    if not isinstance(version, str) or not _VERSION_RE[kind].match(version):
        raise ValueError(
            f"{field_name}={version!r} is not a valid {kind} version "
            f"(expected e.g. {make_component_version(kind, 1)!r})"
        )


# --------------------------------------------------------------------------- #
#                              internal helpers                                #
# --------------------------------------------------------------------------- #
def _freeze_mapping(record: Any, name: str) -> None:
    value = getattr(record, name)
    if isinstance(value, MappingProxyType):
        return
    if not isinstance(value, Mapping):
        raise TypeError(f"{type(record).__name__}.{name} must be a Mapping, "
                        f"got {type(value).__name__}")
    object.__setattr__(record, name, MappingProxyType(dict(value)))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_str_tuple(record: Any, name: str) -> None:
    value = getattr(record, name)
    if not isinstance(value, tuple) or any(not isinstance(v, str) for v in value):
        raise TypeError(f"{type(record).__name__}.{name} must be a tuple of str")


def _require_unique(values: tuple[str, ...], what: str) -> None:
    _require(len(values) == len(set(values)), f"duplicate {what}: {sorted(values)}")


def _validate_confidence(value: float | None, field_name: str, *,
                         allow_none: bool) -> None:
    if value is None:
        _require(allow_none, f"{field_name} must not be None")
        return
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and 0.0 <= float(value) <= 1.0,
             f"{field_name} must be in [0, 1], got {value!r}")


# --------------------------------------------------------------------------- #
#                              serialization                                   #
# --------------------------------------------------------------------------- #
def as_json_dict(value: Any) -> Any:
    """Recursively convert a schema record to plain JSON types. Rejects any
    value outside the schema vocabulary — no raw objects leak into artifacts."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: as_json_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): as_json_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json_dict(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"not schema-serializable: {type(value).__name__}")


def dumps_canonical(value: Any) -> str:
    """Deterministic JSON text for a record (sorted keys, stable separators)."""
    return json.dumps(as_json_dict(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


# --------------------------------------------------------------------------- #
#                             prompt-bank records                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PromptEntry:
    """One reusable, composable captioning instruction (§10.1) — narrower than
    the full captioning contract, never a complete standalone prompt."""

    prompt_id: str
    prompt_text: str
    prompt_hash: str          # sha256 of prompt_text (entry text, NOT cache key)
    name: str
    description: str
    tags: tuple[str, ...] = ()
    applicability_traits: Mapping[str, Any] = field(default_factory=dict)
    conflicts_with: tuple[str, ...] = ()
    status: Literal["active", "retired"] = "active"
    parent_prompt_ids: tuple[str, ...] = ()
    created_by: str = ""
    created_at: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(bool(self.prompt_id), "PromptEntry.prompt_id must be non-empty")
        _require(bool(self.prompt_text), "PromptEntry.prompt_text must be non-empty")
        _require(self.prompt_hash == sha256_text(self.prompt_text),
                 f"PromptEntry.prompt_hash does not match sha256(prompt_text) "
                 f"for {self.prompt_id!r}")
        _require(self.status in ("active", "retired"),
                 f"PromptEntry.status must be 'active' or 'retired', got {self.status!r}")
        _require_str_tuple(self, "tags")
        _require_str_tuple(self, "conflicts_with")
        _require_str_tuple(self, "parent_prompt_ids")
        _require(self.prompt_id not in self.conflicts_with,
                 f"PromptEntry {self.prompt_id!r} cannot conflict with itself")
        _freeze_mapping(self, "applicability_traits")
        _freeze_mapping(self, "provenance")


@dataclass(frozen=True)
class PromptBankSnapshot:
    """Versioned, immutable snapshot of prompt entries (§10.1)."""

    bank_version: str
    entries: tuple[PromptEntry, ...]
    parent_bank_version: str | None = None
    default_entry_ids: tuple[str, ...] = ()
    max_selected_entries: int = 4
    created_at: str = ""
    created_by: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_component_version("bank", self.bank_version,
                                   field_name="PromptBankSnapshot.bank_version")
        validate_component_version("bank", self.parent_bank_version,
                                   field_name="PromptBankSnapshot.parent_bank_version",
                                   allow_none=True)
        _require(all(isinstance(e, PromptEntry) for e in self.entries),
                 "PromptBankSnapshot.entries must be PromptEntry records")
        ids = tuple(e.prompt_id for e in self.entries)
        _require_unique(ids, "PromptBankSnapshot entry prompt_ids")
        _require(self.max_selected_entries >= 1,
                 "PromptBankSnapshot.max_selected_entries must be >= 1")
        _require_str_tuple(self, "default_entry_ids")
        active = {e.prompt_id for e in self.entries if e.status == "active"}
        unknown = set(self.default_entry_ids) - active
        _require(not unknown,
                 f"PromptBankSnapshot.default_entry_ids reference non-active or "
                 f"unknown entries: {sorted(unknown)}")
        _freeze_mapping(self, "provenance")

    def entry(self, prompt_id: str) -> PromptEntry:
        for e in self.entries:
            if e.prompt_id == prompt_id:
                return e
        raise KeyError(prompt_id)


# --------------------------------------------------------------------------- #
#                               router records                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RoutingRule:
    rule_id: str
    priority: int
    conditions: Mapping[str, Any]
    target_prompt_ids: tuple[str, ...]
    enabled: bool = True
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(bool(self.rule_id), "RoutingRule.rule_id must be non-empty")
        _require_str_tuple(self, "target_prompt_ids")
        _require(len(self.target_prompt_ids) >= 1,
                 f"RoutingRule {self.rule_id!r} must target at least one prompt entry")
        _require_unique(self.target_prompt_ids,
                        f"RoutingRule {self.rule_id!r} target_prompt_ids")
        _freeze_mapping(self, "conditions")
        _freeze_mapping(self, "provenance")


@dataclass(frozen=True)
class RouterPolicySnapshot:
    router_version: str
    policy_type: str
    rules: tuple[RoutingRule, ...] = ()
    parent_router_version: str | None = None
    fallback_prompt_ids: tuple[str, ...] = ()
    max_selected_entries: int = 4
    configuration: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_component_version("router", self.router_version,
                                   field_name="RouterPolicySnapshot.router_version")
        validate_component_version("router", self.parent_router_version,
                                   field_name="RouterPolicySnapshot.parent_router_version",
                                   allow_none=True)
        _require(bool(self.policy_type), "RouterPolicySnapshot.policy_type must be non-empty")
        _require(all(isinstance(r, RoutingRule) for r in self.rules),
                 "RouterPolicySnapshot.rules must be RoutingRule records")
        _require_unique(tuple(r.rule_id for r in self.rules),
                        "RouterPolicySnapshot rule_ids")
        _require(self.max_selected_entries >= 1,
                 "RouterPolicySnapshot.max_selected_entries must be >= 1")
        _require_str_tuple(self, "fallback_prompt_ids")
        _freeze_mapping(self, "configuration")
        _freeze_mapping(self, "provenance")


@dataclass(frozen=True)
class SegmentContext:
    """Routing input for one segment. `segment_id` IS the existing repo clip
    key "{start}_{end}" (Stage 4.0 §5) — Phase 4 introduces no new
    segmentation system."""

    video_id: str
    segment_id: str
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    segment_features: Mapping[str, Any] = field(default_factory=dict)
    history_summary: str | None = None
    question: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(bool(self.video_id), "SegmentContext.video_id must be non-empty")
        _require(bool(self.segment_id) and bool(_SEGMENT_ID_RE.match(self.segment_id)),
                 f"SegmentContext.segment_id must be a clip key like '0_10', "
                 f"got {self.segment_id!r}")
        if self.timestamp_start is not None and self.timestamp_end is not None:
            _require(self.timestamp_start < self.timestamp_end,
                     "SegmentContext timestamps must satisfy start < end")
        _freeze_mapping(self, "segment_features")
        _freeze_mapping(self, "metadata")


@dataclass(frozen=True)
class RoutingDecision:
    """Structured multi-entry selection for one segment (§10.2). Never
    unstructured prose; supports zero or more selected entries."""

    video_id: str
    segment_id: str
    bank_version: str
    router_version: str
    selected_prompt_ids: tuple[str, ...]
    prompt_scores: Mapping[str, float] = field(default_factory=dict)
    matched_rule_ids: tuple[str, ...] = ()
    selection_order: tuple[str, ...] = ()
    confidence: float | None = None
    used_fallback: bool = False
    fallback_reason: str | None = None
    decision_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(bool(self.video_id), "RoutingDecision.video_id must be non-empty")
        _require(bool(self.segment_id), "RoutingDecision.segment_id must be non-empty")
        validate_component_version("bank", self.bank_version,
                                   field_name="RoutingDecision.bank_version")
        validate_component_version("router", self.router_version,
                                   field_name="RoutingDecision.router_version")
        _require_str_tuple(self, "selected_prompt_ids")
        _require_unique(self.selected_prompt_ids, "RoutingDecision.selected_prompt_ids")
        _require_str_tuple(self, "matched_rule_ids")
        _require_str_tuple(self, "selection_order")
        # selection_order, when given, is exactly a permutation of the selection
        if self.selection_order:
            _require(sorted(self.selection_order) == sorted(self.selected_prompt_ids),
                     "RoutingDecision.selection_order must be a permutation of "
                     "selected_prompt_ids")
        _validate_confidence(self.confidence, "RoutingDecision.confidence",
                             allow_none=True)
        if self.used_fallback:
            _require(bool(self.fallback_reason),
                     "RoutingDecision.used_fallback requires fallback_reason")
        else:
            _require(self.fallback_reason is None,
                     "RoutingDecision.fallback_reason set but used_fallback is False")
        _freeze_mapping(self, "prompt_scores")
        _freeze_mapping(self, "decision_payload")


# --------------------------------------------------------------------------- #
#                              scaffold records                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScaffoldContract:
    """Immutable hard requirements for a valid captioning prompt (§12.1).
    Never an optimization target."""

    contract_version: str
    required_placeholders: tuple[str, ...]
    required_sections: tuple[str, ...] = ()
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    forbidden_patterns: tuple[str, ...] = ()
    max_prompt_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_component_version("contract", self.contract_version,
                                   field_name="ScaffoldContract.contract_version")
        _require_str_tuple(self, "required_placeholders")
        _require_str_tuple(self, "required_sections")
        _require_str_tuple(self, "forbidden_patterns")
        _require_unique(self.required_placeholders,
                        "ScaffoldContract.required_placeholders")
        if self.max_prompt_tokens is not None:
            _require(self.max_prompt_tokens >= 1,
                     "ScaffoldContract.max_prompt_tokens must be >= 1 or None")
        _freeze_mapping(self, "output_schema")
        _freeze_mapping(self, "metadata")


@dataclass(frozen=True)
class ScaffoldPolicySnapshot:
    """Versioned SOFT composition behavior (§12.2). Distinct from the hard
    contract; the only scaffold state optimization may ever touch."""

    scaffold_version: str
    policy_type: str
    composition_instruction: str = ""
    ordering_policy: str = ""
    conflict_resolution_rules: tuple[str, ...] = ()
    compression_policy: str = ""
    parent_scaffold_version: str | None = None
    configuration: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_component_version("scaffold", self.scaffold_version,
                                   field_name="ScaffoldPolicySnapshot.scaffold_version")
        validate_component_version("scaffold", self.parent_scaffold_version,
                                   field_name="ScaffoldPolicySnapshot.parent_scaffold_version",
                                   allow_none=True)
        _require(bool(self.policy_type),
                 "ScaffoldPolicySnapshot.policy_type must be non-empty")
        _require_str_tuple(self, "conflict_resolution_rules")
        _freeze_mapping(self, "configuration")
        _freeze_mapping(self, "provenance")


@dataclass(frozen=True)
class CompositionTrace:
    """How selected entries became the final prompt (§10.3). preserved and
    omitted partition the selected set — no entry silently disappears."""

    selected_prompt_ids: tuple[str, ...]
    preserved_prompt_ids: tuple[str, ...]
    omitted_prompt_ids: tuple[str, ...] = ()
    conflicts_detected: tuple[str, ...] = ()
    conflict_resolutions: tuple[str, ...] = ()
    ordering_decisions: tuple[str, ...] = ()
    compression_actions: tuple[str, ...] = ()
    raw_policy_output_artifact: str | None = None

    def __post_init__(self) -> None:
        for name in ("selected_prompt_ids", "preserved_prompt_ids",
                     "omitted_prompt_ids", "conflicts_detected",
                     "conflict_resolutions", "ordering_decisions",
                     "compression_actions"):
            _require_str_tuple(self, name)
        _require_unique(self.selected_prompt_ids, "CompositionTrace.selected_prompt_ids")
        _require_unique(self.preserved_prompt_ids, "CompositionTrace.preserved_prompt_ids")
        _require_unique(self.omitted_prompt_ids, "CompositionTrace.omitted_prompt_ids")
        selected = set(self.selected_prompt_ids)
        preserved = set(self.preserved_prompt_ids)
        omitted = set(self.omitted_prompt_ids)
        _require(not (preserved & omitted),
                 "CompositionTrace: an entry cannot be both preserved and omitted")
        _require(preserved | omitted == selected,
                 "CompositionTrace: preserved + omitted must exactly cover "
                 "selected_prompt_ids (every selected entry is traceable)")


@dataclass(frozen=True)
class ComposedCaptionPrompt:
    """Final composed captioning prompt for one segment (§10.3).

    `prompt_hash` = sha256(prompt_text) and is what caption-cache identity
    will later be derived from; the component versions here are provenance
    only and deliberately do NOT enter that hash (Stage 4.1 clarification #5:
    different component versions may share a cache entry when the composed
    text is byte-identical, while remaining distinguishable here)."""

    video_id: str
    segment_id: str
    bank_version: str
    router_version: str
    scaffold_version: str
    contract_version: str
    selected_prompt_ids: tuple[str, ...]
    prompt_text: str
    prompt_hash: str
    composition_trace: CompositionTrace
    validation_errors: tuple[str, ...] = ()
    is_valid: bool = True

    def __post_init__(self) -> None:
        _require(bool(self.video_id), "ComposedCaptionPrompt.video_id must be non-empty")
        _require(bool(self.segment_id), "ComposedCaptionPrompt.segment_id must be non-empty")
        validate_component_version("bank", self.bank_version,
                                   field_name="ComposedCaptionPrompt.bank_version")
        validate_component_version("router", self.router_version,
                                   field_name="ComposedCaptionPrompt.router_version")
        validate_component_version("scaffold", self.scaffold_version,
                                   field_name="ComposedCaptionPrompt.scaffold_version")
        validate_component_version("contract", self.contract_version,
                                   field_name="ComposedCaptionPrompt.contract_version")
        _require_str_tuple(self, "selected_prompt_ids")
        _require_str_tuple(self, "validation_errors")
        _require(isinstance(self.composition_trace, CompositionTrace),
                 "ComposedCaptionPrompt.composition_trace must be a CompositionTrace")
        _require(self.selected_prompt_ids == self.composition_trace.selected_prompt_ids,
                 "ComposedCaptionPrompt.selected_prompt_ids must equal "
                 "composition_trace.selected_prompt_ids")
        _require(self.prompt_hash == sha256_text(self.prompt_text),
                 "ComposedCaptionPrompt.prompt_hash does not match sha256(prompt_text)")
        if self.is_valid:
            _require(not self.validation_errors,
                     "ComposedCaptionPrompt.is_valid=True but validation_errors present")
            _require(bool(self.prompt_text),
                     "ComposedCaptionPrompt.is_valid=True requires non-empty prompt_text")
        else:
            _require(bool(self.validation_errors),
                     "ComposedCaptionPrompt.is_valid=False requires at least one "
                     "validation error (failures are never silent)")


# --------------------------------------------------------------------------- #
#                          Phase 4 typed configuration                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Phase4OptimizationConfig:
    """Optimization switches (§4). `optimize_scaffold` defaults to False and is
    the fixed-vs-optimized scaffold ablation axis (§4.3)."""

    optimize_prompt_bank: bool = True
    optimize_router: bool = True
    optimize_scaffold: bool = False
    max_iterations: int = 1
    max_new_entries_per_iteration: int = 2
    max_router_operations_per_iteration: int = 4
    min_feedback_confidence: float = 0.70

    def __post_init__(self) -> None:
        _require(self.max_iterations >= 1,
                 "Phase4OptimizationConfig.max_iterations must be >= 1")
        _require(self.max_new_entries_per_iteration >= 0,
                 "Phase4OptimizationConfig.max_new_entries_per_iteration must be >= 0")
        _require(self.max_router_operations_per_iteration >= 0,
                 "Phase4OptimizationConfig.max_router_operations_per_iteration must be >= 0")
        _validate_confidence(self.min_feedback_confidence,
                             "Phase4OptimizationConfig.min_feedback_confidence",
                             allow_none=False)


@dataclass(frozen=True)
class Phase4Config:
    """Top-level typed Phase 4 configuration (§17). Dry-run/preview is the
    default; a persistent commit requires the explicit flag (§23). YAML
    loading and persistence belong to later stages — deterministic dict/JSON
    round-trip only here."""

    seed: int = 0
    dry_run: bool = True
    commit: bool = False
    optimization: Phase4OptimizationConfig = field(
        default_factory=Phase4OptimizationConfig)

    def __post_init__(self) -> None:
        _require(isinstance(self.optimization, Phase4OptimizationConfig),
                 "Phase4Config.optimization must be a Phase4OptimizationConfig")
        _require(not (self.dry_run and self.commit),
                 "Phase4Config: dry_run and commit are mutually exclusive")

    @property
    def optimize_scaffold(self) -> bool:
        """Top-level visibility of the ablation switch (§17: never hidden
        inside a scaffold module; copied into every run artifact)."""
        return self.optimization.optimize_scaffold

    def as_json_dict(self) -> dict:
        return as_json_dict(self)

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "Phase4Config":
        opt = data.get("optimization", {})
        return cls(
            seed=data.get("seed", 0),
            dry_run=data.get("dry_run", True),
            commit=data.get("commit", False),
            optimization=Phase4OptimizationConfig(**dict(opt)),
        )
