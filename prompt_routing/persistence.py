"""Stage 4.2 component persistence (PHASE4_PROMPT_ROUTING.md Stage 4.2).

Immutable versioned snapshot files per component, with a small `current.json`
pointer manifest instead of a second mutable copy of the snapshot:

    <root>/
        current.json                      # {"component", "version", "updated_at"}
        snapshots/<version>.json          # canonical record JSON, write-once

Invariants:

- An existing version ID is never overwritten with different content
  (VersionConflictError). Re-saving canonically identical content under the
  same version is an idempotent no-op.
- All writes go through a same-directory temporary file + os.replace, so a
  failed snapshot or pointer write leaves the previously active state
  recoverable and unchanged.
- Preview writes go to a caller-supplied run/preview directory and never
  touch committed snapshots or the current pointer (§23: dry-run default,
  commit is explicit).
- Content digest (optional version suffix) = sha256 over the canonical JSON
  of the record WITH ITS OWN VERSION FIELD REMOVED — the version string may
  embed the digest, so including it would be circular. Parent versions stay
  in the digest (lineage is content). Component versions remain provenance
  identifiers, not caption-cache keys: caption-cache identity derives from
  composed prompt text/hash only.

No routing or composition logic here (Stage 4.2 scope).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    PromptEntry,
    RouterPolicySnapshot,
    RoutingRule,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    as_json_dict,
    make_component_version,
    validate_component_version,
)
from surrogate_rollout.prompt_routing.validators import (
    validate_prompt_bank,
    validate_router_policy,
    validate_scaffold_contract,
    validate_scaffold_policy,
)


class SnapshotStoreError(RuntimeError):
    """Persistence failure (missing snapshot, kind mismatch, digest mismatch)."""


class VersionConflictError(SnapshotStoreError):
    """Attempt to overwrite an existing version ID with different content."""


class SnapshotValidationError(SnapshotStoreError):
    """Record failed component validation before write."""


# --------------------------------------------------------------------------- #
#                            record (de)serialization                          #
# --------------------------------------------------------------------------- #
def _tup(data: Mapping[str, Any], key: str) -> tuple:
    return tuple(data.get(key) or ())


def prompt_entry_from_json(d: Mapping[str, Any]) -> PromptEntry:
    return PromptEntry(
        prompt_id=d["prompt_id"], prompt_text=d["prompt_text"],
        prompt_hash=d["prompt_hash"], name=d["name"], description=d["description"],
        tags=_tup(d, "tags"), applicability_traits=d.get("applicability_traits") or {},
        conflicts_with=_tup(d, "conflicts_with"), status=d.get("status", "active"),
        parent_prompt_ids=_tup(d, "parent_prompt_ids"),
        created_by=d.get("created_by", ""), created_at=d.get("created_at", ""),
        provenance=d.get("provenance") or {},
    )


def prompt_bank_from_json(d: Mapping[str, Any]) -> PromptBankSnapshot:
    return PromptBankSnapshot(
        bank_version=d["bank_version"],
        entries=tuple(prompt_entry_from_json(e) for e in d.get("entries") or ()),
        parent_bank_version=d.get("parent_bank_version"),
        default_entry_ids=_tup(d, "default_entry_ids"),
        max_selected_entries=d.get("max_selected_entries", 4),
        created_at=d.get("created_at", ""), created_by=d.get("created_by", ""),
        provenance=d.get("provenance") or {},
    )


def routing_rule_from_json(d: Mapping[str, Any]) -> RoutingRule:
    return RoutingRule(
        rule_id=d["rule_id"], priority=d["priority"],
        conditions=d.get("conditions") or {},
        target_prompt_ids=_tup(d, "target_prompt_ids"),
        enabled=d.get("enabled", True), provenance=d.get("provenance") or {},
    )


def router_policy_from_json(d: Mapping[str, Any]) -> RouterPolicySnapshot:
    return RouterPolicySnapshot(
        router_version=d["router_version"], policy_type=d["policy_type"],
        rules=tuple(routing_rule_from_json(r) for r in d.get("rules") or ()),
        parent_router_version=d.get("parent_router_version"),
        fallback_prompt_ids=_tup(d, "fallback_prompt_ids"),
        max_selected_entries=d.get("max_selected_entries", 4),
        configuration=d.get("configuration") or {},
        created_at=d.get("created_at", ""), provenance=d.get("provenance") or {},
    )


def scaffold_contract_from_json(d: Mapping[str, Any]) -> ScaffoldContract:
    return ScaffoldContract(
        contract_version=d["contract_version"],
        required_placeholders=_tup(d, "required_placeholders"),
        required_sections=_tup(d, "required_sections"),
        output_schema=d.get("output_schema") or {},
        forbidden_patterns=_tup(d, "forbidden_patterns"),
        max_prompt_tokens=d.get("max_prompt_tokens"),
        metadata=d.get("metadata") or {},
    )


def scaffold_policy_from_json(d: Mapping[str, Any]) -> ScaffoldPolicySnapshot:
    return ScaffoldPolicySnapshot(
        scaffold_version=d["scaffold_version"], policy_type=d["policy_type"],
        composition_instruction=d.get("composition_instruction", ""),
        ordering_policy=d.get("ordering_policy", ""),
        conflict_resolution_rules=_tup(d, "conflict_resolution_rules"),
        compression_policy=d.get("compression_policy", ""),
        parent_scaffold_version=d.get("parent_scaffold_version"),
        configuration=d.get("configuration") or {},
        created_at=d.get("created_at", ""), provenance=d.get("provenance") or {},
    )


# --------------------------------------------------------------------------- #
#                               atomic file IO                                 #
# --------------------------------------------------------------------------- #
def _atomic_write_text(path: str, text: str) -> None:
    """Same-directory tempfile + os.replace. On any failure the target file
    (and therefore the previously active state) is untouched."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _snapshot_text(record: Any) -> str:
    """Deterministic on-disk form (sorted keys, indent for reviewability)."""
    return json.dumps(as_json_dict(record), sort_keys=True, ensure_ascii=False,
                      indent=2) + "\n"


# --------------------------------------------------------------------------- #
#                           component snapshot store                           #
# --------------------------------------------------------------------------- #
class ComponentSnapshotStore:
    """Versioned write-once snapshot store for one component kind."""

    def __init__(self, root: str, *, kind: str, version_field: str,
                 from_json: Callable[[Mapping[str, Any]], Any],
                 validate_fn: Callable[[Any], tuple[str, ...]]) -> None:
        self.root = os.path.abspath(root)
        self.kind = kind
        self.version_field = version_field
        self.from_json = from_json
        self.validate_fn = validate_fn

    # ------------------------------ layout -------------------------------- #
    @property
    def snapshots_dir(self) -> str:
        return os.path.join(self.root, "snapshots")

    @property
    def current_path(self) -> str:
        return os.path.join(self.root, "current.json")

    def snapshot_path(self, version: str) -> str:
        validate_component_version(self.kind, version, field_name="version")
        return os.path.join(self.snapshots_dir, f"{version}.json")

    # ------------------------------ digest -------------------------------- #
    def content_digest(self, record: Any) -> str:
        """sha256 over canonical JSON of the record with its own version field
        removed (the version may embed this digest — see module docstring)."""
        payload = as_json_dict(record)
        payload.pop(self.version_field, None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def assign_version(self, record: Any, number: int) -> Any:
        """Copy of `record` whose version is `<kind>_v<NNNN>_<digest8>`, with
        the digest computed from the record's version-independent content."""
        digest = self.content_digest(record)
        version = make_component_version(self.kind, number, digest)
        return dataclasses.replace(record, **{self.version_field: version})

    def _check_digest_suffix(self, record: Any, version: str) -> None:
        parts = version.split("_")
        if len(parts) < 3:      # no content suffix
            return
        suffix = parts[2]
        digest = self.content_digest(record)
        if not digest.startswith(suffix):
            raise SnapshotStoreError(
                f"{version!r}: content-digest suffix {suffix!r} does not match "
                f"snapshot content digest {digest[:16]}…")

    # ------------------------------ save ---------------------------------- #
    def _record_version(self, record: Any) -> str:
        version = getattr(record, self.version_field)
        validate_component_version(self.kind, version,
                                   field_name=self.version_field)
        return version

    def _validate(self, record: Any) -> None:
        errors = self.validate_fn(record)
        if errors:
            raise SnapshotValidationError(
                f"{self.kind} snapshot failed validation: " + "; ".join(errors))

    def save_snapshot(self, record: Any) -> str:
        """Write-once save. Identical re-save is an idempotent no-op; a
        different payload under an existing version is a conflict."""
        version = self._record_version(record)
        self._validate(record)
        self._check_digest_suffix(record, version)
        text = _snapshot_text(record)
        path = self.snapshot_path(version)
        if os.path.exists(path):
            with open(path) as f:
                if f.read() == text:
                    return path                       # idempotent no-op
            raise VersionConflictError(
                f"{self.kind} version {version!r} already exists with different "
                f"content; changed content must receive a new version")
        _atomic_write_text(path, text)
        return path

    def set_current(self, version: str) -> None:
        path = self.snapshot_path(version)
        if not os.path.exists(path):
            raise SnapshotStoreError(
                f"cannot point current at unsaved {self.kind} version {version!r}")
        pointer = {
            "component": self.kind,
            "version": version,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_text(self.current_path,
                           json.dumps(pointer, sort_keys=True, indent=2) + "\n")

    def commit(self, record: Any) -> str:
        """save_snapshot + set_current. The explicit commit path (§23)."""
        path = self.save_snapshot(record)
        self.set_current(self._record_version(record))
        return path

    # ------------------------------ load ---------------------------------- #
    def load_version(self, version: str) -> Any:
        path = self.snapshot_path(version)
        if not os.path.exists(path):
            raise SnapshotStoreError(f"no {self.kind} snapshot {version!r} at {path}")
        with open(path) as f:
            data = json.load(f)
        record = self.from_json(data)
        actual = self._record_version(record)
        if actual != version:
            raise SnapshotStoreError(
                f"snapshot file {path} claims version {actual!r}, expected {version!r}")
        self._check_digest_suffix(record, version)
        return record

    def current_version(self) -> str | None:
        if not os.path.exists(self.current_path):
            return None
        with open(self.current_path) as f:
            pointer = json.load(f)
        if pointer.get("component") != self.kind:
            raise SnapshotStoreError(
                f"current.json at {self.root} belongs to component "
                f"{pointer.get('component')!r}, not {self.kind!r}")
        return pointer["version"]

    def load_current(self) -> Any:
        version = self.current_version()
        if version is None:
            raise SnapshotStoreError(f"no current {self.kind} pointer at {self.root}")
        return self.load_version(version)

    def list_versions(self) -> tuple[str, ...]:
        """Versions ordered by the parsed numeric component version (then the
        full string as tie-break), not lexicographically — 'v0002' sorts
        before 'v0010' and 'v10000' after 'v9999'."""
        if not os.path.isdir(self.snapshots_dir):
            return ()
        names = [name[:-len(".json")] for name in os.listdir(self.snapshots_dir)
                 if name.endswith(".json") and not name.startswith(".tmp_")]

        def numeric_key(version: str) -> tuple[int, str]:
            m = re.search(r"_v(\d+)", version)
            return (int(m.group(1)) if m else -1, version)

        return tuple(sorted(names, key=numeric_key))

    # ------------------------------ preview ------------------------------- #
    def write_preview(self, record: Any, preview_dir: str) -> str:
        """Dry-run artifact in a run/preview-specific directory. Never touches
        committed snapshots or the current pointer."""
        preview_dir = os.path.abspath(preview_dir)
        if preview_dir == self.root or preview_dir.startswith(self.root + os.sep):
            raise SnapshotStoreError(
                f"preview_dir {preview_dir} must be outside the store root {self.root}")
        version = self._record_version(record)
        self._validate(record)
        self._check_digest_suffix(record, version)
        path = os.path.join(preview_dir, f"preview_{self.kind}_{version}.json")
        _atomic_write_text(path, _snapshot_text(record))
        return path


# --------------------------------------------------------------------------- #
#                             concrete stores                                  #
# --------------------------------------------------------------------------- #
def prompt_bank_store(root: str) -> ComponentSnapshotStore:
    return ComponentSnapshotStore(root, kind="bank", version_field="bank_version",
                                  from_json=prompt_bank_from_json,
                                  validate_fn=validate_prompt_bank)


def router_policy_store(root: str) -> ComponentSnapshotStore:
    return ComponentSnapshotStore(root, kind="router", version_field="router_version",
                                  from_json=router_policy_from_json,
                                  validate_fn=validate_router_policy)


def scaffold_policy_store(root: str) -> ComponentSnapshotStore:
    return ComponentSnapshotStore(root, kind="scaffold",
                                  version_field="scaffold_version",
                                  from_json=scaffold_policy_from_json,
                                  validate_fn=validate_scaffold_policy)


def scaffold_contract_store(root: str) -> ComponentSnapshotStore:
    return ComponentSnapshotStore(root, kind="contract",
                                  version_field="contract_version",
                                  from_json=scaffold_contract_from_json,
                                  validate_fn=validate_scaffold_contract)
