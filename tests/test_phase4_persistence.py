"""Stage 4.2 persistence tests (PHASE4_PROMPT_ROUTING.md Stage 4.2).

Covers: deterministic round trip, immutable snapshots, atomic writes,
invalid active-entry references, retired-entry handling, scaffold policy vs
contract, failed write leaving incumbent unchanged, preview isolation,
parent-version provenance, duplicate active prompt hashes, pointer manifest,
idempotent re-save, and digest-suffix semantics. Every test runs in a pytest
tmp_path — no repository artifact, cache, or Phase 0-3 run directory is
touched.
"""

import dataclasses
import json
import os

import pytest

from surrogate_rollout.prompt_routing.persistence import (
    SnapshotStoreError,
    SnapshotValidationError,
    VersionConflictError,
    prompt_bank_store,
    router_policy_store,
    scaffold_contract_store,
    scaffold_policy_store,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    PromptEntry,
    RouterPolicySnapshot,
    RoutingRule,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    dumps_canonical,
    make_component_version,
)
from surrogate_rollout.prompt_routing.validators import (
    assert_router_compatible,
    validate_router_against_bank,
)
from surrogate_rollout.schemas import sha256_text

BANK_V1 = make_component_version("bank", 1)
BANK_V2 = make_component_version("bank", 2)
ROUTER_V1 = make_component_version("router", 1)
SCAFFOLD_V1 = make_component_version("scaffold", 1)
CONTRACT_V1 = make_component_version("contract", 1)


def entry(prompt_id="pe_temporal", text="Preserve the exact temporal order of actions.",
          **kw) -> PromptEntry:
    return PromptEntry(prompt_id=prompt_id, prompt_text=text,
                       prompt_hash=sha256_text(text), name=prompt_id,
                       description="test entry", **kw)


def bank(version=BANK_V1, entries=None, **kw) -> PromptBankSnapshot:
    if entries is None:
        entries = (entry(),
                   entry("pe_state", "Describe meaningful state changes explicitly."))
    kw.setdefault("max_selected_entries", 4)
    return PromptBankSnapshot(bank_version=version, entries=tuple(entries), **kw)


def router(version=ROUTER_V1, rules=None, **kw) -> RouterPolicySnapshot:
    if rules is None:
        rules = (RoutingRule(rule_id="r1", priority=10,
                             conditions={"has_transcript": True},
                             target_prompt_ids=("pe_temporal",)),)
    kw.setdefault("max_selected_entries", 4)
    return RouterPolicySnapshot(router_version=version, policy_type="rule_based",
                                rules=tuple(rules), **kw)


def contract(version=CONTRACT_V1, **kw) -> ScaffoldContract:
    return ScaffoldContract(
        contract_version=version,
        required_placeholders=("TRANSCRIPT_PLACEHOLDER", "CLIP_START_TIME",
                               "CLIP_END_TIME"),
        output_schema={"clip_description": "str", "subject_registry": "object"},
        **kw)


def scaffold(version=SCAFFOLD_V1, **kw) -> ScaffoldPolicySnapshot:
    return ScaffoldPolicySnapshot(scaffold_version=version,
                                  policy_type="deterministic", **kw)


# --------------------------- deterministic round trip ---------------------- #
def test_round_trip_all_components(tmp_path):
    cases = [
        (prompt_bank_store(str(tmp_path / "banks")), bank()),
        (router_policy_store(str(tmp_path / "routers")), router()),
        (scaffold_policy_store(str(tmp_path / "scaffolds")), scaffold()),
        (scaffold_contract_store(str(tmp_path / "contracts")), contract()),
    ]
    for store, record in cases:
        path = store.commit(record)
        version = getattr(record, store.version_field)
        assert os.path.basename(path) == f"{version}.json"
        loaded = store.load_version(version)
        assert loaded == record
        assert dumps_canonical(loaded) == dumps_canonical(record)
        assert store.load_current() == record
        assert store.current_version() == version


def test_snapshot_file_bytes_deterministic(tmp_path):
    s1 = prompt_bank_store(str(tmp_path / "a"))
    s2 = prompt_bank_store(str(tmp_path / "b"))
    p1 = s1.save_snapshot(bank())
    p2 = s2.save_snapshot(bank())
    assert open(p1).read() == open(p2).read()


# ----------------------------- immutable snapshots ------------------------- #
def test_loaded_snapshot_is_immutable(tmp_path):
    store = prompt_bank_store(str(tmp_path))
    store.commit(bank(provenance={"nested": {"k": [1, 2]}}))
    loaded = store.load_current()
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.bank_version = BANK_V2
    with pytest.raises(TypeError):
        loaded.provenance["nested"]["k"] = "x"
    assert loaded.provenance["nested"]["k"] == (1, 2)


def test_existing_version_never_overwritten_with_different_content(tmp_path):
    store = prompt_bank_store(str(tmp_path))
    store.commit(bank())
    changed = bank(entries=(entry(),))  # same version, different content
    with pytest.raises(VersionConflictError):
        store.save_snapshot(changed)
    assert store.load_current() == bank()  # incumbent intact


def test_identical_resave_is_idempotent_noop(tmp_path):
    store = prompt_bank_store(str(tmp_path))
    p1 = store.save_snapshot(bank())
    before = open(p1).read()
    p2 = store.save_snapshot(bank())
    assert p1 == p2 and open(p2).read() == before


# ------------------------------- atomic writes ----------------------------- #
def test_no_temp_files_after_commit(tmp_path):
    store = router_policy_store(str(tmp_path))
    store.commit(router())
    leftovers = [f for f in os.listdir(store.snapshots_dir) if f.startswith(".tmp_")]
    assert leftovers == []
    assert json.load(open(store.current_path))["component"] == "router"


def test_failed_write_leaves_incumbent_unchanged(tmp_path, monkeypatch):
    store = prompt_bank_store(str(tmp_path))
    store.commit(bank())
    incumbent_text = open(store.snapshot_path(BANK_V1)).read()
    pointer_text = open(store.current_path).read()

    def boom(src, dst):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("surrogate_rollout.prompt_routing.persistence.os.replace", boom)
    with pytest.raises(OSError):
        store.commit(bank(version=BANK_V2, parent_bank_version=BANK_V1))
    monkeypatch.undo()

    assert open(store.snapshot_path(BANK_V1)).read() == incumbent_text
    assert open(store.current_path).read() == pointer_text
    assert not os.path.exists(store.snapshot_path(BANK_V2))
    assert [f for f in os.listdir(store.snapshots_dir) if f.startswith(".tmp_")] == []


def test_failed_pointer_write_keeps_previous_pointer(tmp_path, monkeypatch):
    store = prompt_bank_store(str(tmp_path))
    store.commit(bank())
    pointer_text = open(store.current_path).read()
    store.save_snapshot(bank(version=BANK_V2, parent_bank_version=BANK_V1))

    real_replace = os.replace

    def boom_on_pointer(src, dst):
        if dst.endswith("current.json"):
            raise OSError("simulated pointer failure")
        return real_replace(src, dst)

    monkeypatch.setattr("surrogate_rollout.prompt_routing.persistence.os.replace",
                        boom_on_pointer)
    with pytest.raises(OSError):
        store.set_current(BANK_V2)
    monkeypatch.undo()
    assert open(store.current_path).read() == pointer_text
    assert store.current_version() == BANK_V1


# ------------------------ pointer manifest semantics ----------------------- #
def test_current_pointer_is_manifest_not_snapshot_copy(tmp_path):
    store = prompt_bank_store(str(tmp_path))
    store.commit(bank())
    pointer = json.load(open(store.current_path))
    assert set(pointer) == {"component", "version", "updated_at"}
    assert pointer["version"] == BANK_V1
    assert "entries" not in pointer  # no second mutable snapshot copy


def test_set_current_requires_saved_snapshot(tmp_path):
    store = prompt_bank_store(str(tmp_path))
    with pytest.raises(SnapshotStoreError):
        store.set_current(BANK_V1)


def test_kind_mismatch_pointer_rejected(tmp_path):
    prompt_bank_store(str(tmp_path)).commit(bank())
    with pytest.raises(SnapshotStoreError):
        router_policy_store(str(tmp_path)).current_version()


# --------------------- invalid references / retired entries ---------------- #
def test_router_unknown_target_rejected():
    b = bank()
    r = router(rules=(RoutingRule(rule_id="r_bad", priority=1, conditions={},
                                  target_prompt_ids=("pe_missing",)),))
    errors = validate_router_against_bank(r, b)
    assert any("unknown entry 'pe_missing'" in e for e in errors)
    with pytest.raises(ValueError):
        assert_router_compatible(r, b)


def test_retired_entry_handling():
    retired = entry("pe_old", "Old retired behavior.", status="retired")
    b = bank(entries=(entry(), retired))
    # enabled rule targeting retired entry: error
    r_enabled = router(rules=(RoutingRule(rule_id="r_old", priority=1, conditions={},
                                          target_prompt_ids=("pe_old",)),))
    assert any("retired entry 'pe_old'" in e
               for e in validate_router_against_bank(r_enabled, b))
    # disabled historical rule targeting retired entry: tolerated
    r_disabled = router(rules=(RoutingRule(rule_id="r_old", priority=1, conditions={},
                                           target_prompt_ids=("pe_old",),
                                           enabled=False),))
    assert validate_router_against_bank(r_disabled, b) == ()
    # retired fallback: error
    r_fb = router(fallback_prompt_ids=("pe_old",))
    assert any("fallback references retired entry 'pe_old'" in e
               for e in validate_router_against_bank(r_fb, b))


def test_selection_limit_compatibility():
    b = bank(max_selected_entries=2)
    r = router(max_selected_entries=4)
    assert any("exceeds bank max_selected_entries" in e
               for e in validate_router_against_bank(r, b))
    assert validate_router_against_bank(router(max_selected_entries=2), b) == ()


# ----------------------- duplicate active prompt hashes -------------------- #
def test_duplicate_active_prompt_hash_rejected_at_save(tmp_path):
    text = "Identical instruction text."
    dup = bank(entries=(entry("pe_a", text), entry("pe_b", text)))
    store = prompt_bank_store(str(tmp_path))
    with pytest.raises(SnapshotValidationError):
        store.save_snapshot(dup)
    assert store.list_versions() == ()


def test_duplicate_hash_allowed_when_one_retired(tmp_path):
    text = "Identical instruction text."
    ok = bank(entries=(entry("pe_a", text),
                       entry("pe_b", text, status="retired")))
    store = prompt_bank_store(str(tmp_path))
    store.save_snapshot(ok)
    assert store.list_versions() == (BANK_V1,)


# -------------------- scaffold policy cannot modify contract --------------- #
def test_scaffold_policy_smuggling_contract_fields_rejected(tmp_path):
    store = scaffold_policy_store(str(tmp_path))
    smuggler = scaffold(configuration={"output_schema": {"free": "text"}})
    with pytest.raises(SnapshotValidationError):
        store.save_snapshot(smuggler)
    assert store.list_versions() == ()


def test_scaffold_policy_commit_leaves_contract_store_untouched(tmp_path):
    c_store = scaffold_contract_store(str(tmp_path / "contracts"))
    p_store = scaffold_policy_store(str(tmp_path / "policies"))
    c_store.commit(contract())
    contract_bytes = open(c_store.snapshot_path(CONTRACT_V1)).read()
    p_store.commit(scaffold(configuration={"ordering_bias": "specialized_last"}))
    assert open(c_store.snapshot_path(CONTRACT_V1)).read() == contract_bytes
    assert c_store.current_version() == CONTRACT_V1


# ------------------------------ preview isolation --------------------------- #
def test_preview_does_not_alter_canonical_state(tmp_path):
    store = prompt_bank_store(str(tmp_path / "banks"))
    store.commit(bank())
    preview_dir = str(tmp_path / "runs" / "phase4_preview")
    candidate = bank(version=BANK_V2, parent_bank_version=BANK_V1)
    path = store.write_preview(candidate, preview_dir)
    assert os.path.dirname(path) == os.path.abspath(preview_dir)
    assert store.current_version() == BANK_V1          # pointer untouched
    assert store.list_versions() == (BANK_V1,)         # no committed snapshot
    assert "preview_bank_" in os.path.basename(path)


def test_preview_inside_store_root_rejected(tmp_path):
    store = prompt_bank_store(str(tmp_path / "banks"))
    with pytest.raises(SnapshotStoreError):
        store.write_preview(bank(), str(tmp_path / "banks" / "snapshots"))


# -------------------------- parent-version provenance ----------------------- #
def test_parent_version_provenance_round_trip(tmp_path):
    store = prompt_bank_store(str(tmp_path))
    store.commit(bank())
    v2 = bank(version=BANK_V2, parent_bank_version=BANK_V1,
              provenance={"derived_from": BANK_V1})
    store.commit(v2)
    loaded = store.load_current()
    assert loaded.parent_bank_version == BANK_V1
    assert loaded.provenance["derived_from"] == BANK_V1
    assert store.list_versions() == (BANK_V1, BANK_V2)
    assert store.load_version(BANK_V1).parent_bank_version is None  # history intact


# ------------------------------ digest semantics ---------------------------- #
def test_assign_version_digest_suffix_round_trip(tmp_path):
    store = prompt_bank_store(str(tmp_path))
    versioned = store.assign_version(bank(), 3)
    assert versioned.bank_version.startswith("bank_v0003_")
    store.commit(versioned)
    assert store.load_current() == versioned


def test_digest_excludes_version_string_itself(tmp_path):
    store = prompt_bank_store(str(tmp_path))
    a = store.assign_version(bank(), 1)
    b = store.assign_version(bank(version=BANK_V2), 1)
    assert a.bank_version == b.bank_version  # same content -> same digest


def test_tampered_digest_suffix_rejected(tmp_path):
    store = prompt_bank_store(str(tmp_path))
    fake = bank(version=make_component_version("bank", 3, "deadbeefdeadbeef"))
    with pytest.raises(SnapshotStoreError):
        store.save_snapshot(fake)
