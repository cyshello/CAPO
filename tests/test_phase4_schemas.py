"""Stage 4.1 foundational-schema tests (PHASE4_PROMPT_ROUTING.md Stage 4.1).

Covers: valid construction, immutability, deterministic serialization,
multiple selected prompt IDs, stable version identifiers, invalid confidence,
invalid component-version references, invalid composed-prompt states,
provenance preservation, and configuration defaults/round-trip.
"""

import dataclasses
import json

import pytest

from surrogate_rollout.prompt_routing.schemas import (
    ComposedCaptionPrompt,
    CompositionTrace,
    Phase4Config,
    Phase4OptimizationConfig,
    PromptBankSnapshot,
    PromptEntry,
    RouterPolicySnapshot,
    RoutingDecision,
    RoutingRule,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    SegmentContext,
    as_json_dict,
    dumps_canonical,
    make_component_version,
    validate_component_version,
)
from surrogate_rollout.schemas import sha256_text

BANK_V = make_component_version("bank", 1)
ROUTER_V = make_component_version("router", 1)
SCAFFOLD_V = make_component_version("scaffold", 1)
CONTRACT_V = make_component_version("contract", 1)


# ------------------------------ fixtures ---------------------------------- #
def entry(prompt_id="pe_temporal", text="Preserve the exact temporal order of actions.",
          **kw) -> PromptEntry:
    return PromptEntry(
        prompt_id=prompt_id, prompt_text=text, prompt_hash=sha256_text(text),
        name=prompt_id, description="test entry", **kw)


def bank(entries=None, **kw) -> PromptBankSnapshot:
    if entries is None:
        entries = (entry(), entry("pe_state", "Describe meaningful state changes explicitly."))
    return PromptBankSnapshot(bank_version=BANK_V, entries=tuple(entries), **kw)


def rule(rule_id="r1", targets=("pe_temporal",), **kw) -> RoutingRule:
    return RoutingRule(rule_id=rule_id, priority=10,
                       conditions={"has_transcript": True},
                       target_prompt_ids=tuple(targets), **kw)


def router(**kw) -> RouterPolicySnapshot:
    return RouterPolicySnapshot(router_version=ROUTER_V, policy_type="rule_based",
                                rules=(rule(),), **kw)


def segment(**kw) -> SegmentContext:
    kw.setdefault("timestamp_start", 0.0)
    kw.setdefault("timestamp_end", 10.0)
    return SegmentContext(video_id="0RxMZBLeqRI", segment_id="0_10", **kw)


def decision(selected=("pe_temporal", "pe_state"), **kw) -> RoutingDecision:
    kw.setdefault("selection_order", tuple(selected))
    kw.setdefault("confidence", 0.8)
    kw.setdefault("bank_version", BANK_V)
    kw.setdefault("router_version", ROUTER_V)
    return RoutingDecision(
        video_id="0RxMZBLeqRI", segment_id="0_10",
        selected_prompt_ids=tuple(selected),
        prompt_scores={pid: 0.9 for pid in selected},
        matched_rule_ids=("r1",), **kw)


def contract(**kw) -> ScaffoldContract:
    return ScaffoldContract(
        contract_version=CONTRACT_V,
        required_placeholders=("TRANSCRIPT_PLACEHOLDER", "CLIP_START_TIME",
                               "CLIP_END_TIME"),
        output_schema={"clip_description": "str", "subject_registry": "object"},
        **kw)


def scaffold(**kw) -> ScaffoldPolicySnapshot:
    return ScaffoldPolicySnapshot(scaffold_version=SCAFFOLD_V,
                                  policy_type="deterministic", **kw)


def trace(selected=("pe_temporal", "pe_state"), preserved=None, omitted=(),
          **kw) -> CompositionTrace:
    if preserved is None:
        preserved = tuple(s for s in selected if s not in omitted)
    return CompositionTrace(selected_prompt_ids=tuple(selected),
                            preserved_prompt_ids=tuple(preserved),
                            omitted_prompt_ids=tuple(omitted), **kw)


def composed(text="composed prompt TRANSCRIPT_PLACEHOLDER CLIP_START_TIME CLIP_END_TIME",
             selected=("pe_temporal", "pe_state"), **kw) -> ComposedCaptionPrompt:
    kw.setdefault("composition_trace", trace(selected))
    kw.setdefault("bank_version", BANK_V)
    kw.setdefault("router_version", ROUTER_V)
    kw.setdefault("scaffold_version", SCAFFOLD_V)
    kw.setdefault("contract_version", CONTRACT_V)
    return ComposedCaptionPrompt(
        video_id="0RxMZBLeqRI", segment_id="0_10",
        selected_prompt_ids=tuple(selected),
        prompt_text=text, prompt_hash=sha256_text(text), **kw)


# --------------------------- valid construction ---------------------------- #
def test_valid_construction_all_records():
    b = bank()
    assert b.entry("pe_temporal").status == "active"
    assert router().rules[0].enabled
    assert segment().segment_id == "0_10"
    assert decision().used_fallback is False
    assert contract().max_prompt_tokens is None
    assert scaffold().parent_scaffold_version is None
    c = composed()
    assert c.is_valid and c.validation_errors == ()


def test_zero_selected_entries_allowed():
    d = decision(selected=(), selection_order=())
    assert d.selected_prompt_ids == ()
    t = trace(selected=(), preserved=())
    assert t.preserved_prompt_ids == ()


def test_fallback_consistency():
    d = decision(used_fallback=True, fallback_reason="no rule matched")
    assert d.fallback_reason == "no rule matched"
    with pytest.raises(ValueError):
        decision(used_fallback=True)  # reason required
    with pytest.raises(ValueError):
        decision(fallback_reason="stray reason")  # used_fallback False


# ------------------------------ immutability ------------------------------- #
def test_records_are_frozen():
    for record in (entry(), bank(), rule(), router(), segment(), decision(),
                   contract(), scaffold(), trace(), composed(),
                   Phase4Config(), Phase4OptimizationConfig()):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, dataclasses.fields(record)[0].name, "x")


def test_mapping_fields_are_immutable():
    e = entry(applicability_traits={"needs_transcript": True})
    with pytest.raises(TypeError):
        e.applicability_traits["needs_transcript"] = False
    d = decision()
    with pytest.raises(TypeError):
        d.prompt_scores["pe_temporal"] = 0.0
    with pytest.raises(TypeError):
        contract().output_schema["extra"] = 1


def test_source_mapping_mutation_does_not_leak_in():
    src = {"origin": "unit"}
    e = entry(provenance=src)
    src["origin"] = "mutated"
    assert e.provenance["origin"] == "unit"  # defensive copy at construction


# --------------------------- recursive freezing ---------------------------- #
def test_nested_source_dict_mutation_does_not_leak_in():
    src = {"outer": {"inner": "unit", "deep": {"k": 1}}}
    e = entry(provenance=src)
    src["outer"]["inner"] = "mutated"
    src["outer"]["deep"]["k"] = 999
    assert e.provenance["outer"]["inner"] == "unit"
    assert e.provenance["outer"]["deep"]["k"] == 1


def test_nested_source_list_mutation_does_not_leak_in():
    src = {"steps": ["a", "b"], "nested": [{"k": ["x"]}]}
    e = entry(provenance=src)
    src["steps"].append("c")
    src["nested"][0]["k"].append("y")
    assert e.provenance["steps"] == ("a", "b")
    assert e.provenance["nested"][0]["k"] == ("x",)


def test_nested_values_immutable_through_record():
    e = entry(provenance={"outer": {"inner": 1}, "steps": ["a", "b"]})
    with pytest.raises(TypeError):
        e.provenance["outer"]["inner"] = 2      # nested mapping frozen
    assert isinstance(e.provenance["steps"], tuple)  # list became tuple
    with pytest.raises(AttributeError):
        e.provenance["steps"].append("c")


def test_sets_become_deterministic_tuples():
    a = entry(provenance={"tags": {"b", "a", "c"}})
    b = entry(provenance={"tags": {"c", "a", "b"}})
    assert a.provenance["tags"] == ("a", "b", "c")
    assert dumps_canonical(a) == dumps_canonical(b)


def test_raw_objects_inside_mappings_rejected_at_construction():
    with pytest.raises(TypeError):
        entry(provenance={"handle": object()})
    with pytest.raises(TypeError):
        segment(segment_features={"nested": [{"bad": object()}]})


def test_recursively_frozen_data_serializes_deterministically():
    e = entry(provenance={"z": {"b": [2, 1], "a": {"y", "x"}}, "a": ["m"]})
    d = as_json_dict(e)
    assert d["provenance"] == {"z": {"b": [2, 1], "a": ["x", "y"]}, "a": ["m"]}
    assert isinstance(d["provenance"]["a"], list)  # plain JSON types out
    assert dumps_canonical(e) == dumps_canonical(entry(
        provenance={"a": ["m"], "z": {"a": {"x", "y"}, "b": [2, 1]}}))


# ------------------------- deterministic serialization --------------------- #
def test_deterministic_serialization_across_mapping_order():
    a = decision()
    b = RoutingDecision(
        video_id="0RxMZBLeqRI", segment_id="0_10",
        bank_version=BANK_V, router_version=ROUTER_V,
        selected_prompt_ids=("pe_temporal", "pe_state"),
        prompt_scores={"pe_state": 0.9, "pe_temporal": 0.9},  # reversed insertion
        matched_rule_ids=("r1",), selection_order=("pe_temporal", "pe_state"),
        confidence=0.8)
    assert dumps_canonical(a) == dumps_canonical(b)


def test_serialization_round_trip_is_stable():
    c = composed()
    s1 = dumps_canonical(c)
    s2 = dumps_canonical(c)
    assert s1 == s2
    parsed = json.loads(s1)
    assert parsed["prompt_hash"] == c.prompt_hash
    assert parsed["composition_trace"]["selected_prompt_ids"] == list(
        c.selected_prompt_ids)


def test_serialization_rejects_raw_objects():
    with pytest.raises(TypeError):
        dumps_canonical(object())


# ------------------------- multiple selected prompt IDs -------------------- #
def test_multiple_selected_prompt_ids():
    ids = ("pe_a", "pe_b", "pe_c")
    d = decision(selected=ids, selection_order=("pe_b", "pe_a", "pe_c"))
    assert len(d.selected_prompt_ids) == 3
    assert sorted(d.selection_order) == sorted(ids)
    c = composed(selected=ids, composition_trace=trace(ids))
    assert c.selected_prompt_ids == ids


def test_duplicate_selected_ids_rejected():
    with pytest.raises(ValueError):
        decision(selected=("pe_a", "pe_a"), selection_order=("pe_a", "pe_a"))


def test_selection_order_must_be_permutation():
    with pytest.raises(ValueError):
        decision(selected=("pe_a", "pe_b"), selection_order=("pe_a",))
    with pytest.raises(ValueError):
        decision(selected=("pe_a",), selection_order=("pe_z",))


# ------------------------- stable version identifiers ---------------------- #
def test_make_component_version_deterministic_and_stable():
    assert make_component_version("bank", 1) == "bank_v0001"
    assert make_component_version("scaffold", 12, "ab12cd34ef") == "scaffold_v0012_ab12cd34"
    assert make_component_version("bank", 1) == make_component_version("bank", 1)


def test_version_validation():
    validate_component_version("bank", "bank_v0001", field_name="x")
    for bad in ("bank_v1", "router_v0001", "bankv0001", "", None):
        with pytest.raises(ValueError):
            validate_component_version("bank", bad, field_name="x")
    validate_component_version("bank", None, field_name="x", allow_none=True)
    with pytest.raises(ValueError):
        make_component_version("promptbank", 1)
    with pytest.raises(ValueError):
        make_component_version("bank", -1)


# --------------------------- invalid confidence ---------------------------- #
@pytest.mark.parametrize("value", [-0.1, 1.5, 2.0])
def test_invalid_routing_confidence(value):
    with pytest.raises(ValueError):
        decision(confidence=value)


def test_invalid_feedback_confidence_threshold():
    with pytest.raises(ValueError):
        Phase4OptimizationConfig(min_feedback_confidence=1.2)
    with pytest.raises(ValueError):
        Phase4OptimizationConfig(min_feedback_confidence=-0.5)


# ---------------------- invalid component-version refs --------------------- #
def test_wrong_kind_versions_rejected():
    with pytest.raises(ValueError):
        PromptBankSnapshot(bank_version=ROUTER_V, entries=(entry(),))
    with pytest.raises(ValueError):
        RouterPolicySnapshot(router_version=BANK_V, policy_type="rule_based")
    with pytest.raises(ValueError):
        composed(scaffold_version=CONTRACT_V)
    with pytest.raises(ValueError):
        decision(bank_version="bank_1")


def test_bank_default_entry_ids_must_reference_active_entries():
    retired = entry("pe_old", "Old behavior.", status="retired")
    with pytest.raises(ValueError):
        bank(entries=(entry(), retired), default_entry_ids=("pe_old",))
    with pytest.raises(ValueError):
        bank(default_entry_ids=("pe_missing",))


def test_bank_duplicate_entry_ids_rejected():
    with pytest.raises(ValueError):
        bank(entries=(entry(), entry()))


def test_prompt_entry_hash_must_match_text():
    with pytest.raises(ValueError):
        PromptEntry(prompt_id="pe_bad", prompt_text="text",
                    prompt_hash=sha256_text("other text"),
                    name="pe_bad", description="")


# ------------------------ invalid composed-prompt state -------------------- #
def test_composed_valid_cannot_carry_errors():
    with pytest.raises(ValueError):
        composed(is_valid=True, validation_errors=("missing placeholder",))


def test_composed_invalid_requires_errors():
    with pytest.raises(ValueError):
        composed(is_valid=False)


def test_composed_invalid_with_errors_is_constructible():
    c = composed(is_valid=False,
                 validation_errors=("missing placeholder TRANSCRIPT_PLACEHOLDER",))
    assert not c.is_valid and c.validation_errors


def test_composed_hash_must_match_text():
    text = "some composed text"
    with pytest.raises(ValueError):
        ComposedCaptionPrompt(
            video_id="v", segment_id="0_10",
            bank_version=BANK_V, router_version=ROUTER_V,
            scaffold_version=SCAFFOLD_V, contract_version=CONTRACT_V,
            selected_prompt_ids=("pe_a",), prompt_text=text,
            prompt_hash=sha256_text("different"),
            composition_trace=trace(("pe_a",)))


def test_composed_ids_must_match_trace():
    with pytest.raises(ValueError):
        composed(selected=("pe_a",), composition_trace=trace(("pe_b",)))


def test_trace_partition_enforced():
    with pytest.raises(ValueError):  # preserved+omitted must cover selected
        trace(selected=("pe_a", "pe_b"), preserved=("pe_a",))
    with pytest.raises(ValueError):  # cannot be both preserved and omitted
        trace(selected=("pe_a",), preserved=("pe_a",), omitted=("pe_a",))
    t = trace(selected=("pe_a", "pe_b"), preserved=("pe_a",), omitted=("pe_b",))
    assert t.omitted_prompt_ids == ("pe_b",)


def test_segment_id_is_opaque_non_empty():
    # current integer clip key
    assert SegmentContext(video_id="v", segment_id="0_10").segment_id == "0_10"
    # decimal timestamp key
    assert SegmentContext(video_id="v", segment_id="12.5_20.0").segment_id == "12.5_20.0"
    # other opaque identifier (format checks belong to the DVD adapter)
    assert SegmentContext(video_id="v", segment_id="shot-007").segment_id == "shot-007"
    # empty rejected
    with pytest.raises(ValueError):
        SegmentContext(video_id="v", segment_id="")


def test_segment_timestamps_ordered():
    with pytest.raises(ValueError):
        segment(timestamp_start=10.0, timestamp_end=0.0)


# ------------------------- provenance preservation ------------------------- #
def test_composed_prompt_retains_full_provenance():
    c = composed()
    d = as_json_dict(c)
    assert d["bank_version"] == BANK_V
    assert d["router_version"] == ROUTER_V
    assert d["scaffold_version"] == SCAFFOLD_V
    assert d["contract_version"] == CONTRACT_V
    assert d["selected_prompt_ids"] == ["pe_temporal", "pe_state"]
    assert d["prompt_hash"] == sha256_text(c.prompt_text)


def test_cache_identity_separate_from_provenance():
    """Same composed text under different component versions: identical
    prompt_hash (shared caption-cache identity) but distinguishable records."""
    text = "identical composed text TRANSCRIPT_PLACEHOLDER"
    a = composed(text=text)
    b = ComposedCaptionPrompt(
        video_id="0RxMZBLeqRI", segment_id="0_10",
        bank_version=make_component_version("bank", 2),
        router_version=make_component_version("router", 2),
        scaffold_version=SCAFFOLD_V, contract_version=CONTRACT_V,
        selected_prompt_ids=("pe_temporal", "pe_state"),
        prompt_text=text, prompt_hash=sha256_text(text),
        composition_trace=trace())
    assert a.prompt_hash == b.prompt_hash
    assert dumps_canonical(a) != dumps_canonical(b)  # provenance differs


# ------------------------------ configuration ------------------------------ #
def test_optimize_scaffold_defaults_false():
    cfg = Phase4Config()
    assert cfg.optimization.optimize_prompt_bank is True
    assert cfg.optimization.optimize_router is True
    assert cfg.optimization.optimize_scaffold is False
    assert cfg.optimize_scaffold is False  # visible at top level
    assert cfg.dry_run is True and cfg.commit is False


def test_dry_run_and_commit_mutually_exclusive():
    with pytest.raises(ValueError):
        Phase4Config(dry_run=True, commit=True)
    Phase4Config(dry_run=False, commit=True)  # explicit commit allowed


@pytest.mark.parametrize("flags", [
    dict(optimize_prompt_bank=False),
    dict(optimize_router=False),
    dict(optimize_scaffold=True),
    dict(optimize_prompt_bank=False, optimize_router=False, optimize_scaffold=True),
])
def test_config_round_trip_preserves_all_switches(flags):
    cfg = Phase4Config(seed=7, optimization=Phase4OptimizationConfig(**flags))
    restored = Phase4Config.from_json_dict(json.loads(json.dumps(cfg.as_json_dict())))
    assert restored == cfg
    assert restored.optimization == cfg.optimization
    assert dumps_canonical(restored) == dumps_canonical(cfg)


def test_config_round_trip_via_canonical_json_nested_values():
    cfg = Phase4Config(seed=3, dry_run=False, commit=True,
                       optimization=Phase4OptimizationConfig(
                           optimize_scaffold=True, max_iterations=2,
                           min_feedback_confidence=0.5))
    restored = Phase4Config.from_json_dict(json.loads(dumps_canonical(cfg)))
    assert restored == cfg
    assert restored.optimize_scaffold is True
    assert restored.optimization.min_feedback_confidence == 0.5
