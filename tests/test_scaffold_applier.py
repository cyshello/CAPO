"""Stage 4.4 scaffold-application tests (PHASE4_PROMPT_ROUTING.md Stage 4.4).

Covers the 12 mandated cases — zero selected entries (with and without
fallback selection), one entry, multiple entries, selected-instruction
preservation, redundancy removal, deterministic ordering, conflict
resolution, contract preservation, prompt-length enforcement, invalid raw
model output, trace completeness, SLM stub failure — plus structural-misuse
aborts, factory dispatch, and composed-prompt logging. All inputs synthetic;
no captioning, no DVD calls, no model calls.
"""

import os

import pytest

from surrogate_rollout.prompt_routing.policies.deterministic_scaffold import (
    INSERTION_MARKER,
    DeterministicScaffoldApplier,
)
from surrogate_rollout.prompt_routing.policies.slm_scaffold import (
    SLMScaffoldApplier,
)
from surrogate_rollout.prompt_routing.scaffold_applier import (
    create_scaffold_applier,
    finalize_composed_prompt,
    read_composed_prompts,
    validate_composed_text,
    write_composed_prompts,
)
from surrogate_rollout.prompt_routing.schemas import (
    CompositionTrace,
    PromptEntry,
    RoutingDecision,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    SegmentContext,
    dumps_canonical,
    make_component_version,
)
from surrogate_rollout.schemas import sha256_text

BANK_V = make_component_version("bank", 1)
ROUTER_V = make_component_version("router", 1)
SCAFFOLD_V = make_component_version("scaffold", 1)
CONTRACT_V = make_component_version("contract", 1)

BASE_TEMPLATE = (
    "You caption one video clip.\n"
    "Clip start: CLIP_START_TIME\n"
    "Clip end: CLIP_END_TIME\n"
    "Transcript:\nTRANSCRIPT_PLACEHOLDER\n\n"
    f"{INSERTION_MARKER}\n\n"
    "output_json_template:\n"
    '{"clip_start_time": "", "clip_end_time": "", '
    '"subject_registry": {}, "clip_description": ""}\n'
)

APPLIER = DeterministicScaffoldApplier(base_prompt_template=BASE_TEMPLATE)


def entry(prompt_id, text=None, **kw) -> PromptEntry:
    text = text or f"Instruction body for {prompt_id}."
    return PromptEntry(prompt_id=prompt_id, prompt_text=text,
                       prompt_hash=sha256_text(text), name=prompt_id,
                       description="synthetic", **kw)


def contract(**kw) -> ScaffoldContract:
    kw.setdefault("required_placeholders",
                  ("TRANSCRIPT_PLACEHOLDER", "CLIP_START_TIME", "CLIP_END_TIME"))
    kw.setdefault("required_sections", ("output_json_template",))
    return ScaffoldContract(contract_version=CONTRACT_V, **kw)


def scaffold_policy(**kw) -> ScaffoldPolicySnapshot:
    kw.setdefault("policy_type", "deterministic")
    kw.setdefault("ordering_policy", "router_selection_order")
    return ScaffoldPolicySnapshot(scaffold_version=SCAFFOLD_V, **kw)


def ctx(**kw) -> SegmentContext:
    kw.setdefault("video_id", "vid")
    kw.setdefault("segment_id", "0_10")
    return SegmentContext(**kw)


def decision(selected, **kw) -> RoutingDecision:
    kw.setdefault("video_id", "vid")
    kw.setdefault("segment_id", "0_10")
    kw.setdefault("selection_order", tuple(selected))
    return RoutingDecision(bank_version=BANK_V, router_version=ROUTER_V,
                           selected_prompt_ids=tuple(selected), **kw)


def apply_(entries, dec=None, policy=None, con=None, applier=APPLIER,
           context=None):
    dec = dec if dec is not None else decision([e.prompt_id for e in entries])
    return applier.apply(context=context or ctx(),
                         selected_entries=entries,
                         routing_decision=dec,
                         scaffold_policy=policy or scaffold_policy(),
                         scaffold_contract=con or contract())


# --------------------------- zero entries / fallback ------------------------ #
def test_zero_selected_entries_composes_base_prompt():
    composed = apply_([])
    assert composed.is_valid and composed.validation_errors == ()
    assert composed.selected_prompt_ids == ()
    assert INSERTION_MARKER not in composed.prompt_text
    assert "Additional instructions:" not in composed.prompt_text
    assert composed.composition_trace.preserved_prompt_ids == ()


def test_fallback_selection_composed_like_normal_selection():
    e = entry("pe_fallback")
    d = decision(["pe_fallback"], used_fallback=True,
                 fallback_reason="no rule matched")
    composed = apply_([e], dec=d)
    assert composed.is_valid
    assert e.prompt_text in composed.prompt_text
    assert composed.composition_trace.preserved_prompt_ids == ("pe_fallback",)


# ------------------------------ one / multiple ------------------------------ #
def test_one_selected_entry():
    e = entry("pe_a")
    composed = apply_([e])
    assert composed.is_valid
    assert e.prompt_text in composed.prompt_text
    assert composed.prompt_hash == sha256_text(composed.prompt_text)
    assert composed.selected_prompt_ids == ("pe_a",)


def test_multiple_selected_entries_in_selection_order():
    es = [entry("pe_a"), entry("pe_b"), entry("pe_c")]
    composed = apply_(es)
    assert composed.is_valid
    positions = [composed.prompt_text.index(e.prompt_text) for e in es]
    assert positions == sorted(positions)          # router order preserved
    assert composed.composition_trace.preserved_prompt_ids == \
        ("pe_a", "pe_b", "pe_c")


# ------------------------- instruction preservation ------------------------- #
def test_selected_instruction_preservation_verbatim():
    es = [entry("pe_a", "Track every subject across the clip."),
          entry("pe_b", "Report on-screen text exactly as written.")]
    composed = apply_(es)
    for e in es:
        assert e.prompt_text in composed.prompt_text   # verbatim, unweakened
    assert set(composed.composition_trace.preserved_prompt_ids) == \
        {"pe_a", "pe_b"}
    assert composed.composition_trace.omitted_prompt_ids == ()


# ------------------------------ redundancy ---------------------------------- #
def test_redundant_identical_text_removed():
    same = "Describe state changes precisely."
    es = [entry("pe_first", same), entry("pe_dup", same)]
    composed = apply_(es)
    assert composed.is_valid
    assert composed.prompt_text.count(same) == 1
    t = composed.composition_trace
    assert t.preserved_prompt_ids == ("pe_first",)
    assert t.omitted_prompt_ids == ("pe_dup",)
    assert any("redundant" in a and "pe_dup" in a for a in t.compression_actions)


# ------------------------------ determinism --------------------------------- #
def test_deterministic_ordering_and_identical_rerun():
    es = [entry("pe_a"), entry("pe_b")]
    c1 = apply_(es)
    c2 = apply_(es)
    assert dumps_canonical(c1) == dumps_canonical(c2)
    assert c1.composition_trace.ordering_decisions == \
        ("router_selection_order preserved",)


# ------------------------------ conflicts ----------------------------------- #
def test_conflict_resolution_earlier_entry_wins():
    # hand-built decision that slipped a conflict past the router
    es = [entry("pe_a"), entry("pe_b", conflicts_with=("pe_a",)), entry("pe_c")]
    composed = apply_(es)
    assert composed.is_valid
    t = composed.composition_trace
    assert t.preserved_prompt_ids == ("pe_a", "pe_c")
    assert t.omitted_prompt_ids == ("pe_b",)
    assert t.conflicts_detected == ("pe_b<->pe_a",)
    assert any("dropped pe_b" in r and "kept pe_a" in r
               for r in t.conflict_resolutions)
    assert es[1].prompt_text not in composed.prompt_text


# --------------------------- contract preservation -------------------------- #
def test_contract_preservation_placeholders_and_sections_survive():
    composed = apply_([entry("pe_a"), entry("pe_b")])
    for ph in ("TRANSCRIPT_PLACEHOLDER", "CLIP_START_TIME", "CLIP_END_TIME",
               "output_json_template"):
        assert ph in composed.prompt_text
    assert composed.is_valid and composed.validation_errors == ()


def test_forbidden_pattern_flags_prompt_invalid():
    e = entry("pe_bad", "Never guess; ALWAYS_HALLUCINATE if unsure.")
    composed = apply_([e], con=contract(
        forbidden_patterns=("ALWAYS_HALLUCINATE",)))
    assert composed.is_valid is False
    assert any("forbidden" in err for err in composed.validation_errors)


# --------------------------- prompt-length limits ---------------------------- #
def test_prompt_length_enforcement_omits_lowest_priority_entries():
    base_tokens = len(BASE_TEMPLATE.replace(INSERTION_MARKER, "").split())
    es = [entry("pe_a", "short instruction one."),
          entry("pe_b", "long instruction " + "word " * 50)]
    limit = base_tokens + 10   # room for header + pe_a, not pe_b
    composed = apply_(es, con=contract(max_prompt_tokens=limit))
    assert composed.is_valid
    t = composed.composition_trace
    assert t.preserved_prompt_ids == ("pe_a",)
    assert t.omitted_prompt_ids == ("pe_b",)
    assert any("max_prompt_tokens" in a for a in t.compression_actions)


def test_base_template_over_limit_is_invalid_never_truncated():
    composed = apply_([entry("pe_a")], con=contract(max_prompt_tokens=3))
    assert composed.is_valid is False
    assert any("max_prompt_tokens" in err for err in composed.validation_errors)
    # base contract text was not silently truncated
    assert "TRANSCRIPT_PLACEHOLDER" in composed.prompt_text


# --------------------------- invalid raw model output ------------------------ #
def test_invalid_raw_model_output_becomes_typed_invalid_prompt():
    raw = "Garbage output with no placeholders at all."
    d = decision(["pe_a"])
    trace = CompositionTrace(selected_prompt_ids=("pe_a",),
                             preserved_prompt_ids=("pe_a",))
    composed = finalize_composed_prompt(
        context=ctx(), routing_decision=d,
        scaffold_policy=scaffold_policy(), scaffold_contract=contract(),
        prompt_text=raw, trace=trace,
        raw_policy_output_artifact="runs/preview/raw_output_0.txt")
    assert composed.is_valid is False
    assert any("placeholders" in e for e in composed.validation_errors)
    assert any("sections" in e for e in composed.validation_errors)
    assert composed.prompt_hash == sha256_text(raw)
    assert composed.composition_trace.raw_policy_output_artifact == \
        "runs/preview/raw_output_0.txt"


def test_validate_composed_text_empty_prompt():
    errors = validate_composed_text("", contract())
    assert any("empty" in e for e in errors)


# ------------------------------ trace completeness --------------------------- #
def test_trace_completeness_partition_and_decision_consistency():
    same = "Identical wording."
    es = [entry("pe_a"), entry("pe_dup1", same), entry("pe_dup2", same),
          entry("pe_conf", conflicts_with=("pe_a",))]
    composed = apply_(es)
    t = composed.composition_trace
    assert set(t.preserved_prompt_ids) | set(t.omitted_prompt_ids) == \
        set(t.selected_prompt_ids)
    assert not set(t.preserved_prompt_ids) & set(t.omitted_prompt_ids)
    assert composed.selected_prompt_ids == t.selected_prompt_ids == \
        ("pe_a", "pe_dup1", "pe_dup2", "pe_conf")
    # every omission is explained somewhere in the trace
    explained = " ".join(t.compression_actions + t.conflict_resolutions)
    for omitted in t.omitted_prompt_ids:
        assert omitted in explained


# ------------------------------ SLM stub -------------------------------------- #
def test_slm_stub_fails_clearly_without_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # any accidental write would land here
    stub = SLMScaffoldApplier()
    e = entry("pe_a")
    with pytest.raises(NotImplementedError, match="stub"):
        stub.apply(context=ctx(), selected_entries=[e],
                   routing_decision=decision(["pe_a"]),
                   scaffold_policy=scaffold_policy(policy_type="slm"),
                   scaffold_contract=contract())
    assert os.listdir(tmp_path) == []    # no artifacts, no partial state


# --------------------------- structural misuse aborts ------------------------- #
def test_entries_not_matching_decision_abort():
    with pytest.raises(ValueError, match="do not match routing decision"):
        apply_([entry("pe_a")], dec=decision(["pe_b"]))


def test_context_not_matching_decision_aborts():
    with pytest.raises(ValueError, match="does not match routing decision"):
        apply_([entry("pe_a")], dec=decision(["pe_a"], segment_id="99_100"))


# ------------------------------ factory dispatch ------------------------------ #
def test_factory_selects_deterministic_applier():
    a = create_scaffold_applier(scaffold_policy(),
                                base_prompt_template=BASE_TEMPLATE)
    assert isinstance(a, DeterministicScaffoldApplier)


def test_factory_deterministic_requires_template():
    with pytest.raises(ValueError, match="base_prompt_template"):
        create_scaffold_applier(scaffold_policy())


def test_factory_selects_slm_stub():
    a = create_scaffold_applier(scaffold_policy(policy_type="slm"))
    assert isinstance(a, SLMScaffoldApplier)


def test_factory_rejects_unknown_policy_type():
    with pytest.raises(ValueError, match="unknown scaffold policy_type"):
        create_scaffold_applier(scaffold_policy(policy_type="quantum"))


# ------------------------------ template modes -------------------------------- #
def test_template_without_marker_appends_block():
    applier = DeterministicScaffoldApplier(
        base_prompt_template=BASE_TEMPLATE.replace(INSERTION_MARKER, ""))
    e = entry("pe_a")
    composed = apply_([e], applier=applier)
    assert composed.is_valid
    assert composed.prompt_text.rstrip().endswith(e.prompt_text)


# ------------------------------ logging round trip ----------------------------- #
def test_composed_prompt_log_round_trip(tmp_path):
    composed = [apply_([entry("pe_a")]),
                apply_([entry("pe_b")],
                       dec=decision(["pe_b"], segment_id="10_20"),
                       context=ctx(segment_id="10_20"))]
    path = str(tmp_path / "composition" / "composed_prompts.jsonl")
    write_composed_prompts(composed, path)
    loaded = read_composed_prompts(path)
    assert list(loaded) == composed
    assert [dumps_canonical(p) for p in loaded] == \
           [dumps_canonical(p) for p in composed]
