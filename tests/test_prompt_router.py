"""Stage 4.3 rule-based router tests (PHASE4_PROMPT_ROUTING.md Stage 4.3).

Covers: zero/one/multiple matching rules, duplicate-target deduplication,
conflict handling, max-selected enforcement, disabled rules, retired-target
rejection, deterministic selection order, query-agnostic default behavior,
and routing-decision logging. All contexts are synthetic; no captioning, no
DVD calls.
"""

import pytest

from surrogate_rollout.prompt_routing.policies.rule_based_router import (
    RuleBasedPromptRouter,
)
from surrogate_rollout.prompt_routing.router import (
    read_routing_decisions,
    write_routing_decisions,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    PromptEntry,
    RouterPolicySnapshot,
    RoutingRule,
    SegmentContext,
    dumps_canonical,
    make_component_version,
)
from surrogate_rollout.schemas import sha256_text

BANK_V = make_component_version("bank", 1)
ROUTER_V = make_component_version("router", 1)

ROUTER = RuleBasedPromptRouter()


def entry(prompt_id, text=None, **kw) -> PromptEntry:
    text = text or f"Instruction body for {prompt_id}."
    return PromptEntry(prompt_id=prompt_id, prompt_text=text,
                       prompt_hash=sha256_text(text), name=prompt_id,
                       description="synthetic", **kw)


def bank(*entries, **kw) -> PromptBankSnapshot:
    if not entries:
        entries = (entry("pe_a"), entry("pe_b"), entry("pe_c"), entry("pe_d"))
    kw.setdefault("max_selected_entries", 4)
    return PromptBankSnapshot(bank_version=BANK_V, entries=tuple(entries), **kw)


def rule(rule_id, targets, priority=10, conditions=None, enabled=True) -> RoutingRule:
    return RoutingRule(rule_id=rule_id, priority=priority,
                       conditions=conditions or {},
                       target_prompt_ids=tuple(targets), enabled=enabled)


def policy(*rules, **kw) -> RouterPolicySnapshot:
    kw.setdefault("max_selected_entries", 4)
    return RouterPolicySnapshot(router_version=ROUTER_V, policy_type="rule_based",
                                rules=tuple(rules), **kw)


def ctx(**kw) -> SegmentContext:
    kw.setdefault("segment_features", {"has_transcript": True})
    return SegmentContext(video_id="vid", segment_id="0_10", **kw)


# ------------------------------ rule matching ------------------------------ #
def test_zero_matching_rules_no_fallback():
    d = ROUTER.route(ctx(), bank(),
                     policy(rule("r1", ["pe_a"], conditions={"is_dark": True})))
    assert d.selected_prompt_ids == ()
    assert d.matched_rule_ids == ()
    assert d.used_fallback is False and d.fallback_reason is None


def test_zero_matching_rules_with_fallback():
    d = ROUTER.route(ctx(), bank(),
                     policy(rule("r1", ["pe_a"], conditions={"is_dark": True}),
                            fallback_prompt_ids=("pe_d", "pe_b")))
    assert d.selected_prompt_ids == ("pe_d", "pe_b")  # configured order kept
    assert d.used_fallback is True and d.fallback_reason
    assert d.prompt_scores["pe_d"] == 0.5


def test_one_matching_rule():
    d = ROUTER.route(ctx(), bank(),
                     policy(rule("r1", ["pe_a", "pe_b"],
                                 conditions={"has_transcript": True})))
    assert d.selected_prompt_ids == ("pe_a", "pe_b")
    assert d.matched_rule_ids == ("r1",)
    assert d.selection_order == d.selected_prompt_ids
    assert d.prompt_scores == {"pe_a": 1.0, "pe_b": 1.0}


def test_multiple_matching_rules_priority_order():
    d = ROUTER.route(ctx(), bank(),
                     policy(rule("r_low", ["pe_c"], priority=50),
                            rule("r_high", ["pe_a"], priority=1),
                            rule("r_mid", ["pe_b"], priority=10)))
    assert d.selected_prompt_ids == ("pe_a", "pe_b", "pe_c")
    assert d.matched_rule_ids == ("r_high", "r_mid", "r_low")


def test_duplicate_target_deduplicated():
    d = ROUTER.route(ctx(), bank(),
                     policy(rule("r1", ["pe_a"], priority=1),
                            rule("r2", ["pe_a", "pe_b"], priority=2)))
    assert d.selected_prompt_ids == ("pe_a", "pe_b")
    assert d.matched_rule_ids == ("r1", "r2")  # both matched, one selection


def test_disabled_rules_ignored():
    d = ROUTER.route(ctx(), bank(),
                     policy(rule("r_off", ["pe_a"], enabled=False),
                            rule("r_on", ["pe_b"])))
    assert d.selected_prompt_ids == ("pe_b",)
    assert d.matched_rule_ids == ("r_on",)


# ------------------------------ conflicts ---------------------------------- #
def test_conflicting_candidate_dropped_earlier_wins():
    b = bank(entry("pe_a", conflicts_with=("pe_b",)), entry("pe_b"), entry("pe_c"))
    d = ROUTER.route(ctx(), b,
                     policy(rule("r1", ["pe_a"], priority=1),
                            rule("r2", ["pe_b", "pe_c"], priority=2)))
    assert d.selected_prompt_ids == ("pe_a", "pe_c")
    assert d.decision_payload["conflicts_dropped"] == (
        {"dropped": "pe_b", "kept": "pe_a", "rule_id": "r2"},)
    assert d.prompt_scores["pe_b"] == 1.0  # considered, scored, dropped


def test_conflict_checked_in_both_directions():
    # conflict declared only on pe_b, but pe_a selected first still blocks pe_b
    b = bank(entry("pe_a"), entry("pe_b", conflicts_with=("pe_a",)), entry("pe_c"))
    d = ROUTER.route(ctx(), b,
                     policy(rule("r1", ["pe_a", "pe_b", "pe_c"])))
    assert d.selected_prompt_ids == ("pe_a", "pe_c")


# ------------------------------ budget -------------------------------------- #
def test_max_selected_entries_enforced():
    d = ROUTER.route(ctx(), bank(),
                     policy(rule("r1", ["pe_a", "pe_b", "pe_c", "pe_d"]),
                            max_selected_entries=2))
    assert d.selected_prompt_ids == ("pe_a", "pe_b")
    assert d.decision_payload["budget_dropped"] == ("pe_c", "pe_d")
    assert set(d.prompt_scores) == {"pe_a", "pe_b", "pe_c", "pe_d"}


def test_router_limit_exceeding_bank_limit_aborts():
    with pytest.raises(ValueError):
        ROUTER.route(ctx(), bank(max_selected_entries=2),
                     policy(rule("r1", ["pe_a"]), max_selected_entries=4))


# --------------------------- retired / unknown targets ---------------------- #
def test_retired_target_in_enabled_rule_aborts():
    b = bank(entry("pe_a"), entry("pe_old", status="retired"))
    with pytest.raises(ValueError):
        ROUTER.route(ctx(), b, policy(rule("r1", ["pe_old"])))


def test_retired_target_in_disabled_rule_tolerated():
    b = bank(entry("pe_a"), entry("pe_old", status="retired"))
    d = ROUTER.route(ctx(), b,
                     policy(rule("r_hist", ["pe_old"], enabled=False),
                            rule("r1", ["pe_a"])))
    assert d.selected_prompt_ids == ("pe_a",)


def test_unknown_target_aborts():
    with pytest.raises(ValueError):
        ROUTER.route(ctx(), bank(), policy(rule("r1", ["pe_missing"])))


# ------------------------------ determinism -------------------------------- #
def test_deterministic_selection_order_and_tie_break():
    p = policy(rule("r_b", ["pe_b"], priority=10),
               rule("r_a", ["pe_a"], priority=10))  # same priority: rule_id order
    d1 = ROUTER.route(ctx(), bank(), p)
    d2 = ROUTER.route(ctx(), bank(), p)
    assert d1.selected_prompt_ids == ("pe_a", "pe_b")
    assert dumps_canonical(d1) == dumps_canonical(d2)


# -------------------------- query-agnostic default -------------------------- #
def test_question_rule_skipped_by_default():
    p = policy(rule("r_q", ["pe_a"], conditions={"question_contains": "color"}),
               rule("r_plain", ["pe_b"]))
    d = ROUTER.route(ctx(question="What color is the light?"), bank(), p)
    assert d.selected_prompt_ids == ("pe_b",)
    assert d.decision_payload["skipped_question_rules"] == ("r_q",)
    assert d.decision_payload["question_conditioned"] is False


def test_question_rule_active_when_enabled_in_configuration():
    p = policy(rule("r_q", ["pe_a"], priority=1,
                    conditions={"question_contains": "color"}),
               rule("r_plain", ["pe_b"], priority=10),
               configuration={"question_conditioned": True})
    d_hit = ROUTER.route(ctx(question="What COLOR is the light?"), bank(), p)
    assert d_hit.selected_prompt_ids == ("pe_a", "pe_b")
    d_miss = ROUTER.route(ctx(question="Who is speaking?"), bank(), p)
    assert d_miss.selected_prompt_ids == ("pe_b",)


def test_default_routing_independent_of_question():
    p = policy(rule("r1", ["pe_a"]))
    d1 = ROUTER.route(ctx(question=None), bank(), p)
    d2 = ROUTER.route(ctx(question="Completely different question?"), bank(), p)
    assert d1.selected_prompt_ids == d2.selected_prompt_ids == ("pe_a",)


def test_unsupported_question_condition_rejected():
    p = policy(rule("r_q", ["pe_a"], conditions={"question_length": 5}),
               configuration={"question_conditioned": True})
    with pytest.raises(ValueError):
        ROUTER.route(ctx(question="q?"), bank(), p)


# ------------------------------ decision logging ---------------------------- #
def test_routing_decision_log_round_trip(tmp_path):
    p = policy(rule("r1", ["pe_a", "pe_b"]))
    decisions = [ROUTER.route(ctx(), bank(), p),
                 ROUTER.route(SegmentContext(video_id="vid", segment_id="10_20",
                                             segment_features={"has_transcript": True}),
                              bank(), p)]
    path = str(tmp_path / "routing" / "decisions.jsonl")
    write_routing_decisions(decisions, path)
    loaded = read_routing_decisions(path)
    assert list(loaded) == decisions
    assert [dumps_canonical(d) for d in loaded] == \
           [dumps_canonical(d) for d in decisions]
