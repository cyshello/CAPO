"""Deterministic multi-entry rule-based router (Stage 4.3, PHASE4 §13).

Algorithm (every step deterministic):

1. abort on router/bank incompatibility (§22) — this also guarantees every
   enabled rule targets an existing, active entry (§13.8);
2. evaluate enabled rules in (priority ascending, rule_id) order — lower
   priority number means evaluated (and therefore selected) earlier;
3. a rule matches when every condition key equals the segment's value
   (`segment_features` first, then `metadata`); question-dependent conditions
   (keys starting with "question") are only evaluated when the policy
   configuration sets `question_conditioned: true` — otherwise the rule is
   skipped and recorded, keeping routing query-agnostic by default (§13.10);
4. matching targets are collected in rule order, deduplicated on first
   appearance (§13.3);
5. a candidate conflicting (via `conflicts_with`, either direction) with an
   already-selected entry is dropped and recorded — earlier selection wins
   (§13.4), never silently (§22);
6. the selection is truncated to the policy's `max_selected_entries`
   (compatibility with the bank's limit is enforced in step 1); dropped IDs
   are recorded (§13.5);
7. when no rule selected anything, the configured fallback entries are
   appended (`used_fallback=True`, reason recorded) (§13.7);
8. `prompt_scores` covers every considered candidate — including ones later
   dropped by conflicts or budget: rule-matched candidates score 1.0,
   fallback entries 0.5. Scores are membership evidence for later router
   optimization, not calibrated probabilities.

The final RoutingDecision carries the complete drop/skip accounting in
`decision_payload`, so a decision is auditable without re-running the router.
"""

from __future__ import annotations

from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    RouterPolicySnapshot,
    RoutingDecision,
    RoutingRule,
    SegmentContext,
)
from surrogate_rollout.prompt_routing.validators import assert_router_compatible

_RULE_SCORE = 1.0
_FALLBACK_SCORE = 0.5


class RuleBasedPromptRouter:
    """PromptRouter implementation over RouterPolicySnapshot rules."""

    policy_type = "rule_based"

    def route(
        self,
        context: SegmentContext,
        prompt_bank: PromptBankSnapshot,
        router_policy: RouterPolicySnapshot,
    ) -> RoutingDecision:
        assert_router_compatible(router_policy, prompt_bank)
        question_conditioned = bool(
            router_policy.configuration.get("question_conditioned", False))

        entries = {e.prompt_id: e for e in prompt_bank.entries}
        matched_rule_ids: list[str] = []
        skipped_question_rules: list[str] = []
        selected: list[str] = []
        scores: dict[str, float] = {}
        conflicts_dropped: list[dict] = []

        ordered_rules = sorted(
            (r for r in router_policy.rules if r.enabled),
            key=lambda r: (r.priority, r.rule_id))

        for rule in ordered_rules:
            outcome = self._rule_matches(rule, context, question_conditioned)
            if outcome == "skipped_question":
                skipped_question_rules.append(rule.rule_id)
                continue
            if outcome != "matched":
                continue
            matched_rule_ids.append(rule.rule_id)
            for target in rule.target_prompt_ids:
                scores.setdefault(target, _RULE_SCORE)
                if target in selected:
                    continue                       # §13.3 dedup
                conflict = self._conflicts_with_selected(
                    entries, target, selected)
                if conflict is not None:
                    conflicts_dropped.append(
                        {"dropped": target, "kept": conflict,
                         "rule_id": rule.rule_id})
                    continue
                selected.append(target)

        budget_dropped: list[str] = []
        limit = router_policy.max_selected_entries
        if len(selected) > limit:
            budget_dropped = selected[limit:]
            selected = selected[:limit]

        used_fallback = False
        fallback_reason = None
        if not selected and router_policy.fallback_prompt_ids:
            for target in router_policy.fallback_prompt_ids:
                if target not in selected:
                    selected.append(target)
                    scores.setdefault(target, _FALLBACK_SCORE)
            selected = selected[:limit]
            used_fallback = True
            fallback_reason = ("no enabled rule selected any entry; "
                               "configured fallback entries appended")

        return RoutingDecision(
            video_id=context.video_id,
            segment_id=context.segment_id,
            bank_version=prompt_bank.bank_version,
            router_version=router_policy.router_version,
            selected_prompt_ids=tuple(selected),
            prompt_scores=scores,
            matched_rule_ids=tuple(matched_rule_ids),
            selection_order=tuple(selected),
            confidence=None,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
            decision_payload={
                "policy_type": self.policy_type,
                "question_conditioned": question_conditioned,
                "skipped_question_rules": skipped_question_rules,
                "conflicts_dropped": conflicts_dropped,
                "budget_dropped": budget_dropped,
            },
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _rule_matches(rule: RoutingRule, context: SegmentContext,
                      question_conditioned: bool) -> str:
        """Returns "matched", "unmatched", or "skipped_question"."""
        for key, expected in rule.conditions.items():
            if key.startswith("question"):
                if not question_conditioned:
                    return "skipped_question"      # query-agnostic default
                if key == "question_contains":
                    if not isinstance(expected, str) or \
                            expected.lower() not in (context.question or "").lower():
                        return "unmatched"
                    continue
                raise ValueError(
                    f"rule {rule.rule_id!r}: unsupported question condition "
                    f"{key!r} (only 'question_contains' is defined)")
            actual = context.segment_features.get(key)
            if actual is None:
                actual = context.metadata.get(key)
            if actual != expected:
                return "unmatched"
        return "matched"

    @staticmethod
    def _conflicts_with_selected(entries: dict, target: str,
                                 selected: list[str]) -> str | None:
        """First already-selected entry that conflicts with `target` (checked
        in both directions), or None."""
        target_conflicts = set(entries[target].conflicts_with)
        for kept in selected:
            if kept in target_conflicts or \
                    target in set(entries[kept].conflicts_with):
                return kept
        return None
