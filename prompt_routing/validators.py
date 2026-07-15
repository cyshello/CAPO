"""Stage 4.2 component validation (PHASE4_PROMPT_ROUTING.md Stage 4.2).

Cross-record checks that the Stage 4.1 schemas cannot express at construction
time because they span records: bank-level hash uniqueness, router-vs-bank
reference validity, and scaffold-policy attempts to smuggle contract-owned
fields. Validators return error tuples; persistence raises on non-empty
errors before any write happens (§12.1: invalid state is rejected before
expensive evaluation)."""

from __future__ import annotations

from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    RouterPolicySnapshot,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
)

# Fields owned by the immutable hard contract (§12.1). A scaffold POLICY that
# carries any of these in its free-form configuration is trying to modify the
# contract and must be rejected (§12.2: policy never touches the contract).
CONTRACT_OWNED_KEYS = frozenset({
    "contract_version",
    "required_placeholders",
    "required_sections",
    "output_schema",
    "forbidden_patterns",
    "max_prompt_tokens",
})


def validate_prompt_bank(bank: PromptBankSnapshot) -> tuple[str, ...]:
    errors: list[str] = []
    ids = {e.prompt_id for e in bank.entries}
    seen_hash: dict[str, str] = {}
    for e in bank.entries:
        if e.status == "active":
            other = seen_hash.get(e.prompt_hash)
            if other is not None:
                errors.append(
                    f"duplicate active prompt hash: {other!r} and {e.prompt_id!r} "
                    f"share prompt_hash {e.prompt_hash[:12]}")
            else:
                seen_hash[e.prompt_hash] = e.prompt_id
        unknown = set(e.conflicts_with) - ids
        if unknown:
            errors.append(f"entry {e.prompt_id!r} conflicts_with unknown entries: "
                          f"{sorted(unknown)}")
    return tuple(errors)


def validate_scaffold_policy(policy: ScaffoldPolicySnapshot) -> tuple[str, ...]:
    overlap = CONTRACT_OWNED_KEYS & set(policy.configuration)
    if overlap:
        return (f"scaffold policy {policy.scaffold_version!r} attempts to modify "
                f"contract-owned fields: {sorted(overlap)} (the hard contract is "
                f"not an optimization target)",)
    return ()


def validate_scaffold_contract(contract: ScaffoldContract) -> tuple[str, ...]:
    # Construction already enforces the contract's own invariants.
    return ()


def validate_router_policy(policy: RouterPolicySnapshot) -> tuple[str, ...]:
    # Standalone invariants live in the schema; bank-dependent checks are in
    # validate_router_against_bank.
    return ()


def validate_router_against_bank(router: RouterPolicySnapshot,
                                 bank: PromptBankSnapshot) -> tuple[str, ...]:
    """Router/bank compatibility (Stage 4.2 clarification #6): target and
    fallback existence, active status, and compatible selection limits.

    A retired target inside a DISABLED rule is tolerated (retiring an entry
    must not invalidate historical, switched-off rules); an enabled rule must
    only target active entries."""
    errors: list[str] = []
    entries = {e.prompt_id: e for e in bank.entries}
    for rule in router.rules:
        for target in rule.target_prompt_ids:
            entry = entries.get(target)
            if entry is None:
                errors.append(f"rule {rule.rule_id!r} targets unknown entry {target!r}")
            elif rule.enabled and entry.status != "active":
                errors.append(f"enabled rule {rule.rule_id!r} targets "
                              f"{entry.status} entry {target!r}")
    for target in router.fallback_prompt_ids:
        entry = entries.get(target)
        if entry is None:
            errors.append(f"fallback references unknown entry {target!r}")
        elif entry.status != "active":
            errors.append(f"fallback references {entry.status} entry {target!r}")
    if router.max_selected_entries > bank.max_selected_entries:
        errors.append(
            f"router max_selected_entries={router.max_selected_entries} exceeds "
            f"bank max_selected_entries={bank.max_selected_entries}")
    return tuple(errors)


def assert_router_compatible(router: RouterPolicySnapshot,
                             bank: PromptBankSnapshot) -> None:
    errors = validate_router_against_bank(router, bank)
    if errors:
        raise ValueError(
            f"router {router.router_version!r} incompatible with bank "
            f"{bank.bank_version!r}: " + "; ".join(errors))
