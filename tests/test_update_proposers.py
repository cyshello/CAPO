"""Stage 4.10 independent preview-only proposer tests."""

import json
from pathlib import Path

import pytest

from surrogate_rollout.optimization.prompt_bank_update_proposer import (
    BANK_PAYLOAD_KEYS,
    DeterministicPromptBankUpdateProposer,
)
from surrogate_rollout.optimization.router_update_proposer import (
    ROUTER_OPERATION_KEYS,
    DeterministicRouterUpdateProposer,
)
from surrogate_rollout.optimization.scaffold_update_proposer import (
    DeterministicScaffoldUpdateProposer,
)
from surrogate_rollout.optimization.schemas import (
    FailureAttribution,
    FeedbackBatch,
    FeedbackItem,
    MetaKnowledgeItem,
)
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


FIXTURE = Path(__file__).parents[1] / "prompt_routing" / "fixtures" / \
    "stage4_7_components.json"


@pytest.fixture
def components():
    data = json.loads(FIXTURE.read_text())
    return (
        prompt_bank_from_json(data["prompt_bank"]),
        router_policy_from_json(data["router_policy"]),
        scaffold_policy_from_json(data["scaffold_policy"]),
        scaffold_contract_from_json(data["scaffold_contract"]),
    )


def feedback(component, *, confidence=0.9, traits=None, desired=None):
    feedback_id = f"feedback_{component}"
    attribution_id = f"attribution_{component}"
    issues = (f"{component}_failure",)
    attribution = FailureAttribution(
        attribution_id=attribution_id, evidence_ids=("evidence_1",),
        targets=(component,),
        prompt_bank_issues=(issues if component == "prompt_bank" else ()),
        router_issues=(issues if component == "router" else ()),
        scaffold_issues=(issues if component == "scaffold" else ()),
        supporting_facts=("saved structured fixture",), confidence=confidence,
        rationale="fixture",
    )
    item = FeedbackItem(
        feedback_id=feedback_id, evidence_ids=("evidence_1",),
        attribution_id=attribution_id, target_components=(component,),
        failure_modes=issues, successful_behaviors=(),
        desired_behaviors=(desired or f"improve {component} behavior",),
        avoid_behaviors=(), applicable_segment_traits=traits or {},
        confidence=confidence, rationale="fixture",
    )
    return FeedbackBatch(
        feedback_policy="deterministic_fixture",
        feedback_policy_version="fixture_v0001",
        input_evidence_ids=("evidence_1",), attributions=(attribution,),
        items=(item,), raw_response_artifact=None, parse_errors=(),
    )


def confirmed_scaffold_knowledge():
    return MetaKnowledgeItem(
        knowledge_id="knowledge_scaffold", knowledge_type="composition_pattern",
        condition={"failure": "instruction_loss"},
        principle="group related selected instructions",
        positive_support_ids=("e1", "e2", "e3"), negative_support_ids=(),
        distinct_video_ids=("v1", "v2", "v3"), scope="global_scaffold",
        confidence=0.9, status="confirmed", provenance={"fixture": True},
    )


def test_bank_add_entry_proposal(components):
    bank, _, _, _ = components
    proposal = DeterministicPromptBankUpdateProposer().propose(
        prompt_bank=bank,
        feedback=feedback("prompt_bank", traits={"text_visible": True}),
        meta_knowledge=(),
    )
    assert proposal.is_valid is True
    assert proposal.operations[0].operation_type == "add_entry"
    assert proposal.operations[0].target_ids == ()
    assert proposal.provenance["preview_only"] is True


def test_bank_revise_entry_proposal(components):
    bank, _, _, _ = components
    proposal = DeterministicPromptBankUpdateProposer().propose(
        prompt_bank=bank,
        feedback=feedback(
            "prompt_bank", traits={"target_prompt_id": "pe_text"}),
        meta_knowledge=(),
    )
    operation = proposal.operations[0]
    assert operation.operation_type == "revise_entry"
    assert operation.target_ids == ("pe_text",)


def test_router_rule_proposal(components):
    bank, router, _, _ = components
    proposal = DeterministicRouterUpdateProposer().propose(
        prompt_bank=bank, router_policy=router,
        feedback=feedback("router", traits={
            "routing_conditions": {"text_visible": True},
            "target_prompt_ids": ("pe_text",),
        }),
        meta_knowledge=(),
    )
    operation = proposal.operations[0]
    assert proposal.is_valid is True
    assert operation["operation_type"] == "add_rule"
    assert operation["conditions"] == {"text_visible": True}
    assert operation["target_prompt_ids"] == ("pe_text",)


def test_scaffold_proposal_when_enabled(components):
    _, _, scaffold, contract = components
    proposal = DeterministicScaffoldUpdateProposer().propose(
        scaffold_policy=scaffold, scaffold_contract=contract,
        feedback=feedback("scaffold", desired="group related instructions"),
        meta_knowledge=(confirmed_scaffold_knowledge(),), enabled=True,
    )
    assert proposal.is_valid is True
    assert proposal.skipped_reason == "none"
    assert proposal.candidate_policy.parent_scaffold_version == \
        scaffold.scaffold_version
    assert proposal.candidate_policy.scaffold_version == "scaffold_v0003"


def test_scaffold_disabled_skips_before_policy_call(components):
    _, _, scaffold, contract = components
    calls = []

    def policy(*args):
        calls.append(args)
        return {"ordering_policy": "changed"}

    proposal = DeterministicScaffoldUpdateProposer(
        proposal_policy=policy).propose(
            scaffold_policy=scaffold, scaffold_contract=contract,
            feedback=feedback("scaffold"),
            meta_knowledge=(confirmed_scaffold_knowledge(),), enabled=False)
    assert proposal.skipped_reason == "disabled_by_config"
    assert proposal.candidate_policy is None
    assert proposal.is_valid is True
    assert calls == []


def test_hard_contract_modification_rejected(components):
    _, _, scaffold, contract = components
    proposal = DeterministicScaffoldUpdateProposer(
        proposal_policy=lambda *args: {"required_sections": ("changed",)},
    ).propose(
        scaffold_policy=scaffold, scaffold_contract=contract,
        feedback=feedback("scaffold"),
        meta_knowledge=(confirmed_scaffold_knowledge(),), enabled=True)
    assert proposal.is_valid is False
    assert proposal.candidate_policy is None
    assert "contract-owned" in proposal.validation_errors[0]
    assert contract.required_sections == ()


def test_insufficient_scaffold_support_does_not_call_policy(components):
    _, _, scaffold, contract = components
    calls = []
    proposal = DeterministicScaffoldUpdateProposer(
        proposal_policy=lambda *args: calls.append(args) or {},
    ).propose(
        scaffold_policy=scaffold, scaffold_contract=contract,
        feedback=feedback("scaffold"), meta_knowledge=(), enabled=True)
    assert proposal.skipped_reason == "insufficient_support"
    assert proposal.candidate_policy is None
    assert calls == []


def test_low_confidence_feedback_generates_no_bank_operation(components):
    bank, _, _, _ = components
    proposal = DeterministicPromptBankUpdateProposer().propose(
        prompt_bank=bank, feedback=feedback("prompt_bank", confidence=0.2),
        meta_knowledge=())
    assert proposal.is_valid is False
    assert proposal.operations == ()
    assert "insufficient" in proposal.validation_errors[0]


def test_mock_proposals_are_deterministic(components):
    bank, router, scaffold, contract = components
    bank_feedback = feedback("prompt_bank")
    router_feedback = feedback("router")
    scaffold_feedback = feedback("scaffold")
    knowledge = (confirmed_scaffold_knowledge(),)
    proposers = (
        lambda: DeterministicPromptBankUpdateProposer().propose(
            prompt_bank=bank, feedback=bank_feedback, meta_knowledge=()),
        lambda: DeterministicRouterUpdateProposer().propose(
            prompt_bank=bank, router_policy=router,
            feedback=router_feedback, meta_knowledge=()),
        lambda: DeterministicScaffoldUpdateProposer().propose(
            scaffold_policy=scaffold, scaffold_contract=contract,
            feedback=scaffold_feedback, meta_knowledge=knowledge, enabled=True),
    )
    for propose in proposers:
        assert dumps_canonical(propose()) == dumps_canonical(propose())


def test_proposers_do_not_mutate_inputs(components):
    bank, router, scaffold, contract = components
    originals = dumps_canonical(components)
    DeterministicPromptBankUpdateProposer().propose(
        prompt_bank=bank, feedback=feedback("prompt_bank"), meta_knowledge=())
    DeterministicRouterUpdateProposer().propose(
        prompt_bank=bank, router_policy=router,
        feedback=feedback("router"), meta_knowledge=())
    DeterministicScaffoldUpdateProposer().propose(
        scaffold_policy=scaffold, scaffold_contract=contract,
        feedback=feedback("scaffold"),
        meta_knowledge=(confirmed_scaffold_knowledge(),), enabled=True)
    assert dumps_canonical(components) == originals


def test_no_cross_component_operation_leakage(components):
    bank, router, scaffold, contract = components
    bank_op = DeterministicPromptBankUpdateProposer().propose(
        prompt_bank=bank, feedback=feedback("prompt_bank"),
        meta_knowledge=()).operations[0]
    router_op = DeterministicRouterUpdateProposer().propose(
        prompt_bank=bank, router_policy=router,
        feedback=feedback("router"), meta_knowledge=()).operations[0]
    scaffold_proposal = DeterministicScaffoldUpdateProposer().propose(
        scaffold_policy=scaffold, scaffold_contract=contract,
        feedback=feedback("scaffold"),
        meta_knowledge=(confirmed_scaffold_knowledge(),), enabled=True)
    assert set(bank_op.payload) <= BANK_PAYLOAD_KEYS
    assert not ({"conditions", "target_prompt_ids", "ordering_policy"}
                & set(bank_op.payload))
    assert dumps_canonical(bank_op.payload).find("scaffold") == -1
    assert set(router_op) <= ROUTER_OPERATION_KEYS
    assert not ({"prompt_text", "composition_instruction"} & set(router_op))
    assert scaffold_proposal.candidate_policy.configuration == \
        scaffold.configuration

    malicious_bank = DeterministicPromptBankUpdateProposer().propose(
        prompt_bank=bank, feedback=feedback("prompt_bank", traits={
            "text_visible": True,
            "routing_conditions": {"text_visible": True},
            "target_prompt_ids": ("pe_text",),
        }), meta_knowledge=())
    assert malicious_bank.operations[0].payload["applicability_traits"] == {
        "text_visible": True}

    malicious_router = DeterministicRouterUpdateProposer().propose(
        prompt_bank=bank, router_policy=router,
        feedback=feedback("router", traits={
            "routing_conditions": {"prompt_text": "foreign"},
            "target_prompt_ids": ("pe_text",),
        }), meta_knowledge=())
    assert malicious_router.is_valid is False
    assert malicious_router.operations == ()
