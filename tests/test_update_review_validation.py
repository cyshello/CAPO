"""Stage 4.11 review, candidate-preview, and saved-metric validation tests."""

import json
from pathlib import Path

import pytest

from surrogate_rollout.optimization.candidate_preview import CandidatePreviewBuilder
from surrogate_rollout.optimization.schemas import (
    MetaKnowledgeItem,
    PromptBankOperation,
    PromptBankUpdateProposal,
)
from surrogate_rollout.optimization.update_reviewer import (
    DeterministicUpdateReviewer,
)
from surrogate_rollout.optimization.update_validator import (
    DeterministicUpdateValidator,
)
from surrogate_rollout.prompt_routing.persistence import prompt_bank_from_json
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import QAExample


COMPONENT_FIXTURE = Path(__file__).parents[1] / "prompt_routing" / \
    "fixtures" / "stage4_7_components.json"


def bank_proposal(*, valid=True):
    operation = PromptBankOperation(
        operation_id="bank_op_fixture", operation_type="add_entry",
        target_ids=(), payload={
            "prompt_id": "pe_stage4_11",
            "prompt_text": "Describe visible causal transitions precisely.",
            "name": "Causal transitions",
            "description": "Synthetic Stage 4.11 preview entry.",
            "tags": ("causal",), "applicability_traits": {"causal": True},
            "parent_prompt_ids": (),
        }, source_feedback_ids=("feedback_1",), confidence=0.9)
    return PromptBankUpdateProposal(
        proposal_id="bank_proposal_fixture", input_bank_version="bank_v0002",
        operations=((operation,) if valid else ()),
        validation_errors=(() if valid else ("fixture invalid",)),
        is_valid=valid, provenance={"preview_only": True})


def knowledge(*, status="confirmed", knowledge_type="successful_behavior"):
    return MetaKnowledgeItem(
        knowledge_id=f"knowledge_{status}_{knowledge_type}",
        knowledge_type=knowledge_type,
        condition={"component": "prompt_bank"}, principle="fixture principle",
        positive_support_ids=("support_1",), negative_support_ids=(),
        distinct_video_ids=("proposal_video_1",), scope="local_prompt",
        confidence=0.9, status=status, provenance={"fixture": True})


def example(question_id, video_id):
    return QAExample(
        video_id=video_id, question_id=question_id, question="Question?",
        ground_truth="A", provider_index=0, options=("A", "B"))


def states(*, component="prompt_bank", confirmation_candidate=1.0,
           regression_candidate=1.0, support_videos=(), contract="contract_v0002",
           errors=None, proposal_video="proposal_video_1"):
    kind = {"prompt_bank": "bank", "router": "router",
            "scaffold": "scaffold"}[component]
    incumbent = {
        "version": f"{kind}_v0002", "contract_version": "contract_v0002",
        "scores": {"confirmation_1": 0.0, "regression_1": 1.0},
    }
    candidate = {
        "version": f"{kind}_v0003", "contract_version": contract,
        "proposal_example_ids": ("proposal_1",),
        "proposal_video_ids": (proposal_video,),
        "support_video_ids": support_videos,
        "scores": {
            "confirmation_1": confirmation_candidate,
            "regression_1": regression_candidate,
        },
        "errors": errors or {},
    }
    return incumbent, candidate


def validate(**state_options):
    incumbent, candidate = states(**state_options)
    return DeterministicUpdateValidator().validate(
        incumbent_state=incumbent, candidate_state=candidate,
        component=state_options.get("component", "prompt_bank"),
        confirmation_examples=(example("confirmation_1", "confirmation_video"),),
        regression_examples=(example("regression_1", "regression_video"),))


def test_reviewer_accepts_supported_proposal_for_validation():
    result = DeterministicUpdateReviewer().review(
        proposal=bank_proposal(), component="prompt_bank",
        meta_knowledge=(knowledge(),))
    assert result.decision == "accept_for_validation"


def test_reviewer_rejects_known_conflict():
    result = DeterministicUpdateReviewer().review(
        proposal=bank_proposal(), component="prompt_bank",
        meta_knowledge=(knowledge(knowledge_type="conflict"),))
    assert result.decision == "reject"
    assert result.conflicts == ("fixture principle",)


def test_reviewer_defers_one_weak_sample():
    result = DeterministicUpdateReviewer().review(
        proposal=bank_proposal(), component="prompt_bank", meta_knowledge=())
    assert result.decision == "defer"


def test_reviewer_requests_revision_for_candidate_conflict():
    result = DeterministicUpdateReviewer().review(
        proposal=bank_proposal(), component="prompt_bank",
        meta_knowledge=(knowledge(status="candidate", knowledge_type="conflict"),))
    assert result.decision == "revise"


def test_confirmation_improvement_is_accepted():
    result = validate()
    assert result.decision == "accept"
    assert result.candidate_metrics["confirmation_gain"] == pytest.approx(1.0)


def test_regression_is_rejected():
    result = validate(regression_candidate=0.0)
    assert result.decision == "reject"
    assert result.regressions == ("regression_1",)


def test_scaffold_rejected_for_insufficient_distinct_videos():
    result = validate(component="scaffold", support_videos=("v1", "v2"))
    assert result.decision == "reject"
    assert "insufficient distinct support videos" in " ".join(result.reasons)


def test_scaffold_rejected_for_contract_violation():
    result = validate(
        component="scaffold", support_videos=("v1", "v2", "v3"),
        contract="contract_v0003")
    assert result.decision == "reject"
    assert "fixed contract" in " ".join(result.reasons)


def test_saved_evaluation_failure_is_typed_and_has_priority():
    result = validate(
        component="scaffold", support_videos=("v1",),
        errors={"confirmation_1": "saved evaluator failure"})
    assert result.decision == "evaluation_failed"
    assert result.incumbent_metrics == {}


def test_candidate_preview_leaves_incumbent_untouched():
    data = json.loads(COMPONENT_FIXTURE.read_text())
    incumbent = prompt_bank_from_json(data["prompt_bank"])
    before = dumps_canonical(incumbent)
    preview = CandidatePreviewBuilder().build_prompt_bank(
        incumbent, bank_proposal())
    assert dumps_canonical(incumbent) == before
    assert preview.candidate_version == "bank_v0003"
    assert preview.candidate_state.parent_bank_version == incumbent.bank_version
    assert preview.preview_only is True
    assert preview.canonical_state_mutated is False


def test_fixed_metrics_produce_deterministic_decision():
    first = validate()
    second = validate()
    assert dumps_canonical(first) == dumps_canonical(second)


def test_proposal_and_confirmation_videos_must_be_separate():
    incumbent, candidate = states(proposal_video="shared_video")
    result = DeterministicUpdateValidator().validate(
        incumbent_state=incumbent, candidate_state=candidate,
        component="prompt_bank",
        confirmation_examples=(example("confirmation_1", "shared_video"),),
        regression_examples=(example("regression_1", "regression_video"),))
    assert result.decision == "defer"
    assert "proposal and confirmation videos overlap" in " ".join(result.reasons)
