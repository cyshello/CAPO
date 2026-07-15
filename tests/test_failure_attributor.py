"""Stage 4.9 deterministic component-attribution tests."""

from dataclasses import replace

import pytest

from surrogate_rollout.optimization.failure_attributor import (
    AttributionError,
    DeterministicFailureAttributor,
)
from surrogate_rollout.optimization.policies.mock_feedback import MockFeedbackGenerator
from surrogate_rollout.optimization.schemas import CounterfactualEvidence


def evidence(*, traces=True, legacy=False, scaffold_fact=False):
    trace = {"0_10": {"selected_prompt_ids": ["pe"]}} if traces else {}
    return CounterfactualEvidence(
        evidence_id="ev", video_id="v", question_id="q", ground_truth="A",
        baseline_bank_version=("legacy_phase2_3" if legacy else "bank_v0001"),
        baseline_router_version=("legacy_phase2_3" if legacy else "router_v0001"),
        baseline_scaffold_version=("legacy_phase2_3" if legacy else "scaffold_v0001"),
        candidate_bank_version=("legacy_phase2_3" if legacy else "bank_v0002"),
        candidate_router_version=("legacy_phase2_3" if legacy else "router_v0002"),
        candidate_scaffold_version=("legacy_phase2_3" if legacy else "scaffold_v0002"),
        baseline_selected_prompt_ids={"0_10": ("pe",)},
        candidate_selected_prompt_ids={"0_10": ("pe",)},
        baseline_composed_prompt_refs={"0_10": "a"},
        candidate_composed_prompt_refs={"0_10": "b"},
        baseline_answer="B", candidate_answer="B", baseline_score=0.0,
        candidate_score=0.0, score_delta=0.0,
        outcome_transition="wrong_to_wrong", segment_ids=("0_10",),
        caption_differences=(), baseline_trajectory_refs=(),
        candidate_trajectory_refs=(), rollout_artifact_refs={},
        metadata={"missing_trace_sides": (() if traces else ("baseline", "candidate")),
                  "baseline_composition_traces": trace,
                  "candidate_composition_traces": trace,
                  "scaffold_failure_evidence": scaffold_fact},
    )


def feedback(item, targets, failure_modes=("mock_targeted_failure",)):
    batch = MockFeedbackGenerator(
        {item.evidence_id: tuple(targets)}).generate([item], ())
    changed = replace(batch.items[0], failure_modes=tuple(failure_modes))
    return replace(batch, items=(changed,))


@pytest.mark.parametrize("target", ["prompt_bank", "router"])
def test_prompt_bank_and_router_attribution(target):
    item = evidence()
    result = DeterministicFailureAttributor().attribute(
        [item], feedback(item, (target,)))[0]
    assert result.targets == (target,)
    assert getattr(result, f"{target}_issues")


def test_scaffold_attribution_requires_traces_and_explicit_failure():
    item = evidence(scaffold_fact=True)
    result = DeterministicFailureAttributor().attribute(
        [item], feedback(item, ("scaffold",),
                         ("selected_instruction_omitted",)))[0]
    assert result.targets == ("scaffold",)
    assert result.scaffold_issues == ("selected_instruction_omitted",)


def test_scaffold_label_without_composition_fact_is_insufficient():
    item = evidence()
    result = DeterministicFailureAttributor().attribute(
        [item], feedback(item, ("scaffold",),
                         ("selected_instruction_omitted",)))[0]
    assert result.targets == ("insufficient_evidence",)
    assert result.scaffold_issues == ()


def test_multiple_attribution_is_not_default():
    item = evidence()
    result = DeterministicFailureAttributor().attribute(
        [item], feedback(item, ("prompt_bank", "router")))[0]
    assert result.targets == ("multiple",)
    assert result.prompt_bank_issues and result.router_issues


def test_legacy_missing_traces_prevents_scaffold_inference():
    item = evidence(traces=False, legacy=True)
    result = DeterministicFailureAttributor().attribute(
        [item], feedback(item, ("scaffold",),
                         ("selected_instruction_omitted",)))[0]
    assert result.targets == ("insufficient_evidence",)
    assert result.scaffold_issues == ()
    assert result.confidence <= 0.4


def test_missing_evidence_reference_rejected():
    item = evidence()
    batch = feedback(item, ("router",))
    batch = replace(batch, items=(replace(
        batch.items[0], evidence_ids=("missing",)),))
    with pytest.raises(AttributionError, match="evidence"):
        DeterministicFailureAttributor().attribute([item], batch)


def test_rationale_text_does_not_control_attribution():
    item = evidence()
    batch = feedback(item, ("prompt_bank",))
    batch = replace(batch, items=(replace(
        batch.items[0], rationale="This prose claims a scaffold failure."),))
    result = DeterministicFailureAttributor().attribute([item], batch)[0]
    assert result.targets == ("prompt_bank",)
