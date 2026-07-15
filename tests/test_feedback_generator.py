"""Stage 4.9 feedback policy boundary tests."""

import os

import pytest

from surrogate_rollout.optimization.feedback_generator import FeedbackParseError
from surrogate_rollout.optimization.policies import (
    LLMFeedbackGenerator,
    MockFeedbackGenerator,
    RealFeedbackDisabledError,
)
from surrogate_rollout.optimization.schemas import (
    CounterfactualEvidence,
    FeedbackItem,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


def evidence(evidence_id="ev1"):
    traces = {"0_10": {"selected_prompt_ids": ["pe_default"]}}
    return CounterfactualEvidence(
        evidence_id=evidence_id, video_id="video1", question_id="qa1",
        ground_truth="A", baseline_bank_version="bank_v0001",
        baseline_router_version="router_v0001",
        baseline_scaffold_version="scaffold_v0001",
        candidate_bank_version="bank_v0002",
        candidate_router_version="router_v0002",
        candidate_scaffold_version="scaffold_v0002",
        baseline_selected_prompt_ids={"0_10": ("pe_default",)},
        candidate_selected_prompt_ids={"0_10": ("pe_default", "pe_text")},
        baseline_composed_prompt_refs={"0_10": "hash1"},
        candidate_composed_prompt_refs={"0_10": "hash2"},
        baseline_answer="B", candidate_answer="B", baseline_score=0.0,
        candidate_score=0.0, score_delta=0.0,
        outcome_transition="wrong_to_wrong", segment_ids=("0_10",),
        caption_differences=(), baseline_trajectory_refs=(),
        candidate_trajectory_refs=(), rollout_artifact_refs={},
        metadata={"missing_trace_sides": (),
                  "baseline_composition_traces": traces,
                  "candidate_composition_traces": traces},
    )


def test_deterministic_mock_output():
    generator = MockFeedbackGenerator()
    first = generator.generate([evidence()], ())
    second = generator.generate([evidence()], ())
    assert dumps_canonical(first) == dumps_canonical(second)
    assert first.items[0].target_components == ("insufficient_evidence",)
    assert first.raw_response_artifact is None


def test_invalid_confidence_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        FeedbackItem(
            feedback_id="f", evidence_ids=("e",), attribution_id="a",
            target_components=("router",), failure_modes=(),
            successful_behaviors=(), desired_behaviors=(), avoid_behaviors=(),
            applicable_segment_traits={}, confidence=1.01, rationale="",
        )


def test_real_feedback_is_disabled_without_calling_provider(tmp_path):
    called = []
    generator = LLMFeedbackGenerator(
        artifact_dir=str(tmp_path), enabled=False,
        response_provider=lambda request: called.append(request) or "{}",
    )
    with pytest.raises(RealFeedbackDisabledError, match="disabled"):
        generator.generate([evidence()], ())
    assert called == []


def test_malformed_model_output_is_preserved_and_rejected(tmp_path):
    generator = LLMFeedbackGenerator(artifact_dir=str(tmp_path))
    with pytest.raises(FeedbackParseError) as caught:
        generator.parse_saved_response("not-json")
    assert os.path.exists(caught.value.raw_response_artifact)
    assert open(caught.value.raw_response_artifact).read() == "not-json"


def test_valid_saved_response_preserves_raw_artifact(tmp_path):
    batch = MockFeedbackGenerator().generate([evidence()], ())
    raw = dumps_canonical(batch)
    parsed = LLMFeedbackGenerator(
        artifact_dir=str(tmp_path)).parse_saved_response(raw)
    assert parsed.raw_response_artifact == str(tmp_path / "raw_response.txt")
    assert open(parsed.raw_response_artifact).read() == raw
    assert dumps_canonical(parsed.items) == dumps_canonical(batch.items)
