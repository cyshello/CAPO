import json
from pathlib import Path

import pytest

from surrogate_rollout.captioning.history_aware_baseline import (
    HistoryAwareBaselineCaptionViewBuilder,
    HistoryAwareSegmentCaptioner,
)
from surrogate_rollout.evaluation.dvd_qa import run_dvd_qa
from surrogate_rollout.optimization.baseline_phase import BaselinePhaseRunner
from surrogate_rollout.optimization.confirmation_evaluator import (
    HistoryAwareDVDConfirmationEvaluator,
)
from surrogate_rollout.optimization.final_iteration import Checkpoint3EOrchestrator
from surrogate_rollout.optimization.interventional_feedback import PropertyFeedbackRunner
from surrogate_rollout.optimization.policies.codex_feedback import (
    OpenAIStructuredFeedbackProvider,
)
from surrogate_rollout.optimization.policies.property_proposal import (
    MultiPropertyProposalPolicy,
    OpenAIPropertyProposalProvider,
)
from surrogate_rollout.optimization.property_intervention import (
    PropertyInterventionBatchRunner,
)
from surrogate_rollout.optimization.startup_models import (
    StartupModelValidationError,
    build_startup_model_manifest,
    log_startup_models,
)
from surrogate_rollout.prompt_routing.policies.history_aware_vlm_router import (
    HistoryAwareVLMRouter,
)


class LocalVLM:
    def caption(self, *_args, **_kwargs):
        raise AssertionError("startup validation must not invoke the VLM")


def _components(tmp_path: Path):
    vlm = LocalVLM()
    router = HistoryAwareVLMRouter(vlm, model_id="Qwen/Qwen2.5-VL-7B-Instruct")
    captioner = HistoryAwareSegmentCaptioner(
        vlm, caption_model_id="Qwen/Qwen2.5-VL-7B-Instruct")
    builder = HistoryAwareBaselineCaptionViewBuilder(
        router=router, segment_captioner=captioner)
    baseline = BaselinePhaseRunner(
        proposal_policy=MultiPropertyProposalPolicy(
            response_provider=OpenAIPropertyProposalProvider(model="gpt-4o")),
        history_aware_builder=builder)
    intervention = PropertyInterventionBatchRunner(segment_captioner=captioner)
    feedback = PropertyFeedbackRunner(
        response_provider=OpenAIStructuredFeedbackProvider(model="gpt-4o"))
    confirmation = HistoryAwareDVDConfirmationEvaluator(
        sample_loader=lambda _index: {}, history_aware_builder=builder,
        base_prompt_template="base", merge_prompt="merge",
        sample_source_identity="unit-source",
        cache_root=str(tmp_path / "cache"),
        cache_manifest_path=str(tmp_path / "cache_manifest.jsonl"))
    return baseline, intervention, feedback, confirmation


def test_real_startup_manifest_lists_all_models_and_is_immutable(tmp_path, capsys):
    components = _components(tmp_path)
    manifest = build_startup_model_manifest(
        iteration_id="iter-1", baseline_runner=components[0],
        intervention_runner=components[1], feedback_runner=components[2],
        confirmation_evaluator=components[3])

    assert tuple(manifest["models"]) == (
        "router", "captioner", "property_proposer", "feedback_provider",
        "downstream_qa")
    assert all(item["real_model"] for item in manifest["models"].values())
    assert manifest["models"]["property_proposer"]["model"] == "gpt-4o"
    assert manifest["models"]["feedback_provider"]["model"] == "gpt-4o"
    path = log_startup_models(str(tmp_path / "run"), manifest)
    components[2].response_provider.call_count = 1
    resumed_manifest = build_startup_model_manifest(
        iteration_id="iter-1", baseline_runner=components[0],
        intervention_runner=components[1], feedback_runner=components[2],
        confirmation_evaluator=components[3])
    assert resumed_manifest == manifest
    log_startup_models(str(tmp_path / "run"), manifest)
    assert json.loads(Path(path).read_text()) == manifest
    assert capsys.readouterr().out.count("STARTUP_MODELS ") == 2

    conflicting = dict(manifest, iteration_id="iter-2")
    with pytest.raises(StartupModelValidationError, match="conflicts"):
        log_startup_models(str(tmp_path / "run"), conflicting)


def test_mock_feedback_is_rejected_before_any_provider_call(tmp_path):
    baseline, intervention, feedback, confirmation = _components(tmp_path)

    class MockFeedbackRunner:
        response_provider = lambda self, _request: "{}"

    with pytest.raises(StartupModelValidationError, match="feedback_provider"):
        build_startup_model_manifest(
            iteration_id="iter-1", baseline_runner=baseline,
            intervention_runner=intervention,
            feedback_runner=MockFeedbackRunner(),
            confirmation_evaluator=confirmation)
    assert feedback.response_provider.call_count == 0

    runner = Checkpoint3EOrchestrator(
        baseline_runner=baseline, intervention_runner=intervention,
        feedback_runner=MockFeedbackRunner(),
        confirmation_evaluator=confirmation)
    with pytest.raises(StartupModelValidationError, match="feedback_provider"):
        runner._startup_models(
            iteration_id="iter-1", output_dir=str(tmp_path / "blocked-run"))
    assert not (tmp_path / "blocked-run").exists()


def test_non_openai_proposer_and_qa_double_are_rejected(tmp_path):
    baseline, intervention, feedback, confirmation = _components(tmp_path)

    class SavedProvider:
        model = "saved-response"

        def metadata(self):
            return {"provider": "saved_fixture", "model": self.model,
                    "real_model": False}

    baseline.proposal_policy = MultiPropertyProposalPolicy(
        response_provider=SavedProvider())
    with pytest.raises(StartupModelValidationError, match="property_proposer"):
        build_startup_model_manifest(
            iteration_id="iter-1", baseline_runner=baseline,
            intervention_runner=intervention, feedback_runner=feedback,
            confirmation_evaluator=confirmation)

    baseline, intervention, feedback, confirmation = _components(tmp_path)
    baseline.qa_fn = lambda **_kwargs: None
    with pytest.raises(StartupModelValidationError, match="baseline downstream QA"):
        build_startup_model_manifest(
            iteration_id="iter-1", baseline_runner=baseline,
            intervention_runner=intervention, feedback_runner=feedback,
            confirmation_evaluator=confirmation)
