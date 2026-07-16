"""Stage 4.13 fixed-scaffold iteration boundary tests without real calls."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from surrogate_rollout.optimization.feedback_generator import feedback_batch_from_json
from surrogate_rollout.optimization.policies.codex_feedback import (
    CodexStructuredFeedbackProvider,
    OpenAIStructuredFeedbackProvider,
)
from surrogate_rollout.optimization.real_fixed_scaffold_iteration import (
    RealEvaluationResult,
    RealFixedScaffoldIterationRunner,
    evaluate_routed_states,
)
from surrogate_rollout.optimization.schemas import (
    CounterfactualEvidence,
    NoChangeEvaluationResult,
)
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.schemas import (
    Phase4Config,
    Phase4OptimizationConfig,
)
from surrogate_rollout.schemas import QAExample


ROOT = Path(__file__).parents[1]
COMPONENTS = ROOT / "prompt_routing/fixtures/stage4_7_components.json"
FEEDBACK = ROOT / "optimization/fixtures/stage4_10_feedback.json"


def evidence(evidence_id, component, video):
    traces = {"0_10": {"selected_prompt_ids": ["pe_default"]}}
    return CounterfactualEvidence(
        evidence_id=evidence_id, video_id=video, question_id=evidence_id,
        ground_truth="A", baseline_bank_version="bank_v0002",
        baseline_router_version="router_v0002",
        baseline_scaffold_version="scaffold_v0002",
        candidate_bank_version="bank_v0002",
        candidate_router_version="router_v0002",
        candidate_scaffold_version="scaffold_v0002",
        baseline_selected_prompt_ids={"0_10": ("pe_default",)},
        candidate_selected_prompt_ids={"0_10": ("pe_default",)},
        baseline_composed_prompt_refs={"0_10": "before"},
        candidate_composed_prompt_refs={"0_10": "after"},
        baseline_answer="B", candidate_answer="A", baseline_score=0.0,
        candidate_score=1.0, score_delta=1.0,
        outcome_transition="wrong_to_correct", segment_ids=("0_10",),
        caption_differences=(), baseline_trajectory_refs=(),
        candidate_trajectory_refs=(), rollout_artifact_refs={},
        metadata={
            "baseline_composition_traces": traces,
            "candidate_composition_traces": traces,
            "missing_trace_sides": (),
            "scaffold_failure_evidence": component == "scaffold",
        })


def components():
    data = json.loads(COMPONENTS.read_text())
    return (
        prompt_bank_from_json(data["prompt_bank"]),
        router_policy_from_json(data["router_policy"]),
        scaffold_policy_from_json(data["scaffold_policy"]),
        scaffold_contract_from_json(data["scaffold_contract"]),
    )


def config(**overrides):
    values = {"optimize_prompt_bank": True, "optimize_router": True,
              "optimize_scaffold": False, "max_iterations": 1}
    values.update(overrides)
    return Phase4Config(
        dry_run=True, commit=False,
        optimization=Phase4OptimizationConfig(**values))


def example(question_id):
    return QAExample(
        video_id="validation_video", question_id=question_id,
        question="Question?", ground_truth="A", provider_index=1,
        options=("A", "B"))


def test_fixed_scaffold_runner_emits_skip_and_one_evaluation(tmp_path):
    bank, router, scaffold, contract = components()
    feedback_data = json.loads(FEEDBACK.read_text())["feedback"]
    calls = []
    verifies = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        return RealEvaluationResult(
            incumbent_scores={"confirmation": 0.0, "regression": 1.0},
            candidate_scores={"confirmation": 0.0, "regression": 1.0},
            incumbent_errors={}, candidate_errors={},
            incumbent_artifact="incumbent.jsonl",
            candidate_artifact="candidate.jsonl",
            comparison_artifact="comparison.json", timing={"wall": 0.0})

    result = RealFixedScaffoldIterationRunner(
        feedback_generator=type("Feedback", (), {
            "generate": lambda self, *_: feedback_batch_from_json(feedback_data),
        })(), evaluation_fn=evaluate).run(
            input_bank=bank, input_router=router, input_scaffold=scaffold,
            scaffold_contract=contract,
            evidence=(evidence("evidence_bank", "prompt_bank", "v1"),
                      evidence("evidence_router", "router", "v2"),
                      evidence("evidence_scaffold", "scaffold", "v3")),
            confirmation_examples=(example("confirmation"),),
            regression_examples=(example("regression"),), qa_samples=(),
            base_prompt_template="base", merge_prompt="merge",
            config=config(), output_dir=str(tmp_path), source_revision="abc",
            frozen_manifest_path="manifest.json",
            verify_frozen=lambda: verifies.append(True), gpu="0",
            dvd_max_iterations=1)
    scaffold_proposal = json.loads(Path(
        result.scaffold_proposal_artifact).read_text())
    assert scaffold_proposal["skipped_reason"] == "disabled_by_config"
    assert scaffold_proposal["candidate_policy"] is None
    assert len(calls) == 1
    assert len(verifies) == 3
    assert result.committed_scaffold_version is None


def test_stage4_14_preserves_scaffold_support_threshold(tmp_path):
    bank, router, scaffold, contract = components()
    feedback_data = json.loads(FEEDBACK.read_text())["feedback"]

    def evaluate(**kwargs):
        return RealEvaluationResult(
            incumbent_scores={"confirmation": 1.0, "regression": 1.0},
            candidate_scores={}, incumbent_errors={}, candidate_errors={},
            incumbent_artifact="incumbent.jsonl",
            candidate_artifact="no_change.json",
            comparison_artifact="comparison.json", timing={"incumbent": 0.0},
            no_change=NoChangeEvaluationResult(
                status="no_change",
                reason="equivalent_components_and_composed_prompts",
                component_equivalence={
                    "prompt_bank": True, "router": True,
                    "scaffold": True, "contract": True},
                incumbent_composed_prompt_hashes={"v": ("hash",)},
                candidate_composed_prompt_hashes={"v": ("hash",)},
                candidate_dvd_executed=False,
                eligible_for_validation=False,
                eligible_for_regression_evidence=False))

    result = RealFixedScaffoldIterationRunner(
        feedback_generator=type("Feedback", (), {
            "generate": lambda self, *_: feedback_batch_from_json(feedback_data),
        })(), evaluation_fn=evaluate, optimize_scaffold=True,
        stage="4.14").run(
            input_bank=bank, input_router=router, input_scaffold=scaffold,
            scaffold_contract=contract,
            evidence=(evidence("evidence_bank", "prompt_bank", "v1"),
                      evidence("evidence_router", "router", "v2"),
                      evidence("evidence_scaffold", "scaffold", "v3")),
            confirmation_examples=(example("confirmation"),),
            regression_examples=(example("regression"),), qa_samples=(),
            base_prompt_template="base", merge_prompt="merge",
            config=config(optimize_scaffold=True), output_dir=str(tmp_path),
            source_revision="abc", frozen_manifest_path="manifest.json",
            verify_frozen=lambda: None, gpu="0", dvd_max_iterations=1)

    scaffold_proposal = json.loads(Path(
        result.scaffold_proposal_artifact).read_text())
    scaffold_review = json.loads(Path(
        result.review_artifacts["scaffold"]).read_text())
    assert scaffold_proposal["skipped_reason"] == "insufficient_support"
    assert scaffold_proposal["provenance"]["proposal_policy_calls"] == 0
    assert scaffold_review["decision"] == "defer"
    assert result.validation_artifacts == {}


@pytest.mark.parametrize("bad", [
    {"max_iterations": 2}, {"optimize_scaffold": True},
    {"optimize_prompt_bank": False}, {"optimize_router": False},
])
def test_stage4_13_configuration_is_exact(tmp_path, bad):
    bank, router, scaffold, contract = components()
    runner = RealFixedScaffoldIterationRunner(
        feedback_generator=object(), evaluation_fn=lambda **kwargs: None)
    with pytest.raises(ValueError, match="Stage 4.13"):
        runner.run(
            input_bank=bank, input_router=router, input_scaffold=scaffold,
            scaffold_contract=contract, evidence=(), confirmation_examples=(),
            regression_examples=(), qa_samples=(), base_prompt_template="base",
            merge_prompt="merge", config=config(**bad),
            output_dir=str(tmp_path), source_revision="abc",
            frozen_manifest_path="manifest", verify_frozen=lambda: None,
            gpu="0", dvd_max_iterations=1)


def test_codex_feedback_provider_allows_one_real_call(monkeypatch):
    expected = json.dumps({"feedback": "response"})
    fake_module = type("Module", (), {
        "codex_infer": staticmethod(lambda prompt, model, timeout: expected),
    })
    monkeypatch.setitem(__import__("sys").modules, "codex_infer", fake_module)
    provider = CodexStructuredFeedbackProvider(model="test-model")
    request = json.dumps({"evidence": [{"evidence_id": "e1"}],
                          "meta_knowledge": []})
    assert provider(request) == expected
    assert provider.metadata()["call_count"] == 1
    with pytest.raises(RuntimeError, match="exactly one"):
        provider(request)


def test_openai_feedback_provider_defaults_to_gpt4o_and_allows_one_call(
    monkeypatch,
):
    import surrogate_rollout.optimization.policies.codex_feedback as module

    calls = []
    monkeypatch.setattr(module, "_openai_chat", lambda prompt, **kwargs: (
        calls.append((prompt, kwargs)) or json.dumps({"feedback": "response"})))
    provider = OpenAIStructuredFeedbackProvider(api_key="test-key")
    request = json.dumps({"evidence": [{"evidence_id": "e1"}],
                          "meta_knowledge": []})
    assert provider(request) == json.dumps({"feedback": "response"})
    assert calls[0][1] == {"model": "gpt-4o", "api_key": "test-key"}
    assert provider.metadata()["provider"] == "openai_api"
    assert provider.metadata()["model"] == "gpt-4o"
    with pytest.raises(RuntimeError, match="only one"):
        provider(request)


def test_equivalent_candidate_skips_repeated_dvd_and_records_no_change(
    tmp_path, monkeypatch,
):
    import surrogate_rollout.optimization.real_fixed_scaffold_iteration as module

    bank, router, scaffold, contract = components()
    dvd_calls = []
    monkeypatch.setattr(module, "build_clip_index", lambda *_: (("0_10", {}),))
    monkeypatch.setattr(module, "ensure_backend", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "run_offline_dry_run", lambda **kwargs:
                        SimpleNamespace(composed_prompts=(SimpleNamespace(
                            segment_id="0_10", prompt_hash="same"),)))
    monkeypatch.setattr(module, "RoutedCaptionViewBuilder", lambda: SimpleNamespace(
        build=lambda **kwargs: SimpleNamespace(
            captions_path="captions.json", database_path="database.json",
            routed_view_path="routed_view.json", caption_call_count=1,
            caption_cache_hits=0, caption_seconds=0.1)))

    def fake_dvd(**kwargs):
        dvd_calls.append(kwargs)
        return SimpleNamespace(
            score=1.0, prediction="A", parsed_answer="A", ground_truth="A",
            errors=(), token_usage={}, latency_seconds=0.2,
            captions_path="captions.json", database_path="database.json")

    monkeypatch.setattr(module, "run_dvd_qa", fake_dvd)
    qa = example("confirmation")
    result = evaluate_routed_states(
        incumbent_bank=bank, incumbent_router=router,
        candidate_bank=bank, candidate_router=router,
        incumbent_scaffold=scaffold, candidate_scaffold=scaffold,
        incumbent_contract=contract, candidate_contract=contract,
        qa_samples=((qa, {"sample_id": qa.video_id}),),
        base_prompt_template="base", merge_prompt="merge",
        phase4_config=config(optimize_scaffold=True),
        output_dir=str(tmp_path), source_revision="abc", gpu="0",
        dvd_max_iterations=1)

    assert isinstance(result.no_change, NoChangeEvaluationResult)
    assert result.no_change.candidate_dvd_executed is False
    assert result.no_change.eligible_for_validation is False
    assert result.no_change.eligible_for_regression_evidence is False
    assert result.candidate_scores == {}
    assert len(dvd_calls) == 1
    assert json.loads(Path(result.candidate_artifact).read_text())["status"] == \
        "no_change"


def test_changed_component_does_not_trigger_equivalence_guard(tmp_path, monkeypatch):
    import surrogate_rollout.optimization.real_fixed_scaffold_iteration as module

    bank, router, scaffold, contract = components()
    candidate_bank = replace(
        bank, bank_version="bank_v0003", parent_bank_version=bank.bank_version)
    dvd_calls = []
    monkeypatch.setattr(module, "build_clip_index", lambda *_: (("0_10", {}),))
    monkeypatch.setattr(module, "ensure_backend", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "run_offline_dry_run", lambda **kwargs:
                        SimpleNamespace(composed_prompts=(SimpleNamespace(
                            segment_id="0_10", prompt_hash="same"),)))
    monkeypatch.setattr(module, "RoutedCaptionViewBuilder", lambda: SimpleNamespace(
        build=lambda **kwargs: SimpleNamespace(
            captions_path="captions.json", database_path="database.json",
            routed_view_path="routed_view.json", caption_call_count=1,
            caption_cache_hits=0, caption_seconds=0.1)))

    def fake_dvd(**kwargs):
        dvd_calls.append(kwargs)
        return SimpleNamespace(
            score=1.0, prediction="A", parsed_answer="A", ground_truth="A",
            errors=(), token_usage={}, latency_seconds=0.2,
            captions_path="captions.json", database_path="database.json")

    monkeypatch.setattr(module, "run_dvd_qa", fake_dvd)
    qa = example("confirmation")
    result = evaluate_routed_states(
        incumbent_bank=bank, incumbent_router=router,
        candidate_bank=candidate_bank, candidate_router=router,
        incumbent_scaffold=scaffold, candidate_scaffold=scaffold,
        incumbent_contract=contract, candidate_contract=contract,
        qa_samples=((qa, {"sample_id": qa.video_id}),),
        base_prompt_template="base", merge_prompt="merge",
        phase4_config=config(optimize_scaffold=True),
        output_dir=str(tmp_path), source_revision="abc", gpu="0",
        dvd_max_iterations=1)

    assert result.no_change is None
    assert len(dvd_calls) == 2


def test_real_iteration_does_not_import_canonical_store():
    source = (ROOT / "optimization/real_fixed_scaffold_iteration.py").read_text()
    assert "ComponentSnapshotStore" not in source
    assert "write_current" not in source
