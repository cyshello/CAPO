import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from surrogate_rollout.optimization.final_iteration import Checkpoint3EOrchestrator
from surrogate_rollout.optimization.llm_codebook_updater import (
    CANDIDATE_CODEBOOK_SCHEMA_VERSION,
    SYSTEM_PROMPT_VERSION,
    UPDATER_RESPONSE_SCHEMA_VERSION,
    LLMCodebookUpdaterConflictError,
    LLMCodebookUpdaterParseError,
    MemoryConditionedLLMCodebookUpdater,
    deterministic_property_id,
    parse_updater_plan,
)
from surrogate_rollout.optimization.property_memory import CompactPropertyMemoryRunner
from surrogate_rollout.prompt_routing.persistence import prompt_bank_from_json


ROOT = Path(__file__).parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return str(path.resolve())


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return str(path.resolve())


def bank():
    raw = json.loads((
        ROOT / "prompt_routing/fixtures/stage4_7_components.json").read_text())
    return prompt_bank_from_json(raw["prompt_bank"])


def example(example_id, *, video="v1", iteration=1, effect=None):
    value = {
        "example_id": example_id, "video_id": video,
        "iteration_ordinal": iteration, "summary": "bounded evidence",
        "artifact_refs": [],
    }
    if effect:
        value["effect"] = effect
    return value


def property_memory(property_id, text, *, positive=True, negative=()):
    return {
        "schema_version": "property_memory_v1", "property_id": property_id,
        "property_text": text, "why_created": "seed",
        "intended_behavior": text,
        "origin": {"kind": "seed_or_legacy", "artifact_refs": []},
        "positive_examples": ([example(f"pos-{property_id}")] if positive else []),
        "negative_examples": list(negative), "no_effect_examples": [],
        "routing_examples": {"positive": [], "negative": []},
        "aliases": [], "revision_history": [],
        "latest_decision": {"action": "seed_initialize", "reason": "seed",
                            "artifact_refs": []},
    }


def candidate_memory(candidate_id="cp-new", *, positive=True):
    return {
        "schema_version": "property_memory_v1", "candidate_property_id": candidate_id,
        "source_video_id": "v1",
        "property_text": "Track reusable object completion states.",
        "applicability": {
            "when": "Use when frames show a completion-state transition.",
            "positive_cues": ["A visible object reaches a completed state."],
            "negative_cues": [], "required_modalities": ["frames"],
        },
        "failure_analysis": "The generated description omitted completion.",
        "why_proposed": "Completion state was omitted.",
        "intended_reusable_behavior": "Track reusable object completion states.",
        "related_active_property_ids": [],
        "positive_examples": ([example("candidate-positive", effect="positive")]
                              if positive else []),
        "negative_examples": [], "mixed_examples": [],
        "no_effect_examples": ([example("candidate-none", effect="no_effect")]
                               if not positive else []),
        "artifact_refs": [], "status": "candidate",
        "promoted_to_property_id": None,
    }


def memory_snapshot(tmp_path, *, candidates=None, properties=None, schema="property_memory_v1"):
    source_bank = bank()
    if properties is None:
        properties = [property_memory(entry.prompt_id, entry.prompt_text)
                      for entry in source_bank.entries if entry.status == "active"]
    return write_json(tmp_path / "memory.json", {
        "schema_version": schema,
        "selection_policy_version": "property_memory_selection_v1",
        "active_bank_version": source_bank.bank_version,
        "properties": properties,
        "candidates": list(candidates or ()),
    })


def effects_path(tmp_path, candidate_id="cp-new", *, effect="positive",
                 transition_counts=None):
    transition_counts = transition_counts or {
        "wrong_to_correct": 1, "correct_to_wrong": 0,
        "correct_to_correct": 2, "wrong_to_wrong": 0,
        "runtime_failure": 0, "intervention_failure": 0}
    return write_jsonl(tmp_path / "effects.jsonl", ({
        "schema_version": "property_compact_summary_v1",
        "summary_id": f"effect-{candidate_id}", "kind": "intervention_effect",
        "candidate_property_id": candidate_id, "video_id": "v1",
        "effect": effect, "transition_counts": transition_counts,
        "one_sentence_summary": "The intervention improved one QA.",
        "artifact_refs": [], "iteration_ordinal": 1,
    },))


def action(action_id, name, **overrides):
    value = {
        "action_id": action_id, "action": name,
        "target_property_ids": [], "candidate_ids": [],
        "proposed_property_text": None,
        "reasoning": "Conservative bounded-memory decision.",
        "supporting_memory_example_ids": [], "supporting_evidence_ids": [],
        "behaviors_to_preserve": [], "confidence": 0.8,
    }
    value.update(overrides)
    return value


class MockProvider:
    model = "fixture-codebook-model"

    def __init__(self, actions):
        self.actions = actions
        self.calls = []

    def __call__(self, system_prompt, request):
        self.calls.append((system_prompt, json.loads(request)))
        return json.dumps({"schema_version": UPDATER_RESPONSE_SCHEMA_VERSION,
                           "actions": self.actions})

    def metadata(self):
        return {"provider": "fixture", "model": self.model, "real_model": False}


class SequenceProvider:
    model = "fixture-codebook-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, system_prompt, request):
        self.calls.append((system_prompt, json.loads(request)))
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def metadata(self):
        return {"provider": "fixture-sequence", "model": self.model,
                "real_model": False}


def run_updater(tmp_path, actions, *, candidates=None, properties=None, output="updater",
                effect="positive", transition_counts=None):
    provider = MockProvider(actions)
    updater = MemoryConditionedLLMCodebookUpdater(response_provider=provider)
    result = updater.run(
        iteration_id="iteration-1", prompt_bank=bank(),
        property_memory_snapshot_path=memory_snapshot(
            tmp_path, candidates=candidates, properties=properties),
        intervention_summary_path=effects_path(
            tmp_path, effect=effect, transition_counts=transition_counts),
        output_dir=str(tmp_path / output))
    return result, provider


def candidate_bank(result):
    wrapper = json.loads(Path(
        result["artifacts"]["candidate_codebook"]).read_text())
    assert wrapper["schema_version"] == CANDIDATE_CODEBOOK_SCHEMA_VERSION
    return prompt_bank_from_json(wrapper["candidate_bank"])


def test_strict_updater_plan_parsing():
    with pytest.raises(LLMCodebookUpdaterParseError):
        parse_updater_plan("not json", max_actions=2)
    malformed = action("a1", "no_op")
    malformed["extra"] = True
    with pytest.raises(LLMCodebookUpdaterParseError, match="strict action schema"):
        parse_updater_plan(json.dumps({
            "schema_version": UPDATER_RESPONSE_SCHEMA_VERSION,
            "actions": [malformed]}), max_actions=2)


def test_parse_failure_retries_and_preserves_attempts(tmp_path):
    provider = SequenceProvider([
        json.dumps({"actions": []}),
        json.dumps({"schema_version": UPDATER_RESPONSE_SCHEMA_VERSION,
                    "actions": []}),
    ])
    updater = MemoryConditionedLLMCodebookUpdater(response_provider=provider)
    result = updater.run(
        iteration_id="iteration-1", prompt_bank=bank(),
        property_memory_snapshot_path=memory_snapshot(tmp_path),
        intervention_summary_path=effects_path(tmp_path),
        output_dir=str(tmp_path / "retry"))
    attempts = json.loads(Path(
        result["artifacts"]["response_attempts"]).read_text())
    assert len(provider.calls) == 2
    assert attempts["accepted_attempt_index"] == 2
    assert [row["status"] for row in attempts["attempts"]] == [
        "rejected", "accepted"]
    assert "previous response failed strict parsing" in provider.calls[1][0]


def test_partial_retry_resume_does_not_repeat_completed_provider_call(tmp_path):
    output = tmp_path / "partial"
    first = SequenceProvider([
        json.dumps({"actions": []}), RuntimeError("temporary provider failure")])
    updater = MemoryConditionedLLMCodebookUpdater(response_provider=first)
    memory_path = memory_snapshot(tmp_path)
    effects = effects_path(tmp_path)
    with pytest.raises(RuntimeError, match="temporary provider failure"):
        updater.run(
            iteration_id="iteration-1", prompt_bank=bank(),
            property_memory_snapshot_path=memory_path,
            intervention_summary_path=effects, output_dir=str(output))
    second = SequenceProvider([
        json.dumps({"schema_version": UPDATER_RESPONSE_SCHEMA_VERSION,
                    "actions": []})])
    resumed = MemoryConditionedLLMCodebookUpdater(response_provider=second).run(
        iteration_id="iteration-1", prompt_bank=bank(),
        property_memory_snapshot_path=memory_path,
        intervention_summary_path=effects, output_dir=str(output))
    assert len(first.calls) == 2
    assert len(second.calls) == 1
    attempts = json.loads(Path(
        resumed["artifacts"]["response_attempts"]).read_text())
    assert attempts["accepted_attempt_index"] == 2


def test_supported_add_and_promotion_after_validation(tmp_path):
    add = action(
        "add-1", "add", candidate_ids=["cp-new"],
        proposed_property_text="Track reusable object completion states precisely.",
        supporting_memory_example_ids=["candidate-positive"],
        supporting_evidence_ids=["effect-cp-new"])
    result, provider = run_updater(
        tmp_path, [add], candidates=[candidate_memory()])
    output = candidate_bank(result)
    property_id = deterministic_property_id(
        "cp-new", add["proposed_property_text"])
    assert property_id.startswith("pe_") and len(property_id) == 23
    assert output.entry(property_id).status == "active"
    assert output.entry(property_id).applicability_traits[
        "required_modalities"] == ("frames",)
    mapping = json.loads(Path(result["artifacts"]["property_id_mapping"]).read_text())
    assert mapping["candidate_promotions"] == {"cp-new": property_id}
    promoted = json.loads(Path(
        result["artifacts"]["promoted_property_memory"]).read_text())
    candidate = next(item for item in promoted["candidates"]
                     if item["candidate_property_id"] == "cp-new")
    assert candidate["status"] == "promoted"
    assert provider.calls[0][1]["candidate_memories"][0][
        "candidate_property_id"] == "cp-new"
    assert provider.calls[0][1]["candidate_memories"][0]["failure_analysis"] == (
        "The generated description omitted completion.")
    assert provider.calls[0][1]["schema_version"] == \
        "memory_codebook_updater_request_v4"
    assert "proposed_property_id" not in provider.calls[0][1][
        "output_schema"]["actions"][0]


def test_add_without_correctness_flip_or_positive_label_is_structurally_allowed(tmp_path):
    add = action(
        "add-unsupported", "add", candidate_ids=["cp-new"],
        proposed_property_text="Track reusable object completion states precisely.",
        supporting_memory_example_ids=["candidate-none"],
        supporting_evidence_ids=["effect-cp-new"])
    result, _ = run_updater(
        tmp_path, [add], candidates=[candidate_memory(positive=False)],
        effect="no_effect", transition_counts={
            "wrong_to_correct": 0, "correct_to_wrong": 0,
            "correct_to_correct": 3, "wrong_to_wrong": 0,
            "runtime_failure": 0, "intervention_failure": 0})
    property_id = deterministic_property_id(
        "cp-new", add["proposed_property_text"])
    assert candidate_bank(result).entry(property_id).status == "active"
    report = json.loads(Path(result["artifacts"]["validation_report"]).read_text())
    assert report["status"] == "valid"
    assert report["rejected_actions"] == []
    assert report["non_blocking_warnings"]
    promoted = json.loads(Path(
        result["artifacts"]["promoted_property_memory"]).read_text())
    assert promoted["candidates"][0]["status"] == "promoted"


def test_revise_preserves_identity_and_positive_behavior(tmp_path):
    source = bank().entry("pe_temporal")
    revise = action(
        "revise-1", "revise", target_property_ids=["pe_temporal"],
        candidate_ids=["cp-new"],
        proposed_property_text=(
            "Preserve temporal order and explicit action completion states."),
        supporting_memory_example_ids=["pos-pe_temporal", "candidate-positive"],
        supporting_evidence_ids=["effect-cp-new"],
        behaviors_to_preserve=[source.prompt_text])
    result, _ = run_updater(
        tmp_path, [revise], candidates=[candidate_memory()])
    output = candidate_bank(result)
    assert output.entry("pe_temporal").prompt_text == \
        revise["proposed_property_text"]
    assert output.entry("pe_temporal").prompt_id == "pe_temporal"


def test_merge_aliases_and_complete_old_id_mapping(tmp_path):
    source_bank = bank()
    default = source_bank.entry("pe_default")
    temporal = source_bank.entry("pe_temporal")
    merge = action(
        "merge-1", "merge",
        target_property_ids=["pe_default", "pe_temporal"],
        candidate_ids=["cp-new"],
        proposed_property_text=(
            "Describe visually supported events while preserving temporal order."),
        supporting_memory_example_ids=["pos-pe_default", "pos-pe_temporal",
                                       "candidate-positive"],
        supporting_evidence_ids=["effect-cp-new"],
        behaviors_to_preserve=[default.prompt_text, temporal.prompt_text])
    result, _ = run_updater(
        tmp_path, [merge], candidates=[candidate_memory()])
    output = candidate_bank(result)
    assert output.entry("pe_temporal").status == "retired"
    aliases = output.entry("pe_default").provenance["aliases"]
    assert "pe_temporal" in aliases and temporal.prompt_text in aliases
    mapping = json.loads(Path(result["artifacts"]["property_id_mapping"]).read_text())
    assert mapping["old_to_new_property_ids"] == {
        "pe_default": "pe_default", "pe_temporal": "pe_default",
        "pe_text": "pe_text"}


def test_revise_and_merge_do_not_require_token_overlap(tmp_path):
    revise = action(
        "revise-no-overlap", "revise", target_property_ids=["pe_temporal"],
        candidate_ids=["cp-new"],
        proposed_property_text="Record causally linked scene developments succinctly.",
        supporting_memory_example_ids=["candidate-none"],
        supporting_evidence_ids=["effect-cp-new"])
    revised, _ = run_updater(
        tmp_path / "revise", [revise], candidates=[candidate_memory(positive=False)],
        effect="no_effect", transition_counts={
            "wrong_to_correct": 0, "correct_to_wrong": 0,
            "correct_to_correct": 3, "wrong_to_wrong": 0,
            "runtime_failure": 0, "intervention_failure": 0})
    assert candidate_bank(revised).entry("pe_temporal").prompt_text == \
        revise["proposed_property_text"]

    merge = action(
        "merge-no-overlap", "merge",
        target_property_ids=["pe_default", "pe_temporal"],
        proposed_property_text="Summarize grounded scene developments concisely.",
        supporting_memory_example_ids=["pos-pe_default"])
    merged, _ = run_updater(tmp_path / "merge", [merge])
    assert candidate_bank(merged).entry("pe_temporal").status == "retired"


def test_unknown_references_and_conflicting_mutations_are_structurally_rejected(tmp_path):
    actions = [
        action("unknown-property", "retire", target_property_ids=["pe_missing"],
               supporting_evidence_ids=["effect-cp-new"]),
        action("unknown-evidence", "retire", target_property_ids=["pe_text"],
               supporting_evidence_ids=["missing-effect"]),
        action("revise", "revise", target_property_ids=["pe_temporal"],
               proposed_property_text="Record ordered scene developments clearly.",
               supporting_evidence_ids=["effect-cp-new"]),
        action("conflict", "retire", target_property_ids=["pe_temporal"],
               supporting_evidence_ids=["effect-cp-new"]),
    ]
    result, _ = run_updater(tmp_path, actions)
    report = json.loads(Path(result["artifacts"]["validation_report"]).read_text())
    assert report["status"] == "invalid_with_structural_errors"
    reasons = {row["action_id"]: row["reasons"]
               for row in report["structural_errors"]}
    assert any("unknown or inactive" in reason
               for reason in reasons["unknown-property"])
    assert any("unknown compact evidence" in reason
               for reason in reasons["unknown-evidence"])
    assert any("conflicts" in reason for reason in reasons["conflict"])


def test_runtime_failure_evidence_is_visible_but_not_semantically_gated(tmp_path):
    candidate = candidate_memory(positive=False)
    candidate["no_effect_examples"] = []
    candidate["failure_examples"] = [example(
        "runtime-only", effect="unavailable")]
    revise = action(
        "runtime-cited", "revise", target_property_ids=["pe_temporal"],
        candidate_ids=["cp-new"],
        proposed_property_text="Describe ordered visible developments compactly.",
        supporting_memory_example_ids=["runtime-only"],
        supporting_evidence_ids=["effect-cp-new"])
    result, provider = run_updater(
        tmp_path, [revise], candidates=[candidate], effect="unavailable",
        transition_counts={
            "wrong_to_correct": 0, "correct_to_wrong": 0,
            "correct_to_correct": 0, "wrong_to_wrong": 0,
            "runtime_failure": 3, "intervention_failure": 0})
    loaded = json.loads(Path(
        result["artifacts"]["validation_report"]).read_text())
    assert loaded["accepted_action_ids"] == ["runtime-cited"]
    assert provider.calls[0][1]["evidence_status_contract"][
        "reliability_only"] == ["runtime_failure", "intervention_failure"]
    for name in ("llm_proposed_plan", "structural_validation_result",
                 "structural_errors", "non_blocking_warnings",
                 "final_applied_plan"):
        assert Path(result["artifacts"][name]).exists()


def test_single_harmful_example_retire_is_structurally_allowed(tmp_path):
    source = bank().entry("pe_default")
    properties = [property_memory(entry.prompt_id, entry.prompt_text)
                  for entry in bank().entries]
    properties[0]["negative_examples"] = [
        example("harm-one", video="v1", iteration=1, effect="negative")]
    retire = action(
        "retire-1", "retire", target_property_ids=["pe_default"],
        supporting_memory_example_ids=["harm-one"],
        behaviors_to_preserve=[source.prompt_text])
    noop = action("noop-1", "no_op", candidate_ids=["cp-new"],
                  supporting_memory_example_ids=["candidate-none"])
    result, _ = run_updater(
        tmp_path, [retire, noop], candidates=[candidate_memory(positive=False)],
        properties=properties)
    assert candidate_bank(result).entry("pe_default").status == "retired"
    report = json.loads(Path(result["artifacts"]["validation_report"]).read_text())
    assert report["accepted_action_ids"] == ["retire-1", "noop-1"]
    assert report["rejected_actions"] == []
    assert any("fewer than two" in warning
               for row in report["non_blocking_warnings"]
               for warning in row["warnings"])


def test_exact_resume_and_incompatible_memory_fail_closed(tmp_path):
    noop = action("noop", "no_op")
    result, provider = run_updater(tmp_path, [noop])
    updater = MemoryConditionedLLMCodebookUpdater(response_provider=provider)
    resumed = updater.run(
        iteration_id="iteration-1", prompt_bank=bank(),
        property_memory_snapshot_path=str(tmp_path / "memory.json"),
        intervention_summary_path=str(tmp_path / "effects.jsonl"),
        output_dir=str(tmp_path / "updater"))
    assert resumed["resumed"] is True
    assert len(provider.calls) == 1
    manifest_path = tmp_path / "updater" / "manifest.json"
    incompatible = json.loads(manifest_path.read_text())
    incompatible["schema_version"] = "memory_codebook_checkpoint_manifest_v999"
    manifest_path.write_text(json.dumps(incompatible, sort_keys=True))
    with pytest.raises(LLMCodebookUpdaterConflictError, match="incompatible semantics"):
        updater.run(
            iteration_id="iteration-1", prompt_bank=bank(),
            property_memory_snapshot_path=str(tmp_path / "memory.json"),
            intervention_summary_path=str(tmp_path / "effects.jsonl"),
            output_dir=str(tmp_path / "updater"))
    bad = memory_snapshot(tmp_path / "bad", schema="property_memory_v999")
    with pytest.raises(LLMCodebookUpdaterConflictError, match="incompatible"):
        updater.run(
            iteration_id="bad", prompt_bank=bank(),
            property_memory_snapshot_path=bad,
            intervention_summary_path=effects_path(tmp_path / "bad"),
            output_dir=str(tmp_path / "bad-output"))


class UnusedStage:
    configuration_identity = {"fixture": "unused"}


def orchestration_raw_artifacts(tmp_path, *, with_candidate=True):
    video = "video-a"
    proposals = ([{
        "candidate_property_id": "cp-new",
        "property_text": "Track reusable object completion states.",
        "source_video_id": video, "source_question_ids": ["q1"],
        "source_qa_baseline_correct": False,
        "applicability": {
            "when": "Visible evidence shows an object changing completion state.",
            "positive_cues": ["A completion state is visually observable."],
            "negative_cues": [], "required_modalities": ["frames"],
        },
        "motivating_failure_types": ["missing_state"],
        "covered_by_existing_property_ids": [],
        "proposal_rationale": "Completion state was omitted.",
        "proposer_policy_version": "fixture",
    }] if with_candidate else [])
    proposal_path = write_json(tmp_path / "raw" / "proposals.json", {
        "video_id": video, "proposals": proposals})
    baseline_path = write_json(tmp_path / "raw" / "video_complete.json", {
        "video_id": video, "property_proposal_path": proposal_path})
    results = []
    if with_candidate:
        original_proposal = proposals[0]
        transitions_path = write_json(tmp_path / "raw" / "transitions.json", {
            "candidate_property_id": "cp-new", "source_video_id": video,
            "original_proposal": original_proposal,
            "source_question_ids": ["q1"],
            "source_qa_baseline_correct": False,
            "transition_counts": {"wrong_to_correct": 1, "correct_to_wrong": 0,
                                  "correct_to_correct": 2, "wrong_to_wrong": 0},
            "qas": [
                {"question_id": "q1", "baseline_prediction": "B",
                 "candidate_prediction": "A", "baseline_correct": False,
                 "candidate_correct": True, "transition": "wrong_to_correct",
                 "errors": []},
                {"question_id": "q2", "baseline_prediction": "A",
                 "candidate_prediction": "A", "baseline_correct": True,
                 "candidate_correct": True, "transition": "correct_to_correct",
                 "errors": []},
                {"question_id": "q3", "baseline_prediction": "A",
                 "candidate_prediction": "A", "baseline_correct": True,
                 "candidate_correct": True, "transition": "correct_to_correct",
                 "errors": []},
            ]})
        result_path = write_json(tmp_path / "raw" / "result.json", {
            "status": "completed", "candidate_property_id": "cp-new",
            "source_video_id": video, "original_proposal": original_proposal,
            "transitions_path": transitions_path})
        results.append({"candidate_property_id": "cp-new",
                        "result_path": result_path})
    intervention_path = write_json(tmp_path / "raw" / "intervention.json", {
        "status": "completed", "source_video_id": video, "results": results})
    feedback_path = write_json(tmp_path / "raw" / "feedback.json", {
        "status": "completed", "source_video_id": video, "properties": []})
    return baseline_path, intervention_path, feedback_path


def test_orchestrator_automatically_feeds_memory_snapshot_to_updater(tmp_path):
    add = action(
        "add-1", "add", candidate_ids=["cp-new"],
        proposed_property_text="Track reusable object completion states precisely.",
        supporting_memory_example_ids=[], supporting_evidence_ids=[])
    # Fill IDs after inspecting the automatically generated bounded artifacts.
    class AdaptiveProvider(MockProvider):
        def __call__(self, system_prompt, request):
            payload = json.loads(request)
            candidate = payload["candidate_memories"][0]
            effect = payload["current_intervention_summaries"][0]
            self.actions[0]["supporting_memory_example_ids"] = [
                candidate["positive_examples"][0]["example_id"]]
            self.actions[0]["supporting_evidence_ids"] = [effect["summary_id"]]
            return super().__call__(system_prompt, request)
    provider = AdaptiveProvider([add])
    updater = MemoryConditionedLLMCodebookUpdater(response_provider=provider)
    runner = Checkpoint3EOrchestrator(
        baseline_runner=UnusedStage(), intervention_runner=UnusedStage(),
        feedback_runner=UnusedStage(), confirmation_evaluator=UnusedStage(),
        require_real_models=False,
        property_memory_runner=CompactPropertyMemoryRunner(),
        llm_codebook_updater=updater)
    baseline_path, intervention_path, feedback_path = orchestration_raw_artifacts(tmp_path)
    result = runner.run_memory_conditioned_codebook_checkpoint(
        iteration_id="iteration-1", iteration_ordinal=1, prompt_bank=bank(),
        baseline_video_manifest_paths=(baseline_path,),
        intervention_manifest_paths=(intervention_path,),
        feedback_manifest_paths=(feedback_path,),
        output_dir=str(tmp_path / "checkpoint"), state_dir=str(tmp_path / "state"))
    assert result["router_updated"] is False
    assert result["production_pointer_mutated"] is False
    assert provider.calls[0][1]["candidate_memories"][0][
        "candidate_property_id"] == "cp-new"
    effect = provider.calls[0][1]["current_intervention_summaries"][0]
    assert effect["original_proposal"]["property_text"] == (
        "Track reusable object completion states.")
    assert effect["source_question_ids"] == ["q1"]
    assert effect["source_qa_baseline_correct"] is False
    assert effect["qa_transitions"][0]["transition"] == "wrong_to_correct"
    resumed = runner.run_memory_conditioned_codebook_checkpoint(
        iteration_id="iteration-1", iteration_ordinal=1, prompt_bank=bank(),
        baseline_video_manifest_paths=(baseline_path,),
        intervention_manifest_paths=(intervention_path,),
        feedback_manifest_paths=(feedback_path,),
        output_dir=str(tmp_path / "checkpoint"), state_dir=str(tmp_path / "state"))
    assert resumed["resumed"] is True and len(provider.calls) == 1


def test_zero_proposal_checkpoint_and_legacy_deterministic_constructor_compatible(tmp_path):
    provider = MockProvider([])
    runner = Checkpoint3EOrchestrator(
        baseline_runner=UnusedStage(), intervention_runner=UnusedStage(),
        feedback_runner=UnusedStage(), confirmation_evaluator=UnusedStage(),
        require_real_models=False,
        property_memory_runner=CompactPropertyMemoryRunner(),
        llm_codebook_updater=MemoryConditionedLLMCodebookUpdater(
            response_provider=provider))
    baseline_path, intervention_path, feedback_path = orchestration_raw_artifacts(
        tmp_path, with_candidate=False)
    result = runner.run_memory_conditioned_codebook_checkpoint(
        iteration_id="zero", iteration_ordinal=1, prompt_bank=bank(),
        baseline_video_manifest_paths=(baseline_path,),
        intervention_manifest_paths=(intervention_path,),
        feedback_manifest_paths=(feedback_path,),
        output_dir=str(tmp_path / "zero"), state_dir=str(tmp_path / "state"))
    assert candidate_bank(json.loads(Path(
        result["artifacts"]["llm_codebook_updater_manifest"]).read_text())).bank_version \
        == bank().bank_version
    legacy = Checkpoint3EOrchestrator(
        baseline_runner=UnusedStage(), intervention_runner=UnusedStage(),
        feedback_runner=UnusedStage(), confirmation_evaluator=UnusedStage(),
        require_real_models=False)
    assert legacy.property_memory_runner is None
    assert legacy.llm_codebook_updater is None
    assert SYSTEM_PROMPT_VERSION == "memory_codebook_updater_prompt_v3_deterministic_ids"
