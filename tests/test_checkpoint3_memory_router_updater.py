import json
from dataclasses import replace
from pathlib import Path

import pytest

from surrogate_rollout.optimization.final_iteration import (
    Checkpoint3EOrchestrator,
    FinalIterationConflictError,
)
from surrogate_rollout.optimization.llm_router_updater import (
    LLMRouterUpdaterConflictError,
    LLMRouterUpdaterParseError,
    MemoryConditionedLLMRouterUpdater,
    ROUTER_UPDATER_RESPONSE_SCHEMA_VERSION,
    SYSTEM_PROMPT_VERSION,
    parse_router_updater_plan,
)
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
)
from surrogate_rollout.prompt_routing.policies.history_aware_vlm_router import (
    HistoryAwareVLMRouter,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    SegmentContext,
    as_json_dict,
)
from surrogate_rollout.prompt_routing.structured_router_policy import (
    FIXED_ROUTER_PROTOCOL,
    STRUCTURED_ROUTER_POLICY_SCHEMA_VERSION,
    bootstrap_structured_router_policy,
    install_structured_router_policy,
)
from surrogate_rollout.schemas import sha256_json


ROOT = Path(__file__).parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_json_dict(value), sort_keys=True))
    return str(path.resolve())


def file_hash(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def components():
    value = json.loads((
        ROOT / "prompt_routing/fixtures/stage4_7_components.json").read_text())
    bank = prompt_bank_from_json(value["prompt_bank"])
    router = replace(
        router_policy_from_json(value["router_policy"]),
        policy_type="history_aware_vlm",
        configuration={"max_selected_properties": 3, "max_tokens": 64})
    return bank, router


def route_example(example_id, property_id, polarity, video="v1"):
    return {
        "example_id": example_id, "video_id": video, "iteration_ordinal": 1,
        "summary": f"Demonstrated {polarity} caption and downstream effect.",
        "artifact_refs": [], "property_id": property_id, "polarity": polarity,
    }


def memory(bank, *, schema="property_memory_v1"):
    rows = []
    for entry in bank.entries:
        if entry.status != "active":
            continue
        rows.append({
            "schema_version": "property_memory_v1",
            "property_id": entry.prompt_id, "property_text": entry.prompt_text,
            "why_created": "seed", "intended_behavior": entry.prompt_text,
            "origin": {"kind": "seed_or_legacy", "artifact_refs": []},
            "positive_examples": [], "negative_examples": [],
            "no_effect_examples": [],
            "routing_examples": {
                "positive": [route_example(
                    f"pos-{entry.prompt_id}", entry.prompt_id, "positive")],
                "negative": [route_example(
                    f"neg-{entry.prompt_id}", entry.prompt_id, "negative")],
            },
            "aliases": [], "revision_history": [],
            "latest_decision": {"action": "preserve", "reason": "seed",
                                "artifact_refs": []},
        })
    return {"schema_version": schema, "properties": rows, "candidates": []}


def action(action_id, action_type, target="pe_temporal", value=None, support=()):
    return {
        "action_id": action_id, "target_property_id": target,
        "action_type": action_type, "proposed_value": value,
        "reason": "Bounded demonstrated routing effect.",
        "supporting_memory_example_ids": list(support),
        "supporting_evidence_ids": list(support), "confidence": 0.8,
    }


class Provider:
    model = "fixture-router-updater"

    def __init__(self, actions=(), error=None):
        self.actions, self.error, self.calls = list(actions), error, []

    def __call__(self, system_prompt, request):
        self.calls.append((system_prompt, json.loads(request)))
        if self.error:
            raise self.error
        return json.dumps({
            "schema_version": ROUTER_UPDATER_RESPONSE_SCHEMA_VERSION,
            "actions": self.actions})

    def metadata(self):
        return {"provider": "fixture", "model": self.model, "real_model": False}


class SequenceProvider(Provider):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)

    def __call__(self, system_prompt, request):
        self.calls.append((system_prompt, json.loads(request)))
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def updater_inputs(tmp_path, *, candidate_bank=None, mapping=None, memory_value=None):
    bank, router = components()
    candidate_bank = candidate_bank or bank
    mapping = mapping or {entry.prompt_id: entry.prompt_id for entry in bank.entries}
    wrapper_path = write_json(tmp_path / "candidate_codebook.json", {
        "schema_version": "memory_candidate_codebook_v2",
        "input_bank_version": bank.bank_version,
        "candidate_bank": candidate_bank,
        "candidate_codebook_hash": sha256_json(as_json_dict(candidate_bank)),
        "lineage": {"iteration_id": "iteration-1"},
    })
    mapping_path = write_json(tmp_path / "mapping.json", {
        "schema_version": "property_id_mapping_v2",
        "old_to_new_property_ids": mapping, "candidate_promotions": {}})
    memory_path = write_json(
        tmp_path / "memory.json", memory_value or memory(candidate_bank))
    plan_path = write_json(tmp_path / "codebook_plan.json", {
        "schema_version": "memory_codebook_applied_plan_v1", "actions": []})
    validation_path = write_json(tmp_path / "codebook_validation.json", {
        "schema_version": "memory_codebook_validation_report_v1",
        "accepted_action_ids": [], "rejected_actions": []})
    return bank, router, wrapper_path, mapping_path, memory_path, plan_path, validation_path


def run_updater(tmp_path, actions, **kwargs):
    inputs = updater_inputs(tmp_path / "inputs", **kwargs)
    bank, router, wrapper, mapping_path, memory_path, plan, validation = inputs
    provider = Provider(actions)
    updater = MemoryConditionedLLMRouterUpdater(response_provider=provider)
    result = updater.run(
        iteration_id="iteration-1", parent_router_policy=router,
        candidate_codebook_path=wrapper, property_id_mapping_path=mapping_path,
        property_memory_snapshot_path=memory_path,
        codebook_applied_plan_path=plan,
        codebook_validation_report_path=validation,
        output_dir=str(tmp_path / "router_updater"))
    return result, provider, inputs


def loaded(result, name):
    return json.loads(Path(result["artifacts"][name]).read_text())


def test_strict_router_plan_parsing():
    with pytest.raises(LLMRouterUpdaterParseError):
        parse_router_updater_plan("not json", max_actions=2)
    malformed = action("bad", "no_op")
    malformed["extra"] = True
    with pytest.raises(LLMRouterUpdaterParseError, match="strict schema"):
        parse_router_updater_plan(json.dumps({
            "schema_version": ROUTER_UPDATER_RESPONSE_SCHEMA_VERSION,
            "actions": [malformed]}), max_actions=2)
    empty_target = action("empty-target", "no_op", target="")
    with pytest.raises(LLMRouterUpdaterParseError, match="target_property_id"):
        parse_router_updater_plan(json.dumps({
            "schema_version": ROUTER_UPDATER_RESPONSE_SCHEMA_VERSION,
            "actions": [empty_target]}), max_actions=2)


def test_router_parse_failure_retries_and_preserves_attempts(tmp_path):
    inputs = updater_inputs(tmp_path / "inputs")
    invalid = action("bad", "no_op")
    invalid["target_property_id"] = None
    provider = SequenceProvider([
        json.dumps({"schema_version": ROUTER_UPDATER_RESPONSE_SCHEMA_VERSION,
                    "actions": [invalid]}),
        json.dumps({"schema_version": ROUTER_UPDATER_RESPONSE_SCHEMA_VERSION,
                    "actions": [action("ok", "no_op")]}),
    ])
    result = MemoryConditionedLLMRouterUpdater(response_provider=provider).run(
        iteration_id="iteration-1", parent_router_policy=inputs[1],
        candidate_codebook_path=inputs[2], property_id_mapping_path=inputs[3],
        property_memory_snapshot_path=inputs[4],
        codebook_applied_plan_path=inputs[5],
        codebook_validation_report_path=inputs[6],
        output_dir=str(tmp_path / "router_updater"))
    attempts = loaded(result, "response_attempts")
    assert len(provider.calls) == 2
    assert attempts["accepted_attempt_index"] == 2
    assert [row["status"] for row in attempts["attempts"]] == [
        "rejected", "accepted"]


def test_router_partial_retry_resume_skips_saved_attempt(tmp_path):
    inputs = updater_inputs(tmp_path / "inputs")
    output = tmp_path / "router_updater"
    first = SequenceProvider([
        json.dumps({"actions": []}), RuntimeError("temporary provider failure")])
    with pytest.raises(RuntimeError, match="temporary provider failure"):
        MemoryConditionedLLMRouterUpdater(response_provider=first).run(
            iteration_id="iteration-1", parent_router_policy=inputs[1],
            candidate_codebook_path=inputs[2],
            property_id_mapping_path=inputs[3],
            property_memory_snapshot_path=inputs[4],
            codebook_applied_plan_path=inputs[5],
            codebook_validation_report_path=inputs[6],
            output_dir=str(output))
    second = SequenceProvider([json.dumps({
        "schema_version": ROUTER_UPDATER_RESPONSE_SCHEMA_VERSION,
        "actions": [action("ok", "no_op")],
    })])
    result = MemoryConditionedLLMRouterUpdater(response_provider=second).run(
        iteration_id="iteration-1", parent_router_policy=inputs[1],
        candidate_codebook_path=inputs[2],
        property_id_mapping_path=inputs[3],
        property_memory_snapshot_path=inputs[4],
        codebook_applied_plan_path=inputs[5],
        codebook_validation_report_path=inputs[6],
        output_dir=str(output))
    assert len(first.calls) == 2
    assert len(second.calls) == 1
    assert loaded(result, "response_attempts")["accepted_attempt_index"] == 2


def test_updater_receives_candidate_codebook_mapping_and_bounded_evidence(tmp_path):
    result, provider, _ = run_updater(tmp_path, [action("n", "no_op")])
    request = provider.calls[0][1]
    assert request["schema_version"] == "memory_router_updater_request_v2"
    assert request["property_id_mapping"]["old_to_new_property_ids"][
        "pe_temporal"] == "pe_temporal"
    assert {row["property_id"] for row in request["validated_candidate_codebook"]} == {
        "pe_default", "pe_temporal", "pe_text"}
    assert request["current_compact_routing_evidence"]
    assert result["system_prompt_version"] == SYSTEM_PROMPT_VERSION
    assert "supervision_examples" not in loaded(
        result, "candidate_router_policy")["configuration"]


@pytest.mark.parametrize(("kind", "value", "support", "field"), [
    ("set_selection_guidance", "Select for visible ordered state changes.",
     ("pos-pe_temporal",), "selection_guidance"),
    ("set_avoidance_guidance", "Avoid for unrelated incidental motion.",
     ("neg-pe_temporal",), "avoidance_guidance"),
])
def test_selection_and_avoidance_guidance_updates(
        tmp_path, kind, value, support, field):
    result, _, _ = run_updater(
        tmp_path, [action("a", kind, value=value, support=support)])
    structured = loaded(result, "structured_router_policy")
    row = next(item for item in structured["properties"]
               if item["property_id"] == "pe_temporal")
    assert row[field] == value
    assert structured["protocol"] == {**FIXED_ROUTER_PROTOCOL,
                                       "max_selected_properties": 3}


def test_examples_are_not_semantically_rejected_by_evidence_polarity(tmp_path):
    actions = [
        action("p", "add_positive_example", value=(
            "A visible sequence reaches a clear outcome."),
            support=("pos-pe_temporal",)),
        action("n", "add_negative_example", value=(
            "Unrelated background motion has no ordered state change."),
            support=("neg-pe_temporal",)),
        action("bad", "add_positive_example", value="Several people are present.",
               support=("neg-pe_temporal",)),
        action("conflict", "add_negative_example", value=(
            "A visible sequence reaches a clear outcome."),
            support=("neg-pe_temporal",)),
    ]
    result, _, _ = run_updater(tmp_path, actions)
    row = next(item for item in loaded(result, "structured_router_policy")[
        "properties"] if item["property_id"] == "pe_temporal")
    assert len(row["positive_examples"]) == 2
    assert len(row["negative_examples"]) == 1
    report = loaded(result, "validation_report")
    assert report["accepted_action_ids"] == ["p", "n", "bad"]
    assert report["rejected_actions"][0]["action_id"] == "conflict"
    assert "exact duplicate routing example" in report["rejected_actions"][0][
        "reasons"]


def test_stale_id_rejected_but_general_question_answer_words_are_allowed(tmp_path):
    result, _, _ = run_updater(tmp_path, [
        action("stale", "no_op", target="pe_missing"),
        action("leak", "set_selection_guidance",
               value="Select when the question answer is temporal.",
               support=("pos-pe_temporal",)),
    ])
    reasons = [reason for row in loaded(result, "validation_report")[
        "rejected_actions"] for reason in row["reasons"]]
    assert any("stale" in reason for reason in reasons)
    assert loaded(result, "validation_report")["accepted_action_ids"] == ["leak"]
    row = next(item for item in loaded(result, "structured_router_policy")[
        "properties"] if item["property_id"] == "pe_temporal")
    assert row["selection_guidance"] == "Select when the question answer is temporal."


def test_selection_guidance_accepts_negative_or_no_flip_evidence(tmp_path):
    bank, _ = components()
    value = memory(bank)
    temporal = next(row for row in value["properties"]
                    if row["property_id"] == "pe_temporal")
    temporal["routing_examples"]["positive"] = []
    temporal["routing_examples"]["negative"].append(
        route_example("no-flip-pe_temporal", "pe_temporal", "no_effect"))
    result, _, _ = run_updater(
        tmp_path, [action(
            "selection", "set_selection_guidance",
            value="Select for visible ordered developments.",
            support=("neg-pe_temporal", "no-flip-pe_temporal"))],
        memory_value=value)
    assert loaded(result, "validation_report")["accepted_action_ids"] == ["selection"]


def test_unknown_router_evidence_is_structurally_rejected(tmp_path):
    result, _, _ = run_updater(tmp_path, [action(
        "missing", "set_selection_guidance",
        value="Select for visible ordered developments.",
        support=("missing-evidence",))])
    reasons = loaded(result, "validation_report")["rejected_actions"][0]["reasons"]
    assert any("unknown routing" in reason for reason in reasons)


def test_conflicting_guidance_is_rejected_and_audit_views_are_written(tmp_path):
    result, _, _ = run_updater(tmp_path, [
        action("first", "set_selection_guidance",
               value="Select for visible ordered developments.",
               support=("pos-pe_temporal",)),
        action("second", "set_selection_guidance",
               value="Select only after a visible state transition.",
               support=("neg-pe_temporal",)),
    ])
    report = loaded(result, "structural_validation_result")
    assert report["status"] == "invalid_with_structural_errors"
    assert report["accepted_action_ids"] == ["first"]
    assert any("conflict" in reason for reason in report[
        "structural_errors"][0]["reasons"])
    for name in ("llm_proposed_plan", "structural_validation_result",
                 "structural_errors", "non_blocking_warnings",
                 "final_applied_plan"):
        assert Path(result["artifacts"][name]).exists()


def test_router_example_budget_is_total_not_per_polarity(tmp_path):
    actions = [action(
        f"p-{index}", "add_positive_example",
        value=f"Visible ordered example number {index} reaches an outcome.",
        support=("neg-pe_temporal",)) for index in range(5)]
    result, _, _ = run_updater(tmp_path, actions)
    row = next(item for item in loaded(result, "structured_router_policy")[
        "properties"] if item["property_id"] == "pe_temporal")
    assert len(row["positive_examples"]) == 4
    assert row["negative_examples"] == []
    assert row["positive_examples"] == [
        f"Visible ordered example number {index} reaches an outcome."
        for index in range(1, 5)]


def test_merge_remaps_guidance_aliases_and_retire_removes_guidance():
    bank, router = components()
    structured = bootstrap_structured_router_policy(bank, router)
    for row in structured["properties"]:
        if row["property_id"] == "pe_temporal":
            row["avoidance_guidance"] = "Avoid when no ordering is visible."
    installed = install_structured_router_policy(
        router, bank, structured, lineage={"old_to_new_property_ids": {
            entry.prompt_id: entry.prompt_id for entry in bank.entries}})
    from surrogate_rollout.prompt_routing.structured_router_policy import (
        remap_structured_router_policy,
    )
    merged_entries = tuple(
        replace(entry, status="retired") if entry.prompt_id == "pe_temporal"
        else entry for entry in bank.entries)
    merged = replace(bank, bank_version="bank_v9001", entries=merged_entries)
    value = remap_structured_router_policy(
        installed.configuration["structured_router_policy"],
        {"pe_default": "pe_default", "pe_temporal": "pe_default",
         "pe_text": "pe_text"}, merged)
    default = next(row for row in value["properties"]
                   if row["property_id"] == "pe_default")
    assert "pe_temporal" in default["aliases"]
    retired = remap_structured_router_policy(
        installed.configuration["structured_router_policy"],
        {"pe_default": "pe_default", "pe_temporal": None,
         "pe_text": "pe_text"}, merged)
    assert "pe_temporal" not in {row["property_id"] for row in retired["properties"]}


def test_rendered_prompt_changes_hash_and_real_router_loads_it(tmp_path):
    result, _, inputs = run_updater(tmp_path, [action(
        "a", "set_avoidance_guidance",
        value="Avoid when motion is incidental and unordered.",
        support=("neg-pe_temporal",))])
    rendered = loaded(result, "rendered_router_prompt")
    router = router_policy_from_json(loaded(result, "candidate_router_policy"))
    assert router.configuration["rendered_router_prompt_hash"] == \
        rendered["prompt_sha256"]

    class VLM:
        def __init__(self):
            self.prompt = None

        def caption(self, frames, prompt, **kwargs):
            self.prompt = prompt
            return '{"property_ids": []}'

    vlm = VLM()
    candidate = prompt_bank_from_json(json.loads(
        Path(inputs[2]).read_text())["candidate_bank"])
    decision = HistoryAwareVLMRouter(vlm).route(
        SegmentContext(
            video_id="v", segment_id="0_1",
            segment_features={"frame_references": ("frame.jpg",)},
            history_summary='{"preceding_captions": []}'),
        candidate, router)
    assert vlm.prompt.startswith(rendered["prompt"] + "\n")
    request = json.loads(vlm.prompt.rsplit("\n", 1)[1])
    assert request["router_prompt_identity"]["rendered_prompt_hash"] == \
        rendered["prompt_sha256"]
    assert decision.decision_payload["rendered_router_prompt_hash"] == \
        rendered["prompt_sha256"]


def test_exact_resume_and_prompt_hash_mismatch_fail_closed(tmp_path):
    result, provider, inputs = run_updater(tmp_path, [action("n", "no_op")])
    updater = MemoryConditionedLLMRouterUpdater(response_provider=provider)
    resumed = updater.run(
        iteration_id="iteration-1", parent_router_policy=inputs[1],
        candidate_codebook_path=inputs[2], property_id_mapping_path=inputs[3],
        property_memory_snapshot_path=inputs[4],
        codebook_applied_plan_path=inputs[5],
        codebook_validation_report_path=inputs[6],
        output_dir=str(tmp_path / "router_updater"))
    assert resumed["resumed"] is True and len(provider.calls) == 1
    rendered = Path(result["artifacts"]["rendered_router_prompt"])
    value = json.loads(rendered.read_text())
    value["prompt_sha256"] = "0" * 64
    rendered.write_text(json.dumps(value, sort_keys=True))
    with pytest.raises(LLMRouterUpdaterConflictError, match="artifact mismatch"):
        updater.run(
            iteration_id="iteration-1", parent_router_policy=inputs[1],
            candidate_codebook_path=inputs[2], property_id_mapping_path=inputs[3],
            property_memory_snapshot_path=inputs[4],
            codebook_applied_plan_path=inputs[5],
            codebook_validation_report_path=inputs[6],
            output_dir=str(tmp_path / "router_updater"))


class UnusedStage:
    configuration_identity = {"fixture": "unused"}


def codebook_checkpoint(tmp_path, *, memory_value=None):
    bank, router, wrapper, mapping_path, memory_path, plan, validation = \
        updater_inputs(tmp_path / "cp2", memory_value=memory_value)
    updater_artifacts = {
        "candidate_codebook": wrapper, "property_id_mapping": mapping_path,
        "promoted_property_memory": memory_path, "applied_plan": plan,
        "validation_report": validation,
    }
    updater_manifest = write_json(tmp_path / "cp2" / "updater_manifest.json", {
        "schema_version": "memory_codebook_checkpoint_manifest_v4",
        "status": "completed", "artifacts": updater_artifacts,
        "artifact_hashes": {name: file_hash(path)
                            for name, path in updater_artifacts.items()},
    })
    placeholder = write_json(tmp_path / "cp2" / "placeholder.json", {"ok": True})
    artifacts = {
        "candidate_codebook": wrapper, "property_id_mapping": mapping_path,
        "promoted_property_memory": memory_path,
        "llm_codebook_updater_manifest": updater_manifest,
        "property_memory_manifest": placeholder,
        "property_memory_snapshot": memory_path,
        "intervention_summaries": placeholder,
    }
    checkpoint = write_json(tmp_path / "cp2" / "manifest.json", {
        "schema_version": "memory_conditioned_codebook_iteration_v1",
        "status": "completed", "iteration_id": "iteration-1",
        "input_bank_version": bank.bank_version,
        "input_bank_hash": sha256_json(as_json_dict(bank)),
        "artifacts": artifacts,
        "artifact_hashes": {name: file_hash(path) for name, path in artifacts.items()},
    })
    return bank, router, checkpoint


def orchestrator(provider, *, skip_llm_when_no_evidence=False):
    return Checkpoint3EOrchestrator(
        baseline_runner=UnusedStage(), intervention_runner=UnusedStage(),
        feedback_runner=UnusedStage(), confirmation_evaluator=UnusedStage(),
        require_real_models=False,
        llm_router_updater=MemoryConditionedLLMRouterUpdater(
            response_provider=provider,
            skip_llm_when_no_evidence=skip_llm_when_no_evidence))


def test_atomic_provisional_pair_creation_and_exact_resume(tmp_path):
    bank, router, checkpoint = codebook_checkpoint(tmp_path)
    provider = Provider([action("n", "no_op")])
    runner = orchestrator(provider)
    result = runner.run_memory_conditioned_router_checkpoint(
        iteration_id="iteration-1", parent_router_policy=router,
        codebook_checkpoint_manifest_path=checkpoint,
        output_dir=str(tmp_path / "atomic"), state_dir=str(tmp_path / "state"))
    pair = loaded(result, "atomic_policy_pair")
    assert pair["schema_version"] == "atomic_provisional_policy_pair_v1"
    assert pair["status"] == "committed"
    assert pair["candidate_policy_pair"]["bank_version"] == bank.bank_version
    assert pair["confirmed_production_pointer_mutated"] is False
    pointer = json.loads((tmp_path / "state/memory_conditioned_provisional/current.json").read_text())
    assert pointer["rendered_router_prompt_hash"] == result[
        "rendered_router_prompt_hash"]
    resumed = runner.run_memory_conditioned_router_checkpoint(
        iteration_id="iteration-1", parent_router_policy=router,
        codebook_checkpoint_manifest_path=checkpoint,
        output_dir=str(tmp_path / "atomic"), state_dir=str(tmp_path / "state"))
    assert resumed["resumed"] is True and len(provider.calls) == 1


def test_no_routing_evidence_calls_llm_and_commits_explicit_noop_pair(tmp_path):
    seed_bank, _ = components()
    empty_memory = memory(seed_bank)
    for row in empty_memory["properties"]:
        row["routing_examples"] = {"positive": [], "negative": []}
    bank, router, checkpoint = codebook_checkpoint(
        tmp_path, memory_value=empty_memory)
    provider = Provider([])
    runner = orchestrator(provider)
    result = runner.run_memory_conditioned_router_checkpoint(
        iteration_id="iteration-1", parent_router_policy=router,
        codebook_checkpoint_manifest_path=checkpoint,
        output_dir=str(tmp_path / "atomic"), state_dir=str(tmp_path / "state"))
    pair = loaded(result, "atomic_policy_pair")
    assert pair["status"] == "committed"
    assert pair["candidate_policy_pair"]["bank_version"] == bank.bank_version
    router_manifest = json.loads(Path(
        result["artifacts"]["router_updater_manifest"]).read_text())
    assert router_manifest["execution_mode"] == "llm_provider_no_evidence"
    assert router_manifest["provider_called"] is True
    assert len(provider.calls) == 1
    resumed = runner.run_memory_conditioned_router_checkpoint(
        iteration_id="iteration-1", parent_router_policy=router,
        codebook_checkpoint_manifest_path=checkpoint,
        output_dir=str(tmp_path / "atomic"), state_dir=str(tmp_path / "state"))
    assert resumed["resumed"] is True
    assert len(provider.calls) == 1


def test_router_failure_prevents_codebook_only_pair_commit(tmp_path):
    _, router, checkpoint = codebook_checkpoint(tmp_path)
    runner = orchestrator(Provider(error=RuntimeError("fixture failure")))
    with pytest.raises(RuntimeError, match="fixture failure"):
        runner.run_memory_conditioned_router_checkpoint(
            iteration_id="iteration-1", parent_router_policy=router,
            codebook_checkpoint_manifest_path=checkpoint,
            output_dir=str(tmp_path / "atomic"), state_dir=str(tmp_path / "state"))
    assert (tmp_path / "atomic/failure.json").exists()
    assert not (tmp_path / "atomic/manifest.json").exists()
    assert not (tmp_path / "state/memory_conditioned_provisional/current.json").exists()
    with pytest.raises(FinalIterationConflictError, match="cannot be resumed"):
        runner.run_memory_conditioned_router_checkpoint(
            iteration_id="iteration-1", parent_router_policy=router,
            codebook_checkpoint_manifest_path=checkpoint,
            output_dir=str(tmp_path / "atomic"), state_dir=str(tmp_path / "state"))


def test_incompatible_memory_schema_and_zero_action_noop(tmp_path):
    bad = memory(components()[0], schema="property_memory_v999")
    inputs = updater_inputs(tmp_path / "bad", memory_value=bad)
    updater = MemoryConditionedLLMRouterUpdater(response_provider=Provider([]))
    with pytest.raises(LLMRouterUpdaterConflictError, match="incompatible"):
        updater.run(
            iteration_id="iteration-1", parent_router_policy=inputs[1],
            candidate_codebook_path=inputs[2], property_id_mapping_path=inputs[3],
            property_memory_snapshot_path=inputs[4],
            codebook_applied_plan_path=inputs[5],
            codebook_validation_report_path=inputs[6],
            output_dir=str(tmp_path / "bad-output"))
    result, provider, _ = run_updater(tmp_path / "zero", [])
    assert result["accepted_action_count"] == 0
    assert provider.calls and loaded(result, "structured_router_policy")["properties"]


def test_explicit_empty_evidence_optimization_is_recorded(tmp_path):
    bank, _ = components()
    empty_memory = memory(bank)
    for row in empty_memory["properties"]:
        row["routing_examples"] = {"positive": [], "negative": []}
    provider = Provider(error=AssertionError("provider must not be called"))
    inputs = updater_inputs(
        tmp_path / "inputs", memory_value=empty_memory)
    updater = MemoryConditionedLLMRouterUpdater(
        response_provider=provider, skip_llm_when_no_evidence=True)
    result = updater.run(
        iteration_id="iteration-1", parent_router_policy=inputs[1],
        candidate_codebook_path=inputs[2], property_id_mapping_path=inputs[3],
        property_memory_snapshot_path=inputs[4],
        codebook_applied_plan_path=inputs[5],
        codebook_validation_report_path=inputs[6],
        output_dir=str(tmp_path / "router_updater"))
    assert provider.calls == []
    assert result["provider_called"] is False
    assert result["execution_mode"] == "configured_empty_evidence_skip"
    assert result["deterministic_reason"] == "explicit_skip_llm_when_no_evidence"
    assert loaded(result, "parsed_plan")["actions"] == []
    assert loaded(result, "applied_plan")["actions"] == []
    execution = loaded(result, "execution")
    assert execution["provider_called"] is False
    assert execution["routing_evidence_count"] == 0
    resumed = updater.run(
        iteration_id="iteration-1", parent_router_policy=inputs[1],
        candidate_codebook_path=inputs[2], property_id_mapping_path=inputs[3],
        property_memory_snapshot_path=inputs[4],
        codebook_applied_plan_path=inputs[5],
        codebook_validation_report_path=inputs[6],
        output_dir=str(tmp_path / "router_updater"))
    assert resumed["resumed"] is True
    assert provider.calls == []
