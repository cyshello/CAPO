import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from surrogate_rollout.optimization.final_iteration import (
    Checkpoint3EOrchestrator,
    FinalIterationConflictError,
)
from surrogate_rollout.optimization.iteration_state import (
    ComponentVersions,
    ConfirmedCheckpointState,
)
from surrogate_rollout.optimization.train_roles import (
    EvidenceCoverageState,
    TrainRoleAssignment,
    TrainVideoRecord,
    select_evidence_batch,
)
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.policies.history_aware_vlm_router import (
    HistoryAwareVLMRouter,
)
from surrogate_rollout.prompt_routing.schemas import SegmentContext


ROOT = Path(__file__).parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return str(path.resolve())


def components():
    value = json.loads((
        ROOT / "prompt_routing/fixtures/stage4_7_components.json").read_text())
    return (
        prompt_bank_from_json(value["prompt_bank"]),
        router_policy_from_json(value["router_policy"]),
        scaffold_policy_from_json(value["scaffold_policy"]),
        scaffold_contract_from_json(value["scaffold_contract"]),
    )


def roles():
    evidence = tuple(TrainVideoRecord(
        video_id=f"e{index}",
        provider_indices=(index * 3, index * 3 + 1, index * 3 + 2),
        question_ids=(f"e{index}q1", f"e{index}q2", f"e{index}q3"),
        previously_cached=True) for index in range(8))
    confirmation = tuple(TrainVideoRecord(
        video_id=f"c{index}",
        provider_indices=(100 + index * 3, 101 + index * 3, 102 + index * 3),
        question_ids=(f"c{index}q1", f"c{index}q2", f"c{index}q3"),
        previously_cached=False) for index in range(2))
    return TrainRoleAssignment(
        evidence_videos=evidence, confirmation_videos=confirmation,
        validation_video_ids=tuple(f"v{index}" for index in range(10)),
        test_video_ids=tuple(f"t{index}" for index in range(10)))


def coverage():
    return EvidenceCoverageState(
        rotation_order=tuple(f"e{index}" for index in range(8)),
        used_since_confirmation=tuple(f"e{index}" for index in range(5)),
        rotation_cursor=5, coverage_cycle=0, iterations_since_confirmation=2,
        confirmation_due=False)


def proposal(candidate_id, video_id, text):
    return {
        "candidate_property_id": candidate_id, "property_text": text,
        "source_video_id": video_id,
        "source_question_ids": [f"{video_id}q1"],
        "motivating_failure_types": ["fixture_failure"],
        "covered_by_existing_property_ids": [],
        "proposal_rationale": "fixture",
        "proposer_policy_version": "fixture_v1",
    }


PROPOSALS = {
    "e1": (("cp_e1", "Preserve fine-grained object state changes."),),
    "e2": (("cp_e2", "Describe short causal action sequences precisely."),),
    "e3": (("cp_e3", "Retain visually grounded spatial relationships."),),
    "e4": (("cp_e4", "Capture brief changes in visible object attributes."),),
    "e5": (
        ("cp_action", "Track fine-grained action completion states."),
        ("cp_motion", "Avoid overemphasizing incidental background motion."),
    ),
    "e6": (("cp_text", "Preserve concise transient written information."),),
    "e7": (("cp_retire", "Describe salient object interactions consistently."),),
}


class MockBaselineRunner:
    configuration_identity = {"mock": "baseline_v1"}

    def __init__(self, *, suppress_proposals=False):
        self.calls = 0
        self.caption_calls = 0
        self.proposal_calls = 0
        self.retrieval_calls = 0
        self.suppress_proposals = suppress_proposals
        self.input_versions = []

    def run(self, **kwargs):
        self.calls += 1
        self.input_versions.append((
            kwargs["prompt_bank"].bank_version,
            kwargs["router_policy"].router_version))
        selected, next_coverage = select_evidence_batch(kwargs["coverage_state"])
        output = Path(kwargs["output_dir"])
        video_paths = []
        all_proposal_paths = []
        all_retrieval_paths = []
        for video_id in selected:
            self.caption_calls += 1
            self.proposal_calls += 1
            records = ([] if self.suppress_proposals else [
                proposal(candidate_id, video_id, text)
                for candidate_id, text in PROPOSALS.get(video_id, ())])
            proposal_path = write_json(
                output / "property_proposals" / f"{video_id}.json", {
                    "video_id": video_id, "proposals": records,
                })
            retrievals = []
            for record in records:
                self.retrieval_calls += 1
                retrievals.append(write_json(
                    output / "retrieval" / video_id /
                    record["candidate_property_id"] / "retrieval.json", {
                        "candidate_property_id": record["candidate_property_id"],
                        "source_video_id": video_id,
                        "property_text": record["property_text"],
                        "s_sim": ["0_10"], "input_fingerprint": "fixture",
                    }))
            video_path = write_json(
                output / "baseline" / video_id / "video_complete.json", {
                    "video_id": video_id,
                    "property_proposal_path": proposal_path,
                    "property_retrieval_paths": retrievals,
                })
            video_paths.append(video_path)
            all_proposal_paths.append(proposal_path)
            all_retrieval_paths.extend(retrievals)
        manifest_path = write_json(output / "manifest.json", {
            "status": "completed", "selected_video_ids": selected,
        })
        return SimpleNamespace(
            selected_video_ids=selected, next_coverage_state=next_coverage,
            video_manifest_paths=tuple(video_paths), manifest_path=manifest_path,
            property_proposal_paths=tuple(all_proposal_paths),
            property_retrieval_paths=tuple(all_retrieval_paths))


class MockInterventionRunner:
    configuration_identity = {"mock": "intervention_v1"}

    def __init__(self):
        self.calls = 0
        self.work_items = 0
        self.qa_calls = 0

    def run(self, **kwargs):
        self.calls += 1
        baseline = json.loads(Path(
            kwargs["baseline_video_manifest_path"]).read_text())
        results = []
        for item in kwargs["proposals"]:
            self.work_items += 1
            self.qa_calls += 3
            result_path = write_json(
                Path(kwargs["output_dir"]) / item.candidate_property_id /
                "result.json", {
                    "status": "completed",
                    "candidate_property_id": item.candidate_property_id,
                    "source_video_id": item.source_video_id,
                })
            results.append({
                "status": "completed",
                "candidate_property_id": item.candidate_property_id,
                "source_video_id": item.source_video_id,
                "result_path": result_path,
            })
        manifest_path = write_json(Path(kwargs["output_dir"]) / "manifest.json", {
            "status": "completed",
            "source_video_id": baseline["video_id"],
            "results": results,
        })
        return SimpleNamespace(manifest_path=manifest_path, results=results)


class MockFeedbackRunner:
    configuration_identity = {"mock": "feedback_v1"}

    def __init__(self):
        self.calls = 0
        self.provider_calls = 0

    def run(self, **kwargs):
        self.calls += 1
        baseline = json.loads(Path(
            kwargs["baseline_video_manifest_path"]).read_text())
        properties = []
        for item in kwargs["proposals"]:
            self.provider_calls += 1
            if item.candidate_property_id not in ("cp_motion", "cp_retire"):
                transition = "wrong_to_correct"
                recommendation = "add"
                coverage = "uncovered"
                covered = []
                evidence = "Targeted visual detail supplies reusable caption evidence."
            elif item.candidate_property_id == "cp_motion":
                transition = "correct_to_wrong"
                recommendation = "router_negative"
                coverage = "covered"
                covered = ["pe_temporal"]
                evidence = "Incidental motion emphasis can obscure stable event outcomes."
            else:
                transition = "correct_to_wrong"
                recommendation = "retire"
                coverage = "covered"
                covered = ["pe_text"]
                evidence = "Broad interaction emphasis can suppress more relevant details."
            feedback_id = f"feedback-{item.candidate_property_id}"
            result_path = write_json(
                Path(kwargs["output_dir"]) / item.candidate_property_id /
                "result.json", {
                    "status": "completed",
                    "candidate_property_id": item.candidate_property_id,
                    "property_text": item.property_text,
                    "source_video_id": item.source_video_id,
                    "source_question_ids": list(item.source_question_ids),
                    "qa_feedback": [{
                        "question_id": item.source_question_ids[0],
                        "transition": transition, "status": "accepted",
                        "feedback_id": feedback_id,
                        "recommendation": recommendation,
                        "coverage_assessment": coverage,
                        "covered_by_property_ids": covered,
                        "evidence": evidence, "s_feedback": ["0_10"],
                    }],
                })
            properties.append({
                "candidate_property_id": item.candidate_property_id,
                "result_path": result_path,
            })
        manifest_path = write_json(Path(kwargs["output_dir"]) / "manifest.json", {
            "status": "completed",
            "source_video_id": baseline["video_id"],
            "properties": properties,
        })
        return SimpleNamespace(manifest_path=manifest_path, properties=properties)


class MockConfirmationEvaluator:
    configuration_identity = {"mock": "confirmation_v1"}

    def __init__(self, accepted):
        self.accepted = accepted
        self.calls = 0
        self.qa_calls = 0
        self.requests = []

    def evaluate(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        rows = []
        for video in kwargs["confirmation_videos"]:
            for question_id in video.question_ids:
                self.qa_calls += 1
                candidate = 1.0
                if not self.accepted and not rows:
                    candidate = 0.0
                rows.append({
                    "video_id": video.video_id, "question_id": question_id,
                    "incumbent_score": 1.0, "candidate_score": candidate,
                    "error": None,
                })
        return {"qas": rows, "proposals_generated": 0,
                "feedback_calls": 0, "interventions": 0}


def parent_checkpoint(bank, router, scaffold, contract):
    return ConfirmedCheckpointState(
        checkpoint_id="confirmed-root", coverage_cycle=0,
        component_versions=ComponentVersions(
            bank.bank_version, router.router_version,
            scaffold.scaffold_version, contract.contract_version),
        confirmation_video_ids=("c0", "c1"),
        accepted_provisional_iteration_ids=(),
        confirmation_artifact_path="root-confirmation.json")


def run_iteration(tmp_path, *, accepted, output):
    bank, router, scaffold, contract = components()
    baseline = MockBaselineRunner()
    intervention = MockInterventionRunner()
    feedback = MockFeedbackRunner()
    confirmation = MockConfirmationEvaluator(accepted)
    runner = Checkpoint3EOrchestrator(
        baseline_runner=baseline, intervention_runner=intervention,
        feedback_runner=feedback, confirmation_evaluator=confirmation,
        require_real_models=False)
    result = runner.run(
        iteration_id="iteration-final", roles=roles(), coverage_state=coverage(),
        parent_confirmed=parent_checkpoint(bank, router, scaffold, contract),
        prompt_bank=bank, router_policy=router,
        confirmed_prompt_bank=bank, confirmed_router_policy=router,
        scaffold_policy=scaffold, scaffold_contract=contract,
        output_dir=str(tmp_path / output),
        state_dir=str(tmp_path / f"{output}-policy-state"),
        execution_identity={"fixture": "checkpoint3e"},
        baseline_kwargs={}, intervention_kwargs={})
    return result, runner, baseline, intervention, feedback, confirmation, components()


def feedback_manifest(tmp_path, video_id, specs):
    properties = []
    for index, spec in enumerate(specs):
        candidate_id = f"{video_id}-candidate-{index}"
        result_path = write_json(
            tmp_path / video_id / candidate_id / "result.json", {
                "status": "completed", "candidate_property_id": candidate_id,
                "property_text": spec.get(
                    "property_text", f"Reusable property {video_id} {index}."),
                "source_video_id": video_id,
                "source_question_ids": [f"{video_id}q1"],
                "qa_feedback": [{
                    "question_id": f"{video_id}q1", "status": "accepted",
                    "transition": spec["transition"],
                    "feedback_id": f"feedback-{video_id}-{index}",
                    "recommendation": spec["recommendation"],
                    "covered_by_property_ids": spec.get("covered", []),
                    "evidence": "Reusable visual evidence supports this decision.",
                    "s_feedback": ["0_10"],
                }],
            })
        properties.append({"candidate_property_id": candidate_id,
                           "result_path": result_path})
    return write_json(tmp_path / video_id / "manifest.json", {
        "status": "completed", "source_video_id": video_id,
        "properties": properties,
    })


def checkpoint_from_path(path):
    value = json.loads(Path(path).read_text())
    return ConfirmedCheckpointState(
        checkpoint_id=value["checkpoint_id"],
        coverage_cycle=value["coverage_cycle"],
        component_versions=ComponentVersions(**value["component_versions"]),
        confirmation_video_ids=tuple(value["confirmation_video_ids"]),
        accepted_provisional_iteration_ids=tuple(
            value["accepted_provisional_iteration_ids"]),
        confirmation_artifact_path=value["confirmation_artifact_path"],
        parent_confirmed_checkpoint_id=value.get(
            "parent_confirmed_checkpoint_id"),
        status=value.get("status", "confirmed"))


def run_three_iteration_cycle(tmp_path, *, accepted):
    bank, router, scaffold, contract = components()
    parent = parent_checkpoint(bank, router, scaffold, contract)
    baseline = MockBaselineRunner()
    intervention = MockInterventionRunner()
    feedback = MockFeedbackRunner()
    confirmation = MockConfirmationEvaluator(accepted)
    runner = Checkpoint3EOrchestrator(
        baseline_runner=baseline, intervention_runner=intervention,
        feedback_runner=feedback, confirmation_evaluator=confirmation,
        require_real_models=False)
    state_dir = tmp_path / "policy-state"
    state = EvidenceCoverageState(
        rotation_order=tuple(f"e{index}" for index in range(8)))
    common = {
        "roles": roles(), "parent_confirmed": parent,
        "scaffold_policy": scaffold, "scaffold_contract": contract,
        "state_dir": str(state_dir),
        "execution_identity": {"fixture": "three-iteration-cycle"},
        "baseline_kwargs": {}, "intervention_kwargs": {},
    }
    first = runner.run(
        iteration_id="iteration-1", coverage_state=state,
        prompt_bank=bank, router_policy=router,
        confirmed_prompt_bank=bank, confirmed_router_policy=router,
        output_dir=str(tmp_path / "iteration-1"), **common)
    second = runner.run(
        iteration_id="iteration-2", coverage_state=first.next_coverage_state,
        output_dir=str(tmp_path / "iteration-2"), **common)
    return {
        "runner": runner, "baseline": baseline,
        "intervention": intervention, "feedback": feedback,
        "confirmation": confirmation, "bank": bank, "router": router,
        "scaffold": scaffold, "contract": contract, "parent": parent,
        "state_dir": state_dir, "common": common,
        "first": first, "second": second,
    }


def finish_three_iteration_cycle(fixture, tmp_path):
    third = fixture["runner"].run(
        iteration_id="iteration-3",
        coverage_state=fixture["second"].next_coverage_state,
        output_dir=str(tmp_path / "iteration-3"), **fixture["common"])
    fixture["third"] = third
    return third


def test_active_policy_resolution_accumulates_and_accepts_latest_pair(tmp_path):
    fixture = run_three_iteration_cycle(tmp_path, accepted=True)
    first, second = fixture["first"], fixture["second"]
    bank0, router0 = fixture["bank"], fixture["router"]
    bank1 = prompt_bank_from_json(json.loads(Path(
        first.provisional_bank_path).read_text()))
    router1 = router_policy_from_json(json.loads(Path(
        first.provisional_router_path).read_text()))
    bank2 = prompt_bank_from_json(json.loads(Path(
        second.provisional_bank_path).read_text()))
    router2 = router_policy_from_json(json.loads(Path(
        second.provisional_router_path).read_text()))
    assert fixture["baseline"].input_versions == [
        (bank0.bank_version, router0.router_version),
        (bank1.bank_version, router1.router_version),
    ]
    assert bank2.parent_bank_version == bank1.bank_version
    assert router2.parent_router_version == router1.router_version
    reference = json.loads(Path(
        second.active_provisional_reference_path).read_text())
    assert reference["status"] == "active"
    assert reference["provisional_lineage"] == ["iteration-1", "iteration-2"]

    counts = (fixture["baseline"].calls, fixture["intervention"].calls,
              fixture["feedback"].calls, fixture["confirmation"].calls)
    resumed = fixture["runner"].run(
        iteration_id="iteration-2",
        coverage_state=first.next_coverage_state,
        output_dir=str(tmp_path / "iteration-2"), **fixture["common"])
    assert resumed.resumed
    assert counts == (fixture["baseline"].calls, fixture["intervention"].calls,
                      fixture["feedback"].calls,
                      fixture["confirmation"].calls)

    third = finish_three_iteration_cycle(fixture, tmp_path)
    bank3 = prompt_bank_from_json(json.loads(Path(
        third.provisional_bank_path).read_text()))
    router3 = router_policy_from_json(json.loads(Path(
        third.provisional_router_path).read_text()))
    assert fixture["baseline"].input_versions[-1] == (
        bank2.bank_version, router2.router_version)
    request = fixture["confirmation"].requests[0]
    assert request["incumbent_bank"].bank_version == bank0.bank_version
    assert request["incumbent_router"].router_version == router0.router_version
    assert request["candidate_bank"].bank_version == bank3.bank_version
    assert request["candidate_router"].router_version == router3.router_version
    assert third.status == "confirmed"
    pointer = json.loads(Path(third.confirmed_pointer_path).read_text())
    assert pointer["bank"]["version"] == bank3.bank_version
    assert pointer["router"]["version"] == router3.router_version
    closed = json.loads(Path(
        third.active_provisional_reference_path).read_text())
    assert closed["status"] == "accepted"
    assert closed["provisional_lineage"] == [
        "iteration-1", "iteration-2", "iteration-3"]

    confirmed = checkpoint_from_path(third.confirmed_checkpoint_path)
    next_baseline = MockBaselineRunner(suppress_proposals=True)
    next_runner = Checkpoint3EOrchestrator(
        baseline_runner=next_baseline,
        intervention_runner=MockInterventionRunner(),
        feedback_runner=MockFeedbackRunner(),
        confirmation_evaluator=MockConfirmationEvaluator(True),
        require_real_models=False)
    next_result = next_runner.run(
        iteration_id="iteration-4", roles=roles(),
        coverage_state=third.next_coverage_state,
        parent_confirmed=confirmed,
        prompt_bank=bank3, router_policy=router3,
        scaffold_policy=fixture["scaffold"],
        scaffold_contract=fixture["contract"],
        state_dir=str(fixture["state_dir"]),
        output_dir=str(tmp_path / "iteration-4"),
        execution_identity={"fixture": "new-cycle"},
        baseline_kwargs={}, intervention_kwargs={})
    assert next_result.status == "provisional"
    assert next_baseline.input_versions == [
        (bank3.bank_version, router3.router_version)]


def test_rejected_cycle_restores_exact_parent_and_new_cycle_starts_from_it(
        tmp_path):
    fixture = run_three_iteration_cycle(tmp_path, accepted=False)
    pointer_path = Path(fixture["second"].confirmed_pointer_path)
    original_pointer = pointer_path.read_text()
    third = finish_three_iteration_cycle(fixture, tmp_path)
    assert third.status == "rolled_back"
    assert pointer_path.read_text() == original_pointer
    assert Path(third.active_bank_path).resolve() == Path(
        json.loads(original_pointer)["bank"]["path"]).resolve()
    assert Path(third.active_router_path).resolve() == Path(
        json.loads(original_pointer)["router"]["path"]).resolve()
    closed = json.loads(Path(
        third.active_provisional_reference_path).read_text())
    assert closed["status"] == "rejected"
    assert closed["provisional_lineage"] == [
        "iteration-1", "iteration-2", "iteration-3"]

    next_baseline = MockBaselineRunner(suppress_proposals=True)
    next_runner = Checkpoint3EOrchestrator(
        baseline_runner=next_baseline,
        intervention_runner=MockInterventionRunner(),
        feedback_runner=MockFeedbackRunner(),
        confirmation_evaluator=MockConfirmationEvaluator(True),
        require_real_models=False)
    next_runner.run(
        iteration_id="iteration-after-reject", roles=roles(),
        coverage_state=third.next_coverage_state,
        parent_confirmed=fixture["parent"],
        prompt_bank=fixture["bank"], router_policy=fixture["router"],
        scaffold_policy=fixture["scaffold"],
        scaffold_contract=fixture["contract"],
        state_dir=str(fixture["state_dir"]),
        output_dir=str(tmp_path / "iteration-after-reject"),
        execution_identity={"fixture": "after-reject"},
        baseline_kwargs={}, intervention_kwargs={})
    assert next_baseline.input_versions == [(
        fixture["bank"].bank_version, fixture["router"].router_version)]


@pytest.mark.parametrize("conflict", ("stale", "branched", "hash"))
def test_active_provisional_reference_conflicts_fail_closed(tmp_path, conflict):
    root = tmp_path / conflict
    fixture = run_three_iteration_cycle(root, accepted=True)
    reference_path = Path(
        fixture["second"].active_provisional_reference_path)
    reference = json.loads(reference_path.read_text())
    if conflict == "stale":
        reference["coverage_state_hash"] = "0" * 64
    elif conflict == "branched":
        reference["provisional_lineage"] = ["iteration-1", "iteration-1"]
    else:
        reference["active_bank"]["hash"] = "f" * 64
    reference_path.write_text(json.dumps(reference, sort_keys=True))
    with pytest.raises(FinalIterationConflictError):
        finish_three_iteration_cycle(fixture, root)


def test_explicit_policy_cannot_overwrite_active_provisional_cycle(tmp_path):
    fixture = run_three_iteration_cycle(tmp_path, accepted=True)
    with pytest.raises(
            FinalIterationConflictError,
            match="explicit policy cannot overwrite"):
        fixture["runner"].run(
            iteration_id="branched-iteration",
            coverage_state=fixture["second"].next_coverage_state,
            prompt_bank=fixture["bank"], router_policy=fixture["router"],
            output_dir=str(tmp_path / "branched"), **fixture["common"])


def test_fixture_end_to_end_accept_reject_rollback_and_resume(tmp_path):
    accepted = run_iteration(tmp_path, accepted=True, output="accepted")
    result, runner, baseline, intervention, feedback, confirmation, original = accepted
    bank, router, scaffold, contract = original
    assert result.status == "confirmed"
    assert result.selected_video_ids == ("e5", "e6", "e7")
    assert baseline.caption_calls == 3
    assert baseline.proposal_calls == 3
    assert baseline.retrieval_calls == 4
    assert intervention.work_items == 4
    assert intervention.qa_calls == 12
    assert feedback.provider_calls == 4
    assert confirmation.qa_calls == 6
    assert not result.next_coverage_state.confirmation_due
    assert result.next_coverage_state.coverage_cycle == 1

    plan = json.loads(Path(result.update_plan_path).read_text())
    actions = [item["action"] for item in plan["codebook_decisions"]]
    assert actions.count("add") == 2
    assert "retire" not in actions
    assert any("retirement deferred" in item["rationale"]
               for item in plan["codebook_decisions"])
    polarities = {item["polarity"] for item in plan["router_supervision"]}
    assert polarities == {"positive", "negative"}
    provisional = json.loads(Path(result.provisional_state_path).read_text())
    assert provisional["parent_confirmed_checkpoint_id"] == "confirmed-root"
    assert provisional["provisional_versions"]["scaffold_version"] == \
        scaffold.scaffold_version
    candidate_bank = prompt_bank_from_json(json.loads(
        Path(result.provisional_bank_path).read_text()))
    candidate_router = router_policy_from_json(json.loads(
        Path(result.provisional_router_path).read_text()))
    assert candidate_bank.bank_version != bank.bank_version
    assert len(candidate_bank.entries) == len(bank.entries) + 2
    supervision = candidate_router.configuration["supervision_examples"]
    assert {item["polarity"] for item in supervision} == {"positive", "negative"}
    assert all("question_id" not in item and "source_video_id" not in item
               for item in supervision)

    counts = (baseline.calls, baseline.caption_calls, baseline.retrieval_calls,
              intervention.calls, intervention.qa_calls, feedback.calls,
              feedback.provider_calls, confirmation.calls, confirmation.qa_calls)
    resumed = runner.run(
        iteration_id="iteration-final", roles=roles(), coverage_state=coverage(),
        parent_confirmed=parent_checkpoint(bank, router, scaffold, contract),
        prompt_bank=bank, router_policy=router,
        confirmed_prompt_bank=bank, confirmed_router_policy=router,
        scaffold_policy=scaffold, scaffold_contract=contract,
        output_dir=str(tmp_path / "accepted"),
        state_dir=str(tmp_path / "accepted-policy-state"),
        execution_identity={"fixture": "checkpoint3e"},
        baseline_kwargs={}, intervention_kwargs={})
    assert resumed.resumed
    assert counts == (
        baseline.calls, baseline.caption_calls, baseline.retrieval_calls,
        intervention.calls, intervention.qa_calls, feedback.calls,
        feedback.provider_calls, confirmation.calls, confirmation.qa_calls)

    rejected = run_iteration(tmp_path, accepted=False, output="rejected")[0]
    assert rejected.status == "rolled_back"
    rollback_bank = prompt_bank_from_json(json.loads(
        Path(rejected.active_bank_path).read_text()))
    rollback_router = router_policy_from_json(json.loads(
        Path(rejected.active_router_path).read_text()))
    assert rollback_bank.bank_version == bank.bank_version
    assert rollback_router.router_version == router.router_version
    decision = json.loads(Path(rejected.confirmation_decision_path).read_text())
    assert decision["rollback_to_checkpoint_id"] == "confirmed-root"
    assert decision["correct_to_wrong_question_ids"]
    evaluation = json.loads(Path(decision["evaluation_artifact_path"]).read_text())
    assert evaluation["proposals_generated"] == 0
    assert evaluation["feedback_calls"] == 0
    assert evaluation["interventions"] == 0


def test_codebook_action_rules_and_retirement_support(tmp_path):
    bank, router, _, _ = components()
    runner = Checkpoint3EOrchestrator(
        baseline_runner=MockBaselineRunner(),
        intervention_runner=MockInterventionRunner(),
        feedback_runner=MockFeedbackRunner(),
        confirmation_evaluator=MockConfirmationEvaluator(True),
        require_real_models=False)

    retire_manifests = (
        feedback_manifest(tmp_path / "retire", "e0", [{
            "transition": "correct_to_wrong", "recommendation": "retire",
            "covered": ["pe_text"],
        }]),
        feedback_manifest(tmp_path / "retire", "e1", [{
            "transition": "correct_to_wrong", "recommendation": "retire",
            "covered": ["pe_text"],
        }]),
        feedback_manifest(tmp_path / "retire", "e2", []),
    )
    retire_plan = runner.aggregate_updates(
        prompt_bank=bank, router_policy=router,
        feedback_manifest_paths=retire_manifests)
    retire = [item for item in retire_plan.codebook_decisions
              if item.action == "retire"]
    assert len(retire) == 1
    assert retire[0].supporting_video_ids == ("e0", "e1")
    retired_bank, retired_router = runner.apply_update_plan(
        prompt_bank=bank, router_policy=router, plan=retire_plan)
    assert retired_bank.entry("pe_text").status == "retired"
    assert retired_router.router_version != router.router_version

    single_manifests = (
        feedback_manifest(tmp_path / "single", "e0", [{
            "transition": "correct_to_wrong", "recommendation": "retire",
            "covered": ["pe_text"],
        }]),
        feedback_manifest(tmp_path / "single", "e1", []),
        feedback_manifest(tmp_path / "single", "e2", []),
    )
    single_plan = runner.aggregate_updates(
        prompt_bank=bank, router_policy=router,
        feedback_manifest_paths=single_manifests)
    assert all(item.action != "retire" for item in single_plan.codebook_decisions)

    merge_manifests = (
        feedback_manifest(tmp_path / "merge", "e0", [{
            "transition": "wrong_to_correct", "recommendation": "merge",
            "covered": ["pe_temporal", "pe_text"],
            "property_text": "Combine temporal and written detail precisely.",
        }]),
        feedback_manifest(tmp_path / "merge", "e1", []),
        feedback_manifest(tmp_path / "merge", "e2", []),
    )
    merge_plan = runner.aggregate_updates(
        prompt_bank=bank, router_policy=router,
        feedback_manifest_paths=merge_manifests)
    assert any(item.action == "merge" for item in merge_plan.codebook_decisions)
    merged_bank, merged_router = runner.apply_update_plan(
        prompt_bank=bank, router_policy=router, plan=merge_plan)
    merge_decision = next(item for item in merge_plan.codebook_decisions
                          if item.action == "merge")
    assert merged_bank.entry(merge_decision.resulting_property_id).status == "active"
    assert merged_bank.entry("pe_temporal").status == "retired"
    assert merged_bank.entry("pe_text").status == "retired"
    assert merged_router.router_version != router.router_version

    revise_manifests = (
        feedback_manifest(tmp_path / "revise", "e0", [{
            "transition": "wrong_to_correct", "recommendation": "revise",
            "covered": ["pe_temporal"],
            "property_text": "Track temporal changes without incidental motion.",
        }]),
        feedback_manifest(tmp_path / "revise", "e1", []),
        feedback_manifest(tmp_path / "revise", "e2", []),
    )
    revise_plan = runner.aggregate_updates(
        prompt_bank=bank, router_policy=router,
        feedback_manifest_paths=revise_manifests)
    assert any(item.action == "revise" for item in revise_plan.codebook_decisions)
    revised_bank, _ = runner.apply_update_plan(
        prompt_bank=bank, router_policy=router, plan=revise_plan)
    assert revised_bank.entry("pe_temporal").prompt_text == \
        "Track temporal changes without incidental motion."

    covered_add = (
        feedback_manifest(tmp_path / "covered", "e0", [{
            "transition": "wrong_to_correct", "recommendation": "add",
            "covered": ["pe_temporal"],
        }]),
        feedback_manifest(tmp_path / "covered", "e1", []),
        feedback_manifest(tmp_path / "covered", "e2", []),
    )
    covered_plan = runner.aggregate_updates(
        prompt_bank=bank, router_policy=router,
        feedback_manifest_paths=covered_add)
    assert all(item.action != "add" for item in covered_plan.codebook_decisions)


def test_history_router_receives_compact_supervision_without_qa_data():
    bank, router, _, _ = components()

    class VLM:
        def __init__(self):
            self.prompt = None

        def caption(self, frames, prompt, max_tokens):
            self.prompt = prompt
            return '{"property_ids": []}'

    vlm = VLM()
    policy = replace(
        router, policy_type="history_aware_vlm",
        configuration={
            **dict(router.configuration),
            "supervision_examples": ({
                "property_id": "pe_temporal", "polarity": "negative",
                "evidence": "Incidental motion can obscure stable outcomes.",
                "context_hash": "context-hash",
                "source_video_id": "must-not-leak",
                "source_question_id": "must-not-leak",
            },),
        })
    HistoryAwareVLMRouter(vlm).route(
        SegmentContext(
            video_id="video", segment_id="0_10",
            segment_features={"frame_references": ("frame.jpg",)},
            history_summary='{"preceding_captions": []}'),
        bank, policy)
    request = json.loads(vlm.prompt.split("\n", 1)[1])
    assert request["router_supervision"] == [{
        "property_id": "pe_temporal", "polarity": "negative",
        "evidence": "Incidental motion can obscure stable outcomes.",
        "context_hash": "context-hash",
    }]
    assert "question" not in json.dumps(request["router_supervision"])
    assert "must-not-leak" not in json.dumps(request["router_supervision"])
