import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from surrogate_rollout.optimization.property_memory import (
    COMPACT_SUMMARY_SCHEMA_VERSION,
    PROPERTY_MEMORY_SCHEMA_VERSION,
    CompactPropertyMemoryRunner,
    PropertyMemoryBounds,
    PropertyMemoryConflictError,
)
from surrogate_rollout.prompt_routing.persistence import prompt_bank_from_json
from surrogate_rollout.prompt_routing.schemas import PromptEntry
from surrogate_rollout.schemas import sha256_text


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
    source = prompt_bank_from_json(raw["prompt_bank"])
    texts = {
        "pe_default": "Track apple color changes.",
        "pe_temporal": "Capture visible sign words.",
        "pe_text": "Describe background motion.",
    }
    entries = tuple(replace(
        entry, prompt_text=texts[entry.prompt_id],
        prompt_hash=sha256_text(texts[entry.prompt_id]), created_by="seed")
        for entry in source.entries)
    return replace(source, entries=entries, created_by="seed")


def baseline_fixture(tmp_path, video_id="video-a", *, qas=None, proposals=()):
    root = tmp_path / f"baseline-{video_id}"
    routes = [
        {"segment_id": "0_10", "selected_prompt_ids": ["pe_default"]},
        {"segment_id": "10_20", "selected_prompt_ids": ["pe_temporal"]},
        {"segment_id": "20_30", "selected_prompt_ids": ["pe_text"]},
        {"segment_id": "30_40", "selected_prompt_ids": ["pe_text"]},
    ]
    route_path = write_jsonl(root / "composed_prompts.jsonl", routes)
    routing_path = write_json(root / "routing_manifest.json", {
        "composed_prompts_path": route_path})
    captions_path = write_json(root / "captions.json", {
        "0_10": {"caption": "The apple color changes from green to red."},
        "10_20": {"caption": "A visible sign shows OPEN."},
        "20_30": {"caption": "A person sits at a table."},
        "30_40": {"caption": "Background motion passes behind the person."},
    })
    if qas is None:
        qas = (
            ("q-strong", "0_10", "The apple color changes from green to red."),
            ("q-weak", "10_20", "The word OPEN determines the response."),
            ("q-none", "20_30", "The person is sitting."),
        )
    qa_rows = []
    for question_id, segment_id, reasoning in qas:
        trajectory_path = write_jsonl(
            root / "qa" / question_id / "trajectory.jsonl",
            ({"event": "reasoning", "text": reasoning,
              "segment_id": segment_id},))
        result_path = write_json(
            root / "qa" / question_id / "result.json",
            {"prediction": "A", "score": 1.0})
        qa_rows.append({
            "video_id": video_id, "question_id": question_id,
            "prediction": "A", "parsed_answer": "A", "is_correct": True,
            "errors": [], "used_segments": [segment_id],
            "trajectory_path": trajectory_path, "result_path": result_path,
        })
    qas_path = write_jsonl(root / "baseline_qas.jsonl", qa_rows)
    proposal_path = write_json(root / "proposals.json", {
        "video_id": video_id, "proposals": list(proposals)})
    return write_json(root / "video_complete.json", {
        "video_id": video_id, "routing_manifest_path": routing_path,
        "captions_path": captions_path, "baseline_qas_path": qas_path,
        "property_proposal_path": proposal_path,
    })


def proposal(candidate_id, video_id, text="Track object completion states."):
    return {
        "candidate_property_id": candidate_id, "property_text": text,
        "source_video_id": video_id, "source_question_ids": ["q1"],
        "motivating_failure_types": ["missing_state"],
        "covered_by_existing_property_ids": ["pe_default"],
        "proposal_rationale": "Reusable state completion was omitted.",
        "proposer_policy_version": "fixture_v1",
    }


def intervention_fixture(tmp_path, video_id, candidates):
    results = []
    root = tmp_path / f"interventions-{video_id}"
    for candidate_id, counts in candidates:
        qas = []
        for label, count in counts.items():
            qas.extend({"question_id": f"{candidate_id}-{label}-{index}",
                        "transition": label} for index in range(count))
        transitions_path = write_json(
            root / candidate_id / "transitions.json", {
                "candidate_property_id": candidate_id,
                "source_video_id": video_id,
                "transition_counts": counts, "qas": qas,
            })
        result_path = write_json(root / candidate_id / "result.json", {
            "schema_version": "property_intervention_result_v1",
            "status": "completed", "candidate_property_id": candidate_id,
            "source_video_id": video_id, "transitions_path": transitions_path,
            "transition_counts": counts,
        })
        results.append({"candidate_property_id": candidate_id,
                        "result_path": result_path})
    return write_json(root / "manifest.json", {
        "status": "completed", "source_video_id": video_id,
        "results": results,
    })


def counts(wc=0, cw=0, cc=0, ww=0):
    return {
        "wrong_to_correct": wc, "correct_to_wrong": cw,
        "correct_to_correct": cc, "wrong_to_wrong": ww,
    }


def load_snapshot(result):
    return json.loads(Path(result["property_memory_snapshot_path"]).read_text())


def memory(snapshot, property_id):
    return next(item for item in snapshot["properties"]
                if item["property_id"] == property_id)


def test_correct_qa_credit_strong_weak_none_and_route_alone(tmp_path):
    baseline = baseline_fixture(tmp_path)
    result = CompactPropertyMemoryRunner().run(
        iteration_id="i1", iteration_ordinal=1, prompt_bank=bank(),
        baseline_video_manifest_paths=(baseline,),
        intervention_manifest_paths=(), output_dir=str(tmp_path / "memory"))
    rows = [json.loads(line) for line in Path(
        result["credit_summary_path"]).read_text().splitlines()]
    by_key = {(row["question_id"], row["property_id"]): row for row in rows}
    assert by_key[("q-strong", "pe_default")]["credit"] == "strong"
    assert by_key[("q-weak", "pe_temporal")]["credit"] == "weak"
    assert by_key[("q-none", "pe_text")]["credit"] == "none"
    assert by_key[("q-strong", "pe_text")]["credit"] == "none"
    assert by_key[("q-strong", "pe_text")]["supporting_segment_ids"] == []


def test_intervention_effects_and_candidate_memory_stay_separate(tmp_path):
    props = (
        proposal("cp-pos", "video-a"), proposal("cp-neg", "video-a"),
        proposal("cp-mix", "video-a"), proposal("cp-none", "video-a"),
    )
    baseline = baseline_fixture(tmp_path, proposals=props)
    intervention = intervention_fixture(tmp_path, "video-a", (
        ("cp-pos", counts(wc=1, cc=2)),
        ("cp-neg", counts(cw=1, cc=2)),
        ("cp-mix", counts(wc=1, cw=1, ww=1)),
        ("cp-none", counts(cc=2, ww=1)),
    ))
    result = CompactPropertyMemoryRunner().run(
        iteration_id="i1", iteration_ordinal=1, prompt_bank=bank(),
        baseline_video_manifest_paths=(baseline,),
        intervention_manifest_paths=(intervention,),
        output_dir=str(tmp_path / "memory"))
    effects = [json.loads(line) for line in Path(
        result["effect_summary_path"]).read_text().splitlines()]
    assert {item["candidate_property_id"]: item["effect"] for item in effects} == {
        "cp-pos": "positive", "cp-neg": "negative",
        "cp-mix": "mixed", "cp-none": "no_effect",
    }
    snapshot = load_snapshot(result)
    assert {item["candidate_property_id"] for item in snapshot["candidates"]} == {
        "cp-pos", "cp-neg", "cp-mix", "cp-none"}
    assert all(item["status"] == "candidate" for item in snapshot["candidates"])
    assert all(item["property_id"] not in {"cp-pos", "cp-neg", "cp-mix", "cp-none"}
               for item in snapshot["properties"])


def test_promoted_and_related_effects_accumulate_without_changing_plan(tmp_path):
    source_bank = bank()
    props = (proposal("cp-add", "video-a"), proposal("cp-harm", "video-a"),
             proposal("cp-no", "video-a"))
    baseline = baseline_fixture(tmp_path, proposals=props)
    intervention = intervention_fixture(tmp_path, "video-a", (
        ("cp-add", counts(wc=1, cc=2)),
        ("cp-harm", counts(cw=1, cc=2)),
        ("cp-no", counts(cc=2, ww=1)),
    ))
    learned_text = props[0]["property_text"]
    learned = PromptEntry(
        prompt_id="pe_learned", prompt_text=learned_text,
        prompt_hash=sha256_text(learned_text), name="Learned",
        description="Learned", created_by="checkpoint3e")
    resulting = replace(
        source_bank, bank_version="bank_v0003_deadbeef",
        parent_bank_version=source_bank.bank_version,
        entries=(*source_bank.entries, learned))
    plan = {
        "codebook_decisions": [
            {"action": "add", "candidate_lineage": [{
                "source_video_id": "video-a", "candidate_property_id": "cp-add"}],
             "target_property_ids": [], "resulting_property_id": "pe_learned",
             "rationale": "positive evidence"},
            {"action": "no_op", "candidate_lineage": [{
                "source_video_id": "video-a", "candidate_property_id": "cp-harm"}],
             "target_property_ids": ["pe_default"], "resulting_property_id": None,
             "rationale": "covered harmful evidence"},
            {"action": "no_op", "candidate_lineage": [{
                "source_video_id": "video-a", "candidate_property_id": "cp-no"}],
             "target_property_ids": ["pe_default"], "resulting_property_id": None,
             "rationale": "covered no effect"},
        ],
        "router_supervision": [],
    }
    plan_path = write_json(tmp_path / "update_plan.json", plan)
    result = CompactPropertyMemoryRunner().run(
        iteration_id="i1", iteration_ordinal=1, prompt_bank=source_bank,
        baseline_video_manifest_paths=(baseline,),
        intervention_manifest_paths=(intervention,),
        output_dir=str(tmp_path / "memory"), update_plan=plan,
        update_plan_path=plan_path, resulting_prompt_bank=resulting)
    snapshot = load_snapshot(result)
    learned_memory = memory(snapshot, "pe_learned")
    default_memory = memory(snapshot, "pe_default")
    assert learned_memory["origin"]["kind"] == "promoted_candidate"
    assert any(item["effect"] == "positive"
               for item in learned_memory["positive_examples"])
    assert any(item["effect"] == "negative"
               for item in default_memory["negative_examples"])
    assert any(item["effect"] == "no_effect"
               for item in default_memory["no_effect_examples"])
    promoted = next(item for item in snapshot["candidates"]
                    if item["candidate_property_id"] == "cp-add")
    assert promoted["status"] == "promoted"
    assert promoted["promoted_to_property_id"] == "pe_learned"


def test_bounded_selection_prefers_distinct_videos_and_audits_eviction(tmp_path):
    qas_a = (
        ("qa1", "0_10", "The apple color changes from green to red."),
        ("qa2", "0_10", "The apple color changes from green to red."),
        ("qa3", "0_10", "The apple color changes from green to red."),
    )
    baseline_a = baseline_fixture(tmp_path, "video-a", qas=qas_a)
    baseline_b = baseline_fixture(tmp_path, "video-b", qas=(
        ("qb1", "0_10", "The apple color changes from green to red."),))
    runner = CompactPropertyMemoryRunner(bounds=PropertyMemoryBounds(
        strong_positive=2, weak_positive=0, harmful=1, no_effect=1,
        routing_positive=1, routing_negative=1))
    result = runner.run(
        iteration_id="i1", iteration_ordinal=1, prompt_bank=bank(),
        baseline_video_manifest_paths=(baseline_a, baseline_b),
        intervention_manifest_paths=(), output_dir=str(tmp_path / "memory"))
    examples = memory(load_snapshot(result), "pe_default")["positive_examples"]
    assert {item["video_id"] for item in examples} == {"video-a", "video-b"}
    audit = json.loads(Path(result["selection_audit_path"]).read_text())
    evicted = [item for item in audit["decisions"]
               if item["property_id"] == "pe_default" and not item["retained"]]
    assert evicted
    assert all("raw compact summary remains immutable" in item["reason"]
               for item in evicted)


def test_seed_origin_artifact_hashes_deterministic_rebuild_and_resume(tmp_path):
    baseline = baseline_fixture(tmp_path)
    runner = CompactPropertyMemoryRunner()
    result = runner.run(
        iteration_id="i1", iteration_ordinal=1, prompt_bank=bank(),
        baseline_video_manifest_paths=(baseline,), intervention_manifest_paths=(),
        output_dir=str(tmp_path / "memory-a"))
    snapshot = load_snapshot(result)
    assert memory(snapshot, "pe_default")["origin"]["kind"] == "seed_or_legacy"
    row = json.loads(Path(result["credit_summary_path"]).read_text().splitlines()[0])
    assert row["schema_version"] == COMPACT_SUMMARY_SCHEMA_VERSION
    for ref in row["artifact_refs"]:
        assert hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
    resumed = runner.run(
        iteration_id="i1", iteration_ordinal=1, prompt_bank=bank(),
        baseline_video_manifest_paths=(baseline,), intervention_manifest_paths=(),
        output_dir=str(tmp_path / "memory-a"))
    assert resumed["resumed"] is True
    rebuilt = runner.run(
        iteration_id="i1", iteration_ordinal=1, prompt_bank=bank(),
        baseline_video_manifest_paths=(baseline,), intervention_manifest_paths=(),
        output_dir=str(tmp_path / "memory-b"))
    left = json.loads(Path(result["property_memory_snapshot_path"]).read_text())
    right = json.loads(Path(rebuilt["property_memory_snapshot_path"]).read_text())
    left.pop("selection_audit_ref")
    right.pop("selection_audit_ref")
    for value in (left, right):
        for ref in value["summary_artifact_refs"]:
            ref.pop("path")
    assert left == right


def test_parent_accumulation_and_incompatible_schema_fail_closed(tmp_path):
    first_baseline = baseline_fixture(tmp_path, "video-a", qas=(
        ("q1", "0_10", "The apple color changes from green to red."),))
    runner = CompactPropertyMemoryRunner()
    first = runner.run(
        iteration_id="i1", iteration_ordinal=1, prompt_bank=bank(),
        baseline_video_manifest_paths=(first_baseline,),
        intervention_manifest_paths=(), output_dir=str(tmp_path / "memory-1"))
    second_baseline = baseline_fixture(tmp_path, "video-b", qas=(
        ("q2", "20_30", "The person is sitting."),))
    second = runner.run(
        iteration_id="i2", iteration_ordinal=2, prompt_bank=bank(),
        baseline_video_manifest_paths=(second_baseline,),
        intervention_manifest_paths=(), output_dir=str(tmp_path / "memory-2"),
        parent_memory_path=first["property_memory_snapshot_path"])
    default = memory(load_snapshot(second), "pe_default")
    assert any(item["video_id"] == "video-a" for item in default["positive_examples"])
    broken = json.loads(Path(first["property_memory_snapshot_path"]).read_text())
    broken["schema_version"] = "property_memory_v999"
    broken_path = write_json(tmp_path / "broken-parent.json", broken)
    with pytest.raises(PropertyMemoryConflictError, match="incompatible parent"):
        runner.run(
            iteration_id="i3", iteration_ordinal=3, prompt_bank=bank(),
            baseline_video_manifest_paths=(second_baseline,),
            intervention_manifest_paths=(), output_dir=str(tmp_path / "memory-3"),
            parent_memory_path=broken_path)


def test_effect_and_routing_examples_accumulate_across_iterations(tmp_path):
    source_bank = bank()
    first_proposal = proposal("cp-positive", "video-a")
    first_baseline = baseline_fixture(
        tmp_path, "video-a", proposals=(first_proposal,))
    first_intervention = intervention_fixture(tmp_path, "video-a", (
        ("cp-positive", counts(wc=1, cc=2)),))
    first_plan = {
        "codebook_decisions": [{
            "action": "no_op", "candidate_lineage": [{
                "source_video_id": "video-a",
                "candidate_property_id": "cp-positive"}],
            "target_property_ids": ["pe_default"],
            "resulting_property_id": None, "rationale": "covered positive",
        }],
        "router_supervision": [{
            "property_id": "pe_default", "polarity": "positive",
            "evidence": "Color-change captions helped answer state questions.",
            "context_hash": "ctx-positive", "source_video_id": "video-a",
            "source_question_id": "q1", "attributed_segment_ids": ["0_10"],
            "source_feedback_id": "feedback-positive",
        }],
    }
    first_plan_path = write_json(tmp_path / "first-plan.json", first_plan)
    feedback_request = write_json(tmp_path / "feedback" / "request.json", {"x": 1})
    feedback_raw = write_json(tmp_path / "feedback" / "raw.json", {"raw": True})
    feedback_parsed = write_json(tmp_path / "feedback" / "parsed.json", {
        "feedback_id": "feedback-positive"})
    feedback_result = write_json(tmp_path / "feedback" / "result.json", {
        "qa_feedback": [{
            "feedback_id": "feedback-positive", "request_path": feedback_request,
            "raw_output_path": feedback_raw, "parsed_output_path": feedback_parsed,
        }]})
    feedback_manifest = write_json(tmp_path / "feedback" / "manifest.json", {
        "properties": [{"result_path": feedback_result}]})
    runner = CompactPropertyMemoryRunner()
    first = runner.run(
        iteration_id="i1", iteration_ordinal=1, prompt_bank=source_bank,
        baseline_video_manifest_paths=(first_baseline,),
        intervention_manifest_paths=(first_intervention,),
        feedback_manifest_paths=(feedback_manifest,),
        output_dir=str(tmp_path / "memory-1"), update_plan=first_plan,
        update_plan_path=first_plan_path)

    second_props = (proposal("cp-negative", "video-b"),
                    proposal("cp-none", "video-b"))
    second_baseline = baseline_fixture(
        tmp_path, "video-b", proposals=second_props)
    second_intervention = intervention_fixture(tmp_path, "video-b", (
        ("cp-negative", counts(cw=1, cc=2)),
        ("cp-none", counts(cc=2, ww=1)),
    ))
    second_plan = {
        "codebook_decisions": [{
            "action": "no_op", "candidate_lineage": [{
                "source_video_id": "video-b", "candidate_property_id": candidate}],
            "target_property_ids": ["pe_default"], "resulting_property_id": None,
            "rationale": "covered observation",
        } for candidate in ("cp-negative", "cp-none")],
        "router_supervision": [{
            "property_id": "pe_default", "polarity": "negative",
            "evidence": "The routed detail distracted from the stable outcome.",
            "context_hash": "ctx-negative", "source_video_id": "video-b",
            "source_question_id": "q2", "attributed_segment_ids": ["10_20"],
        }],
    }
    second_plan_path = write_json(tmp_path / "second-plan.json", second_plan)
    second = runner.run(
        iteration_id="i2", iteration_ordinal=2, prompt_bank=source_bank,
        baseline_video_manifest_paths=(second_baseline,),
        intervention_manifest_paths=(second_intervention,),
        output_dir=str(tmp_path / "memory-2"), update_plan=second_plan,
        update_plan_path=second_plan_path,
        parent_memory_path=first["property_memory_snapshot_path"])
    retained = memory(load_snapshot(second), "pe_default")
    assert any(item.get("effect") == "positive"
               for item in retained["positive_examples"])
    assert any(item.get("effect") == "negative"
               for item in retained["negative_examples"])
    assert any(item.get("effect") == "no_effect"
               for item in retained["no_effect_examples"])
    assert len(retained["routing_examples"]["positive"]) == 1
    assert len(retained["routing_examples"]["negative"]) == 1
    assert "feedback_parsed_output" in {
        ref["role"] for ref in retained["routing_examples"]["positive"][0][
            "artifact_refs"]}


def test_zero_proposal_noop_build_does_not_invent_candidate_memory(tmp_path):
    baseline = baseline_fixture(tmp_path, proposals=())
    empty_interventions = intervention_fixture(tmp_path, "video-a", ())
    result = CompactPropertyMemoryRunner().run(
        iteration_id="i1", iteration_ordinal=1, prompt_bank=bank(),
        baseline_video_manifest_paths=(baseline,),
        intervention_manifest_paths=(empty_interventions,),
        output_dir=str(tmp_path / "memory"),
        update_plan={"codebook_decisions": [], "router_supervision": []})
    snapshot = load_snapshot(result)
    assert snapshot["candidates"] == []
    assert Path(result["effect_summary_path"]).read_text() == ""
    assert snapshot["schema_version"] == PROPERTY_MEMORY_SCHEMA_VERSION


def test_incomplete_memory_output_fails_closed(tmp_path):
    baseline = baseline_fixture(tmp_path)
    output = tmp_path / "partial-memory"
    output.mkdir()
    (output / "partial.json").write_text("{}")
    with pytest.raises(PropertyMemoryConflictError, match="incomplete memory"):
        CompactPropertyMemoryRunner().run(
            iteration_id="i1", iteration_ordinal=1, prompt_bank=bank(),
            baseline_video_manifest_paths=(baseline,),
            intervention_manifest_paths=(), output_dir=str(output))


def test_exact_resume_detects_nested_raw_artifact_change(tmp_path):
    baseline = baseline_fixture(tmp_path)
    output = tmp_path / "memory"
    runner = CompactPropertyMemoryRunner()
    result = runner.run(
        iteration_id="i1", iteration_ordinal=1, prompt_bank=bank(),
        baseline_video_manifest_paths=(baseline,), intervention_manifest_paths=(),
        output_dir=str(output))
    credit = json.loads(Path(result["credit_summary_path"]).read_text().splitlines()[0])
    trajectory = next(ref["path"] for ref in credit["artifact_refs"]
                      if ref["role"] == "qa_reasoning")
    Path(trajectory).write_text('{"event":"changed"}\n')
    with pytest.raises(PropertyMemoryConflictError, match="raw artifact hash mismatch"):
        runner.run(
            iteration_id="i1", iteration_ordinal=1, prompt_bank=bank(),
            baseline_video_manifest_paths=(baseline,), intervention_manifest_paths=(),
            output_dir=str(output))
