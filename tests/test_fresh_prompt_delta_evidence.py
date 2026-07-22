import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from surrogate_rollout.optimization.fresh_prompt_delta_evidence import (
    FreshPromptDeltaError, LLMPromptDeltaProposer,
    OpenAICompatiblePromptDeltaProposalBackend,
    PromptDeltaExecutionPlan, PromptDeltaInterventionRunner,
    PromptDeltaProposalBackend,
    PromptDeltaRequestPreflight, build_prompt_delta_proposal_request,
    build_prompt_delta_single_qa_request,
    _qa_segment_selection_records,
    _source_qa_classification_hash,
    PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
    reconstruct_prompt_delta_history_snapshot,
)
from surrogate_rollout.optimization.checkpoint_g_factory import (
    _exact_counter, _load_configuration,
)
from surrogate_rollout.optimization.llm_episode_feedback import (
    SavedEpisodeFeedbackArtifactResolver,
)
from surrogate_rollout.optimization.schemas import (
    InterventionClipRecord, InterventionEpisode, PromptDelta,
    QAInterventionOutcome,
)
from surrogate_rollout.prompt_routing.schemas import (
    ComposedCaptionPrompt, CompositionTrace, dumps_canonical,
)
from surrogate_rollout.schemas import sha256_text
from surrogate_rollout.optimization.baseline_phase import (
    load_completed_baseline_for_read_only_resume,
)


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return str(path.resolve())


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(dumps_canonical(row) + "\n" for row in rows),
                    encoding="utf-8")
    return str(path.resolve())


def _baseline(tmp_path: Path):
    segment_ids = ("0_10", "10_20", "20_30")
    prompts = []
    for segment_id in segment_ids:
        text = f"base prompt {segment_id}"
        prompts.append(ComposedCaptionPrompt(
            video_id="v1", segment_id=segment_id,
            bank_version="bank_v9999", router_version="router_v9999",
            scaffold_version="scaffold_v0001",
            contract_version="contract_v0001", selected_prompt_ids=(),
            prompt_text=text, prompt_hash=sha256_text(text),
            composition_trace=CompositionTrace(
                selected_prompt_ids=(), preserved_prompt_ids=())))
    composed = _jsonl(tmp_path / "composed.jsonl", prompts)
    routing = _json(tmp_path / "routing.json", {
        "composed_prompts_path": composed})
    histories = _jsonl(tmp_path / "histories.jsonl", ({
        "segment_id": segment_id,
        "history": [{"segment_id": "prior", "caption": "shared history"}],
        "history_hash": "history-shared",
        "opaque_metadata": "preserved",
    } for segment_id in segment_ids))
    captions = _json(tmp_path / "captions.json", {
        segment_id: {"caption": f"caption {segment_id}"}
        for segment_id in segment_ids if segment_id != "20_30"})
    trajectory_qa1 = _jsonl(tmp_path / "qa1/trajectory.jsonl", [{
        "role": "assistant", "content": "Inspect 00:00:10 and 00:00:00."}])
    trajectory_qa2 = _jsonl(tmp_path / "qa2/trajectory.jsonl", [{
        "role": "assistant", "content": "Inspect 00:00:10."}])
    _jsonl(tmp_path / "qa1/tool_events.jsonl", [{
        "tool": "frame_inspect_tool",
        "args": {"time_ranges_hhmmss": [["00:00:10", "00:00:19"]]},
    }])
    _jsonl(tmp_path / "qa2/tool_events.jsonl", [{
        "tool": "frame_inspect_tool",
        "args": {"time_ranges_hhmmss": [["00:00:00", "00:00:30"]]},
    }])
    qas = _jsonl(tmp_path / "qas.jsonl", ({
        "question_id": "qa1", "question": "What happened?",
        "options": ["A", "B"], "ground_truth": "A", "prediction": "B",
        "is_correct": False, "trajectory_path": trajectory_qa1,
        "reference_sets": {
            "explicitly_cited_segments": ["10_20", "0_10"],
            "frame_inspected_segments": ["0_10"],
            "retrieved_segments": ["20_30"],
            "returned_segments": ["20_30"],
            "consumed_segments": ["10_20", "0_10", "20_30"],
        },
        "used_segments": ["10_20", "0_10"], "provider_index": 1,
    }, {
        "question_id": "qa2", "question": "Who remained?",
        "options": ["A", "B"], "ground_truth": "B", "prediction": "B",
        "is_correct": True, "trajectory_path": trajectory_qa2,
        "reference_sets": {
            "explicitly_cited_segments": ["10_20"],
            "frame_inspected_segments": [],
            "retrieved_segments": ["20_30"],
            "returned_segments": ["20_30"],
            "consumed_segments": ["10_20", "20_30"],
        },
        "used_segments": ["10_20"], "provider_index": 2,
    }))
    manifest = _json(tmp_path / "video_complete.json", {
        "video_id": "v1", "routing_manifest_path": routing,
        "frozen_histories_path": histories, "captions_path": captions,
        "baseline_qas_path": qas})
    return manifest


class FakeBackend:
    policy_version = "delta_policy_v1"

    def __init__(self, raw=None, *, full_fits=True, oversized_qa=()):
        self.raw = raw
        self.full_fits = full_fits
        self.oversized_qa = set(oversized_qa)
        self.calls = 0
        self.payloads = []

    @property
    def configuration_identity(self):
        return {"backend": "deterministic_fake", "policy_version": self.policy_version}

    def preflight(self, system, payload):
        value = json.loads(payload)
        qas = tuple(value["request_scope"]["qa_ids"])
        fits = (self.full_fits if value["request_scope"]["mode"] == "all_qas"
                else (not bool(set(qas) & self.oversized_qa) or
                      "[TRUNCATED_TO_CONTEXT]" in payload))
        total = 50 if fits else 150
        return PromptDeltaRequestPreflight(
            system_prompt_tokens=10, user_payload_tokens=total - 20,
            total_input_tokens=total, reserved_output_tokens=10,
            context_limit=100, remaining_tokens=100-total-10,
            fits_context=fits)

    def generate(self, system, payload, *, minimum_proposals,
                 maximum_proposals):
        self.calls += 1
        value = json.loads(payload)
        self.payloads.append(value)
        qa_id = value["request_scope"]["qa_ids"][0]
        configured_raw = (
            self.raw.get(qa_id) if isinstance(self.raw, dict) else self.raw)
        if configured_raw is not None:
            return configured_raw
        return json.dumps({"proposals": [{
            "instruction": f"Inspect visible continuity for {qa_id}.",
            "source_qa_ids": [qa_id],
            "proposer_diagnosis": f"Stored trajectory evidence for {qa_id}.",
        }]})


def test_fresh_request_and_proposal_are_complete_and_deterministic(tmp_path):
    manifest = _baseline(tmp_path)
    source_before = {path: path.read_bytes() for path in tmp_path.glob("*.json*")}
    system, payload, digest = build_prompt_delta_proposal_request(
        manifest, parent_meta_prompt_id="meta_parent",
        parent_meta_prompt_text="Current parent instruction.",
        global_inspection_boundary_tolerance_seconds=10)
    assert "not a" in system and "codebook" in system
    assert payload["parent_meta_prompt_id"] == "meta_parent"
    assert payload["parent_meta_prompt_text"] == "Current parent instruction."
    assert payload["schema_version"].startswith("prompt_delta_proposal_request_v5")
    assert [row["segment_id"] for row in payload["segment_catalog"]] == [
        "0_10", "10_20"]
    assert "20_30" not in {
        row["segment_id"] for row in payload["segment_catalog"]}
    assert all("used_segment_refs" not in row for row in payload["qa_records"])
    assert [row["qa_id"] for row in payload["qa_records"]] == ["qa1", "qa2"]
    assert payload["qa_records"][0]["intervention_candidate_segment_refs"] == [
        "0_10", "10_20"]
    assert payload["qa_records"][1]["intervention_candidate_segment_refs"] == [
        "10_20"]
    assert len(payload["segment_catalog"]) == 2
    assert len(payload["history_catalog"]["snapshots"]) == 1
    assert reconstruct_prompt_delta_history_snapshot(
        payload, segment_id="10_20")["opaque_metadata"] == "preserved"
    assert dumps_canonical(reconstruct_prompt_delta_history_snapshot(
        payload, segment_id="0_10")) == dumps_canonical(
            json.loads(Path(tmp_path / "histories.jsonl").read_text().splitlines()[0]))
    assert digest == build_prompt_delta_proposal_request(
        manifest, parent_meta_prompt_id="meta_parent",
        parent_meta_prompt_text="Current parent instruction.",
        global_inspection_boundary_tolerance_seconds=10)[2]
    backend = FakeBackend({"qa1": json.dumps({"proposals": [{
        "instruction": "Inspect the visible handoff.",
        "source_qa_ids": ["qa1"],
        "proposer_diagnosis": "The handoff was absent.",
    }]})})
    proposer = LLMPromptDeltaProposer(
        backend=backend, maximum_deltas_per_qa=1,
        selection_policy=PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
        global_inspection_boundary_tolerance_seconds=10)
    plans = proposer.propose(
        manifest, parent_meta_prompt_id="meta_parent",
        parent_meta_prompt_text="Current parent instruction.",
        output_directory=str(tmp_path / "proposal"))
    assert plans[0].selected_segment_ids == ("0_10", "10_20")
    assert plans[0].prompt_delta.source_qa_ids == ("qa1",)
    assert "property" not in dumps_canonical(plans[0])
    assert proposer.propose(
        manifest, parent_meta_prompt_id="meta_parent",
        parent_meta_prompt_text="Current parent instruction.",
        output_directory=str(tmp_path / "proposal")) == plans
    assert backend.calls == 2
    assert [row["request_scope"]["qa_ids"] for row in backend.payloads] == [
        ["qa1"], ["qa2"]]
    assert all(len(row["qa_records"]) == 1 for row in backend.payloads)
    assert backend.payloads[0]["qa_records"][0]["qa_id"] == "qa1"
    assert backend.payloads[1]["qa_records"][0]["qa_id"] == "qa2"
    audit = json.loads(
        (tmp_path / "proposal/selection_audit.json").read_text())
    qa1_audit = next(row for row in audit["qa_records"]
                     if row["qa_id"] == "qa1")
    qa2_audit = next(row for row in audit["qa_records"]
                     if row["qa_id"] == "qa2")
    assert qa1_audit["localized_frame_inspected_segments"] == ["10_20"]
    assert qa1_audit["assistant_timestamp_cited_segments"] == [
        "0_10", "10_20"]
    assert qa2_audit["global_frame_inspected_segments"] == [
        "0_10", "10_20", "20_30"]
    assert qa2_audit["localized_frame_inspected_segments"] == []
    assert qa2_audit["intervention_candidate_segments"] == ["10_20"]
    assert qa2_audit["frame_inspect_calls"][0]["classification"] == "global"
    request_identity = json.loads(
        (tmp_path / "proposal/request_manifest.json").read_text())[
            "request_identity"]
    assert request_identity["global_inspection_classification_hash"] == \
        audit["global_inspection_classification_hash"]
    assert request_identity[
        "global_inspection_boundary_tolerance_seconds"] == 10
    assert {path: path.read_bytes() for path in source_before} == source_before


def test_schema_valid_candidates_are_not_normalized_or_deduplicated(
        tmp_path):
    manifest = _baseline(tmp_path)
    backend = FakeBackend({"qa1": json.dumps({"proposals": [{
        "instruction": "Inspect   the visible handoff.",
        "source_qa_ids": ["qa1"], "proposer_diagnosis": "first",
    }, {
        "instruction": "Inspect the visible handoff.",
        "source_qa_ids": ["qa1"], "proposer_diagnosis": "second",
    }]})})
    proposer = LLMPromptDeltaProposer(
        backend=backend, maximum_deltas_per_qa=2,
        selection_policy=PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
        global_inspection_boundary_tolerance_seconds=10)
    plans = proposer.propose(
        manifest, parent_meta_prompt_id="meta_parent",
        parent_meta_prompt_text="Current parent instruction.",
        output_directory=str(tmp_path / "dedup"))
    qa1_plans = [plan for plan in plans
                 if plan.prompt_delta.source_qa_ids == ("qa1",)]
    assert len(qa1_plans) == 2
    assert [plan.prompt_delta.instruction for plan in qa1_plans] == [
        "Inspect   the visible handoff.", "Inspect the visible handoff."]
    assert len({plan.prompt_delta.delta_id for plan in qa1_plans}) == 2


def test_provider_preflight_uses_exact_injected_counter_without_call():
    class Count:
        system_prompt_tokens = 7
        user_payload_tokens = 11
        total_input_tokens = 25

    transports = []
    backend = OpenAICompatiblePromptDeltaProposalBackend(
        provider="fixture_provider", provider_endpoint="memory://fixture",
        model_id="gpt-fixture", context_limit=40,
        maximum_output_tokens=10, generation_settings={"temperature": 0.0},
        policy_version="normalized_v2", exact_token_counter=lambda _m: Count(),
        response_transport=lambda body: transports.append(body),
        tokenizer_identity="exact-fixture-tokenizer", maximum_calls=1)
    measured = backend.preflight("system", "payload")
    assert measured == PromptDeltaRequestPreflight(
        system_prompt_tokens=7, user_payload_tokens=11,
        total_input_tokens=25, reserved_output_tokens=10,
        context_limit=40, remaining_tokens=5, fits_context=True)
    assert transports == []


def test_proposer_always_calls_each_qa_in_stable_isolated_order(tmp_path):
    manifest = _baseline(tmp_path)
    backend = FakeBackend(full_fits=False)
    proposer = LLMPromptDeltaProposer(
        backend=backend, maximum_deltas_per_qa=2,
        selection_policy=PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
        global_inspection_boundary_tolerance_seconds=10)
    plans = proposer.propose(
        manifest, parent_meta_prompt_id="meta_parent",
        parent_meta_prompt_text="Current parent instruction.",
        output_directory=str(tmp_path / "split"))

    assert [row["request_scope"]["qa_ids"] for row in backend.payloads] == [
        ["qa1"], ["qa2"]]
    assert [plan.prompt_delta.source_qa_ids for plan in plans] == [
        ("qa1",), ("qa2",)]
    manifest_value = json.loads(
        (tmp_path / "split/request_manifest.json").read_text())
    assert manifest_value["request_identity"]["split_mode"] == \
        "isolated_per_qa"
    assert manifest_value[
        "truncation_filtering_sampling_or_intra_qa_split"] is False
    assert all(row["minimum_proposals"] == 1 for row in
               manifest_value["requests"])
    assert all(row["maximum_proposals"] == 2 for row in
               manifest_value["requests"])
    serialized = [dumps_canonical(row) for row in backend.payloads]
    assert "Who remained?" not in serialized[0]
    assert "What happened?" not in serialized[1]


def test_three_eligible_qas_create_three_sibling_free_requests(tmp_path):
    manifest = Path(_baseline(tmp_path))
    baseline = json.loads(manifest.read_text())
    qa_path = Path(baseline["baseline_qas_path"])
    qas = [json.loads(line) for line in qa_path.read_text().splitlines()]
    qa3_trajectory = _jsonl(tmp_path / "qa3/trajectory.jsonl", ({
        "role": "assistant", "content": "Inspect 00:00:00."},))
    qas.append({
        **qas[0], "question_id": "qa3", "question": "Where did it occur?",
        "provider_index": 3, "trajectory_path": qa3_trajectory,
        "reference_sets": {
            **qas[0]["reference_sets"],
            "explicitly_cited_segments": ["0_10"],
            "frame_inspected_segments": [],
        },
    })
    qa_path.write_text(
        "".join(dumps_canonical(row) + "\n" for row in qas),
        encoding="utf-8")
    source_before = {path: path.read_bytes() for path in tmp_path.rglob("*")
                     if path.is_file()}
    backend = FakeBackend()
    proposer = LLMPromptDeltaProposer(
        backend=backend, maximum_deltas_per_qa=2,
        selection_policy=PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
        global_inspection_boundary_tolerance_seconds=10)

    plans = proposer.propose(
        str(manifest), parent_meta_prompt_id="meta_parent",
        parent_meta_prompt_text="Current parent instruction.",
        output_directory=str(tmp_path / "three-qa-proposals"))

    assert backend.calls == 3
    assert [row["request_scope"]["qa_ids"] for row in backend.payloads] == [
        ["qa1"], ["qa2"], ["qa3"]]
    questions = ["What happened?", "Who remained?", "Where did it occur?"]
    for index, payload in enumerate(backend.payloads):
        serialized = dumps_canonical(payload)
        assert payload["qa_records"][0]["question"] == questions[index]
        assert all(question not in serialized
                   for question in questions if question != questions[index])
    assert {plan.prompt_delta.source_qa_ids for plan in plans} == {
        ("qa1",), ("qa2",), ("qa3",)}
    assert {path: path.read_bytes() for path in source_before} == source_before


def test_completed_per_qa_raw_responses_are_reused_on_resume(tmp_path):
    manifest = _baseline(tmp_path)
    backend = FakeBackend()
    proposer = LLMPromptDeltaProposer(
        backend=backend, maximum_deltas_per_qa=1,
        selection_policy=PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
        global_inspection_boundary_tolerance_seconds=10)
    output = tmp_path / "resume"
    first = proposer.propose(
        manifest, parent_meta_prompt_id="meta_parent",
        parent_meta_prompt_text="Current parent instruction.",
        output_directory=str(output))
    (output / "proposal_plans.json").unlink()
    backend.calls = 0
    backend.payloads.clear()

    resumed = proposer.propose(
        manifest, parent_meta_prompt_id="meta_parent",
        parent_meta_prompt_text="Current parent instruction.",
        output_directory=str(output))

    assert resumed == first
    assert backend.calls == 0
    assert backend.payloads == []


def test_candidate_joint_view_recaption_and_all_sibling_qa_rerun(
        tmp_path, monkeypatch):
    manifest = _baseline(tmp_path / "baseline")
    monkeypatch.setattr(
        "surrogate_rollout.optimization.fresh_prompt_delta_evidence."
        "validate_composed_text", lambda _text, _contract: ())

    caption_calls = []
    qa_calls = []

    class _Captioner:
        configuration_identity = {"captioner": "fixture"}

        def caption(self, **kwargs):
            caption_calls.append(kwargs["segment_id"])
            return SimpleNamespace(parsed={
                "clip_description": f"changed {kwargs['segment_id']}"})

    class _Mixed:
        def build(self, **kwargs):
            selected = sorted(kwargs["selected_clip_ids"])
            return SimpleNamespace(
                captions_path=str(tmp_path / "mixed-captions.json"),
                captions_hash="mixed-hash", database_path=str(
                    tmp_path / "mixed-database.json"),
                selected_clip_ids=selected, replaced_clip_ids=selected)

    def qa_fn(**kwargs):
        qa_calls.append(kwargs["question_id"])
        qa_dir = Path(kwargs["run_dir"])
        qa_dir.mkdir(parents=True, exist_ok=True)
        _jsonl(qa_dir / "trajectory.jsonl", ({
            "role": "assistant", "content": "done"},))
        return SimpleNamespace(
            errors=(), prediction=("A" if kwargs["question_id"] == "qa1"
                                   else "B"),
            score=(1.0 if kwargs["question_id"] == "qa1" else 0.0))

    delta = PromptDelta(
        "delta-qa1", "Describe the visible continuity.", ("qa1",),
        "The source trajectory omitted continuity.")
    plan = PromptDeltaExecutionPlan(
        prompt_delta=delta, selected_segment_ids=("0_10", "10_20"),
        selection_policy=PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
        frame_inspection_classification_hash=_source_qa_classification_hash(
            next(row for row in _qa_segment_selection_records(
                manifest, global_inspection_boundary_tolerance_seconds=10)
                 if row["qa_id"] == "qa1"), tolerance_seconds=10),
        global_inspection_boundary_tolerance_seconds=10)
    runner = PromptDeltaInterventionRunner(
        segment_captioner=_Captioner(), sample_loader=lambda _index: {},
        merge_prompt="merge", caption_cache_root=str(tmp_path / "cache"),
        caption_cache_manifest_path=str(tmp_path / "cache.jsonl"),
        composition_separator="\n", qa_fn=qa_fn,
        clip_index_fn=lambda _sample, _video: [
            ("0_10", {}), ("10_20", {}), ("20_30", {})],
        mixed_view_builder=_Mixed(), dvd_max_iterations=1, gpu=None,
        scaffold_contract=None)

    episode = runner.run(
        baseline_video_manifest_path=manifest,
        parent_meta_prompt_id="meta_parent", plan=plan,
        output_directory=str(tmp_path / "intervention"))

    assert caption_calls == ["0_10", "10_20"]
    assert qa_calls == ["qa1", "qa2"]
    assert [row.is_source_qa for row in episode.qa_outcomes] == [True, False]
    assert [(row.baseline_correct, row.intervention_correct)
            for row in episode.qa_outcomes] == [(False, True), (True, False)]
    assert episode.mixed_view_identity == {
        "delta_id": "delta-qa1", "captions_hash": "mixed-hash",
        "selected_segment_ids": ("0_10", "10_20"),
        "replaced_segment_ids": ("0_10", "10_20"),
    }


def test_oversized_single_qa_is_recorded_without_payload_modification(tmp_path):
    manifest = _baseline(tmp_path)
    backend = FakeBackend(full_fits=False, oversized_qa={"qa2"})
    proposer = LLMPromptDeltaProposer(
        backend=backend, maximum_deltas_per_qa=2,
        selection_policy=PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
        global_inspection_boundary_tolerance_seconds=10)
    plans = proposer.propose(
        manifest, parent_meta_prompt_id="meta_parent",
        parent_meta_prompt_text="Current parent instruction.",
        output_directory=str(tmp_path / "overflow"))
    assert backend.calls == 1
    assert [plan.prompt_delta.source_qa_ids for plan in plans] == [
        ("qa1",)]
    assert (tmp_path / "overflow/context_ineligible.json").exists()
    request_manifest = json.loads(
        (tmp_path / "overflow/request_manifest.json").read_text())
    assert request_manifest[
        "truncation_filtering_sampling_or_intra_qa_split"] is False
    assert request_manifest["context_ineligible_qa_ids"] == ["qa2"]
    assert request_manifest[
        "truncation_filtering_sampling_or_intra_qa_split"] is False
    audit = json.loads(
        (tmp_path / "overflow/selection_audit.json").read_text())
    qa2 = next(row for row in audit["qa_records"] if row["qa_id"] == "qa2")
    assert qa2["skip_reason"] == "context_ineligible"
    assert qa2["token_preflight"]["fits_context"] is False


def test_fresh_proposer_fails_closed_on_invalid_source_scope(tmp_path):
    manifest = _baseline(tmp_path)
    backend = FakeBackend(json.dumps({"proposals": [{
        "instruction": "x", "source_qa_ids": ["unknown"],
        "proposer_diagnosis": "y"}]}))
    proposer = LLMPromptDeltaProposer(
        backend=backend, maximum_deltas_per_qa=1,
        selection_policy=PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
        global_inspection_boundary_tolerance_seconds=10)
    with pytest.raises(FreshPromptDeltaError, match="content is invalid"):
        proposer.propose(
            manifest, parent_meta_prompt_id="meta_parent",
            parent_meta_prompt_text="Current parent instruction.",
            output_directory=str(tmp_path / "bad"))
    assert backend.calls == 1


def test_empty_localized_evidence_skips_provider_and_records_no_evidence(tmp_path):
    manifest = Path(_baseline(tmp_path))
    baseline = json.loads(manifest.read_text())
    qa_path = Path(baseline["baseline_qas_path"])
    qas = [json.loads(line) for line in qa_path.read_text().splitlines()]
    for qa in qas:
        qa["reference_sets"]["explicitly_cited_segments"] = []
        qa["reference_sets"]["frame_inspected_segments"] = []
        trajectory = Path(qa["trajectory_path"])
        trajectory.write_text(
            dumps_canonical({"role": "assistant", "content": "No timestamp."})
            + "\n")
        events = trajectory.parent / "tool_events.jsonl"
        if events.exists():
            events.unlink()
    qa_path.write_text("".join(dumps_canonical(row) + "\n" for row in qas))
    backend = FakeBackend()
    proposer = LLMPromptDeltaProposer(
        backend=backend, maximum_deltas_per_qa=2,
        selection_policy=PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
        global_inspection_boundary_tolerance_seconds=10)

    plans = proposer.propose(
        str(manifest), parent_meta_prompt_id="meta_parent",
        parent_meta_prompt_text="Current parent instruction.",
        output_directory=str(tmp_path / "no_evidence"))

    assert plans == ()
    assert backend.calls == 0
    assert backend.payloads == []
    request_manifest = json.loads(
        (tmp_path / "no_evidence/request_manifest.json").read_text())
    assert request_manifest["status"] == "no_eligible_proposal_evidence"
    assert request_manifest["requests"] == []
    audit = json.loads(
        (tmp_path / "no_evidence/selection_audit.json").read_text())
    assert audit["qa_without_localized_evidence"] == 2
    assert all(row["no_localized_evidence"] for row in audit["qa_records"])


def test_rejected_frame_inspect_arguments_are_audit_only_not_localization(
        tmp_path):
    manifest = Path(_baseline(tmp_path))
    baseline = json.loads(manifest.read_text())
    qas = [json.loads(line) for line in Path(
        baseline["baseline_qas_path"]).read_text().splitlines()]
    qa1 = qas[0]
    trajectory = Path(qa1["trajectory_path"])
    trajectory.write_text(
        dumps_canonical({"role": "assistant", "content": "No timestamp."})
        + "\n")
    _jsonl(trajectory.parent / "tool_events.jsonl", ({
        "tool": "frame_inspect_tool",
        "args": {"time_ranges_hhmmss": [[0, 10]]},
        "status": "argument_validation_error",
        "execution_performed": False,
        "error": "endpoint must be an HH:MM:SS string",
    }, {
        "tool": "frame_inspect_tool",
        "args": {"time_ranges_hhmmss": [["00:00:10", "00:00:19"]]},
        "status": "completed", "execution_performed": True,
    }))

    rows = _qa_segment_selection_records(
        str(manifest), global_inspection_boundary_tolerance_seconds=10)
    row = next(item for item in rows if item["qa_id"] == "qa1")

    assert row["frame_inspect_calls"][0]["classification"] == \
        "invalid_not_executed"
    assert row["frame_inspect_calls"][0]["segment_ids"] == []
    assert row["localized_frame_inspected_segments"] == ["10_20"]
    assert row["intervention_candidate_segments"] == ["10_20"]


def test_all_oversized_localized_qas_skip_without_provider_calls(tmp_path):
    manifest = _baseline(tmp_path)
    backend = FakeBackend(full_fits=False, oversized_qa={"qa1", "qa2"})
    proposer = LLMPromptDeltaProposer(
        backend=backend, maximum_deltas_per_qa=2,
        selection_policy=PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
        global_inspection_boundary_tolerance_seconds=10)

    plans = proposer.propose(
        manifest, parent_meta_prompt_id="meta_parent",
        parent_meta_prompt_text="Current parent instruction.",
        output_directory=str(tmp_path / "all_context_ineligible"))

    assert plans == ()
    assert backend.calls == 0
    result = json.loads((tmp_path /
        "all_context_ineligible/proposal_plans.json").read_text())
    assert result["status"] == "no_eligible_proposal_evidence"
    audit = json.loads((tmp_path /
        "all_context_ineligible/selection_audit.json").read_text())
    assert audit["eligible_qa_count"] == 0
    assert audit["context_ineligible_qa_count"] == 2


def test_intervention_rejects_retrieved_but_not_explicitly_cited_segment(
        tmp_path):
    manifest = _baseline(tmp_path)
    backend = FakeBackend({"qa1": json.dumps({"proposals": [{
        "instruction": "Inspect continuity.", "source_qa_ids": ["qa1"],
        "proposer_diagnosis": "diagnosis"}]})})
    proposer = LLMPromptDeltaProposer(
        backend=backend, maximum_deltas_per_qa=1,
        selection_policy=PROMPT_DELTA_SEGMENT_SELECTION_POLICY,
        global_inspection_boundary_tolerance_seconds=10)
    valid_plan = proposer.propose(
        manifest, parent_meta_prompt_id="meta_parent",
        parent_meta_prompt_text="Current parent instruction.",
        output_directory=str(tmp_path / "proposal_for_scope"))[0]
    plan = replace(valid_plan, selected_segment_ids=("20_30",))
    runner = PromptDeltaInterventionRunner(
        segment_captioner=object(), sample_loader=lambda _index: {},
        merge_prompt="merge", caption_cache_root=str(tmp_path / "cache"),
        caption_cache_manifest_path=str(tmp_path / "cache.jsonl"),
        composition_separator="\n", dvd_max_iterations=1, gpu=None,
        scaffold_contract=None)

    with pytest.raises(FreshPromptDeltaError, match="not localized"):
        runner.run(
            baseline_video_manifest_path=manifest,
            parent_meta_prompt_id="meta_parent", plan=plan,
            output_directory=str(tmp_path / "intervention"))


def test_saved_resolver_dispatches_fresh_manifest_without_mutation(tmp_path):
    baseline = _baseline(tmp_path / "bundle")
    before = Path(baseline).read_bytes()
    baseline_value = json.loads(Path(baseline).read_text())
    baseline_qas = [json.loads(line) for line in Path(
        baseline_value["baseline_qas_path"]).read_text().splitlines()]
    baseline_trajectories = {
        row["question_id"]: row["trajectory_path"] for row in baseline_qas}
    intervention = _json(tmp_path / "intervention.json", {
        "schema_version": "fresh_prompt_delta_intervention_v1",
        "video_id": "v1", "qa_records": [{
            "qa_id": "qa1", "baseline_correct": False,
            "intervention_answer": "A",
            "intervention_correct": True,
            "intervention_trajectory_path": None,
        }, {
            "qa_id": "qa2", "baseline_correct": True,
            "intervention_answer": "B",
            "intervention_correct": True,
            "intervention_trajectory_path": None,
        }]})
    delta = PromptDelta("d1", "delta", ("qa1",), "diagnosis")
    episode = InterventionEpisode(
        episode_id="e1", video_id="v1", parent_meta_prompt_id="m1",
        prompt_delta=delta,
        clips=(InterventionClipRecord(
            "10_20", [10, 20], {"history": []}, "base", delta,
            "before", "after"),),
        qa_outcomes=(QAInterventionOutcome(
            "qa1", True, "B", "A", False, True,
            baseline_trajectories["qa1"], None),
            QAInterventionOutcome(
                "qa2", False, "B", "B", True, True,
                baseline_trajectories["qa2"], None)),
        baseline_run_ref=baseline, intervention_run_ref=intervention)
    resolved = SavedEpisodeFeedbackArtifactResolver().resolve_qas(episode)
    assert [row["transition"] for row in resolved] == [
        "wrong_to_correct", "correct_to_correct"]
    assert resolved[0]["baseline_trajectory"]["availability"] == "available"
    assert resolved[0]["intervention_trajectory"]["availability"] == "unavailable"
    assert Path(baseline).read_bytes() == before


def test_fresh_component_schema_and_exact_counter_contract(tmp_path):
    path = tmp_path / "component.json"
    _json(path, {"schema_version": "fresh_prompt_delta_component_config_v1"})
    assert _load_configuration(str(path))["schema_version"].startswith("fresh_")
    counter, tokenizer_identity = _exact_counter("gpt-4o")
    assert callable(counter)
    assert tokenizer_identity
    count = counter(({"role": "system", "content": "system"},
                     {"role": "user", "content": "user"}))
    assert count.total_input_tokens > 0


def test_completed_baseline_resume_is_read_only(tmp_path):
    root = tmp_path / "baseline"
    video_root = root / "baseline" / "v1"
    required_paths = {}
    for name in ("routing_manifest_path", "caption_view_path", "captions_path",
                 "frames_path", "frozen_histories_path"):
        required_paths[name] = _json(video_root / f"{name}.json", {})
    trajectory = _jsonl(video_root / "qa/trajectory.jsonl", ({
        "role": "assistant", "content": "done",
    },))
    qas = _jsonl(video_root / "baseline_qas.jsonl", ({
        "question_id": "qa1", "prediction": "A", "parsed_answer": "A",
        "errors": [], "trajectory_path": trajectory,
    },))
    video_manifest = _json(video_root / "video_complete.json", {
        "video_id": "v1", "qa_count": 1,
        "baseline_qas_path": qas, **required_paths,
    })
    provisional = _json(root / "iteration_state/provisional.json", {})
    _json(root / "manifest.json", {
        "status": "completed", "run_id": "baseline_old_identity",
        "selected_video_ids": ["v1"], "baseline_qa_count": 1,
        "video_manifest_paths": [video_manifest],
        "prompt_delta_source_paths": [],
            "provisional_state_path": provisional,
            "next_coverage_state": {
                "rotation_order": [
                    "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"],
                "used_since_confirmation": ["v1"],
            "rotation_cursor": 0, "coverage_cycle": 0,
            "iterations_since_confirmation": 1,
            "confirmation_due": False,
        },
    })
    before = {path: path.read_bytes() for path in root.rglob("*")
              if path.is_file()}

    result = load_completed_baseline_for_read_only_resume(
        str(root), selected_video_ids=("v1",))

    assert result is not None and result.resumed is True
    assert result.run_id == "baseline_old_identity"
    assert result.baseline_qa_count == 1
    assert {path: path.read_bytes() for path in before} == before
