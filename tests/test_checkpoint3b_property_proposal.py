import json
from pathlib import Path

import pytest
from PIL import Image

from surrogate_rollout.optimization.policies.property_proposal import (
    MultiPropertyProposalPolicy,
    OpenAIPropertyProposalProvider,
    ProposalEvidenceBounds,
    PropertyProposalConflictError,
    PropertyProposalParseError,
    build_proposal_input,
    build_proposal_request,
    parse_proposal_output,
)
from surrogate_rollout.optimization.property_proposal import VideoProposalContext
from surrogate_rollout.optimization.property_proposal import (
    candidate_property_proposal_from_json,
)
from surrogate_rollout.prompt_routing.persistence import prompt_bank_from_json
from surrogate_rollout.prompt_routing.schemas import as_json_dict


ROOT = Path(__file__).parents[1]


def context(tmp_path, *, prediction="B"):
    data = json.loads((
        ROOT / "prompt_routing/fixtures/stage4_7_components.json").read_text())
    rows = (
        {
            "question_id": "q1", "question": "What color is the moving car?",
            "options": ["red", "green", "blue"], "ground_truth": "green car",
            "prediction": prediction, "is_correct": False,
            "reasoning_trace": [{
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-secret",
                    "type": "function",
                    "function": {
                        "name": "frame_inspect_tool",
                        "arguments": json.dumps({
                            "segment_id": "0_10", "database": "/secret/db"}),
                }}],
            }],
            "used_segments": ["0_10"],
            "reference_sets": {"frame_inspected_segments": ["0_10"]},
        },
        {
            "question_id": "q2", "question": "What happens after the door opens?",
            "options": ["A", "B", "C"], "ground_truth": "B",
            "prediction": "B", "is_correct": True,
            "reasoning_trace": [{"step": "ordered actions"}],
            "used_segments": ["10_20"],
            "reference_sets": {"returned_segments": ["10_20"]},
        },
        {
            "question_id": "q3", "question": "Which label appears briefly?",
            "options": ["EXIT", "SHOP", "HOME"], "ground_truth": "EXIT",
            "prediction": "EXIT", "is_correct": True,
            "reasoning_trace": [], "used_segments": ["20_30"],
            "reference_sets": {"consumed_segments": ["20_30"]},
        },
    )
    frame_references = {}
    for index, segment_id in enumerate(("0_10", "10_20", "20_30")):
        paths = []
        for frame_index in range(3):
            path = tmp_path / f"frame-{index}-{frame_index}.jpg"
            Image.new("RGB", (32, 24), (index * 50, frame_index * 50, 10)).save(path)
            paths.append(str(path))
        frame_references[segment_id] = tuple(paths)
    return VideoProposalContext(
        video_id="video-1", baseline_run_id="baseline-1",
        baseline_qa_results=rows,
        captions={
            "0_10": {"caption": "A car moves."},
            "10_20": {"caption": "A door opens."},
            "20_30": {"caption": "A sign is visible."},
        },
        frame_references=frame_references, frozen_histories=(),
        prompt_bank=prompt_bank_from_json(data["prompt_bank"]),
        proposal_artifact_dir=str(tmp_path / "proposal"),
    )


def with_all_qas_correct(ctx):
    rows = tuple({
        **row,
        "prediction": row["ground_truth"],
        "parsed_answer": row["ground_truth"],
        "is_correct": True,
        "runtime_valid": True,
    } for row in ctx.baseline_qa_results)
    return type(ctx)(
        video_id=ctx.video_id, baseline_run_id=ctx.baseline_run_id,
        baseline_qa_results=rows, captions=ctx.captions,
        frame_references=ctx.frame_references,
        frozen_histories=ctx.frozen_histories, prompt_bank=ctx.prompt_bank,
        proposal_artifact_dir=ctx.proposal_artifact_dir)


def proposal(candidate_id, text, qids=("q1",), **over):
    source_qa_slot = over.pop("source_qa_slot", "qa_1")
    del qids
    value = {
        "source_qa_slot": source_qa_slot,
        "suggested_property_id": candidate_id,
        "property_text": text,
        "applicability": {
            "when": "Use when visible evidence contains a salient omitted detail.",
            "positive_cues": ["A visible detail is absent from the description."],
            "negative_cues": [],
            "required_modalities": ["frames"],
        },
        "covered_by_existing_property_ids": [],
        "failure_analysis": "The baseline omitted a reusable visual detail.",
    }
    value.update(over)
    return value


class Provider:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, request):
        self.calls.append(json.loads(request))
        return json.dumps(self.response)


class SequenceProvider(Provider):
    def __init__(self, responses):
        super().__init__(None)
        self.responses = list(responses)

    def __call__(self, request):
        self.calls.append(json.loads(request))
        return json.dumps(self.responses[len(self.calls) - 1])


@pytest.mark.parametrize("rows, expected", [
    ([proposal("cp_color", "Record salient object colors precisely.")], 1),
    ([proposal("cp_color", "Record salient object colors precisely."),
      proposal("cp_sequence", "Preserve causal action order.", ("q1", "q2"))], 2),
])
def test_zero_one_and_multiple_proposals_with_complete_video_request(
        tmp_path, rows, expected):
    provider = Provider({"proposals": rows})
    policy = MultiPropertyProposalPolicy(
        response_provider=provider, max_proposals=4)
    result = tuple(policy.propose(context(tmp_path)))
    assert len(result) == expected
    request = provider.calls[0]
    assert len(request["qas"]) == 3
    assert request["qas"][0]["answer_choices"] == ["red", "green", "blue"]
    assert request["qas"][0]["ground_truth"] == "green car"
    serialized = json.dumps(request, sort_keys=True)
    for forbidden in (
        "source_video_id", "segment_id", "question_id", "priority_rank",
        "payload_truncation", "call-secret", "/secret/db"):
        assert forbidden not in serialized
    assert request["qas"][0]["used_segment_evidence"][0][
        "representative_image"]["base64_data"]
    if expected == 2:
        assert result[1].source_question_ids == ("q1",)
    artifact_dir = Path(context(tmp_path).proposal_artifact_dir)
    assert (artifact_dir / "raw_output.txt").exists()
    identity = json.loads((artifact_dir / "input_identity.json").read_text())
    assert identity["source_video_id"] == "video-1"
    assert identity["qas"][0]["question_id"] == "q1"
    packing = json.loads((artifact_dir / "evidence_packing.json").read_text())
    assert packing["included_interval_count"] == 1
    assert packing["omitted_interval_count"] == 0
    assert packing["per_qa"][0]["included_interval_count"] == 1
    parsed = json.loads((artifact_dir / "parsed_output.json").read_text())
    assert len(parsed["proposals"]) == expected


def test_runtime_valid_incorrect_qa_cannot_complete_with_zero_proposals(tmp_path):
    provider = Provider({"proposals": []})
    policy = MultiPropertyProposalPolicy(
        response_provider=provider, max_proposals=2)

    with pytest.raises(
            PropertyProposalParseError,
            match=r"every eligible QA requires 1\.\.2"):
        policy.propose(context(tmp_path))

    assert len(provider.calls) == 3
    request = provider.calls[0]
    assert request["proposal_cardinality"] == {
        "eligible_qa_count": 1,
        "minimum_accepted_proposals": 1,
        "maximum_accepted_proposals": 2,
        "zero_proposal_allowed": False,
    }
    assert request["qas"][0]["proposal_requirement"]["status"] == (
        "required_intervention_candidate")
    artifact_dir = Path(context(tmp_path).proposal_artifact_dir)
    assert json.loads((artifact_dir / "parsed_output.json").read_text()) == {
        "proposals": []}
    validation = json.loads((artifact_dir / "validation.json").read_text())
    assert validation["status"] == "rejected"
    retry_manifest = json.loads((
        artifact_dir / "missing_slot_retries.json").read_text())
    assert len(retry_manifest["attempts"]) == 2
    assert all(call["retry"]["required_source_qa_slots"] == ["qa_1"]
               for call in provider.calls[1:])


def test_missing_qa_slot_is_retried_without_repeating_completed_slots(tmp_path):
    ctx = context(tmp_path)
    rows = list(ctx.baseline_qa_results)
    rows[1] = {**rows[1], "runtime_valid": True,
               "reference_sets": {"frame_inspected_segments": ["10_20"]}}
    rows[2] = {**rows[2], "runtime_valid": True,
               "reference_sets": {"frame_inspected_segments": ["20_30"]}}
    eligible = type(ctx)(
        video_id=ctx.video_id, baseline_run_id=ctx.baseline_run_id,
        baseline_qa_results=tuple(rows), captions=ctx.captions,
        frame_references=ctx.frame_references,
        frozen_histories=ctx.frozen_histories, prompt_bank=ctx.prompt_bank,
        proposal_artifact_dir=ctx.proposal_artifact_dir)
    provider = SequenceProvider((
        {"proposals": [
            proposal("cp_first", "Preserve the first visible cue precisely.",
                     source_qa_slot="qa_1"),
            proposal("cp_third", "Preserve the third visible cue precisely.",
                     source_qa_slot="qa_3"),
        ]},
        {"proposals": [proposal(
            "cp_second", "Preserve the second visible cue precisely.",
            source_qa_slot="qa_2")]},
    ))
    policy = MultiPropertyProposalPolicy(
        response_provider=provider, max_proposals=4)

    result = tuple(policy.propose(eligible))

    assert [item.source_question_ids for item in result] == [
        ("q1",), ("q3",), ("q2",)]
    assert len(provider.calls) == 2
    assert [row["qa_slot"] for row in provider.calls[1]["qas"]] == ["qa_2"]
    assert provider.calls[1]["retry"] == {
        "schema_version": "property_proposal_missing_slot_retry_v1",
        "policy_version": "missing_qa_slot_retry_v1",
        "attempt_index": 1,
        "required_source_qa_slots": ["qa_2"],
    }
    artifact_dir = Path(eligible.proposal_artifact_dir)
    retry_manifest = json.loads((
        artifact_dir / "missing_slot_retries.json").read_text())
    assert retry_manifest["final_missing_source_qa_slots"] == []
    assert retry_manifest["attempts"][0]["accepted_proposal_count"] == 1
    assert (artifact_dir / "missing_slot_retries" /
            "attempt_001" / "raw_output.txt").exists()

    assert tuple(policy.propose(eligible)) == result
    assert len(provider.calls) == 2


@pytest.mark.parametrize("count", [1, 2])
def test_runtime_valid_incorrect_qa_accepts_one_or_two_proposals(tmp_path, count):
    rows = [
        proposal(f"cp_{index}", f"Record salient visible detail {index} precisely.")
        for index in range(count)
    ]
    result = MultiPropertyProposalPolicy(
        response_provider=Provider({"proposals": rows}),
        max_proposals=2,
    ).propose(context(tmp_path))

    assert len(result) == count


def test_two_eligible_qas_require_two_to_four_candidates(tmp_path):
    ctx = context(tmp_path)
    rows = list(ctx.baseline_qa_results)
    rows[1] = {
        **rows[1],
        "prediction": "A",
        "parsed_answer": "A",
        "is_correct": False,
        "runtime_valid": True,
        "reference_sets": {"frame_inspected_segments": ["10_20"]},
    }
    expanded = type(ctx)(
        video_id=ctx.video_id, baseline_run_id=ctx.baseline_run_id,
        baseline_qa_results=tuple(rows), captions=ctx.captions,
        frame_references=ctx.frame_references,
        frozen_histories=ctx.frozen_histories, prompt_bank=ctx.prompt_bank,
        proposal_artifact_dir=ctx.proposal_artifact_dir)

    request = build_proposal_request(
        expanded, max_proposals=4, max_trace_events_per_qa=3,
        max_captions=3, max_payload_chars=40000)

    assert request["proposal_cardinality"] == {
        "eligible_qa_count": 2,
        "minimum_accepted_proposals": 2,
        "maximum_accepted_proposals": 4,
        "zero_proposal_allowed": False,
    }


def test_explicitly_invalid_incorrect_qa_may_return_zero_proposals(tmp_path):
    ctx = context(tmp_path)
    rows = list(ctx.baseline_qa_results)
    rows[0] = {
        **rows[0], "runtime_valid": True, "annotation_reliable": False,
    }
    invalid = type(ctx)(
        video_id=ctx.video_id, baseline_run_id=ctx.baseline_run_id,
        baseline_qa_results=tuple(rows), captions=ctx.captions,
        frame_references=ctx.frame_references,
        frozen_histories=ctx.frozen_histories, prompt_bank=ctx.prompt_bank,
        proposal_artifact_dir=ctx.proposal_artifact_dir)
    provider = Provider({"proposals": []})

    assert MultiPropertyProposalPolicy(
        response_provider=provider, max_proposals=2).propose(invalid) == ()
    requirement = provider.calls[0]["qas"][0]["proposal_requirement"]
    assert requirement["status"] == "ineligible"
    assert requirement["invalid_reasons"] == ["unreliable_annotation"]


def test_eligible_correct_qa_requires_candidate_and_preserves_correctness(tmp_path):
    ctx = with_all_qas_correct(context(tmp_path))
    provider = Provider({"proposals": [proposal(
        "cp_concise", "Consolidate repetitive visible details concisely.")]})

    result = MultiPropertyProposalPolicy(
        response_provider=provider, max_proposals=2,
    ).propose(ctx)

    assert len(result) == 1
    assert result[0].source_question_ids == ("q1",)
    assert result[0].source_qa_baseline_correct is True
    assert provider.calls[0]["qas"][0]["proposal_requirement"]["status"] == (
        "required_intervention_candidate")


def test_every_eligible_correct_and_incorrect_qa_gets_its_own_candidate(tmp_path):
    ctx = context(tmp_path)
    rows = list(ctx.baseline_qa_results)
    rows[1] = {**rows[1], "runtime_valid": True,
               "reference_sets": {"frame_inspected_segments": ["10_20"]}}
    rows[2] = {**rows[2], "runtime_valid": True,
               "reference_sets": {"frame_inspected_segments": ["20_30"]}}
    eligible = type(ctx)(
        video_id=ctx.video_id, baseline_run_id=ctx.baseline_run_id,
        baseline_qa_results=tuple(rows), captions=ctx.captions,
        frame_references=ctx.frame_references,
        frozen_histories=ctx.frozen_histories, prompt_bank=ctx.prompt_bank,
        proposal_artifact_dir=ctx.proposal_artifact_dir)
    provider = Provider({"proposals": [
        proposal("cp_recover", "Emphasize visually supported object color.",
                 source_qa_slot="qa_1"),
        proposal("cp_order", "Clarify temporal ordering without repetition.",
                 source_qa_slot="qa_2"),
        proposal("cp_concise", "Suppress irrelevant speculative detail.",
                 source_qa_slot="qa_3"),
    ]})

    result = MultiPropertyProposalPolicy(
        response_provider=provider, max_proposals=4).propose(eligible)

    assert [item.source_question_ids for item in result] == [
        ("q1",), ("q2",), ("q3",)]
    assert [item.source_qa_baseline_correct for item in result] == [False, True, True]
    assert provider.calls[0]["proposal_cardinality"]["eligible_qa_count"] == 3


def test_proposal_request_bounds_reasoning_and_used_segment_evidence(tmp_path):
    ctx = context(tmp_path)
    large_rows = tuple({
        **row,
        "reasoning_trace": tuple(
            {"step": f"{index}-" + "x" * 1000} for index in range(20)),
    } for row in ctx.baseline_qa_results)
    bounded = VideoProposalContext(
        video_id=ctx.video_id,
        baseline_run_id=ctx.baseline_run_id,
        baseline_qa_results=large_rows,
        captions=ctx.captions,
        frame_references=ctx.frame_references,
        frozen_histories=ctx.frozen_histories,
        prompt_bank=ctx.prompt_bank,
        proposal_artifact_dir=ctx.proposal_artifact_dir,
    )

    request = build_proposal_request(
        bounded,
        max_proposals=1,
        max_trace_events_per_qa=20,
        max_captions=30,
        max_payload_chars=40000,
    )

    assert all(len(row["bounded_reasoning"]) <= 3 for row in request["qas"])
    assert [len(row["used_segment_evidence"]) for row in request["qas"]] == [1, 0, 0]
    assert all(len(item) <= 1201 for row in request["qas"]
               for item in row["bounded_reasoning"])
    assert "payload_truncation" not in request
    assert request["output_schema"]["proposals"][0][
        "covered_by_existing_property_ids"] == []
    assert "non-binding hints" in request["task"]
    assert "current frames" in request["task"]
    assert "current transcript" in request["task"]
    assert "preceding caption history" in request["task"]
    assert "external knowledge" in request["task"]
    assert "required_intervention_candidate" in request["task"]
    assert "both correct and incorrect baselines" in request["task"]
    assert "Only the downstream updater" in request["task"]


def test_openai_provider_builds_exact_multimodal_body(tmp_path):
    request = build_proposal_request(
        context(tmp_path), max_proposals=1, max_trace_events_per_qa=3,
        max_captions=3, max_payload_chars=40000)
    provider = OpenAIPropertyProposalProvider(model="gpt-test", api_key="unused")
    body = provider.build_request_body(request)
    assert body["messages"][0]["role"] == "system"
    assert "current transcript" in body["messages"][0]["content"]
    content = body["messages"][1]["content"]

    assert content[0]["type"] == "text"
    assert sum(item["type"] == "image_url" for item in content) == 1
    assert all(item["image_url"]["url"].startswith("data:image/jpeg;base64,")
               for item in content if item["type"] == "image_url")
    assert "base64_data" not in content[0]["text"]


def test_policy_persists_private_identity_and_exact_provider_body(tmp_path):
    class MultimodalProvider:
        supports_multimodal_request = True

        def __init__(self):
            self.calls = []
            self.response = {"proposals": []}

        def build_request_body(self, request):
            return {"exact": True, "request_schema": request["schema_version"]}

        def __call__(self, request):
            self.calls.append(request)
            return json.dumps(self.response)

    provider = MultimodalProvider()
    ctx = context(tmp_path)
    provider_response = proposal(
        "cp_color", "Record salient object colors precisely.")
    provider.response = {"proposals": [provider_response]}
    result = MultiPropertyProposalPolicy(response_provider=provider).propose(ctx)

    assert len(result) == 1
    assert isinstance(provider.calls[0], dict)
    artifact_dir = Path(ctx.proposal_artifact_dir)
    assert json.loads((artifact_dir / "provider_request.json").read_text()) == {
        "exact": True,
        "request_schema": "multimodal_property_proposal_request_v10",
    }
    model_request = (artifact_dir / "request.json").read_text()
    private_identity = (artifact_dir / "input_identity.json").read_text()
    assert "source_video_id" not in model_request
    assert "source_video_id" in private_identity


def test_proposal_uses_all_cited_and_inspected_segments_only(tmp_path):
    ctx = context(tmp_path)
    extra_frames = dict(ctx.frame_references)
    extra_captions = dict(ctx.captions)
    for index, segment_id in enumerate(("30_40", "40_50"), start=3):
        paths = []
        for frame_index in range(3):
            path = tmp_path / f"frame-{index}-{frame_index}.jpg"
            Image.new("RGB", (32, 24), (index * 40, frame_index * 40, 10)).save(path)
            paths.append(str(path))
        extra_frames[segment_id] = tuple(paths)
        extra_captions[segment_id] = {"caption": f"Evidence for {segment_id}."}
    rows = list(ctx.baseline_qa_results)
    rows[0] = {
        **rows[0],
        "used_segments": ["0_10", "10_20", "20_30", "30_40", "40_50"],
        "reference_sets": {
            "explicitly_cited_segments": ["40_50", "0_10", "30_40"],
            "frame_inspected_segments": ["20_30", "40_50"],
            "returned_segments": ["10_20"],
        },
    }
    expanded = VideoProposalContext(
        video_id=ctx.video_id, baseline_run_id=ctx.baseline_run_id,
        baseline_qa_results=tuple(rows), captions=extra_captions,
        frame_references=extra_frames, frozen_histories=ctx.frozen_histories,
        prompt_bank=ctx.prompt_bank,
        proposal_artifact_dir=ctx.proposal_artifact_dir,
    )

    request, identity = build_proposal_input(
        expanded, max_proposals=1,
        bounds=ProposalEvidenceBounds(max_text_payload_chars=40000))

    assert request["schema_version"] == "multimodal_property_proposal_request_v10"
    assert len(request["qas"][0]["used_segment_evidence"]) == 4
    assert [row["segment_id"] for row in identity["qas"][0][
        "selected_segment_evidence"]] == ["0_10", "30_40", "40_50", "20_30"]
    assert identity["evidence_selection_policy"] == (
        "compact_cited_then_inspected_evidence_v2")
    assert [row["evidence_role"] for row in request["qas"][0][
        "used_segment_evidence"]] == [
            "explicitly_cited", "explicitly_cited", "explicitly_cited",
            "inspected_only"]
    assert "segment_id" not in json.dumps(request)


def test_compact_evidence_normalizes_truncates_and_omits_duplicate_transcript(
        tmp_path):
    ctx = context(tmp_path)
    rows = list(ctx.baseline_qa_results)
    rows[0] = {
        **rows[0],
        "reference_sets": {
            "explicitly_cited_segments": ["10_20"],
            "frame_inspected_segments": ["0_10", "10_20"],
        },
    }
    captions = dict(ctx.captions)
    captions["10_20"] = {"caption": (
        "A person   opens\n the door.\n\nTranscript during this video clip: "
        "A person opens the door.")}
    captions["0_10"] = {"caption": (
        "D" * 400 + "\nmore detail\n\nTranscript during this video clip: "
        + "T" * 200)}
    compact = type(ctx)(
        video_id=ctx.video_id, baseline_run_id=ctx.baseline_run_id,
        baseline_qa_results=tuple(rows), captions=captions,
        frame_references=ctx.frame_references,
        frozen_histories=ctx.frozen_histories, prompt_bank=ctx.prompt_bank,
        proposal_artifact_dir=ctx.proposal_artifact_dir)

    request, identity = build_proposal_input(
        compact, max_proposals=1,
        bounds=ProposalEvidenceBounds(max_text_payload_chars=40000))

    evidence = request["qas"][0]["used_segment_evidence"]
    assert evidence[0] == {
        "timestamp_range": "[10–20]",
        "evidence_role": "explicitly_cited",
        "generated_description": "A person opens the door.",
        "representative_image": evidence[0]["representative_image"],
    }
    assert evidence[1]["evidence_role"] == "inspected_only"
    assert evidence[1]["summary"].startswith("[0–10] ")
    assert "\n" not in evidence[1]["summary"]
    description, transcript = evidence[1]["summary"].split(" | tx: ")
    assert len(description.removeprefix("[0–10] ")) <= 250
    assert len(transcript) <= 120
    private = identity["qas"][0]["selected_segment_evidence"]
    assert private[0]["baseline_caption"] == captions["10_20"]["caption"]
    assert private[1]["baseline_caption"] == captions["0_10"]["caption"]


def test_compact_evidence_budget_is_deterministic_and_preserves_full_provenance(
        tmp_path):
    ctx = context(tmp_path)
    segment_ids = tuple(f"{index}_{index + 1}" for index in range(80))
    rows = list(ctx.baseline_qa_results)
    rows[0] = {
        **rows[0],
        "reference_sets": {"frame_inspected_segments": list(reversed(segment_ids))},
    }
    captions = {
        segment_id: {"caption": (
            f"Description for interval {segment_id}. " + "d" * 400
            + "\n\nTranscript during this video clip: " + "t" * 200)}
        for segment_id in segment_ids
    }
    frame_references = {segment_id: (f"/{segment_id}.jpg",)
                        for segment_id in segment_ids}
    bounded = type(ctx)(
        video_id=ctx.video_id, baseline_run_id=ctx.baseline_run_id,
        baseline_qa_results=tuple(rows), captions=captions,
        frame_references=frame_references,
        frozen_histories=ctx.frozen_histories, prompt_bank=ctx.prompt_bank,
        proposal_artifact_dir=ctx.proposal_artifact_dir)

    def frame_loader(path, _max_bytes, _transform):
        return {
            "mime_type": "image/jpeg", "base64_data": "AA==",
            "source_sha256": "source-" + path,
            "transformed_sha256": "transformed-" + path,
            "transformed_bytes": 1, "transform_configuration": {},
        }

    bounds = ProposalEvidenceBounds(max_text_payload_chars=30000)
    first_request, first_identity = build_proposal_input(
        bounded, max_proposals=1, bounds=bounds, frame_loader=frame_loader)
    second_request, second_identity = build_proposal_input(
        bounded, max_proposals=1, bounds=bounds, frame_loader=frame_loader)

    assert first_request == second_request
    assert first_identity == second_identity
    packing = first_identity["evidence_packing"]
    assert packing["original_interval_count"] == 80
    assert 0 < packing["included_interval_count"] < 80
    assert packing["omitted_interval_count"] == (
        80 - packing["included_interval_count"])
    assert first_identity["text_payload_chars"] <= 30000
    private = first_identity["qas"][0]["selected_segment_evidence"]
    assert len(private) == 80
    assert sum(row["included_in_payload"] for row in private) == (
        packing["included_interval_count"])
    assert any(row["omission_reason"] == "evidence_text_budget_exhausted"
               for row in private)
    summaries = [row["summary"] for row in first_request["qas"][0][
        "used_segment_evidence"]]
    starts = [int(summary[1:].split("–", 1)[0]) for summary in summaries]
    assert starts == sorted(starts)
    assert "segment_id" not in json.dumps(first_request)


def test_lexical_overlap_and_existing_property_similarity_are_not_vetoes(tmp_path):
    ctx = context(tmp_path)
    raw = json.dumps({"proposals": [
        proposal("cp_video", "In this video, record the answer option A."),
        proposal("cp_answer", "Describe the green car as the correct answer."),
        proposal(
            "cp_existing",
            "Describe only visually supported events and entities precisely."),
    ]})
    accepted, rejected = parse_proposal_output(
        raw, ctx, max_proposals=4, max_property_text_chars=240)
    assert len(accepted) == 3
    assert rejected == ()
    assert {item.suggested_property_id for item in accepted} == {
        "cp_video", "cp_answer", "cp_existing"}


def test_coverage_hints_are_non_binding_and_persisted_for_later_feedback(tmp_path):
    ctx = context(tmp_path)
    provider = Provider({"proposals": [proposal(
        "cp_roles",
        "Describe visually observable instructional interactions and roles clearly.",
        covered_by_existing_property_ids=["pe_default", "pe_temporal"],
        failure_analysis=(
            "This may overlap general visual and temporal properties, but the "
            "intervention should determine whether explicit role distinctions help."),
    )]})
    result = tuple(MultiPropertyProposalPolicy(
        response_provider=provider).propose(ctx))

    assert len(result) == 1
    assert result[0].candidate_property_id.startswith("candidate_")
    assert result[0].suggested_property_id == "cp_roles"
    assert result[0].coverage_hints == ("pe_default", "pe_temporal")
    assert result[0].covered_by_existing_property_ids == result[0].coverage_hints
    assert result[0].coverage_assessment == "deferred_to_intervention"
    parsed = json.loads((Path(ctx.proposal_artifact_dir) /
                         "parsed_output.json").read_text())
    assert parsed["proposals"][0]["coverage_hints"] == [
        "pe_default", "pe_temporal"]
    assert "covered_by_existing_property_ids" not in parsed["proposals"][0]


def test_coverage_contradiction_is_deferred_but_unexecutable_instruction_is_rejected(
        tmp_path):
    ctx = context(tmp_path)
    raw = json.dumps({"proposals": [proposal(
        "cp_contradiction", "Describe visually observable teaching roles.",
        covered_by_existing_property_ids=["pe_default"],
        failure_analysis="No existing property covers this instruction.")]})
    accepted, rejected = parse_proposal_output(
        raw, ctx, max_proposals=4, max_property_text_chars=240)
    assert len(accepted) == 1
    assert rejected == ()
    unexecutable = proposal(
        "cp_history", "Research external historical sources before captioning.")
    with pytest.raises(PropertyProposalParseError, match="unavailable"):
        parse_proposal_output(
            json.dumps({"proposals": [unexecutable]}), ctx,
            max_proposals=4, max_property_text_chars=240)


def test_duplicate_and_malformed_proposals_fail_strictly(tmp_path):
    ctx = context(tmp_path)
    duplicated = json.dumps({"proposals": [
        proposal("cp_a", "Record salient colors."),
        proposal("cp_b", "Record salient colors."),
    ]})
    with pytest.raises(PropertyProposalParseError, match="exact duplicate"):
        parse_proposal_output(
            duplicated, ctx, max_proposals=4, max_property_text_chars=240)
    malformed = json.dumps({"proposals": [{"property_text": "missing fields"}]})
    with pytest.raises(PropertyProposalParseError, match="exactly"):
        parse_proposal_output(
            malformed, ctx, max_proposals=4, max_property_text_chars=240)


def test_nonbinding_suggested_id_collision_is_allowed(tmp_path):
    accepted, rejected = parse_proposal_output(
        json.dumps({"proposals": [proposal(
            "pe_default", "Describe visually observable teaching roles.")]}),
        context(tmp_path), max_proposals=1, max_property_text_chars=240)

    assert len(accepted) == 1
    assert accepted[0].candidate_property_id.startswith("candidate_")
    assert accepted[0].suggested_property_id == "pe_default"
    assert rejected == ()


def test_candidate_handle_is_stable_opaque_and_source_specific(tmp_path):
    raw = json.dumps({"proposals": [proposal(
        "pe_visual_context", "Record salient visible context precisely.")]})
    first, _ = parse_proposal_output(
        raw, context(tmp_path), max_proposals=1, max_property_text_chars=240)
    repeated, _ = parse_proposal_output(
        raw, context(tmp_path), max_proposals=1, max_property_text_chars=240)
    other = context(tmp_path)
    other = type(other)(
        video_id="video-2", baseline_run_id=other.baseline_run_id,
        baseline_qa_results=other.baseline_qa_results,
        captions=other.captions, frame_references=other.frame_references,
        frozen_histories=other.frozen_histories, prompt_bank=other.prompt_bank,
        proposal_artifact_dir=other.proposal_artifact_dir)
    second_video, _ = parse_proposal_output(
        raw, other, max_proposals=1, max_property_text_chars=240)

    assert first[0].candidate_property_id == repeated[0].candidate_property_id
    assert first[0].candidate_property_id != second_video[0].candidate_property_id
    assert first[0].suggested_property_id == "pe_visual_context"
    assert first[0].candidate_property_id != first[0].suggested_property_id


@pytest.mark.parametrize("applicability, match", [
    (None, "exactly"),
    ({"when": "", "positive_cues": ["cue"], "negative_cues": [],
      "required_modalities": ["frames"]}, "when"),
    ({"when": "condition", "positive_cues": [], "negative_cues": [],
      "required_modalities": ["frames"]}, "positive_cues"),
    ({"when": "condition", "positive_cues": ["cue", "cue"],
      "negative_cues": [], "required_modalities": ["frames"]}, "duplicates"),
    ({"when": "condition", "positive_cues": ["cue"],
      "negative_cues": ["no", "no"], "required_modalities": ["frames"]},
     "duplicates"),
    ({"when": "condition", "positive_cues": ["cue"], "negative_cues": [],
      "required_modalities": []}, "required_modalities"),
    ({"when": "condition", "positive_cues": ["cue"], "negative_cues": [],
      "required_modalities": ["frames", "frames"]}, "duplicates"),
    ({"when": "condition", "positive_cues": ["cue"], "negative_cues": [],
      "required_modalities": ["audio"]}, "unsupported"),
    ({"when": "condition", "positive_cues": ["cue"], "negative_cues": [],
      "required_modalities": ["frames"], "unknown": True}, "exactly"),
])
def test_applicability_schema_is_strict(tmp_path, applicability, match):
    row = proposal("cp_app", "Preserve supported details precisely.")
    if applicability is None:
        del row["applicability"]
    else:
        row["applicability"] = applicability
    with pytest.raises(PropertyProposalParseError, match=match):
        parse_proposal_output(
            json.dumps({"proposals": [row]}), context(tmp_path),
            max_proposals=1, max_property_text_chars=240)


def test_unknown_top_level_and_empty_failure_analysis_are_rejected(tmp_path):
    unknown = proposal("cp_unknown", "Preserve supported details precisely.")
    unknown["legacy_field"] = "not canonical"
    with pytest.raises(PropertyProposalParseError, match="exactly"):
        parse_proposal_output(json.dumps({"proposals": [unknown]}), context(tmp_path),
                              max_proposals=1, max_property_text_chars=240)
    empty = proposal("cp_empty", "Preserve supported details precisely.",
                     failure_analysis=" ")
    with pytest.raises(PropertyProposalParseError, match="failure_analysis"):
        parse_proposal_output(json.dumps({"proposals": [empty]}), context(tmp_path),
                              max_proposals=1, max_property_text_chars=240)


def test_unresolved_placeholder_is_structurally_rejected(tmp_path):
    row = proposal("cp_placeholder", "Describe <missing behavior> precisely.")
    with pytest.raises(PropertyProposalParseError, match="unresolved placeholder"):
        parse_proposal_output(
            json.dumps({"proposals": [row]}), context(tmp_path),
            max_proposals=1, max_property_text_chars=240)


@pytest.mark.parametrize("when", [
    "Use at 01:23.",
    "Use for segment 10_20.",
    "Use when option A is correct.",
    "Use when the ground truth is green.",
    "Use when discussing green car.",
    "Use for historical documentaries.",
])
def test_applicability_lexical_overlap_is_not_a_preintervention_veto(tmp_path, when):
    row = proposal("cp_leak", "Preserve supported details precisely.")
    row["applicability"]["when"] = when
    accepted, rejected = parse_proposal_output(
        json.dumps({"proposals": [row]}), context(tmp_path),
        max_proposals=1, max_property_text_chars=240)
    assert len(accepted) == 1
    assert rejected == ()


def test_source_specific_failure_analysis_is_allowed_as_provenance(tmp_path):
    row = proposal(
        "cp_transcript", "Preserve attributed claims present in the transcript.",
        applicability={
            "when": "Use when a transcript contains an attributed claim absent from frames.",
            "positive_cues": ["A non-empty transcript attributes a claim to a speaker."],
            "negative_cues": ["The generated description already preserves the claim."],
            "required_modalities": ["transcript"],
        },
        failure_analysis=(
            "In this video the green car answer was supported by source evidence, but "
            "the generated description omitted it."),
    )
    accepted, rejected = parse_proposal_output(
        json.dumps({"proposals": [row]}), context(tmp_path),
        max_proposals=1, max_property_text_chars=240)
    assert rejected == ()
    assert accepted[0].failure_analysis.startswith("In this video")


def test_source_specific_terms_are_deferred_to_intervention_and_updater(
        tmp_path):
    ctx = context(tmp_path)
    rows = list(ctx.baseline_qa_results)
    rows[0] = {
        **rows[0],
        "question": "What claim is made about tuberculosis transmission?",
        "options": ["airborne", "waterborne", "unknown"],
        "ground_truth": "airborne",
    }
    source_context = type(ctx)(
        video_id=ctx.video_id, baseline_run_id=ctx.baseline_run_id,
        baseline_qa_results=tuple(rows), captions=ctx.captions,
        frame_references=ctx.frame_references, frozen_histories=ctx.frozen_histories,
        prompt_bank=ctx.prompt_bank, proposal_artifact_dir=ctx.proposal_artifact_dir)
    row = proposal(
        "cp_topic", "Preserve transcript claims about tuberculosis.",
        applicability={
            "when": "Use when tuberculosis is discussed in the transcript.",
            "positive_cues": ["The transcript mentions tuberculosis."],
            "negative_cues": [], "required_modalities": ["transcript"],
        })
    accepted, rejected = parse_proposal_output(
        json.dumps({"proposals": [row]}), source_context,
        max_proposals=1, max_property_text_chars=240)
    assert len(accepted) == 1
    assert rejected == ()


def test_transcript_suffix_is_structurally_separated_without_reclassification(tmp_path):
    ctx = context(tmp_path)
    captions = dict(ctx.captions)
    captions["0_10"] = {"caption": (
        "A speaker is visible.\n\nTranscript during this video clip: "
        "The speaker attributes the claim to a study.")}
    rows = list(ctx.baseline_qa_results)
    rows[0] = {
        **rows[0],
        "reference_sets": {"explicitly_cited_segments": ["0_10"]},
    }
    split_context = type(ctx)(
        video_id=ctx.video_id, baseline_run_id=ctx.baseline_run_id,
        baseline_qa_results=tuple(rows), captions=captions,
        frame_references=ctx.frame_references, frozen_histories=ctx.frozen_histories,
        prompt_bank=ctx.prompt_bank, proposal_artifact_dir=ctx.proposal_artifact_dir)
    request = build_proposal_request(
        split_context, max_proposals=1, max_trace_events_per_qa=3,
        max_captions=3, max_payload_chars=40000)
    evidence = request["qas"][0]["used_segment_evidence"][0]
    assert evidence["timestamp_range"] == "[0–10]"
    assert evidence["generated_description"] == "A speaker is visible."
    assert evidence["transcript"] == (
        "The speaker attributes the claim to a study.")
    assert "baseline_caption" not in evidence
    assert "mere presence" in request["task"]


def test_canonical_round_trip_preserves_applicability_and_omits_legacy_fields(tmp_path):
    accepted, _ = parse_proposal_output(
        json.dumps({"proposals": [proposal(
            "cp_roundtrip", "Preserve supported details precisely.")]}),
        context(tmp_path), max_proposals=1, max_property_text_chars=240)
    serialized = as_json_dict(accepted[0])
    loaded = candidate_property_proposal_from_json(serialized)
    assert loaded == accepted[0]
    assert loaded.applicability == accepted[0].applicability
    assert loaded.failure_analysis == accepted[0].failure_analysis
    assert "motivating_failure_types" not in serialized
    assert "proposal_rationale" not in serialized


def test_legacy_artifact_loads_only_at_compatibility_boundary():
    loaded = candidate_property_proposal_from_json({
        "candidate_property_id": "legacy-candidate",
        "property_text": "Preserve visible state changes.",
        "source_video_id": "legacy-video",
        "source_question_ids": ["q1"],
        "motivating_failure_types": ["missing_state"],
        "covered_by_existing_property_ids": ["pe_temporal"],
        "proposal_rationale": "A visible state change was omitted.",
        "proposer_policy_version": "legacy-v1",
    })
    serialized = as_json_dict(loaded)
    assert loaded.applicability.when.startswith("Apply when available captioning input")
    assert loaded.failure_analysis == "A visible state change was omitted."
    assert serialized["coverage_hints"] == ["pe_temporal"]
    assert "motivating_failure_types" not in serialized
    assert "proposal_rationale" not in serialized


def test_candidate_identity_includes_applicability(tmp_path):
    first_row = proposal("cp_same", "Preserve supported details precisely.")
    second_row = proposal("cp_same", "Preserve supported details precisely.")
    second_row["applicability"]["when"] = (
        "Use when transcript evidence contains a salient omitted detail.")
    second_row["applicability"]["required_modalities"] = ["transcript"]
    first, _ = parse_proposal_output(
        json.dumps({"proposals": [first_row]}), context(tmp_path),
        max_proposals=1, max_property_text_chars=240)
    second, _ = parse_proposal_output(
        json.dumps({"proposals": [second_row]}), context(tmp_path),
        max_proposals=1, max_property_text_chars=240)
    assert first[0].candidate_property_id != second[0].candidate_property_id


def test_exact_proposal_resume_skips_provider_and_stale_input_fails_closed(tmp_path):
    provider = Provider({"proposals": [
        proposal("cp_color", "Record salient object colors precisely.")]})
    policy = MultiPropertyProposalPolicy(response_provider=provider)
    ctx = context(tmp_path)
    first = tuple(policy.propose(ctx))
    second = tuple(policy.propose(ctx))
    assert first == second
    assert len(provider.calls) == 1

    stale = context(tmp_path, prediction="C")
    with pytest.raises(PropertyProposalConflictError):
        policy.propose(stale)
    assert len(provider.calls) == 1


def test_ineligible_zero_proposal_resume_remains_provider_free(tmp_path):
    provider = Provider({"proposals": []})
    policy = MultiPropertyProposalPolicy(response_provider=provider)
    ctx = context(tmp_path)
    rows = tuple({**row, "annotation_reliable": False}
                 for row in ctx.baseline_qa_results)
    ctx = type(ctx)(
        video_id=ctx.video_id, baseline_run_id=ctx.baseline_run_id,
        baseline_qa_results=rows, captions=ctx.captions,
        frame_references=ctx.frame_references,
        frozen_histories=ctx.frozen_histories, prompt_bank=ctx.prompt_bank,
        proposal_artifact_dir=ctx.proposal_artifact_dir)

    assert policy.propose(ctx) == ()
    assert policy.propose(ctx) == ()
    assert len(provider.calls) == 1
