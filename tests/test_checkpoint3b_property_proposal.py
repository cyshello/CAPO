import json
from pathlib import Path

import pytest
from PIL import Image

from surrogate_rollout.optimization.policies.property_proposal import (
    MultiPropertyProposalPolicy,
    OpenAIPropertyProposalProvider,
    PropertyProposalConflictError,
    PropertyProposalParseError,
    build_proposal_request,
    parse_proposal_output,
)
from surrogate_rollout.optimization.property_proposal import VideoProposalContext
from surrogate_rollout.prompt_routing.persistence import prompt_bank_from_json


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


def proposal(candidate_id, text, qids=("q1",), **over):
    del qids
    value = {
        "candidate_property_id": candidate_id,
        "property_text": text,
        "motivating_failure_types": ["missing_visual_attribute"],
        "covered_by_existing_property_ids": [],
        "proposal_rationale": "The baseline omitted a reusable visual detail.",
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


@pytest.mark.parametrize("rows, expected", [
    ([], 0),
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
        assert result[1].source_question_ids == ("q1", "q2", "q3")
    artifact_dir = Path(context(tmp_path).proposal_artifact_dir)
    assert (artifact_dir / "raw_output.txt").exists()
    identity = json.loads((artifact_dir / "input_identity.json").read_text())
    assert identity["source_video_id"] == "video-1"
    assert identity["qas"][0]["question_id"] == "q1"
    parsed = json.loads((artifact_dir / "parsed_output.json").read_text())
    assert len(parsed["proposals"]) == expected


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
    assert all(len(row["used_segment_evidence"]) <= 3 for row in request["qas"])
    assert all(len(item) <= 1201 for row in request["qas"]
               for item in row["bounded_reasoning"])
    assert "payload_truncation" not in request
    assert request["output_schema"]["proposals"][0][
        "covered_by_existing_property_ids"] == []
    assert "non-binding hints" in request["task"]
    assert "non-visual knowledge" in request["task"]


def test_openai_provider_builds_exact_multimodal_body(tmp_path):
    request = build_proposal_request(
        context(tmp_path), max_proposals=1, max_trace_events_per_qa=3,
        max_captions=3, max_payload_chars=40000)
    provider = OpenAIPropertyProposalProvider(model="gpt-test", api_key="unused")
    body = provider.build_request_body(request)
    content = body["messages"][0]["content"]

    assert content[0]["type"] == "text"
    assert sum(item["type"] == "image_url" for item in content) == 3
    assert all(item["image_url"]["url"].startswith("data:image/jpeg;base64,")
               for item in content if item["type"] == "image_url")
    assert "base64_data" not in content[0]["text"]


def test_policy_persists_private_identity_and_exact_provider_body(tmp_path):
    class MultimodalProvider:
        supports_multimodal_request = True

        def __init__(self):
            self.calls = []

        def build_request_body(self, request):
            return {"exact": True, "request_schema": request["schema_version"]}

        def __call__(self, request):
            self.calls.append(request)
            return json.dumps({"proposals": []})

    provider = MultimodalProvider()
    ctx = context(tmp_path)
    result = MultiPropertyProposalPolicy(response_provider=provider).propose(ctx)

    assert result == ()
    assert isinstance(provider.calls[0], dict)
    artifact_dir = Path(ctx.proposal_artifact_dir)
    assert json.loads((artifact_dir / "provider_request.json").read_text()) == {
        "exact": True,
        "request_schema": "multimodal_property_proposal_request_v3",
    }
    model_request = (artifact_dir / "request.json").read_text()
    private_identity = (artifact_dir / "input_identity.json").read_text()
    assert "source_video_id" not in model_request
    assert "source_video_id" in private_identity


def test_instance_specific_answer_leaking_and_codebook_duplicates_are_rejected(tmp_path):
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
    assert accepted == ()
    assert {item["candidate_property_id"] for item in rejected} == {
        "cp_video", "cp_answer", "cp_existing"}
    exact = next(item for item in rejected
                 if item["candidate_property_id"] == "cp_existing")
    assert exact["reason"] == "exact_property_text_match:pe_default"


def test_coverage_hints_are_non_binding_and_persisted_for_later_feedback(tmp_path):
    ctx = context(tmp_path)
    provider = Provider({"proposals": [proposal(
        "cp_roles",
        "Describe visually observable instructional interactions and roles clearly.",
        covered_by_existing_property_ids=["pe_default", "pe_temporal"],
        proposal_rationale=(
            "This may overlap general visual and temporal properties, but the "
            "intervention should determine whether explicit role distinctions help."),
    )]})
    result = tuple(MultiPropertyProposalPolicy(
        response_provider=provider).propose(ctx))

    assert len(result) == 1
    assert result[0].coverage_hints == ("pe_default", "pe_temporal")
    assert result[0].covered_by_existing_property_ids == result[0].coverage_hints
    assert result[0].coverage_assessment == "deferred_to_intervention"
    parsed = json.loads((Path(ctx.proposal_artifact_dir) /
                         "parsed_output.json").read_text())
    assert parsed["proposals"][0]["coverage_hints"] == [
        "pe_default", "pe_temporal"]
    assert "covered_by_existing_property_ids" not in parsed["proposals"][0]


def test_contradictory_coverage_and_nonvisual_knowledge_are_explicitly_rejected(
        tmp_path):
    ctx = context(tmp_path)
    raw = json.dumps({"proposals": [
        proposal(
            "cp_contradiction", "Describe visually observable teaching roles.",
            covered_by_existing_property_ids=["pe_default"],
            proposal_rationale="No existing property covers this instruction."),
        proposal(
            "cp_history", "Provide historical context for depicted events.",
            proposal_rationale="External context could explain the scene."),
    ]})
    accepted, rejected = parse_proposal_output(
        raw, ctx, max_proposals=4, max_property_text_chars=240)

    assert accepted == ()
    assert {item["candidate_property_id"]: item["reason"] for item in rejected} == {
        "cp_contradiction": "contradictory_coverage_hints_without_uncertainty",
        "cp_history": (
            "requires non-visual or external/background/historical knowledge"),
    }


def test_duplicate_and_malformed_proposals_fail_strictly(tmp_path):
    ctx = context(tmp_path)
    duplicated = json.dumps({"proposals": [
        proposal("cp_a", "Record salient colors."),
        proposal("cp_b", "Record salient colors."),
    ]})
    with pytest.raises(PropertyProposalParseError, match="duplicated"):
        parse_proposal_output(
            duplicated, ctx, max_proposals=4, max_property_text_chars=240)
    malformed = json.dumps({"proposals": [{"property_text": "missing fields"}]})
    with pytest.raises(PropertyProposalParseError, match="exactly"):
        parse_proposal_output(
            malformed, ctx, max_proposals=4, max_property_text_chars=240)


def test_active_property_id_collision_is_deterministically_rejected(tmp_path):
    accepted, rejected = parse_proposal_output(
        json.dumps({"proposals": [proposal(
            "pe_default", "Describe visually observable teaching roles.")]}),
        context(tmp_path), max_proposals=1, max_property_text_chars=240)

    assert accepted == ()
    assert rejected == ({
        "candidate_property_id": "pe_default",
        "reason": "candidate_property_id_collides_with_active_property",
    },)


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


def test_zero_proposal_resume_remains_provider_free(tmp_path):
    provider = Provider({"proposals": []})
    policy = MultiPropertyProposalPolicy(response_provider=provider)
    ctx = context(tmp_path)

    assert policy.propose(ctx) == ()
    assert policy.propose(ctx) == ()
    assert len(provider.calls) == 1
