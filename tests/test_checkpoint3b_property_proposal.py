import json
from pathlib import Path

import pytest

from surrogate_rollout.optimization.policies.property_proposal import (
    MultiPropertyProposalPolicy,
    PropertyProposalConflictError,
    PropertyProposalParseError,
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
            "reasoning_trace": [{"step": "looked at motion"}],
            "used_segments": ["0_10"],
        },
        {
            "question_id": "q2", "question": "What happens after the door opens?",
            "options": ["A", "B", "C"], "ground_truth": "B",
            "prediction": "B", "is_correct": True,
            "reasoning_trace": [{"step": "ordered actions"}],
            "used_segments": ["10_20"],
        },
        {
            "question_id": "q3", "question": "Which label appears briefly?",
            "options": ["EXIT", "SHOP", "HOME"], "ground_truth": "EXIT",
            "prediction": "EXIT", "is_correct": True,
            "reasoning_trace": [], "used_segments": ["20_30"],
        },
    )
    return VideoProposalContext(
        video_id="video-1", baseline_run_id="baseline-1",
        baseline_qa_results=rows,
        captions={
            "0_10": {"caption": "A car moves."},
            "10_20": {"caption": "A door opens."},
            "20_30": {"caption": "A sign is visible."},
        },
        frame_references={}, frozen_histories=(),
        prompt_bank=prompt_bank_from_json(data["prompt_bank"]),
        proposal_artifact_dir=str(tmp_path / "proposal"),
    )


def proposal(candidate_id, text, qids=("q1",), **over):
    value = {
        "candidate_property_id": candidate_id,
        "property_text": text,
        "source_video_id": "video-1",
        "source_question_ids": list(qids),
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
    assert request["qas"][0]["question_id"] == "q1"  # incorrect first
    assert request["qas"][0]["answer_choices"] == ["red", "green", "blue"]
    assert request["qas"][0]["gold_answer"] == "green car"
    if expected == 2:
        assert result[1].source_question_ids == ("q1", "q2")
    artifact_dir = Path(context(tmp_path).proposal_artifact_dir)
    assert (artifact_dir / "raw_output.txt").exists()
    parsed = json.loads((artifact_dir / "parsed_output.json").read_text())
    assert len(parsed["proposals"]) == expected


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
