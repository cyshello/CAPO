import json
from pathlib import Path

import pytest
from PIL import Image

from surrogate_rollout.optimization.interventional_feedback import (
    FeedbackEvidenceBounds,
    FrameTransformConfig,
    InterventionalFeedbackError,
    InterventionalFeedbackConflictError,
    PropertyFeedbackRunner,
    load_bounded_frame,
)
from surrogate_rollout.optimization.property_proposal import CandidatePropertyProposal
from surrogate_rollout.prompt_routing.persistence import prompt_bank_from_json


ROOT = Path(__file__).parents[1]
SEGMENTS = ("0_10", "10_20", "20_30")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return str(path.resolve())


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return str(path.resolve())


def prompt_bank():
    value = json.loads((
        ROOT / "prompt_routing/fixtures/stage4_7_components.json").read_text())
    return prompt_bank_from_json(value["prompt_bank"])


def proposal(candidate_id):
    return CandidatePropertyProposal(
        candidate_property_id=candidate_id,
        property_text=f"Track reusable fine-grained actions for {candidate_id}.",
        source_video_id="video-1", source_question_ids=("q1", "q2"),
        motivating_failure_types=("missed_action",),
        covered_by_existing_property_ids=(), proposal_rationale="fixture",
        proposer_policy_version="fixture_v1")


def refs(*segments):
    return {
        "retrieved_segments": [], "returned_segments": [],
        "frame_inspected_segments": [], "explicitly_cited_segments": [],
        "consumed_segments": list(segments),
    }


def fixture(tmp_path, candidates=("cp_a",), *, empty_candidate=None):
    baseline_dir = tmp_path / "baseline"
    frame_map = {}
    histories = []
    captions = {}
    for segment_id in SEGMENTS:
        frame_paths = []
        for index in range(2):
            path = baseline_dir / "frames" / f"{segment_id}-{index}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"frame-{segment_id}-{index}".encode())
            frame_paths.append(str(path.resolve()))
        frame_map[segment_id] = frame_paths
        histories.append({
            "segment_id": segment_id, "history_hash": f"history-{segment_id}",
            "history": [
                {"segment_id": "prior-1", "caption": "x" * 80},
                {"segment_id": "prior-2", "caption": "recent history"},
            ],
        })
        captions[segment_id] = {"caption": f"baseline caption {segment_id} " + "b" * 80}
    captions["subject_registry"] = {}
    captions_path = write_json(baseline_dir / "captions.json", captions)
    frames_path = write_json(baseline_dir / "frames.json", frame_map)
    histories_path = write_jsonl(
        baseline_dir / "frozen_histories.jsonl", histories)

    before_used = {
        "q1": ("0_10", "20_30"),
        "q2": ("10_20",),
        "q3": ("20_30",),
    }
    baseline_qas = []
    for index, question_id in enumerate(("q1", "q2", "q3"), 1):
        trajectory = write_jsonl(
            baseline_dir / "qa" / question_id / "trajectory.jsonl",
            [{"step": 1, "segment": before_used[question_id][0],
              "detail": "r" * 100}, {"step": 2, "detail": "other"}])
        baseline_qas.append({
            "video_id": "video-1", "question_id": question_id,
            "provider_index": index, "question": f"What occurs in case {index}?",
            "options": ["Alpha", "Beta"], "ground_truth": "Alpha",
            "prediction": "Beta" if question_id != "q2" else "Alpha",
            "is_correct": question_id == "q2", "used_segments": before_used[question_id],
            "reference_sets": refs(*before_used[question_id]),
            "trajectory_path": trajectory,
        })
    baseline_qas_path = write_jsonl(
        baseline_dir / "baseline_qas.jsonl", baseline_qas)
    baseline_manifest = write_json(baseline_dir / "video_complete.json", {
        "video_id": "video-1", "video_fingerprint": "fixture-baseline",
        "baseline_qas_path": baseline_qas_path, "captions_path": captions_path,
        "frames_path": frames_path, "frozen_histories_path": histories_path,
    })

    proposals = tuple(proposal(candidate_id) for candidate_id in candidates)
    retrieval_paths = []
    intervention_results = []
    for item in proposals:
        selected = (("10_20",) if item.candidate_property_id == empty_candidate
                    else ("0_10", "10_20"))
        retrieval_paths.append(write_json(
            tmp_path / "retrieval" / item.candidate_property_id / "retrieval.json", {
                "candidate_property_id": item.candidate_property_id,
                "source_video_id": "video-1", "property_text": item.property_text,
                "s_sim": selected,
                "input_fingerprint": f"retrieval-{item.candidate_property_id}",
            }))
        candidate_dir = tmp_path / "interventions" / "video-1" / item.candidate_property_id
        write_json(candidate_dir / "work_item.json", {"candidate": item.candidate_property_id})
        write_jsonl(candidate_dir / "composed_prompts.jsonl", [{
            "segment_id": segment_id,
            "selected_prompt_ids": ["pe_default", "pe_temporal",
                                    item.candidate_property_id],
        } for segment_id in selected])
        write_jsonl(candidate_dir / "frozen_histories.jsonl", [
            history for history in histories if history["segment_id"] in selected])
        mixed = dict(captions)
        for segment_id in selected:
            mixed[segment_id] = {
                "caption": f"candidate caption {item.candidate_property_id} " + "c" * 80}
        mixed_path = write_json(candidate_dir / "mixed.json", mixed)
        after_used = {
            "q1": (() if item.candidate_property_id == empty_candidate else ("10_20",)),
            "q2": (() if item.candidate_property_id == empty_candidate else ("10_20", "20_30")),
            "q3": ("20_30",),
        }
        qa_rows = []
        transitions = {
            "q1": "wrong_to_correct",
            "q2": ("correct_to_correct"
                   if item.candidate_property_id == empty_candidate
                   else "correct_to_wrong"),
            "q3": "wrong_to_wrong",
        }
        for question_id in ("q1", "q2", "q3"):
            trajectory = write_jsonl(
                candidate_dir / "qa" / question_id / "trajectory.jsonl",
                [{"step": 1, "segments": after_used[question_id],
                  "detail": "z" * 100}, {"step": 2, "detail": "other"}])
            qa_rows.append({
                "question_id": question_id,
                "transition": transitions[question_id],
                "candidate_prediction": (
                    "Alpha" if question_id == "q1" else "Beta"),
                "reference_sets": refs(*after_used[question_id]),
                "trajectory_path": trajectory,
            })
        transitions_path = write_json(candidate_dir / "transitions.json", {
            "candidate_property_id": item.candidate_property_id,
            "source_video_id": "video-1", "qas": qa_rows,
        })
        result_path = write_json(candidate_dir / "result.json", {
            "candidate_property_id": item.candidate_property_id,
            "source_video_id": "video-1", "status": "completed",
            "parent_baseline_identity": "parent-fixture",
            "mixed_captions_path": mixed_path,
            "transitions_path": transitions_path,
        })
        intervention_results.append({
            "candidate_property_id": item.candidate_property_id,
            "source_video_id": "video-1", "status": "completed",
            "result_path": result_path,
        })
    intervention_manifest = write_json(
        tmp_path / "interventions" / "manifest.json", {
            "source_video_id": "video-1", "status": "completed",
            "results": intervention_results,
        })
    return {
        "baseline": baseline_manifest, "proposals": proposals,
        "retrievals": tuple(retrieval_paths),
        "intervention": intervention_manifest,
    }


class MockProvider:
    def __init__(self, *, leak=False):
        self.calls = []
        self.leak = leak

    def __call__(self, request_text):
        request = json.loads(request_text)
        self.calls.append(request)
        if self.leak:
            evidence = "The video-1 answer option proves this property helps."
        elif request["transition"] == "wrong_to_correct":
            evidence = "Fine-grained action tracking supplies reusable visual evidence."
        else:
            evidence = "Overemphasizing incidental motion can obscure stable event outcomes."
        positive = request["transition"] == "wrong_to_correct"
        return json.dumps({
            "assessment": "credit" if positive else "blame",
            "coverage_assessment": "uncovered" if positive else "covered",
            "covered_by_property_ids": [] if positive else ["pe_temporal"],
            "evidence": evidence,
            "recommendation": "add" if positive else "router_negative",
        })


class FlakyProvider(MockProvider):
    def __init__(self):
        super().__init__()
        self.fail = True

    def __call__(self, request_text):
        if self.fail:
            self.calls.append(json.loads(request_text))
            raise RuntimeError("fixture transient provider failure")
        return super().__call__(request_text)


def frame_loader(path, max_bytes, transform):
    data = Path(path).read_bytes()
    assert len(data) <= max_bytes
    transformed_hash = __import__("hashlib").sha256(
        data + json.dumps(transform.__dict__, sort_keys=True).encode()).hexdigest()
    return {
        "frame_id": Path(path).name, "fixture_pixels": data.decode(),
        "transformed_sha256": transformed_hash,
        "transformed_bytes": len(data),
        "transform_configuration": transform.__dict__,
    }


def run_feedback(tmp_path, artifacts, provider, *, output="feedback", bounds=None,
                 frame_transform=None, retry_failed=False):
    return PropertyFeedbackRunner(
        response_provider=provider, frame_loader=frame_loader,
        bounds=bounds, frame_transform=frame_transform).run(
            iteration_id="iteration-1",
            baseline_video_manifest_path=artifacts["baseline"],
            proposals=artifacts["proposals"],
            retrieval_artifact_paths=artifacts["retrievals"],
            intervention_manifest_path=artifacts["intervention"],
            prompt_bank=prompt_bank(), output_dir=str(tmp_path / output),
            retry_failed=retry_failed)


def test_flip_only_feedback_intersection_bounds_aggregation_and_resume(tmp_path):
    artifacts = fixture(tmp_path)
    provider = MockProvider()
    bounds = FeedbackEvidenceBounds(
        max_frames_per_segment=1, max_history_items=1, max_caption_chars=24,
        max_reasoning_events=1, max_reasoning_event_chars=32)
    result = run_feedback(tmp_path, artifacts, provider, bounds=bounds)
    aggregate = result.properties[0]
    assert aggregate.positive_flip_count == 1
    assert aggregate.negative_flip_count == 1
    assert aggregate.unchanged_count == 1
    assert aggregate.accepted_feedback_count == 2
    assert aggregate.supporting_s_feedback == ("0_10", "10_20")
    assert aggregate.coverage_assessment == "mixed"
    assert aggregate.covered_by_property_ids == ("pe_temporal",)
    assert {item["recommendation"] for item in aggregate.recommendation_evidence} == {
        "add", "router_negative"}
    assert len(provider.calls) == 2
    positive, negative = provider.calls
    assert [item["segment_id"] for item in positive["segments"]] == ["0_10", "10_20"]
    assert [item["segment_id"] for item in negative["segments"]] == ["10_20"]
    for request in provider.calls:
        assert all(len(segment["frames"]) == 1 for segment in request["segments"])
        assert all(len(segment["frozen_history"]) == 1
                   for segment in request["segments"])
        assert all(len(segment["before_caption"]) <= 24
                   for segment in request["segments"])
        assert len(request["reasoning_excerpts"]["before"]) <= 1
        assert all(len(item) <= 32
                   for item in request["reasoning_excerpts"]["before"])
    unchanged = next(item for item in aggregate.qa_feedback
                     if item.question_id == "q3")
    assert unchanged.status == "not_eligible_unchanged"
    assert unchanged.request_path is None

    resumed = run_feedback(tmp_path, artifacts, provider, bounds=bounds)
    assert resumed.properties[0].resumed
    assert len(provider.calls) == 2


def test_empty_s_feedback_rejected_and_sibling_aggregates_independently(tmp_path):
    artifacts = fixture(tmp_path, ("cp_empty", "cp_b"), empty_candidate="cp_empty")
    provider = MockProvider()
    result = run_feedback(tmp_path, artifacts, provider)
    empty, sibling = result.properties
    assert empty.accepted_feedback_count == 0
    assert empty.rejected_feedback_count == 1
    assert {item.rejection_reason for item in empty.qa_feedback
            if item.status == "rejected"} == {"empty_s_feedback"}
    assert sibling.accepted_feedback_count == 2
    assert len(provider.calls) == 2


def test_leaking_feedback_is_rejected_with_exact_artifacts(tmp_path):
    artifacts = fixture(tmp_path)
    provider = MockProvider(leak=True)
    result = run_feedback(tmp_path, artifacts, provider)
    aggregate = result.properties[0]
    assert aggregate.accepted_feedback_count == 0
    assert aggregate.rejected_feedback_count == 2
    for item in aggregate.qa_feedback:
        if item.transition not in ("wrong_to_correct", "correct_to_wrong"):
            continue
        assert Path(item.request_path).exists()
        assert Path(item.raw_output_path).read_text()
        assert json.loads(Path(item.parsed_output_path).read_text()) == {"parsed": None}
        rejection = json.loads(Path(item.rejection_path).read_text())
        assert "source lineage" in rejection["rejection_reason"]


def test_changed_frozen_input_fails_closed(tmp_path):
    artifacts = fixture(tmp_path)
    provider = MockProvider()
    run_feedback(tmp_path, artifacts, provider)
    baseline = json.loads(Path(artifacts["baseline"]).read_text())
    captions = Path(baseline["captions_path"])
    captions.write_text(captions.read_text() + "\n")
    with pytest.raises(InterventionalFeedbackConflictError):
        run_feedback(tmp_path, artifacts, provider)


def test_conflicting_partial_qa_artifact_fails_closed(tmp_path):
    artifacts = fixture(tmp_path)
    provider = MockProvider()
    result = run_feedback(tmp_path, artifacts, provider)
    aggregate = result.properties[0]
    Path(result.manifest_path).unlink()
    Path(aggregate.result_path).unlink()
    accepted = next(item for item in aggregate.qa_feedback
                    if item.status == "accepted")
    request = json.loads(Path(accepted.request_path).read_text())
    request["transition"] = "correct_to_wrong"
    Path(accepted.request_path).write_text(json.dumps(request, sort_keys=True))
    with pytest.raises(InterventionalFeedbackConflictError):
        run_feedback(tmp_path, artifacts, provider)


def test_bounded_frame_transformation_is_deterministic_and_hashed(tmp_path):
    path = tmp_path / "pattern.png"
    image = Image.new("RGB", (128, 96))
    image.putdata([
        ((x * 17 + y * 3) % 256, (x * 5 + y * 11) % 256,
         (x * 13 + y * 7) % 256)
        for y in range(96) for x in range(128)
    ])
    image.save(path, format="PNG")
    transform = FrameTransformConfig(
        max_dimension=64, initial_quality=80, minimum_quality=40,
        quality_step=10, max_resize_steps=2)
    first = load_bounded_frame(str(path), 2500, transform)
    second = load_bounded_frame(str(path), 2500, transform)
    assert first == second
    assert first["transformed_sha256"]
    assert first["transformed_bytes"] <= 2500
    assert max(first["transformed_dimensions"]) <= 64
    assert first["transform_configuration"]["max_dimension"] == 64


def test_frame_transform_configuration_is_request_identity(tmp_path):
    artifacts = fixture(tmp_path)
    provider = MockProvider()
    run_feedback(
        tmp_path, artifacts, provider,
        frame_transform=FrameTransformConfig(max_dimension=640))
    with pytest.raises(InterventionalFeedbackConflictError):
        run_feedback(
            tmp_path, artifacts, provider,
            frame_transform=FrameTransformConfig(max_dimension=320))


def test_frame_rejected_only_after_bounded_transform_exhausted(tmp_path):
    path = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path, format="PNG")
    with pytest.raises(InterventionalFeedbackError, match="after bounded transformation"):
        load_bounded_frame(
            str(path), 10,
            FrameTransformConfig(
                max_dimension=1, initial_quality=35, minimum_quality=35,
                max_resize_steps=0))


def test_failed_property_retained_and_explicit_retry_is_isolated(tmp_path):
    artifacts = fixture(tmp_path)
    provider = FlakyProvider()
    failed = run_feedback(tmp_path, artifacts, provider)
    original = failed.properties[0]
    assert original.status == "failed"
    original_bytes = Path(original.result_path).read_bytes()

    provider.fail = False
    retained = run_feedback(tmp_path, artifacts, provider)
    assert retained.properties[0].status == "failed"
    assert retained.properties[0].resumed
    assert Path(original.result_path).read_bytes() == original_bytes

    retried = run_feedback(
        tmp_path, artifacts, provider, retry_failed=True)
    assert retried.properties[0].status == "completed"
    assert "/retries/retry_001/" in retried.properties[0].result_path
    assert Path(original.result_path).read_bytes() == original_bytes
