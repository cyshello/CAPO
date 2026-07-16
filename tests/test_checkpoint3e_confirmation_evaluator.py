import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from surrogate_rollout import config
from surrogate_rollout.captioning.history_aware_baseline import (
    HistoryAwareBaselineCaptionViewBuilder,
    HistoryAwareSegmentCaptioner,
)
from surrogate_rollout.optimization.confirmation_evaluator import (
    ConfirmationEvaluationConflictError,
    HistoryAwareDVDConfirmationEvaluator,
)
from surrogate_rollout.optimization.final_iteration import Checkpoint3EOrchestrator
from surrogate_rollout.optimization.iteration_state import (
    ComponentVersions,
    ConfirmedCheckpointState,
)
from surrogate_rollout.optimization.train_roles import (
    TrainRoleAssignment,
    TrainVideoRecord,
)
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.schemas import RoutingDecision


ROOT = Path(__file__).parents[1]
BASE_PROMPT = (
    "TRANSCRIPT_PLACEHOLDER CLIP_START_TIME CLIP_END_TIME "
    "ROUTED_INSTRUCTIONS_PLACEHOLDER")


def components():
    value = json.loads((
        ROOT / "prompt_routing/fixtures/stage4_7_components.json").read_text())
    bank = prompt_bank_from_json(value["prompt_bank"])
    router = router_policy_from_json(value["router_policy"])
    return (
        bank, router,
        scaffold_policy_from_json(value["scaffold_policy"]),
        scaffold_contract_from_json(value["scaffold_contract"]),
    )


class PolicyRouter:
    def __init__(self):
        self.calls = []

    def route(self, context, prompt_bank, router_policy):
        selected = tuple(router_policy.configuration.get("test_select_ids") or ())
        self.calls.append({
            "video_id": context.video_id, "segment_id": context.segment_id,
            "frames": tuple(context.segment_features["frame_references"]),
            "history": json.loads(context.history_summary),
            "bank_version": prompt_bank.bank_version,
            "router_version": router_policy.router_version,
        })
        return RoutingDecision(
            video_id=context.video_id, segment_id=context.segment_id,
            bank_version=prompt_bank.bank_version,
            router_version=router_policy.router_version,
            selected_prompt_ids=selected,
            selection_order=selected,
            used_fallback=not selected,
            fallback_reason=("fixture_base_only" if not selected else None))


class CaptionVLM:
    def __init__(self):
        self.calls = []

    def caption(self, images, prompt, **kwargs):
        policy = "candidate" if "on-screen text" in prompt else "parent"
        history = json.loads(prompt.split(
            "FROZEN_PRECEDING_CAPTION_HISTORY_JSON:\n", 1)[1])
        self.calls.append({
            "images": tuple(images), "prompt": prompt,
            "history": history, "policy": policy, "kwargs": kwargs,
        })
        index = len(history["preceding_captions"])
        return json.dumps({
            "clip_description": f"{policy} caption {index}",
            "subject_registry": {"S1": policy},
        })


class QASpy:
    def __init__(self, *, reject_candidate=False):
        self.calls = []
        self.reject_candidate = reject_candidate

    def __call__(self, **kwargs):
        state = ("candidate" if "/candidate/" in kwargs["captions_path"]
                 else "parent")
        self.calls.append({
            "state": state, "question_id": kwargs["question_id"],
            "video_id": kwargs["sample"]["extra"]["videoID"],
            "sample": kwargs["sample"],
            "max_iterations": kwargs["max_iterations"],
            "gpu": kwargs["gpu"],
        })
        score = 0.0 if self.reject_candidate and state == "candidate" and \
            len([call for call in self.calls if call["state"] == "candidate"]) == 1 \
            else 1.0
        return SimpleNamespace(
            prediction="A" if score else "B", parsed_answer="A" if score else "B",
            ground_truth="A", score=score, errors=[], latency_seconds=0.01)


def fixture_inputs(tmp_path):
    records = []
    samples = {}
    clips = {}
    for video_index in range(2):
        video_id = f"confirm-{video_index}"
        video_path = tmp_path / f"{video_id}.mp4"
        video_path.write_bytes(f"video-{video_index}".encode())
        segments = []
        for segment_index, segment_id in enumerate(("0_10", "10_20")):
            frame = tmp_path / f"{video_id}-{segment_index}.jpg"
            frame.write_bytes(f"frame-{video_index}-{segment_index}".encode())
            segments.append((segment_id, {
                "files": [str(frame)],
                "transcript": f"transcript {video_index} {segment_index}",
            }))
        clips[video_id] = segments
        indices = tuple(100 + video_index * 3 + offset for offset in range(3))
        question_ids = tuple(f"{video_id}-q{offset}" for offset in range(3))
        for provider_index, question_id in zip(indices, question_ids):
            samples[provider_index] = {
                "sample_id": question_id,
                "video_path": str(video_path),
                "question": f"Question {question_id}?",
                "options": ["A", "B", "C", "D"],
                "answer": "A",
                "extra": {"videoID": video_id, "subtitle_path": None},
            }
        records.append(TrainVideoRecord(
            video_id=video_id, provider_indices=indices,
            question_ids=question_ids, previously_cached=False))
    return tuple(records), samples, clips


def make_evaluator(tmp_path, *, reject_candidate=False,
                   history_block_seconds=300.0):
    records, samples, clips = fixture_inputs(tmp_path)
    router = PolicyRouter()
    vlm = CaptionVLM()
    builder = HistoryAwareBaselineCaptionViewBuilder(
        router=router,
        segment_captioner=HistoryAwareSegmentCaptioner(
            vlm, caption_model_id="fixture-qwen",
            backend_id="fixture-history-backend"),
        merge_fn=lambda values: {"count": len(values)})
    qa = QASpy(reject_candidate=reject_candidate)
    evaluator = HistoryAwareDVDConfirmationEvaluator(
        sample_loader=lambda index: samples[index],
        history_aware_builder=builder,
        clip_index_fn=lambda _sample, video_id: clips[video_id],
        qa_fn=qa, base_prompt_template=BASE_PROMPT, merge_prompt="merge",
        sample_source_identity="fixture-confirmation-split-v1",
        cache_root=str(tmp_path / "shared-cache"),
        cache_manifest_path=str(tmp_path / "cache-manifest.jsonl"),
        history_block_seconds=history_block_seconds,
        max_history_captions=5, dvd_max_iterations=4,
        downstream_qa_configuration={
            "reasoning_model": "fixture-dvd", "tool_policy": "fixed"},
    )
    return evaluator, records, router, vlm, qa


def policies():
    bank, router, scaffold, contract = components()
    parent_router = replace(
        router, configuration={"test_select_ids": []})
    candidate_bank = replace(
        bank, bank_version="bank_v0003",
        parent_bank_version=bank.bank_version)
    candidate_router = replace(
        router, router_version="router_v0003",
        parent_router_version=router.router_version,
        configuration={"test_select_ids": ["pe_text"]})
    return (bank, parent_router, candidate_bank, candidate_router,
            scaffold, contract)


def evaluate(evaluator, records, tmp_path, *, same_policy=False):
    bank, parent_router, candidate_bank, candidate_router, scaffold, contract = \
        policies()
    if same_policy:
        candidate_bank, candidate_router = bank, parent_router
    result = evaluator.evaluate(
        confirmation_videos=records,
        incumbent_bank=bank, incumbent_router=parent_router,
        candidate_bank=candidate_bank, candidate_router=candidate_router,
        scaffold_policy=scaffold, scaffold_contract=contract,
        output_dir=str(tmp_path / "confirmation"))
    return result, json.loads(Path(result["manifest_path"]).read_text())


def test_shared_bundle_on_policy_histories_and_cache_isolation(tmp_path):
    evaluator, records, router, vlm, qa = make_evaluator(tmp_path)
    result, manifest = evaluate(evaluator, records, tmp_path)
    bundle = json.loads(Path(result["input_bundle_path"]).read_text())

    assert bundle["confirmation_video_ids"] == [
        "confirm-0", "confirm-1"]
    assert len(bundle["videos"]) == 2
    assert all(len(video["qas"]) == 3 for video in bundle["videos"])
    assert all(len(video["segments"]) == 2 for video in bundle["videos"])
    assert all(frame["content_hash"] for video in bundle["videos"]
               for segment in video["segments"] for frame in segment["frames"])
    assert {
        "frame_sampling_identity", "caption_model_hash",
        "caption_backend_hash", "router_adapter_hash",
        "caption_decoding_hash", "history_configuration_hash",
        "downstream_qa_configuration_hash", "scaffold_version",
        "contract_version",
    } <= set(bundle["runtime_configuration"])
    assert manifest["parent"]["input_bundle_hash"] == result["input_bundle_hash"]
    assert manifest["candidate"]["input_bundle_hash"] == result["input_bundle_hash"]
    assert manifest["parent"]["runtime_configuration_hash"] == \
        manifest["candidate"]["runtime_configuration_hash"]

    parent_calls = vlm.calls[:4]
    candidate_calls = vlm.calls[4:]
    assert [call["images"] for call in parent_calls] == \
        [call["images"] for call in candidate_calls]
    parent_second = manifest["parent"]["videos"][0]["segments"][1]
    candidate_second = manifest["candidate"]["videos"][0]["segments"][1]
    assert parent_second["preceding_captions"][0]["caption"].startswith(
        "parent caption 0")
    assert candidate_second["preceding_captions"][0]["caption"].startswith(
        "candidate caption 0")
    assert parent_second["history_hash"] != candidate_second["history_hash"]

    for parent_video, candidate_video in zip(
            manifest["parent"]["videos"], manifest["candidate"]["videos"]):
        for parent_segment, candidate_segment in zip(
                parent_video["segments"], candidate_video["segments"]):
            assert parent_segment["cache_identity"] != \
                candidate_segment["cache_identity"]
            assert parent_segment["caption_result_path"] != \
                candidate_segment["caption_result_path"]
    assert len(router.calls) == 8
    assert len(qa.calls) == 12
    assert {(call["state"], call["video_id"]) for call in qa.calls} == {
        (state, video) for state in ("parent", "candidate")
        for video in ("confirm-0", "confirm-1")}
    assert len(result["qas"]) == 6
    assert result["proposals_generated"] == 0
    assert result["interventions"] == 0
    assert result["feedback_calls"] == 0


def test_exact_full_identity_reuses_cache_and_completed_resume_is_call_free(
        tmp_path):
    evaluator, records, router, vlm, qa = make_evaluator(tmp_path)
    result, manifest = evaluate(evaluator, records, tmp_path, same_policy=True)
    assert all(video["caption_calls"] == 0
               for video in manifest["candidate"]["videos"])
    assert all(video["caption_cache_hits"] == 2
               for video in manifest["candidate"]["videos"])
    for parent_video, candidate_video in zip(
            manifest["parent"]["videos"], manifest["candidate"]["videos"]):
        for parent_segment, candidate_segment in zip(
                parent_video["segments"], candidate_video["segments"]):
            assert parent_segment["cache_identity"] == \
                candidate_segment["cache_identity"]
            assert parent_segment["caption_result_path"] == \
                candidate_segment["caption_result_path"]

    counts = (len(router.calls), len(vlm.calls), len(qa.calls))
    resumed, _ = evaluate(evaluator, records, tmp_path, same_policy=True)
    assert resumed["resumed"]
    assert counts == (len(router.calls), len(vlm.calls), len(qa.calls))


def test_component_versions_separate_cache_when_composed_prompt_is_same(tmp_path):
    evaluator, records, _, _, _ = make_evaluator(tmp_path)
    bank, parent_router, candidate_bank, candidate_router, scaffold, contract = \
        policies()
    candidate_router = replace(
        candidate_router, configuration={"test_select_ids": []})
    result = evaluator.evaluate(
        confirmation_videos=records,
        incumbent_bank=bank, incumbent_router=parent_router,
        candidate_bank=candidate_bank, candidate_router=candidate_router,
        scaffold_policy=scaffold, scaffold_contract=contract,
        output_dir=str(tmp_path / "confirmation"))
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    parent = manifest["parent"]["videos"][0]["segments"][0]
    candidate = manifest["candidate"]["videos"][0]["segments"][0]
    assert parent["composed_prompt_hash"] == candidate["composed_prompt_hash"]
    assert parent["cache_identity"] != candidate["cache_identity"]
    assert parent["caption_result_path"] != candidate["caption_result_path"]


def test_composed_prompt_change_separates_cache_with_same_versions(tmp_path):
    evaluator, records, _, _, _ = make_evaluator(tmp_path)
    bank, parent_router, _, _, scaffold, contract = policies()
    candidate_router = replace(
        parent_router, configuration={"test_select_ids": ["pe_text"]})
    result = evaluator.evaluate(
        confirmation_videos=records,
        incumbent_bank=bank, incumbent_router=parent_router,
        candidate_bank=bank, candidate_router=candidate_router,
        scaffold_policy=scaffold, scaffold_contract=contract,
        output_dir=str(tmp_path / "confirmation"))
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    parent = manifest["parent"]["videos"][0]["segments"][0]
    candidate = manifest["candidate"]["videos"][0]["segments"][0]
    assert parent["cache_identity"]["bank_version"] == \
        candidate["cache_identity"]["bank_version"]
    assert parent["cache_identity"]["router_version"] == \
        candidate["cache_identity"]["router_version"]
    assert parent["composed_prompt_hash"] != candidate["composed_prompt_hash"]
    assert parent["caption_result_path"] != candidate["caption_result_path"]


def test_paired_transitions_and_rejection_criterion_are_persisted(tmp_path):
    evaluator, records, _, _, qa = make_evaluator(
        tmp_path, reject_candidate=True)
    result, manifest = evaluate(evaluator, records, tmp_path)
    transitions = [row["transition"] for row in result["qas"]]
    assert transitions.count("correct_to_wrong") == 1
    assert transitions.count("correct_to_correct") == 5
    assert not manifest["criterion"]["accepted"]
    assert not manifest["criterion"]["zero_correct_to_wrong"]
    assert len(qa.calls) == 12
    assert manifest["proposals_generated"] == 0
    assert manifest["interventions"] == 0
    assert manifest["feedback_calls"] == 0


@pytest.mark.parametrize("changed", ("sampling", "model", "backend", "decoding"))
def test_changed_live_runtime_fails_closed_before_caption_or_qa(
        tmp_path, monkeypatch, changed):
    evaluator, records, router, vlm, qa = make_evaluator(tmp_path)
    if changed == "sampling":
        monkeypatch.setattr(config, "SAMPLE_FPS", config.SAMPLE_FPS + 1)
    elif changed == "model":
        evaluator.builder.segment_captioner.caption_model_id = "changed-model"
    elif changed == "backend":
        evaluator.builder.segment_captioner.backend_id = "changed-backend"
    else:
        monkeypatch.setattr(config, "CAPTION_DECODING", {
            **config.CAPTION_DECODING, "max_tokens": 999})
    with pytest.raises(ConfirmationEvaluationConflictError):
        evaluate(evaluator, records, tmp_path)
    assert not router.calls and not vlm.calls and not qa.calls


def test_changed_history_configuration_conflicts_with_completed_artifact(tmp_path):
    evaluator, records, _, _, _ = make_evaluator(tmp_path)
    evaluate(evaluator, records, tmp_path)
    changed, records_again, router, vlm, qa = make_evaluator(
        tmp_path, history_block_seconds=120.0)
    assert records_again == records
    with pytest.raises(ConfirmationEvaluationConflictError):
        evaluate(changed, records_again, tmp_path)
    assert not router.calls and not vlm.calls and not qa.calls


def test_changed_sampled_frame_content_invalidates_completed_bundle(tmp_path):
    evaluator, records, router, vlm, qa = make_evaluator(tmp_path)
    result, _ = evaluate(evaluator, records, tmp_path)
    bundle = json.loads(Path(result["input_bundle_path"]).read_text())
    frame = Path(bundle["videos"][0]["segments"][0]["frames"][0]["path"])
    frame.write_bytes(b"changed-frame-content")
    counts = (len(router.calls), len(vlm.calls), len(qa.calls))
    with pytest.raises(ConfirmationEvaluationConflictError):
        evaluate(evaluator, records, tmp_path)
    assert counts == (len(router.calls), len(vlm.calls), len(qa.calls))


def test_real_orchestrator_factory_wires_concrete_confirmation(tmp_path):
    evaluator, records, _, _, _ = make_evaluator(tmp_path)
    runner = Checkpoint3EOrchestrator.with_real_confirmation(
        baseline_runner=object(), intervention_runner=object(),
        feedback_runner=object(), confirmation_kwargs={
            "sample_loader": evaluator.sample_loader,
            "history_aware_builder": evaluator.builder,
            "base_prompt_template": evaluator.base_prompt_template,
            "merge_prompt": evaluator.merge_prompt,
            "sample_source_identity": evaluator.sample_source_identity,
            "cache_root": evaluator.cache_root,
            "cache_manifest_path": evaluator.cache_manifest_path,
            "clip_index_fn": evaluator.clip_index_fn,
            "qa_fn": evaluator.qa_fn,
            "history_block_seconds": evaluator.history_block_seconds,
            "max_history_captions": evaluator.max_history_captions,
            "dvd_max_iterations": evaluator.dvd_max_iterations,
        })
    assert isinstance(
        runner.confirmation_evaluator, HistoryAwareDVDConfirmationEvaluator)
    assert len(records) == 2


@pytest.mark.parametrize("accepted", (True, False))
def test_concrete_confirmation_selects_complete_candidate_or_parent_pair(
        tmp_path, accepted):
    evaluator, confirmation_records, _, _, _ = make_evaluator(
        tmp_path, reject_candidate=not accepted)
    bank, parent_router, candidate_bank, candidate_router, scaffold, contract = \
        policies()
    roles = TrainRoleAssignment(
        evidence_videos=tuple(TrainVideoRecord(
            video_id=f"e{index}",
            provider_indices=(index * 3, index * 3 + 1, index * 3 + 2),
            question_ids=(f"e{index}q0", f"e{index}q1", f"e{index}q2"),
            previously_cached=True) for index in range(8)),
        confirmation_videos=confirmation_records,
        validation_video_ids=tuple(f"v{index}" for index in range(10)),
        test_video_ids=tuple(f"t{index}" for index in range(10)))
    parent = ConfirmedCheckpointState(
        checkpoint_id="confirmed-c0", coverage_cycle=0,
        component_versions=ComponentVersions(
            bank.bank_version, parent_router.router_version,
            scaffold.scaffold_version, contract.contract_version),
        confirmation_video_ids=tuple(
            record.video_id for record in confirmation_records),
        accepted_provisional_iteration_ids=(),
        confirmation_artifact_path="confirmed-c0.json")
    runner = Checkpoint3EOrchestrator(
        baseline_runner=object(), intervention_runner=object(),
        feedback_runner=object(), confirmation_evaluator=evaluator,
        require_real_models=False)
    decision, checkpoint_path, active_bank_path, active_router_path = \
        runner._run_confirmation(
            parent_confirmed=parent, roles=roles,
            incumbent_bank=bank, incumbent_router=parent_router,
            candidate_bank=candidate_bank, candidate_router=candidate_router,
            scaffold_policy=scaffold, scaffold_contract=contract,
            iteration_id="iteration-final", coverage_cycle=0,
            pending_provisional_iteration_ids=("iteration-final",),
            output_dir=str(tmp_path / "decision"),
            provisional_bank_path=str(tmp_path / "candidate-bank.json"),
            provisional_router_path=str(tmp_path / "candidate-router.json"))
    assert decision.accepted is accepted
    checkpoint = json.loads(Path(checkpoint_path).read_text())
    if accepted:
        assert decision.active_bank_version == candidate_bank.bank_version
        assert decision.active_router_version == candidate_router.router_version
        assert active_bank_path == str(tmp_path / "candidate-bank.json")
        assert active_router_path == str(tmp_path / "candidate-router.json")
        assert checkpoint["component_versions"]["bank_version"] == \
            candidate_bank.bank_version
        assert checkpoint["component_versions"]["router_version"] == \
            candidate_router.router_version
    else:
        assert decision.active_bank_version == bank.bank_version
        assert decision.active_router_version == parent_router.router_version
        assert checkpoint["checkpoint_id"] == parent.checkpoint_id
        assert json.loads(Path(active_bank_path).read_text())["bank_version"] == \
            bank.bank_version
        assert json.loads(Path(active_router_path).read_text())[
            "router_version"] == parent_router.router_version
