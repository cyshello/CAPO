import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from surrogate_rollout.cache.caption_cache import (
    build_history_aware_cache_key,
    key_as_dict,
    new_history_aware_cache_dir,
)
from surrogate_rollout.captioning.history_aware_baseline import build_history_snapshot
from surrogate_rollout.mixed_views.builder import MixedViewBuilder
from surrogate_rollout.optimization.property_intervention import (
    PropertyInterventionBatchRunner,
    PropertyInterventionConflictError,
)
from surrogate_rollout.optimization.property_proposal import CandidatePropertyProposal
from surrogate_rollout.prompt_routing.intervention_composer import (
    InterventionCompositionError,
    compose_forced_property,
)
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.policies.deterministic_scaffold import (
    DeterministicScaffoldApplier,
)
from surrogate_rollout.prompt_routing.router import write_routing_decisions
from surrogate_rollout.prompt_routing.scaffold_applier import write_composed_prompts
from surrogate_rollout.prompt_routing.schemas import (
    RoutingDecision,
    SegmentContext,
)
from surrogate_rollout.schemas import ReferenceSets


ROOT = Path(__file__).parents[1]
BASE_PROMPT = (
    "Caption TRANSCRIPT_PLACEHOLDER from CLIP_START_TIME to CLIP_END_TIME.\n"
    "ROUTED_INSTRUCTIONS_PLACEHOLDER"
)
SEGMENTS = ("0_10", "10_20", "20_30")


def components():
    data = json.loads((
        ROOT / "prompt_routing/fixtures/stage4_7_components.json").read_text())
    return (
        prompt_bank_from_json(data["prompt_bank"]),
        router_policy_from_json(data["router_policy"]),
        scaffold_policy_from_json(data["scaffold_policy"]),
        scaffold_contract_from_json(data["scaffold_contract"]),
    )


def proposal(candidate_id, text):
    return CandidatePropertyProposal(
        candidate_property_id=candidate_id, property_text=text,
        source_video_id="video-1", source_question_ids=("q1",),
        motivating_failure_types=("fixture_failure",),
        covered_by_existing_property_ids=(), proposal_rationale="fixture",
        proposer_policy_version="fixture_v1")


def incumbent_prompts(selected=("pe_default", "pe_temporal")):
    bank, router, scaffold, contract = components()
    output = []
    for segment_id in SEGMENTS:
        context = SegmentContext(video_id="video-1", segment_id=segment_id)
        decision = RoutingDecision(
            video_id="video-1", segment_id=segment_id,
            bank_version=bank.bank_version, router_version=router.router_version,
            selected_prompt_ids=tuple(selected), selection_order=tuple(selected))
        output.append(DeterministicScaffoldApplier(
            base_prompt_template=BASE_PROMPT).apply(
                context=context,
                selected_entries=tuple(bank.entry(item) for item in selected),
                routing_decision=decision, scaffold_policy=scaffold,
                scaffold_contract=contract))
    return tuple(output)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return str(path.resolve())


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return str(path.resolve())


def baseline_fixture(tmp_path):
    base = tmp_path / "baseline"
    prompts = incumbent_prompts()
    composed_path = base / "composed_prompts.jsonl"
    write_composed_prompts(prompts, str(composed_path))
    decisions_path = base / "routing_decisions.jsonl"
    write_routing_decisions((), str(decisions_path))

    preceding = []
    histories = []
    captions = {}
    cache_rows = []
    for index, segment_id in enumerate(SEGMENTS):
        history = build_history_snapshot(
            segment_id=segment_id, block_seconds=300,
            preceding=preceding, max_history_captions=30)
        histories.append(history)
        caption = f"baseline {segment_id}"
        captions[segment_id] = {"caption": caption}
        preceding.append({"segment_id": segment_id, "caption": caption})
        result_path = write_json(base / "baseline_cache" / segment_id / "caption.json", {
            "parsed": {
                "clip_description": caption,
                "subject_registry": {"segment": segment_id, "source": "baseline"},
            }
        })
        cache_rows.append({
            "segment_id": segment_id, "result_path": result_path,
            "cache_dir": str((base / "baseline_cache" / segment_id).resolve()),
        })
    captions["subject_registry"] = {"baseline": True}
    captions_path = write_json(base / "captions.json", captions)
    histories_path = write_jsonl(base / "frozen_histories.jsonl", histories)
    cache_keys_path = write_jsonl(base / "caption_cache_keys.jsonl", cache_rows)
    qas = [
        {"question_id": "q1", "provider_index": 1, "prediction": "B",
         "is_correct": False},
        {"question_id": "q2", "provider_index": 2, "prediction": "A",
         "is_correct": True},
        {"question_id": "q3", "provider_index": 3, "prediction": "B",
         "is_correct": False},
    ]
    qas_path = write_jsonl(base / "baseline_qas.jsonl", qas)
    routing_manifest_path = write_json(base / "routing_manifest.json", {
        "composed_prompts_path": str(composed_path.resolve()),
        "routing_decisions_path": str(decisions_path.resolve()),
        "caption_cache_keys_path": cache_keys_path,
    })
    bank, router, scaffold, contract = components()
    video_manifest = write_json(base / "video_complete.json", {
        "video_id": "video-1", "video_fingerprint": "baseline-fixture",
        "component_versions": {
            "bank_version": bank.bank_version,
            "router_version": router.router_version,
            "scaffold_version": scaffold.scaffold_version,
            "contract_version": contract.contract_version,
        },
        "captions_path": captions_path,
        "frozen_histories_path": histories_path,
        "baseline_qas_path": qas_path,
        "routing_manifest_path": routing_manifest_path,
    })
    return video_manifest, histories


def retrieval_fixture(tmp_path, item, selected, identity=None):
    return write_json(tmp_path / "retrieval" / item.candidate_property_id /
                      "retrieval.json", {
        "candidate_property_id": item.candidate_property_id,
        "source_video_id": item.source_video_id,
        "property_text": item.property_text,
        "s_sim": list(selected),
        "input_fingerprint": identity or f"retrieval-{item.candidate_property_id}",
    })


class FakeSegmentCaptioner:
    caption_model_id = "fixture-caption-model"
    backend_id = "fixture-caption-backend"

    def __init__(self, fail_candidates=()):
        self.fail_candidates = set(fail_candidates)
        self.calls = []

    def caption(self, **kwargs):
        candidate_id = kwargs["composed_prompt"].selected_prompt_ids[-1]
        self.calls.append({
            "candidate_id": candidate_id,
            "segment_id": kwargs["segment_id"],
            "history": json.loads(json.dumps(kwargs["history_snapshot"])),
            "selected_ids": kwargs["composed_prompt"].selected_prompt_ids,
            "intervention_identity": kwargs["intervention_identity_hash"],
        })
        if candidate_id in self.fail_candidates:
            raise RuntimeError("fixture selected-segment failure")
        cache_dir = Path(kwargs["cache_root"]) / candidate_id / kwargs["segment_id"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            parsed={
                "clip_description": (
                    f"candidate caption {candidate_id} {kwargs['segment_id']}"),
                "subject_registry": {
                    "segment": kwargs["segment_id"], "source": candidate_id},
            },
            cache_key={
                "intervention_identity_hash": kwargs["intervention_identity_hash"]},
            cache_dir=str(cache_dir), cache_hit=False, caption_seconds=0.01)


class FakeQA:
    OUTCOMES = {
        "cp_a": {"q1": True, "q2": False, "q3": False},
        "cp_b": {"q1": False, "q2": True, "q3": True},
    }

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        captions = json.loads(Path(kwargs["captions_path"]).read_text())
        candidate_id = next(
            value["caption"].split()[2] for key, value in captions.items()
            if key in SEGMENTS and "candidate caption" in value["caption"])
        question_id = kwargs["question_id"]
        correct = self.OUTCOMES[candidate_id][question_id]
        self.calls.append((candidate_id, question_id))
        run_dir = Path(kwargs["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "trajectory.jsonl").write_text("{}\n")
        (run_dir / "result.json").write_text("{}")
        return SimpleNamespace(
            score=1.0 if correct else 0.0,
            prediction="A" if correct else "B", ground_truth="A",
            references=ReferenceSets(consumed_segments={"0_10"}),
            errors=[], latency_seconds=0.01)


def clip_index(sample, video_id):
    assert video_id == "video-1"
    return [(segment_id, {
        "files": [f"/{segment_id}.jpg"], "transcript": f"tx {segment_id}"})
        for segment_id in SEGMENTS]


def sample_loader(index):
    return {
        "sample_id": f"sample-{index}", "video_path": "/video/video-1.mp4",
        "extra": {"videoID": "video-1", "subtitle_path": None},
        "question": f"question {index}", "answer": "A", "options": ["A", "B"],
    }


def test_force_add_preserves_order_allows_plus_one_and_fails_budget():
    bank, router, scaffold, contract = components()
    incumbent = incumbent_prompts(("pe_default", "pe_temporal", "pe_text"))[0]
    item = proposal("cp_extra", "Record fine-grained hand-object interactions.")
    composed = compose_forced_property(
        incumbent=incumbent, candidate=item, prompt_bank=bank,
        router_policy=router, scaffold_policy=scaffold,
        scaffold_contract=contract, base_prompt_template=BASE_PROMPT)
    assert composed.selected_prompt_ids == (
        "pe_default", "pe_temporal", "pe_text", "cp_extra")
    positions = [composed.prompt_text.index(text) for text in (
        bank.entry("pe_default").prompt_text,
        bank.entry("pe_temporal").prompt_text,
        bank.entry("pe_text").prompt_text,
        item.property_text,
    )]
    assert positions == sorted(positions)
    with pytest.raises(InterventionCompositionError, match="max_prompt_tokens"):
        compose_forced_property(
            incumbent=incumbent, candidate=item, prompt_bank=bank,
            router_policy=router, scaffold_policy=scaffold,
            scaffold_contract=replace(contract, max_prompt_tokens=5),
            base_prompt_template=BASE_PROMPT)


def test_intervention_cache_identity_preserves_baseline_path(tmp_path):
    values = {
        "video_id": "video-1", "video_path": "/missing/video.mp4",
        "caption_prompt": "prompt", "merge_prompt": "merge",
        "subtitle_path": None, "segment_id": "0_10",
        "history_hash": "h" * 64, "composed_prompt_hash": "c" * 64,
        "bank_version": "bank-v1", "router_version": "router-v1",
        "scaffold_version": "scaffold-v1", "contract_version": "contract-v1",
        "backend_id": "fixture-backend", "history_config_hash": "g" * 64,
    }
    baseline_key = build_history_aware_cache_key(**values)
    candidate_key = build_history_aware_cache_key(
        **values, intervention_identity_hash="i" * 64)
    baseline_path = new_history_aware_cache_dir(
        baseline_key, str(tmp_path / "cache"))
    candidate_path = new_history_aware_cache_dir(
        candidate_key, str(tmp_path / "cache"))
    assert "intervention_identity_hash" not in key_as_dict(baseline_key)
    assert key_as_dict(candidate_key)["intervention_identity_hash"] == "i" * 64
    assert f"_i{'i' * 12}_" in candidate_path
    assert candidate_path != baseline_path


def test_independent_selected_only_interventions_history_mixed_view_and_resume(tmp_path):
    baseline_path, histories = baseline_fixture(tmp_path)
    a = proposal("cp_a", "Track fine-grained actions precisely.")
    b = proposal("cp_b", "Retain brief interaction outcomes precisely.")
    retrievals = (
        retrieval_fixture(tmp_path, a, ("0_10", "10_20")),
        retrieval_fixture(tmp_path, b, ("20_30",)),
    )
    captioner = FakeSegmentCaptioner()
    qa = FakeQA()
    registry_merge_calls = []

    def merge_registries(values):
        registry_merge_calls.append(values)
        return {"ordered": values}

    runner = PropertyInterventionBatchRunner(
        segment_captioner=captioner, qa_fn=qa, clip_index_fn=clip_index,
        mixed_view_builder=MixedViewBuilder(merge_fn=merge_registries),
        caption_cache_root=str(tmp_path / "candidate_cache"),
        caption_cache_manifest_path=str(tmp_path / "cache_manifest.jsonl"))
    bank, router, scaffold, contract = components()
    result = runner.run(
        iteration_id="iteration-1", baseline_video_manifest_path=baseline_path,
        proposals=(a, b), retrieval_artifact_paths=retrievals,
        sample_loader=sample_loader, prompt_bank=bank, router_policy=router,
        scaffold_policy=scaffold, scaffold_contract=contract,
        base_prompt_template=BASE_PROMPT, merge_prompt="merge",
        output_dir=str(tmp_path / "interventions"))

    assert [item.status for item in result.results] == ["completed", "completed"]
    assert [(call["candidate_id"], call["segment_id"]) for call in captioner.calls] == [
        ("cp_a", "0_10"), ("cp_a", "10_20"), ("cp_b", "20_30")]
    assert all(call["selected_ids"][-1] == call["candidate_id"]
               for call in captioner.calls)
    second_history = next(call["history"] for call in captioner.calls
                          if call["candidate_id"] == "cp_a"
                          and call["segment_id"] == "10_20")
    assert second_history == json.loads(json.dumps(histories[1]))
    assert second_history["history"][0]["caption"] == "baseline 0_10"
    assert len({call["intervention_identity"] for call in captioner.calls}) == 3

    a_captions = json.loads(Path(result.results[0].mixed_captions_path).read_text())
    assert a_captions["0_10"]["caption"].startswith("candidate caption cp_a")
    assert a_captions["10_20"]["caption"].startswith("candidate caption cp_a")
    assert a_captions["20_30"]["caption"] == "baseline 20_30"
    registry_sources = [item["source"]
                        for item in a_captions["subject_registry"]["ordered"]]
    assert registry_sources == ["cp_a", "cp_a", "baseline"]
    assert len(qa.calls) == 6
    assert len(registry_merge_calls) == 2

    transitions_a = json.loads(Path(result.results[0].transitions_path).read_text())
    assert transitions_a["transition_counts"] == {
        "correct_to_correct": 0, "correct_to_wrong": 1,
        "wrong_to_correct": 1, "wrong_to_wrong": 1}
    transitions_b = json.loads(Path(result.results[1].transitions_path).read_text())
    assert transitions_b["transition_counts"] == {
        "correct_to_correct": 1, "correct_to_wrong": 0,
        "wrong_to_correct": 1, "wrong_to_wrong": 1}

    caption_calls = len(captioner.calls)
    qa_calls = len(qa.calls)
    resumed = runner.run(
        iteration_id="iteration-1", baseline_video_manifest_path=baseline_path,
        proposals=(a, b), retrieval_artifact_paths=retrievals,
        sample_loader=sample_loader, prompt_bank=bank, router_policy=router,
        scaffold_policy=scaffold, scaffold_contract=contract,
        base_prompt_template=BASE_PROMPT, merge_prompt="merge",
        output_dir=str(tmp_path / "interventions"))
    assert all(item.resumed for item in resumed.results)
    assert len(captioner.calls) == caption_calls
    assert len(qa.calls) == qa_calls
    assert len(registry_merge_calls) == 2


def test_changed_property_or_parent_baseline_fails_closed(tmp_path):
    baseline_path, _ = baseline_fixture(tmp_path)
    item = proposal("cp_a", "Track fine-grained actions precisely.")
    retrieval_path = retrieval_fixture(tmp_path, item, ("0_10",))
    runner = PropertyInterventionBatchRunner(
        segment_captioner=FakeSegmentCaptioner(), qa_fn=FakeQA(),
        clip_index_fn=clip_index,
        mixed_view_builder=MixedViewBuilder(merge_fn=lambda values: values),
        caption_cache_root=str(tmp_path / "cache"),
        caption_cache_manifest_path=str(tmp_path / "cache_manifest.jsonl"))
    bank, router, scaffold, contract = components()
    arguments = {
        "iteration_id": "iteration-1",
        "baseline_video_manifest_path": baseline_path,
        "sample_loader": sample_loader,
        "prompt_bank": bank,
        "router_policy": router,
        "scaffold_policy": scaffold,
        "scaffold_contract": contract,
        "base_prompt_template": BASE_PROMPT,
        "merge_prompt": "merge",
        "output_dir": str(tmp_path / "interventions"),
    }
    runner.run(
        **arguments, proposals=(item,),
        retrieval_artifact_paths=(retrieval_path,))

    baseline = json.loads(Path(baseline_path).read_text())
    captions_path = Path(baseline["captions_path"])
    original_captions = captions_path.read_text()
    captions_path.write_text(original_captions + "\n")
    with pytest.raises(PropertyInterventionConflictError):
        runner.run(
            **arguments, proposals=(item,),
            retrieval_artifact_paths=(retrieval_path,))
    captions_path.write_text(original_captions)

    changed = proposal("cp_a", "Record fine-grained hand-object interactions.")
    changed_retrieval = retrieval_fixture(tmp_path, changed, ("0_10",))
    with pytest.raises(PropertyInterventionConflictError):
        runner.run(
            **arguments, proposals=(changed,),
            retrieval_artifact_paths=(changed_retrieval,))


def test_selected_segment_failure_isolated_without_incumbent_fallback(tmp_path):
    baseline_path, _ = baseline_fixture(tmp_path)
    good = proposal("cp_a", "Track fine-grained actions precisely.")
    failed = proposal("cp_fail", "Record subtle object state changes.")
    retrievals = (
        retrieval_fixture(tmp_path, good, ("0_10",)),
        retrieval_fixture(tmp_path, failed, ("10_20",)),
    )
    captioner = FakeSegmentCaptioner(fail_candidates={"cp_fail"})
    qa = FakeQA()
    runner = PropertyInterventionBatchRunner(
        segment_captioner=captioner, qa_fn=qa, clip_index_fn=clip_index,
        mixed_view_builder=MixedViewBuilder(merge_fn=lambda values: values),
        caption_cache_root=str(tmp_path / "cache"),
        caption_cache_manifest_path=str(tmp_path / "manifest.jsonl"))
    bank, router, scaffold, contract = components()
    result = runner.run(
        iteration_id="iteration-1", baseline_video_manifest_path=baseline_path,
        proposals=(good, failed), retrieval_artifact_paths=retrievals,
        sample_loader=sample_loader, prompt_bank=bank, router_policy=router,
        scaffold_policy=scaffold, scaffold_contract=contract,
        base_prompt_template=BASE_PROMPT, merge_prompt="merge",
        output_dir=str(tmp_path / "interventions"))
    assert result.results[0].status == "completed"
    assert result.results[1].status == "failed"
    assert result.results[1].failure_type == "selected_segment_caption_failed"
    assert result.results[1].mixed_captions_path is None
    assert qa.calls == [("cp_a", "q1"), ("cp_a", "q2"), ("cp_a", "q3")]
    assert Path(result.results[1].result_path).exists()

    changed = retrieval_fixture(
        tmp_path, good, ("0_10",), identity="changed-retrieval")
    with pytest.raises(PropertyInterventionConflictError):
        runner.run(
            iteration_id="iteration-1",
            baseline_video_manifest_path=baseline_path,
            proposals=(good, failed),
            retrieval_artifact_paths=(changed, retrievals[1]),
            sample_loader=sample_loader, prompt_bank=bank, router_policy=router,
            scaffold_policy=scaffold, scaffold_contract=contract,
            base_prompt_template=BASE_PROMPT, merge_prompt="merge",
            output_dir=str(tmp_path / "interventions"))
