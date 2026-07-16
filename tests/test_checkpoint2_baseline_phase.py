import json
from pathlib import Path
from types import SimpleNamespace

from surrogate_rollout.optimization.baseline_phase import BaselinePhaseRunner
from surrogate_rollout.optimization.property_proposal import (
    CandidatePropertyProposal,
)
from surrogate_rollout.optimization.train_roles import derive_train_roles, initial_coverage
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.schemas import Phase4Config, PostInterventionMode
from surrogate_rollout.schemas import DVDRunResult, ReferenceSets


ROOT = Path(__file__).parents[1]


class FakeCaptionViewBuilder:
    def __init__(self):
        self.calls = []

    def build(self, *, sample, segment_composed_prompt_map, work_root, merge_prompt):
        video_id = sample["extra"]["videoID"]
        self.calls.append(video_id)
        directory = Path(work_root) / video_id / "view"
        directory.mkdir(parents=True, exist_ok=True)
        segment_ids = tuple(segment_composed_prompt_map)
        captions_path = directory / "captions.json"
        captions_path.write_text(json.dumps({
            segment_id: {"caption": f"caption {video_id} {segment_id}"}
            for segment_id in segment_ids
        }))
        routed_path = directory / "routed_view.json"
        routed_path.write_text("{}")
        return SimpleNamespace(
            captions_path=str(captions_path), database_path=str(directory / "db.json"),
            routed_view_path=str(routed_path), segment_ids=segment_ids)


class FixtureProposalPolicy:
    policy_version = "fixture_property_proposer_v1"

    def __init__(self):
        self.calls = []

    def propose(self, context):
        self.calls.append(context.video_id)
        qa_ids = tuple(row["question_id"] for row in context.baseline_qa_results)
        count = len(self.calls) - 1
        return tuple(CandidatePropertyProposal(
            property_id=f"property_{context.video_id}_{index}",
            source_video_id=context.video_id,
            source_qa_ids=(qa_ids[index % len(qa_ids)],),
            instruction=f"record property {index}",
            failure_evidence=f"fixture failure {index}",
            coverage_assessment="missing_from_codebook",
            proposer_policy_version=self.policy_version,
        ) for index in range(count))


class FixturePropertyRetrievalRunner:
    policy_version = "fixture_property_retrieval_v1"
    configuration_identity = {"top_k": 5, "method": "fixture"}

    def __init__(self):
        self.calls = []

    def run(self, *, sample, proposals, output_dir):
        video_id = sample["extra"]["videoID"]
        self.calls.append((video_id, tuple(item.property_id for item in proposals)))
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for item in proposals:
            path = directory / item.property_id / "retrieval.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
            paths.append(str(path.resolve()))
        manifest = directory / "manifest.json"
        manifest.write_text("{}")
        return SimpleNamespace(
            retrieval_artifact_paths=tuple(paths),
            manifest_path=str(manifest.resolve()))


def components():
    data = json.loads((ROOT / "prompt_routing/fixtures/stage4_7_components.json").read_text())
    return (
        prompt_bank_from_json(data["prompt_bank"]),
        router_policy_from_json(data["router_policy"]),
        scaffold_policy_from_json(data["scaffold_policy"]),
        scaffold_contract_from_json(data["scaffold_contract"]),
    )


def test_baseline_materializes_three_videos_once_runs_nine_qas_and_resumes(tmp_path):
    manifest = json.loads((ROOT / "split_manifest.json").read_text())
    roles = derive_train_roles(manifest)
    samples = {}
    for video in roles.evidence_videos[:3]:
        for index, question_id in zip(video.provider_indices, video.question_ids):
            samples[index] = {
                "sample_id": f"sample-{index}",
                "question": f"question {index}", "answer": "A",
                "options": ["A", "B"], "video_path": f"/video/{video.video_id}.mp4",
                "extra": {"videoID": video.video_id, "subtitle_path": None},
                "question_id": question_id,
            }

    routing_calls = []

    def routing_fn(**kwargs):
        routing_calls.append(kwargs["contexts"][0].video_id)
        directory = Path(kwargs["output_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        path.write_text("{}")
        prompts = tuple(SimpleNamespace(
            segment_id=context.segment_id,
            prompt_text="prompt TRANSCRIPT_PLACEHOLDER CLIP_START_TIME CLIP_END_TIME")
            for context in kwargs["contexts"])
        return SimpleNamespace(composed_prompts=prompts, manifest_path=str(path))

    qa_calls = []

    def qa_fn(**kwargs):
        qa_calls.append(kwargs["question_id"])
        return DVDRunResult(
            question_id=kwargs["question_id"],
            video_id=kwargs["sample"]["extra"]["videoID"],
            prediction="A", parsed_answer="A", ground_truth="A", score=1.0,
            trajectory=[], references=ReferenceSets(consumed_segments={"0_10"}),
            total_segments=2, token_usage={}, latency_seconds=0.01,
            caption_cache_tag="fixture", captions_path=kwargs["captions_path"],
            database_path=kwargs["database_path"], errors=[])

    def clip_index_fn(sample, video_id):
        return [("0_10", {"files": [tmp_path / f"{video_id}-0.jpg"]}),
                ("10_20", {"files": [tmp_path / f"{video_id}-1.jpg"]})]

    caption_builder = FakeCaptionViewBuilder()
    proposer = FixtureProposalPolicy()
    retrieval_runner = FixturePropertyRetrievalRunner()
    runner = BaselinePhaseRunner(
        routing_fn=routing_fn, caption_view_builder=caption_builder,
        qa_fn=qa_fn, clip_index_fn=clip_index_fn, proposal_policy=proposer,
        property_retrieval_runner=retrieval_runner)
    bank, router, scaffold, contract = components()
    output = tmp_path / "iteration"
    result = runner.run(
        roles=roles, coverage_state=initial_coverage(roles),
        sample_loader=samples.__getitem__, prompt_bank=bank,
        router_policy=router, scaffold_policy=scaffold,
        scaffold_contract=contract, phase4_config=Phase4Config(),
        base_prompt_template="base", merge_prompt="merge",
        output_dir=str(output), parent_confirmed_checkpoint_id="confirmed_0",
        history_block_size=2, source_revision="fixture")

    assert len(set(result.selected_video_ids)) == 3
    assert result.baseline_qa_count == 9
    assert routing_calls == list(result.selected_video_ids)
    assert caption_builder.calls == list(result.selected_video_ids)
    assert len(qa_calls) == 9
    assert proposer.calls == list(result.selected_video_ids)
    assert [len(item[1]) for item in retrieval_runner.calls] == [0, 1, 2]
    assert len(result.property_retrieval_paths) == 3
    assert [len(json.loads(Path(path).read_text())["proposals"])
            for path in result.property_proposal_paths] == [0, 1, 2]
    for path in result.video_manifest_paths:
        video = json.loads(Path(path).read_text())
        assert video["caption_materializations"] == 1
        histories = Path(video["frozen_histories_path"]).read_text().splitlines()
        assert len(histories) == 2
        assert json.loads(histories[0])["preceding_segment_ids"] == []
        assert json.loads(histories[1])["preceding_segment_ids"] == ["0_10"]
    summary = json.loads(Path(result.manifest_path).read_text())
    assert summary["calls"] == {
        "confirmation_evaluations": 0,
        "dvd_reasoning": 9,
        "interventional_feedback_models": 0,
        "property_proposal_policy": 3,
        "real_vlm_router": 0,
        "video_caption_materializations": 3,
    }

    resumed = runner.run(
        roles=roles, coverage_state=initial_coverage(roles),
        sample_loader=samples.__getitem__, prompt_bank=bank,
        router_policy=router, scaffold_policy=scaffold,
        scaffold_contract=contract, phase4_config=Phase4Config(),
        base_prompt_template="base", merge_prompt="merge",
        output_dir=str(output), parent_confirmed_checkpoint_id="confirmed_0",
        history_block_size=2, source_revision="fixture")
    assert resumed.resumed
    assert len(routing_calls) == len(caption_builder.calls) == 3
    assert len(qa_calls) == 9
    assert len(retrieval_runner.calls) == 3

    Path(result.manifest_path).unlink()
    Path(result.video_manifest_paths[-1]).unlink()
    partial = runner.run(
        roles=roles, coverage_state=initial_coverage(roles),
        sample_loader=samples.__getitem__, prompt_bank=bank,
        router_policy=router, scaffold_policy=scaffold,
        scaffold_contract=contract, phase4_config=Phase4Config(),
        base_prompt_template="base", merge_prompt="merge",
        output_dir=str(output), parent_confirmed_checkpoint_id="confirmed_0",
        history_block_size=2, source_revision="fixture")
    assert not partial.resumed
    assert len(routing_calls) == len(caption_builder.calls) == 4
    assert len(qa_calls) == 12
    assert len(retrieval_runner.calls) == 4
    assert Path(partial.manifest_path).exists()

    bounded_captioner = FakeCaptionViewBuilder()
    bounded_proposer = FixtureProposalPolicy()
    bounded_retrieval = FixturePropertyRetrievalRunner()
    bounded_runner = BaselinePhaseRunner(
        routing_fn=routing_fn, caption_view_builder=bounded_captioner,
        qa_fn=qa_fn, clip_index_fn=clip_index_fn,
        proposal_policy=bounded_proposer,
        property_retrieval_runner=bounded_retrieval)
    bounded_output = tmp_path / "bounded"
    bounded_video = roles.evidence_videos[0].video_id
    bounded = bounded_runner.run(
        roles=roles, coverage_state=initial_coverage(roles),
        sample_loader=samples.__getitem__, prompt_bank=bank,
        router_policy=router, scaffold_policy=scaffold,
        scaffold_contract=contract,
        phase4_config=Phase4Config(
            post_intervention_mode=PostInterventionMode.QA_ONLY),
        base_prompt_template="base", merge_prompt="merge",
        output_dir=str(bounded_output),
        parent_confirmed_checkpoint_id="bounded-input",
        bounded_smoke_video_id=bounded_video,
        history_block_size=2, source_revision="fixture")
    assert bounded.selected_video_ids == (bounded_video,)
    assert bounded.baseline_qa_count == 3
    resumed_later_mode = bounded_runner.run(
        roles=roles, coverage_state=initial_coverage(roles),
        sample_loader=samples.__getitem__, prompt_bank=bank,
        router_policy=router, scaffold_policy=scaffold,
        scaffold_contract=contract,
        phase4_config=Phase4Config(
            post_intervention_mode=PostInterventionMode.FEEDBACK_ONLY),
        base_prompt_template="base", merge_prompt="merge",
        output_dir=str(bounded_output),
        parent_confirmed_checkpoint_id="bounded-input",
        bounded_smoke_video_id=bounded_video,
        history_block_size=2, source_revision="fixture")
    assert resumed_later_mode.resumed is True
    assert bounded_captioner.calls == [bounded_video]
    assert len(bounded_proposer.calls) == len(bounded_retrieval.calls) == 1
