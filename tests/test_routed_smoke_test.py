"""Focused Stage 4.7 routed evaluator smoke-runner tests."""

import json
import os
from pathlib import Path

import pytest

from surrogate_rollout.cache.caption_cache import prompt_hash as cache_prompt_hash
from surrogate_rollout.captioning.candidate_captions import CandidateCaptionSet
from surrogate_rollout.prompt_routing.routed_caption_view import (
    RoutedCaptionViewBuilder,
)
from surrogate_rollout.prompt_routing.routed_smoke_test import (
    FALLBACK_CONDITION,
    ROUTED_CONDITION,
    RoutedEvaluatorSmokeRunner,
    RoutedSmokeTestError,
    load_smoke_components,
)
from surrogate_rollout.schemas import DVDRunResult, ReferenceSets

FIXTURE = Path(__file__).parents[1] / "prompt_routing" / "fixtures" / \
    "stage4_7_components.json"
VIDEO_ID = "small_video"
CLIPS = ("0_10", "10_20")
BASE_PROMPT = (
    "Clip CLIP_START_TIME to CLIP_END_TIME.\n"
    "Transcript: TRANSCRIPT_PLACEHOLDER\n"
)
MERGE_PROMPT = "synthetic merge prompt"


class FakeCaptioner:
    def __init__(self):
        self.calls = []

    def __call__(self, *, sample, candidate_prompt, merge_prompt, clip_ids,
                 cache_root):
        self.calls.append((candidate_prompt, set(clip_ids), cache_root))
        strong_hash = cache_prompt_hash(candidate_prompt, merge_prompt)
        return CandidateCaptionSet(
            video_id=VIDEO_ID,
            prompt_hash=strong_hash,
            ckpt_dir=f"/cache/{strong_hash[:8]}/ckpt",
            clips=list(CLIPS),
            parsed={key: {
                "clip_description": f"{strong_hash[:8]} caption {key}",
                "subject_registry": {"segment": key},
            } for key in clip_ids},
            caption_call_count=len(clip_ids),
            cache_hits=0,
            caption_seconds=0.1,
            cache_key={
                "video_id": VIDEO_ID,
                "segment_id": "*",
                "prompt_hash": strong_hash,
                "caption_model_id": "fake",
                "decoding_hash": "d" * 64,
                "source_hash": "s" * 64,
            },
        )


class Harness:
    def __init__(self):
        self.backend_calls = []
        self.qa_calls = []
        self.captioner = FakeCaptioner()

    @staticmethod
    def clip_index(sample, video_id):
        return [(key, {}) for key in CLIPS]

    def backend(self, gpu, *, preload_captioner):
        self.backend_calls.append((gpu, preload_captioner))

    def qa(self, *, captions_path, sample, run_dir, question_id,
           database_path, max_iterations, gpu):
        self.qa_calls.append({
            "captions_path": captions_path,
            "question_id": question_id,
            "database_path": database_path,
        })
        os.makedirs(run_dir, exist_ok=True)
        condition = ROUTED_CONDITION if ROUTED_CONDITION in captions_path \
            else FALLBACK_CONDITION
        prediction = "B" if condition == ROUTED_CONDITION else "A"
        score = 1.0 if prediction == sample["answer"] else 0.0
        for name, content in {
            "trajectory.jsonl": "{}\n",
            "tool_events.jsonl": "",
            "llm_calls.jsonl": "",
            "references.json": "{}\n",
            "result.json": json.dumps({"prediction": prediction}) + "\n",
        }.items():
            Path(run_dir, name).write_text(content)
        return DVDRunResult(
            question_id=question_id,
            video_id=VIDEO_ID,
            prediction=prediction,
            parsed_answer=prediction,
            ground_truth=sample["answer"],
            score=score,
            trajectory=[],
            references=ReferenceSets(),
            total_segments=len(CLIPS),
            token_usage={"total": 10 if condition == ROUTED_CONDITION else 8},
            latency_seconds=0.2,
            caption_cache_tag="routed",
            captions_path=captions_path,
            database_path=database_path,
            errors=[],
        )

    def runner(self):
        view_builder = RoutedCaptionViewBuilder(
            caption_fn=self.captioner,
            clip_index_fn=self.clip_index,
            merge_fn=lambda registries: {
                "segments": [registry["segment"] for registry in registries]},
        )
        return RoutedEvaluatorSmokeRunner(
            backend_fn=self.backend,
            qa_fn=self.qa,
            clip_index_fn=self.clip_index,
            caption_view_builder=view_builder,
        )


def sample(video_id=VIDEO_ID, answer="B"):
    return {
        "sample_id": video_id,
        "video_path": "unused.mp4",
        "question": "Synthetic question?",
        "options": ["A. no", "B. yes"],
        "answer": answer,
    }


def run(tmp_path, qa_samples=None):
    harness = Harness()
    result = harness.runner().run(
        qa_samples=qa_samples or (("dataset/short/0", sample()),),
        base_prompt_template=BASE_PROMPT,
        merge_prompt=MERGE_PROMPT,
        work_root=str(tmp_path / "smoke"),
        gpu="7",
        candidate_cache_root=str(tmp_path / "caption_cache"),
        source_revision="abc123",
        **load_smoke_components(str(FIXTURE)),
    )
    return result, harness


def test_two_conditions_use_one_frozen_component_set_and_shared_qa_path(tmp_path):
    result, harness = run(tmp_path)
    assert harness.backend_calls == [("7", True)]
    assert len(harness.qa_calls) == 2
    comparison = json.loads(Path(result.comparison_path).read_text())
    assert comparison["frozen_component_versions"] == {
        "bank": "bank_v0002", "router": "router_v0002",
        "scaffold": "scaffold_v0002", "contract": "contract_v0002"}
    row = comparison["comparisons"][0]
    assert row[FALLBACK_CONDITION]["prediction"] == "A"
    assert row[ROUTED_CONDITION]["prediction"] == "B"
    assert row["score_delta"] == 1.0


def test_fallback_and_multi_entry_routing_composition_are_recorded(tmp_path):
    result, _ = run(tmp_path)
    conditions = {condition.condition: condition for condition in result.conditions}
    fallback_dir = Path(conditions[FALLBACK_CONDITION].routing_composition_dir)
    routed_dir = Path(conditions[ROUTED_CONDITION].routing_composition_dir)
    fallback_decision = json.loads(
        (fallback_dir / "routing/decisions.jsonl").read_text().splitlines()[0])
    routed_decision = json.loads(
        (routed_dir / "routing/decisions.jsonl").read_text().splitlines()[0])
    assert fallback_decision["used_fallback"] is True
    assert fallback_decision["selected_prompt_ids"] == ["pe_default"]
    assert routed_decision["used_fallback"] is False
    assert routed_decision["selected_prompt_ids"] == [
        "pe_default", "pe_temporal", "pe_text"]
    assert (routed_dir / "composition/traces.jsonl").is_file()
    assert (routed_dir / "composition/composed_prompts.jsonl").is_file()


def test_cache_hash_prediction_score_cost_and_timing_artifacts(tmp_path):
    result, _ = run(tmp_path)
    for condition in result.conditions:
        summary = json.loads(Path(condition.summary_path).read_text())
        assert summary["composed_prompt_hashes"]
        assert len(summary["caption_cache_keys"][0]["prompt_hash"]) == 64
        assert summary["qa_results"][0]["prediction"] in ("A", "B")
        assert summary["qa_results"][0]["score"] in (0.0, 1.0)
        assert summary["cost"]["caption_call_count"] == len(CLIPS)
        assert summary["cost"]["token_usage_by_question"]
        assert summary["timing"]["caption_seconds"] == 0.1
        assert summary["timing"]["reasoning_seconds"] == 0.2


def test_manifest_is_complete_and_no_later_stage_artifacts(tmp_path):
    result, _ = run(tmp_path)
    manifest = json.loads(Path(result.manifest_path).read_text())
    actual = {
        str(path.relative_to(result.work_root))
        for path in Path(result.work_root).rglob("*")
        if path.is_file() and path != Path(result.manifest_path)
    }
    assert set(manifest["artifacts"]) == actual
    assert manifest["later_stages_run"] == []
    forbidden = {"evidence", "feedback", "proposals", "optimization"}
    assert not forbidden & {part for path in actual for part in Path(path).parts}


@pytest.mark.parametrize("count", [0, 4])
def test_requires_one_to_three_qas(tmp_path, count):
    harness = Harness()
    qas = tuple((f"q/{i}", sample()) for i in range(count))
    with pytest.raises(RoutedSmokeTestError, match="one to three"):
        harness.runner().run(
            qa_samples=qas,
            base_prompt_template=BASE_PROMPT,
            merge_prompt=MERGE_PROMPT,
            work_root=str(tmp_path / "smoke"),
            **load_smoke_components(str(FIXTURE)),
        )


def test_requires_all_qas_from_one_video(tmp_path):
    harness = Harness()
    with pytest.raises(RoutedSmokeTestError, match="one video"):
        harness.runner().run(
            qa_samples=(("q/0", sample()), ("q/1", sample("other_video"))),
            base_prompt_template=BASE_PROMPT,
            merge_prompt=MERGE_PROMPT,
            work_root=str(tmp_path / "smoke"),
            **load_smoke_components(str(FIXTURE)),
        )
