"""Checkpoint G real DVD meta-prompt confirmation adapter tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from surrogate_rollout.optimization.dvd_meta_prompt_confirmation import (
    DVDMetaPromptConfirmationEvaluator,
    DVDMetaPromptConfirmationVideo,
    LazyDVDMetaPromptConfirmationEvaluator,
)
from surrogate_rollout.optimization.confirmation_evaluator import (
    ConfirmationEvaluationConflictError,
)
from surrogate_rollout.optimization.checkpoint_g_factory import (
    CheckpointGConfigurationError,
    _load_configuration,
    _real_identity,
)
from surrogate_rollout.optimization.prompt_delta_iteration import (
    MetaPromptConfirmationCase,
    build_confirmation_request,
)
from surrogate_rollout.optimization.schemas import MetaPromptVersion
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)


class _VLM:
    pass


@dataclass
class _QAResult:
    prediction: str
    parsed_answer: str
    ground_truth: str
    score: float
    errors: tuple[str, ...] = ()
    latency_seconds: float = 0.1


class _Runtime:
    def __init__(self, root: Path):
        self.root = root
        self.builder = SimpleNamespace(
            router=SimpleNamespace(vlm=_VLM()), free_form_generator=None)
        self.dvd_max_iterations = 3
        self.gpu = "0"
        self.configuration_identity = {"caption_model": "caption-real"}
        self.generators = []
        self.qa_calls = []
        self.qa_fn = self._qa

    def _qa(self, **kwargs):
        self.qa_calls.append(kwargs)
        candidate = "/candidate/" in kwargs["run_dir"]
        qa = kwargs["question_id"]
        score = 1.0 if candidate or qa.endswith("1") else 0.0
        return _QAResult("A" if score else "B", "A", "A", score)

    def _materialize_bundle(self, *, confirmation_videos, scaffold, contract,
                            output_dir):
        path = Path(output_dir) / "input_bundle.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        videos = []
        for video in confirmation_videos:
            videos.append({
                "video_id": video.video_id,
                "qas": [{"question_id": q, "provider_index": i,
                         "sample": {"extra": {"videoID": video.video_id}}}
                        for q, i in zip(video.question_ids,
                                        video.provider_indices)],
            })
        value = {"bundle_hash": "bundle-hash", "videos": videos}
        path.write_text(json.dumps(value), encoding="utf-8")
        return str(path), value

    def _caption_state(self, *, state_name, bundle, bank, router, scaffold,
                       contract, output_dir, free_form_generator):
        self.generators.append((state_name, free_form_generator.meta_prompt_id,
                                free_form_generator.template))
        videos = []
        for video in bundle["videos"]:
            root = Path(output_dir) / state_name / video["video_id"]
            root.mkdir(parents=True, exist_ok=True)
            captions = root / "captions.json"
            captions.write_text(json.dumps({"caption": state_name}),
                                encoding="utf-8")
            routing = root / "routing.json"
            routing.write_text("{}", encoding="utf-8")
            videos.append({
                "video_id": video["video_id"],
                "captions_path": str(captions),
                "captions_hash": state_name + "-captions",
                "database_path": str(root / "database.json"),
                "routing_manifest_path": str(routing),
                "caption_view_hash": state_name + "-view",
                "caption_calls": 2, "caption_cache_hits": 0,
                "segments": [{"cache_identity_hash": state_name + "-cache"}],
            })
        artifact = Path(output_dir) / state_name / "caption_state.json"
        artifact.write_text("{}", encoding="utf-8")
        return {"videos": videos, "artifact_path": str(artifact)}

    @staticmethod
    def _validate_paired_states(parent, candidate, bundle):
        assert [v["video_id"] for v in parent["videos"]] == \
               [v["video_id"] for v in candidate["videos"]]


def _components():
    value = json.loads(Path(
        "prompt_routing/fixtures/stage4_7_components.json").read_text())
    return (prompt_bank_from_json(value["prompt_bank"]),
            router_policy_from_json(value["router_policy"]),
            scaffold_policy_from_json(value["scaffold_policy"]),
            scaffold_contract_from_json(value["scaffold_contract"]))


def _versions():
    return (
        MetaPromptVersion("parent", None, "PARENT EXACT", "t", "parent"),
        MetaPromptVersion("candidate", "parent", "CANDIDATE EXACT", "t",
                          "provisional"),
    )


def _adapter(tmp_path):
    runtime = _Runtime(tmp_path)
    bank, router, scaffold, contract = _components()
    videos = (
        DVDMetaPromptConfirmationVideo("v1", (1, 2, 3), ("q1", "q2", "q3")),
        DVDMetaPromptConfirmationVideo("v2", (4, 5, 6), ("q4", "q5", "q6")),
    )
    adapter = DVDMetaPromptConfirmationEvaluator(
        evaluator=runtime, confirmation_videos=videos,
        compatibility_bank=bank, compatibility_router=router,
        scaffold_policy=scaffold, scaffold_contract=contract,
        prompt_generator_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        prompt_generator_backend_id="local-vllm",
        prompt_generator_max_tokens=192,
        model_identity="paired-runtime-v1",
        decoding_settings={"temperature": 0.0},
        cache_reset_identity="clean-paired-v1",
        evaluation_pipeline_identity="dvd-paired-v1")
    cases = tuple(MetaPromptConfirmationCase(
        f"case-{q}", video.video_id, q, f"bundle:{i}")
        for video in videos for q, i in zip(
            video.question_ids, video.provider_indices))
    parent, candidate = _versions()
    request = build_confirmation_request(
        parent=parent, candidate=candidate, cases=cases,
        model_identity="paired-runtime-v1",
        decoding_settings={"temperature": 0.0},
        cache_reset_identity="clean-paired-v1",
        evaluation_pipeline_identity="dvd-paired-v1")
    return adapter, runtime, request, parent, candidate


def test_real_adapter_applies_each_meta_prompt_and_aggregates_pairs(tmp_path):
    adapter, runtime, request, parent, candidate = _adapter(tmp_path)
    result = adapter.evaluate(
        request=request, parent=parent, candidate=candidate,
        output_directory=str(tmp_path / "out"))
    assert runtime.generators == [
        ("parent", "parent", "PARENT EXACT"),
        ("candidate", "candidate", "CANDIDATE EXACT")]
    assert len(result.outcomes) == 6
    assert len(runtime.qa_calls) == 12
    manifest = json.loads((tmp_path / "out/dvd_confirmation_manifest.json").read_text())
    assert manifest["legacy_property_routing_used"] is False
    assert manifest["aggregate"]["evaluated_qa_count"] == 6
    assert manifest["aggregate"]["wrong_to_correct"] == 5
    assert all(row["parent_prompt_artifact_ref"] !=
               row["candidate_prompt_artifact_ref"]
               for row in manifest["qa_results"])


def test_resume_avoids_caption_and_qa_reinvocation(tmp_path):
    adapter, runtime, request, parent, candidate = _adapter(tmp_path)
    kwargs = dict(request=request, parent=parent, candidate=candidate,
                  output_directory=str(tmp_path / "out"))
    first = adapter.evaluate(**kwargs)
    generator_count, qa_count = len(runtime.generators), len(runtime.qa_calls)
    second = adapter.evaluate(**kwargs)
    assert first == second
    assert len(runtime.generators) == generator_count
    assert len(runtime.qa_calls) == qa_count


def test_partial_resume_reuses_completed_paired_qa_rows(tmp_path):
    adapter, runtime, request, parent, candidate = _adapter(tmp_path)
    kwargs = dict(request=request, parent=parent, candidate=candidate,
                  output_directory=str(tmp_path / "out"))
    adapter.evaluate(**kwargs)
    qa_count = len(runtime.qa_calls)
    (tmp_path / "out/dvd_confirmation_manifest.json").unlink()
    adapter.evaluate(**kwargs)
    assert len(runtime.qa_calls) == qa_count


def test_runtime_identity_and_confirmation_set_mismatch_fail_before_calls(tmp_path):
    adapter, runtime, request, parent, candidate = _adapter(tmp_path)
    bad_request = type(request)(
        **{**request.__dict__, "model_identity": "different"})
    with pytest.raises(ConfirmationEvaluationConflictError):
        adapter.evaluate(request=bad_request, parent=parent,
                         candidate=candidate,
                         output_directory=str(tmp_path / "bad"))
    assert runtime.generators == []
    assert runtime.qa_calls == []


def test_execution_failure_is_persisted_without_losing_pair(tmp_path):
    adapter, runtime, request, parent, candidate = _adapter(tmp_path)
    original = runtime.qa_fn
    def fail_one(**kwargs):
        if kwargs["question_id"] == "q2" and "/candidate/" in kwargs["run_dir"]:
            raise RuntimeError("intentional QA failure")
        return original(**kwargs)
    runtime.qa_fn = fail_one
    result = adapter.evaluate(
        request=request, parent=parent, candidate=candidate,
        output_directory=str(tmp_path / "out"))
    row = next(item for item in result.outcomes if item.qa_id == "q2")
    assert row.candidate_correct is None
    assert "intentional QA failure" in row.candidate_error
    manifest = json.loads((tmp_path / "out/dvd_confirmation_manifest.json").read_text())
    assert manifest["aggregate"]["execution_failures"] == ["q2"]


def test_lazy_runtime_is_not_constructed_before_confirmation_call(tmp_path):
    adapter, _runtime, request, parent, candidate = _adapter(tmp_path)
    calls = []
    lazy = LazyDVDMetaPromptConfirmationEvaluator(
        configuration_identity={"runtime": "real"},
        factory=lambda: calls.append("constructed") or adapter)
    assert calls == []
    lazy.evaluate(request=request, parent=parent, candidate=candidate,
                  output_directory=str(tmp_path / "out"))
    assert calls == ["constructed"]


def test_real_factory_config_schema_and_mock_identities_fail_fast(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema_version": "wrong"}))
    with pytest.raises(CheckpointGConfigurationError, match="unsupported"):
        _load_configuration(str(path))
    with pytest.raises(CheckpointGConfigurationError, match="reviewed real"):
        _real_identity("fixture-model", "model")
