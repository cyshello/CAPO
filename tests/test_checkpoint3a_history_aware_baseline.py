import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from surrogate_rollout.captioning.history_aware_baseline import (
    CAPTION_CACHE_SCHEMA_VERSION,
    CAPTION_OUTPUT_CONTRACT,
    CAPTION_OUTPUT_CONTRACT_VERSION,
    CAPTION_PARSE_NORMALIZATION_VERSION,
    CAPTION_PARSE_SCHEMA_VERSION,
    CAPTION_RETRY_POLICY_VERSION,
    HistoryAwareBaselineCaptionViewBuilder,
    HistoryAwareSegmentCaptioner,
    _parse_caption_output_with_metadata,
    build_caption_output_contract,
    build_history_snapshot,
)
from surrogate_rollout.prompt_routing.scaffold_applier import create_scaffold_applier
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.policies.history_aware_vlm_router import (
    HistoryAwareRouterError,
    HistoryAwareVLMRouter,
    STRUCTURED_OUTPUT_VERSION,
    build_router_output_schema,
    parse_router_output,
)
from surrogate_rollout.prompt_routing.schemas import SegmentContext


ROOT = Path(__file__).parents[1]


def test_caption_parser_accepts_description_without_subject_registry():
    parsed, result = _parse_caption_output_with_metadata(json.dumps({
        "clip_description": "A person demonstrates an action."
    }))

    assert parsed == {
        "clip_description": "A person demonstrates an action.",
        "subject_registry": {},
    }
    assert result["status"] == "valid"
    assert result["subject_registry_populated"] is False
    assert "default_empty_subject_registry" in result["normalizations"]


def test_default_caption_contract_requests_an_empty_registry():
    assert 'Set "subject_registry" to an empty object {}' in \
        CAPTION_OUTPUT_CONTRACT


def test_optional_caption_contract_allows_larger_model_registry():
    contract = build_caption_output_contract("optional")

    assert '"subject_registry" is optional' in contract
    assert "A valid populated registry is allowed" in contract


def test_caption_parser_preserves_populated_registry_for_larger_models():
    registry = {"person_1": {"name": "demonstrator"}}
    parsed, result = _parse_caption_output_with_metadata(json.dumps({
        "clip_description": "A person demonstrates an action.",
        "subject_registry": registry,
    }))

    assert parsed["subject_registry"] == registry
    assert result["subject_registry_populated"] is True
    assert "default_empty_subject_registry" not in result["normalizations"]


def test_caption_parser_safely_unwraps_one_object_array():
    parsed, result = _parse_caption_output_with_metadata(
        "```json\n" + json.dumps([{
            "clip_description": "A map appears on screen."
        }]) + "\n```")

    assert parsed["clip_description"] == "A map appears on screen."
    assert parsed["subject_registry"] == {}
    assert result["original_top_level_type"] == "list"
    assert "unwrap_singleton_object_array" in result["normalizations"]


def test_caption_parser_normalizes_only_invalid_percent_escape():
    parsed, result = _parse_caption_output_with_metadata(
        r'{"clip_description":"A percentage symbol (\%) is visible."}')
    assert parsed["clip_description"] == "A percentage symbol (%) is visible."
    assert result["status"] == "valid"
    assert "replace_invalid_percent_escape" in result["normalizations"]

    escaped, escaped_result = _parse_caption_output_with_metadata(
        r'{"clip_description":"A literal backslash (\\%) is visible."}')
    assert escaped["clip_description"] == r"A literal backslash (\%) is visible."
    assert "replace_invalid_percent_escape" not in \
        escaped_result["normalizations"]

    invalid, invalid_result = _parse_caption_output_with_metadata(
        r'{"clip_description":"An unrelated invalid escape \q remains."}')
    assert invalid == {}
    assert invalid_result["error"] == "invalid_json"


@pytest.mark.parametrize("value,error", [
    ([], "invalid_top_level_array"),
    ([{"clip_description": "one"}, {"clip_description": "two"}],
     "invalid_top_level_array"),
    ({"subject_registry": {}}, "missing_clip_description"),
    ({"clip_description": "   "}, "missing_clip_description"),
])
def test_caption_parser_rejects_ambiguous_or_descriptionless_output(value, error):
    parsed, result = _parse_caption_output_with_metadata(json.dumps(value))

    assert parsed == {}
    assert result["status"] == "invalid"
    assert result["error"] == error


def components():
    data = json.loads((
        ROOT / "prompt_routing/fixtures/stage4_7_components.json").read_text())
    bank = prompt_bank_from_json(data["prompt_bank"])
    router = replace(
        router_policy_from_json(data["router_policy"]),
        policy_type="history_aware_vlm",
        configuration={"max_selected_properties": 3, "max_tokens": 64},
    )
    return (
        bank,
        router,
        scaffold_policy_from_json(data["scaffold_policy"]),
        scaffold_contract_from_json(data["scaffold_contract"]),
    )


class RouterVLM:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def caption(self, images, prompt, **kwargs):
        self.calls.append((tuple(images), prompt, kwargs))
        return self.output


def context():
    return SegmentContext(
        video_id="video-1", segment_id="0_10",
        timestamp_start=0.0, timestamp_end=10.0,
        segment_features={"frame_references": ["frame-0.jpg"]},
        history_summary=json.dumps({"preceding_captions": []}),
        question="SECRET QUESTION MUST NOT LEAK",
        metadata={"answer": "SECRET ANSWER", "used_segments": ["10_20"]},
    )


def test_router_strict_json_deduplicates_and_restores_codebook_order():
    bank, policy, _, _ = components()
    vlm = RouterVLM(json.dumps({
        "property_ids": ["pe_text", "pe_default", "pe_text"]}))
    decision = HistoryAwareVLMRouter(vlm).route(context(), bank, policy)

    assert decision.selected_prompt_ids == ("pe_default", "pe_text")
    request_prompt = vlm.calls[0][1]
    assert "SECRET QUESTION" not in request_prompt
    assert "SECRET ANSWER" not in request_prompt
    assert "used_segments" not in request_prompt
    assert vlm.calls[0][0] == ("frame-0.jpg",)
    schema = vlm.calls[0][2]["json_schema"]
    assert schema == build_router_output_schema(
        tuple(entry.prompt_id for entry in bank.entries if entry.status == "active"),
        3,
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["property_ids"]["maxItems"] == 3
    request = json.loads(request_prompt.split("\n", 1)[1])
    assert set(request) == {
        "active_codebook", "current_segment", "max_selected_properties",
        "output_schema", "preceding_caption_history", "schema_version",
    }
    assert request["max_selected_properties"] == 3
    assert request["output_schema"] == schema
    assert HistoryAwareVLMRouter(vlm).configuration_identity[
        "structured_output"]["version"] == STRUCTURED_OUTPUT_VERSION


def test_qwen_sampling_enforces_json_schema_without_fallback():
    from surrogate_rollout.captioning.qwen25_vl import Qwen25VLCaptioner

    captioner = object.__new__(Qwen25VLCaptioner)
    captioner.default_max_tokens = 256
    schema = build_router_output_schema(("pe_default",), 1)
    sampling = captioner._sampling(
        64, 0.0, 1.0, json_schema=schema)

    assert sampling.structured_outputs.json == schema
    assert sampling.structured_outputs.disable_fallback is True
    assert sampling.structured_outputs.disable_additional_properties is True


def test_qwen_sampling_remains_unconstrained_without_schema():
    from surrogate_rollout.captioning.qwen25_vl import Qwen25VLCaptioner

    captioner = object.__new__(Qwen25VLCaptioner)
    captioner.default_max_tokens = 256
    sampling = captioner._sampling(64, 0.0, 1.0)

    assert sampling.structured_outputs is None


def test_empty_property_ids_uses_only_base_caption_prompt():
    bank, policy, scaffold_policy, contract = components()
    decision = HistoryAwareVLMRouter(
        RouterVLM('{"property_ids": []}')).route(context(), bank, policy)
    base = (
        "BASE TRANSCRIPT_PLACEHOLDER CLIP_START_TIME CLIP_END_TIME\n"
        "ROUTED_INSTRUCTIONS_PLACEHOLDER")
    composed = create_scaffold_applier(
        scaffold_policy, base_prompt_template=base).apply(
            context=context(), selected_entries=(), routing_decision=decision,
            scaffold_policy=scaffold_policy, scaffold_contract=contract)
    assert decision.selected_prompt_ids == ()
    assert composed.prompt_text == base.replace(
        "ROUTED_INSTRUCTIONS_PLACEHOLDER", "")
    assert "Additional routed caption instructions:" not in composed.prompt_text


def test_history_blocks_use_floor_of_segment_start_boundary_rule():
    before = build_history_snapshot(
        segment_id="299_300", block_seconds=300, preceding=[],
        max_history_captions=30)
    boundary = build_history_snapshot(
        segment_id="300_310", block_seconds=300, preceding=[],
        max_history_captions=30)
    assert before["block_index"] == 0
    assert before["block_start_seconds"] == 0
    assert boundary["block_index"] == 1
    assert boundary["block_start_seconds"] == 300


@pytest.mark.parametrize("raw", [
    "```json\n{\"property_ids\": []}\n```",
    '{"property_ids": [], "explanation": "no"}',
    '{"property_ids": "pe_default"}',
    "not json",
])
def test_router_rejects_non_strict_output(raw):
    with pytest.raises(HistoryAwareRouterError):
        parse_router_output(raw)


def test_router_rejects_unknown_or_over_budget_properties():
    bank, policy, _, _ = components()
    with pytest.raises(HistoryAwareRouterError, match="unknown"):
        HistoryAwareVLMRouter(RouterVLM(json.dumps({
            "property_ids": ["not-active"]
        }))).route(context(), bank, policy)
    limited = replace(policy, configuration={"max_selected_properties": 1})
    with pytest.raises(HistoryAwareRouterError, match="maximum"):
        HistoryAwareVLMRouter(RouterVLM(json.dumps({
            "property_ids": ["pe_default", "pe_text"]
        }))).route(context(), bank, limited)


class SharedMockVLM:
    def __init__(self):
        self.calls = []
        self.caption_number = 0

    def caption(self, images, prompt, **kwargs):
        kind = ("router" if "history_aware_vlm_router_request_v1" in prompt
                else "caption")
        self.calls.append({
            "kind": kind, "images": tuple(images), "prompt": prompt,
            "kwargs": kwargs,
        })
        if kind == "router":
            return json.dumps({"property_ids": ["pe_temporal"]})
        self.caption_number += 1
        return json.dumps({
            "clip_description": f"generated caption {self.caption_number}",
            "subject_registry": {"S1": f"subject {self.caption_number}"},
        })


def test_sequential_history_is_shared_persisted_cached_and_resumable(tmp_path):
    bank, policy, scaffold, contract = components()
    vlm = SharedMockVLM()
    builder = HistoryAwareBaselineCaptionViewBuilder(
        router=HistoryAwareVLMRouter(vlm),
        segment_captioner=HistoryAwareSegmentCaptioner(vlm),
        merge_fn=lambda values: {"merged_count": len(values)},
    )
    clips = [
        ("0_10", {"files": [tmp_path / "f0.jpg"], "transcript": "zero"}),
        ("10_20", {"files": [tmp_path / "f1.jpg"], "transcript": "one"}),
        ("300_310", {"files": [tmp_path / "f2.jpg"], "transcript": "two"}),
    ]
    sample = {
        "sample_id": "sample",
        "video_path": str(tmp_path / "video.mp4"),
        "extra": {"videoID": "video-1", "subtitle_path": None},
    }
    work = tmp_path / "work"
    artifact = builder.build(
        sample=sample, clip_index=clips, prompt_bank=bank,
        router_policy=policy, scaffold_policy=scaffold,
        scaffold_contract=contract,
        base_prompt_template=(
            "TRANSCRIPT_PLACEHOLDER CLIP_START_TIME CLIP_END_TIME "
            "ROUTED_INSTRUCTIONS_PLACEHOLDER"),
        merge_prompt="merge", work_root=str(work),
        history_block_seconds=300, max_history_captions=30,
        candidate_cache_root=str(tmp_path / "cache"),
        cache_manifest_path=str(tmp_path / "cache_manifest.jsonl"),
    )

    assert [call["kind"] for call in vlm.calls] == [
        "router", "caption", "router", "caption", "router", "caption"]
    histories = [json.loads(line) for line in
                 Path(artifact.frozen_histories_path).read_text().splitlines()]
    assert histories[0]["preceding_segment_ids"] == []
    assert histories[1]["preceding_segment_ids"] == ["0_10"]
    assert histories[2]["preceding_segment_ids"] == []
    for index in range(3):
        router_request = json.loads(
            vlm.calls[index * 2]["prompt"].split("\n", 1)[1])
        serialized = json.dumps(
            router_request["preceding_caption_history"],
            sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        assert serialized == histories[index]["serialized_history"]
        assert histories[index]["serialized_history"] in \
            vlm.calls[index * 2 + 1]["prompt"]
        assert CAPTION_OUTPUT_CONTRACT in vlm.calls[index * 2 + 1]["prompt"]
    assert artifact.caption_call_count == artifact.router_call_count == 3
    assert artifact.caption_cache_hits == 0
    assert Path(artifact.routing_manifest_path).exists()
    assert Path(artifact.routed_view_path).exists()
    captions = json.loads(Path(artifact.captions_path).read_text())
    assert captions["subject_registry"] == {"merged_count": 3}
    routing_manifest = json.loads(Path(
        artifact.routing_manifest_path).read_text())
    first_cache = json.loads(Path(json.loads(
        Path(routing_manifest["caption_cache_keys_path"])
        .read_text().splitlines()[0]
    )["result_path"]).read_text())
    assert first_cache["schema_version"] == CAPTION_CACHE_SCHEMA_VERSION
    assert first_cache["caption_output_contract_version"] == \
        CAPTION_OUTPUT_CONTRACT_VERSION
    assert first_cache["parse_result"]["schema_version"] == \
        CAPTION_PARSE_SCHEMA_VERSION
    identity = builder.segment_captioner.configuration_identity
    assert identity["caption_output_contract_version"] == \
        CAPTION_OUTPUT_CONTRACT_VERSION
    assert identity["caption_parse_schema_version"] == \
        CAPTION_PARSE_SCHEMA_VERSION
    assert identity["caption_parse_normalization_version"] == \
        CAPTION_PARSE_NORMALIZATION_VERSION
    assert identity["caption_parse_max_retries"] == 5
    assert identity["subject_registry_mode"] == "empty"

    before = len(vlm.calls)
    resumed = builder.build(
        sample=sample, clip_index=clips, prompt_bank=bank,
        router_policy=policy, scaffold_policy=scaffold,
        scaffold_contract=contract,
        base_prompt_template=(
            "TRANSCRIPT_PLACEHOLDER CLIP_START_TIME CLIP_END_TIME "
            "ROUTED_INSTRUCTIONS_PLACEHOLDER"),
        merge_prompt="merge", work_root=str(work),
        history_block_seconds=300, max_history_captions=30,
        candidate_cache_root=str(tmp_path / "cache"),
        cache_manifest_path=str(tmp_path / "cache_manifest.jsonl"),
    )
    assert len(vlm.calls) == before
    assert resumed.resumed_segment_count == 3
    assert resumed.router_call_count == resumed.caption_call_count == 0


def test_parallel_captioning_schedules_history_blocks_and_merges_deterministically(
        tmp_path):
    bank, policy, scaffold, contract = components()
    worker_calls = {}

    def fixture_executor(assignments, common):
        manifests = []
        worker_bank = prompt_bank_from_json(common["prompt_bank"])
        worker_policy = router_policy_from_json(common["router_policy"])
        worker_scaffold = scaffold_policy_from_json(common["scaffold_policy"])
        worker_contract = scaffold_contract_from_json(common["scaffold_contract"])
        for gpu, jobs in assignments:
            vlm = SharedMockVLM()
            worker_calls[gpu] = vlm.calls
            worker = HistoryAwareBaselineCaptionViewBuilder(
                router=HistoryAwareVLMRouter(vlm),
                segment_captioner=HistoryAwareSegmentCaptioner(vlm),
                merge_fn=lambda values: {"merged_count": len(values)},
            )
            for _, block_clips, block_root, fragment_path in jobs:
                artifact = worker.build(
                    sample=common["sample"], clip_index=list(block_clips),
                    prompt_bank=worker_bank, router_policy=worker_policy,
                    scaffold_policy=worker_scaffold,
                    scaffold_contract=worker_contract,
                    base_prompt_template=common["base_prompt_template"],
                    merge_prompt=common["merge_prompt"], work_root=block_root,
                    history_block_seconds=common["history_block_seconds"],
                    max_history_captions=common["max_history_captions"],
                    candidate_cache_root=common["candidate_cache_root"],
                    cache_manifest_path=fragment_path,
                )
                manifests.append(artifact.routing_manifest_path)
        return tuple(manifests)

    parent_vlm = SharedMockVLM()
    builder = HistoryAwareBaselineCaptionViewBuilder(
        router=HistoryAwareVLMRouter(parent_vlm),
        segment_captioner=HistoryAwareSegmentCaptioner(parent_vlm),
        merge_fn=lambda values: {"merged_count": len(values)},
        parallel_gpus=("4", "5"), parallel_executor=fixture_executor,
    )
    clips = [
        ("0_10", {"files": [tmp_path / "f0.jpg"], "transcript": "zero"}),
        ("10_20", {"files": [tmp_path / "f1.jpg"], "transcript": "one"}),
        ("300_310", {"files": [tmp_path / "f2.jpg"], "transcript": "two"}),
        ("600_610", {"files": [tmp_path / "f3.jpg"], "transcript": "three"}),
    ]
    sample = {
        "sample_id": "sample", "video_path": str(tmp_path / "video.mp4"),
        "extra": {"videoID": "video-1", "subtitle_path": None},
    }
    artifact = builder.build(
        sample=sample, clip_index=clips, prompt_bank=bank,
        router_policy=policy, scaffold_policy=scaffold,
        scaffold_contract=contract,
        base_prompt_template=(
            "TRANSCRIPT_PLACEHOLDER CLIP_START_TIME CLIP_END_TIME "
            "ROUTED_INSTRUCTIONS_PLACEHOLDER"),
        merge_prompt="merge", work_root=str(tmp_path / "parallel"),
        history_block_seconds=300, max_history_captions=30,
        candidate_cache_root=str(tmp_path / "cache"),
        cache_manifest_path=str(tmp_path / "cache_manifest.jsonl"),
    )

    assert parent_vlm.calls == []
    assert set(worker_calls) == {"4", "5"}
    assert artifact.segment_ids == tuple(segment_id for segment_id, _ in clips)
    histories = [json.loads(line) for line in
                 Path(artifact.frozen_histories_path).read_text().splitlines()]
    assert [item["preceding_segment_ids"] for item in histories] == [
        [], ["0_10"], [], []]
    manifest = json.loads(Path(artifact.routing_manifest_path).read_text())
    assert manifest["parallel_captioning"] == {
        "scheduler_version": "history_block_gpu_scheduler_v1",
        "unit": "history_block", "worker_count": 2,
        "gpu_ids": ["4", "5"], "block_count": 3,
        "merge_order": "ascending_segment_start",
    }
    assert manifest["router_calls"] == manifest["caption_calls"] == 4
    assert len(Path(tmp_path / "cache_manifest.jsonl").read_text().splitlines()) == 4
    captions = json.loads(Path(artifact.captions_path).read_text())
    assert captions["subject_registry"] == {"merged_count": 4}

    reassigned = HistoryAwareBaselineCaptionViewBuilder(
        router=HistoryAwareVLMRouter(parent_vlm),
        segment_captioner=HistoryAwareSegmentCaptioner(parent_vlm),
        merge_fn=lambda values: {"merged_count": len(values)},
        parallel_gpus=("7", "6"), parallel_executor=fixture_executor,
    ).build(
        sample=sample, clip_index=clips, prompt_bank=bank,
        router_policy=policy, scaffold_policy=scaffold,
        scaffold_contract=contract,
        base_prompt_template=(
            "TRANSCRIPT_PLACEHOLDER CLIP_START_TIME CLIP_END_TIME "
            "ROUTED_INSTRUCTIONS_PLACEHOLDER"),
        merge_prompt="merge", work_root=str(tmp_path / "parallel"),
        history_block_seconds=300, max_history_captions=30,
        candidate_cache_root=str(tmp_path / "cache"),
        cache_manifest_path=str(tmp_path / "cache_manifest.jsonl"),
    )
    assert reassigned.captions_hash == artifact.captions_hash
    assert reassigned.resumed_segment_count == 4
    assert reassigned.router_call_count == reassigned.caption_call_count == 0
    assert len(Path(tmp_path / "cache_manifest.jsonl").read_text().splitlines()) == 4
    reassigned_manifest = json.loads(
        Path(reassigned.routing_manifest_path).read_text())
    assert reassigned_manifest["parallel_captioning"]["gpu_ids"] == ["7", "6"]


def test_parallel_gpu_ids_must_be_unique(tmp_path):
    vlm = SharedMockVLM()
    with pytest.raises(ValueError, match="unique"):
        HistoryAwareBaselineCaptionViewBuilder(
            router=HistoryAwareVLMRouter(vlm),
            segment_captioner=HistoryAwareSegmentCaptioner(vlm),
            parallel_gpus=("4", "4"),
        )


def test_segment_captioner_can_reuse_persistent_remote_worker():
    class ForbiddenLocalVLM:
        def caption(self, *_args, **_kwargs):
            raise AssertionError("persistent dispatch must not load a parent VLM")

    class RemoteDispatcher:
        def __init__(self):
            self.calls = []

        def caption_segment(self, **kwargs):
            self.calls.append(kwargs)
            return "remote-result"

    remote = RemoteDispatcher()
    captioner = HistoryAwareSegmentCaptioner(
        ForbiddenLocalVLM(), remote_dispatcher=remote)
    result = captioner.caption(
        sample={"video_path": "video.mp4"}, video_id="video-1",
        segment_id="0_10", clip_info={"files": ["frame.jpg"]},
        composed_prompt=object(), history_snapshot={"history_hash": "h"},
        merge_prompt="merge")

    assert result == "remote-result"
    assert remote.calls[0]["segment_id"] == "0_10"


def _caption_once(tmp_path, vlm, cache_root):
    video = tmp_path / "video.mp4"
    frame = tmp_path / "frame.jpg"
    video.write_bytes(b"video")
    frame.write_bytes(b"frame")
    # Keep the provider identity stable across resume assertions even when the
    # test swaps in a guard object that must never be called.
    captioner = HistoryAwareSegmentCaptioner(
        vlm, backend_id="test.caption.retry.backend")
    result = captioner.caption(
        sample={
            "video_path": str(video),
            "extra": {"videoID": "video-1", "subtitle_path": None},
        },
        video_id="video-1", segment_id="0_10",
        clip_info={"files": [str(frame)], "transcript": "spoken words"},
        composed_prompt=SimpleNamespace(
            prompt_text=("TRANSCRIPT_PLACEHOLDER CLIP_START_TIME "
                         "CLIP_END_TIME"),
            prompt_hash="prompt-hash", bank_version="bank-v1",
            router_version="router-v1", scaffold_version="scaffold-v1",
            contract_version="contract-v1"),
        history_snapshot=build_history_snapshot(
            segment_id="0_10", block_seconds=300, preceding=[],
            max_history_captions=30),
        merge_prompt="merge", cache_root=str(cache_root),
        cache_manifest_path=str(tmp_path / "manifest.jsonl"))
    return result, captioner


def test_caption_parse_failure_retries_until_valid(tmp_path):
    class RetryVLM:
        def __init__(self):
            self.calls = 0

        def caption(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return '{"clip_description": "unterminated'
            return json.dumps({
                "clip_description": "valid retry", "subject_registry": {}})

    vlm = RetryVLM()
    result, _ = _caption_once(tmp_path, vlm, tmp_path / "cache")
    assert vlm.calls == 2
    assert result.model_call_count == 2
    assert result.retry_count == 1
    assert result.parsed["clip_description"].startswith("valid retry")
    artifact = json.loads(Path(result.result_path).read_text())
    assert artifact["attempt_count"] == 2
    assert artifact["retry_count"] == 1
    assert artifact["retry_exhausted"] is False
    assert [row["parse_result"]["status"] for row in artifact["attempts"]] == [
        "invalid", "valid"]


def test_caption_parse_failure_stops_after_five_retries(tmp_path):
    class InvalidVLM:
        def __init__(self):
            self.calls = 0

        def caption(self, *_args, **_kwargs):
            self.calls += 1
            return '{"clip_description": "unterminated'

    vlm = InvalidVLM()
    result, _ = _caption_once(tmp_path, vlm, tmp_path / "cache")
    assert vlm.calls == 6  # initial attempt plus five retries
    assert result.model_call_count == 6
    assert result.retry_count == 5
    assert result.parsed == {}
    artifact = json.loads(Path(result.result_path).read_text())
    assert artifact["attempt_count"] == 6
    assert artifact["retry_exhausted"] is True
    assert all(row["parse_result"]["error"] == "invalid_json"
               for row in artifact["attempts"])

    class ForbiddenVLM:
        def caption(self, *_args, **_kwargs):
            raise AssertionError("exhausted retry artifact must resume exactly")

    resumed, _ = _caption_once(tmp_path, ForbiddenVLM(), tmp_path / "cache")
    assert resumed.cache_hit is True
    assert resumed.model_call_count == 0
    assert resumed.retry_count == 5
    assert resumed.parsed == {}


def test_legacy_invalid_cache_is_immutable_and_retried_in_sidecar(tmp_path):
    class MutableVLM:
        def __init__(self, output):
            self.output = output
            self.calls = 0

        def caption(self, *_args, **_kwargs):
            self.calls += 1
            return self.output

    invalid_vlm = MutableVLM('{"clip_description": "unterminated')
    first, _ = _caption_once(tmp_path, invalid_vlm, tmp_path / "cache")
    base_path = Path(first.result_path)
    legacy = json.loads(base_path.read_text())
    legacy["schema_version"] = "history_aware_caption_cache_v2"
    for field in (
            "caption_retry_policy_version", "caption_parse_max_retries",
            "attempt_count", "retry_count", "retry_exhausted", "attempts"):
        legacy.pop(field, None)
    base_path.write_text(json.dumps(legacy, sort_keys=True))
    immutable_legacy_text = base_path.read_text()

    valid_vlm = MutableVLM(json.dumps({
        "clip_description": "recovered", "subject_registry": {}}))
    retried, _ = _caption_once(tmp_path, valid_vlm, tmp_path / "cache")
    assert valid_vlm.calls == 1
    assert retried.retry_count == 1
    assert retried.result_path != str(base_path)
    assert f"/parse_retries/{CAPTION_RETRY_POLICY_VERSION}/" in \
        retried.result_path
    assert base_path.read_text() == immutable_legacy_text
    retry_artifact = json.loads(Path(retried.result_path).read_text())
    assert retry_artifact["attempts"][0]["source"] == \
        "immutable_cached_initial_attempt"

    no_call_vlm = MutableVLM("must not be called")
    resumed, _ = _caption_once(tmp_path, no_call_vlm, tmp_path / "cache")
    assert no_call_vlm.calls == 0
    assert resumed.cache_hit is True
    assert resumed.result_path == retried.result_path


def test_cached_percent_escape_is_reparsed_in_new_sidecar_without_model_call(
        tmp_path):
    class VLM:
        def __init__(self, output):
            self.output = output
            self.calls = 0

        def caption(self, *_args, **_kwargs):
            self.calls += 1
            return self.output

    first, _ = _caption_once(
        tmp_path, VLM('{"clip_description":"unterminated'), tmp_path / "cache")
    base_path = Path(first.result_path)
    legacy = json.loads(base_path.read_text())
    percent_raw = r'{"clip_description":"A percentage symbol (\%) is visible."}'
    legacy.update({
        "caption_retry_policy_version": "caption_parse_retry_v1",
        "raw_output": percent_raw,
        "parsed": {},
        "parse_result": {
            "schema_version": CAPTION_PARSE_SCHEMA_VERSION,
            "status": "invalid", "error": "invalid_json",
            "normalizations": [],
        },
    })
    legacy.pop("caption_parse_normalization_version", None)
    base_path.write_text(json.dumps(legacy, sort_keys=True))
    immutable_legacy = base_path.read_text()

    forbidden = VLM("must not be called")
    recovered, _ = _caption_once(tmp_path, forbidden, tmp_path / "cache")

    assert forbidden.calls == 0
    assert recovered.model_call_count == 0
    assert recovered.parsed["clip_description"].startswith(
        "A percentage symbol (%) is visible.")
    assert recovered.result_path != str(base_path)
    assert f"/parse_retries/{CAPTION_RETRY_POLICY_VERSION}/" in \
        recovered.result_path
    assert base_path.read_text() == immutable_legacy
    sidecar = json.loads(Path(recovered.result_path).read_text())
    assert sidecar["caption_parse_normalization_version"] == \
        CAPTION_PARSE_NORMALIZATION_VERSION
    assert sidecar["attempts"][0]["source"] == \
        "immutable_cached_initial_attempt"
    assert sidecar["attempts"][0]["parsed_valid"] is True


def test_dvd_raw_frame_inspection_uses_pool_proxy_and_standalone_is_preserved():
    import dvd_backend

    vlm = SharedMockVLM()
    builder = HistoryAwareBaselineCaptionViewBuilder(
        router=HistoryAwareVLMRouter(vlm),
        segment_captioner=HistoryAwareSegmentCaptioner(vlm),
        parallel_gpus=("4", "5"),
    )
    original = dvd_backend._captioner
    standalone = object()
    try:
        dvd_backend._captioner = standalone
        assert dvd_backend.get_captioner() is standalone
        assert builder._parent_dvd_backend is None

        dvd_backend._captioner = None
        calls = []
        builder.caption_raw_vlm = lambda **kwargs: calls.append(kwargs) or "remote"
        builder._install_parent_vlm_proxy()
        assert dvd_backend.get_captioner().caption(
            ["frame.jpg"], "inspect", max_tokens=512) == "remote"
        assert calls == [{
            "images": ["frame.jpg"], "prompt": "inspect", "max_tokens": 512}]
        builder._restore_parent_vlm_proxy()
        assert dvd_backend._captioner is None
    finally:
        dvd_backend._captioner = original


def test_persistent_pool_demultiplexes_concurrent_video_requests():
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    class AliveProcess:
        def is_alive(self):
            return True

    builder = HistoryAwareBaselineCaptionViewBuilder(
        router=object(), segment_captioner=object(), parallel_gpus=("4", "5"))
    builder._pool_processes = {"4": AliveProcess(), "5": AliveProcess()}
    builder._pool_result_queue = queue.Queue()
    barrier = threading.Barrier(2)

    class CommandQueue:
        def put(self, command):
            barrier.wait(timeout=2.0)
            job = command["jobs"][0]
            builder._pool_result_queue.put({
                "request_id": command["request_id"],
                "gpu": str(job[0]),
                "completed": ((job[0], job[2]),),
            })

    builder._pool_command_queues = {"4": CommandQueue(), "5": CommandQueue()}
    builder._pool_result_thread = threading.Thread(
        target=builder._route_pool_results, daemon=True)
    builder._pool_result_thread.start()
    try:
        def dispatch(gpu, index):
            return builder._run_gpu_workers(
                ((gpu, ((index, (), f"manifest-{index}", "fragment"),)),), {})

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(
                lambda row: dispatch(*row), (("4", 4), ("5", 5))))
        assert results == (("manifest-4",), ("manifest-5",))
    finally:
        builder._pool_result_queue.put({"request_id": "__parent_stop__"})
        builder._pool_result_thread.join(timeout=2.0)
        builder._pool_processes.clear()
