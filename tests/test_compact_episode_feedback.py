"""Checkpoint D1.5 compact episode-feedback request tests."""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from surrogate_rollout.optimization.llm_episode_feedback import (
    COMPACT_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION,
    COMPACT_EPISODE_FEEDBACK_SYSTEM_INSTRUCTION,
    EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION,
    EpisodeFeedbackRequestError,
    LLMEpisodeFeedbackGenerator,
    LegacyEpisodeFeedbackArtifactResolver,
    build_compact_episode_feedback_request,
    build_episode_feedback_request,
    reconstruct_compact_history_snapshots,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json, sha256_text
from test_legacy_intervention_adapter import convert, legacy_bundle
from test_llm_episode_feedback import FakeBackend, POLICY_VERSION


def compact_request(tmp_path, *, episode_transform=None):
    bundle = legacy_bundle(tmp_path)
    episode = convert(bundle)
    if episode_transform is not None:
        episode = episode_transform(episode)
    request = build_compact_episode_feedback_request(
        episode, artifact_resolver=LegacyEpisodeFeedbackArtifactResolver())
    return bundle, episode, request


def history_snapshot(segment_id, items):
    items = [dict(item) for item in items]
    value = {
        "schema_version": "frozen_local_caption_history_v2",
        "segment_id": segment_id,
        "block_index": 0,
        "block_start_seconds": 0.0,
        "block_end_seconds": 60.0,
        "max_history_captions": 64,
        "preceding_captions": items,
        "preceding_segment_ids": [item["segment_id"] for item in items],
        "history": items,
        "source": "sequential_history_aware_baseline",
    }
    serialized = dumps_canonical({
        key: value[key] for key in (
            "schema_version", "block_index", "block_start_seconds",
            "block_end_seconds", "max_history_captions",
            "preceding_captions")})
    value["serialized_history"] = serialized
    value["history_hash"] = sha256_text(serialized)
    return value


def overlapping_episode(episode):
    first = {"segment_id": "history-a", "caption": "exact caption A"}
    second = {"segment_id": "history-b", "caption": "exact caption B"}
    clips = (
        dataclasses.replace(
            episode.clips[0],
            history_snapshot=history_snapshot("0_10", [first, first])),
        dataclasses.replace(
            episode.clips[1],
            history_snapshot=history_snapshot("10_20", [first, second])),
    )
    return dataclasses.replace(episode, clips=clips)


def test_all_clips_qas_and_episode_delta_remain_once(tmp_path):
    _, episode, request = compact_request(tmp_path)
    payload = request.user_payload["episode"]
    assert [item["segment_id"] for item in payload["clips"]] == [
        item.segment_id for item in episode.clips]
    assert [item["qa_id"] for item in payload["qas"]] == [
        item.qa_id for item in episode.qa_outcomes]
    assert payload["qa_transition_summary"] == {
        "correct_to_correct": ["benchmark/train/2"],
        "wrong_to_wrong": ["benchmark/train/3"],
        "wrong_to_correct": ["benchmark/train/1"],
        "correct_to_wrong": [],
    }
    assert payload["prompt_delta"]["instruction"] == \
        episode.prompt_delta.instruction
    assert all("applied_prompt_delta_instruction" not in item
               for item in payload["clips"])
    assert dumps_canonical(request.user_payload).count(
        episode.prompt_delta.instruction) == 1


def test_clip_delta_mismatch_is_rejected_before_projection(tmp_path):
    def conflict(episode):
        delta = dataclasses.replace(
            episode.prompt_delta, instruction="conflicting instruction")
        clips = (dataclasses.replace(
            episode.clips[0], prompt_delta=delta), *episode.clips[1:])
        return dataclasses.replace(episode, clips=clips)

    bundle = legacy_bundle(tmp_path)
    episode = conflict(convert(bundle))
    with pytest.raises(EpisodeFeedbackRequestError, match="conflicts"):
        build_compact_episode_feedback_request(
            episode, artifact_resolver=LegacyEpisodeFeedbackArtifactResolver())


def test_history_catalog_deduplicates_content_and_preserves_occurrences(tmp_path):
    _, _, request = compact_request(
        tmp_path, episode_transform=overlapping_episode)
    payload = request.user_payload["episode"]
    assert [item["history_item_id"] for item in payload["history_catalog"]] == [
        "history-a", "history-b"]
    assert payload["clips"][0]["history_item_ids"] == [
        "history-a", "history-a"]
    assert payload["clips"][1]["history_item_ids"] == [
        "history-a", "history-b"]
    assert request.size_statistics.unique_history_item_count == 2
    assert request.size_statistics.total_history_item_occurrences == 4


def test_history_reconstruction_is_canonical_equal_for_every_clip(tmp_path):
    _, episode, request = compact_request(
        tmp_path, episode_transform=overlapping_episode)
    reconstructed = reconstruct_compact_history_snapshots(request.user_payload)
    assert len(reconstructed) == len(episode.clips)
    assert [dumps_canonical(item) for item in reconstructed] == [
        dumps_canonical(item.history_snapshot) for item in episode.clips]


def test_different_exact_history_items_are_not_merged(tmp_path):
    _, _, request = compact_request(
        tmp_path, episode_transform=overlapping_episode)
    catalog = request.user_payload["episode"]["history_catalog"]
    assert catalog[0]["content"] != catalog[1]["content"]
    assert catalog[0]["history_item_id"] != catalog[1]["history_item_id"]


def test_same_stable_history_id_with_different_content_fails_fast(tmp_path):
    def collision(episode):
        left = {"segment_id": "same", "caption": "left"}
        right = {"segment_id": "same", "caption": "right"}
        clips = (
            dataclasses.replace(
                episode.clips[0],
                history_snapshot=history_snapshot("0_10", [left])),
            dataclasses.replace(
                episode.clips[1],
                history_snapshot=history_snapshot("10_20", [right])),
        )
        return dataclasses.replace(episode, clips=clips)

    bundle = legacy_bundle(tmp_path)
    with pytest.raises(EpisodeFeedbackRequestError, match="collision"):
        build_compact_episode_feedback_request(
            collision(convert(bundle)),
            artifact_resolver=LegacyEpisodeFeedbackArtifactResolver())


def test_catalog_first_appearance_order_and_payload_are_deterministic(tmp_path):
    bundle = legacy_bundle(tmp_path)
    episode = overlapping_episode(convert(bundle))
    resolver = LegacyEpisodeFeedbackArtifactResolver()
    first = build_compact_episode_feedback_request(
        episode, artifact_resolver=resolver)
    second = build_compact_episode_feedback_request(
        episode, artifact_resolver=resolver)
    assert first.user_request == second.user_request
    assert first.payload_hash == second.payload_hash
    assert first.payload_hash == sha256_json(first.user_payload)


def test_all_tool_events_arguments_hits_and_evidence_are_preserved(tmp_path):
    bundle, _, request = compact_request(tmp_path)
    trajectory = request.user_payload["episode"]["qas"][0][
        "baseline_trajectory"]
    raw_events = [json.loads(line) for line in Path(
        bundle["baseline_qas"]).parent.joinpath(
            "qa", "0", "tool_events.jsonl").read_text().splitlines()
                  if line.strip()]
    assert len(trajectory["tool_events"]) == len(raw_events)
    event = trajectory["tool_events"][0]
    assert event["args"] == raw_events[0]["args"]
    assert event["hits"] == raw_events[0]["hits"]
    assert event["returned_segment_ids"] == ["0_10"]
    assert event["returned_evidence"] == ["baseline evidence for 0_10"]


def test_references_final_answer_and_unavailable_trajectory_are_preserved(tmp_path):
    _, _, request = compact_request(tmp_path)
    qas = request.user_payload["episode"]["qas"]
    available = qas[0]["baseline_trajectory"]
    assert available["reference_sets"] == {
        "explicitly_cited_segments": ["0_10"],
        "retrieved_segments": ["0_10"],
    }
    assert available["reference_evidence"] == [{
        "segment": "0_10", "set": "retrieved_segments",
        "reason": "clip_search_tool_hit", "event_index": 0,
    }]
    assert available["referenced_segment_ids"] == ["0_10"]
    assert available["retrieved_segment_ids"] == ["0_10"]
    assert available["final_response"]["finish_arguments"] == {"answer": "B"}
    unavailable = qas[1]["baseline_trajectory"]
    assert unavailable["availability"] == "unavailable"
    assert unavailable["source"] is None
    assert unavailable["assistant_steps"] == []
    assert unavailable["tool_events"] == []
    assert unavailable["final_response"] is None
    assert unavailable["projection_statistics"]["raw_character_count"] == 0


def test_agent_boilerplate_and_duplicate_raw_wrappers_are_absent(tmp_path):
    bundle, _, request = compact_request(tmp_path)
    raw_path = Path(bundle["baseline_qas"]).parent / "qa" / "0" / \
        "trajectory.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines()
                if line.strip()]
    payload_text = dumps_canonical(request.user_payload)
    assert raw_rows[0]["content"] not in payload_text
    assert raw_rows[1]["content"] not in payload_text
    assert '"tool_calls"' not in payload_text
    assert '"tool_call_id"' not in payload_text
    assert "Fixture agent system prompt and tool definitions." not in payload_text


def test_provider_metadata_and_multi_tool_wrappers_are_deduplicated(tmp_path):
    bundle = legacy_bundle(tmp_path)
    episode = convert(bundle)
    trajectory_path = Path(episode.qa_outcomes[0].baseline_trajectory_ref)
    rows = [json.loads(line) for line in trajectory_path.read_text().splitlines()
            if line.strip()]
    tool_call_row = next(row for row in rows if row.get("tool_calls") and
                         row["tool_calls"][0]["function"]["name"] != "finish")
    tool_call_row["annotations"] = []
    tool_call_row["refusal"] = None
    tool_call_row["tool_calls"].append({
        "id": "baseline-extra-call",
        "type": "function",
        "function": {
            "name": "clip_search_tool",
            "arguments": json.dumps({"top_k": 8}),
        },
    })
    first_tool_index = next(index for index, row in enumerate(rows)
                            if row.get("role") == "tool")
    rows.insert(first_tool_index + 1, {
        "role": "tool", "name": "clip_search_tool",
        "tool_call_id": "baseline-extra-call",
        "content": "second exact returned evidence",
    })
    trajectory_path.write_text("".join(
        dumps_canonical(row) + "\n" for row in rows))
    events_path = trajectory_path.with_name("tool_events.jsonl")
    events = [json.loads(line) for line in events_path.read_text().splitlines()
              if line.strip()]
    events.append({
        "tool": "clip_search_tool", "args": {"top_k": 8},
        "hits": [], "n_hits": 0, "error": None,
    })
    events_path.write_text("".join(
        dumps_canonical(row) + "\n" for row in events))

    request = build_compact_episode_feedback_request(
        episode, artifact_resolver=LegacyEpisodeFeedbackArtifactResolver())
    trajectory = request.user_payload["episode"]["qas"][0][
        "baseline_trajectory"]

    assert len(trajectory["tool_events"]) == 2
    assert trajectory["tool_events"][1]["returned_evidence"] == [
        "second exact returned evidence"]
    assert trajectory["unclassified_messages"] == []
    assert trajectory["projection_statistics"][
        "excluded_duplicate_wrapper_count"] == 4


def test_unrecognized_message_is_exactly_preserved_not_silently_dropped(tmp_path):
    bundle = legacy_bundle(tmp_path)
    episode = convert(bundle)
    trajectory_path = Path(episode.qa_outcomes[0].baseline_trajectory_ref)
    unknown = {"role": "developer", "content": {"exact": [1, 2, 3]}}
    trajectory_path.write_text(
        trajectory_path.read_text() + dumps_canonical(unknown) + "\n")
    request = build_compact_episode_feedback_request(
        episode, artifact_resolver=LegacyEpisodeFeedbackArtifactResolver())
    projected = request.user_payload["episode"]["qas"][0][
        "baseline_trajectory"]
    assert projected["unclassified_messages"][-1]["message"] == unknown
    assert projected["projection_statistics"][
        "unclassified_message_count"] == 1


def test_audit_source_references_and_hashes_are_exact(tmp_path):
    _, episode, request = compact_request(tmp_path)
    trajectory = request.user_payload["episode"]["qas"][0][
        "baseline_trajectory"]
    source = trajectory["source"]
    ref = Path(episode.qa_outcomes[0].baseline_trajectory_ref)
    tool_ref = ref.with_name("tool_events.jsonl")
    assert source["trajectory_ref"] == str(ref.resolve())
    assert source["trajectory_sha256"] == hashlib.sha256(
        ref.read_bytes()).hexdigest()
    assert source["tool_events_ref"] == str(tool_ref.resolve())
    assert source["tool_events_sha256"] == hashlib.sha256(
        tool_ref.read_bytes()).hexdigest()
    assert source["reference_sets_sha256"] == sha256_json(
        trajectory["reference_sets"])
    assert source["reference_evidence_sha256"] == sha256_json(
        trajectory["reference_evidence"])


def test_complete_and_compact_schema_versions_and_sizes_are_distinct(tmp_path):
    bundle = legacy_bundle(tmp_path)
    episode = convert(bundle)
    resolver = LegacyEpisodeFeedbackArtifactResolver()
    complete = build_episode_feedback_request(
        episode, artifact_resolver=resolver)
    compact = build_compact_episode_feedback_request(
        episode, artifact_resolver=resolver)
    assert complete.user_payload["schema_version"] == \
        EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION
    assert compact.user_payload["schema_version"] == \
        COMPACT_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION
    assert compact.size_statistics.complete_payload_hash == complete.payload_hash
    assert compact.payload_hash != complete.payload_hash
    assert compact.size_statistics.complete_request_character_count == len(
        dumps_canonical(complete.messages))
    assert compact.size_statistics.compact_request_character_count == len(
        dumps_canonical(compact.messages))
    assert compact.size_statistics.compact_request_character_count < \
        compact.size_statistics.complete_request_character_count


def test_compact_size_statistics_match_actual_serialization(tmp_path):
    _, _, request = compact_request(
        tmp_path, episode_transform=overlapping_episode)
    stats = request.size_statistics
    episode_payload = request.user_payload["episode"]
    assert stats.history_catalog_character_count == len(dumps_canonical(
        episode_payload["history_catalog"]))
    assert stats.history_reference_character_count == sum(
        len(dumps_canonical(clip["history_item_ids"]))
        for clip in episode_payload["clips"])
    trajectories = [
        qa[side] for qa in episode_payload["qas"]
        for side in ("baseline_trajectory", "intervention_trajectory")]
    assert stats.compact_trajectory_character_count == sum(
        len(dumps_canonical(item)) for item in trajectories)
    assert stats.raw_trajectory_character_count == sum(
        item["projection_statistics"]["raw_character_count"]
        for item in trajectories)
    assert stats.removed_system_tool_schema_character_count is None
    assert stats.removed_duplicate_wrapper_character_count is None
    assert stats.repeated_prompt_delta_character_count_removed > 0


def test_compact_request_uses_projection_aware_system_instruction(tmp_path):
    _, _, request = compact_request(tmp_path)
    assert request.system_instruction == \
        COMPACT_EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
    assert "history_catalog" in request.system_instruction
    assert "omit agent boilerplate" in request.system_instruction
    assert "Do not infer that omitted boilerplate was evidence" in \
        request.system_instruction


def test_compact_request_is_compatible_with_existing_strict_parser(tmp_path):
    bundle = legacy_bundle(tmp_path)
    episode = convert(bundle)
    backend = FakeBackend()
    result = LLMEpisodeFeedbackGenerator(
        response_provider=backend,
        artifact_resolver=LegacyEpisodeFeedbackArtifactResolver(),
        policy_version=POLICY_VERSION,
        request_representation="compact").generate_with_trace(episode)
    assert result.feedback.episode_id == episode.episode_id
    assert result.request.user_payload["schema_version"] == \
        COMPACT_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION
    assert backend.calls == [(
        result.request.system_instruction, result.request.user_request)]


def test_compact_build_does_not_mutate_episode_or_source_bundle(tmp_path):
    bundle = legacy_bundle(tmp_path)
    episode = convert(bundle)
    before_episode = dumps_canonical(episode)
    files = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    before_files = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    build_compact_episode_feedback_request(
        episode, artifact_resolver=LegacyEpisodeFeedbackArtifactResolver())
    after_files = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    assert dumps_canonical(episode) == before_episode
    assert after_files == before_files
