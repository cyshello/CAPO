"""Checkpoint D1.6 model-visible/audit request separation tests."""

import copy
import dataclasses
import hashlib
import json
from pathlib import Path

from surrogate_rollout.optimization.llm_episode_feedback import (
    EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION,
    MODEL_COMPACT_EPISODE_FEEDBACK_AUDIT_SCHEMA_VERSION,
    MODEL_COMPACT_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION,
    LLMEpisodeFeedbackGenerator,
    LegacyEpisodeFeedbackArtifactResolver,
    build_compact_episode_feedback_request,
    build_episode_feedback_request,
    build_model_compact_episode_feedback_request,
    reconstruct_compact_history_snapshots,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json
from test_compact_episode_feedback import overlapping_episode
from test_legacy_intervention_adapter import convert, legacy_bundle
from test_llm_episode_feedback import FakeBackend, POLICY_VERSION


def model_request(tmp_path, *, episode_transform=None):
    bundle = legacy_bundle(tmp_path)
    episode = convert(bundle)
    if episode_transform is not None:
        episode = episode_transform(episode)
    request = build_model_compact_episode_feedback_request(
        episode, artifact_resolver=LegacyEpisodeFeedbackArtifactResolver())
    return bundle, episode, request


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_default_remains_complete_and_model_compact_requires_explicit_choice(
        tmp_path):
    episode = convert(legacy_bundle(tmp_path))
    default_result = LLMEpisodeFeedbackGenerator(
        response_provider=FakeBackend(),
        artifact_resolver=LegacyEpisodeFeedbackArtifactResolver(),
        policy_version=POLICY_VERSION,
    ).generate_with_trace(episode)
    model_result = LLMEpisodeFeedbackGenerator(
        response_provider=FakeBackend(),
        artifact_resolver=LegacyEpisodeFeedbackArtifactResolver(),
        policy_version=POLICY_VERSION,
        request_representation="model_compact",
    ).generate_with_trace(episode)
    assert default_result.request.user_payload["schema_version"] == \
        EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION
    assert model_result.request.model_payload["schema_version"] == \
        MODEL_COMPACT_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION


def test_model_payload_preserves_all_clips_qas_histories_and_delta_once(tmp_path):
    _, episode, request = model_request(
        tmp_path, episode_transform=overlapping_episode)
    payload = request.model_payload["episode"]
    assert [row["segment_id"] for row in payload["clips"]] == [
        clip.segment_id for clip in episode.clips]
    assert [row["qa_id"] for row in payload["qas"]] == [
        qa.qa_id for qa in episode.qa_outcomes]
    assert payload["qa_transition_summary"] == {
        "correct_to_correct": ["benchmark/train/2"],
        "wrong_to_wrong": ["benchmark/train/3"],
        "wrong_to_correct": ["benchmark/train/1"],
        "correct_to_wrong": [],
    }
    assert payload["history_catalog"]
    assert payload["clips"][0]["history_item_ids"] == [
        "history-a", "history-a"]
    assert payload["prompt_delta"]["instruction"] == \
        episode.prompt_delta.instruction
    assert dumps_canonical(request.model_payload).count(
        episode.prompt_delta.instruction) == 1


def test_history_reconstruction_uses_separate_audit_and_is_exact(tmp_path):
    _, episode, request = model_request(
        tmp_path, episode_transform=overlapping_episode)
    reconstructed = reconstruct_compact_history_snapshots(
        request.model_payload, request.audit_metadata)
    assert [dumps_canonical(item) for item in reconstructed] == [
        dumps_canonical(clip.history_snapshot) for clip in episode.clips]
    assert all("history_snapshot_metadata" not in clip
               for clip in request.model_payload["episode"]["clips"])


def test_model_trajectory_keeps_execution_evidence_without_audit_fields(tmp_path):
    _, _, request = model_request(tmp_path)
    trajectory = request.model_payload["episode"]["qas"][0][
        "baseline_trajectory"]
    assert trajectory["tool_events"][0]["args"] == {"top_k": 16}
    assert trajectory["tool_events"][0]["returned_segment_ids"] == ["0_10"]
    assert trajectory["tool_events"][0]["returned_evidence"] == [
        "baseline evidence for 0_10"]
    assert trajectory["referenced_segment_ids"] == ["0_10"]
    assert trajectory["retrieved_segment_ids"] == ["0_10"]
    assert "reference_sets" not in trajectory
    assert "reference_evidence" not in trajectory
    trajectory_audit = request.audit_metadata["trajectory_projection"][0][
        "baseline_trajectory"]
    assert trajectory_audit["reference_sets"] == {
        "explicitly_cited_segments": ["0_10"],
        "retrieved_segments": ["0_10"],
    }
    assert trajectory_audit["reference_evidence"] == [{
        "segment": "0_10", "set": "retrieved_segments",
        "reason": "clip_search_tool_hit", "event_index": 0,
    }]
    assert trajectory["final_response"]["finish_arguments"] == {"answer": "B"}
    assert "source" not in trajectory
    assert "projection_statistics" not in trajectory
    assert "event_index" not in trajectory["tool_events"][0]
    assert "message_index" not in trajectory["final_response"]


def test_model_payload_contains_no_paths_hashes_or_history_debug_duplicates(
        tmp_path):
    bundle, _, request = model_request(tmp_path)
    keys = set(_all_keys(request.model_payload))
    assert not {"source", "projection_statistics", "serialized_history",
                "history_hash", "preceding_captions",
                "preceding_segment_ids", "history"} & keys
    assert not any(key.endswith("_sha256") for key in keys)
    payload_text = dumps_canonical(request.model_payload)
    assert str(Path(bundle["result"]).resolve()) not in payload_text
    assert str(Path(bundle["baseline_manifest"]).resolve()) not in payload_text


def test_audit_metadata_contains_source_references_hashes_and_projection_data(
        tmp_path):
    _, episode, request = model_request(tmp_path)
    audit = request.audit_metadata
    assert audit["schema_version"] == \
        MODEL_COMPACT_EPISODE_FEEDBACK_AUDIT_SCHEMA_VERSION
    for name, path in (("baseline_run", episode.baseline_run_ref),
                       ("intervention_run", episode.intervention_run_ref)):
        source = audit["source_artifacts"][name]
        assert source["path"] == str(Path(path).resolve())
        assert source["sha256"] == hashlib.sha256(
            Path(path).read_bytes()).hexdigest()
    trajectory_audit = audit["trajectory_projection"][0][
        "baseline_trajectory"]
    assert trajectory_audit["source"]["trajectory_ref"] == str(Path(
        episode.qa_outcomes[0].baseline_trajectory_ref).resolve())
    assert trajectory_audit["projection_statistics"]["raw_message_count"] == 5


def test_unknown_trajectory_messages_remain_model_visible(tmp_path):
    bundle = legacy_bundle(tmp_path)
    episode = convert(bundle)
    trajectory_path = Path(episode.qa_outcomes[0].baseline_trajectory_ref)
    unknown = {"role": "developer", "content": {"exact": [1, 2, 3]}}
    trajectory_path.write_text(
        trajectory_path.read_text() + dumps_canonical(unknown) + "\n")
    request = build_model_compact_episode_feedback_request(
        episode, artifact_resolver=LegacyEpisodeFeedbackArtifactResolver())
    projected = request.model_payload["episode"]["qas"][0][
        "baseline_trajectory"]
    assert projected["unclassified_messages"][-1] == unknown
    assert request.audit_metadata["trajectory_projection"][0][
        "baseline_trajectory"]["projection_debug"][
            "unclassified_message_indices"][-1] == 5


def test_audit_only_change_does_not_change_model_payload_hash(tmp_path):
    _, _, request = model_request(tmp_path)
    changed_audit = copy.deepcopy(request.audit_metadata)
    changed_audit["source_artifacts"]["baseline_run"]["path"] = \
        "/audit-only/relocated.json"
    changed = dataclasses.replace(
        request,
        audit_metadata=changed_audit,
        audit_metadata_hash=sha256_json(changed_audit),
    )
    assert changed.model_payload_hash == request.model_payload_hash
    assert changed.audit_metadata_hash != request.audit_metadata_hash


def test_semantic_change_changes_model_payload_hash(tmp_path):
    bundle = legacy_bundle(tmp_path)
    episode = convert(bundle)
    resolver = LegacyEpisodeFeedbackArtifactResolver()
    original = build_model_compact_episode_feedback_request(
        episode, artifact_resolver=resolver)
    clips = (dataclasses.replace(
        episode.clips[0], intervention_caption="semantically changed caption"),
        *episode.clips[1:])
    changed = build_model_compact_episode_feedback_request(
        dataclasses.replace(episode, clips=clips), artifact_resolver=resolver)
    assert changed.model_payload_hash != original.model_payload_hash


def test_fake_backend_receives_only_canonical_model_payload(tmp_path):
    episode = convert(legacy_bundle(tmp_path))
    backend = FakeBackend()
    result = LLMEpisodeFeedbackGenerator(
        response_provider=backend,
        artifact_resolver=LegacyEpisodeFeedbackArtifactResolver(),
        policy_version=POLICY_VERSION,
        request_representation="model_compact",
    ).generate_with_trace(episode)
    sent = json.loads(backend.calls[0][1])
    assert sent == result.request.model_payload
    assert backend.calls[0][1] == dumps_canonical(result.request.model_payload)
    assert "audit_metadata" not in sent
    identity = {
        "episode_id": episode.episode_id,
        "feedback_policy_version": POLICY_VERSION,
        "request_payload_hash": result.request.model_payload_hash,
    }
    assert result.feedback.feedback_id == \
        "episode_feedback_" + sha256_json(identity)[:20]


def test_size_statistics_and_three_representations_are_exact(tmp_path):
    bundle = legacy_bundle(tmp_path)
    episode = convert(bundle)
    resolver = LegacyEpisodeFeedbackArtifactResolver()
    complete = build_episode_feedback_request(
        episode, artifact_resolver=resolver)
    compact = build_compact_episode_feedback_request(
        episode, artifact_resolver=resolver)
    model = build_model_compact_episode_feedback_request(
        episode, artifact_resolver=resolver)
    stats = model.size_statistics
    assert stats.complete_request_character_count == len(
        dumps_canonical(complete.messages))
    assert stats.compact_request_character_count == len(
        dumps_canonical(compact.messages))
    assert stats.model_request_character_count == len(
        dumps_canonical(model.messages))
    assert stats.model_payload_character_count == len(
        dumps_canonical(model.model_payload))
    assert stats.audit_metadata_character_count == len(
        dumps_canonical(model.audit_metadata))
    assert stats.model_payload_hash == sha256_json(model.model_payload)
    assert stats.audit_metadata_hash == sha256_json(model.audit_metadata)
    assert stats.model_request_character_count < \
        stats.compact_request_character_count < \
        stats.complete_request_character_count


def test_build_is_deterministic_and_does_not_mutate_episode_or_sources(tmp_path):
    bundle = legacy_bundle(tmp_path)
    episode = convert(bundle)
    before_episode = dumps_canonical(episode)
    files = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    before_files = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    resolver = LegacyEpisodeFeedbackArtifactResolver()
    first = build_model_compact_episode_feedback_request(
        episode, artifact_resolver=resolver)
    second = build_model_compact_episode_feedback_request(
        episode, artifact_resolver=resolver)
    after_files = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    assert first.user_request == second.user_request
    assert first.model_payload_hash == second.model_payload_hash
    assert first.audit_metadata_hash == second.audit_metadata_hash
    assert dumps_canonical(episode) == before_episode
    assert after_files == before_files
