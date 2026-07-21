"""Checkpoint B read-only adapter tests over a synthetic legacy v1 bundle.

The fixture is created under pytest ``tmp_path`` and mirrors the minimum fields
written by the real baseline/property-intervention artifact writers.  It
contains no frames, trajectory payload content, model calls, or run state.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from surrogate_rollout.optimization.legacy_intervention_adapter import (
    EPISODE_IDENTITY_VERSION,
    LegacyInterventionConversionError,
    legacy_property_intervention_to_episode,
)
from surrogate_rollout.optimization.schemas import (
    InterventionClipRecord,
    intervention_episode_from_json,
)
from surrogate_rollout.prompt_routing.schemas import (
    ComposedCaptionPrompt,
    CompositionTrace,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_json, sha256_text


CANDIDATE_ID = "candidate-stable-001"
SOURCE_QA_ID = "benchmark/train/1"
SIBLING_QA_IDS = (SOURCE_QA_ID, "benchmark/train/2", "benchmark/train/3")
INSTRUCTION = "Describe the visible handoff and resulting object ownership."
DIAGNOSIS = "The baseline captions omitted a visible object handoff."
SEGMENT_IDS = ("0_10", "10_20")


def write_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return str(path.resolve())


def write_jsonl(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(dumps_canonical(row) + "\n" for row in rows), encoding="utf-8")
    return str(path.resolve())


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rewrite_json(path: str, transform) -> None:
    value = read_json(path)
    transform(value)
    Path(path).write_text(
        json.dumps(value, sort_keys=True), encoding="utf-8")


def legacy_bundle(tmp_path: Path) -> dict[str, str]:
    baseline_dir = tmp_path / "baseline" / "video-1"
    intervention_dir = tmp_path / "intervention" / CANDIDATE_ID

    prompts = []
    histories = []
    baseline_captions = {}
    mixed_captions = {}
    preceding = []
    for segment_id in ("0_10", "10_20", "20_30"):
        text = f"Exact incumbent composed prompt for {segment_id}."
        prompts.append(ComposedCaptionPrompt(
            video_id="video-1", segment_id=segment_id,
            bank_version="bank_v0001", router_version="router_v0001",
            scaffold_version="scaffold_v0001", contract_version="contract_v0001",
            selected_prompt_ids=(), prompt_text=text,
            prompt_hash=sha256_text(text),
            composition_trace=CompositionTrace(
                selected_prompt_ids=(), preserved_prompt_ids=()),
        ))
        history = {
            "schema_version": "frozen_local_caption_history_v2",
            "segment_id": segment_id,
            "block_index": 0,
            "block_start_seconds": 0.0,
            "block_end_seconds": 60.0,
            "max_history_captions": 64,
            "preceding_captions": list(preceding),
            "preceding_segment_ids": [item["segment_id"] for item in preceding],
            "history": list(preceding),
            "source": "sequential_history_aware_baseline",
        }
        serialized_history = dumps_canonical({
            key: history[key] for key in (
                "schema_version", "block_index", "block_start_seconds",
                "block_end_seconds", "max_history_captions",
                "preceding_captions")})
        history["serialized_history"] = serialized_history
        history["history_hash"] = sha256_text(serialized_history)
        if segment_id in SEGMENT_IDS:
            histories.append(history)
        baseline_caption = f"baseline caption {segment_id}"
        baseline_captions[segment_id] = {"caption": baseline_caption}
        mixed_captions[segment_id] = {
            "caption": (f"intervention caption {segment_id}"
                        if segment_id in SEGMENT_IDS else baseline_caption)}
        preceding.append({"segment_id": segment_id, "caption": baseline_caption})

    composed_path = write_jsonl(
        baseline_dir / "routing" / "composed_prompts.jsonl", prompts)
    routing_path = write_json(
        baseline_dir / "routing" / "routing_manifest.json",
        {"composed_prompts_path": composed_path})
    captions_path = write_json(
        baseline_dir / "captions.json", baseline_captions)
    proposal = {
        "candidate_property_id": CANDIDATE_ID,
        "suggested_property_id": "readable-legacy-name",
        "property_text": INSTRUCTION,
        "source_video_id": "video-1",
        "source_question_ids": [SOURCE_QA_ID],
        "proposal_rationale": DIAGNOSIS,
        "motivating_failure_types": ["legacy_fixture_failure"],
        "coverage_hints": [],
        "proposer_policy_version": "legacy_fixture_v1",
    }
    proposal_path = write_json(
        baseline_dir / "property_proposals.json",
        {"video_id": "video-1", "proposals": [proposal]})

    baseline_qas = []
    transition_qas = []
    for index, qa_id in enumerate(SIBLING_QA_IDS):
        baseline_answer = ("B", "A", "C")[index]
        intervention_answer = ("A", "A", "B")[index]
        baseline_correct = (False, True, False)[index]
        intervention_correct = (True, True, False)[index]
        referenced_segment = SEGMENT_IDS[index % len(SEGMENT_IDS)]
        if index == 1:
            baseline_trajectory = None
            intervention_trajectory = None
            baseline_reference_sets = {}
            intervention_reference_sets = {}
        else:
            baseline_trajectory = write_jsonl(
                baseline_dir / "qa" / str(index) / "trajectory.jsonl", ({
                    "role": "system",
                    "content": (
                        "Fixture agent system prompt and tool definitions. " * 30),
                }, {
                    "role": "user",
                    "content": (
                        "Generic stored QA instructions and tool usage guidance. " * 20
                        + f"Stored question {index}? A. first B. second C. third"),
                }, {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"baseline-call-{index}",
                        "type": "function",
                        "function": {"name": "clip_search_tool",
                                     "arguments": "{\"top_k\": 16}"},
                    }],
                }, {
                    "role": "tool",
                    "name": "clip_search_tool",
                    "tool_call_id": f"baseline-call-{index}",
                    "content": f"baseline evidence for {referenced_segment}",
                }, {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"baseline-finish-{index}",
                        "type": "function",
                        "function": {"name": "finish",
                                     "arguments": json.dumps({
                                         "answer": baseline_answer})},
                    }],
                }))
            intervention_trajectory = write_jsonl(
                intervention_dir / "qa" / str(index) / "trajectory.jsonl", ({
                    "role": "system",
                    "content": (
                        "Fixture agent system prompt and tool definitions. " * 30),
                }, {
                    "role": "user",
                    "content": (
                        "Generic stored QA instructions and tool usage guidance. " * 20
                        + f"Stored question {index}? A. first B. second C. third"),
                }, {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"intervention-call-{index}",
                        "type": "function",
                        "function": {"name": "clip_search_tool",
                                     "arguments": "{\"top_k\": 16}"},
                    }],
                }, {
                    "role": "tool",
                    "name": "clip_search_tool",
                    "tool_call_id": f"intervention-call-{index}",
                    "content": f"intervention evidence for {referenced_segment}",
                }, {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"intervention-finish-{index}",
                        "type": "function",
                        "function": {"name": "finish",
                                     "arguments": json.dumps({
                                         "answer": intervention_answer})},
                    }],
                }))
            baseline_reference_sets = {
                "explicitly_cited_segments": [referenced_segment],
                "retrieved_segments": [referenced_segment],
            }
            intervention_reference_sets = {
                "explicitly_cited_segments": [referenced_segment],
                "retrieved_segments": [referenced_segment],
            }
            reference_evidence = [{
                    "segment": referenced_segment,
                    "set": "retrieved_segments",
                    "reason": "clip_search_tool_hit",
                    "event_index": 0,
                }]
            write_json(
                Path(baseline_trajectory).parent / "references.json", {
                    **baseline_reference_sets, "evidence": reference_evidence})
            write_json(
                Path(intervention_trajectory).parent / "references.json", {
                    **intervention_reference_sets,
                    "evidence": reference_evidence})
            write_jsonl(
                Path(baseline_trajectory).parent / "tool_events.jsonl", ({
                    "tool": "clip_search_tool",
                    "args": {"top_k": 16},
                    "hits": [{"time_start_secs": float(index * 10),
                              "time_end_secs": float(index * 10 + 10)}],
                    "n_hits": 1,
                    "error": None,
                },))
            write_jsonl(
                Path(intervention_trajectory).parent / "tool_events.jsonl", ({
                    "tool": "clip_search_tool",
                    "args": {"top_k": 16},
                    "hits": [{"time_start_secs": float(index * 10),
                              "time_end_secs": float(index * 10 + 10)}],
                    "n_hits": 1,
                    "error": None,
                },))
        baseline_qas.append({
            "question_id": qa_id,
            "question": f"Stored question {index}?",
            "options": ["A. first", "B. second", "C. third"],
            "ground_truth": "A",
            "prediction": baseline_answer,
            "is_correct": baseline_correct,
            "trajectory_path": baseline_trajectory,
            "reference_sets": baseline_reference_sets,
        })
        transition_qas.append({
            "question_id": qa_id,
            "ground_truth": "A",
            "baseline_prediction": baseline_answer,
            "candidate_prediction": intervention_answer,
            "baseline_correct": baseline_correct,
            "candidate_correct": intervention_correct,
            "transition": (
                "wrong_to_correct" if index == 0 else
                "correct_to_correct" if index == 1 else "wrong_to_wrong"),
            "trajectory_path": intervention_trajectory,
            "reference_sets": intervention_reference_sets,
        })
    baseline_qas_path = write_jsonl(
        baseline_dir / "baseline_qas.jsonl", baseline_qas)
    baseline_manifest_path = write_json(
        baseline_dir / "video_complete.json", {
            "video_id": "video-1",
            "captions_path": captions_path,
            "baseline_qas_path": baseline_qas_path,
            "routing_manifest_path": routing_path,
            "property_proposal_path": proposal_path,
        })

    histories_path = write_jsonl(
        intervention_dir / "frozen_histories.jsonl", histories)
    mixed_captions_path = write_json(
        intervention_dir / "mixed_view" / "captions.json", mixed_captions)
    transitions_path = write_json(
        intervention_dir / "transitions.json", {
            "schema_version": "property_intervention_transitions_v1",
            "candidate_property_id": CANDIDATE_ID,
            "source_video_id": "video-1",
            "transition_counts": {
                "wrong_to_correct": 1, "correct_to_correct": 1,
                "wrong_to_wrong": 1, "correct_to_wrong": 0},
            "qas": transition_qas,
        })
    result_path = write_json(
        intervention_dir / "result.json", {
            "schema_version": "property_intervention_result_v1",
            "status": "completed",
            "input_fingerprint": "intervention-fingerprint-001",
            "candidate_property_id": CANDIDATE_ID,
            "source_video_id": "video-1",
            "parent_baseline_identity": "baseline-identity-001",
            "selected_segment_ids": list(SEGMENT_IDS),
            "mixed_captions_path": mixed_captions_path,
            "transitions_path": transitions_path,
        })
    return {
        "baseline_manifest": baseline_manifest_path,
        "result": result_path,
        "histories": histories_path,
        "mixed_captions": mixed_captions_path,
        "transitions": transitions_path,
        "proposal": proposal_path,
        "baseline_qas": baseline_qas_path,
        "composed_prompts": composed_path,
    }


def convert(bundle):
    return legacy_property_intervention_to_episode(
        intervention_result_path=bundle["result"],
        baseline_video_manifest_path=bundle["baseline_manifest"],
        parent_meta_prompt_id="meta-parent-001")


def test_valid_legacy_artifact_converts_to_one_episode_with_exact_mapping(tmp_path):
    bundle = legacy_bundle(tmp_path)
    episode = convert(bundle)

    assert episode.video_id == "video-1"
    assert episode.parent_meta_prompt_id == "meta-parent-001"
    assert episode.prompt_delta.delta_id == CANDIDATE_ID
    assert episode.prompt_delta.instruction == INSTRUCTION
    assert episode.prompt_delta.source_qa_ids == (SOURCE_QA_ID,)
    assert episode.prompt_delta.proposer_diagnosis == DIAGNOSIS
    assert tuple(clip.segment_id for clip in episode.clips) == SEGMENT_IDS
    assert episode.clips[1].time_range == {
        "start_seconds": 10.0, "end_seconds": 20.0}
    assert episode.clips[1].history_snapshot["history"][0]["caption"] == \
        "baseline caption 0_10"
    assert episode.clips[1].base_prompt == \
        "Exact incumbent composed prompt for 10_20."
    assert episode.clips[1].baseline_caption == "baseline caption 10_20"
    assert episode.clips[1].intervention_caption == \
        "intervention caption 10_20"
    assert all(clip.prompt_delta.instruction == episode.prompt_delta.instruction
               for clip in episode.clips)


def test_all_saved_sibling_qa_outcomes_and_null_trajectories_are_preserved(tmp_path):
    episode = convert(legacy_bundle(tmp_path))
    assert tuple(item.qa_id for item in episode.qa_outcomes) == SIBLING_QA_IDS
    assert tuple(item.is_source_qa for item in episode.qa_outcomes) == (
        True, False, False)
    unavailable = episode.qa_outcomes[1]
    assert unavailable.baseline_trajectory_ref is None
    assert unavailable.intervention_trajectory_ref is None
    assert episode.qa_outcomes[0].baseline_answer == "B"
    assert episode.qa_outcomes[0].intervention_answer == "A"
    assert episode.qa_outcomes[0].baseline_correct is False
    assert episode.qa_outcomes[0].intervention_correct is True


def test_qa_transition_is_not_attributed_to_clip(tmp_path):
    episode = convert(legacy_bundle(tmp_path))
    clip_fields = {field.name for field in dataclasses.fields(
        InterventionClipRecord)}
    assert not clip_fields.intersection({
        "transition", "qa_transition", "qa_outcome", "qa_outcomes"})
    assert len(episode.qa_outcomes) == 3


def test_missing_required_clip_source_field_fails_fast(tmp_path):
    bundle = legacy_bundle(tmp_path)
    rewrite_json(bundle["mixed_captions"], lambda value: value["0_10"].pop(
        "caption"))
    with pytest.raises(LegacyInterventionConversionError, match="caption"):
        convert(bundle)


def test_missing_stable_candidate_id_fails_fast(tmp_path):
    bundle = legacy_bundle(tmp_path)
    rewrite_json(bundle["result"], lambda value: value.pop(
        "candidate_property_id"))
    with pytest.raises(LegacyInterventionConversionError, match="candidate_property_id"):
        convert(bundle)


def test_duplicate_segment_id_rejected(tmp_path):
    bundle = legacy_bundle(tmp_path)
    rewrite_json(bundle["result"], lambda value: value[
        "selected_segment_ids"].append("0_10"))
    with pytest.raises(LegacyInterventionConversionError, match="duplicate segment ID"):
        convert(bundle)


def test_duplicate_qa_id_rejected(tmp_path):
    bundle = legacy_bundle(tmp_path)

    def duplicate(value):
        value["qas"].append(dict(value["qas"][0]))

    rewrite_json(bundle["transitions"], duplicate)
    with pytest.raises(LegacyInterventionConversionError, match="duplicate question_id"):
        convert(bundle)


def test_missing_source_qa_outcome_is_reported_not_fabricated(tmp_path):
    bundle = legacy_bundle(tmp_path)
    rewrite_json(bundle["transitions"], lambda value: value.update({
        "qas": [row for row in value["qas"]
                if row["question_id"] != SOURCE_QA_ID]}))
    with pytest.raises(LegacyInterventionConversionError, match="will not fabricate"):
        convert(bundle)


def test_unresolved_trajectory_reference_rejected(tmp_path):
    bundle = legacy_bundle(tmp_path)

    def break_trajectory(value):
        value["qas"][0]["trajectory_path"] = "missing/trajectory.jsonl"

    rewrite_json(bundle["transitions"], break_trajectory)
    with pytest.raises(LegacyInterventionConversionError, match="unresolved"):
        convert(bundle)


def test_repeated_conversion_has_same_episode_id_and_canonical_json(tmp_path):
    bundle = legacy_bundle(tmp_path)
    first = convert(bundle)
    second = convert(bundle)
    expected_identity = {
        "schema_version": EPISODE_IDENTITY_VERSION,
        "candidate_property_id": CANDIDATE_ID,
        "parent_baseline_identity": "baseline-identity-001",
        "intervention_input_fingerprint": "intervention-fingerprint-001",
    }
    assert first.episode_id == (
        f"legacy_episode_{sha256_json(expected_identity)[:20]}")
    assert first.episode_id == second.episode_id
    assert dumps_canonical(first) == dumps_canonical(second)


def test_different_intervention_identity_has_different_episode_id(tmp_path):
    bundle = legacy_bundle(tmp_path)
    first = convert(bundle)
    rewrite_json(bundle["result"], lambda value: value.update({
        "input_fingerprint": "intervention-fingerprint-002"}))
    second = convert(bundle)
    assert first.episode_id != second.episode_id


def test_conversion_does_not_mutate_legacy_bundle(tmp_path):
    bundle = legacy_bundle(tmp_path)
    before = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in bundle.values()
    }
    convert(bundle)
    after = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in bundle.values()
    }
    assert after == before


def test_converted_episode_json_round_trip(tmp_path):
    episode = convert(legacy_bundle(tmp_path))
    restored = intervention_episode_from_json(
        json.loads(dumps_canonical(episode)))
    assert restored == episode
    assert dumps_canonical(restored) == dumps_canonical(episode)
