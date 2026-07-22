"""Grounded compact feedback memory and cumulative bank tests."""

import dataclasses
import json
from pathlib import Path

import pytest

import scripts.migrate_episode_feedback_memory as migration
from scripts.migrate_episode_feedback_memory import main as migrate_main
from surrogate_rollout.optimization.episode_feedback import (
    DeterministicMockEpisodeFeedbackGenerator,
)
from surrogate_rollout.optimization.feedback_memory import (
    append_compact_feedback_memory_bank,
    build_compact_feedback_memories,
    build_compact_feedback_updater_projection,
    load_compact_feedback_memory_bank,
    load_representative_full_feedback,
    validate_compact_feedback_memory_record,
    append_parent_feedback_memory_bank,
    build_episode_feedback_memory_record,
    load_parent_feedback_memory_bank,
    select_historical_feedback_memories,
)
from surrogate_rollout.optimization.meta_prompt_updater import (
    DeterministicMockMetaPromptUpdater,
)
from surrogate_rollout.optimization.schemas import (
    CompactFeedbackMemory,
    CompactFeedbackMemoryRecord,
    CompactFeedbackProvenance,
    MetaPromptVersion,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from test_episode_feedback import episode_fixture


def _record(*, attribution="unresolved", effect="wrong_to_correct",
            memory_id="memory-1"):
    return CompactFeedbackMemoryRecord(
        memory_id=memory_id,
        memory=CompactFeedbackMemory(
            runtime_condition="A chart is visible while prior context omits its trend.",
            description_change="Describe the axes, compared values, and trend direction.",
            effect=effect, attribution=attribution),
        provenance=CompactFeedbackProvenance(
            parent_meta_prompt_id="parent-1", iteration_id="iteration-1",
            episode_id="episode-1", video_id="video-1",
            qa_ids=("qa-1",), segment_ids=("10_20",),
            feedback_id="feedback-1"),
        metadata={"source_artifact_refs": {}, "source_artifact_sha256": {}},
    )


def test_word_sentence_markdown_and_causal_limits():
    validate_compact_feedback_memory_record(_record())
    with pytest.raises(ValueError, match="30 words"):
        CompactFeedbackMemory(
            runtime_condition=" ".join(["word"] * 31),
            description_change="Describe visible evidence.",
            effect="wrong_to_correct", attribution="unresolved")
    for text in ("First sentence. Second sentence.", "- list item",
                 "The description caused the answer change."):
        with pytest.raises(ValueError):
            CompactFeedbackMemory(
                runtime_condition="A chart is visible.",
                description_change=text, effect="wrong_to_correct",
                attribution="unresolved")


def test_grounding_wins_and_noop_invalid_are_deterministic():
    episode = episode_fixture()
    feedback = DeterministicMockEpisodeFeedbackGenerator().generate(episode)
    records = build_compact_feedback_memories(
        feedback=feedback, episode=episode, iteration_id="iteration-x")
    assert records
    assert all("caused" not in item.memory.description_change.lower()
               for item in records)
    assert all(item.provenance.qa_ids for item in records)
    # Fixture trajectory references are intentionally unavailable.
    assert {item.memory.attribution for item in records} == {"invalid"}

    contradictory = dataclasses.replace(
        feedback,
        generator_diagnosis="The caption caused the QA improvement.")
    conflicts = build_compact_feedback_memories(
        feedback=contradictory, episode=episode,
        iteration_id="iteration-conflict")
    assert all(
        "feedback uses causal wording without direct trajectory grounding" in
        item.metadata["natural_language_grounding_conflicts"]
        for item in conflicts)
    assert all("caused" not in item.memory.description_change.lower()
               for item in conflicts)

    segment_id = episode.clips[0].segment_id
    id_bearing = dataclasses.replace(
        feedback,
        observations=(dataclasses.replace(
            feedback.observations[0],
            statement=f"Observed a description change in {segment_id}.",
            supporting_segment_ids=(segment_id,),
            supporting_qa_ids=(), evidence_type="caption_change",
            transition_type=None),),
        recommended_strategy_change=(
            f"When reviewing {episode.qa_outcomes[0].qa_id}, inspect context."))
    sanitized = build_compact_feedback_memories(
        feedback=id_bearing, episode=episode,
        iteration_id="iteration-identifiers")
    assert all(segment_id not in item.memory.description_change
               for item in sanitized)
    assert all("description change" in item.memory.description_change
               for item in sanitized)
    assert all("grounded target segment" in item.memory.description_change
               for item in sanitized)
    assert all(episode.qa_outcomes[0].qa_id not in item.memory.runtime_condition
               for item in sanitized)

    unchanged = dataclasses.replace(episode, clips=tuple(
        dataclasses.replace(clip, intervention_caption=clip.baseline_caption)
        for clip in episode.clips))
    no_op_feedback = DeterministicMockEpisodeFeedbackGenerator().generate(unchanged)
    no_op = build_compact_feedback_memories(
        feedback=no_op_feedback, episode=unchanged,
        iteration_id="iteration-noop")
    assert {item.memory.attribution for item in no_op} == {"no_op"}
    assert all("unchanged" in item.memory.description_change for item in no_op)


def test_direct_requires_exact_returned_caption_evidence(tmp_path):
    episode = episode_fixture()
    clip = episode.clips[0]
    outcomes = []
    for index, outcome in enumerate(episode.qa_outcomes):
        refs = []
        for side in ("baseline", "intervention"):
            root = tmp_path / f"{index}_{side}"
            root.mkdir()
            trajectory = root / "trajectory.jsonl"
            trajectory.write_text("{}\n")
            evidence = [clip.intervention_caption] if side == "intervention" else []
            (root / "references.json").write_text(json.dumps({
                "returned_segments": [clip.segment_id],
                "retrieved_segments": [clip.segment_id],
                "evidence": evidence,
            }))
            refs.append(str(trajectory))
        outcomes.append(dataclasses.replace(
            outcome, baseline_trajectory_ref=refs[0],
            intervention_trajectory_ref=refs[1]))
    episode = dataclasses.replace(episode, qa_outcomes=tuple(outcomes))
    feedback = DeterministicMockEpisodeFeedbackGenerator().generate(episode)
    records = build_compact_feedback_memories(
        feedback=feedback, episode=episode, iteration_id="iteration-direct")
    assert {item.memory.attribution for item in records} == {"direct"}
    assert all(not any(term in item.memory.description_change.lower()
                       for term in ("caused", "led to", "corrected"))
               for item in records)


def test_bank_append_dedup_projection_and_full_feedback_lookup(tmp_path):
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(json.dumps({"feedback_id": "feedback-1"}))
    record = dataclasses.replace(_record(), metadata={
        "source_artifact_refs": {"feedback": str(feedback_path)},
        "source_artifact_sha256": {
            "feedback": __import__("hashlib").sha256(
                feedback_path.read_bytes()).hexdigest()},
    })
    first = append_compact_feedback_memory_bank(
        str(tmp_path / "bank"), (record,))
    second = append_compact_feedback_memory_bank(
        str(tmp_path / "bank"), (record,))
    assert first["record_count"] == second["record_count"] == 1
    assert second["added_memory_ids"] == []
    assert load_compact_feedback_memory_bank(str(tmp_path / "bank")) == (record,)
    assert load_representative_full_feedback(record)["feedback_id"] == \
        "feedback-1"

    projection = build_compact_feedback_updater_projection((
        record,
        dataclasses.replace(
            _record(memory_id="memory-2", effect="wrong_to_wrong"),
            provenance=dataclasses.replace(
                _record().provenance, qa_ids=("qa-2",))),
    ))
    assert projection["memories"] == []
    assert projection["aggregates"] == [{
        "attribution": "unresolved", "count": 2,
        "effects": {"wrong_to_correct": 1, "correct_to_wrong": 0,
                    "unchanged": 1, "mixed": 0},
    }]
    assert projection["full_feedback_omitted"] is True


def test_updater_uses_compact_memory_not_full_feedback():
    episode = episode_fixture()
    feedback = DeterministicMockEpisodeFeedbackGenerator().generate(episode)
    memories = build_compact_feedback_memories(
        feedback=feedback, episode=episode, iteration_id="iteration-u")
    parent = MetaPromptVersion(
        "parent", None, "Inspect frames and bounded history.",
        "2026-07-21T00:00:00Z", "parent")
    result = DeterministicMockMetaPromptUpdater().update(
        parent, (feedback,), feedback_memories=memories)
    assert "compact_feedback_memory" in result.request.payload
    assert "feedbacks" not in result.request.payload
    serialized = dumps_canonical(result.request.payload)
    assert feedback.generator_diagnosis not in serialized


def test_provider_authored_episode_memory_is_one_per_episode_and_deduplicated(
        tmp_path):
    episode = episode_fixture()
    feedback = dataclasses.replace(
        DeterministicMockEpisodeFeedbackGenerator().generate(episode),
        compact_memory_text=(
            "Context: A very long, imperfect memory may contain causal words, "
            "Markdown-like text, or multiple sentences.\n"
            "Experience: It is stored without a semantic rewrite because the "
            "provider authored it."))
    record = build_episode_feedback_memory_record(
        feedback=feedback, episode=episode, iteration_id="iteration-1",
        parent_meta_prompt_id="parent-1")
    assert record is not None
    assert record.memory_text == feedback.compact_memory_text
    first = append_parent_feedback_memory_bank(
        str(tmp_path / "bank"), "parent-1", (record,))
    second = append_parent_feedback_memory_bank(
        str(tmp_path / "bank"), "parent-1", (record,))
    assert first["record_count"] == second["record_count"] == 1
    assert second["added_memory_ids"] == []
    assert load_parent_feedback_memory_bank(
        str(tmp_path / "bank"), "parent-1") == (record,)
    assert load_parent_feedback_memory_bank(
        str(tmp_path / "bank"), "another-parent") == ()


def test_historical_selection_excludes_current_and_uses_recent_stable_prefix():
    episode = episode_fixture()
    feedback = DeterministicMockEpisodeFeedbackGenerator().generate(episode)
    old = build_episode_feedback_memory_record(
        feedback=feedback, episode=episode, iteration_id="iteration-old",
        parent_meta_prompt_id="parent-1")
    newer_episode = dataclasses.replace(
        episode, episode_id="episode-new",
        prompt_delta=dataclasses.replace(
            episode.prompt_delta, delta_id="candidate-new"))
    newer_feedback = dataclasses.replace(
        feedback, feedback_id="feedback-new", episode_id="episode-new")
    newer = build_episode_feedback_memory_record(
        feedback=newer_feedback, episode=newer_episode,
        iteration_id="iteration-new", parent_meta_prompt_id="parent-1")
    current_episode = dataclasses.replace(
        episode, episode_id="episode-current",
        prompt_delta=dataclasses.replace(
            episode.prompt_delta, delta_id="candidate-current"))
    current_feedback = dataclasses.replace(
        feedback, feedback_id="feedback-current", episode_id="episode-current")
    current = build_episode_feedback_memory_record(
        feedback=current_feedback, episode=current_episode,
        iteration_id="iteration-current", parent_meta_prompt_id="parent-1")
    selected, manifest = select_historical_feedback_memories(
        (old, newer, current), current_iteration_id="iteration-current",
        maximum_serialized_characters=len(dumps_canonical({
            "iteration_id": newer.iteration_id,
            "episode_id": newer.episode_id,
            "memory_id": newer.memory_id,
            "memory_text": newer.memory_text,
        })))
    assert [item["memory_id"] for item in selected] == [newer.memory_id]
    assert manifest["selected_memory_ids"] == [newer.memory_id]
    assert manifest["excluded_memory_ids"] == [old.memory_id]
    assert current.memory_id not in dumps_canonical(manifest)


def test_migration_is_idempotent_and_sources_are_unchanged(tmp_path, monkeypatch):
    episode = episode_fixture()
    feedback = DeterministicMockEpisodeFeedbackGenerator().generate(episode)
    episode_path = tmp_path / "episode.json"
    feedback_path = tmp_path / "feedback.json"
    episode_path.write_text(dumps_canonical(episode))
    feedback_path.write_text(dumps_canonical(dataclasses.replace(
        feedback, compact_memory_text=None)))
    before = {path: path.read_bytes() for path in (
        episode_path, feedback_path)}
    calls = []

    def fake_call(args, body):
        calls.append(body)
        return dumps_canonical({
            "compact_memory_text": (
                "Context: A chart is visible in bounded context.\n"
                "Experience: Added visible labels with uncertain utility.")
        }), {"usage": {"input_tokens": 1, "output_tokens": 1}}

    monkeypatch.setattr(migration, "_call_once", fake_call)
    args = [
        "--iteration-id", "migration-iteration",
        "--episode-artifact", str(episode_path),
        "--feedback-artifact", str(feedback_path),
        "--memory-bank-dir", str(tmp_path / "bank"),
        "--output-dir", str(tmp_path / "migration"),
        "--provider", "openai_api",
        "--api-endpoint", "https://example.invalid/v1/chat/completions",
        "--model-id", "fixture-model",
        "--temperature", "0",
        "--maximum-output-tokens", "128",
        "--policy-version", "fixture-memory-v1",
        "--timeout-seconds", "10",
    ]
    assert migrate_main(args) == migrate_main(args) == 0
    assert len(calls) == 1
    bank = load_parent_feedback_memory_bank(
        str(tmp_path / "bank"), episode.parent_meta_prompt_id)
    assert len(bank) == 1
    assert all(path.read_bytes() == value for path, value in before.items())
