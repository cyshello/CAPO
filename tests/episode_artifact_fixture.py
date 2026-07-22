import json
from pathlib import Path

from surrogate_rollout.optimization.schemas import (
    InterventionClipRecord,
    InterventionEpisode,
    PromptDelta,
    QAInterventionOutcome,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


def _write_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _write_jsonl(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(dumps_canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return str(path)


def _trajectory(path: Path, segment_id: str = "0_10") -> str:
    _write_jsonl(path.with_name("tool_events.jsonl"), [{
        "tool": "clip_search_tool",
        "args": {"event_description": "fixture query"},
        "hits": [{
            "time_start_secs": 0,
            "time_end_secs": 10,
        }],
    }])
    _write_json(path.with_name("references.json"), {
        "retrieved_segments": [segment_id],
        "returned_segments": [segment_id],
        "frame_inspected_segments": [],
        "explicitly_cited_segments": [],
        "consumed_segments": [segment_id],
    })
    return _write_jsonl(path, [
        {
            "role": "assistant",
            "content": "I should inspect the localized evidence.",
            "tool_calls": [{
                "id": "call-1",
                "function": {
                    "name": "clip_search_tool",
                    "arguments": json.dumps({
                        "event_description": "fixture query"}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "clip_search_tool",
            "content": "baseline evidence for 0_10",
        },
        {"role": "assistant", "content": "Final answer: A"},
    ])


def fresh_episode_bundle(tmp_path: Path) -> InterventionEpisode:
    root = Path(tmp_path)
    delta = PromptDelta(
        delta_id="delta-001",
        instruction="Describe the visible object transfer.",
        source_qa_ids=("benchmark/train/1",),
        proposer_diagnosis="The baseline omitted the transfer.",
    )
    clips = (
        InterventionClipRecord(
            segment_id="0_10",
            time_range={"start_seconds": 0.0, "end_seconds": 10.0},
            history_snapshot={"history": []},
            base_prompt="Base prompt zero.",
            prompt_delta=delta,
            baseline_caption="A person holds an object.",
            intervention_caption="A person passes an object.",
        ),
        InterventionClipRecord(
            segment_id="10_20",
            time_range={"start_seconds": 10.0, "end_seconds": 20.0},
            history_snapshot={"history": [{"segment_id": "0_10"}]},
            base_prompt="Base prompt one.",
            prompt_delta=delta,
            baseline_caption="The person walks away.",
            intervention_caption="The person walks away.",
        ),
    )
    states = (
        ("benchmark/train/1", True, False, True, "B", "A"),
        ("benchmark/train/2", False, True, False, "A", "B"),
        ("benchmark/train/3", False, True, True, "C", "C"),
        ("benchmark/train/4", False, False, False, "D", "C"),
    )
    baseline_rows = []
    qa_records = []
    outcomes = []
    for index, (qa_id, is_source, before, after, base_answer, int_answer) in enumerate(states):
        base_traj = _trajectory(root / "baseline" / f"qa{index}" / "trajectory.jsonl")
        int_traj = _trajectory(root / "intervention" / f"qa{index}" / "trajectory.jsonl")
        baseline_rows.append({
            "question_id": qa_id,
            "question": f"Stored question {index}?",
            "options": ["A. first", "B. second", "C. third", "D. fourth"],
            "ground_truth": "A",
            "prediction": base_answer,
            "is_correct": before,
            "trajectory_path": base_traj,
            "reference_sets": {
                "retrieved_segments": ["0_10"],
                "returned_segments": ["0_10"],
                "frame_inspected_segments": [],
                "explicitly_cited_segments": [],
                "consumed_segments": ["0_10"],
            },
        })
        qa_records.append({
            "qa_id": qa_id,
            "intervention_answer": int_answer,
            "baseline_correct": before,
            "intervention_correct": after,
            "transition": (
                "wrong_to_correct" if not before and after else
                "correct_to_wrong" if before and not after else
                "correct_to_correct" if before and after else
                "wrong_to_wrong"
            ),
            "intervention_trajectory_path": int_traj,
        })
        outcomes.append(QAInterventionOutcome(
            qa_id=qa_id,
            is_source_qa=is_source,
            baseline_answer=base_answer,
            intervention_answer=int_answer,
            baseline_correct=before,
            intervention_correct=after,
            baseline_trajectory_ref=base_traj,
            intervention_trajectory_ref=int_traj,
        ))
    baseline_qas = _write_jsonl(root / "baseline_qas.jsonl", baseline_rows)
    baseline_manifest = _write_json(root / "baseline_video_manifest.json", {
        "video_id": "video-001",
        "baseline_qas_path": baseline_qas,
    })
    intervention_manifest = _write_json(root / "intervention_manifest.json", {
        "schema_version": "fresh_prompt_delta_intervention_v1",
        "video_id": "video-001",
        "qa_records": qa_records,
    })
    return InterventionEpisode(
        episode_id="fresh-episode-001",
        video_id="video-001",
        parent_meta_prompt_id="meta-parent-001",
        prompt_delta=delta,
        clips=clips,
        qa_outcomes=tuple(outcomes),
        baseline_run_ref=baseline_manifest,
        intervention_run_ref=intervention_manifest,
    )
