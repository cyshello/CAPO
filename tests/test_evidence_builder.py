"""Stage 4.8 offline counterfactual-evidence builder tests."""

import json

import pytest

from surrogate_rollout.optimization.evidence_builder import (
    CounterfactualEvidenceBuilder,
    EvidenceBuildError,
    SavedEvaluationArtifact,
    load_normalized_evaluations,
    load_phase2_3_condition,
    read_evidence_jsonl,
    write_evidence_jsonl,
    write_normalized_evaluations,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


def captions(path, values):
    path.write_text(json.dumps({
        **{segment: {"caption": text} for segment, text in values.items()},
        "subject_registry": {},
    }))
    return str(path)


def record(tmp_path, side, *, video="v1", question="q1", answer="A",
           score=1.0, config="cfg", selected=None, composed=None, traces=None,
           fallback=False, caption_values=None):
    caption_path = tmp_path / f"{side}_{video}_{question}.json"
    captions(caption_path, caption_values or {"0_10": f"{side} caption"})
    return SavedEvaluationArtifact(
        artifact_id=f"{side}:{video}:{question}",
        condition_name=side,
        source_format="synthetic_saved",
        video_id=video,
        question_id=question,
        ground_truth="A",
        answer=answer,
        score=score,
        bank_version="bank_v0001",
        router_version="router_v0001",
        scaffold_version="scaffold_v0001",
        selected_prompt_ids=selected or {"0_10": (f"pe_{side}",)},
        composed_prompt_refs=composed or {"0_10": f"hash_{side}"},
        composition_traces=traces if traces is not None else {
            "0_10": {"selected_prompt_ids": [f"pe_{side}"]}},
        captions_path=str(caption_path),
        trajectory_refs=(f"/{side}/trajectory.jsonl",),
        rollout_artifact_refs={"result": f"/{side}/result.json"},
        config_fingerprint=config,
        used_fallback=fallback,
        fallback_segments=(("0_10",) if fallback else ()),
        metadata={"saved": True},
    )


def test_matched_pairing_is_stable_and_unmatched_rejected(tmp_path):
    b1 = record(tmp_path, "baseline", question="q1")
    b2 = record(tmp_path, "baseline", question="q2")
    c1 = record(tmp_path, "candidate", question="q1")
    c2 = record(tmp_path, "candidate", question="q2")
    evidence = CounterfactualEvidenceBuilder().build([b2, b1], [c1, c2])
    assert [item.question_id for item in evidence] == ["q1", "q2"]
    with pytest.raises(EvidenceBuildError, match="pairing mismatch"):
        CounterfactualEvidenceBuilder().build([b1, b2], [c1])


def test_configuration_mismatch_rejected(tmp_path):
    baseline = record(tmp_path, "baseline", config="config_a")
    candidate = record(tmp_path, "candidate", config="config_b")
    with pytest.raises(EvidenceBuildError, match="configuration mismatch"):
        CounterfactualEvidenceBuilder().build([baseline], [candidate])


@pytest.mark.parametrize(("before_answer", "before_score", "after_answer",
                          "after_score", "transition"), [
    ("A", 1.0, "A", 1.0, "correct_to_correct"),
    ("A", 1.0, "B", 0.0, "correct_to_wrong"),
    ("B", 0.0, "A", 1.0, "wrong_to_correct"),
    ("B", 0.0, "B", 0.0, "wrong_to_wrong"),
    (None, 0.0, "A", 1.0, "unknown"),
])
def test_all_outcome_transitions_retained(
    tmp_path, before_answer, before_score, after_answer, after_score, transition,
):
    baseline = record(tmp_path, "baseline", answer=before_answer,
                      score=before_score)
    candidate = record(tmp_path, "candidate", answer=after_answer,
                       score=after_score)
    item = CounterfactualEvidenceBuilder().build([baseline], [candidate])[0]
    assert item.outcome_transition == transition
    assert item.score_delta == after_score - before_score


def test_missing_traces_are_explicit_not_fatal(tmp_path):
    baseline = record(tmp_path, "baseline", traces={})
    candidate = record(tmp_path, "candidate")
    item = CounterfactualEvidenceBuilder().build([baseline], [candidate])[0]
    assert item.metadata["missing_trace_sides"] == ("baseline",)
    assert item.metadata["baseline_composition_traces"] == {}


def test_fallback_provenance_retained(tmp_path):
    baseline = record(tmp_path, "baseline", fallback=True)
    candidate = record(tmp_path, "candidate")
    item = CounterfactualEvidenceBuilder().build([baseline], [candidate])[0]
    assert item.metadata["baseline_used_fallback"] is True
    assert item.metadata["baseline_fallback_segments"] == ("0_10",)
    assert item.metadata["candidate_used_fallback"] is False


def test_evidence_ids_and_caption_differences_are_deterministic(tmp_path):
    baseline = record(tmp_path, "baseline", caption_values={
        "10_20": "same", "0_10": "old"})
    candidate = record(tmp_path, "candidate", caption_values={
        "0_10": "new", "10_20": "same"})
    builder = CounterfactualEvidenceBuilder()
    first = builder.build([baseline], [candidate])[0]
    second = builder.build([baseline], [candidate])[0]
    assert first.evidence_id == second.evidence_id
    assert [d.segment_id for d in first.caption_differences] == ["0_10"]
    assert first.caption_differences[0].difference_summary == "caption_changed"


def test_selected_entry_differences_retained(tmp_path):
    baseline = record(tmp_path, "baseline",
                      selected={"0_10": ("pe_default",)})
    candidate = record(tmp_path, "candidate",
                       selected={"0_10": ("pe_default", "pe_temporal")})
    item = CounterfactualEvidenceBuilder().build([baseline], [candidate])[0]
    assert item.baseline_selected_prompt_ids["0_10"] == ("pe_default",)
    assert item.metadata["selected_entry_differences"]["0_10"][
        "candidate"] == ("pe_default", "pe_temporal")


def test_composed_prompt_differences_retained(tmp_path):
    baseline = record(tmp_path, "baseline", composed={"0_10": "hash_a"})
    candidate = record(tmp_path, "candidate", composed={"0_10": "hash_b"})
    item = CounterfactualEvidenceBuilder().build([baseline], [candidate])[0]
    assert item.baseline_composed_prompt_refs["0_10"] == "hash_a"
    assert item.metadata["composed_prompt_differences"]["0_10"] == {
        "baseline": "hash_a", "candidate": "hash_b"}


def test_evidence_and_normalized_serialization_round_trip(tmp_path):
    baseline = record(tmp_path, "baseline")
    candidate = record(tmp_path, "candidate")
    normalized_path = tmp_path / "normalized.jsonl"
    write_normalized_evaluations([baseline, candidate], str(normalized_path))
    loaded = load_normalized_evaluations(str(normalized_path))
    assert [dumps_canonical(item) for item in loaded] == \
        [dumps_canonical(baseline), dumps_canonical(candidate)]

    evidence = CounterfactualEvidenceBuilder().build([baseline], [candidate])
    evidence_path = tmp_path / "evidence.jsonl"
    write_evidence_jsonl(evidence, str(evidence_path))
    round_tripped = read_evidence_jsonl(str(evidence_path))
    assert [dumps_canonical(item) for item in round_tripped] == \
        [dumps_canonical(item) for item in evidence]


def test_phase2_3_loader_normalizes_filtered_saved_rows(tmp_path):
    (tmp_path / "run_config.json").write_text(json.dumps({"seed": 0}))
    cap = captions(tmp_path / "captions.json", {"0_10": "caption"})
    rows = [
        {"candidate_name": "p", "rollout_mode": "full",
         "selection_policy": "full_rollout", "video_id": "v", "qa_id": "q",
         "parsed_answer": "A", "score": 1.0, "captions_path": cap,
         "trajectory_path": "/tmp/t.jsonl", "fallback_type": "none"},
        {"candidate_name": "other", "rollout_mode": "full",
         "selection_policy": "full_rollout", "video_id": "v", "qa_id": "q",
         "parsed_answer": "B", "score": 0.0, "captions_path": cap,
         "fallback_type": "none"},
    ]
    results = tmp_path / "results.jsonl"
    results.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loaded = load_phase2_3_condition(
        str(results), condition_name="full",
        filters={"candidate_name": "p", "rollout_mode": "full"})
    assert len(loaded) == 1
    assert loaded[0].source_format == "phase2_3"
    assert loaded[0].bank_version == "legacy_phase2_3"
