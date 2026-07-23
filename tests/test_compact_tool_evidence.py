"""Compaction must be structural. Nothing may depend on what evidence says."""

import json
import random
import re
import string

import pytest

from surrogate_rollout.optimization import compact_tool_evidence as cte
from surrogate_rollout.optimization.compact_tool_evidence import (
    CompactToolEvidenceError,
    build_compact_qa_evidence,
    build_compact_tool_calls,
    build_trajectory_delta,
)

REGISTRY = json.dumps({
    "subject_registry": {"Ronaldo": ["a striped shirt", "a red jersey"]},
    "scene_registry": {"stadium": ["floodlights"]},
})


def _event(tool, query_key, query, evidence, segment_ids, **extra):
    return {
        "tool": tool,
        "args": {query_key: query},
        "returned_evidence": list(evidence),
        "returned_segment_ids": list(segment_ids),
        **extra,
    }


def _trajectory(*events):
    return {"availability": "available", "tool_events": list(events)}


def _clip_search(query, evidence, segment_ids):
    return _event("clip_search_tool", "event_description", query,
                  evidence, segment_ids)


def _global_browse(query, evidence, segment_ids):
    return _event("global_browse_tool", "query", query, evidence, segment_ids)


def _frame_inspect(query, evidence, segment_ids, **extra):
    return _event("frame_inspect_tool", "question", query, evidence,
                  segment_ids, **extra)


# --------------------------------------------------------------------------- #
#                     no semantic judgement anywhere                           #
# --------------------------------------------------------------------------- #
def _scramble(text):
    """Replace every word with a random one, preserving structure exactly."""
    def replace(match):
        word = match.group(0)
        return "".join(random.choice(string.ascii_lowercase)
                       for _ in range(len(word)))
    return re.sub(r"[A-Za-z]+", replace, text)


def _shape(record):
    """Everything except the evidence text itself."""
    return {key: value for key, value in record.items()
            if key != "returned_evidence_excerpts"}


def test_rewriting_every_word_changes_nothing_structural():
    random.seed(0)
    original = _trajectory(
        _clip_search("what happens", [REGISTRY], ["0_10", "10_20"]),
        _global_browse("summary", ["timeline of the match"], ["20_30"]),
    )
    scrambled = _trajectory(
        _clip_search(_scramble("what happens"), [_scramble(REGISTRY)],
                     ["0_10", "10_20"]),
        _global_browse(_scramble("summary"), [_scramble("timeline of the match")],
                       ["20_30"]),
    )
    left = build_compact_tool_calls(original, changed_segment_ids={"10_20"})
    right = build_compact_tool_calls(scrambled, changed_segment_ids={"10_20"})
    assert [_shape(item) for item in left] == [
        {**_shape(item), "query": _shape(left[index])["query"]}
        for index, item in enumerate(right)]
    assert [len(item["returned_evidence_excerpts"]) for item in left] == \
        [len(item["returned_evidence_excerpts"]) for item in right]


def test_no_evidence_item_is_dropped_except_exact_duplicates():
    trajectory = _trajectory(
        _clip_search("a", ["alpha", "beta"], ["0_10"]),
        _clip_search("b", ["gamma"], ["10_20"]),
    )
    records = build_compact_tool_calls(trajectory, changed_segment_ids=set())
    kept = {unit for record in records
            for unit in record["returned_evidence_excerpts"]}
    assert kept == {"alpha", "beta", "gamma"}


def test_every_distinct_unit_of_a_registry_survives():
    trajectory = _trajectory(_global_browse("q", [REGISTRY], ["0_10"]))
    records = build_compact_tool_calls(trajectory, changed_segment_ids=set())
    rebuilt = {}
    for unit in records[0]["returned_evidence_excerpts"]:
        # Units stay decoded, so no second round of string escaping is paid.
        rebuilt.update(unit)
    assert rebuilt == json.loads(REGISTRY)


def test_two_qas_differing_only_in_wording_compact_identically():
    def build(word):
        return build_compact_tool_calls(
            _trajectory(_clip_search("q", [f"{word} appears"], ["0_10"])),
            changed_segment_ids=set())
    # Same length, so only the wording differs.
    left, right = build("Ronaldo"), build("Beckham")
    assert _shape(left[0]) == _shape(right[0])


def test_module_makes_no_model_or_network_call():
    source = open(cte.__file__, encoding="utf-8").read()
    for forbidden in ("requests", "openai", "http", "urllib", "socket",
                      "subprocess", "codex"):
        assert forbidden not in source


def test_compaction_is_deterministic():
    trajectory = _trajectory(
        _global_browse("q", [REGISTRY], ["0_10"]),
        _clip_search("r", ["alpha"], ["10_20"]),
    )
    first = build_compact_tool_calls(trajectory, changed_segment_ids={"0_10"})
    second = build_compact_tool_calls(trajectory, changed_segment_ids={"0_10"})
    assert first == second


# --------------------------------------------------------------------------- #
#                     deterministic structural processing                      #
# --------------------------------------------------------------------------- #
def test_duplicate_evidence_within_one_event_is_counted_not_repeated():
    trajectory = _trajectory(_clip_search("q", ["alpha", "alpha"], ["0_10"]))
    record = build_compact_tool_calls(trajectory, changed_segment_ids=set())[0]
    assert record["returned_evidence_excerpts"] == ["alpha"]
    assert record["duplicate_evidence_units_removed"] == 1
    assert record["returned_evidence_item_count"] == 2


def test_repeated_identical_event_is_merged_with_a_count():
    call = _global_browse("q", [REGISTRY], ["0_10"])
    records = build_compact_tool_calls(
        _trajectory(call, dict(call)), changed_segment_ids=set())
    assert len(records) == 1
    assert records[0]["repeated_event_occurrences"] == 2


def test_evidence_repeated_by_a_later_event_points_back():
    records = build_compact_tool_calls(
        _trajectory(_global_browse("first", [REGISTRY], ["0_10"]),
                    _global_browse("second", [REGISTRY], ["20_30"])),
        changed_segment_ids=set(), location={"side": "baseline"})
    assert records[1]["returned_evidence_excerpts"] == []
    assert records[1]["evidence_repeated_from_events"] == [
        {"side": "baseline", "event_index": 0}]


def test_timestamp_blocks_split_only_on_existing_boundaries():
    text = "0_10 a man walks\n10_20 he sits down\n"
    records = build_compact_tool_calls(
        _trajectory(_clip_search("q", [text], ["0_10"])),
        changed_segment_ids=set())
    assert records[0]["returned_evidence_excerpts"] == [
        "0_10 a man walks", "10_20 he sits down"]


def test_unsplittable_text_is_kept_whole():
    text = "one continuous sentence with no structure at all"
    records = build_compact_tool_calls(
        _trajectory(_clip_search("q", [text], ["0_10"])),
        changed_segment_ids=set())
    assert records[0]["returned_evidence_excerpts"] == [text]


def test_changed_segment_overlap_is_recorded():
    records = build_compact_tool_calls(
        _trajectory(_clip_search("q", ["alpha"], ["0_10", "10_20", "20_30"])),
        changed_segment_ids={"10_20", "90_100"})
    assert records[0]["returned_segment_ids"] == ["0_10", "10_20", "20_30"]
    assert records[0]["changed_segment_ids_returned"] == ["10_20"]


def test_events_keep_their_original_order_and_index():
    records = build_compact_tool_calls(
        _trajectory(_clip_search("first", ["a"], ["0_10"]),
                    _global_browse("second", ["b"], ["10_20"]),
                    _frame_inspect("third", ["c"], ["20_30"])),
        changed_segment_ids=set())
    assert [item["event_index"] for item in records] == [0, 1, 2]
    assert [item["query"] for item in records] == ["first", "second", "third"]


# --------------------------------------------------------------------------- #
#                          evidence provenance                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory,expected", [
    (_clip_search, "caption_backed"),
    (_global_browse, "aggregate_registry"),
    (_frame_inspect, "frame_backed"),
])
def test_source_type_follows_tool_identity(factory, expected):
    records = build_compact_tool_calls(
        _trajectory(factory("q", ["a"], ["0_10"])), changed_segment_ids=set())
    assert records[0]["evidence_source"] == expected


def test_frame_inspect_is_not_caption_exposure_by_default():
    records = build_compact_tool_calls(
        _trajectory(_frame_inspect("q", [REGISTRY], ["0_10"])),
        changed_segment_ids={"0_10"})
    assert records[0]["evidence_source"] == "frame_backed"


def test_frame_inspect_counts_as_caption_exposure_only_when_metadata_says_so():
    records = build_compact_tool_calls(
        _trajectory(_frame_inspect("q", ["a"], ["0_10"],
                                   caption_text_provided=True)),
        changed_segment_ids={"0_10"})
    assert records[0]["evidence_source"] == "caption_backed"


def test_unknown_tool_is_provenance_unknown_not_rejected():
    records = build_compact_tool_calls(
        _trajectory(_event("some_new_tool", "query", "q", ["a"], ["0_10"])),
        changed_segment_ids=set())
    assert records[0]["evidence_source"] == "provenance_unknown"


# --------------------------------------------------------------------------- #
#                            trajectory delta                                  #
# --------------------------------------------------------------------------- #
def test_trajectory_delta_counts_and_jaccard():
    baseline = _trajectory(_clip_search("q", ["a"], ["0_10", "10_20"]))
    intervention = _trajectory(_clip_search("q", ["a"], ["10_20", "20_30"]))
    delta = build_trajectory_delta(
        baseline, intervention, changed_segment_ids={"10_20"})
    assert delta["returned_segments_added"] == 1
    assert delta["returned_segments_removed"] == 1
    assert delta["returned_segments_jaccard"] == pytest.approx(
        1 / 3, abs=1e-4)


def test_delta_separates_changed_segments_seen_by_both_runs():
    baseline = _trajectory(_clip_search("q", ["a"], ["0_10"]))
    intervention = _trajectory(_clip_search("q", ["a"], ["0_10", "10_20"]))
    delta = build_trajectory_delta(
        baseline, intervention, changed_segment_ids={"0_10", "10_20"})
    assert delta["changed_segments_returned_in_both"] == ["0_10"]
    assert delta["changed_segments_returned_only_in_intervention"] == ["10_20"]


def test_frame_inspection_is_tracked_separately_from_returned_captions():
    baseline = _trajectory(_frame_inspect("q", ["a"], ["0_10"]))
    intervention = _trajectory(_frame_inspect("q", ["a"], ["20_30"]))
    delta = build_trajectory_delta(
        baseline, intervention, changed_segment_ids=set())
    assert delta["returned_segments_jaccard"] == 1.0
    assert delta["frame_inspected_segments_jaccard"] == 0.0
    assert delta["frame_inspected_segments_added"] == 1


def test_qa_evidence_bundles_both_sides_and_the_delta():
    evidence = build_compact_qa_evidence(
        _trajectory(_clip_search("q", ["a"], ["0_10"])),
        _trajectory(_clip_search("q", ["b"], ["0_10"])),
        changed_segment_ids={"0_10"})
    assert set(evidence) == {
        "compact_tool_evidence_version", "baseline_tool_calls",
        "intervention_tool_calls", "trajectory_delta"}


def test_malformed_trajectory_is_rejected_rather_than_guessed():
    with pytest.raises(CompactToolEvidenceError):
        build_compact_tool_calls({"tool_events": [42]},
                                 changed_segment_ids=set())
    with pytest.raises(CompactToolEvidenceError):
        build_compact_tool_calls(
            {"tool_events": [{"tool": "clip_search_tool",
                              "returned_evidence": "not an array"}]},
            changed_segment_ids=set())


def test_one_registry_survives_once_across_a_whole_episode():
    """The same blob returned by every QA of a video is stored once."""
    shared: dict = {}
    first = build_compact_qa_evidence(
        _trajectory(_global_browse("q", [REGISTRY], ["0_10"])),
        _trajectory(_global_browse("q", [REGISTRY], ["0_10"])),
        changed_segment_ids={"0_10"}, seen_units=shared, qa_id="qa/1")
    second = build_compact_qa_evidence(
        _trajectory(_global_browse("q", [REGISTRY], ["0_10"])),
        _trajectory(_global_browse("q", [REGISTRY], ["0_10"])),
        changed_segment_ids={"0_10"}, seen_units=shared, qa_id="qa/2")
    assert first["baseline_tool_calls"][0]["returned_evidence_excerpts"]
    for later in (first["intervention_tool_calls"][0],
                  second["baseline_tool_calls"][0],
                  second["intervention_tool_calls"][0]):
        assert later["returned_evidence_excerpts"] == []
        assert later["evidence_repeated_from_events"] == [
            {"qa_id": "qa/1", "side": "baseline", "event_index": 0}]


def test_sharing_never_drops_a_unit_that_differs():
    shared: dict = {}
    other = json.dumps({"subject_registry": {"Ronaldo": ["a blue jersey"]}})
    build_compact_qa_evidence(
        _trajectory(_global_browse("q", [REGISTRY], ["0_10"])),
        _trajectory(_global_browse("q", [REGISTRY], ["0_10"])),
        changed_segment_ids=set(), seen_units=shared, qa_id="qa/1")
    second = build_compact_qa_evidence(
        _trajectory(_global_browse("q", [other], ["0_10"])),
        _trajectory(_global_browse("q", [other], ["0_10"])),
        changed_segment_ids=set(), seen_units=shared, qa_id="qa/2")
    assert second["baseline_tool_calls"][0]["returned_evidence_excerpts"] == [
        json.loads(other)]


def test_compact_flag_off_leaves_the_request_shape_untouched():
    """The opt-in must not change what an existing run sends."""
    import inspect
    from surrogate_rollout.optimization.llm_episode_feedback import (
        build_lean_episode_feedback_request,
    )
    signature = inspect.signature(build_lean_episode_feedback_request)
    parameter = signature.parameters["compact_tool_evidence"]
    assert parameter.default is False
    source = inspect.getsource(build_lean_episode_feedback_request)
    # Every compact-only field is written inside the guarded branch.
    guard = source.index("if compact_tool_evidence or feedback_view:")
    for field in ("trajectory_delta", "compact_tool_evidence_version"):
        assert source.index(field) > guard
    assert signature.parameters["feedback_view"].default is False
