import json
import os

from surrogate_rollout.references.extractor import (
    clips_for_range,
    extract_references,
    hhmmss_to_seconds,
)

CLIPS = [f"{i * 10}_{(i + 1) * 10}" for i in range(20)]  # 0..200s


def test_hhmmss_parsing():
    assert hhmmss_to_seconds("00:01:30") == 90
    assert hhmmss_to_seconds("01:00:00") == 3600
    assert hhmmss_to_seconds("02:15") == 135          # MM:SS tolerated
    assert hhmmss_to_seconds("00:00:05.500") == 5     # fractional dropped


def test_clips_for_range_overlap():
    assert clips_for_range(15, 35, CLIPS) == ["10_20", "20_30", "30_40"]
    assert clips_for_range(10, 10, CLIPS) == ["10_20"]  # point at boundary
    assert clips_for_range(0, 5, CLIPS) == ["0_10"]


def test_frame_inspect_event_extraction():
    events = [{
        "tool": "frame_inspect_tool",
        "args": {"question": "q", "time_ranges_hhmmss": [["00:00:15", "00:00:35"]]},
        "hits": [],
    }]
    refs = extract_references([], events, CLIPS)
    assert refs.frame_inspected_segments == {"10_20", "20_30", "30_40"}
    assert refs.explicitly_cited_segments == refs.frame_inspected_segments
    assert refs.retrieved_segments == set()
    assert refs.consumed_segments == refs.frame_inspected_segments


def test_retrieval_events_returned_vs_consumed():
    events = [
        {"tool": "clip_search_tool", "args": {},
         "hits": [{"time_start_secs": 40.0, "time_end_secs": 50.0}]},
        {"tool": "global_browse_tool", "args": {},
         "hits": [{"time_start_secs": 100.0, "time_end_secs": 110.0}]},
    ]
    refs = extract_references([], events, CLIPS)
    assert refs.retrieved_segments == {"40_50", "100_110"}
    # only clip_search captions reach the orchestrator verbatim
    assert refs.returned_segments == {"40_50"}
    # global_browse hits were consumed by the tool's internal LLM
    assert refs.consumed_segments == {"40_50", "100_110"}
    assert refs.explicitly_cited_segments == set()


def test_assistant_text_citations():
    messages = [
        {"role": "assistant", "content": "The event happens at 00:01:30 in the video."},
        {"role": "tool", "content": "at 00:02:00 something"},  # tool text ignored
    ]
    refs = extract_references(messages, [], CLIPS)
    assert refs.explicitly_cited_segments == {"90_100"}


def test_evidence_records_explain_every_segment():
    events = [{
        "tool": "clip_search_tool", "args": {"event_description": "x"},
        "hits": [{"time_start_secs": 0.0, "time_end_secs": 10.0}],
    }]
    refs = extract_references([], events, CLIPS)
    segs_with_evidence = {e["segment"] for e in refs.evidence}
    assert refs.retrieved_segments <= segs_with_evidence


def test_phase0_real_trajectory_smoke():
    """Real Phase 0 trajectory: one global_browse call, no sidecar events
    (instrumentation did not exist yet) -> only assistant-text citations
    may appear; extraction must not crash on real message shapes."""
    path = os.path.join(os.path.dirname(__file__), "data", "phase0_trajectory.jsonl")
    messages = [json.loads(l) for l in open(path)]
    clips = [f"{i * 10}_{(i + 1) * 10}" for i in range(184)]
    refs = extract_references(messages, [], clips)
    assert refs.retrieved_segments == set()
    assert isinstance(refs.explicitly_cited_segments, set)
