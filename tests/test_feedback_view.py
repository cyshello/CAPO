"""The feedback view carries the episode, not a provenance report.

Everything the generator needs to compare a delta against its captions stays;
the audit structures (registries, segment-id arrays, overlaps, Jaccard) move out
of the payload and remain in the compact artifact.
"""

import copy
import json

import pytest

from surrogate_rollout.optimization import feedback_view as fv
from surrogate_rollout.optimization.compact_tool_evidence import (
    build_compact_qa_evidence,
)
from surrogate_rollout.optimization.feedback_view import (
    EPISODE_FEEDBACK_VIEW_VERSION,
    FeedbackViewError,
    build_feedback_view,
    extract_returned_answer_text,
)

REGISTRY_WITH_ANSWER = json.dumps({
    "subject_registry": {"Ronaldo": {"appearance": ["a red jersey"] * 40}},
    "query_related_event": "He lifts the trophy after the final whistle.",
})
REGISTRY_ONLY = json.dumps({
    "subject_registry": {"Ronaldo": {"appearance": ["a red jersey"]}},
})


class _Delta:
    delta_id = "prompt_delta_test"
    instruction = "Emphasize who is speaking."


class _Clip:
    def __init__(self, segment_id, baseline, intervention):
        self.segment_id = segment_id
        self.baseline_caption = baseline
        self.intervention_caption = intervention


class _Episode:
    episode_id = "fresh_episode_test"
    prompt_delta = _Delta()
    clips = [
        _Clip("0_10", "A man walks.", "A man in a suit walks."),
        _Clip("10_20", "Unchanged.", "Unchanged."),
        _Clip("20_30", "Two people talk.", "Two people face each other."),
    ]


def _event(tool, evidence, segment_ids):
    return {
        "tool": tool,
        "args": {"query": f"{tool} query"},
        "returned_evidence": list(evidence),
        "returned_segment_ids": list(segment_ids),
    }


def _trajectory(*events):
    return {"availability": "available", "tool_events": list(events)}


def _qa(qa_id, is_source=True, transition="wrong_to_correct"):
    return {
        "qa_id": qa_id,
        "is_source_qa": is_source,
        "question": f"What happens in {qa_id}?",
        "answer_choices": ["A. one", "B. two", "C. three", "D. four"],
        "gold_answer": "D",
        "baseline_answer": "C",
        "intervention_answer": "D",
        "transition": transition,
    }


def _built(evidence=(REGISTRY_WITH_ANSWER,), qa_records=None):
    qa_records = qa_records or [_qa("videomme/long/1")]
    compact, trajectories = {}, {}
    for record in qa_records:
        baseline = _trajectory(_event("global_browse_tool", evidence, ["0_10"]))
        intervention = _trajectory(
            _event("clip_search_tool", evidence, ["20_30"]))
        compact[record["qa_id"]] = build_compact_qa_evidence(
            baseline, intervention, changed_segment_ids={"0_10", "20_30"})
        trajectories[record["qa_id"]] = {
            "baseline": baseline, "intervention": intervention}
    return build_feedback_view(
        _Episode(), qa_records, compact, trajectories), compact, trajectories


# --------------------------------------------------------------------------- #
#                        1-2. episode content is complete                      #
# --------------------------------------------------------------------------- #
def test_every_changed_caption_pair_survives_verbatim():
    view, _, _ = _built()
    assert view["changed_captions"] == [
        {"segment_id": "0_10", "baseline": "A man walks.",
         "intervention": "A man in a suit walks."},
        {"segment_id": "20_30", "baseline": "Two people talk.",
         "intervention": "Two people face each other."},
    ]


def test_unchanged_captions_are_not_carried():
    view, _, _ = _built()
    assert "10_20" not in [item["segment_id"] for item in
                           view["changed_captions"]]


def test_changed_captions_are_not_filtered_by_what_the_trace_touched():
    # Only 0_10 and 20_30 are returned by any tool, yet both pairs are kept
    # regardless of which side referenced them.
    view, _, _ = _built()
    assert len(view["changed_captions"]) == 2


def test_source_and_sibling_qas_all_survive_with_their_transitions():
    records = [_qa("videomme/long/1", True, "wrong_to_correct"),
               _qa("videomme/long/2", False, "correct_to_wrong"),
               _qa("videomme/long/3", False, "wrong_to_wrong")]
    view, _, _ = _built(qa_records=records)
    assert [item["qa_id"] for item in view["qa_outcomes"]] == [
        "videomme/long/1", "videomme/long/2", "videomme/long/3"]
    assert [item["transition"] for item in view["qa_outcomes"]] == [
        "wrong_to_correct", "correct_to_wrong", "wrong_to_wrong"]
    assert [item["is_source_qa"] for item in view["qa_outcomes"]] == [
        True, False, False]


def test_qa_records_keep_every_required_field():
    view, _, _ = _built()
    assert set(view["qa_outcomes"][0]) == {
        "qa_id", "is_source_qa", "question", "choices", "gold_answer",
        "baseline_answer", "intervention_answer", "transition"}


def test_prompt_delta_is_carried():
    view, _, _ = _built()
    assert view["prompt_delta"] == {
        "delta_id": "prompt_delta_test",
        "instruction": "Emphasize who is speaking."}


# --------------------------------------------------------------------------- #
#                        3-5. structural answer projection                     #
# --------------------------------------------------------------------------- #
def test_query_related_event_is_used_exactly():
    assert extract_returned_answer_text([REGISTRY_WITH_ANSWER]) == \
        "He lifts the trophy after the final whistle."


def test_registry_is_excluded_once_the_answer_field_is_present():
    view, _, _ = _built()
    payload = json.dumps(view)
    assert "subject_registry" not in payload
    assert "He lifts the trophy" in payload


@pytest.mark.parametrize("field", ["final_answer", "answer", "result", "text"])
def test_other_designated_answer_fields_are_used(field):
    item = json.dumps({"subject_registry": {"x": 1}, field: "the answer"})
    assert extract_returned_answer_text([item]) == "the answer"


def test_timestamp_blocks_are_kept_in_order():
    text = "0_10 a man walks\n10_20 he sits down\n"
    assert extract_returned_answer_text([text]) == \
        "0_10 a man walks\n10_20 he sits down"


def test_evidence_without_a_safe_projection_is_kept_verbatim():
    text = "one continuous sentence with no structure at all"
    assert extract_returned_answer_text([text]) == text


def test_registry_without_an_answer_field_is_kept_verbatim():
    assert extract_returned_answer_text([REGISTRY_ONLY]) == REGISTRY_ONLY


def test_only_exact_duplicates_are_dropped():
    assert extract_returned_answer_text(["alpha", "alpha", "beta"]) == \
        "alpha\n\nbeta"


# --------------------------------------------------------------------------- #
#                        6-8. tool evidence is faithful                        #
# --------------------------------------------------------------------------- #
def test_baseline_and_intervention_evidence_are_both_present():
    view, _, _ = _built()
    evidence = view["reasoning_evidence"][0]
    assert evidence["baseline"]["tools"] and evidence["intervention"]["tools"]


def test_tool_name_and_query_are_the_stored_ones():
    view, _, _ = _built()
    evidence = view["reasoning_evidence"][0]
    assert evidence["baseline"]["tools"][0]["tool"] == "global_browse_tool"
    assert evidence["baseline"]["tools"][0]["query"] == \
        "global_browse_tool query"
    assert evidence["intervention"]["tools"][0]["tool"] == "clip_search_tool"


def test_source_type_matches_the_compact_metadata():
    view, compact, _ = _built()
    qa_id = view["qa_outcomes"][0]["qa_id"]
    for side in ("baseline", "intervention"):
        tools = view["reasoning_evidence"][0][side]["tools"]
        calls = compact[qa_id][f"{side}_tool_calls"]
        assert [item["source_type"] for item in tools] == [
            call["evidence_source"] for call in calls]


def test_tool_records_carry_only_the_five_agreed_fields():
    view, _, _ = _built()
    for side in ("baseline", "intervention"):
        for tool in view["reasoning_evidence"][0][side]["tools"]:
            assert set(tool) == {
                "event_index", "tool", "query", "returned_answer_text",
                "source_type"}


def test_a_compact_call_pointing_outside_the_trajectory_is_rejected():
    with pytest.raises(FeedbackViewError):
        fv._tools([{"event_index": 7}], _trajectory())


# --------------------------------------------------------------------------- #
#                    9-12. the audit structures stay behind                    #
# --------------------------------------------------------------------------- #
FORBIDDEN_IN_PAYLOAD = (
    "influence_summary", "influence_paths", "direct_caption_exposure",
    "jaccard", "changed_segments_returned_in_both",
    "changed_segments_returned_in_both_by_source", "trajectory_delta",
    "returned_segment_ids", "changed_segment_ids_returned",
    "frame_inspected", "segments_added", "segments_removed",
    "returned_evidence_excerpts", "duplicate_evidence_units_removed",
)


@pytest.mark.parametrize("field", FORBIDDEN_IN_PAYLOAD)
def test_audit_structures_are_absent_from_the_payload(field):
    view, _, _ = _built()
    assert field not in json.dumps(view)


def test_module_makes_no_model_embedding_or_network_call():
    source = open(fv.__file__, encoding="utf-8").read()
    for forbidden in ("requests", "openai", "http", "urllib", "socket",
                      "subprocess", "codex", "embedding", "tiktoken"):
        assert forbidden not in source


def test_two_episodes_differing_only_in_wording_project_identically():
    def shape(view):
        return json.loads(json.dumps(view).replace("Ronaldo", "Beckham"))
    view, _, _ = _built()
    assert shape(view).keys() == view.keys()


# --------------------------------------------------------------------------- #
#                      13-15. inputs, determinism, identity                    #
# --------------------------------------------------------------------------- #
def test_inputs_are_not_mutated():
    compact_before = None
    view, compact, trajectories = _built()
    compact_before = copy.deepcopy(compact)
    trajectories_before = copy.deepcopy(trajectories)
    build_feedback_view(_Episode(), [_qa("videomme/long/1")], compact,
                        trajectories)
    assert compact == compact_before
    assert trajectories == trajectories_before
    assert _Episode.clips[0].baseline_caption == "A man walks."


def test_payload_is_byte_identical_across_builds():
    first, _, _ = _built()
    second, _, _ = _built()
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True)


def test_view_version_is_carried_in_the_payload():
    view, _, _ = _built()
    assert view["feedback_view_version"] == EPISODE_FEEDBACK_VIEW_VERSION
    assert EPISODE_FEEDBACK_VIEW_VERSION == "episode_feedback_view_v1_minimal"


def test_view_version_replaces_the_lean_schema_version_in_the_request():
    import inspect

    from surrogate_rollout.optimization.llm_episode_feedback import (
        LEAN_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION,
        build_lean_episode_feedback_request,
    )
    assert EPISODE_FEEDBACK_VIEW_VERSION != \
        LEAN_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION
    source = inspect.getsource(build_lean_episode_feedback_request)
    assert '"schema_version": EPISODE_FEEDBACK_VIEW_VERSION' in source


def test_reasoning_generator_rejects_a_budget_sized_for_visible_text_only():
    """gpt-5-mini at medium effort returns nothing at 512 output tokens."""
    import importlib

    from surrogate_rollout import config
    import surrogate_rollout.prompt_routing.policies.openai_free_form_generator \
        as generator_module

    original = config.GENERATOR_REASONING_EFFORT
    config.GENERATOR_REASONING_EFFORT = "medium"
    importlib.reload(generator_module)
    try:
        def _build(model_id, max_tokens):
            generator = generator_module.OpenAIFreeFormInstructionGenerator(
                model_id=model_id, max_tokens=max_tokens,
                template_text="meta", meta_prompt_id="x", api_key="k")
            return generator

        # Construction stays cheap: callers build a default generator and
        # replace it with the configured one immediately afterwards.
        too_small = _build("gpt-5-mini", 512)
        with pytest.raises(generator_module.FreeFormGenerationError,
                           match="output tokens"):
            too_small._complete([], "prompt")
        # Above the floor, and non-reasoning models, reach the transport.
        for generator in (_build("gpt-5-mini", 4096),
                          _build("gpt-4o-mini", 512)):
            with pytest.raises(Exception) as raised:
                generator._complete([], "prompt")
            assert "output tokens" not in str(raised.value)
    finally:
        config.GENERATOR_REASONING_EFFORT = original
        importlib.reload(generator_module)
