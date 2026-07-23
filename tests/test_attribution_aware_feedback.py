"""The attribution contract: a lesson exists only when the evidence carries it.

The strategy fields stand or fall together. `supported` means the generator
claims a transferable rule and must supply all four; every other status is an
abstention and must supply none of them. The parser enforces that instead of
quietly repairing a half-populated response.
"""

import json

import pytest

from surrogate_rollout.optimization.compact_tool_evidence import (
    build_trajectory_delta,
)
from surrogate_rollout.optimization.llm_episode_feedback import (
    EPISODE_FEEDBACK_RESPONSE_SCHEMA_VERSION,
    EPISODE_FEEDBACK_SYSTEM_INSTRUCTION,
    validate_attribution_contract,
)
from surrogate_rollout.optimization.policies.episode_feedback_provider import (
    episode_feedback_response_json_schema,
)
from surrogate_rollout.optimization.schemas import (
    EPISODE_FEEDBACK_ATTRIBUTION_STATUSES,
    EPISODE_FEEDBACK_STRATEGY_FIELDS,
)

NON_SUPPORTED = [status for status in EPISODE_FEEDBACK_ATTRIBUTION_STATUSES
                 if status != "supported"]


def _event(tool, segment_ids):
    return {
        "tool": tool,
        "args": {"query": "q"},
        "returned_evidence": ["evidence"],
        "returned_segment_ids": list(segment_ids),
    }


def _trajectory(*events):
    return {"availability": "available", "tool_events": list(events)}


def _supported():
    return {
        "attribution_status": "supported",
        "observable_trigger": "when two people face each other",
        "caption_operation": "state who faces whom",
        "recommended_strategy_change": "when two people face each other, "
                                       "state who faces whom",
        "compact_memory_text": "facing pairs were named",
    }


# --------------------------------------------------------------------------- #
#                 1-4. per-source overlap is provenance-correct                #
# --------------------------------------------------------------------------- #
def test_per_source_overlap_is_computed_per_provenance_class():
    baseline = _trajectory(
        _event("clip_search_tool", ["0_10", "10_20"]),
        _event("global_browse_tool", ["20_30"]),
        _event("frame_inspect_tool", ["30_40"]))
    intervention = _trajectory(
        _event("clip_search_tool", ["0_10"]),
        _event("global_browse_tool", ["20_30"]),
        _event("frame_inspect_tool", ["30_40"]))
    by_source = build_trajectory_delta(
        baseline, intervention,
        changed_segment_ids={"0_10", "10_20", "20_30", "30_40"},
    )["changed_segments_returned_in_both_by_source"]
    assert by_source["caption_backed"] == ["0_10"]
    assert by_source["aggregate_registry"] == ["20_30"]
    assert by_source["frame_backed"] == ["30_40"]
    assert by_source["provenance_unknown"] == []


def test_registry_segment_is_not_counted_as_caption_exposure():
    both = _trajectory(_event("global_browse_tool", ["0_10"]))
    by_source = build_trajectory_delta(
        both, both, changed_segment_ids={"0_10"},
    )["changed_segments_returned_in_both_by_source"]
    assert by_source["aggregate_registry"] == ["0_10"]
    assert by_source["caption_backed"] == []


def test_frame_backed_segment_is_not_counted_as_caption_exposure():
    both = _trajectory(_event("frame_inspect_tool", ["0_10"]))
    by_source = build_trajectory_delta(
        both, both, changed_segment_ids={"0_10"},
    )["changed_segments_returned_in_both_by_source"]
    assert by_source["frame_backed"] == ["0_10"]
    assert by_source["caption_backed"] == []


def test_frame_inspect_declaring_caption_input_is_caption_backed():
    event = _event("frame_inspect_tool", ["0_10"])
    event["caption_text_provided"] = True
    both = _trajectory(event)
    by_source = build_trajectory_delta(
        both, both, changed_segment_ids={"0_10"},
    )["changed_segments_returned_in_both_by_source"]
    assert by_source["caption_backed"] == ["0_10"]
    assert by_source["frame_backed"] == []


def test_a_segment_may_appear_under_several_sources():
    both = _trajectory(
        _event("clip_search_tool", ["0_10"]),
        _event("global_browse_tool", ["0_10"]))
    by_source = build_trajectory_delta(
        both, both, changed_segment_ids={"0_10"},
    )["changed_segments_returned_in_both_by_source"]
    assert by_source["caption_backed"] == ["0_10"]
    assert by_source["aggregate_registry"] == ["0_10"]


def test_every_source_key_is_always_present_and_sorted():
    both = _trajectory(_event("clip_search_tool", ["30_40", "0_10", "10_20"]))
    delta = build_trajectory_delta(
        both, both, changed_segment_ids={"0_10", "10_20", "30_40"})
    by_source = delta["changed_segments_returned_in_both_by_source"]
    assert set(by_source) == {
        "caption_backed", "frame_backed", "aggregate_registry",
        "provenance_unknown"}
    assert by_source["caption_backed"] == sorted(by_source["caption_backed"])


def test_per_source_overlap_is_deterministic():
    both = _trajectory(_event("clip_search_tool", ["10_20", "0_10"]))
    first = build_trajectory_delta(
        both, both, changed_segment_ids={"0_10", "10_20"})
    second = build_trajectory_delta(
        both, both, changed_segment_ids={"0_10", "10_20"})
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True)


def test_overlap_requires_the_segment_on_both_sides():
    baseline = _trajectory(_event("clip_search_tool", ["0_10"]))
    intervention = _trajectory(_event("clip_search_tool", ["10_20"]))
    by_source = build_trajectory_delta(
        baseline, intervention, changed_segment_ids={"0_10", "10_20"},
    )["changed_segments_returned_in_both_by_source"]
    assert by_source["caption_backed"] == []


# --------------------------------------------------------------------------- #
#                    5-6. the strategy fields move together                    #
# --------------------------------------------------------------------------- #
def test_supported_requires_all_four_strategy_fields():
    validate_attribution_contract(_supported())


@pytest.mark.parametrize("missing", EPISODE_FEEDBACK_STRATEGY_FIELDS)
def test_supported_with_any_null_strategy_field_is_rejected(missing):
    value = _supported()
    value[missing] = None
    with pytest.raises(ValueError, match="requires non-empty"):
        validate_attribution_contract(value)


@pytest.mark.parametrize("missing", EPISODE_FEEDBACK_STRATEGY_FIELDS)
def test_supported_with_a_blank_strategy_field_is_rejected(missing):
    value = _supported()
    value[missing] = "   "
    with pytest.raises(ValueError, match="requires non-empty"):
        validate_attribution_contract(value)


@pytest.mark.parametrize("status", NON_SUPPORTED)
def test_abstaining_status_requires_all_four_to_be_null(status):
    value = {name: None for name in EPISODE_FEEDBACK_STRATEGY_FIELDS}
    value["attribution_status"] = status
    validate_attribution_contract(value)


@pytest.mark.parametrize("status", NON_SUPPORTED)
@pytest.mark.parametrize("populated", EPISODE_FEEDBACK_STRATEGY_FIELDS)
def test_abstaining_status_may_not_smuggle_a_rule(status, populated):
    value = {name: None for name in EPISODE_FEEDBACK_STRATEGY_FIELDS}
    value["attribution_status"] = status
    value[populated] = "a rule that should not be here"
    with pytest.raises(ValueError, match="requires null"):
        validate_attribution_contract(value)


def test_unknown_attribution_status_is_rejected_not_defaulted():
    value = _supported()
    value["attribution_status"] = "probably_fine"
    with pytest.raises(ValueError, match="attribution_status must be one of"):
        validate_attribution_contract(value)


# --------------------------------------------------------------------------- #
#                     7-8. prompt and schema stay in step                      #
# --------------------------------------------------------------------------- #
def test_every_field_the_prompt_names_exists_in_the_strict_schema():
    schema = episode_feedback_response_json_schema()
    named = {"attribution_status", "observable_trigger", "caption_operation",
             "recommended_strategy_change", "compact_memory_text",
             "outcome_summary", "generator_diagnosis"}
    for field in named:
        assert field in EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
        assert field in schema["properties"]
        assert field in schema["required"]
    assert schema["additionalProperties"] is False


def test_every_status_the_schema_allows_is_described_by_the_prompt():
    schema = episode_feedback_response_json_schema()
    enum = schema["properties"]["attribution_status"]["enum"]
    assert set(enum) == set(EPISODE_FEEDBACK_ATTRIBUTION_STATUSES)
    for status in enum:
        assert status in EPISODE_FEEDBACK_SYSTEM_INSTRUCTION


def test_prompt_treats_source_type_as_metadata_not_as_a_gate():
    """The minimal view carries source_type; it must not block feedback."""
    lowered = " ".join(EPISODE_FEEDBACK_SYSTEM_INSTRUCTION.lower().split())
    assert "source_type" in lowered
    assert "it is not a gate" in lowered
    assert "do not refuse to write feedback" in lowered


def test_prompt_no_longer_forces_a_rule_out_of_an_outcome():
    # The prompt is hard-wrapped, so compare on collapsed whitespace.
    lowered = " ".join(EPISODE_FEEDBACK_SYSTEM_INSTRUCTION.lower().split())
    assert "otherwise return null for all four" in lowered
    for forced in ("must produce a do rule", "always produce",
                   "preserve-plus-safeguard"):
        assert forced not in lowered


def test_prompt_forbids_segment_level_causal_claims():
    lowered = " ".join(EPISODE_FEEDBACK_SYSTEM_INSTRUCTION.lower().split())
    assert "do not claim that one segment caused one qa transition" in lowered
    # The ban list is written once, as a run-on clause.
    assert "never phrase them as: when it is important, relevant, useful" \
        in lowered
    for banned in ("when it helps answer the question",
                   "when it resolves an error",
                   "when it improves performance"):
        assert banned in lowered


def test_response_schema_version_records_the_new_contract():
    assert EPISODE_FEEDBACK_RESPONSE_SCHEMA_VERSION == \
        "episode_feedback_response_v6_attribution_aware"


def test_prompt_change_moves_the_recorded_system_prompt_hash():
    """The manifest hash the resume guard compares is over this prompt text."""
    from surrogate_rollout.schemas import sha256_text
    current = sha256_text(EPISODE_FEEDBACK_SYSTEM_INSTRUCTION)
    superseded = sha256_text(
        "Analyze one prompt-delta intervention episode and produce concise, "
        "evidence-linked feedback about the tested visual prompt-generator "
        "behavior.")
    assert current != superseded
    assert len(current) == 64
