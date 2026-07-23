"""Two regressions from the first gpt-5-mini feedback/update run.

The runner reached for a `candidate` attribute the update result does not have,
after the result was already on disk -- a paid run that finished its work and
then crashed writing its last file.

And an abstaining feedback still shipped its prose to the updater, which reasoned
about it and let it shape a rule the abstention did not support.
"""

import importlib.util
from pathlib import Path

import pytest

from surrogate_rollout.optimization.meta_prompt_updater import (
    MetaPromptUpdateResult,
    _updater_feedback_projection,
)
from surrogate_rollout.optimization.schemas import (
    EpisodeFeedback,
    EpisodeFeedbackEvidence,
)

_RUNNER_PATH = (Path(__file__).resolve().parents[1] / "scripts" /
                "run_feedback_and_update_once.py")


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_feedback_and_update_once", _RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Decision:
    def __init__(self, text):
        self.candidate_meta_prompt = text
        self.decision = "update" if text else "no_update"


class _Update:
    def __init__(self, text):
        self.decision = _Decision(text)
        self.candidate_meta_prompt_id = "meta_prompt_test" if text else None
        self.candidate_status = "provisional" if text else None


# --------------------------------------------------------------------------- #
#                        the runner's final write                              #
# --------------------------------------------------------------------------- #
def test_candidate_is_read_from_the_decision_not_the_result():
    record = _runner().candidate_record(_Update("line one\nline two"))
    assert record["text"] == "line one\nline two"
    assert record["candidate_meta_prompt_id"] == "meta_prompt_test"
    assert record["candidate_status"] == "provisional"
    assert record["line_count"] == 2


def test_no_update_writes_no_candidate():
    assert _runner().candidate_record(_Update(None)) is None


def test_the_update_result_still_has_no_candidate_attribute():
    """The bug was reaching for this; assert the shape that caused it."""
    assert "candidate" not in MetaPromptUpdateResult.__dataclass_fields__
    assert "candidate_meta_prompt_id" in \
        MetaPromptUpdateResult.__dataclass_fields__


def test_runner_never_reaches_for_the_missing_attribute():
    source = _RUNNER_PATH.read_text(encoding="utf-8")
    assert "update.candidate\n" not in source
    assert "update.candidate " not in source
    assert "update.candidate)" not in source


# --------------------------------------------------------------------------- #
#                   what an abstaining feedback may contribute                 #
# --------------------------------------------------------------------------- #
def _feedback(status, rule=None):
    evidence = EpisodeFeedbackEvidence(
        statement="Observed: the intervention captions named the trophy.",
        supporting_segment_ids=("0_10",),
        supporting_qa_ids=("videomme/long/1",),
        evidence_type="caption_change",
        confidence="Direct stored evidence.", transition_type=None)
    return EpisodeFeedback(
        feedback_id="episode_feedback_test",
        episode_id="fresh_episode_test",
        outcome_summary="One QA improved.",
        observations=(evidence,),
        counterevidence=(),
        generator_diagnosis="A caption change may have helped.",
        recommended_strategy_change=rule,
        confidence="Local evidence only.",
        compact_memory_text="trophies were named" if rule else None,
        attribution_status=status,
        observable_trigger="when a trophy is held up" if rule else None,
        caption_operation="name the trophy" if rule else None)


def test_supported_feedback_still_carries_its_prose():
    projection = _updater_feedback_projection(
        _feedback("supported", "when a trophy is held up, name it"), None)
    assert projection["caption_or_trajectory_evidence"] == [
        "Observed: the intervention captions named the trophy."]
    assert projection["recommended_strategy_change"] == \
        "when a trophy is held up, name it"


@pytest.mark.parametrize("status", [
    "trajectory_confounded", "no_changed_caption_exposure",
    "insufficient_evidence"])
def test_abstaining_feedback_contributes_no_prose(status):
    projection = _updater_feedback_projection(_feedback(status), None)
    assert projection["caption_or_trajectory_evidence"] == []
    assert projection["recommended_strategy_change"] is None
    assert projection["observable_trigger"] is None
    assert projection["caption_operation"] is None


@pytest.mark.parametrize("status", [
    "trajectory_confounded", "no_changed_caption_exposure",
    "insufficient_evidence"])
def test_abstaining_feedback_still_reports_what_happened(status):
    projection = _updater_feedback_projection(_feedback(status), None)
    assert projection["attribution_status"] == status
    assert "qa_transition_counts" in projection
    assert "episode_effect" in projection


# --------------------------------------------------------------------------- #
#                       the updater asks for a rewrite                         #
# --------------------------------------------------------------------------- #
def _updater_prompt():
    from surrogate_rollout.optimization.meta_prompt_updater import (
        META_PROMPT_UPDATER_SYSTEM_INSTRUCTION,
    )
    return " ".join(META_PROMPT_UPDATER_SYSTEM_INSTRUCTION.lower().split())


def test_updater_prompt_requires_a_full_rewrite_within_twelve_lines():
    lowered = _updater_prompt()
    assert "rewrite; do not append" in lowered
    assert "return the complete meta-prompt, rewritten as a whole" in lowered
    assert "candidate_meta_prompt must be at most 12 lines" in lowered


def test_updater_prompt_and_request_agree_on_the_status_vocabulary():
    """The prompt reads attribution_status; the request has to speak it."""
    lowered = _updater_prompt()
    assert "attribution_status" in lowered
    assert "transition counts and nothing else" in lowered
    supported = _updater_feedback_projection(
        _feedback("supported", "when a trophy is held up, name it"), None)
    assert supported["attribution_status"] == "supported"
    abstained = _updater_feedback_projection(
        _feedback("trajectory_confounded"), None)
    assert abstained["attribution_status"] == "trajectory_confounded"
