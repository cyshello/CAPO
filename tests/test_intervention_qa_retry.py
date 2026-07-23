"""An unscoreable DVD answer is re-run, not escalated to a driver restart.

The agent occasionally finishes without naming an option. Observed once in
roughly 270 runs. Ending the process there cost a GPU drain, a backoff, a vLLM
warmup and one restart from a bounded budget, to redo a call that takes seconds
and usually succeeds on the next attempt.

What must not change: a QA that is unusable on every attempt still fails. The
alternative -- recording it as an answer -- would put a transition the model
never made into the evidence the optimizer learns from.
"""

import pytest

from surrogate_rollout.optimization.fresh_prompt_delta_evidence import (
    FreshPromptDeltaError,
    PromptDeltaInterventionRunner,
)

PARSE_FAILURE = [{
    "stage": "answer_parsing", "type": "ParseFailure",
    "error": 'no option letter in "Unable to determine from the video."',
}]


class _Result:
    def __init__(self, prediction, errors=()):
        self.prediction = prediction
        self.errors = list(errors)
        self.score = 1.0


class _Mixed:
    captions_path = "/tmp/captions.json"
    database_path = "/tmp/database.json"


def _runner(results):
    calls = []

    def qa_fn(**kwargs):
        calls.append(kwargs)
        return results[min(len(calls) - 1, len(results) - 1)]

    runner = PromptDeltaInterventionRunner.__new__(PromptDeltaInterventionRunner)
    runner.qa_fn = qa_fn
    runner.qa_retry_attempts = 3
    runner.dvd_max_iterations = 15
    runner.gpu = "0"
    return runner, calls


def _run(runner):
    return runner._run_qa_with_retries(
        qa_id="videomme/long/473", sample={}, qa_dir="/tmp/qa", mixed=_Mixed())


def test_a_usable_answer_is_returned_on_the_first_call():
    runner, calls = _runner([_Result("D")])
    assert _run(runner).prediction == "D"
    assert len(calls) == 1


def test_an_unscoreable_answer_is_retried_and_recovers():
    runner, calls = _runner([_Result(None, PARSE_FAILURE), _Result("D")])
    assert _run(runner).prediction == "D"
    assert len(calls) == 2


def test_a_retry_writes_to_its_own_directory():
    """The first attempt's artifacts stay put; the retry does not overwrite."""
    runner, calls = _runner([_Result(None, PARSE_FAILURE), _Result("D")])
    _run(runner)
    assert calls[0]["run_dir"] == "/tmp/qa"
    assert calls[1]["run_dir"] == "/tmp/qa__retry1"


def test_a_persistently_unusable_qa_still_fails():
    runner, calls = _runner([_Result(None, PARSE_FAILURE)])
    with pytest.raises(FreshPromptDeltaError, match="after 3 attempts"):
        _run(runner)
    assert len(calls) == 3


def test_the_failure_still_names_the_qa_and_the_stored_errors():
    runner, _ = _runner([_Result(None, PARSE_FAILURE)])
    with pytest.raises(FreshPromptDeltaError) as raised:
        _run(runner)
    assert "videomme/long/473" in str(raised.value)
    assert "ParseFailure" in str(raised.value)


def test_retries_can_be_switched_off():
    runner, calls = _runner([_Result(None, PARSE_FAILURE)])
    runner.qa_retry_attempts = 1
    with pytest.raises(FreshPromptDeltaError, match="after 1 attempts"):
        _run(runner)
    assert len(calls) == 1


def test_a_missing_prediction_without_errors_is_also_retried():
    runner, calls = _runner([_Result(None), _Result("B")])
    assert _run(runner).prediction == "B"
    assert len(calls) == 2


def test_the_default_attempt_count_is_configured_not_hard_coded():
    from surrogate_rollout import config
    assert config.INTERVENTION_QA_RETRY_ATTEMPTS >= 2
