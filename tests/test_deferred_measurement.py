"""The held-out measurement leaves the optimization loop.

It never decided anything -- promotion happens before it and does not read it --
so making the loop wait for it only meant a scoring failure could stop a run.
Under the deferred policy the iteration promotes, records what still has to be
scored, and returns; a separate worker drains the queue on its own GPUs.

The property that matters: nothing the measurement side does can stop the
optimization side.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from surrogate_rollout.optimization.prompt_delta_iteration import (
    DEFERRED_MEASUREMENT_POLICY,
    DEFERRED_MEASUREMENT_REQUEST_SCHEMA_VERSION,
    PROMOTION_POLICIES,
    PromptDeltaIterationOrchestrator,
)

_WORKER_PATH = (Path(__file__).resolve().parents[1] / "scripts" /
                "run_measurement_worker.py")


def _worker():
    spec = importlib.util.spec_from_file_location(
        "run_measurement_worker", _WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _queue(tmp_path, count=1, name="it"):
    queue = tmp_path / "queue"
    for state in ("pending", "running", "done", "failed"):
        (queue / state).mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (queue / "pending" / f"{name}{index}.json").write_text(json.dumps({
            "schema_version": DEFERRED_MEASUREMENT_REQUEST_SCHEMA_VERSION,
            "iteration_id": f"{name}{index}",
            "meta_prompt_id": f"meta_prompt_{index}",
            "meta_prompt": {
                "meta_prompt_id": f"meta_prompt_{index}",
                "parent_meta_prompt_id": None,
                "text": "task section",
                "created_at": "2026-07-23T00:00:00Z",
                "status": "confirmed",
            },
            "cases": [{
                "case_id": "confirmation_v_qa",
                "video_id": "v", "qa_id": "qa",
                "input_ref": "split_manifest.json#provider_index=1",
            }],
            "confirmation_set_id": "confirmation_set_test",
            "model_identity": "captioner=x",
            "decoding_settings": {},
            "cache_reset_identity": "reset",
            "evaluation_pipeline_identity": "pipeline",
            "measurement_output_directory": str(tmp_path / f"out{index}"),
            "measurement_output_path": str(
                tmp_path / f"out{index}" / "active_measurement.json"),
        }), encoding="utf-8")
    return queue


def _Measurement():
    """The real result type: the worker persists it exactly as the loop did."""
    from surrogate_rollout.optimization.dvd_meta_prompt_confirmation import (
        MetaPromptMeasurementOutcome,
        MetaPromptMeasurementResult,
    )
    return MetaPromptMeasurementResult(
        measurement_id="measurement_0", meta_prompt_id="meta_prompt_0",
        confirmation_set_id="confirmation_set_test",
        model_identity="captioner=x", decoding_settings={},
        cache_reset_identity="reset", evaluation_pipeline_identity="pipeline",
        outcomes=(
            MetaPromptMeasurementOutcome(
                case_id="a", video_id="v", qa_id="qa", correct=True,
                error=None),
            MetaPromptMeasurementOutcome(
                case_id="b", video_id="v", qa_id="qb", correct=False,
                error=None)))


# --------------------------------------------------------------------------- #
#                       the policy exists and is guarded                       #
# --------------------------------------------------------------------------- #
def test_the_deferred_policy_is_selectable():
    assert DEFERRED_MEASUREMENT_POLICY in PROMOTION_POLICIES


def test_the_deferred_policy_needs_a_queue_directory():
    with pytest.raises(ValueError, match="measurement_queue_directory"):
        PromptDeltaIterationOrchestrator(
            feedback_generator=object(), updater=object(),
            confirmation_evaluator=object(),
            promotion_policy=DEFERRED_MEASUREMENT_POLICY)


def test_the_deferred_policy_does_not_need_an_evaluator(tmp_path):
    """The scoring runs elsewhere, so this process needs no evaluator at all."""
    PromptDeltaIterationOrchestrator(
        feedback_generator=object(), updater=object(),
        confirmation_evaluator=object(),
        promotion_policy=DEFERRED_MEASUREMENT_POLICY,
        measurement_queue_directory=str(tmp_path))


def test_the_inline_policy_still_requires_an_evaluator():
    with pytest.raises(ValueError, match="requires a measurement_evaluator"):
        PromptDeltaIterationOrchestrator(
            feedback_generator=object(), updater=object(),
            confirmation_evaluator=object(),
            promotion_policy="always_promote_measured_v1")


# --------------------------------------------------------------------------- #
#                    the worker survives whatever it is given                  #
# --------------------------------------------------------------------------- #
def test_a_measured_request_moves_to_done(tmp_path):
    worker = _worker()
    queue = _queue(tmp_path)
    handled = worker._drain_once(
        queue, type("E", (), {"measure": lambda *a, **k: _Measurement()})(), 3)
    assert handled == 1
    assert list((queue / "done").glob("*.json"))
    assert not list((queue / "pending").glob("*.json"))
    assert not list((queue / "running").glob("*.json"))


def test_a_failing_measurement_never_raises(tmp_path):
    worker = _worker()
    queue = _queue(tmp_path)

    class _Boom:
        def measure(self, **kwargs):
            raise RuntimeError("GPU fell over")

    # The call returns normally; the optimization side would carry on.
    assert worker._drain_once(queue, _Boom(), 3) == 1
    parked = list((queue / "pending").glob("*.json"))
    assert parked, "a first failure is retried, not discarded"
    record = json.loads(parked[0].read_text())
    assert record["attempts"] == 1
    assert "GPU fell over" in record["last_error"]
    assert "Traceback" in record["last_traceback"]


def test_a_request_is_parked_after_the_attempt_limit(tmp_path):
    worker = _worker()
    queue = _queue(tmp_path)

    class _Boom:
        def measure(self, **kwargs):
            raise RuntimeError("still broken")

    for _ in range(3):
        worker._drain_once(queue, _Boom(), 3)
    assert list((queue / "failed").glob("*.json"))
    assert not list((queue / "pending").glob("*.json"))


def test_one_bad_request_does_not_block_the_others(tmp_path):
    worker = _worker()
    queue = _queue(tmp_path, count=3)
    seen = []

    class _Flaky:
        def measure(self, **kwargs):
            seen.append(kwargs)
            if len(seen) == 1:
                raise RuntimeError("first one fails")
            return _Measurement()

    worker._drain_once(queue, _Flaky(), 3)
    assert len(list((queue / "done").glob("*.json"))) == 2
    assert len(list((queue / "pending").glob("*.json"))) == 1


def test_an_unreadable_request_is_parked_rather_than_crashing(tmp_path):
    worker = _worker()
    queue = _queue(tmp_path)
    list((queue / "pending").glob("*.json"))[0].write_text("{not json",
                                                          encoding="utf-8")
    assert worker._drain_once(queue, object(), 1) == 1
    assert list((queue / "failed").glob("*.json"))


def test_a_stale_claim_is_recovered_on_startup(tmp_path):
    worker = _worker()
    queue = _queue(tmp_path)
    claimed = list((queue / "pending").glob("*.json"))[0]
    claimed.rename(queue / "running" / claimed.name)
    worker._recover_running(queue)
    assert list((queue / "pending").glob("*.json"))
    assert not list((queue / "running").glob("*.json"))


def test_an_empty_queue_is_not_an_error(tmp_path):
    worker = _worker()
    queue = tmp_path / "queue"
    for state in ("pending", "running", "done", "failed"):
        (queue / state).mkdir(parents=True, exist_ok=True)
    assert worker._drain_once(queue, object(), 3) == 0


# --------------------------------------------------------------------------- #
#              the iteration-0 point fills the worker's first gap              #
# --------------------------------------------------------------------------- #
def _enqueue_script():
    path = (Path(__file__).resolve().parents[1] / "scripts" /
            "enqueue_measurement.py")
    spec = importlib.util.spec_from_file_location("enqueue_measurement", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _enqueue_inputs(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({
        "meta_prompt_id": "meta_prompt_init", "parent_meta_prompt_id": None,
        "text": "task section", "created_at": "2026-07-23T00:00:00Z",
        "status": "parent"}), encoding="utf-8")
    (tmp_path / "cases.json").write_text(json.dumps([{
        "case_id": "confirmation_v_qa", "video_id": "v", "qa_id": "qa",
        "input_ref": "split_manifest.json#provider_index=1"}]), encoding="utf-8")
    (tmp_path / "decoding.json").write_text("{}", encoding="utf-8")
    return [
        "--queue-dir", str(tmp_path / "queue"),
        "--meta-prompt", str(tmp_path / "meta.json"),
        "--confirmation-cases", str(tmp_path / "cases.json"),
        "--output-dir", str(tmp_path / "parent_measurement"),
        "--iteration-id", "iteration_0_initial_meta_prompt",
        "--model-identity", "captioner=x",
        "--decoding-settings", str(tmp_path / "decoding.json"),
        "--cache-reset-identity", "reset",
        "--evaluation-pipeline-identity", "pipeline",
    ]


def test_iteration_zero_is_queued_not_measured_inline(tmp_path):
    assert _enqueue_script().main(_enqueue_inputs(tmp_path)) == 0
    queued = list((tmp_path / "queue" / "pending").glob("*.json"))
    assert len(queued) == 1
    request = json.loads(queued[0].read_text())
    assert request["schema_version"] == \
        DEFERRED_MEASUREMENT_REQUEST_SCHEMA_VERSION
    assert request["meta_prompt_id"] == "meta_prompt_init"
    assert request["confirmation_set_id"].startswith("confirmation_set_")


def test_queueing_iteration_zero_twice_does_not_duplicate_work(tmp_path):
    script = _enqueue_script()
    argv = _enqueue_inputs(tmp_path)
    script.main(argv)
    script.main(argv)
    assert len(list((tmp_path / "queue" / "pending").glob("*.json"))) == 1


def test_an_already_measured_iteration_zero_is_not_requeued(tmp_path):
    argv = _enqueue_inputs(tmp_path)
    measured = tmp_path / "parent_measurement"
    measured.mkdir(parents=True, exist_ok=True)
    (measured / "active_measurement.json").write_text("{}", encoding="utf-8")
    assert _enqueue_script().main(argv) == 0
    assert not list((tmp_path / "queue" / "pending").glob("*.json"))
