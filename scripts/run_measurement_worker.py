#!/usr/bin/env python
"""Drain the held-out measurement queue on its own GPUs.

Operator-run only: this performs local GPU work and paid provider calls.

The optimization loop promotes and enqueues; this worker scores. They share
nothing but a directory of JSON requests, so the two halves can sit on separate
GPUs or separate machines, and neither waits for the other.

Failure policy: a measurement is a report, never a gate. A request that fails is
moved aside with its traceback and the worker moves to the next one. Nothing
here can stop the optimization run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from surrogate_rollout.optimization.checkpoint_g_factory import (  # noqa: E402
    build_checkpoint_g_components,
)
from surrogate_rollout.optimization.prompt_delta_iteration import (  # noqa: E402
    DEFERRED_MEASUREMENT_REQUEST_SCHEMA_VERSION,
    MetaPromptConfirmationCase,
)
from surrogate_rollout.optimization.schemas import (  # noqa: E402
    meta_prompt_version_from_json,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical  # noqa: E402


def _object(path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(dumps_canonical(value) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--component-config", required=True,
                        help="Resolved component config; its runtime.worker_gpus "
                             "decides which GPUs this worker uses.")
    parser.add_argument("--worker-gpus", default=None,
                        help="Comma-separated GPU override for this worker.")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true",
                        help="Drain what is pending now, then exit.")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Retries before a request is parked in failed/.")
    return parser.parse_args(argv)


def _claim(path: Path, running: Path) -> Path | None:
    """Move a request into running/ so two workers cannot take the same one."""
    running.mkdir(parents=True, exist_ok=True)
    target = running / path.name
    try:
        os.rename(path, target)
    except OSError:
        return None
    return target


def _measure(request: dict, evaluator) -> dict:
    meta_prompt = meta_prompt_version_from_json(request["meta_prompt"])
    cases = tuple(MetaPromptConfirmationCase(**item)
                  for item in request["cases"])
    output_directory = request["measurement_output_directory"]
    os.makedirs(output_directory, exist_ok=True)
    measurement = evaluator.measure(
        meta_prompt=meta_prompt, cases=cases,
        confirmation_set_id=request["confirmation_set_id"],
        model_identity=request["model_identity"],
        decoding_settings=request["decoding_settings"],
        cache_reset_identity=request["cache_reset_identity"],
        evaluation_pipeline_identity=request["evaluation_pipeline_identity"],
        output_directory=output_directory)
    _write(Path(request["measurement_output_path"]), measurement)
    correct = sum(1 for item in measurement.outcomes if item.correct)
    return {
        "meta_prompt_id": measurement.meta_prompt_id,
        "measurement_id": measurement.measurement_id,
        "accuracy": measurement.accuracy,
        "correct": correct,
        "case_count": len(measurement.outcomes),
    }


def _drain_once(queue: Path, evaluator, max_attempts: int) -> int:
    pending = sorted((queue / "pending").glob("*.json"))
    handled = 0
    for path in pending:
        claimed = _claim(path, queue / "running")
        if claimed is None:
            continue
        name = claimed.name
        try:
            request = _object(claimed)
            if request.get("schema_version") != \
                    DEFERRED_MEASUREMENT_REQUEST_SCHEMA_VERSION:
                raise ValueError(
                    f"unexpected request schema: {request.get('schema_version')}")
            print(f"[measure] {name} -> {request['meta_prompt_id']}", flush=True)
            started = time.time()
            summary = _measure(request, evaluator)
            request["result"] = summary
            request["measured_at_seconds"] = round(time.time() - started, 1)
            _write(queue / "done" / name, request)
            claimed.unlink(missing_ok=True)
            print(f"[measure] {name} accuracy={summary['accuracy']:.4f} "
                  f"({summary['correct']}/{summary['case_count']}) in "
                  f"{request['measured_at_seconds']}s", flush=True)
        except BaseException as exc:  # noqa: BLE001 - the queue must survive
            attempts = 0
            try:
                attempts = int(_object(claimed).get("attempts", 0))
            except Exception:  # noqa: BLE001 - unreadable request
                pass
            attempts += 1
            record = {}
            try:
                record = _object(claimed)
            except Exception:  # noqa: BLE001
                record = {"unreadable_request": str(claimed)}
            record["attempts"] = attempts
            record["last_error"] = f"{type(exc).__name__}: {exc}"
            record["last_traceback"] = traceback.format_exc()
            destination = "pending" if attempts < max_attempts else "failed"
            _write(queue / destination / name, record)
            claimed.unlink(missing_ok=True)
            print(f"[measure] {name} attempt {attempts} failed -> "
                  f"{destination}: {record['last_error'][:200]}", flush=True)
            if isinstance(exc, KeyboardInterrupt):
                raise
        handled += 1
    return handled


def _recover_running(queue: Path) -> None:
    """A worker that died mid-measurement leaves a claim behind; take it back."""
    running = queue / "running"
    if not running.is_dir():
        return
    for path in sorted(running.glob("*.json")):
        shutil.move(str(path), str(queue / "pending" / path.name))
        print(f"[measure] recovered stale claim {path.name}", flush=True)


def main(argv=None) -> int:
    args = _args(argv)
    queue = Path(args.queue_dir).resolve()
    for name in ("pending", "running", "done", "failed"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    _recover_running(queue)

    factory_args = argparse.Namespace(component_config=args.component_config)
    if args.worker_gpus:
        factory_args.worker_gpus = args.worker_gpus
    _feedback, _updater, evaluator = build_checkpoint_g_components(factory_args)
    if evaluator is None:
        raise SystemExit("component config produced no measurement evaluator")

    print(f"[measure] queue={queue} gpus={args.worker_gpus or 'from config'}",
          flush=True)
    while True:
        handled = _drain_once(queue, evaluator, args.max_attempts)
        if args.once:
            return 0
        if handled == 0:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
