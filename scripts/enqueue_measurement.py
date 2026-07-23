#!/usr/bin/env python
"""Queue one held-out measurement for the measurement worker to run.

Used for the iteration-0 point: the starting meta-prompt has to be scored on
the held-out set, but running it inline blocks the optimization loop for hours
on the loop's own GPUs. Queued, it becomes the first thing the measurement
worker picks up -- which is also the window in which that worker would
otherwise be idle, waiting for the first candidate.

Writes the request and exits. Performs no model calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from surrogate_rollout.optimization.prompt_delta_iteration import (  # noqa: E402
    DEFERRED_MEASUREMENT_REQUEST_SCHEMA_VERSION,
)
from surrogate_rollout.optimization.schemas import (  # noqa: E402
    meta_prompt_version_from_json,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical  # noqa: E402
from surrogate_rollout.schemas import sha256_json  # noqa: E402


def _object(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--meta-prompt", required=True)
    parser.add_argument("--confirmation-cases", required=True)
    parser.add_argument("--output-dir", required=True,
                        help="Where the measurement artifacts are written.")
    parser.add_argument("--iteration-id", required=True)
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--decoding-settings", required=True)
    parser.add_argument("--cache-reset-identity", required=True)
    parser.add_argument("--evaluation-pipeline-identity", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _args(argv)
    meta_prompt = meta_prompt_version_from_json(_object(args.meta_prompt))
    cases = _object(args.confirmation_cases) if os.path.isdir(
        args.confirmation_cases) else json.loads(
            Path(args.confirmation_cases).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise SystemExit("confirmation cases must be a non-empty array")

    queue = Path(args.queue_dir).resolve()
    for name in ("pending", "running", "done", "failed"):
        (queue / name).mkdir(parents=True, exist_ok=True)

    output = Path(args.output_dir).resolve()
    name = f"{args.iteration_id}__{meta_prompt.meta_prompt_id}.json"
    for state in ("done", "failed", "running", "pending"):
        if (queue / state / name).is_file():
            print(f"already queued or complete: {queue / state / name}")
            return 0
    if (output / "active_measurement.json").is_file():
        print(f"already measured: {output / 'active_measurement.json'}")
        return 0

    request = {
        "schema_version": DEFERRED_MEASUREMENT_REQUEST_SCHEMA_VERSION,
        "iteration_id": args.iteration_id,
        "meta_prompt_id": meta_prompt.meta_prompt_id,
        "meta_prompt": json.loads(dumps_canonical(meta_prompt)),
        # Computed exactly as the orchestrator does, so a queued measurement
        # and an inline one carry the same confirmation-set identity.
        "confirmation_set_id": "confirmation_set_" + sha256_json(cases)[:20],
        "cases": cases,
        "model_identity": args.model_identity,
        "decoding_settings": _object(args.decoding_settings),
        "cache_reset_identity": args.cache_reset_identity,
        "evaluation_pipeline_identity": args.evaluation_pipeline_identity,
        "measurement_output_directory": str(output),
        "measurement_output_path": str(output / "active_measurement.json"),
    }
    path = queue / "pending" / name
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(dumps_canonical(request) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(f"queued {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
