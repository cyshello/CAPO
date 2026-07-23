#!/usr/bin/env python3
"""Score one meta-prompt on the held-out confirmation cases. Decision-free.

Used for the iteration-0 point of a learning curve: the starting meta-prompt is
measured before any update, so the curve begins at the unoptimized baseline.
Nothing here promotes, rejects, or writes optimizer state.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from surrogate_rollout.optimization.meta_prompt_defaults import (
    resolve_meta_prompt_artifact_path,
)
from surrogate_rollout.optimization.prompt_delta_iteration import (
    MetaPromptConfirmationCase,
)
from surrogate_rollout.optimization.schemas import meta_prompt_version_from_json
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-prompt", help="MetaPromptVersion JSON; omit for the "
                                        "repository-owned initial prompt.")
    p.add_argument("--confirmation-cases", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--component-factory", required=True)
    p.add_argument("--component-config", required=True)
    p.add_argument("--decoding-settings", required=True)
    p.add_argument("--model-identity", required=True)
    p.add_argument("--cache-reset-identity", required=True)
    p.add_argument("--evaluation-pipeline-identity", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        meta_prompt = meta_prompt_version_from_json(
            _object(resolve_meta_prompt_artifact_path(args.meta_prompt)))
        cases = tuple(
            MetaPromptConfirmationCase(**item)
            for item in json.loads(
                Path(args.confirmation_cases).read_text(encoding="utf-8")))
        decoding = json.loads(
            Path(args.decoding_settings).read_text(encoding="utf-8"))

        module_name, name = args.component_factory.split(":", 1)
        factory = getattr(importlib.import_module(module_name), name)
        components = factory(args)
        evaluator = components[2]

        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        result = evaluator.measure(
            meta_prompt=meta_prompt, cases=cases,
            confirmation_set_id="confirmation_set_" + sha256_json(
                json.loads(dumps_canonical(cases)))[:20],
            model_identity=args.model_identity,
            decoding_settings=decoding,
            cache_reset_identity=args.cache_reset_identity,
            evaluation_pipeline_identity=args.evaluation_pipeline_identity,
            output_directory=str(output))
        summary = {
            "schema_version": "meta_prompt_measurement_summary_v1",
            "status": "completed",
            "meta_prompt_id": result.meta_prompt_id,
            "measurement_id": result.measurement_id,
            "case_count": len(cases),
            "evaluated_qa_count": result.evaluated_count,
            "accuracy": result.accuracy,
            "measurement_manifest_path": str(
                output / "dvd_measurement_manifest.json"),
        }
        (output / "measurement_summary.json").write_text(
            dumps_canonical(summary) + "\n", encoding="utf-8")
        print(dumps_canonical(summary))
    except Exception as exc:  # noqa: BLE001 - operator-facing launcher
        print(f"meta-prompt measurement failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
