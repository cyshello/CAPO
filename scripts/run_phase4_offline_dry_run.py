"""Stage 4.5 offline routing-to-composition dry run.

Documented deterministic fixture command (run from the repository parent):

    /home/intern/.conda/envs/local_llm_vllm/bin/python -m \
      surrogate_rollout.scripts.run_phase4_offline_dry_run \
      --fixture --output-dir /tmp/phase4_stage4_5

The command makes no captioning, DVD reasoning, model, or optimization calls.
Use ``--input-bundle PATH`` instead of ``--fixture`` to replay another saved
bundle with the same JSON structure.
"""

from __future__ import annotations

import argparse
import os
import subprocess

from surrogate_rollout.prompt_routing.offline_dry_run import (
    load_fixture_bundle,
    run_offline_dry_run,
)

DEFAULT_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "prompt_routing", "fixtures", "stage4_5_fixture.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline SegmentContext -> routing -> composition dry run")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--fixture", action="store_true",
        help="use the repository's deterministic Stage 4.5 fixture")
    source.add_argument(
        "--input-bundle",
        help="saved Stage 4.5 JSON input bundle")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    bundle = load_fixture_bundle(
        DEFAULT_FIXTURE if args.fixture else args.input_bundle)
    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
        text=True, cwd=os.path.dirname(os.path.dirname(__file__))).stdout.strip()
    result = run_offline_dry_run(
        output_dir=args.output_dir, source_revision=source_revision, **bundle)
    print(result.manifest_path)


if __name__ == "__main__":
    main()
