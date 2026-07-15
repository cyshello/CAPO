"""Run the Stage 4.7 routed evaluator integration smoke test.

Example (repository parent, one QA on the cached eight-clip short video):

    /home/intern/.conda/envs/local_llm_vllm/bin/python -m \
      surrogate_rollout.scripts.run_phase4_routed_smoke_test \
      --split short --indices 2 --gpu 0 \
      --work-root surrogate_rollout/runs/phase4_stage4_7_smoke

This command captions two frozen composition conditions and evaluates both via
the existing DVD reasoning path.  It does not construct evidence or perform
feedback, proposals, component updates, or optimization.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time

from surrogate_rollout import config
from surrogate_rollout.prompt_routing.routed_smoke_test import (
    RoutedEvaluatorSmokeRunner,
    load_smoke_components,
)

DEFAULT_COMPONENTS = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "prompt_routing", "fixtures", "stage4_7_components.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4.7 fallback-vs-routed DVD integration smoke")
    parser.add_argument("--split", default="short")
    parser.add_argument("--indices", default="2",
                        help="one to three comma-separated provider indices")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--components-file", default=DEFAULT_COMPONENTS)
    parser.add_argument("--work-root", default=os.path.join(
        config.RUNS_ROOT, f"phase4_stage4_7_{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--max-iterations", type=int,
                        default=config.MAX_ITERATIONS)
    args = parser.parse_args()

    from data_provider import get_provider
    from dvd_prompt import get_prompts

    indices = tuple(int(value.strip()) for value in args.indices.split(",")
                    if value.strip())
    if not 1 <= len(indices) <= 3:
        raise SystemExit("--indices must contain one to three provider indices")
    provider = get_provider(config.BENCHMARK, split=args.split)
    qa_samples = tuple(
        (f"{config.BENCHMARK}/{args.split}/{index}", provider[index])
        for index in indices)
    components = load_smoke_components(args.components_file)
    prompts = get_prompts()
    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
        text=True, cwd=os.path.dirname(os.path.dirname(__file__))).stdout.strip()
    result = RoutedEvaluatorSmokeRunner().run(
        qa_samples=qa_samples,
        base_prompt_template=prompts.caption_prompt,
        merge_prompt=prompts.merge_prompt,
        work_root=args.work_root,
        gpu=args.gpu,
        source_revision=source_revision,
        max_iterations=args.max_iterations,
        **components,
    )
    print(result.manifest_path)
    print(result.comparison_path)


if __name__ == "__main__":
    main()
