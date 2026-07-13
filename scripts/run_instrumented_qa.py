"""Run instrumented DVD QAs (grouped per video for engine reuse).

Usage:
    conda run -n local_llm_vllm python -m surrogate_rollout.scripts.run_instrumented_qa \
        --indices 9,10,11 --gpu 1 --out-root runs/phase1_trajectories
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from surrogate_rollout import config
from surrogate_rollout.dvd_runner import run_qa_instrumented

if config.PROMPT_SENS_ROOT not in sys.path:
    sys.path.insert(0, config.PROMPT_SENS_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", required=True, help="comma-separated provider indices")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--model", default=config.ORCHESTRATOR_TOOL_MODEL)
    ap.add_argument("--no-openai-tools", dest="openai_tools", action="store_false")
    ap.add_argument("--prompts", default=None, help="JSON file of prompt overrides")
    ap.add_argument("--out-root", default=os.path.join(config.RUNS_ROOT, "instrumented"))
    args = ap.parse_args()

    from data_provider import get_provider

    provider = get_provider(config.BENCHMARK, split=config.BENCHMARK_SPLIT)
    overrides = {}
    if args.prompts:
        with open(args.prompts) as f:
            overrides = json.load(f)

    for idx in [int(i) for i in args.indices.split(",") if i.strip()]:
        sample = provider[idx]
        vid = sample.get("extra", {}).get("videoID") or sample["sample_id"]
        run_dir = os.path.join(args.out_root, f"{vid}_q{idx}")
        res = run_qa_instrumented(
            sample,
            run_dir=run_dir,
            gpu=args.gpu,
            prompt_overrides=overrides,
            tool_calling_model=args.model,
            use_openai_tools=args.openai_tools,
            question_id=f"{config.BENCHMARK}/{config.BENCHMARK_SPLIT}/{idx}",
        )
        print("RESULT_JSON " + json.dumps({
            "index": idx, "video": vid, "score": res.score,
            "parsed": res.parsed_answer, "gold": res.ground_truth,
            "n_tool_events": len(res.references.evidence),
            "errors": [e["type"] for e in res.errors],
            "run_dir": run_dir,
        }), flush=True)


if __name__ == "__main__":
    main()
