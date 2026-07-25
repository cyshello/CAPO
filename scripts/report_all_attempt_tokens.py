#!/usr/bin/env python
"""Total tokens across every attempt, not one run.

Ledger directories are per run timestamp, and a crashed attempt's tokens were
still paid for, so the experiment's real cost is the sum over all of them --
the 08:12 DNS death, the 18:12 provider-500 death, and the run that finishes.
Each directory is aggregated with the run's own summariser, so the per-run
numbers here match that run's token_usage_summary.json exactly.

Local components (the Qwen captioner) are counted but priced at zero.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from surrogate_rollout import token_ledger  # noqa: E402

# Every iteration writes its own ledger dir (run_fresh_prompt_delta_iteration.sh
# exports SR_TOKEN_LEDGER_DIR=<run root>_tokens), and a restarted attempt appends
# new per-PID files to the same one. Discovering them keeps later iterations in
# the total without editing this list.
DEFAULT_LEDGER_GLOB = "runs/fresh_prompt_delta_iteration_*_tokens"


def default_ledger_directories(root: str | None = None) -> list[str]:
    base = Path(root or ".")
    return sorted(str(path) for path in base.glob(DEFAULT_LEDGER_GLOB)
                  if path.is_dir())
# USD per million tokens, provider list price.
PRICES = {"gpt-5-mini": (0.25, 2.00), "gpt-4o": (2.50, 10.00),
          "gpt-4o-mini": (0.15, 0.60)}
LOCAL_COMPONENTS = ("qwen",)


def _model_of(component: str, models: dict[str, str]) -> str | None:
    return models.get(component)


def _models_per_component(directory: str) -> dict[str, str]:
    """The model id each component recorded, read from the ledger rows."""
    found: dict[str, str] = {}
    for path in sorted(Path(directory).glob("*.jsonl")):
        component = path.name.split(".")[0]
        if component in found:
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                model = row.get("model")
                if model:
                    found[component] = str(model)
                    break
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger_dirs", nargs="*",
                        default=default_ledger_directories())
    parser.add_argument("--json", action="store_true",
                        help="Emit the combined report as JSON.")
    args = parser.parse_args(argv)

    combined: dict[str, dict[str, int]] = {}
    models: dict[str, str] = {}
    per_run = []
    files = 0
    for directory in args.ledger_dirs:
        if not os.path.isdir(directory):
            print(f"skipped (absent): {directory}", file=sys.stderr)
            continue
        summary = token_ledger.aggregate(directory)
        files += summary["ledger_files"]
        models.update(_models_per_component(directory))
        per_run.append((directory, summary))
        for name, counts in summary["components"].items():
            bucket = combined.setdefault(
                name, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            for field in bucket:
                bucket[field] += int(counts[field])

    usd = 0.0
    for name, counts in combined.items():
        model = _model_of(name, models)
        if name in LOCAL_COMPONENTS or model not in PRICES:
            continue
        prompt_price, completion_price = PRICES[model]
        usd += (counts["prompt_tokens"] / 1e6 * prompt_price
                + counts["completion_tokens"] / 1e6 * completion_price)

    report = {
        "schema_version": "all_attempt_token_report_v1",
        "ledger_directories": [d for d, _ in per_run],
        "ledger_files": files,
        "components": {
            name: {**counts,
                   "model": _model_of(name, models),
                   "total_tokens": counts["prompt_tokens"]
                   + counts["completion_tokens"]}
            for name, counts in sorted(combined.items())},
        "total": {
            "calls": sum(c["calls"] for c in combined.values()),
            "prompt_tokens": sum(c["prompt_tokens"] for c in combined.values()),
            "completion_tokens": sum(
                c["completion_tokens"] for c in combined.values()),
        },
        "paid_usd_list_price": round(usd, 2),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    for directory, summary in per_run:
        total = summary["total"]
        print(f"{Path(directory).name}: {total['calls']:>6} calls  "
              f"{total['total_tokens']:>12,} tokens  "
              f"({summary['ledger_files']} files)")
    print("-" * 78)
    header = f"{'component':<18}{'model':<14}{'calls':>8}{'prompt':>14}{'completion':>13}"
    print(header)
    for name, counts in report["components"].items():
        print(f"{name:<18}{str(counts['model'] or '-'):<14}"
              f"{counts['calls']:>8,}{counts['prompt_tokens']:>14,}"
              f"{counts['completion_tokens']:>13,}")
    total = report["total"]
    print(f"{'TOTAL':<18}{'':<14}{total['calls']:>8,}"
          f"{total['prompt_tokens']:>14,}{total['completion_tokens']:>13,}")
    print(f"\npaid (list price, local captioner excluded): "
          f"${report['paid_usd_list_price']:,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
