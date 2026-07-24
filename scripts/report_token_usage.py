#!/usr/bin/env python3
"""Aggregate a training iteration's token ledger and report the totals.

Scans every ``*.jsonl`` under a ledger directory (the ledger's own records plus
any ``SR_QWEN_USAGE_LOG`` / ``SR_OPENAI_USAGE_LOG`` files pointed there) and sums
prompt/completion tokens per source and in total. Writes
``token_usage_summary.json`` into the ledger dir (or --output) and prints a
table.

Usage:
  python scripts/report_token_usage.py <ledger_dir> [--output PATH]
  # or, if SR_TOKEN_LEDGER_DIR is set:
  python scripts/report_token_usage.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from surrogate_rollout import token_ledger


def _fmt(n: int) -> str:
    return f"{n:,}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ledger_dir", nargs="?", default=token_ledger.ledger_directory(),
        help="Ledger directory; defaults to $SR_TOKEN_LEDGER_DIR.")
    parser.add_argument(
        "--output", default=None,
        help="Where to write the summary JSON; default <ledger_dir>/"
             "token_usage_summary.json.")
    args = parser.parse_args(argv)

    if not args.ledger_dir:
        raise SystemExit(
            "no ledger directory (pass one or set SR_TOKEN_LEDGER_DIR)")

    summary = token_ledger.aggregate(args.ledger_dir)
    output = args.output or os.path.join(
        args.ledger_dir, "token_usage_summary.json")
    try:
        Path(output).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    except OSError as exc:
        print(f"warning: could not write {output}: {exc}", file=sys.stderr)

    width = max([len("component")] +
                [len(name) for name in summary["components"]] + [len("TOTAL")])
    header = (f"{'component':<{width}}  {'prompt':>14}  {'completion':>14}  "
              f"{'total':>14}  {'calls':>8}")
    print(header)
    print("-" * len(header))
    for name, c in summary["components"].items():
        print(f"{name:<{width}}  {_fmt(c['prompt_tokens']):>14}  "
              f"{_fmt(c['completion_tokens']):>14}  "
              f"{_fmt(c['total_tokens']):>14}  {_fmt(c['calls']):>8}")
    t = summary["total"]
    print("-" * len(header))
    print(f"{'TOTAL':<{width}}  {_fmt(t['prompt_tokens']):>14}  "
          f"{_fmt(t['completion_tokens']):>14}  {_fmt(t['total_tokens']):>14}  "
          f"{_fmt(t['calls']):>8}")
    print(f"\nsummary: {output}  (from {summary['ledger_files']} ledger files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
