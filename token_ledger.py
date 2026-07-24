"""Process-safe token-usage ledger for a training iteration.

Purely additive accounting: nothing here changes model behaviour or existing
artifacts. Each model call appends one JSONL line (per component, per process) to
``SR_TOKEN_LEDGER_DIR``; a post-run aggregator sums every line into a per-source
and total token report.

Why file-based: the captioner and DVD orchestrator run in GPU-worker
subprocesses, so an in-memory counter in the main process cannot see them. A
per-process JSONL (``{component}.{pid}.jsonl``) is the only seam that survives
``fork``/``spawn``. This mirrors the pattern the repo already uses for
``SR_QWEN_USAGE_LOG`` and ``SR_OPENAI_USAGE_LOG`` (whose lines this aggregator
also reads), so those two env-gated logs just point into the same directory.

record() is a no-op unless ``SR_TOKEN_LEDGER_DIR`` is set, and it never raises:
token accounting must not be able to break a training call.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Any, Mapping

_ENV_DIR = "SR_TOKEN_LEDGER_DIR"


def ledger_directory() -> str | None:
    value = os.environ.get(_ENV_DIR, "").strip()
    return value or None


def record(component: str, model: Any, usage: Mapping[str, Any] | None) -> None:
    """Append one usage line for ``component``. No-op if the ledger is off.

    ``usage`` is an OpenAI-style block (or the captioner's equivalent); only
    ``prompt_tokens`` and ``completion_tokens`` are required — anything else is
    carried through verbatim for later inspection.
    """
    directory = ledger_directory()
    if not directory or not isinstance(usage, Mapping):
        return
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None and completion is None:
        return
    try:
        os.makedirs(directory, exist_ok=True)
        row = {
            "component": str(component),
            "model": None if model is None else str(model),
            "prompt_tokens": int(prompt or 0),
            "completion_tokens": int(completion or 0),
        }
        path = os.path.join(
            directory, f"{_safe(component)}.{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    except Exception:
        # Accounting is best-effort; never propagate into the model call.
        pass


def _safe(component: str) -> str:
    return "".join(
        c if c.isalnum() or c in "-_" else "_" for c in str(component)) or "call"


def _component_of(filename: str) -> str:
    # "{component}.{pid}.jsonl" -> component; also tags the env-log files
    # (qwen.<pid>.jsonl, dvd_openai.<pid>.jsonl) by their configured prefix.
    base = os.path.basename(filename)
    if base.endswith(".jsonl"):
        base = base[: -len(".jsonl")]
    parts = base.split(".")
    if len(parts) >= 2 and parts[-1].isdigit():
        return ".".join(parts[:-1]) or "unknown"
    return base or "unknown"


def aggregate(directory: str | None = None) -> dict[str, Any]:
    """Sum every ``*.jsonl`` under the ledger dir into a token report.

    Reads the ledger's own records and the ``SR_QWEN_USAGE_LOG`` /
    ``SR_OPENAI_USAGE_LOG`` JSONL files when they are pointed at the same dir;
    all three share ``prompt_tokens``/``completion_tokens`` fields.
    """
    directory = directory or ledger_directory()
    per_component: dict[str, dict[str, int]] = defaultdict(
        lambda: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
    files = 0
    if directory and os.path.isdir(directory):
        for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
            files += 1
            component = _component_of(path)
            try:
                with open(path, encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        bucket = per_component[component]
                        bucket["prompt_tokens"] += int(
                            row.get("prompt_tokens") or 0)
                        bucket["completion_tokens"] += int(
                            row.get("completion_tokens") or 0)
                        bucket["calls"] += 1
            except OSError:
                continue
    components = {
        name: {
            **counts,
            "total_tokens": counts["prompt_tokens"] + counts["completion_tokens"],
        }
        for name, counts in sorted(per_component.items())
    }
    total_prompt = sum(c["prompt_tokens"] for c in components.values())
    total_completion = sum(c["completion_tokens"] for c in components.values())
    total_calls = sum(c["calls"] for c in components.values())
    return {
        "schema_version": "token_usage_summary_v1",
        "ledger_directory": directory,
        "ledger_files": files,
        "components": components,
        "total": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "calls": total_calls,
        },
    }
