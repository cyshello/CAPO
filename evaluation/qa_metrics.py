"""Single shared answer parser + QA scoring (CLAUDE.md §24: one parser for all
conditions; parsing failures distinguishable from semantic failures)."""

from __future__ import annotations

import re

_LETTER = re.compile(r"[A-E]")


def parse_letter(answer: str | None) -> str | None:
    """First option letter A-E — identical rule to
    prompt_sensitivity/run_video_qas.py (baseline parity)."""
    if not answer:
        return None
    m = _LETTER.search(answer.strip())
    return m.group(0) if m else None


def score_mcq(raw_answer: str | None, gold: str | None) -> tuple[float, str | None, str]:
    """Return (score, parsed, failure_kind). failure_kind is one of
    "" (correct), "wrong_answer", "parse_failure", "no_gold"."""
    parsed = parse_letter(raw_answer)
    if gold is None:
        return 0.0, parsed, "no_gold"
    if parsed is None:
        return 0.0, None, "parse_failure"
    return (1.0, parsed, "") if parsed == gold else (0.0, parsed, "wrong_answer")
