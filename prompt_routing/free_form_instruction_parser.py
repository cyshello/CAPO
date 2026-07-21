"""Strict plain-text parser for generated caption instructions.

The generator produces one string-valued instruction, so its runtime output
contract does not use a JSON envelope.  After surrounding whitespace is
removed, every non-empty response is preserved literally.  JSON-looking text,
Markdown fences, and near-JSON are not interpreted, extracted, or repaired.

This module is intentionally isolated: no model calls, no config reads, no
imports from the routing or optimization packages.
"""

from __future__ import annotations

from dataclasses import dataclass

PARSER_VERSION = "free_form_instruction_plain_text_parser_v2"


@dataclass(frozen=True)
class InstructionParseOutcome:
    """Result of one deterministic parse attempt over a raw generator reply."""

    ok: bool
    instruction: str | None
    parser_path: str
    error: str | None = None


def parse_generated_instruction(raw: str | None) -> InstructionParseOutcome:
    """Return the complete non-empty response as an opaque instruction."""
    text = (raw or "").strip()
    if not text:
        return InstructionParseOutcome(
            ok=False, instruction=None, parser_path="empty",
            error="generator returned an empty response")
    return InstructionParseOutcome(
        ok=True, instruction=text, parser_path="plain_text")
