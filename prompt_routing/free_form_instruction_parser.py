"""Deterministic parser for free-form caption-instruction generator output.

The free-form baseline generator (free_form_instruction_generator.py) prompts
the local VLM to return ``{"caption_instruction": "..."}``, but Qwen2.5-VL
JSON output may be unreliable. This module implements the documented fallback
ladder — no LLM repair call, no retries, no silent behavior: the chosen path
is always recorded in ``parser_path`` and an unusable response is an explicit
failure outcome, never a hidden default.

Fallback order (first success wins):

1. ``direct_json``    — strict ``json.loads`` of the whole response;
2. ``fenced_json``    — same after stripping one Markdown code fence;
3. ``near_json_field``— extract the ``caption_instruction`` string value from
   near-JSON text (regex over an escaped JSON string literal);
4. ``plain_text``     — the cleaned non-empty response used verbatim;
5. ``empty``          — no usable text: ``ok=False`` (caller decides whether
   to raise; the shared pipeline treats it as an explicit generation failure).

This module is intentionally isolated: no model calls, no config reads, no
imports from the routing or optimization packages.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

INSTRUCTION_KEY = "caption_instruction"
PARSER_VERSION = "free_form_instruction_parser_v1"

# A JSON string literal (handles escaped quotes/backslashes) after the key.
_NEAR_JSON_FIELD_RE = re.compile(
    r'"caption_instruction"\s*:\s*"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class InstructionParseOutcome:
    """Result of one deterministic parse attempt over a raw generator reply."""

    ok: bool
    instruction: str | None
    parser_path: str
    error: str | None = None


def _strip_markdown_fence(text: str) -> str:
    """Remove one surrounding ``` fence (with optional language tag), mirroring
    the caption-output normalization convention in history_aware_baseline."""
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _instruction_from_json_value(value: object) -> str | None:
    """The instruction string if `value` matches the requested contract."""
    if not isinstance(value, dict):
        return None
    instruction = value.get(INSTRUCTION_KEY)
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()
    return None


def parse_generated_instruction(raw: str | None) -> InstructionParseOutcome:
    """Parse a raw generator response through the documented fallback ladder."""
    text = (raw or "").strip()
    if not text:
        return InstructionParseOutcome(
            ok=False, instruction=None, parser_path="empty",
            error="generator returned an empty response")

    # 1. direct strict JSON.
    try:
        instruction = _instruction_from_json_value(json.loads(text))
        if instruction is not None:
            return InstructionParseOutcome(
                ok=True, instruction=instruction, parser_path="direct_json")
    except json.JSONDecodeError:
        pass

    # 2. strip one Markdown fence and parse again.
    if text.startswith("```"):
        unfenced = _strip_markdown_fence(text)
        if unfenced:
            try:
                instruction = _instruction_from_json_value(json.loads(unfenced))
                if instruction is not None:
                    return InstructionParseOutcome(
                        ok=True, instruction=instruction,
                        parser_path="fenced_json")
            except json.JSONDecodeError:
                pass
            text = unfenced  # later fallbacks work on the cleaned text

    # 3. extract the field value from near-JSON text.
    match = _NEAR_JSON_FIELD_RE.search(text)
    if match:
        try:
            extracted = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            extracted = ""
        if isinstance(extracted, str) and extracted.strip():
            return InstructionParseOutcome(
                ok=True, instruction=extracted.strip(),
                parser_path="near_json_field")

    # 4. cleaned non-empty plain text used as the instruction.
    if text:
        return InstructionParseOutcome(
            ok=True, instruction=text, parser_path="plain_text")

    # 5. nothing usable after cleaning.
    return InstructionParseOutcome(
        ok=False, instruction=None, parser_path="empty",
        error="generator response had no usable text after cleaning")
