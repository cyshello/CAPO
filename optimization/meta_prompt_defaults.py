"""Canonical initial meta-prompt for prompt-delta training entry points."""

from __future__ import annotations

import json
from pathlib import Path

from surrogate_rollout.optimization.schemas import (
    MetaPromptVersion,
    meta_prompt_version_from_json,
)


INITIAL_META_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "init_meta_prompt.json")


def load_initial_meta_prompt() -> MetaPromptVersion:
    value = json.loads(INITIAL_META_PROMPT_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("initial meta-prompt artifact must be a JSON object")
    return meta_prompt_version_from_json(value)


def resolve_meta_prompt_artifact_path(value: str | None) -> Path:
    """Resolve an explicit parent or the repository-owned initial parent."""
    path = Path(value).resolve() if value else INITIAL_META_PROMPT_PATH.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("meta-prompt artifact must be a JSON object")
    meta_prompt_version_from_json(parsed)
    return path
