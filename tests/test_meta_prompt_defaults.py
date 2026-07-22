"""Repository-owned initial meta-prompt and training default tests."""

import json

from scripts.run_meta_prompt_update_once import _parse_args as updater_args
from surrogate_rollout.optimization.meta_prompt_defaults import (
    INITIAL_META_PROMPT_PATH,
    load_initial_meta_prompt,
    resolve_meta_prompt_artifact_path,
)
from surrogate_rollout.prompt_routing import static_meta_replace_body as smrb
from surrogate_rollout.schemas import sha256_text


def test_initial_meta_prompt_is_repository_owned_and_schema_valid():
    assert INITIAL_META_PROMPT_PATH.name == "init_meta_prompt.json"
    assert INITIAL_META_PROMPT_PATH.parent.name == "prompts"
    assert INITIAL_META_PROMPT_PATH.parent.parent.name == "optimization"
    version = load_initial_meta_prompt()
    # The repository-owned parent is the static replace-body meta prompt, the
    # same text the private baseline repo runs (pinned by sha on both sides).
    assert version.meta_prompt_id == "meta_prompt_4e7ca02d27e84339e6e5"
    assert version.parent_meta_prompt_id is None
    assert version.status == "parent"
    assert sha256_text(version.text) == smrb.META_PROMPT_SHA256
    assert resolve_meta_prompt_artifact_path(None) == \
        INITIAL_META_PROMPT_PATH.resolve()


def test_runtime_default_template_uses_canonical_initial_prompt_text():
    value = json.loads(INITIAL_META_PROMPT_PATH.read_text(encoding="utf-8"))
    assert smrb.META_PROMPT_TEXT == value["text"]


def test_updater_training_cli_parent_is_optional_and_defaults_later():
    parsed = updater_args([
        "--provider", "openai_api",
        "--api-endpoint", "https://example.invalid/v1/chat/completions",
        "--model-id", "fixture-model",
        "--feedback-artifact", "feedback.json",
        "--updater-policy-version", "fixture-policy",
        "--temperature", "0.0",
        "--maximum-output-tokens", "1",
        "--context-limit", "128000",
        "--candidate-created-at", "2026-07-22T00:00:00Z",
        "--output-dir", "output",
        "--timeout-seconds", "1",
    ])
    assert parsed.parent_meta_prompt is None


def test_explicit_meta_prompt_still_overrides_default(tmp_path):
    explicit = tmp_path / "explicit.json"
    explicit.write_text(INITIAL_META_PROMPT_PATH.read_text(encoding="utf-8"))
    assert resolve_meta_prompt_artifact_path(str(explicit)) == explicit.resolve()
