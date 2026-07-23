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
    # The repository-owned parent is the starting point of the optimization, so
    # its text changes between experiments. What must hold is that its id is
    # derived from its own text: promoted versions and the per-parent feedback
    # memory bank are keyed by that id, and a stale id would collide with the
    # artifacts of the text it used to carry.
    assert version.meta_prompt_id == \
        "meta_prompt_" + sha256_text(version.text)[:20]
    assert version.parent_meta_prompt_id is None
    assert version.status == "parent"
    assert version.text.strip()
    assert resolve_meta_prompt_artifact_path(None) == \
        INITIAL_META_PROMPT_PATH.resolve()


def test_generator_fallback_template_is_the_baseline_repo_parity_text():
    """The two texts are allowed to diverge, and each has one job.

    `smrb.META_PROMPT_TEXT` stays pinned to the private baseline repo so the
    parity condition remains reproducible; it is only the fallback used when no
    parent is supplied. Every optimization run passes its parent explicitly, so
    the repository-owned artifact is what actually drives those runs.
    """
    assert sha256_text(smrb.META_PROMPT_TEXT) == smrb.META_PROMPT_SHA256
    value = json.loads(INITIAL_META_PROMPT_PATH.read_text(encoding="utf-8"))
    assert value["text"].strip()


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
