#!/usr/bin/env python3
"""Ablate the meta-prompt UPDATER system prompt on frozen, already-generated feedback.

Holds parent meta-prompt, the ordered EpisodeFeedback set, AND the historical
compact-memory bank fixed; varies only the updater's system prompt. For each
`--variant NAME=PATH` it wraps the updater backend with
``SystemPromptOverrideUpdaterBackend`` (the repo's zero-touch swap) and calls
``LLMMetaPromptUpdater.update`` directly — NOT ``execute_meta_prompt_update_once``,
which silently drops ``historical_memories``. The baked v5 prompt is included as
the ``baseline`` variant when ``--include-baseline`` is passed.

Nothing is regenerated. Inputs are reused from a completed iteration:
  --feedback-dir   a run dir (globs feedback/*/feedback.json in sorted order)
  --parent-meta-prompt   the MetaPromptVersion JSON the iteration ran under
  --memory-bank-dir + (parent id taken from the parent file's meta_prompt_id)

Example (iteration fresh_prompt_delta_20260723_151911):
  export OPENAI_API_KEY=...
  python scripts/ablate_updater_prompt.py \
    --feedback-dir runs/fresh_prompt_delta_iteration_20260723_151911_output \
    --parent-meta-prompt runs/prompt_delta_20video_5iter_val15_20260723_151910_state/versions/meta_prompt_3a5944363e2089fe29f7.json \
    --memory-bank-dir runs/prompt_delta_20video_5iter_val15_20260723_151910_feedback_memory \
    --current-iteration-id fresh_prompt_delta_20260723_151911 \
    --variant conditional=optimization/prompts/meta_prompt_updater_system_v6_conditional.txt \
    --include-baseline \
    --api-endpoint https://api.openai.com/v1/chat/completions \
    --model-id gpt-5 --output-root runs/updater_ablation_20260724_151911_mem
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from surrogate_rollout.optimization.meta_prompt_update_execution import (
    OpenAICompatibleMetaPromptUpdaterBackend,
)
from surrogate_rollout.optimization.meta_prompt_updater import (
    LLMMetaPromptUpdater,
    META_PROMPT_UPDATER_SYSTEM_INSTRUCTION,
)
from surrogate_rollout.optimization.schemas import (
    episode_feedback_from_json,
    meta_prompt_version_from_json,
)
from surrogate_rollout.optimization.feedback_memory import (
    load_parent_feedback_memory_bank,
)
from surrogate_rollout.optimization.policies.episode_feedback_provider import (
    ExactProviderInputTokenCount,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.caption_prompt_opt.updater_backend import (
    SystemPromptOverrideUpdaterBackend,
)


class OpenAIChatTransport:
    """Two-attempt-max transport (updater may issue one corrective retry)."""

    def __init__(self, *, endpoint: str, api_key: str, timeout_seconds: int) -> None:
        if not endpoint:
            raise ValueError("endpoint must be non-empty")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in the environment")
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.call_count = 0

    def __call__(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        self.call_count += 1
        request = urllib.request.Request(
            self.endpoint,
            data=dumps_canonical(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"provider HTTP {exc.code}: {detail}") from exc
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError("provider response envelope is not an object")
        return value


def _resolve_feedback_paths(args: argparse.Namespace) -> list[str]:
    if args.feedback_dir:
        root = Path(args.feedback_dir)
        found = sorted(root.glob("*/feedback.json")) or \
            sorted((root / "feedback").glob("*/feedback.json"))
        if not found:
            raise SystemExit(f"no */feedback.json under {root}")
        return [str(p) for p in found]
    if not args.feedback_artifact:
        raise SystemExit("pass --feedback-dir or one or more --feedback-artifact")
    return list(args.feedback_artifact)


def _parse_variants(pairs: list[str]) -> dict[str, str]:
    variants: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--variant must be NAME=PATH, got {pair!r}")
        name, path = pair.split("=", 1)
        name = name.strip()
        if not name or name == "baseline":
            raise SystemExit(f"invalid variant name {name!r} (reserved: baseline)")
        variants[name] = path.strip()
    return variants


def _make_updater(args: argparse.Namespace, override_text: str | None):
    transport = OpenAIChatTransport(
        endpoint=args.api_endpoint,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        timeout_seconds=args.timeout_seconds,
    )
    import tiktoken
    try:
        encoding = tiktoken.encoding_for_model(args.model_id)
    except Exception:
        encoding = tiktoken.get_encoding("o200k_base")

    def exact_counter(messages):
        system = len(encoding.encode(messages[0]["content"]))
        user = len(encoding.encode(messages[1]["content"]))
        total = 3 + sum(
            3 + len(encoding.encode(item["role"])) +
            len(encoding.encode(item["content"])) for item in messages)
        return ExactProviderInputTokenCount(system, user, total)

    backend = OpenAICompatibleMetaPromptUpdaterBackend(
        provider="openai_api",
        model_id=args.model_id,
        maximum_output_tokens=args.maximum_output_tokens,
        generation_settings={"temperature": args.temperature},
        updater_policy_version=args.updater_policy_version,
        tokenizer_identity=encoding.name,
        exact_token_counter=exact_counter,
        context_limit=args.context_limit,
        response_transport=transport,
    )
    if override_text is not None:
        backend = SystemPromptOverrideUpdaterBackend(
            backend, override_text,
            replaced_base=META_PROMPT_UPDATER_SYSTEM_INSTRUCTION)
    updater = LLMMetaPromptUpdater(
        backend=backend, updater_policy_version=args.updater_policy_version)
    return updater, transport


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-dir")
    parser.add_argument("--feedback-artifact", action="append")
    parser.add_argument("--parent-meta-prompt", required=True,
                        help="MetaPromptVersion JSON the iteration ran under.")
    parser.add_argument("--memory-bank-dir",
                        help="Parent feedback memory bank dir. Omit to run with "
                             "no historical memory (not faithful to the loop).")
    parser.add_argument("--current-iteration-id", default="updater_ablation")
    parser.add_argument("--historical-char-budget", type=int, default=None)
    parser.add_argument("--variant", action="append", default=[],
                        help="NAME=PATH to a candidate updater system prompt.")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--api-endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--updater-policy-version", default="updater_ablation")
    parser.add_argument("--temperature", type=float, default=0.0)
    # Reasoning models (gpt-5*) spend reasoning tokens from this same budget; the
    # loop config uses 32000. A small value truncates to empty content, which the
    # updater masks as no_update (see config.py:113-115).
    parser.add_argument("--maximum-output-tokens", type=int, default=32000)
    parser.add_argument("--context-limit", type=int, default=128000)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)

    feedback_paths = _resolve_feedback_paths(args)
    feedbacks = tuple(
        episode_feedback_from_json(json.loads(Path(p).read_text(encoding="utf-8")))
        for p in feedback_paths)
    parent = meta_prompt_version_from_json(
        json.loads(Path(args.parent_meta_prompt).read_text(encoding="utf-8")))

    if args.memory_bank_dir:
        historical = load_parent_feedback_memory_bank(
            args.memory_bank_dir, parent.meta_prompt_id)
    else:
        historical = ()
    print(f"parent={parent.meta_prompt_id}  feedbacks={len(feedbacks)}  "
          f"historical_memories={len(historical)}", flush=True)

    variants = _parse_variants(args.variant)
    if variants and args.include_baseline:
        variants = {"baseline": None, **variants}  # type: ignore[dict-item]
    elif not variants:
        variants = {"baseline": None}  # type: ignore[dict-item]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    runs: dict[str, str | None] = {}

    for name, prompt_path in variants.items():
        override = None if prompt_path is None else \
            Path(prompt_path).read_text(encoding="utf-8")
        updater, transport = _make_updater(args, override)
        vdir = output_root / name
        vdir.mkdir()
        try:
            result = updater.update(
                parent, feedbacks,
                historical_memories=historical,
                current_iteration_id=args.current_iteration_id,
                historical_memory_character_budget=args.historical_char_budget)
            (vdir / "request_messages.json").write_text(
                dumps_canonical(result.request.messages), encoding="utf-8")
            (vdir / "request_payload.json").write_text(
                dumps_canonical(result.request.payload), encoding="utf-8")
            (vdir / "raw_response.txt").write_text(
                result.raw_response or "", encoding="utf-8")
            # Persist provider response + usage so finish_reason / reasoning-token
            # exhaustion is visible (updater.backend proxies these attributes).
            provider_response = getattr(
                updater.backend, "last_provider_response", None)
            usage = getattr(updater.backend, "last_usage", None)
            if provider_response is not None:
                (vdir / "provider_response.json").write_text(
                    dumps_canonical(provider_response), encoding="utf-8")
            if usage:
                (vdir / "usage.json").write_text(
                    dumps_canonical(usage), encoding="utf-8")
            (vdir / "decision.json").write_text(dumps_canonical({
                "decision": result.decision.decision,
                "candidate_meta_prompt": result.decision.candidate_meta_prompt,
                "change_summary": result.decision.change_summary,
                "rationale": result.decision.rationale,
                "supporting_feedback_ids": result.decision.supporting_feedback_ids,
                "provider_calls": transport.call_count,
            }), encoding="utf-8")
            runs[name] = result.decision.candidate_meta_prompt
            # A no_update whose rationale is the double-rejection wrapper is not a
            # genuine no_update: the provider returned unusable/empty content.
            masked = (result.decision.decision == "no_update"
                      and "rejected twice by the updater contract"
                      in (result.decision.rationale or ""))
            if masked:
                finish = None
                if isinstance(provider_response, dict):
                    ch = provider_response.get("choices") or [{}]
                    finish = ch[0].get("finish_reason")
                print(f"\n===== VARIANT: {name}  PROVIDER ERROR (masked as "
                      f"no_update) =====", file=sys.stderr)
                print(f"  finish_reason={finish}  usage={usage}", file=sys.stderr)
                print(f"  {result.decision.rationale}", file=sys.stderr)
                runs[name] = None
                continue
            print(f"\n===== VARIANT: {name}  (decision={result.decision.decision}, "
                  f"calls={transport.call_count}) =====")
            print(result.decision.candidate_meta_prompt or "(no_update)")
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            runs[name] = None
            print(f"\n===== VARIANT: {name}  FAILED: {exc} =====", file=sys.stderr)

    (output_root / "ablation_summary.json").write_text(dumps_canonical({
        "parent_meta_prompt_id": parent.meta_prompt_id,
        "feedback_artifacts": feedback_paths,
        "memory_bank_dir": args.memory_bank_dir,
        "historical_memory_count": len(historical),
        "current_iteration_id": args.current_iteration_id,
        "variants": {n: (variants[n] or "baked_v5") for n in variants},
        "candidates": runs,
    }), encoding="utf-8")
    print(f"\nsummary: {output_root / 'ablation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
