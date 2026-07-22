#!/usr/bin/env python3
"""Run one strict meta-prompt updater call into a new local E2 directory."""

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
    MetaPromptUpdateExecutionError,
    OpenAICompatibleMetaPromptUpdaterBackend,
    execute_meta_prompt_update_once,
)
from surrogate_rollout.optimization.meta_prompt_defaults import (
    resolve_meta_prompt_artifact_path,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.optimization.policies.episode_feedback_provider import (
    ExactProviderInputTokenCount,
)


class ProviderTransportError(RuntimeError):
    def __init__(self, reason: str, *, raw_error: str) -> None:
        self.raw_error = raw_error
        super().__init__(reason)


class SingleOpenAIChatTransport:
    """Exactly-one-attempt transport; no retry, repair, or fallback."""

    def __init__(self, *, endpoint: str, api_key: str, timeout_seconds: int) -> None:
        if not endpoint:
            raise ValueError("endpoint must be non-empty")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in the environment")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.call_count = 0

    def __call__(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.call_count:
            raise RuntimeError("provider transport is limited to one call")
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
            raw_error = exc.read().decode("utf-8", errors="replace")
            raise ProviderTransportError(
                f"provider returned HTTP {exc.code}", raw_error=raw_error) from exc
        except BaseException as exc:
            raise ProviderTransportError(
                f"provider transport failed: {type(exc).__name__}: {exc}",
                raw_error=f"{type(exc).__name__}: {exc}") from exc
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderTransportError(
                "provider response envelope is not JSON", raw_error=raw) from exc
        if not isinstance(value, dict):
            raise ProviderTransportError(
                "provider response envelope is not an object", raw_error=raw)
        return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call one injected strict-JSON meta-prompt updater and atomically "
            "write a provisional or no-update review artifact. No promotion, "
            "retry, repair, fallback, or runtime integration is performed."))
    parser.add_argument("--provider", required=True, choices=("openai_api",))
    parser.add_argument("--api-endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--parent-meta-prompt",
        help=("MetaPromptVersion JSON. Omit to use "
              "optimization/prompts/init_meta_prompt.json."))
    parser.add_argument(
        "--feedback-artifact", required=True, action="append",
        help="Ordered EpisodeFeedback JSON; repeat to preserve input order.")
    parser.add_argument("--updater-policy-version", required=True)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--maximum-output-tokens", required=True, type=int)
    parser.add_argument("--context-limit", required=True, type=int)
    parser.add_argument("--candidate-created-at", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        transport = SingleOpenAIChatTransport(
            endpoint=args.api_endpoint,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            timeout_seconds=args.timeout_seconds,
        )
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(args.model_id)
        except Exception as exc:
            raise ValueError(
                f"exact tokenizer is unavailable for {args.model_id!r}: {exc}") \
                from exc

        def exact_counter(messages):
            system = len(encoding.encode(messages[0]["content"]))
            user = len(encoding.encode(messages[1]["content"]))
            total = 3 + sum(
                3 + len(encoding.encode(item["role"])) +
                len(encoding.encode(item["content"])) for item in messages)
            return ExactProviderInputTokenCount(system, user, total)

        backend = OpenAICompatibleMetaPromptUpdaterBackend(
            provider=args.provider,
            model_id=args.model_id,
            maximum_output_tokens=args.maximum_output_tokens,
            generation_settings={"temperature": args.temperature},
            updater_policy_version=args.updater_policy_version,
            tokenizer_identity=encoding.name,
            exact_token_counter=exact_counter,
            context_limit=args.context_limit,
            response_transport=transport,
        )
        parent_path = resolve_meta_prompt_artifact_path(
            args.parent_meta_prompt)
        result = execute_meta_prompt_update_once(
            parent_artifact_path=parent_path,
            feedback_artifact_paths=args.feedback_artifact,
            output_directory=args.output_dir,
            backend=backend,
            updater_policy_version=args.updater_policy_version,
            candidate_created_at=args.candidate_created_at,
        )
    except (MetaPromptUpdateExecutionError, OSError, TypeError, ValueError) as exc:
        print(f"meta-prompt update failed: {exc}", file=sys.stderr)
        return 1
    print(dumps_canonical({
        "decision": result.decision.decision,
        "candidate_meta_prompt_id": result.candidate_meta_prompt_id,
        "candidate_status": result.candidate_status,
        "output_directory": os.path.abspath(args.output_dir),
        "provider_call_count": backend.call_count,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
