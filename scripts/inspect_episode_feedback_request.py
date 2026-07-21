#!/usr/bin/env python3
"""Read-only D2 inspection of one saved legacy intervention bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from surrogate_rollout.optimization.legacy_intervention_adapter import (
    legacy_property_intervention_to_episode,
)
from surrogate_rollout.optimization.llm_episode_feedback import (
    LegacyEpisodeFeedbackArtifactResolver,
)
from surrogate_rollout.optimization.policies.episode_feedback_provider import (
    ExactProviderInputTokenCount,
    OpenAICompatibleEpisodeFeedbackProviderAdapter,
    prepare_and_measure,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and exactly tokenize model_compact without a call")
    parser.add_argument("--intervention-result", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--parent-meta-prompt-id", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--tokenizer-identity", required=True)
    parser.add_argument("--context-limit", required=True, type=int)
    parser.add_argument("--maximum-output-tokens", required=True, type=int)
    parser.add_argument("--generation-settings-json", required=True)
    parser.add_argument("--feedback-policy-version", required=True)
    return parser.parse_args()


def _source_files(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(sorted({
        item.resolve() for path in paths for item in path.parent.rglob("*")
        if item.is_file()
    }, key=str))


def _aggregate_sha256(paths: tuple[Path, ...]) -> str:
    rows = "".join(
        f"{path}:{hashlib.sha256(path.read_bytes()).hexdigest()}\n"
        for path in paths)
    return hashlib.sha256(rows.encode()).hexdigest()


def _load_generation_settings(raw: str) -> Mapping[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or not value:
        raise ValueError("generation settings must be a non-empty JSON object")
    return value


def main() -> int:
    args = _parse_args()
    intervention = Path(args.intervention_result).resolve()
    baseline = Path(args.baseline_manifest).resolve()
    files = _source_files((intervention, baseline))
    before = _aggregate_sha256(files)

    # Local-only prevents inspection from selecting or downloading a tokenizer.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True)
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            "the explicitly selected tokenizer has no chat_template; exact "
            "provider-message token counting is not configured")

    def exact_token_counter(
        messages: tuple[Mapping[str, str], ...],
    ) -> ExactProviderInputTokenCount:
        system_tokens = len(tokenizer.encode(
            messages[0]["content"], add_special_tokens=False))
        user_tokens = len(tokenizer.encode(
            messages[1]["content"], add_special_tokens=False))
        formatted_tokens = tokenizer.apply_chat_template(
            [dict(item) for item in messages], tokenize=True,
            add_generation_prompt=True)
        return ExactProviderInputTokenCount(
            system_prompt_tokens=system_tokens,
            user_payload_tokens=user_tokens,
            total_input_tokens=len(formatted_tokens),
        )

    def forbidden_transport(_: Mapping[str, Any]) -> str:
        raise AssertionError("read-only inspection must not call a provider")

    adapter = OpenAICompatibleEpisodeFeedbackProviderAdapter(
        provider=args.provider,
        model_id=args.model_id,
        tokenizer_identity=args.tokenizer_identity,
        exact_token_counter=exact_token_counter,
        context_limit=args.context_limit,
        maximum_output_tokens=args.maximum_output_tokens,
        generation_settings=_load_generation_settings(
            args.generation_settings_json),
        feedback_policy_version=args.feedback_policy_version,
        response_transport=forbidden_transport,
    )
    episode = legacy_property_intervention_to_episode(
        intervention_result_path=str(intervention),
        baseline_video_manifest_path=str(baseline),
        parent_meta_prompt_id=args.parent_meta_prompt_id)
    inspection = prepare_and_measure(
        episode, provider_adapter=adapter,
        artifact_resolver=LegacyEpisodeFeedbackArtifactResolver())
    after = _aggregate_sha256(files)
    if before != after:
        raise RuntimeError("source artifact aggregate SHA-256 changed")
    stats = inspection.token_statistics
    print(json.dumps({
        "provider": args.provider,
        "model_id": args.model_id,
        "tokenizer_identity": args.tokenizer_identity,
        "episode_id": episode.episode_id,
        "clip_count": len(episode.clips),
        "qa_count": len(episode.qa_outcomes),
        "system_prompt_tokens": stats.system_prompt_tokens,
        "user_payload_tokens": stats.user_payload_tokens,
        "total_input_tokens": stats.total_input_tokens,
        "reserved_output_tokens": stats.reserved_output_tokens,
        "context_limit": stats.context_limit,
        "remaining_tokens": stats.remaining_tokens,
        "fits_context": stats.fits_context,
        "request_identity": inspection.prepared_request.request_identity,
        "model_payload_hash": (
            inspection.model_compact_request.model_payload_hash),
        "source_aggregate_sha256_before": before,
        "source_aggregate_sha256_after": after,
        "provider_calls": adapter.call_count,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
