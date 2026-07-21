#!/usr/bin/env python3
"""Run exactly one prompt-delta episode-feedback provider request.

This standalone operator command does not integrate feedback into runtime state.
It writes only to a newly created, explicitly selected output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from surrogate_rollout.optimization.legacy_intervention_adapter import (
    legacy_property_intervention_to_episode,
)
from surrogate_rollout.optimization.episode_feedback import (
    EPISODE_FEEDBACK_ELIGIBILITY_POLICY_VERSION,
    evaluate_episode_feedback_eligibility,
)
from surrogate_rollout.optimization.llm_episode_feedback import (
    LLMEpisodeFeedbackGenerator,
    LegacyEpisodeFeedbackArtifactResolver,
    build_model_compact_episode_feedback_request,
)
from surrogate_rollout.optimization.policies.episode_feedback_provider import (
    ExactProviderInputTokenCount,
    OpenAICompatibleEpisodeFeedbackProviderAdapter,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send one model_compact episode-feedback request with no retry, "
            "repair, fallback, persistence, updater, or runtime integration."))
    parser.add_argument("--intervention-result", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--parent-meta-prompt-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provider", required=True, choices=("openai_api",))
    parser.add_argument("--api-endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--request-representation", required=True,
        choices=("model_compact",))
    parser.add_argument("--context-limit", required=True, type=int)
    parser.add_argument("--maximum-output-tokens", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--feedback-policy-version", required=True)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    return parser.parse_args()


def _write(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


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


def _token_counter(model_id: str):
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model_id)
    except Exception as exc:
        raise RuntimeError(
            f"no local tokenizer mapping is available for {model_id!r}: {exc}") \
            from exc

    def count(
        messages: tuple[Mapping[str, str], ...],
    ) -> ExactProviderInputTokenCount:
        system_tokens = len(encoding.encode(messages[0]["content"]))
        user_tokens = len(encoding.encode(messages[1]["content"]))
        # OpenAI's documented text-chat accounting: three framing tokens per
        # message plus three tokens priming the assistant response.
        total = 3 + sum(
            3 + len(encoding.encode(item["role"])) +
            len(encoding.encode(item["content"]))
            for item in messages)
        return ExactProviderInputTokenCount(
            system_prompt_tokens=system_tokens,
            user_payload_tokens=user_tokens,
            total_input_tokens=total)

    return count, encoding.name


class _SingleOpenAITransport:
    def __init__(
        self, *, endpoint: str, api_key: str, timeout: int,
        output_dir: Path,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.output_dir = output_dir
        self.call_count = 0
        self.api_response: Mapping[str, Any] | None = None

    def __call__(self, request_body: Mapping[str, Any]) -> str:
        if self.call_count:
            raise RuntimeError("provider transport is limited to exactly one call")
        self.call_count += 1
        canonical_request = dumps_canonical(request_body)
        _write(self.output_dir / "request.json", canonical_request)
        request = urllib.request.Request(
            self.endpoint,
            data=canonical_request.encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                    request, timeout=self.timeout) as response:
                raw_envelope = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw_error = exc.read().decode("utf-8", errors="replace")
            _write(self.output_dir / "raw_error.txt", raw_error)
            raise RuntimeError(
                f"OpenAI API returned HTTP {exc.code}: {raw_error}") from exc
        except Exception as exc:
            _write(
                self.output_dir / "raw_error.txt",
                f"{type(exc).__name__}: {exc}")
            raise
        _write(self.output_dir / "provider_response.json", raw_envelope)
        payload = json.loads(raw_envelope)
        if not isinstance(payload, dict):
            raise RuntimeError("provider response envelope is not a JSON object")
        self.api_response = payload
        _write(
            self.output_dir / "usage.json",
            dumps_canonical(payload.get("usage", {})))
        raw_response = payload["choices"][0]["message"]["content"]
        if not isinstance(raw_response, str):
            raise RuntimeError("provider response content is not a string")
        _write(self.output_dir / "raw_response.txt", raw_response)
        return raw_response


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(
            f"output directory already exists; refusing overwrite: {output_dir}",
            file=sys.stderr)
        return 2

    intervention = Path(args.intervention_result).resolve()
    baseline = Path(args.baseline_manifest).resolve()
    source_files = _source_files((intervention, baseline))
    source_before = _aggregate_sha256(source_files)
    manifest: dict[str, Any] = {
        "status": "preparing",
        "provider": args.provider,
        "model": args.model_id,
        "request_representation": args.request_representation,
        "maximum_output_tokens": args.maximum_output_tokens,
        "temperature": args.temperature,
        "feedback_policy_version": args.feedback_policy_version,
        "source_aggregate_sha256_before": source_before,
        "source_aggregate_sha256_after": None,
        "provider_call_count": 0,
    }
    _write(output_dir / "manifest.json", dumps_canonical(manifest))

    transport: _SingleOpenAITransport | None = None
    episode = None
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set in the environment")
        episode = legacy_property_intervention_to_episode(
            intervention_result_path=str(intervention),
            baseline_video_manifest_path=str(baseline),
            parent_meta_prompt_id=args.parent_meta_prompt_id)
        request = build_model_compact_episode_feedback_request(
            episode, artifact_resolver=LegacyEpisodeFeedbackArtifactResolver())
        counter, tokenizer_identity = _token_counter(args.model_id)
        transport = _SingleOpenAITransport(
            endpoint=args.api_endpoint,
            api_key=api_key,
            timeout=args.timeout_seconds,
            output_dir=output_dir)
        adapter = OpenAICompatibleEpisodeFeedbackProviderAdapter(
            provider=args.provider,
            model_id=args.model_id,
            tokenizer_identity=tokenizer_identity,
            exact_token_counter=counter,
            context_limit=args.context_limit,
            maximum_output_tokens=args.maximum_output_tokens,
            generation_settings={"temperature": args.temperature},
            feedback_policy_version=args.feedback_policy_version,
            response_transport=transport)
        prepared, token_statistics = adapter.prepare_messages(
            request.system_instruction, request.user_request)
        _write(output_dir / "request.json", prepared.serialized_request)
        manifest.update({
            "status": "prepared",
            "episode_id": episode.episode_id,
            "request_payload_hash": request.payload_hash,
            "canonical_provider_request_sha256": hashlib.sha256(
                prepared.serialized_request.encode()).hexdigest(),
            "token_statistics": token_statistics,
        })
        _write(output_dir / "manifest.json", dumps_canonical(manifest))
        generator = LLMEpisodeFeedbackGenerator(
            response_provider=adapter,
            artifact_resolver=LegacyEpisodeFeedbackArtifactResolver(),
            policy_version=args.feedback_policy_version,
            request_representation=args.request_representation)
        result = generator.generate_with_trace(episode)
        eligibility = evaluate_episode_feedback_eligibility(
            result.feedback, episode)
        _write(
            output_dir / "semantic_eligibility.json",
            dumps_canonical(eligibility))
        if not eligibility.eligible:
            raise RuntimeError(
                "episode feedback failed semantic eligibility: "
                f"{list(eligibility.reasons)!r}")
        _write(
            output_dir / "parsed_feedback.json",
            dumps_canonical(result.feedback))
        api_response = transport.api_response or {}
        usage = api_response.get("usage", {})
        _write(output_dir / "usage.json", dumps_canonical(usage))
        source_after = _aggregate_sha256(source_files)
        if source_before != source_after:
            raise RuntimeError("source artifact aggregate SHA-256 changed")
        manifest.update({
            "status": "success",
            "episode_id": episode.episode_id,
            "provider_call_count": transport.call_count,
            "adapter_call_count": adapter.call_count,
            "provider_model": api_response.get("model"),
            "api_response_id": api_response.get("id"),
            "request_payload_hash": request.payload_hash,
            "canonical_provider_request_sha256": hashlib.sha256(
                prepared.serialized_request.encode()).hexdigest(),
            "raw_response_characters": len(result.raw_response),
            "feedback_id": result.feedback.feedback_id,
            "observation_count": len(result.feedback.observations),
            "counterevidence_count": len(result.feedback.counterevidence),
            "supporting_id_validation": "passed",
            "semantic_eligibility": "passed",
            "semantic_eligibility_artifact": str(
                output_dir / "semantic_eligibility.json"),
            "token_statistics": token_statistics,
            "source_aggregate_sha256_after": source_after,
        })
        _write(output_dir / "manifest.json", dumps_canonical(manifest))
        print(dumps_canonical(manifest))
        return 0
    except Exception as exc:
        source_after = _aggregate_sha256(source_files)
        if episode is not None and not (
                output_dir / "semantic_eligibility.json").exists():
            _write(
                output_dir / "semantic_eligibility.json",
                dumps_canonical({
                    "schema_version": (
                        EPISODE_FEEDBACK_ELIGIBILITY_POLICY_VERSION),
                    "feedback_id": None,
                    "episode_id": episode.episode_id,
                    "feedback_sha256": None,
                    "eligible": False,
                    "reasons": [f"{type(exc).__name__}: {exc}"],
                }))
        manifest.update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "semantic_eligibility": "failed",
            "provider_call_count": transport.call_count if transport else 0,
            "source_aggregate_sha256_after": source_after,
        })
        _write(output_dir / "manifest.json", dumps_canonical(manifest))
        if not (output_dir / "raw_error.txt").exists():
            _write(output_dir / "raw_error.txt", manifest["error"])
        if not (output_dir / "usage.json").exists():
            _write(output_dir / "usage.json", dumps_canonical({}))
        print(manifest["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
