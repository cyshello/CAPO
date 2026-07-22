#!/usr/bin/env python3
"""Generate one provider-authored memory for each legacy detailed feedback."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
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

from surrogate_rollout.optimization.feedback_memory import (
    append_parent_feedback_memory_bank,
    build_episode_feedback_memory_record,
    load_parent_feedback_memory_bank,
)
from surrogate_rollout.optimization.schemas import (
    episode_feedback_from_json,
    intervention_episode_from_json,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json


MIGRATION_SYSTEM_INSTRUCTION = """Write one short historical experience from the supplied detailed intervention feedback and deterministic episode facts.

Use roughly 2–3 lines of natural language. First describe a situation observable from current frames and bounded preceding caption history. Then describe the attempted description change and observed downstream result. Preserve uncertainty when caption contribution was not confirmed. Omit QA IDs, segment IDs, long reasoning, confidence, and repeated recommendations. Do not assert benefit or causality without supporting evidence.

Return exactly one strict JSON object with compact_memory_text, whose value is a string or null. Do not return Markdown or additional fields."""


def _sha(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: str) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _args(argv=None):
    parser = argparse.ArgumentParser(
        description=("Create one provider-authored compact memory per legacy "
                     "EpisodeFeedback and append a parent-scoped bank. No "
                     "semantic validation, repair, retry, or source mutation."))
    parser.add_argument("--iteration-id", required=True)
    parser.add_argument("--episode-artifact", action="append", required=True)
    parser.add_argument("--feedback-artifact", action="append", required=True)
    parser.add_argument("--memory-bank-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provider", choices=("openai_api",), required=True)
    parser.add_argument("--api-endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--maximum-output-tokens", type=int, required=True)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _response_schema() -> Mapping[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["compact_memory_text"],
        "properties": {
            "compact_memory_text": {"type": ["string", "null"]},
        },
    }


def _migration_payload(episode, feedback) -> Mapping[str, Any]:
    detailed = json.loads(dumps_canonical(feedback))
    detailed.pop("compact_memory_text", None)
    return json.loads(dumps_canonical({
        "schema_version": "episode_feedback_memory_migration_request_v1",
        "episode": {
            "episode_id": episode.episode_id,
            "video_id": episode.video_id,
            "parent_meta_prompt_id": episode.parent_meta_prompt_id,
            "candidate_id": episode.prompt_delta.delta_id,
            "prompt_delta_instruction": episode.prompt_delta.instruction,
            "clips": [{
                "segment_id": clip.segment_id,
                "time_range": clip.time_range,
                "history_snapshot": clip.history_snapshot,
                "base_prompt": clip.base_prompt,
                "baseline_caption": clip.baseline_caption,
                "intervention_caption": clip.intervention_caption,
            } for clip in episode.clips],
            "qa_outcomes": episode.qa_outcomes,
        },
        "detailed_feedback": detailed,
    }))


def _provider_body(args, payload) -> Mapping[str, Any]:
    return json.loads(dumps_canonical({
        "model": args.model_id,
        "messages": [
            {"role": "system", "content": MIGRATION_SYSTEM_INSTRUCTION},
            {"role": "user", "content": dumps_canonical(payload)},
        ],
        "max_tokens": args.maximum_output_tokens,
        "temperature": args.temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "episode_feedback_memory_migration_v1",
                "strict": True,
                "schema": _response_schema(),
            },
        },
    }))


def _call_once(args, body: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY must be set in the environment")
    request = urllib.request.Request(
        args.api_endpoint, data=dumps_canonical(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(
                request, timeout=args.timeout_seconds) as response:
            envelope_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"provider HTTP {exc.code}: {raw}") from exc
    envelope = json.loads(envelope_text)
    raw = envelope["choices"][0]["message"]["content"]
    if not isinstance(raw, str):
        raise TypeError("provider message content must be a string")
    return raw, envelope


def _parse_memory(raw: str) -> str | None:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"compact_memory_text"}:
        raise ValueError("memory response must contain only compact_memory_text")
    memory = value["compact_memory_text"]
    if memory is not None and not isinstance(memory, str):
        raise TypeError("compact_memory_text must be a string or null")
    return memory.strip() or None if isinstance(memory, str) else None


def main(argv=None) -> int:
    args = _args(argv)
    try:
        if len(args.episode_artifact) != len(args.feedback_artifact):
            raise ValueError("episode and feedback artifact counts must match")
        if args.maximum_output_tokens <= 0 or args.timeout_seconds <= 0:
            raise ValueError("token and timeout settings must be positive")
        output = os.path.abspath(args.output_dir)
        manifest_path = os.path.join(output, "migration_manifest.json")
        if os.path.isfile(manifest_path):
            saved = _object(manifest_path)
            if saved.get("status") == "completed":
                print(dumps_canonical(saved))
                return 0
            raise RuntimeError("migration manifest already exists but is incomplete")
        pairs = tuple(zip(args.episode_artifact, args.feedback_artifact))
        source_paths = tuple(os.path.abspath(path) for pair in pairs for path in pair)
        before = {path: _sha(path) for path in source_paths}
        requests = []
        prepared = []
        for index, (episode_path, feedback_path) in enumerate(pairs):
            episode = intervention_episode_from_json(_object(episode_path))
            feedback = episode_feedback_from_json(_object(feedback_path))
            if feedback.episode_id != episode.episode_id:
                raise ValueError("feedback and episode IDs differ")
            payload = _migration_payload(episode, feedback)
            body = _provider_body(args, payload)
            request_path = os.path.join(output, f"{index:03d}_{episode.episode_id}",
                                        "request.json")
            requests.append({
                "episode": episode, "feedback": feedback,
                "feedback_path": os.path.abspath(feedback_path),
                "body": body, "request_path": request_path,
            })
            prepared.append({
                "episode_id": episode.episode_id,
                "feedback_id": feedback.feedback_id,
                "request_path": request_path,
                "request_sha256": sha256_json(body),
                "request_characters": len(dumps_canonical(body)),
            })
            os.makedirs(os.path.dirname(request_path), exist_ok=True)
            _atomic_write_text(request_path, dumps_canonical(body) + "\n")
        if args.dry_run:
            after = {path: _sha(path) for path in source_paths}
            if before != after:
                raise RuntimeError("source artifacts changed during preflight")
            manifest = {
                "schema_version": "episode_feedback_memory_migration_v2",
                "status": "dry_run",
                "iteration_id": args.iteration_id,
                "provider": args.provider,
                "model_id": args.model_id,
                "policy_version": args.policy_version,
                "prepared_requests": prepared,
                "provider_call_count": 0,
                "source_hashes_before": before,
                "source_hashes_after": after,
            }
            _atomic_write_text(manifest_path, dumps_canonical(manifest) + "\n")
            print(dumps_canonical(manifest))
            return 0
        records = []
        response_rows = []
        for index, item in enumerate(requests):
            raw, envelope = _call_once(args, item["body"])
            stage = os.path.dirname(item["request_path"])
            _atomic_write_text(os.path.join(stage, "raw_response.json"), raw + "\n")
            memory_text = _parse_memory(raw)
            feedback = dataclasses.replace(
                item["feedback"], compact_memory_text=memory_text)
            record = build_episode_feedback_memory_record(
                feedback=feedback, episode=item["episode"],
                iteration_id=args.iteration_id,
                feedback_artifact_ref=item["feedback_path"])
            if record is not None:
                records.append(record)
            response_rows.append({
                "episode_id": item["episode"].episode_id,
                "feedback_id": feedback.feedback_id,
                "memory_id": record.memory_id if record else None,
                "memory_text": memory_text,
                "usage": envelope.get("usage"),
            })
        parent_ids = {item.parent_meta_prompt_id for item in records}
        if len(parent_ids) > 1:
            raise ValueError("migration inputs span multiple parent meta-prompts")
        bank = None
        if records:
            parent_id = records[0].parent_meta_prompt_id
            bank = append_parent_feedback_memory_bank(
                args.memory_bank_dir, parent_id, records)
            bank_count = len(load_parent_feedback_memory_bank(
                args.memory_bank_dir, parent_id))
        else:
            bank_count = 0
        after = {path: _sha(path) for path in source_paths}
        if before != after:
            raise RuntimeError("source artifacts changed during migration")
        manifest = {
            "schema_version": "episode_feedback_memory_migration_v2",
            "status": "completed",
            "iteration_id": args.iteration_id,
            "provider": args.provider,
            "model_id": args.model_id,
            "policy_version": args.policy_version,
            "prepared_requests": prepared,
            "responses": response_rows,
            "provider_call_count": len(requests),
            "memory_bank": bank,
            "memory_bank_record_count": bank_count,
            "source_hashes_before": before,
            "source_hashes_after": after,
        }
        _atomic_write_text(manifest_path, dumps_canonical(manifest) + "\n")
    except Exception as exc:
        print(f"feedback-memory migration failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    print(dumps_canonical(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
