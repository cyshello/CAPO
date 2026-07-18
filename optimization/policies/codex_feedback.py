"""Real structured feedback through the repository's configured Codex CLI."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Sequence

from surrogate_rollout import config
from surrogate_rollout.optimization.schemas import CounterfactualEvidence
FEEDBACK_POLICY_VERSION = "codex_structured_feedback_v0001"
SCAFFOLD_FEEDBACK_POLICY_VERSION = "codex_structured_feedback_v0002_scaffold"
OPENAI_FEEDBACK_POLICY_VERSION = "openai_structured_feedback_v0001"
OPENAI_SCAFFOLD_FEEDBACK_POLICY_VERSION = \
    "openai_structured_feedback_v0002_scaffold"


def _load_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env_path = os.path.join(config.PROMPT_SENS_ROOT, ".env")
    if os.path.isfile(env_path):
        with open(env_path) as env_file:
            for line in env_file:
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENAI_API_KEY is not configured")


def _feedback_prompt(request: str, *, allow_scaffold_updates: bool,
                     policy_version: str) -> str:
    payload = json.loads(request)
    evidence: Sequence[dict] = payload["evidence"]
    evidence_ids = [item["evidence_id"] for item in evidence]
    scaffold_instruction = (
        "Scaffold attribution is permitted only when complete composition "
        "traces and an explicit structured scaffold-failure fact support it. "
        "Do not combine unrelated videos to fabricate a global pattern."
        if allow_scaffold_updates else
        "Do not propose scaffold changes in this fixed-scaffold iteration."
    )
    allowed_targets = (
        "prompt_bank OR router OR scaffold OR insufficient_evidence"
        if allow_scaffold_updates else
        "prompt_bank OR router OR insufficient_evidence")
    return f"""You are the real structured-feedback policy for one offline prompt-routing optimization iteration.

Analyze only the supplied frozen counterfactual evidence. Do not claim causal component attribution when the records do not support it. Legacy records without composition traces cannot support scaffold attribution. A neutral or ambiguous comparison may be insufficient evidence.

Current reusable prompt IDs are pe_default, pe_temporal, and pe_text. For a supported prompt-bank revision, put the existing target in applicable_segment_traits.target_prompt_id. For a supported router change, put a JSON object in applicable_segment_traits.routing_conditions and a list of existing prompt IDs in applicable_segment_traits.target_prompt_ids. Keep desired behaviors reusable and concise. {scaffold_instruction}

Return ONLY one JSON object with exactly this top-level structure:
{{
  "feedback_policy": "real_openai_structured",
  "feedback_policy_version": "{policy_version}",
  "input_evidence_ids": {json.dumps(evidence_ids)},
  "attributions": [],
  "items": [
    {{
      "feedback_id": "feedback_real_unique_id",
      "evidence_ids": ["one or more exact evidence IDs"],
      "attribution_id": "attribution_real_matching_unique_id",
      "target_components": ["{allowed_targets}"],
      "failure_modes": ["typed_snake_case_label"],
      "successful_behaviors": [],
      "desired_behaviors": ["one reusable behavior"],
      "avoid_behaviors": ["one behavior to avoid"],
      "applicable_segment_traits": {{}},
      "confidence": 0.0,
      "rationale": "brief evidence-grounded rationale"
    }}
  ],
  "raw_response_artifact": null,
  "parse_errors": []
}}

Requirements:
- Include every input evidence ID in input_evidence_ids exactly as provided.
- Every feedback evidence ID must be one of those IDs.
- Use a unique feedback_id and attribution_id per item.
- Confidence must be in [0,1].
- Prefer separate feedback items when distinct evidence records support distinct local observations.
- Never use rationale prose as a substitute for structured target or trait fields.

Frozen request:
{json.dumps(payload, sort_keys=True, ensure_ascii=False)}
"""


def _openai_chat(prompt: str, *, model: str, api_key: str,
                 timeout: int = 600,
                 response_format: dict | None = None) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": response_format or {"type": "json_object"},
    }).encode()
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"OpenAI feedback call failed with HTTP {exc.code}: {detail}") from exc
    return payload["choices"][0]["message"]["content"]


class OpenAIStructuredFeedbackProvider:
    """Real structured feedback through the OpenAI API."""

    def __init__(self, *, model: str = config.FEEDBACK_MODEL,
                 allow_scaffold_updates: bool = False,
                 api_key: str | None = None, max_calls: int = 1) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.model = model
        self.allow_scaffold_updates = allow_scaffold_updates
        self.api_key = api_key
        self.max_calls = max_calls
        self.policy_version = (
            OPENAI_SCAFFOLD_FEEDBACK_POLICY_VERSION
            if allow_scaffold_updates else OPENAI_FEEDBACK_POLICY_VERSION)
        self.call_count = 0

    def __call__(self, request: str) -> str:
        if self.call_count >= self.max_calls:
            message = ("only one feedback call is permitted"
                       if self.max_calls == 1
                       else "configured feedback call limit reached")
            raise RuntimeError(message)
        self.call_count += 1
        prompt = _feedback_prompt(
            request, allow_scaffold_updates=self.allow_scaffold_updates,
            policy_version=self.policy_version)
        return _openai_chat(
            prompt, model=self.model,
            api_key=self.api_key or _load_openai_key())

    def metadata(self) -> dict:
        return {
            "provider": "openai_api", "model": self.model,
            "policy_version": self.policy_version,
            "allow_scaffold_updates": self.allow_scaffold_updates,
            "max_calls": self.max_calls, "call_count": self.call_count,
            "real_model": True,
        }


class CodexStructuredFeedbackProvider:
    """Callable response provider for ``LLMFeedbackGenerator``."""

    def __init__(
        self,
        *,
        model: str = config.TEXT_FALLBACK_MODEL,
        allow_scaffold_updates: bool = False,
    ) -> None:
        self.model = model
        self.allow_scaffold_updates = allow_scaffold_updates
        self.policy_version = (SCAFFOLD_FEEDBACK_POLICY_VERSION
                               if allow_scaffold_updates
                               else FEEDBACK_POLICY_VERSION)
        self.call_count = 0

    def __call__(self, request: str) -> str:
        if self.call_count:
            raise RuntimeError("Stage 4.13 permits exactly one feedback call")
        self.call_count += 1
        prompt = _feedback_prompt(
            request, allow_scaffold_updates=self.allow_scaffold_updates,
            policy_version=self.policy_version)
        if config.PROMPT_SENS_ROOT not in sys.path:
            sys.path.insert(0, config.PROMPT_SENS_ROOT)
        from codex_infer import codex_infer

        return codex_infer(prompt, model=self.model, timeout=600)

    def metadata(self) -> dict:
        return {
            "provider": "codex_cli",
            "model": self.model,
            "policy_version": self.policy_version,
            "allow_scaffold_updates": self.allow_scaffold_updates,
            "call_count": self.call_count,
            "real_model": True,
        }
