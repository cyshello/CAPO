"""Real structured feedback through the repository's configured Codex CLI."""

from __future__ import annotations

import json
import sys
from typing import Sequence

from surrogate_rollout import config
from surrogate_rollout.optimization.schemas import CounterfactualEvidence
FEEDBACK_POLICY_VERSION = "codex_structured_feedback_v0001"


class CodexStructuredFeedbackProvider:
    """Callable response provider for ``LLMFeedbackGenerator``."""

    def __init__(self, *, model: str = config.TEXT_FALLBACK_MODEL) -> None:
        self.model = model
        self.call_count = 0

    def __call__(self, request: str) -> str:
        if self.call_count:
            raise RuntimeError("Stage 4.13 permits exactly one feedback call")
        self.call_count += 1
        payload = json.loads(request)
        evidence: Sequence[dict] = payload["evidence"]
        evidence_ids = [item["evidence_id"] for item in evidence]
        prompt = f"""You are the real structured-feedback policy for one offline prompt-routing optimization iteration.

Analyze only the supplied frozen counterfactual evidence. Do not claim causal component attribution when the records do not support it. Legacy records without composition traces cannot support scaffold attribution. A neutral or ambiguous comparison may be insufficient evidence.

Current reusable prompt IDs are pe_default, pe_temporal, and pe_text. For a supported prompt-bank revision, put the existing target in applicable_segment_traits.target_prompt_id. For a supported router change, put a JSON object in applicable_segment_traits.routing_conditions and a list of existing prompt IDs in applicable_segment_traits.target_prompt_ids. Keep desired behaviors reusable and concise. Do not propose scaffold changes in this fixed-scaffold iteration.

Return ONLY one JSON object with exactly this top-level structure:
{{
  "feedback_policy": "real_codex_structured",
  "feedback_policy_version": "{FEEDBACK_POLICY_VERSION}",
  "input_evidence_ids": {json.dumps(evidence_ids)},
  "attributions": [],
  "items": [
    {{
      "feedback_id": "feedback_real_unique_id",
      "evidence_ids": ["one or more exact evidence IDs"],
      "attribution_id": "attribution_real_matching_unique_id",
      "target_components": ["prompt_bank OR router OR insufficient_evidence"],
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
        if config.PROMPT_SENS_ROOT not in sys.path:
            sys.path.insert(0, config.PROMPT_SENS_ROOT)
        from codex_infer import codex_infer

        return codex_infer(prompt, model=self.model, timeout=600)

    def metadata(self) -> dict:
        return {
            "provider": "codex_cli",
            "model": self.model,
            "policy_version": FEEDBACK_POLICY_VERSION,
            "call_count": self.call_count,
            "real_model": True,
        }
