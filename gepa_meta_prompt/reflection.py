"""Reflective mutation of the meta-prompt for GEPA search.

The GEPA adapter feeds the current (parent) meta-prompt plus rendered execution
feedback from a minibatch of videos to a :class:`ReflectionMutator`, which
returns a new candidate meta-prompt text. The adapter depends only on the
Protocol; :class:`OpenAICompatibleReflectionMutator` is one concrete backend and
can be swapped without touching the search engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

REFLECTION_SYSTEM_INSTRUCTION = """You improve the instruction ("meta-prompt") given to a visual- and history-conditioned caption-instruction generator.

At runtime that generator sees only sampled frames from the current video segment and a bounded window of preceding captions. Using the meta-prompt, it writes ONE segment-specific captioning instruction. A separate captioner then produces the caption, and long-video question answering runs over the resulting captions. The meta-prompt is the only thing you may change.

You are given the current meta-prompt and per-question execution feedback from real runs (which questions the current meta-prompt answered correctly or incorrectly, and any execution errors). Diagnose what caption behavior the meta-prompt should encourage or avoid, then rewrite the meta-prompt so that it is more likely to yield captions that preserve information useful for later question answering.

Constraints:
- The generator can rely only on current frames, bounded preceding caption history, and this meta-prompt. Do not require the question, correctness labels, transcripts, OCR, tools, or any other unavailable input.
- Do not embed specific questions, answers, dataset wording, video IDs, segment IDs, or QA IDs. Keep the meta-prompt general.
- Keep the generator's output contract intact: it must still return strict JSON of the form {"caption_instruction": "..."} and must not answer downstream questions.
- Prefer the smallest change that addresses the observed failures; do not rewrite wholesale without reason.

Return only the full revised meta-prompt text. Do not add commentary, explanation, or Markdown fences."""


def render_video_feedback(video_score) -> str:
    """Feedback block for one per-video absolute evaluation (GEPA adapter path).

    ``video_score`` is a ``GepaVideoScore`` (duck-typed here to avoid importing
    the DVD-bound module into the reflection layer).
    """

    lines = [
        f"Video {video_score.video_id}: accuracy {video_score.accuracy:.3f} "
        f"over {video_score.evaluated_qa_count} questions "
        f"(caption_calls={video_score.caption_calls}, "
        f"cache_hits={video_score.caption_cache_hits})."
    ]
    for row in video_score.qa_results:
        if row.errors:
            verdict = f"execution_error: {'; '.join(row.errors)}"
        elif row.is_correct is True:
            verdict = "correct"
        elif row.is_correct is False:
            verdict = f"incorrect (predicted {row.prediction!r}, " \
                      f"truth {row.ground_truth!r})"
        else:
            verdict = "unknown"
        lines.append(f"  - question {row.question_id}: {verdict}")
    return "\n".join(lines)


class ReflectionMutator(Protocol):
    """Propose a revised meta-prompt from feedback on the current one."""

    def propose(
        self, *, parent_text: str, feedback_blocks: Sequence[str],
        instance_ids: Sequence[str],
    ) -> str:
        ...


@dataclass
class OpenAICompatibleReflectionMutator:
    """Reflective mutator backed by an OpenAI-compatible chat completion.

    ``response_transport`` takes a request body and returns the raw response
    envelope (same shape the Checkpoint-G transport already speaks); this keeps
    the mutator free of any HTTP/retry policy, which the caller owns.
    """

    model_id: str
    generation_settings: Mapping[str, Any]
    response_transport: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    maximum_output_tokens: int
    policy_version: str = "gepa_openai_reflection_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(self.generation_settings, Mapping):
            raise TypeError("generation_settings must be a mapping")
        if not callable(self.response_transport):
            raise TypeError("response_transport must be callable")
        if not isinstance(self.maximum_output_tokens, int) or \
                isinstance(self.maximum_output_tokens, bool) or \
                self.maximum_output_tokens <= 0:
            raise ValueError("maximum_output_tokens must be a positive integer")

    def _user_content(
        self, parent_text: str, feedback_blocks: Sequence[str],
    ) -> str:
        payload = {
            "current_meta_prompt": parent_text,
            "execution_feedback": list(feedback_blocks),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def propose(
        self, *, parent_text: str, feedback_blocks: Sequence[str],
        instance_ids: Sequence[str],
    ) -> str:
        if not isinstance(parent_text, str) or not parent_text:
            raise ValueError("parent_text must be a non-empty string")
        if not feedback_blocks:
            raise ValueError("feedback_blocks must be non-empty")
        body = {
            "model": self.model_id,
            "max_tokens": self.maximum_output_tokens,
            "messages": [
                {"role": "system", "content": REFLECTION_SYSTEM_INSTRUCTION},
                {"role": "user",
                 "content": self._user_content(parent_text, feedback_blocks)},
            ],
            **dict(self.generation_settings),
        }
        envelope = self.response_transport(body)
        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("reflection response has no message content") from exc
        if not isinstance(content, str):
            raise TypeError("reflection message content must be a string")
        text = content.strip()
        if not text:
            raise ValueError("reflection produced an empty meta-prompt")
        return text
