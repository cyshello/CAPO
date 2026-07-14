"""Prompt-delta -> visual retrieval queries (PHASE2_3 §8).

Extracts newly emphasized visual concepts from the baseline-vs-candidate
caption-prompt difference and combines them with the downstream question. The
caption prompts are summarized by the text LLM, never passed to the CLIP text
encoder directly. Cached by question/options/baseline-prompt/candidate-prompt
hashes plus model and prompt version.
"""

from __future__ import annotations

from typing import Callable, Sequence

from surrogate_rollout import config
from surrogate_rollout.retrieval.question_queries import (
    QueryGenerationError,
    _cached_or_generate,
    _clean_queries,
)
from surrogate_rollout.schemas import sha256_text

PROMPT_DELTA_QUERY_PROMPT = """Two captioner prompts caption the same video. A downstream agent answers a question from those captions. Identify what the CANDIDATE prompt newly emphasizes relative to the BASELINE prompt, then write short visual search queries (for a CLIP-style image-text retriever) targeting video moments where that new emphasis could change the caption in ways relevant to the question.

BASELINE captioner prompt:
---
{baseline_prompt}
---

CANDIDATE captioner prompt:
---
{candidate_prompt}
---

Question:
{question}

Answer options:
{options}

Rules:
- Output ONLY a JSON object: {{"newly_emphasized_concepts": ["...", ...], "queries": ["...", ...]}}
- 4 to {max_queries} queries.
- Queries must be visually retrievable from single frames: concrete entities, attributes, colors, actions, on-screen text.
- Only include concepts the candidate prompt emphasizes MORE than the baseline.
"""


def generate_prompt_delta_queries(
    question: str,
    answer_options: Sequence[str],
    baseline_prompt: str,
    candidate_prompt: str,
    *,
    llm_fn: Callable[[str], str] | None = None,
    model: str = config.QUERY_GENERATOR_MODEL,
    cache_root: str | None = None,
    max_queries: int = config.MAX_RETRIEVAL_QUERIES,
) -> dict:
    """Returns {"newly_emphasized_concepts": [...], "queries": [...]}."""
    key = {
        "question_hash": sha256_text(question),
        "answer_options_hash": sha256_text("\n".join(answer_options)),
        "baseline_prompt_hash": sha256_text(baseline_prompt),
        "candidate_prompt_hash": sha256_text(candidate_prompt),
        "model": model,
        "prompt_version": config.PROMPT_DELTA_QUERY_PROMPT_VERSION,
    }
    prompt = PROMPT_DELTA_QUERY_PROMPT.format(
        baseline_prompt=baseline_prompt,
        candidate_prompt=candidate_prompt,
        question=question,
        options="\n".join(answer_options) or "(none)",
        max_queries=max_queries,
    )

    def parse(obj: dict) -> dict:
        return {
            "newly_emphasized_concepts": [
                str(c).strip() for c in (obj.get("newly_emphasized_concepts") or [])
                if str(c).strip()
            ],
            "queries": _clean_queries(obj.get("queries"), max_queries),
        }

    return _cached_or_generate("prompt_delta_queries", key, prompt,
                               llm_fn, cache_root, parse)


__all__ = ["generate_prompt_delta_queries", "QueryGenerationError"]
