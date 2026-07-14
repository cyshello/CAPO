"""Selection policies and their composition (PHASE2_3 §10).

`build_selector(policy_name, ...)` assembles a ClipSelector from the trace
selector, CLIP retrievers, and the random budget-matched baseline. Retrieval
dependencies (text-embedding fn, query-generator llm fn) are injected here so
selectors themselves stay deterministic and testable.
"""

from __future__ import annotations

import hashlib
import random
from typing import Callable, Sequence

import numpy as np

from surrogate_rollout import config
from surrogate_rollout.retrieval.clip_retriever import retrieve_clips
from surrogate_rollout.retrieval.prompt_delta_queries import generate_prompt_delta_queries
from surrogate_rollout.retrieval.question_queries import generate_question_queries
from surrogate_rollout.selection.base import (
    SOURCE_PROMPT_DELTA,
    SOURCE_QUESTION,
    SOURCE_RANDOM,
    SelectionContext,
    SelectionResult,
    merge_results,
)
from surrogate_rollout.selection.trace import TraceReferenceSelector


class QuestionClipRetriever:
    """question_clip baseline: question-derived queries against the visual index."""

    source = SOURCE_QUESTION

    def __init__(self, embed_texts_fn: Callable[[Sequence[str]], np.ndarray],
                 llm_fn: Callable[[str], str] | None = None,
                 cache_root: str | None = None,
                 top_k: int = config.RETRIEVAL_TOP_K) -> None:
        self.embed_texts_fn = embed_texts_fn
        self.llm_fn = llm_fn
        self.cache_root = cache_root
        self.top_k = top_k

    def _queries(self, context: SelectionContext) -> list[str]:
        return generate_question_queries(
            context.question, context.answer_options,
            llm_fn=self.llm_fn, cache_root=self.cache_root)

    def select(self, context: SelectionContext) -> SelectionResult:
        if context.visual_index is None:
            raise ValueError(f"{type(self).__name__} requires context.visual_index")
        queries = self._queries(context)
        return retrieve_clips(
            context.visual_index, self.embed_texts_fn(queries),
            source=self.source, top_k=self.top_k,
            max_fraction=context.budget.max_clip_retrieval_fraction,
            queries=queries)


class PromptDeltaClipRetriever(QuestionClipRetriever):
    """prompt_delta_clip: queries from the baseline-vs-candidate prompt delta."""

    source = SOURCE_PROMPT_DELTA

    def _queries(self, context: SelectionContext) -> list[str]:
        out = generate_prompt_delta_queries(
            context.question, context.answer_options,
            context.baseline_prompt, context.candidate_prompt,
            llm_fn=self.llm_fn, cache_root=self.cache_root)
        return out["queries"]


class RandomBudgetMatchedSelector:
    """Budget-matched random baseline: uniformly samples the same number of
    clips a real policy would recaption (context.budget_matched_count), with a
    deterministic per-(video, qa, candidate, seed) RNG."""

    def select(self, context: SelectionContext) -> SelectionResult:
        total = len(context.all_clip_ids)
        n = context.budget_matched_count
        if n is None:
            n = context.budget.max_clips(total)
        n = min(n, total)
        seed_material = "|".join([
            context.video_id, context.qa_id,
            hashlib.sha256(context.candidate_prompt.encode()).hexdigest(),
            str(context.random_seed),
        ])
        rng = random.Random(seed_material)
        chosen = rng.sample(sorted(context.all_clip_ids,
                                   key=lambda k: float(k.split("_")[0])), n)
        result = SelectionResult()
        for cid in sorted(chosen, key=lambda k: float(k.split("_")[0])):
            result.add(cid, SOURCE_RANDOM)
        result.metadata["budget_matched_count"] = n
        return result


class UnionSelector:
    def __init__(self, selectors: Sequence) -> None:
        self.selectors = list(selectors)

    def select(self, context: SelectionContext) -> SelectionResult:
        return merge_results(*(s.select(context) for s in self.selectors))


SELECTION_POLICIES = (
    "trace_only",
    "question_clip",
    "trace_plus_question_clip",
    "prompt_delta_clip",
    "trace_plus_prompt_delta_clip",
    "random_budget_matched",
)


def build_selector(
    policy: str,
    *,
    embed_texts_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
    llm_fn: Callable[[str], str] | None = None,
    query_cache_root: str | None = None,
    top_k: int = config.RETRIEVAL_TOP_K,
):
    if policy not in SELECTION_POLICIES:
        raise ValueError(f"unknown selection policy: {policy}")

    def question():
        return QuestionClipRetriever(embed_texts_fn, llm_fn, query_cache_root, top_k)

    def delta():
        return PromptDeltaClipRetriever(embed_texts_fn, llm_fn, query_cache_root, top_k)

    if policy == "trace_only":
        return TraceReferenceSelector()
    if policy == "question_clip":
        return question()
    if policy == "trace_plus_question_clip":
        return UnionSelector([TraceReferenceSelector(), question()])
    if policy == "prompt_delta_clip":
        return delta()
    if policy == "trace_plus_prompt_delta_clip":
        return UnionSelector([TraceReferenceSelector(), delta()])
    return RandomBudgetMatchedSelector()
