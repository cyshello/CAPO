"""Selective surrogate rollout (PHASE2_3 §12).

Pipeline: baseline trajectory -> trace references -> (question / prompt-delta)
CLIP retrieval -> selection union -> budget -> recaption selected clips only ->
mixed caption view -> content-hash-keyed DB -> unchanged DVD QA.

Only caption generation is approximated; the downstream rollout is the same
shared reasoning path FullRolloutEvaluator uses. Zero-reference fallbacks are
explicit and machine-readable (fallback_type: none / clip_only / full_rollout /
unsupported) — never silent (CLAUDE.md §24).
"""

from __future__ import annotations

import json
import os
import time

from surrogate_rollout import config
from surrogate_rollout.captioning.candidate_captions import (
    CandidatePromptError,
    caption_clips,
    legacy_ckpt_dir,
    validate_candidate_prompt,
)
from surrogate_rollout.evaluation.dvd_qa import prepare_video_workdir, run_dvd_qa
from surrogate_rollout.evaluation.full_rollout import FullRolloutEvaluator, qa_slug
from surrogate_rollout.evaluation.rollout_evaluator import (
    FALLBACK_CLIP_ONLY,
    FALLBACK_FULL_ROLLOUT,
    FALLBACK_NONE,
    FALLBACK_UNSUPPORTED,
    EvaluationRequest,
    EvaluationResult,
)
from surrogate_rollout.mixed_views.builder import MixedViewBuilder
from surrogate_rollout.selection.base import (
    SOURCE_FRAME_INSPECTION,
    SOURCE_TRACE_RETRIEVED,
    SOURCE_TRACE_RETURNED,
    SOURCE_TRACE_STRICT,
    SelectionContext,
)
from surrogate_rollout.selection.budget import apply_budget
from surrogate_rollout.selection.union import build_selector

_TRACE_SOURCES = {
    SOURCE_FRAME_INSPECTION, SOURCE_TRACE_STRICT,
    SOURCE_TRACE_RETURNED, SOURCE_TRACE_RETRIEVED,
}
_RETRIEVAL_POLICIES = {
    "question_clip", "trace_plus_question_clip",
    "prompt_delta_clip", "trace_plus_prompt_delta_clip",
}
_TRACE_POLICIES = {
    "trace_only", "trace_plus_question_clip", "trace_plus_prompt_delta_clip",
}


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class SelectiveSurrogateRolloutEvaluator:
    """RolloutEvaluator producing surrogate_rollout_score.

    `embedder` (retrieval text encoder) and `llm_fn` (query generator) are
    injectable; both default to the configured SigLIP / codex implementations.
    `zero_reference_fallback` decides what an empty selection becomes:
    "full_rollout" or "unsupported"."""

    rollout_mode = "selective"

    def __init__(
        self,
        *,
        embedder=None,
        llm_fn=None,
        query_cache_root: str | None = None,
        visual_index_root: str | None = None,
        merge_fn=None,
        zero_reference_fallback: str = FALLBACK_FULL_ROLLOUT,
        retrieval_top_k: int = config.RETRIEVAL_TOP_K,
    ) -> None:
        if zero_reference_fallback not in (FALLBACK_FULL_ROLLOUT, FALLBACK_UNSUPPORTED):
            raise ValueError(f"invalid zero_reference_fallback: {zero_reference_fallback}")
        self._embedder = embedder
        self.llm_fn = llm_fn
        self.query_cache_root = query_cache_root
        self.visual_index_root = visual_index_root
        self.merge_fn = merge_fn
        self.zero_reference_fallback = zero_reference_fallback
        self.retrieval_top_k = retrieval_top_k
        self._full = FullRolloutEvaluator(merge_fn=merge_fn)

    # ------------------------------------------------------------------ #
    def _embedder_lazy(self):
        if self._embedder is None:
            from surrogate_rollout.retrieval.visual_index import SiglipEmbedder

            self._embedder = SiglipEmbedder()
        return self._embedder

    def _visual_index(self, request: EvaluationRequest):
        if request.selection_policy not in _RETRIEVAL_POLICIES:
            return None
        from dvd_captioning import video_duration_seconds

        from surrogate_rollout.evaluation.dvd_qa import resolve_frames_dir
        from surrogate_rollout.retrieval.visual_index import load_or_build_visual_index

        frames_dir = resolve_frames_dir(request.sample, request.video_id)
        return load_or_build_visual_index(
            video_id=request.video_id, frames_dir=frames_dir,
            duration=video_duration_seconds(request.sample["video_path"]),
            embedder=self._embedder_lazy(), root=self.visual_index_root)

    def _baseline_clip_ids(self, request: EvaluationRequest) -> list[str]:
        with open(request.baseline_captions_path) as f:
            captions = json.load(f)
        return [k for k in captions
                if k not in ("subject_registry", "character_registry")]

    def _full_fallback(self, request: EvaluationRequest, reason: str) -> EvaluationResult:
        result = self._full.evaluate(request)
        result.rollout_mode = self.rollout_mode
        result.selection_policy = request.selection_policy
        result.fallback_type = FALLBACK_FULL_ROLLOUT
        result.fallback_reason = reason
        return result

    def _unsupported(self, request: EvaluationRequest, reason: str,
                     t_start: float) -> EvaluationResult:
        return EvaluationResult(
            video_id=request.video_id, qa_id=request.qa_id,
            rollout_mode=self.rollout_mode,
            selection_policy=request.selection_policy,
            answer=None, parsed_answer=None, is_correct=False, score=0.0,
            fallback_type=FALLBACK_UNSUPPORTED, fallback_reason=reason,
            latency_seconds=time.time() - t_start,
            errors=[{"stage": "selection", "type": "Unsupported", "error": reason}],
        )

    # ------------------------------------------------------------------ #
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        t_start = time.time()
        try:
            validate_candidate_prompt(request.candidate_prompt)
        except CandidatePromptError as e:
            return EvaluationResult(
                video_id=request.video_id, qa_id=request.qa_id,
                rollout_mode=self.rollout_mode,
                selection_policy=request.selection_policy,
                answer=None, parsed_answer=None, is_correct=False, score=0.0,
                errors=[{"stage": "candidate_validation",
                         "type": "CandidatePromptError", "error": str(e)}],
            )

        if not request.baseline_trajectory_dir and \
                request.selection_policy in _TRACE_POLICIES:
            return self._unsupported(
                request, "no baseline trajectory for trace-based policy", t_start)

        trajectory = _load_jsonl(os.path.join(
            request.baseline_trajectory_dir or "", "trajectory.jsonl"))
        tool_events = _load_jsonl(os.path.join(
            request.baseline_trajectory_dir or "", "tool_events.jsonl"))
        all_clip_ids = self._baseline_clip_ids(request)

        context = SelectionContext(
            video_id=request.video_id,
            qa_id=request.qa_id,
            question=request.question,
            answer_options=tuple(request.answer_options),
            baseline_prompt=request.baseline_prompt,
            candidate_prompt=request.candidate_prompt,
            baseline_trajectory=tuple(trajectory),
            baseline_tool_events=tuple(tool_events),
            all_clip_ids=tuple(all_clip_ids),
            reference_policy=request.reference_policy,
            budget=request.budget,
            visual_index=self._visual_index(request),
            random_seed=request.random_seed,
            budget_matched_count=request.budget_matched_count,
        )
        selector = build_selector(
            request.selection_policy,
            embed_texts_fn=(self._embedder_lazy().embed_texts
                            if request.selection_policy in _RETRIEVAL_POLICIES else None),
            llm_fn=self.llm_fn,
            query_cache_root=self.query_cache_root,
            top_k=self.retrieval_top_k,
        )
        selection = selector.select(context)
        selection = apply_budget(selection, request.budget, len(all_clip_ids))

        # --------------------- zero-reference fallbacks -------------------- #
        fallback_type, fallback_reason = FALLBACK_NONE, None
        has_trace = any(_TRACE_SOURCES & set(s)
                        for s in selection.sources.values())
        if request.selection_policy in _TRACE_POLICIES and not has_trace:
            if selection.clip_ids:  # retrieval carried the selection
                fallback_type = FALLBACK_CLIP_ONLY
                fallback_reason = "zero trace references; CLIP-only selection"
            elif self.zero_reference_fallback == FALLBACK_FULL_ROLLOUT:
                return self._full_fallback(
                    request, "zero trace references and empty CLIP selection")
            else:
                return self._unsupported(
                    request, "zero trace references and empty CLIP selection", t_start)
        elif not selection.clip_ids:
            if self.zero_reference_fallback == FALLBACK_FULL_ROLLOUT:
                return self._full_fallback(request, "empty selection")
            return self._unsupported(request, "empty selection", t_start)

        # ------------------------- recaption + view ------------------------ #
        prepare_video_workdir(request.work_root, request.video_id, request.sample)
        selected = set(selection.clip_ids)
        cap_set = caption_clips(
            sample=request.sample,
            candidate_prompt=request.candidate_prompt,
            merge_prompt=request.merge_prompt,
            clip_ids=selected,
        )

        baseline_ckpt = None
        if request.merge_prompt is None:
            from dvd_prompt import get_prompts

            merge_prompt = get_prompts().merge_prompt
        else:
            merge_prompt = request.merge_prompt
        baseline_ckpt = legacy_ckpt_dir(
            request.sample, request.video_id, request.baseline_prompt, merge_prompt)

        view_name = (f"mixed_{request.selection_policy}"
                     f"_p{cap_set.prompt_hash[:12]}_{qa_slug(request.qa_id)}")
        artifact = MixedViewBuilder(merge_fn=self.merge_fn).build(
            request.baseline_captions_path,
            cap_set.parsed,
            selected,
            request.work_root,
            video_id=request.video_id,
            baseline_ckpt_dir=baseline_ckpt,
            view_name=view_name,
        )

        run_dir = os.path.join(
            request.work_root, "runs",
            f"{qa_slug(request.qa_id)}_{request.selection_policy}"
            f"_p{cap_set.prompt_hash[:8]}")
        t_reason = time.time()
        run_result = run_dvd_qa(
            captions_path=artifact.captions_path,
            sample=request.sample,
            run_dir=run_dir,
            question_id=request.qa_id,
            database_path=artifact.database_path,
            gpu=request.gpu,
        )
        reasoning_seconds = time.time() - t_reason

        total = len(all_clip_ids)
        with open(os.path.join(run_dir, "selection.json"), "w") as f:
            json.dump({
                "selected_clip_ids": sorted(selected),
                "sources": selection.sources,
                "scores": selection.scores,
                "metadata": selection.metadata,
                "fallback_type": fallback_type,
                "fallback_reason": fallback_reason,
            }, f, indent=2, default=str)

        return EvaluationResult(
            video_id=request.video_id,
            qa_id=request.qa_id,
            rollout_mode=self.rollout_mode,
            selection_policy=request.selection_policy,
            answer=run_result.prediction,
            parsed_answer=run_result.parsed_answer,
            is_correct=run_result.score == 1.0,
            score=run_result.score,
            selected_clip_ids=sorted(selected),
            recaptioned_clip_ids=artifact.replaced_clip_ids,
            recaption_fraction=len(selected) / total if total else 0.0,
            total_clips=total,
            selection_sources=selection.sources,
            fallback_type=fallback_type,
            fallback_reason=fallback_reason,
            captions_hash=artifact.captions_hash,
            captions_path=artifact.captions_path,
            database_path=artifact.database_path,
            trajectory_path=os.path.join(run_dir, "trajectory.jsonl"),
            caption_call_count=cap_set.caption_call_count,
            caption_cache_hits=cap_set.cache_hits,
            caption_seconds=cap_set.caption_seconds,
            reasoning_seconds=reasoning_seconds,
            latency_seconds=time.time() - t_start,
            errors=run_result.errors,
            run_result=run_result,
        )
