"""Full candidate-prompt rollout (PHASE2_3 §2): the ground-truth evaluator.

Every clip of the video is captioned under the candidate prompt (through the
strong-key harness cache, so repeat evaluations hit cache), the subject
registry is re-merged, the exact captions.json is snapshotted into the work
root, and the unchanged DVD agent runs over a vector DB keyed by that
snapshot's content hash. Legacy caches are read (baseline-prompt prefill) but
never written.
"""

from __future__ import annotations

import os
import time

from surrogate_rollout import config
from surrogate_rollout.captioning.candidate_captions import (
    CandidatePromptError,
    caption_clips,
    validate_candidate_prompt,
)
from surrogate_rollout.cache.caption_cache import prompt_hash as compute_prompt_hash
from surrogate_rollout.evaluation.dvd_qa import prepare_video_workdir, run_dvd_qa
from surrogate_rollout.evaluation.rollout_evaluator import (
    FALLBACK_NONE,
    EvaluationRequest,
    EvaluationResult,
)
from surrogate_rollout.mixed_views.builder import (
    caption_entry_from_parsed,
    default_merge_fn,
    write_captions_json,
)


def qa_slug(qa_id: str) -> str:
    return qa_id.replace("/", "-")


class FullRolloutEvaluator:
    """RolloutEvaluator producing full_rollout_score."""

    rollout_mode = "full"

    def __init__(self, merge_fn=None) -> None:
        self.merge_fn = merge_fn or default_merge_fn

    def _resolve_merge_prompt(self, request: EvaluationRequest) -> str:
        if request.merge_prompt is not None:
            return request.merge_prompt
        from dvd_prompt import get_prompts

        return get_prompts().merge_prompt

    def build_full_view(self, request: EvaluationRequest) -> tuple[str, object]:
        """Materialize the full candidate caption view for the request's video
        (reused across the video's QAs within the work root). Returns
        (captions_path, CandidateCaptionSet)."""
        merge_prompt = self._resolve_merge_prompt(request)
        cap_set = caption_clips(
            sample=request.sample,
            candidate_prompt=request.candidate_prompt,
            merge_prompt=merge_prompt,
        )
        p12 = cap_set.prompt_hash[:12]
        out_dir = os.path.join(request.work_root, request.video_id,
                               f"captions_full_p{p12}")
        captions_path = os.path.join(out_dir, "captions.json")
        if not os.path.exists(captions_path):
            captions: dict = {}
            registries = []
            for k in cap_set.clips:
                entry = caption_entry_from_parsed(cap_set.parsed.get(k) or {})
                if entry is not None:
                    captions[k] = entry
                    reg = (cap_set.parsed.get(k) or {}).get("subject_registry")
                    if reg:
                        registries.append(reg)
            captions["subject_registry"] = self.merge_fn(registries)
            captions_path, _ = write_captions_json(captions, out_dir)
        return captions_path, cap_set

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        t_start = time.time()
        try:
            validate_candidate_prompt(request.candidate_prompt)
        except CandidatePromptError as e:
            # documented failure score, no silent repair (CLAUDE.md §17)
            return EvaluationResult(
                video_id=request.video_id, qa_id=request.qa_id,
                rollout_mode=self.rollout_mode, selection_policy="full_rollout",
                answer=None, parsed_answer=None, is_correct=False, score=0.0,
                errors=[{"stage": "candidate_validation",
                         "type": "CandidatePromptError", "error": str(e)}],
            )

        prepare_video_workdir(request.work_root, request.video_id, request.sample)
        captions_path, cap_set = self.build_full_view(request)
        from surrogate_rollout.cache.caption_cache import captions_content_hash

        chash = captions_content_hash(captions_path)

        run_dir = os.path.join(
            request.work_root, "runs",
            f"{qa_slug(request.qa_id)}_full_p{cap_set.prompt_hash[:8]}")
        t_reason = time.time()
        run_result = run_dvd_qa(
            captions_path=captions_path,
            sample=request.sample,
            run_dir=run_dir,
            question_id=request.qa_id,
            gpu=request.gpu,
        )
        reasoning_seconds = time.time() - t_reason

        total = len(cap_set.clips)
        return EvaluationResult(
            video_id=request.video_id,
            qa_id=request.qa_id,
            rollout_mode=self.rollout_mode,
            selection_policy="full_rollout",
            answer=run_result.prediction,
            parsed_answer=run_result.parsed_answer,
            is_correct=run_result.score == 1.0,
            score=run_result.score,
            selected_clip_ids=list(cap_set.clips),
            recaptioned_clip_ids=list(cap_set.clips),
            recaption_fraction=1.0,
            total_clips=total,
            selection_sources={k: ["full_rollout"] for k in cap_set.clips},
            fallback_type=FALLBACK_NONE,
            captions_hash=chash,
            captions_path=captions_path,
            database_path=run_result.database_path,
            trajectory_path=os.path.join(run_dir, "trajectory.jsonl"),
            caption_call_count=cap_set.caption_call_count,
            caption_cache_hits=cap_set.cache_hits,
            caption_seconds=cap_set.caption_seconds,
            reasoning_seconds=reasoning_seconds,
            latency_seconds=time.time() - t_start,
            errors=run_result.errors,
            run_result=run_result,
        )


def baseline_prompt_hash(baseline_prompt: str, merge_prompt: str) -> str:
    return compute_prompt_hash(baseline_prompt, merge_prompt)
