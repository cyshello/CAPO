"""Official-GEPA integration adapter for the free-form DVD meta-prompt.

Implements the ``gepa.core.adapter.GEPAAdapter`` contract so the installed
``gepa`` engine owns candidate selection, minibatch sampling, Pareto tracking,
and the rollout budget. The single optimizable component is ``meta_prompt`` (the
free-form caption-instruction generator template).

Responsibilities:

* ``evaluate`` — instantiate the free-form generator from the candidate's
  ``meta_prompt`` text and score each video in the batch absolutely
  (mean QA accuracy), reusing :mod:`.dvd_single_video_evaluator`. Per-example
  failures return a 0.0 score with an error trajectory rather than raising, per
  the GEPA error-handling contract.
* ``make_reflective_dataset`` — turn captured per-video trajectories into the
  ``{Inputs, Generated Outputs, Feedback}`` records the proposer consumes.
* ``propose_new_texts`` — reflect on that dataset with the injected
  :class:`ReflectionMutator` and return the revised ``meta_prompt`` text. Because
  the adapter supplies this, the engine needs no separate ``reflection_lm``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from gepa import EvaluationBatch

from surrogate_rollout.gepa_meta_prompt.dvd_single_video_evaluator import (
    GepaVideoInstance,
    caption_and_score_video,
    generator_meta_prompt_id,
)
from surrogate_rollout.gepa_meta_prompt.reflection import (
    ReflectionMutator,
    render_video_feedback,
)
from surrogate_rollout.prompt_routing.free_form_instruction_generator import (
    VLMFreeFormInstructionGenerator,
)

META_PROMPT_COMPONENT = "meta_prompt"


class DVDMetaPromptGEPAAdapter:
    """GEPA adapter scoring meta-prompt candidates on real DVD videos."""

    def __init__(
        self, *, evaluator: Any, bank: Any, router: Any, scaffold: Any,
        contract: Any, generator_model_id: str, generator_backend_id: str,
        generator_max_tokens: int, mutator: ReflectionMutator,
        work_root: str, qa_cache_root: str,
    ) -> None:
        if not isinstance(generator_model_id, str) or not generator_model_id:
            raise ValueError("generator_model_id must be a non-empty string")
        if not isinstance(generator_backend_id, str) or not generator_backend_id:
            raise ValueError("generator_backend_id must be a non-empty string")
        if not isinstance(generator_max_tokens, int) or \
                isinstance(generator_max_tokens, bool) or \
                generator_max_tokens <= 0:
            raise ValueError("generator_max_tokens must be a positive integer")
        self._evaluator = evaluator
        self._bank = bank
        self._router = router
        self._scaffold = scaffold
        self._contract = contract
        self._generator_model_id = generator_model_id
        self._generator_backend_id = generator_backend_id
        self._generator_max_tokens = generator_max_tokens
        self._mutator = mutator
        self._work_root = work_root
        self._qa_cache_root = qa_cache_root
        # Present + non-None => the GEPA engine uses this instead of a default
        # reflection_lm proposer (see gepa/api.py adapter_has_propose).
        self.propose_new_texts = self._propose_new_texts

    # -- program construction / scoring -----------------------------------

    def _generator(self, meta_prompt_text: str) -> VLMFreeFormInstructionGenerator:
        return VLMFreeFormInstructionGenerator(
            self._evaluator.builder.router.vlm,
            max_tokens=self._generator_max_tokens,
            template_text=meta_prompt_text,
            meta_prompt_id=generator_meta_prompt_id(meta_prompt_text),
            model_id=self._generator_model_id,
            backend_id=self._generator_backend_id)

    def _candidate_dir(self, meta_prompt_text: str) -> str:
        return generator_meta_prompt_id(meta_prompt_text)

    def evaluate(
        self, batch: list[GepaVideoInstance], candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        meta_prompt_text = candidate[META_PROMPT_COMPONENT]
        generator = self._generator(meta_prompt_text)
        candidate_dir = self._candidate_dir(meta_prompt_text)
        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] = []
        for instance in batch:
            try:
                video_score = caption_and_score_video(
                    evaluator=self._evaluator, video=instance,
                    generator=generator, bank=self._bank, router=self._router,
                    scaffold=self._scaffold, contract=self._contract,
                    work_root=f"{self._work_root}/{candidate_dir}",
                    qa_cache_root=self._qa_cache_root)
                score = float(video_score.accuracy)
                output = {
                    "video_id": instance.video_id,
                    "accuracy": score,
                    "captions_hash": video_score.captions_hash,
                    "qa": [{
                        "question_id": row.question_id,
                        "is_correct": row.is_correct,
                        "prediction": row.prediction,
                        "ground_truth": row.ground_truth,
                        "errors": list(row.errors),
                    } for row in video_score.qa_results],
                }
                trajectory = {
                    "video_id": instance.video_id,
                    "feedback": render_video_feedback(video_score),
                    "accuracy": score,
                    "captions_hash": video_score.captions_hash,
                }
            except Exception as exc:  # noqa: BLE001 - systemic-safe per contract
                score = 0.0
                output = {"video_id": instance.video_id, "error": str(exc)}
                trajectory = {
                    "video_id": instance.video_id,
                    "feedback": (f"Video {instance.video_id}: evaluation failed "
                                 f"({type(exc).__name__}: {exc})."),
                    "accuracy": 0.0, "captions_hash": None,
                }
            outputs.append(output)
            scores.append(score)
            trajectories.append(trajectory)
        return EvaluationBatch(
            outputs=outputs, scores=scores,
            trajectories=trajectories if capture_traces else None)

    # -- reflection --------------------------------------------------------

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if eval_batch.trajectories is None:
            raise ValueError("reflective dataset requires captured trajectories")
        records = []
        for trajectory, output in zip(eval_batch.trajectories, eval_batch.outputs):
            records.append({
                "Inputs": {"video_id": trajectory.get("video_id", "")},
                "Generated Outputs": {
                    "accuracy": trajectory.get("accuracy"),
                    "captions_hash": trajectory.get("captions_hash"),
                },
                "Feedback": trajectory.get("feedback", ""),
            })
        return {META_PROMPT_COMPONENT: records}

    def _propose_new_texts(
        self, candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        if META_PROMPT_COMPONENT not in components_to_update:
            return {}
        parent_text = candidate[META_PROMPT_COMPONENT]
        records = list(reflective_dataset.get(META_PROMPT_COMPONENT, ()))
        if not records:
            raise ValueError("no reflective records for the meta_prompt component")
        feedback_blocks = [str(record.get("Feedback", "")) for record in records]
        instance_ids = [str(record.get("Inputs", {}).get("video_id", ""))
                        for record in records]
        new_text = self._mutator.propose(
            parent_text=parent_text, feedback_blocks=feedback_blocks,
            instance_ids=instance_ids)
        return {META_PROMPT_COMPONENT: new_text}
