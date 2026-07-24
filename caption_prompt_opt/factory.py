"""Component factory for the caption-prompt optimization path.

Returns ``(feedback_generator, updater, confirmation_evaluator)`` for the
generic ``scripts/run_prompt_delta_iteration.py`` entry, exactly like
``surrogate_rollout.optimization.checkpoint_g_factory.build_checkpoint_g_components``.

Reuse vs. new:
- feedback generator: reused verbatim from the existing factory;
- updater: the existing OpenAI updater backend is reused, wrapped so the baked
  meta-prompt system instruction is replaced with the caption-prompt updater
  prompt (``prompts/caption_prompt_updater_system_v1.txt``);
- confirmation: rebuilt here so it uses ``StaticInstructionGenerator`` (caption
  prompt emitted verbatim) instead of the OpenAI free-form generator.

Nothing under ``surrogate_rollout`` is modified. Concurrency isolation from a
live meta-prompt run is the operator's responsibility via the component config
(distinct ``cache_root``, ``sample_source_identity``, ``cache_reset_identity``,
etc.) and the entry's distinct ``--output-dir/--state-dir/--feedback-memory-bank-dir``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from surrogate_rollout import config
from surrogate_rollout.captioning.history_aware_baseline import (
    HistoryAwareBaselineCaptionViewBuilder,
)
from surrogate_rollout.evaluation.dvd_qa import ensure_backend
from surrogate_rollout.optimization.checkpoint_g_factory import (
    build_checkpoint_g_components,
    _object,
    _required,
)
from surrogate_rollout.optimization.confirmation_evaluator import (
    HistoryAwareDVDConfirmationEvaluator,
)
from surrogate_rollout.optimization.dvd_meta_prompt_confirmation import (
    DVDMetaPromptConfirmationVideo,
    LazyDVDMetaPromptConfirmationEvaluator,
)
from surrogate_rollout.optimization.meta_prompt_updater import (
    LLMMetaPromptUpdater,
    META_PROMPT_UPDATER_SYSTEM_INSTRUCTION,
)
from surrogate_rollout.prompt_routing.persistence import (
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)

from caption_prompt_opt.confirmation import StaticGeneratorConfirmationEvaluator
from caption_prompt_opt.updater_backend import SystemPromptOverrideUpdaterBackend

_PROMPT_DIRECTORY = Path(__file__).resolve().parent / "prompts"
CAPTION_UPDATER_POLICY_VERSION = "caption_prompt_updater_v1"


def _load_updater_prompt() -> str:
    path = _PROMPT_DIRECTORY / "caption_prompt_updater_system_v1.txt"
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"caption-prompt updater prompt is empty: {path}")
    return text


CAPTION_UPDATER_SYSTEM_INSTRUCTION = _load_updater_prompt()


def _wrap_updater(updater: LLMMetaPromptUpdater) -> LLMMetaPromptUpdater:
    """Reuse the existing updater backend, swapping only its system prompt."""
    return LLMMetaPromptUpdater(
        backend=SystemPromptOverrideUpdaterBackend(
            updater.backend,
            CAPTION_UPDATER_SYSTEM_INSTRUCTION,
            replaced_base=META_PROMPT_UPDATER_SYSTEM_INSTRUCTION),
        updater_policy_version=CAPTION_UPDATER_POLICY_VERSION)


def _build_static_confirmation(args) -> LazyDVDMetaPromptConfirmationEvaluator:
    """Mirror checkpoint_g_factory's confirmation construction, static generator.

    The construction is deferred exactly like the incumbent factory so that no
    provider backend is spun up until an update decision actually needs the
    held-out confirmation. Only the final evaluator class differs.
    """
    cfg = _object(json.loads(Path(args.component_config).read_text(
        encoding="utf-8")), "component_config")
    runtime = dict(_object(_required(cfg, "runtime"), "runtime"))

    worker_gpus_override = getattr(args, "worker_gpus", None)
    if worker_gpus_override is not None:
        resolved = tuple(v.strip() for v in str(worker_gpus_override).split(",")
                         if v.strip())
        if not resolved or len(set(resolved)) != len(resolved):
            raise ValueError("worker_gpus override must be unique and non-empty")
        runtime["worker_gpus"] = list(resolved)

    gpus = tuple(str(item) for item in _required(runtime, "worker_gpus"))
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("worker_gpus must be unique")

    components_path = Path(str(_required(runtime, "scaffold_components_path")))
    components = json.loads(components_path.read_text(encoding="utf-8"))
    scaffold = scaffold_policy_from_json(components["scaffold_policy"])
    contract = scaffold_contract_from_json(components["scaffold_contract"])
    videos = tuple(DVDMetaPromptConfirmationVideo(
        video_id=str(item["video_id"]),
        provider_indices=tuple(int(v) for v in item["provider_indices"]),
        question_ids=tuple(str(v) for v in item["question_ids"]))
        for item in _required(cfg, "confirmation_videos"))
    bank_version = "bank_v9999"
    router_version = "router_v9999"

    def construct_confirmation():
        for path in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
            if path not in os.sys.path:
                os.sys.path.insert(0, path)
        from data_provider import get_provider
        from dvd_prompt import get_prompts
        provider_data = get_provider(
            str(_required(runtime, "benchmark")),
            split=str(_required(runtime, "benchmark_split")))
        ensure_backend(
            gpus[0], preload_captioner=False, preload_embedder=True,
            text_backend=str(_required(runtime, "dvd_text_backend")),
            use_openai_tools=bool(_required(runtime, "dvd_use_openai_tools")))
        builder_kwargs = {
            "parallel_gpus": gpus,
            "routing_mode": "free_form_generator",
        }
        if "worker_result_timeout_seconds" in runtime:
            builder_kwargs.update({
                "worker_result_timeout_seconds": float(
                    runtime["worker_result_timeout_seconds"]),
                "worker_log_directory": os.path.join(
                    str(_required(runtime, "cache_root")),
                    "worker_logs", "confirmation"),
            })
        builder = HistoryAwareBaselineCaptionViewBuilder.from_local_qwen(
            **builder_kwargs)
        prompts = get_prompts()
        dvd = HistoryAwareDVDConfirmationEvaluator(
            sample_loader=provider_data.__getitem__,
            history_aware_builder=builder,
            base_prompt_template=prompts.caption_prompt,
            merge_prompt=prompts.merge_prompt,
            sample_source_identity=str(_required(
                runtime, "sample_source_identity")),
            cache_root=str(_required(runtime, "cache_root")),
            cache_manifest_path=str(_required(runtime, "cache_manifest_path")),
            history_block_seconds=float(_required(
                runtime, "history_block_seconds")),
            max_history_captions=int(_required(runtime, "max_history_captions")),
            dvd_max_iterations=int(_required(runtime, "dvd_max_iterations")),
            gpu=gpus[0], frame_sampling_configuration={
                "sample_fps": runtime["sample_fps"],
                "clip_seconds": runtime["clip_seconds"],
                "segment_boundary_rule": "dvd_build_clips_fixed_duration",
                "frame_order_rule": "sorted_decoded_frame_paths"},
            downstream_qa_configuration={
                "text_backend": runtime["dvd_text_backend"],
                "use_openai_tools": runtime["dvd_use_openai_tools"]})
        return StaticGeneratorConfirmationEvaluator(
            evaluator=dvd, confirmation_videos=videos,
            bank_version=bank_version, router_version=router_version,
            scaffold_policy=scaffold, scaffold_contract=contract,
            # Vestigial under the static path (the generator makes no model
            # call); kept non-empty to satisfy the base contract and to mark
            # the identity as the static caption-prompt path.
            prompt_generator_model_id=str(runtime.get(
                "prompt_generator_model_id", "static")),
            prompt_generator_backend_id=str(runtime.get(
                "prompt_generator_backend_id", "static")),
            prompt_generator_max_tokens=int(runtime.get(
                "prompt_generator_max_tokens", 1)),
            model_identity=str(_required(runtime, "paired_model_identity")),
            decoding_settings=_object(
                _required(runtime, "paired_decoding_settings"),
                "runtime.paired_decoding_settings"),
            cache_reset_identity=str(_required(runtime, "cache_reset_identity")),
            evaluation_pipeline_identity=str(_required(
                runtime, "evaluation_pipeline_identity")))

    return LazyDVDMetaPromptConfirmationEvaluator(
        configuration_identity={
            "policy_version": StaticGeneratorConfirmationEvaluator.policy_version,
            "runtime": runtime,
            "confirmation_videos": videos,
            "legacy_property_routing_used": False,
            "construction": "deferred_until_update_decision",
            "caption_prompt_path": True,
        }, factory=construct_confirmation)


def build_caption_prompt_components(args) -> tuple[Any, Any, Any]:
    """Feedback (reused), caption-prompt updater, static-generator confirmation."""
    # build_checkpoint_g_components cross-checks the prepared runtime's
    # prompt_generator identity against config.PROMPT_GENERATOR_{MODEL,BACKEND}_ID
    # and aborts on a mismatch. On the static path the prepared runtime says
    # 'static', so align config to it in this process only (the generator is
    # never called; this is the same alignment run_static_evidence does for the
    # evidence process's own cross-check).
    config.PROMPT_GENERATOR_MODEL_ID = "static"
    config.PROMPT_GENERATOR_BACKEND_ID = "static"
    feedback, updater, _incumbent_confirmation = build_checkpoint_g_components(
        args)
    return feedback, _wrap_updater(updater), _build_static_confirmation(args)
