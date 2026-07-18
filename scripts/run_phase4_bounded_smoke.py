"""Run one isolated real Phase 4 evidence-video smoke without confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_PARENT = os.path.dirname(REPO_ROOT)
if REPO_PARENT not in sys.path:
    sys.path.insert(0, REPO_PARENT)

from surrogate_rollout import config
from surrogate_rollout.captioning.history_aware_baseline import (
    HistoryAwareBaselineCaptionViewBuilder,
)
from surrogate_rollout.optimization.baseline_phase import BaselinePhaseRunner
from surrogate_rollout.optimization.bounded_smoke import BoundedSmokeRunner
from surrogate_rollout.optimization.confirmation_evaluator import (
    HistoryAwareDVDConfirmationEvaluator,
)
from surrogate_rollout.optimization.final_iteration import Checkpoint3EOrchestrator
from surrogate_rollout.optimization.llm_codebook_updater import (
    MemoryConditionedLLMCodebookUpdater,
)
from surrogate_rollout.optimization.llm_router_updater import (
    MemoryConditionedLLMRouterUpdater,
)
from surrogate_rollout.optimization.interventional_feedback import PropertyFeedbackRunner
from surrogate_rollout.optimization.policies.codex_feedback import (
    OpenAIStructuredFeedbackProvider,
)
from surrogate_rollout.optimization.policies.property_proposal import (
    MultiPropertyProposalPolicy,
    OpenAIPropertyProposalProvider,
)
from surrogate_rollout.optimization.policies.openai_update import (
    OpenAIJSONUpdateProvider,
    codebook_updater_response_schema,
    router_updater_response_schema,
)
from surrogate_rollout.optimization.property_memory import (
    CompactPropertyMemoryRunner,
)
from surrogate_rollout.optimization.property_intervention import (
    PropertyInterventionBatchRunner,
)
from surrogate_rollout.optimization.train_roles import derive_train_roles, initial_coverage
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.schemas import (
    Phase4Config,
    Phase4OptimizationConfig,
    PostInterventionMode,
)
from surrogate_rollout.retrieval.property_retrieval import (
    PropertyFrameRetriever,
    PropertyRetrievalBatchRunner,
)
from surrogate_rollout.retrieval.visual_index import (
    SiglipEmbedder,
    load_or_build_visual_index,
)


ROOT = REPO_ROOT
DEFAULT_COMPONENTS = os.path.join(
    ROOT, "prompt_routing", "fixtures", "stage4_7_components.json")
DEFAULT_VIDEO_ID = "0RxMZBLeqRI"


def _load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _build_retrieval_runner(*, cache_dir: str, video_duration_seconds):
    """Keep SigLIP off GPUs reserved for the persistent Qwen worker pool."""
    retrieval_embedder = SiglipEmbedder(
        model_id=config.VISUAL_INDEX_MODEL_ID, device="cpu")

    def visual_index_loader(sample: dict, video_id: str):
        from surrogate_rollout.evaluation.dvd_qa import resolve_frames_dir

        frames_dir = resolve_frames_dir(sample, video_id)
        return load_or_build_visual_index(
            video_id=video_id, frames_dir=frames_dir,
            duration=video_duration_seconds(sample["video_path"]),
            embedder=retrieval_embedder,
            root=os.path.join(cache_dir, "visual_indexes"),
            model_id=config.VISUAL_INDEX_MODEL_ID)

    return PropertyRetrievalBatchRunner(
        retriever=PropertyFrameRetriever(
            embedder=retrieval_embedder, top_k=1),
        visual_index_loader=visual_index_loader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--post-intervention-mode", required=True,
        choices=tuple(item.value for item in PostInterventionMode))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument("--components", default=DEFAULT_COMPONENTS)
    parser.add_argument("--split-manifest", default=config.SPLIT_MANIFEST_PATH)
    parser.add_argument(
        "--gpu", default=None,
        help="primary GPU for downstream QA and selective intervention")
    parser.add_argument(
        "--gpus", default=None,
        help="comma-separated GPUs for history-block-parallel baseline captioning")
    parser.add_argument("--feedback-model", default=config.FEEDBACK_MODEL)
    parser.add_argument("--dvd-max-iterations", type=int, default=10)
    args = parser.parse_args()
    caption_gpus = tuple(
        value.strip() for value in str(args.gpus or "").split(",")
        if value.strip())
    if len(caption_gpus) != len(set(caption_gpus)):
        raise SystemExit("--gpus must contain unique GPU IDs")
    primary_gpu = str(args.gpu or (caption_gpus[0] if caption_gpus else "0"))
    if not caption_gpus:
        caption_gpus = (primary_gpu,)

    components = _load_json(args.components)
    roles = derive_train_roles(_load_json(args.split_manifest))
    record = next((item for item in roles.evidence_videos
                   if item.video_id == args.video_id), None)
    if record is None:
        raise SystemExit("--video-id must identify a frozen evidence-pool video")

    for path in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from data_provider import get_provider
    from dvd_captioning import video_duration_seconds
    from dvd_prompt import get_prompts
    from surrogate_rollout.evaluation.dvd_qa import ensure_backend

    provider = get_provider(config.BENCHMARK, split=config.BENCHMARK_SPLIT)
    sample_loader = provider.__getitem__
    samples = tuple(sample_loader(index) for index in record.provider_indices)
    actual_ids = tuple(
        item.get("extra", {}).get("videoID") or item.get("sample_id")
        for item in samples)
    if actual_ids != (record.video_id,) * 3:
        raise RuntimeError("frozen provider indices no longer identify one video")

    bank = prompt_bank_from_json(components["prompt_bank"])
    source_router = router_policy_from_json(components["router_policy"])
    router = replace(
        source_router, router_version="router_v9002",
        parent_router_version=source_router.router_version,
        policy_type="history_aware_vlm",
        configuration={
            **source_router.configuration,
            "max_selected_properties": 3,
            "max_tokens": 256,
            "bounded_smoke_bootstrap": True,
        },
        provenance={
            "bounded_smoke": True,
            "source_router_version": source_router.router_version,
        })
    scaffold = scaffold_policy_from_json(components["scaffold_policy"])
    contract = scaffold_contract_from_json(components["scaffold_contract"])
    prompts = get_prompts()

    output_dir = os.path.abspath(args.output_dir)
    state_dir = os.path.abspath(args.state_dir)
    cache_dir = os.path.abspath(args.cache_dir)
    ensure_backend(
        primary_gpu, preload_captioner=len(caption_gpus) == 1,
        text_backend="codex", use_openai_tools=False)
    history_builder = HistoryAwareBaselineCaptionViewBuilder.from_local_qwen(
        parallel_gpus=caption_gpus)
    proposal_policy = MultiPropertyProposalPolicy(
        response_provider=OpenAIPropertyProposalProvider(
            model=args.feedback_model), max_proposals=1)

    retrieval_runner = _build_retrieval_runner(
        cache_dir=cache_dir,
        video_duration_seconds=video_duration_seconds)
    baseline_runner = BaselinePhaseRunner(
        proposal_policy=proposal_policy,
        history_aware_builder=history_builder,
        property_retrieval_runner=retrieval_runner)
    intervention_runner = PropertyInterventionBatchRunner(
        segment_captioner=history_builder.segment_captioner,
        caption_cache_root=os.path.join(cache_dir, "captions"),
        caption_cache_manifest_path=os.path.join(
            cache_dir, "caption_cache_manifest.jsonl"))
    feedback_runner = PropertyFeedbackRunner(
        response_provider=OpenAIStructuredFeedbackProvider(
            model=args.feedback_model, max_calls=3))
    confirmation_evaluator = HistoryAwareDVDConfirmationEvaluator(
        sample_loader=sample_loader, history_aware_builder=history_builder,
        base_prompt_template=prompts.caption_prompt,
        merge_prompt=prompts.merge_prompt,
        sample_source_identity=os.path.abspath(args.split_manifest),
        cache_root=os.path.join(cache_dir, "confirmation_disabled"),
        cache_manifest_path=os.path.join(
            cache_dir, "confirmation_disabled_manifest.jsonl"),
        gpu=primary_gpu, dvd_max_iterations=args.dvd_max_iterations,
        downstream_qa_configuration={
            "text_backend": "codex", "use_openai_tools": False})
    codebook_update_provider = OpenAIJSONUpdateProvider(
        model=args.feedback_model, max_calls=3,
        response_schema_name="phase4_codebook_update_plan",
        response_schema=codebook_updater_response_schema(max_actions=32))
    router_update_provider = OpenAIJSONUpdateProvider(
        model=args.feedback_model, max_calls=3,
        response_schema_name="phase4_router_update_plan",
        response_schema=router_updater_response_schema(max_actions=48))
    update_engine = Checkpoint3EOrchestrator(
        baseline_runner=baseline_runner,
        intervention_runner=intervention_runner,
        feedback_runner=feedback_runner,
        confirmation_evaluator=confirmation_evaluator,
        property_memory_runner=CompactPropertyMemoryRunner(),
        llm_codebook_updater=MemoryConditionedLLMCodebookUpdater(
            response_provider=codebook_update_provider),
        llm_router_updater=MemoryConditionedLLMRouterUpdater(
            response_provider=router_update_provider))
    runner = BoundedSmokeRunner(
        baseline_runner=baseline_runner,
        intervention_runner=intervention_runner,
        feedback_runner=feedback_runner,
        update_engine=update_engine,
        memory_conditioned_update=True)
    phase4_config = Phase4Config(
        seed=0, dry_run=True, commit=False,
        post_intervention_mode=PostInterventionMode(
            args.post_intervention_mode),
        optimization=Phase4OptimizationConfig(
            optimize_prompt_bank=True, optimize_router=True,
            optimize_scaffold=False, max_iterations=1,
            max_new_entries_per_iteration=1,
            max_router_operations_per_iteration=1))
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    try:
        result = runner.run(
            iteration_id=f"bounded-smoke-{record.video_id}",
            video_id=record.video_id, roles=roles,
            coverage_state=initial_coverage(roles),
            phase4_config=phase4_config,
            prompt_bank=bank, router_policy=router,
            scaffold_policy=scaffold, scaffold_contract=contract,
            output_dir=output_dir, state_dir=state_dir, cache_dir=cache_dir,
            execution_identity={
                "source_revision": revision, "seed": 0,
                "provider_indices": record.provider_indices,
                "question_ids": record.question_ids,
                "feedback_model": args.feedback_model,
                "codebook_update_provider":
                    codebook_update_provider.metadata(),
                "router_update_provider": router_update_provider.metadata(),
                "dvd_max_iterations": args.dvd_max_iterations,
                "gpu": primary_gpu,
                "retrieval_device": "cpu",
                "dvd_text_backend": "codex",
                "dvd_use_openai_tools": False,
                "base_prompt_hash": _text_hash(prompts.caption_prompt),
                "merge_prompt_hash": _text_hash(prompts.merge_prompt),
                "components_path": os.path.abspath(args.components),
                "split_manifest_path": os.path.abspath(args.split_manifest),
            },
            baseline_kwargs={
                "sample_loader": sample_loader,
                "base_prompt_template": prompts.caption_prompt,
                "merge_prompt": prompts.merge_prompt,
                "parent_confirmed_checkpoint_id": "bounded_smoke_input",
                "source_revision": revision, "gpu": primary_gpu,
                "dvd_max_iterations": args.dvd_max_iterations,
            },
            intervention_kwargs={
                "sample_loader": sample_loader,
                "base_prompt_template": prompts.caption_prompt,
                "merge_prompt": prompts.merge_prompt,
                "gpu": primary_gpu,
                "dvd_max_iterations": args.dvd_max_iterations,
            })
    finally:
        worker_processes = tuple(
            (gpu, process.pid, process)
            for gpu, process in history_builder._pool_processes.items())
        history_builder.close_worker_pool()
        gpu_memory_after_close = {}
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True)
            for line in query.stdout.splitlines():
                index, memory = (item.strip() for item in line.split(",", 1))
                if index in caption_gpus:
                    gpu_memory_after_close[index] = int(memory)
        except BaseException:
            gpu_memory_after_close = {gpu: None for gpu in caption_gpus}
        all_workers_released = (
            not history_builder._pool_processes and
            not any(process.is_alive() for _, _, process in worker_processes) and
            set(gpu_memory_after_close) == set(caption_gpus) and
            all(memory is not None and memory < 100
                for memory in gpu_memory_after_close.values()))
        cleanup_path = os.path.join(output_dir, "worker_cleanup.json")
        os.makedirs(os.path.dirname(cleanup_path), exist_ok=True)
        with open(cleanup_path, "w") as handle:
            json.dump({
                "schema_version": "bounded_smoke_worker_cleanup_v1",
                "configured_gpus": caption_gpus,
                "workers_before_close": tuple({
                    "gpu": gpu, "pid": pid,
                } for gpu, pid, _ in worker_processes),
                "workers_alive_after_close": tuple({
                    "gpu": gpu, "pid": pid,
                } for gpu, pid, process in worker_processes
                if process.is_alive()),
                "pool_empty_after_close": not history_builder._pool_processes,
                "gpu_memory_after_close_mib": gpu_memory_after_close,
                "all_workers_released": all_workers_released,
                "component_update_provider_calls": {
                    "codebook": codebook_update_provider.call_count,
                    "router": router_update_provider.call_count,
                },
            }, handle, sort_keys=True, ensure_ascii=False, indent=2)
            handle.write("\n")
        if sys.exc_info()[0] is None and not all_workers_released:
            raise RuntimeError(
                "bounded smoke workers exited but GPU memory was not released")
    print(result.manifest_path, flush=True)


if __name__ == "__main__":
    main()
