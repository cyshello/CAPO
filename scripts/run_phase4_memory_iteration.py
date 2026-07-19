"""Run one configurable memory-conditioned Phase 4 evidence iteration.

This launcher never runs confirmation and never writes confirmed/current.json.
Use --dry-run-plan to freeze and inspect selection without model startup.
"""

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
from surrogate_rollout.evaluation.dvd_qa import (
    DVD_EMBEDDER_PRELOAD_POLICY_VERSION,
    DVD_QA_CONCURRENCY_POLICY_VERSION,
)
from surrogate_rollout.optimization.final_iteration import Checkpoint3EOrchestrator
from surrogate_rollout.optimization.production_iteration import (
    MemoryConditionedProductionIterationRunner,
    ProductionIterationError,
    resolve_video_selection,
)
from surrogate_rollout.optimization.stage_logging import IterationStageLogger
from surrogate_rollout.optimization.train_roles import (
    EvidenceCoverageState, derive_train_roles,
)
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
    as_json_dict,
    dumps_canonical,
)
from surrogate_rollout.prompt_routing.structured_router_policy import (
    validate_rendered_prompt_configuration,
)
from surrogate_rollout.schemas import sha256_text


DEFAULT_COMPONENTS = os.path.join(
    REPO_ROOT, "prompt_routing", "fixtures", "stage4_7_components.json")


def _read_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_inventory() -> dict[str, int]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True)
    result = {}
    for line in query.stdout.splitlines():
        index, memory = (item.strip() for item in line.split(",", 1))
        result[index] = int(memory)
    return result


def _validate_gpu_configuration(
    *, gpus: tuple[str, ...], inventory: dict[str, int],
    max_parallel_videos: int, require_free: bool,
    embedding_gpu: str | None = None,
) -> None:
    if not gpus or len(gpus) != len(set(gpus)):
        raise ProductionIterationError("--gpus must contain unique GPU IDs")
    missing = set(gpus) - set(inventory)
    if missing:
        raise ProductionIterationError(f"unavailable GPUs: {sorted(missing)}")
    if not 1 <= max_parallel_videos <= len(gpus):
        raise ProductionIterationError(
            "--max-parallel-videos exceeds the usable worker count")
    if embedding_gpu is not None:
        if embedding_gpu in gpus:
            raise ProductionIterationError(
                "--embedding-gpu must not overlap --gpus caption workers")
        if embedding_gpu not in inventory:
            raise ProductionIterationError(
                f"unavailable embedding GPU: {embedding_gpu}")
    if require_free:
        occupied = {gpu: inventory[gpu] for gpu in gpus if inventory[gpu] >= 100}
        if embedding_gpu is not None and inventory[embedding_gpu] >= 100:
            occupied[embedding_gpu] = inventory[embedding_gpu]
        if occupied:
            raise ProductionIterationError(
                f"configured GPUs are not free: {occupied}")


def _embedding_device(embedding_gpu: str | None) -> str:
    return ("cpu" if embedding_gpu is None else
            f"isolated_process:physical_gpu={embedding_gpu}:logical_device=cuda:0")


def _load_parent_pair(*, components: dict, state_dir: str):
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
            "production_bootstrap": True,
        },
        provenance={
            "production_bootstrap": True,
            "source_router_version": source_router.router_version,
        })
    pointer_path = os.path.join(
        os.path.abspath(state_dir), "memory_conditioned_provisional", "current.json")
    if not os.path.exists(pointer_path):
        return bank, router, sha256_text(dumps_canonical({
            "bank": as_json_dict(bank), "router": as_json_dict(router)}))
    pointer = _read_json(pointer_path)
    if pointer.get("schema_version") != \
            "memory_conditioned_provisional_pair_pointer_v1":
        raise ProductionIterationError("incompatible parent policy-pair pointer")
    for name in ("bank", "router"):
        path = str(pointer.get(f"{name}_path") or "")
        if not os.path.exists(path) or _file_hash(path) != pointer.get(f"{name}_hash"):
            raise ProductionIterationError(f"parent {name} path/hash mismatch")
    bank = prompt_bank_from_json(_read_json(pointer["bank_path"]))
    router = router_policy_from_json(_read_json(pointer["router_path"]))
    if router.policy_type != "history_aware_vlm" or \
            not router.configuration.get("rendered_router_prompt_hash"):
        raise ProductionIterationError(
            "parent router is missing the rendered-prompt identity")
    try:
        validate_rendered_prompt_configuration(router, bank)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ProductionIterationError(
            "parent router has incompatible rendered-prompt semantics") from exc
    return bank, router, str(pointer["pair_manifest_hash"])


def _coverage_state(value: dict) -> EvidenceCoverageState:
    return EvidenceCoverageState(
        rotation_order=tuple(value["rotation_order"]),
        used_since_confirmation=tuple(value.get("used_since_confirmation") or ()),
        rotation_cursor=int(value.get("rotation_cursor", 0)),
        coverage_cycle=int(value.get("coverage_cycle", 0)),
        iterations_since_confirmation=int(value.get("iterations_since_confirmation", 0)),
        confirmation_due=bool(value.get("confirmation_due", False)),
    )


def _build_runtime(args, *, gpus, bank, router, components, split_hash):
    from surrogate_rollout.captioning.history_aware_baseline import (
        HistoryAwareBaselineCaptionViewBuilder,
    )
    from surrogate_rollout.optimization.baseline_phase import BaselinePhaseRunner
    from surrogate_rollout.optimization.interventional_feedback import (
        PropertyFeedbackRunner,
    )
    from surrogate_rollout.optimization.llm_codebook_updater import (
        MemoryConditionedLLMCodebookUpdater,
    )
    from surrogate_rollout.optimization.llm_router_updater import (
        MemoryConditionedLLMRouterUpdater,
    )
    from surrogate_rollout.optimization.policies.codex_feedback import (
        OpenAIStructuredFeedbackProvider,
    )
    from surrogate_rollout.optimization.policies.openai_update import (
        OpenAIJSONUpdateProvider,
        codebook_updater_response_schema,
        router_updater_response_schema,
    )
    from surrogate_rollout.optimization.policies.property_proposal import (
        MultiPropertyProposalPolicy, OpenAIPropertyProposalProvider,
    )
    from surrogate_rollout.optimization.property_intervention import (
        PropertyInterventionBatchRunner,
    )
    from surrogate_rollout.optimization.property_memory import (
        CompactPropertyMemoryRunner,
    )
    from surrogate_rollout.retrieval.property_retrieval import (
        PropertyFrameRetriever, PropertyRetrievalBatchRunner,
    )
    from surrogate_rollout.retrieval.visual_index import (
        ProcessIsolatedSiglipEmbedder, SiglipEmbedder,
        load_or_build_visual_index,
    )

    for path in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from data_provider import get_provider
    from dvd_captioning import video_duration_seconds
    from dvd_prompt import get_prompts
    from surrogate_rollout.evaluation.dvd_qa import ensure_backend, resolve_frames_dir

    primary_gpu = gpus[0]
    ensure_backend(
        primary_gpu, preload_captioner=False,
        preload_embedder=True,
        text_backend=config.DVD_TEXT_BACKEND,
        use_openai_tools=config.DVD_USE_OPENAI_TOOLS)
    provider = get_provider(config.BENCHMARK, split=config.BENCHMARK_SPLIT)
    sample_loader = provider.__getitem__
    prompts = get_prompts()
    history_builder = HistoryAwareBaselineCaptionViewBuilder.from_local_qwen(
        parallel_gpus=gpus,
        routing_mode=getattr(args, "routing_mode", "property_bank"))
    retrieval_embedder = (
        SiglipEmbedder(model_id=config.VISUAL_INDEX_MODEL_ID, device="cpu")
        if args.embedding_gpu is None else
        ProcessIsolatedSiglipEmbedder(
            model_id=config.VISUAL_INDEX_MODEL_ID,
            physical_gpu=args.embedding_gpu))

    def visual_index_loader(sample: dict, video_id: str):
        return load_or_build_visual_index(
            video_id=video_id,
            frames_dir=resolve_frames_dir(sample, video_id),
            duration=video_duration_seconds(sample["video_path"]),
            embedder=retrieval_embedder,
            root=os.path.join(os.path.abspath(args.cache_dir), "visual_indexes"),
            model_id=config.VISUAL_INDEX_MODEL_ID)

    retrieval_runner = PropertyRetrievalBatchRunner(
        retriever=PropertyFrameRetriever(
            embedder=retrieval_embedder,
            top_k=args.property_retrieval_top_k),
        visual_index_loader=visual_index_loader)
    proposal_policy = MultiPropertyProposalPolicy(
        response_provider=OpenAIPropertyProposalProvider(
            model=args.feedback_model),
        max_proposals=args.max_proposals_per_video)
    baseline_runner = BaselinePhaseRunner(
        proposal_policy=proposal_policy,
        history_aware_builder=history_builder,
        property_retrieval_runner=retrieval_runner)
    intervention_runner = PropertyInterventionBatchRunner(
        segment_captioner=history_builder.segment_captioner,
        caption_cache_root=os.path.join(os.path.abspath(args.cache_dir), "captions"),
        caption_cache_manifest_path=os.path.join(
            os.path.abspath(args.cache_dir), "caption_cache_manifest.jsonl"))
    feedback_runner = PropertyFeedbackRunner(
        response_provider=OpenAIStructuredFeedbackProvider(
            model=args.feedback_model))
    codebook_provider = OpenAIJSONUpdateProvider(
        model=args.feedback_model, max_calls=3,
        response_schema_name="phase4_codebook_update_plan",
        response_schema=codebook_updater_response_schema(max_actions=32))
    router_provider = OpenAIJSONUpdateProvider(
        model=args.feedback_model, max_calls=3,
        response_schema_name="phase4_router_update_plan",
        response_schema=router_updater_response_schema(max_actions=48))
    scaffold = scaffold_policy_from_json(components["scaffold_policy"])
    contract = scaffold_contract_from_json(components["scaffold_contract"])
    update_engine = Checkpoint3EOrchestrator.with_real_confirmation(
        baseline_runner=baseline_runner,
        intervention_runner=intervention_runner,
        feedback_runner=feedback_runner,
        confirmation_kwargs={
            "sample_loader": sample_loader,
            "history_aware_builder": history_builder,
            "base_prompt_template": prompts.caption_prompt,
            "merge_prompt": prompts.merge_prompt,
            "sample_source_identity": split_hash,
            "cache_root": os.path.join(
                os.path.abspath(args.cache_dir), "confirmation_not_run"),
            "cache_manifest_path": os.path.join(
                os.path.abspath(args.cache_dir),
                "confirmation_not_run_manifest.jsonl"),
            "gpu": primary_gpu,
            "dvd_max_iterations": args.dvd_max_iterations,
            "downstream_qa_configuration": {
                "text_backend": config.DVD_TEXT_BACKEND,
                "use_openai_tools": config.DVD_USE_OPENAI_TOOLS},
        },
        property_memory_runner=CompactPropertyMemoryRunner(),
        llm_codebook_updater=MemoryConditionedLLMCodebookUpdater(
            response_provider=codebook_provider),
        llm_router_updater=MemoryConditionedLLMRouterUpdater(
            response_provider=router_provider))
    runner = MemoryConditionedProductionIterationRunner(
        baseline_runner=baseline_runner,
        intervention_runner=intervention_runner,
        feedback_runner=feedback_runner,
        update_engine=update_engine,
        history_builder=history_builder)
    return {
        "runner": runner, "history_builder": history_builder,
        "retrieval_embedder": retrieval_embedder,
        "sample_loader": sample_loader, "prompts": prompts,
        "scaffold": scaffold, "contract": contract,
        "codebook_provider": codebook_provider,
        "router_provider": router_provider,
    }


def _write_cleanup(
    output_dir: str, history_builder, gpus, *, calls,
    embedding_gpu: str | None = None,
    embedding_model_closed: bool = True,
    embedding_worker_pid: int | None = None,
    embedding_worker_released: bool = True,
) -> None:
    root = os.path.join(os.path.abspath(output_dir), "worker_cleanup")
    os.makedirs(root, exist_ok=True)
    attempt = len([name for name in os.listdir(root) if name.endswith(".json")]) + 1
    workers = tuple((gpu, process.pid, process)
                    for gpu, process in history_builder._pool_processes.items()) \
        if history_builder is not None else ()
    if history_builder is not None:
        history_builder.close_worker_pool()
    inventory = {}
    try:
        inventory = _gpu_inventory()
    except BaseException:
        inventory = {gpu: None for gpu in gpus}
    configured = {gpu: inventory.get(gpu) for gpu in gpus}
    embedding_memory = (
        inventory.get(embedding_gpu) if embedding_gpu is not None else None)
    released = (
        (history_builder is None or not history_builder._pool_processes)
        and not any(process.is_alive() for _, _, process in workers)
        and embedding_worker_released
        and all(value is not None and value < 100 for value in configured.values()))
    path = os.path.join(root, f"attempt_{attempt:03d}.json")
    with open(path, "w") as handle:
        json.dump({
            "schema_version": "production_worker_cleanup_v2",
            "configured_gpus": gpus,
            "embedding_gpu": embedding_gpu,
            "embedding_model_closed": embedding_model_closed,
            "embedding_worker_pid": embedding_worker_pid,
            "embedding_worker_released": embedding_worker_released,
            "embedding_gpu_memory_before_parent_exit_mib": embedding_memory,
            "workers_before_close": tuple({"gpu": gpu, "pid": pid}
                                          for gpu, pid, _ in workers),
            "workers_alive_after_close": tuple(
                {"gpu": gpu, "pid": pid} for gpu, pid, process in workers
                if process.is_alive()),
            "gpu_memory_after_close_mib": configured,
            "all_workers_released": released,
            "component_update_provider_calls": calls,
        }, handle, sort_keys=True, indent=2)
        handle.write("\n")
    if sys.exc_info()[0] is None and not released:
        raise RuntimeError("production workers did not release every configured GPU")


def _run_baseline_only(
    *, selected, coverage_state, roles, bank, router, scaffold, contract,
    phase4_config, output_dir, cache_dir, gpus, history_builder,
    baseline_kwargs, routing_mode,
) -> str:
    """Caption + baseline QA only, no optimization-time stage.

    Reuses BaselinePhaseRunner with a no-op proposer and no retrieval runner so
    the free-form ablation's accuracy is measured without invoking any paid
    optimizer LLM stage (proposal/retrieval/intervention/feedback/updaters).
    Videos run sequentially, each using every worker GPU. Returns the summary
    JSON path.
    """
    from surrogate_rollout.optimization.baseline_phase import BaselinePhaseRunner

    output_dir = os.path.abspath(output_dir)
    cache_dir = os.path.abspath(cache_dir)
    runner = BaselinePhaseRunner(
        proposal_policy=None,             # -> NoOpPropertyProposalPolicy
        history_aware_builder=history_builder,
        property_retrieval_runner=None)   # skip similarity retrieval
    caption_cache_root = os.path.join(cache_dir, "captions")
    cache_manifest_path = os.path.join(cache_dir, "caption_cache_manifest.jsonl")
    per_video: dict[str, dict] = {}
    for video_id in selected:
        result = runner.run(
            roles=roles, coverage_state=coverage_state,
            prompt_bank=bank, router_policy=router,
            scaffold_policy=scaffold, scaffold_contract=contract,
            phase4_config=phase4_config,
            output_dir=os.path.join(output_dir, "baseline_videos", video_id),
            bounded_smoke_video_id=video_id,
            caption_cache_root=caption_cache_root,
            cache_manifest_path=cache_manifest_path,
            worker_gpus=gpus,
            **dict(baseline_kwargs))
        manifest = _read_json(result.video_manifest_paths[0])
        with open(manifest["baseline_qas_path"]) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        correct = sum(1 for row in rows if row.get("is_correct"))
        per_video[video_id] = {
            "correct": correct, "total": len(rows),
            "baseline_qas_path": manifest["baseline_qas_path"],
            "video_manifest_path": result.video_manifest_paths[0]}
    total_correct = sum(item["correct"] for item in per_video.values())
    total = sum(item["total"] for item in per_video.values())
    summary = {
        "schema_version": "free_form_baseline_eval_v1",
        "routing_mode": routing_mode,
        "benchmark": config.BENCHMARK,
        "split": config.BENCHMARK_SPLIT,
        "selected_video_ids": list(selected),
        "per_video": per_video,
        "correct": total_correct, "total": total,
        "accuracy": (total_correct / total) if total else None,
        "optimizer_executed": False,
    }
    summary_path = os.path.join(output_dir, "baseline_only_eval.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--num-videos", type=int)
    parser.add_argument("--video-ids")
    parser.add_argument("--max-parallel-videos", type=int)
    parser.add_argument("--gpus", required=True)
    parser.add_argument(
        "--embedding-gpu",
        help="dedicated SigLIP GPU; must not overlap --gpus (default: CPU)")
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--components", default=DEFAULT_COMPONENTS)
    parser.add_argument("--split-manifest", default=config.SPLIT_MANIFEST_PATH)
    parser.add_argument("--feedback-model", default=config.FEEDBACK_MODEL)
    parser.add_argument("--dvd-max-iterations", type=int, default=10)
    parser.add_argument(
        "--routing-mode", default="property_bank",
        choices=("property_bank", "free_form_generator"),
        help="Experimental ablation only: 'free_form_generator' replaces the "
             "property router+bank instruction with an on-the-fly generated "
             "instruction. Default is the existing property_bank behavior. "
             "Use a separate --output-dir/--state-dir/--cache-dir for "
             "free-form runs.")
    parser.add_argument(
        "--max-proposals-per-video", type=int,
        default=config.MAX_PROPERTY_PROPOSALS_PER_VIDEO)
    parser.add_argument(
        "--property-retrieval-top-k", type=int,
        default=config.PROPERTY_RETRIEVAL_TOP_K)
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Caption + baseline QA only: measure accuracy without running any "
             "optimization-time stage (property proposal, retrieval, "
             "intervention, feedback, codebook/router updaters). Uses a no-op "
             "proposer and no retrieval runner, so no paid optimizer LLM calls "
             "are made. Writes baseline_only_eval.json with per-video and "
             "overall accuracy. Intended for the free-form ablation vs "
             "property-bank captioning comparison.")
    args = parser.parse_args()
    if args.baseline_only and args.dry_run_plan:
        raise SystemExit("--baseline-only cannot be combined with --dry-run-plan")

    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    inventory = _gpu_inventory()
    components = _read_json(args.components)
    split_hash = _file_hash(args.split_manifest)
    roles = derive_train_roles(_read_json(args.split_manifest))
    explicit = tuple(item.strip() for item in (args.video_ids or "").split(",")
                     if item.strip())
    saved_identity_path = os.path.join(
        os.path.abspath(args.output_dir), "iteration_identity.json")
    if os.path.exists(saved_identity_path):
        saved_wrapper = _read_json(saved_identity_path)
        saved = saved_wrapper.get("identity") or {}
        if saved.get("iteration_id") != args.iteration_id or \
                saved.get("split_manifest_hash") != split_hash or \
                int(saved.get("selection_seed")) != args.selection_seed:
            raise SystemExit("saved output identity conflicts with requested iteration")
        selected = tuple(saved.get("ordered_video_ids") or ())
        if explicit and explicit != selected:
            raise SystemExit("--video-ids conflicts with saved ordered video list")
        if args.num_videos is not None and args.num_videos != len(selected):
            raise SystemExit("--num-videos conflicts with saved iteration size")
        selection_state = _coverage_state(saved["coverage_state"])
        next_coverage = _coverage_state(saved["next_coverage_state"])
        saved_components = saved.get("components") or {}
        bank = prompt_bank_from_json(saved_components["bank"])
        router = router_policy_from_json(saved_components["router"])
        parent_pair_hash = str(saved["parent_policy_pair_hash"])
    else:
        bank, router, parent_pair_hash = _load_parent_pair(
            components=components, state_dir=args.state_dir)
        selection_state = \
            MemoryConditionedProductionIterationRunner.load_selection_state(
                roles=roles, state_dir=args.state_dir,
                selection_seed=args.selection_seed)
        selected, next_coverage = resolve_video_selection(
            roles=roles, state=selection_state,
            num_videos=args.num_videos, video_ids=explicit)
    max_parallel = args.max_parallel_videos or min(len(selected), len(gpus))
    try:
        _validate_gpu_configuration(
            gpus=gpus, inventory=inventory,
            max_parallel_videos=max_parallel,
            require_free=not args.dry_run_plan,
            embedding_gpu=args.embedding_gpu)
    except ProductionIterationError as exc:
        raise SystemExit(str(exc)) from exc
    phase4_config = Phase4Config(
        seed=args.selection_seed, dry_run=False, commit=False,
        post_intervention_mode=PostInterventionMode.PROVISIONAL_UPDATE,
        optimization=Phase4OptimizationConfig(
            optimize_prompt_bank=True, optimize_router=True,
            optimize_scaffold=False, max_iterations=1))
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        check=True, capture_output=True, text=True).stdout.strip()
    execution_identity = {
        "source_revision": revision,
        "split_manifest_hash": split_hash,
        "feedback_model": args.feedback_model,
        "dvd_max_iterations": args.dvd_max_iterations,
        "gpu_inventory_at_plan": {
            gpu: inventory[gpu] for gpu in
            (*gpus, *((args.embedding_gpu,) if args.embedding_gpu else ()))},
        "caption_model": config.CAPTION_MODEL_ID,
        "caption_decoding": config.CAPTION_DECODING,
        "retrieval_model": config.VISUAL_INDEX_MODEL_ID,
        "retrieval_device": _embedding_device(args.embedding_gpu),
        "embedding_gpu": args.embedding_gpu,
        "embedding_process_isolation": args.embedding_gpu is not None,
        "max_proposals_per_video": args.max_proposals_per_video,
        "property_retrieval_top_k": args.property_retrieval_top_k,
        "dvd_clip_search_top_k": config.DVD_CLIP_SEARCH_TOP_K,
        "dvd_clip_search_policy_version": config.DVD_CLIP_SEARCH_POLICY_VERSION,
        "dvd_embedding_preload_policy_version":
            DVD_EMBEDDER_PRELOAD_POLICY_VERSION,
        "dvd_qa_concurrency_policy_version":
            DVD_QA_CONCURRENCY_POLICY_VERSION,
        "codebook_updater_prompt_version": "memory_codebook_updater_prompt_v3_deterministic_ids",
        "router_updater_prompt_version": "memory_router_updater_prompt_v2_semantic_llm",
        "structured_router_policy_version": "structured_router_policy_v2_total_examples",
        "rendered_router_prompt_version": "rendered_router_prompt_v1",
    }
    runtime = None
    history_builder = None
    calls = {"codebook": 0, "router": 0}
    try:
        if args.dry_run_plan:
            runner = MemoryConditionedProductionIterationRunner(
                baseline_runner=None, intervention_runner=None,
                feedback_runner=None,
                update_engine=Checkpoint3EOrchestrator(
                    baseline_runner=None, intervention_runner=None,
                    feedback_runner=None, confirmation_evaluator=None,
                    require_real_models=False),
                history_builder=None, require_real_models=False)
            scaffold = scaffold_policy_from_json(components["scaffold_policy"])
            contract = scaffold_contract_from_json(components["scaffold_contract"])
            baseline_kwargs = intervention_kwargs = {}
        else:
            runtime = _build_runtime(
                args, gpus=gpus, bank=bank, router=router,
                components=components, split_hash=split_hash)
            runner = runtime["runner"]
            history_builder = runtime["history_builder"]
            scaffold, contract = runtime["scaffold"], runtime["contract"]
            baseline_kwargs = {
                "sample_loader": runtime["sample_loader"],
                "base_prompt_template": runtime["prompts"].caption_prompt,
                "merge_prompt": runtime["prompts"].merge_prompt,
                "parent_confirmed_checkpoint_id": "memory_production_parent",
                "source_revision": revision,
                "gpu": gpus[0],
                "dvd_max_iterations": args.dvd_max_iterations,
            }
            intervention_kwargs = {
                "sample_loader": runtime["sample_loader"],
                "base_prompt_template": runtime["prompts"].caption_prompt,
                "merge_prompt": runtime["prompts"].merge_prompt,
                "gpu": gpus[0],
                "dvd_max_iterations": args.dvd_max_iterations,
            }
        if args.baseline_only:
            summary_path = _run_baseline_only(
                selected=selected, coverage_state=selection_state, roles=roles,
                bank=bank, router=router, scaffold=scaffold, contract=contract,
                phase4_config=phase4_config, output_dir=args.output_dir,
                cache_dir=args.cache_dir, gpus=gpus,
                history_builder=history_builder,
                baseline_kwargs=baseline_kwargs,
                routing_mode=getattr(args, "routing_mode", "property_bank"))
            print(summary_path, flush=True)
        else:
            result = runner.run(
                iteration_id=args.iteration_id, roles=roles,
                coverage_state=selection_state,
                next_coverage_state=next_coverage,
                selected_video_ids=selected,
                selection_seed=args.selection_seed,
                split_manifest_path=args.split_manifest,
                split_manifest_hash=split_hash,
                parent_policy_pair_hash=parent_pair_hash,
                prompt_bank=bank, router_policy=router,
                scaffold_policy=scaffold, scaffold_contract=contract,
                phase4_config=phase4_config,
                output_dir=args.output_dir, state_dir=args.state_dir,
                cache_dir=args.cache_dir, gpus=gpus,
                max_parallel_videos=max_parallel,
                execution_identity=execution_identity,
                baseline_kwargs=baseline_kwargs,
                intervention_kwargs=intervention_kwargs,
                dry_run_plan=args.dry_run_plan)
            if runtime is not None:
                calls = {
                    "codebook": runtime["codebook_provider"].call_count,
                    "router": runtime["router_provider"].call_count,
                }
            print(result.manifest_path, flush=True)
    finally:
        if not args.dry_run_plan:
            cleanup_log = IterationStageLogger(
                os.path.join(os.path.abspath(args.output_dir), "iteration.log"))
            cleanup_log.emit(
                event="START", stage="worker_cleanup",
                message="iteration model worker 및 embedding process 종료 시작",
                gpu_ids=gpus, embedding_gpu=args.embedding_gpu)
            embedding_model_closed = args.embedding_gpu is None
            embedding_worker_pid = None
            embedding_worker_released = args.embedding_gpu is None
            if runtime is not None:
                calls = {
                    "codebook": runtime["codebook_provider"].call_count,
                    "router": runtime["router_provider"].call_count,
                }
            try:
                if runtime is not None:
                    embedding_worker_pid = getattr(
                        runtime["retrieval_embedder"], "worker_pid", None)
                    runtime["retrieval_embedder"].close()
                    embedding_model_closed = True
                    embedding_worker_released = not bool(getattr(
                        runtime["retrieval_embedder"], "is_alive", False))
            finally:
                from surrogate_rollout.retrieval.visual_index import (
                    close_all_isolated_siglip_embedders,
                )
                orphan_records = close_all_isolated_siglip_embedders()
                if orphan_records:
                    embedding_worker_pid = int(orphan_records[-1]["pid"])
                    embedding_worker_released = all(
                        bool(item["released"]) for item in orphan_records)
                    embedding_model_closed = embedding_worker_released
                try:
                    _write_cleanup(
                        args.output_dir, history_builder, gpus, calls=calls,
                        embedding_gpu=args.embedding_gpu,
                        embedding_model_closed=embedding_model_closed,
                        embedding_worker_pid=embedding_worker_pid,
                        embedding_worker_released=embedding_worker_released)
                except BaseException as exc:
                    cleanup_log.emit(
                        event="ERROR", stage="worker_cleanup",
                        message="iteration process 종료 실패",
                        error_type=type(exc).__name__, error=str(exc))
                    raise
                cleanup_log.emit(
                    event="END", stage="worker_cleanup",
                    message="모든 iteration process 종료 완료; launcher exit",
                    gpu_ids=gpus, embedding_gpu=args.embedding_gpu)


if __name__ == "__main__":
    main()
