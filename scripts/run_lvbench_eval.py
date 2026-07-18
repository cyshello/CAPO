"""Full-benchmark caption + QA accuracy eval (no roles, no optimizer).

Evaluates one captioning routing mode (default: the ``free_form_generator``
ablation) over EVERY QA of a provider split:

  1. build the query-independent caption view once per video, then
  2. run each QA through the shared DVD QA evaluator (``run_dvd_qa``).

No property proposal / retrieval / intervention / feedback / updater stage runs,
and there is NO evidence-pool / rotation / three-QA-per-video assumption, so it
works on LVBench (variable QA count per video, 103 videos / 1549 QAs) which the
optimization launcher (``run_phase4_memory_iteration.py``) cannot load.

Reuses the existing services unchanged: ``build_clip_index``, the history-aware
caption view builder, and ``run_dvd_qa``. Captioning and per-QA results are
resumable from their on-disk artifacts, so a re-run performs no repeated work.

Example (LVBench pilot, free-form ablation):

    SR_BENCHMARK=lvbench SR_BENCHMARK_SPLIT=test \
    python -u scripts/run_lvbench_eval.py \
      --output-dir runs/freeform_lvbench_pilot_output \
      --cache-dir  runs/freeform_lvbench_pilot_cache \
      --gpus 4,5,6,7 \
      --routing-mode free_form_generator \
      --video-ids 28CIeC8cZks,KbahC-QCKU8,7HjNIPIgUg4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_PARENT = os.path.dirname(REPO_ROOT)
if REPO_PARENT not in sys.path:
    sys.path.insert(0, REPO_PARENT)

from surrogate_rollout import config
from surrogate_rollout.captioning.candidate_captions import build_clip_index
from surrogate_rollout.captioning.history_aware_baseline import (
    HistoryAwareBaselineCaptionViewBuilder,
)
from surrogate_rollout.evaluation.dvd_qa import ensure_backend, run_dvd_qa
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)

DEFAULT_COMPONENTS = os.path.join(
    REPO_ROOT, "prompt_routing", "fixtures", "stage4_7_components.json")


def _read_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")


def _bootstrap_components(components: dict):
    """Load the shared components for a captioning-only run.

    Free-form routing bypasses the router entirely, but the history-aware
    builder's ``build`` signature still needs a valid ``history_aware_vlm``
    router object; the property-bank comparison would instead supply the
    optimized pair. Here we take the frozen fixture pair as-is.
    """
    from dataclasses import replace

    bank = prompt_bank_from_json(components["prompt_bank"])
    source_router = router_policy_from_json(components["router_policy"])
    router = replace(
        source_router,
        policy_type="history_aware_vlm",
        configuration={
            **source_router.configuration,
            "max_selected_properties": 3,
            "max_tokens": 256,
        })
    scaffold = scaffold_policy_from_json(components["scaffold_policy"])
    contract = scaffold_contract_from_json(components["scaffold_contract"])
    return bank, router, scaffold, contract


def _grouped_samples(provider) -> "dict[str, list[tuple[int, dict]]]":
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index in range(len(provider)):
        sample = provider[index]
        video_id = sample.get("extra", {}).get("videoID") or sample["sample_id"]
        groups[str(video_id)].append((index, sample))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--components", default=DEFAULT_COMPONENTS)
    parser.add_argument(
        "--routing-mode", default="free_form_generator",
        choices=("free_form_generator", "property_bank"))
    parser.add_argument(
        "--video-ids",
        help="Comma-separated video IDs to evaluate (pilot); default = all.")
    parser.add_argument(
        "--max-videos", type=int,
        help="Cap the number of videos (applied after --video-ids filtering).")
    parser.add_argument("--dvd-max-iterations", type=int, default=10)
    args = parser.parse_args()

    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpus must contain unique GPU IDs")
    output_dir = os.path.abspath(args.output_dir)
    cache_dir = os.path.abspath(args.cache_dir)
    caption_cache_root = os.path.join(cache_dir, "captions")
    cache_manifest_path = os.path.join(cache_dir, "caption_cache_manifest.jsonl")

    for path in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from data_provider import get_provider
    from dvd_prompt import get_prompts

    ensure_backend(
        gpus[0], preload_captioner=False,
        text_backend=config.DVD_TEXT_BACKEND,
        use_openai_tools=config.DVD_USE_OPENAI_TOOLS)
    provider = get_provider(config.BENCHMARK, split=config.BENCHMARK_SPLIT)
    prompts = get_prompts()
    components = _read_json(args.components)
    bank, router, scaffold, contract = _bootstrap_components(components)
    history_builder = HistoryAwareBaselineCaptionViewBuilder.from_local_qwen(
        parallel_gpus=gpus, routing_mode=args.routing_mode)

    groups = _grouped_samples(provider)
    video_ids = list(groups)
    if args.video_ids:
        wanted = [item.strip() for item in args.video_ids.split(",")
                  if item.strip()]
        unknown = [item for item in wanted if item not in groups]
        if unknown:
            raise SystemExit(f"unknown --video-ids: {unknown}")
        video_ids = wanted
    if args.max_videos is not None:
        video_ids = video_ids[:args.max_videos]

    per_video: dict[str, dict] = {}
    try:
        for position, video_id in enumerate(video_ids, start=1):
            items = groups[video_id]
            sample0 = items[0][1]
            video_dir = os.path.join(output_dir, "videos", video_id)
            clip_index = build_clip_index(dict(sample0), video_id)
            caption_view = history_builder.build(
                sample=dict(sample0),
                clip_index=clip_index,
                prompt_bank=bank, router_policy=router,
                scaffold_policy=scaffold, scaffold_contract=contract,
                base_prompt_template=prompts.caption_prompt,
                merge_prompt=prompts.merge_prompt,
                work_root=os.path.join(video_dir, "history_aware_baseline"),
                candidate_cache_root=caption_cache_root,
                cache_manifest_path=cache_manifest_path,
                worker_gpus=gpus)

            rows = []
            for provider_index, sample in items:
                question_id = str(sample["sample_id"])
                qa_dir = os.path.join(
                    video_dir, "qa", question_id.replace("/", "-"))
                result_path = os.path.join(qa_dir, "result.json")
                if os.path.exists(result_path):  # resume: reuse prior QA
                    record = _read_json(result_path)
                    score = float(record.get("score") or 0.0)
                    prediction = record.get("parsed_answer")
                else:
                    result = run_dvd_qa(
                        captions_path=caption_view.captions_path,
                        sample=dict(sample), run_dir=qa_dir,
                        question_id=question_id,
                        database_path=caption_view.database_path,
                        max_iterations=args.dvd_max_iterations, gpu=gpus[0])
                    score = float(result.score)
                    prediction = result.parsed_answer
                rows.append({
                    "question_id": question_id,
                    "provider_index": provider_index,
                    "question_type": sample.get("extra", {}).get(
                        "question_type"),
                    "ground_truth": sample.get("answer"),
                    "prediction": prediction,
                    "score": score,
                    "correct": score > 0.0,
                })

            correct = sum(1 for row in rows if row["correct"])
            per_video[video_id] = {
                "correct": correct, "total": len(rows),
                "captions_path": caption_view.captions_path,
                "qas": rows,
            }
            _write_json(
                os.path.join(video_dir, "video_eval.json"),
                {"video_id": video_id, **per_video[video_id]})
            running_correct = sum(v["correct"] for v in per_video.values())
            running_total = sum(v["total"] for v in per_video.values())
            print(
                f"[{position}/{len(video_ids)}] {video_id}: "
                f"{correct}/{len(rows)} | running "
                f"{running_correct}/{running_total} "
                f"({running_correct / running_total:.4f})",
                flush=True)
    finally:
        history_builder.close_worker_pool()

    total_correct = sum(item["correct"] for item in per_video.values())
    total = sum(item["total"] for item in per_video.values())
    by_type: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in per_video.values():
        for row in item["qas"]:
            bucket = by_type[str(row["question_type"])]
            bucket["total"] += 1
            bucket["correct"] += int(row["correct"])
    summary = {
        "schema_version": "benchmark_caption_qa_eval_v1",
        "benchmark": config.BENCHMARK,
        "split": config.BENCHMARK_SPLIT,
        "routing_mode": args.routing_mode,
        "dvd_text_backend": config.DVD_TEXT_BACKEND,
        "num_videos": len(per_video),
        "correct": total_correct, "total": total,
        "accuracy": (total_correct / total) if total else None,
        "accuracy_by_question_type": {
            key: {**value,
                  "accuracy": value["correct"] / value["total"]
                  if value["total"] else None}
            for key, value in sorted(by_type.items())},
        "per_video": {key: {"correct": value["correct"],
                            "total": value["total"]}
                      for key, value in per_video.items()},
        "optimizer_executed": False,
    }
    summary_path = os.path.join(output_dir, "benchmark_eval.json")
    _write_json(summary_path, summary)
    print(summary_path, flush=True)
    print(
        f"OVERALL {config.BENCHMARK}/{config.BENCHMARK_SPLIT} "
        f"[{args.routing_mode}]: {total_correct}/{total} = "
        f"{summary['accuracy']}",
        flush=True)


if __name__ == "__main__":
    main()
