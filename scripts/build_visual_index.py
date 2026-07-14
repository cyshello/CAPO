"""Precompute cached SigLIP visual indexes for split videos (PHASE2_3 §6).

Usage:
    conda run -n local_llm_vllm python -m surrogate_rollout.scripts.build_visual_index \
        --split train --gpu 2
    # or explicit provider indices:
    ... --indices 9,10,11
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from surrogate_rollout import config

if config.PROMPT_SENS_ROOT not in sys.path:
    sys.path.insert(0, config.PROMPT_SENS_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, choices=["train", "validation", "test"])
    ap.add_argument("--indices", default=None, help="comma-separated provider indices")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--model", default=config.VISUAL_INDEX_MODEL_ID)
    args = ap.parse_args()
    if not args.split and not args.indices:
        ap.error("need --split or --indices")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from data_provider import get_provider
    from dvd_captioning import video_duration_seconds

    from surrogate_rollout.evaluation.dvd_qa import resolve_frames_dir
    from surrogate_rollout.retrieval.visual_index import (
        SiglipEmbedder,
        load_or_build_visual_index,
    )

    provider = get_provider(config.BENCHMARK, split=config.BENCHMARK_SPLIT)

    indices: list[int] = []
    if args.indices:
        indices = [int(i) for i in args.indices.split(",") if i.strip()]
    else:
        with open(config.SPLIT_MANIFEST_PATH) as f:
            manifest = json.load(f)
        for v in manifest["splits"][args.split]:
            indices.append(v["provider_indices"][0])  # one per video suffices

    embedder = SiglipEmbedder(args.model)
    done: set[str] = set()
    for idx in indices:
        sample = provider[idx]
        video_id = sample.get("extra", {}).get("videoID") or sample["sample_id"]
        if video_id in done:
            continue
        done.add(video_id)
        frames_dir = resolve_frames_dir(sample, video_id)
        duration = video_duration_seconds(sample["video_path"])
        index = load_or_build_visual_index(
            video_id=video_id, frames_dir=frames_dir, duration=duration,
            embedder=embedder, model_id=args.model)
        print(f"{video_id}: {len(index.frame_names)} frames, "
              f"{len(index.all_clip_ids)} clips, dim={index.embeddings.shape[1]}",
              flush=True)


if __name__ == "__main__":
    main()
