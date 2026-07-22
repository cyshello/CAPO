"""Build the deterministic video-level split manifest (CLAUDE.md §5).

train / validation / test: 10 videos x 3 QAs each. The five videos already
captioned by earlier prompt_sensitivity experiments are pinned to train and
flagged `previously_cached` — they must never appear in validation or test.
Videos without subtitles are NOT excluded (DVD runs them in notx mode); their
subtitle availability is recorded per video.

Usage:
    conda run -n local_llm_vllm python -m surrogate_rollout.scripts.make_split_manifest
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime

from surrogate_rollout import config

if config.PROMPT_SENS_ROOT not in sys.path:
    sys.path.insert(0, config.PROMPT_SENS_ROOT)


def make_split(video_ids: list[str], pinned_train: list[str],
               per_split: int, seed: int) -> dict[str, list[str]]:
    """Pure deterministic split. `video_ids` order does not matter (sorted
    internally); pinned videos land in train and are excluded from sampling."""
    pool = sorted(set(video_ids) - set(pinned_train))
    missing_pins = [v for v in pinned_train if v not in set(video_ids)]
    if missing_pins:
        raise ValueError(f"pinned videos not in dataset: {missing_pins}")
    n_needed = (per_split - len(pinned_train)) + 2 * per_split
    if len(pool) < n_needed:
        raise ValueError(f"pool too small: {len(pool)} < {n_needed}")

    rng = random.Random(seed)
    sampled = rng.sample(pool, n_needed)
    n_train_rest = per_split - len(pinned_train)
    return {
        "train": sorted(pinned_train) + sorted(sampled[:n_train_rest]),
        "validation": sorted(sampled[n_train_rest:n_train_rest + per_split]),
        "test": sorted(sampled[n_train_rest + per_split:]),
    }


def main() -> None:
    from data_provider import get_provider

    provider = get_provider(config.BENCHMARK, split=config.BENCHMARK_SPLIT)
    videos: dict[str, dict] = {}
    for i in range(len(provider)):
        s = provider[i]
        vid = s.get("extra", {}).get("videoID") or s["sample_id"]
        info = videos.setdefault(vid, {
            "video_id": vid,
            "provider_indices": [],
            "sample_ids": [],
            "subtitle_available": False,
            "video_available": False,
        })
        info["provider_indices"].append(i)
        info["sample_ids"].append(s.get("sample_id"))
        if not info["video_available"]:
            info["video_available"] = os.path.exists(s["video_path"])
            sp = s.get("extra", {}).get("subtitle_path")
            info["subtitle_available"] = bool(sp) and os.path.exists(sp)

    excluded = {v: "video file missing"
                for v, info in videos.items() if not info["video_available"]}
    usable = [v for v in videos if v not in excluded]

    split = make_split(usable, list(config.PREVIOUSLY_CACHED_VIDEOS),
                       config.VIDEOS_PER_SPLIT, config.SPLIT_SEED)

    manifest = {
        "benchmark": config.BENCHMARK,
        "benchmark_split": config.BENCHMARK_SPLIT,
        "seed": config.SPLIT_SEED,
        "videos_per_split": config.VIDEOS_PER_SPLIT,
        "generated_at": datetime.now().astimezone().isoformat(),
        "pinned_previously_cached": sorted(config.PREVIOUSLY_CACHED_VIDEOS),
        "excluded_videos": excluded,   # video_id -> reason (empty when none)
        "splits": {},
    }
    for name, vids in split.items():
        manifest["splits"][name] = [
            {
                "video_id": v,
                "previously_cached": v in config.PREVIOUSLY_CACHED_VIDEOS,
                "subtitle_available": videos[v]["subtitle_available"],
                "provider_indices": videos[v]["provider_indices"],
                "sample_ids": videos[v]["sample_ids"],
                "question_ids": [
                    f"{config.BENCHMARK}/{config.BENCHMARK_SPLIT}/{i}"
                    for i in videos[v]["provider_indices"]
                ],
            }
            for v in vids
        ]

    with open(config.SPLIT_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    for name, entries in manifest["splits"].items():
        n_q = sum(len(e["provider_indices"]) for e in entries)
        n_sub = sum(e["subtitle_available"] for e in entries)
        print(f"{name}: {len(entries)} videos, {n_q} QAs, {n_sub} with subtitles")
    print(f"excluded: {len(excluded)}")
    print(f"manifest -> {config.SPLIT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
