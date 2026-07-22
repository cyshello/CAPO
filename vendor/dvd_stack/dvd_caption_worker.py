"""Data-parallel captioning worker: one GPU, one shard of clips.

Run as a subprocess by dvd_captioning._caption_shards_dp. Each worker pins a
single GPU, builds its own Qwen2.5-VL engine, captions its shard in one batch,
writes per-clip JSON to the shared ckpt dir, then exits (freeing the GPU).

Prompts are pre-rendered by the parent (so prompt overrides apply); the worker
needs no access to the DVD prompt registry. Invocation:

    python dvd_caption_worker.py <tasks_json> <ckpt_dir> <gpu> <max_tokens> <max_images>

<tasks_json> is a list of {"key", "files", "prompt", "transcript"}.
"""

from __future__ import annotations

import json
import os
import re
import sys


def _strip_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_json(text: str) -> dict:
    t = _strip_fences(text)
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError("unbalanced JSON")


def main() -> None:
    tasks_path, ckpt_dir, gpu, max_tokens, max_images = sys.argv[1:6]
    # Pin the GPU before importing torch/vllm.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    max_tokens = int(max_tokens)
    max_images = int(max_images)

    with open(tasks_path) as f:
        tasks = json.load(f)
    if not tasks:
        return

    from captioner import build_captioner

    # Footprint must match dvd_backend.get_captioner().
    captioner = build_captioner(
        max_model_len=12288,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        max_images_per_prompt=max_images,
        image_max_pixels=200704,
    )
    outputs = captioner.caption_batch(
        [t["files"] for t in tasks],
        [t["prompt"] for t in tasks],
        max_tokens=max_tokens,
    )
    for task, raw in zip(tasks, outputs):
        try:
            parsed = _extract_json(raw)
            parsed["clip_description"] = (
                parsed.get("clip_description", "")
                + f"\n\nTranscript during this video clip: {task['transcript']}."
            )
        except Exception:
            parsed = {}  # deterministic parse failure: cache empty so we don't
            # re-caption this clip on every run (temperature 0 -> same output)
        with open(os.path.join(ckpt_dir, f"{task['key']}.json"), "w") as f:
            json.dump(parsed, f)


if __name__ == "__main__":
    main()
