"""Determine whether "EngineCore_DP0 died unexpectedly" is a real premature
engine death or only a shutdown-time message.

Sequence mirrors the DVD run that produced the message:
1. build the in-process Qwen engine (get_captioner, as build_captions_json's
   before_merge does);
2. caption once (proves the engine works);
3. run one codex CLI call (the registry-merge step that preceded the message);
4. wait ~90s (the Phase 0 message appeared ~60s after engine init);
5. caption again — succeeds only if the engine is still alive;
6. call frame_inspect_tool directly on a cached video DB (the real dependent).

Usage:
    conda run -n local_llm_vllm python -m surrogate_rollout.scripts.repro_vllm_engine --gpu 1
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

from surrogate_rollout import config

for _p in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--wait", type=int, default=90)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import dvd.config as dvd_config
    from dvd_backend import install_backend, get_captioner

    dvd_config.LITE_MODE = False
    install_backend(config.TEXT_FALLBACK_MODEL, tensor_parallel_size=1)

    video = "0RxMZBLeqRI"
    workdir = os.path.join(config.DVD_RUN_WORKSPACE, video)
    frames = sorted(glob.glob(os.path.join(workdir, "frames_fps1", "*.jpg")))[:4]
    assert frames, "no cached frames found"

    print("STEP 1: building engine...")
    cap = get_captioner()
    print("STEP 2: caption #1...")
    out1 = cap.caption(frames, "Describe these frames in one sentence.", max_tokens=64)
    print(f"  ok ({len(out1)} chars)")

    print("STEP 3: codex call (registry-merge analogue)...")
    from codex_infer import codex_infer

    t0 = time.time()
    resp = codex_infer("Reply with the single word OK.", model=config.TEXT_FALLBACK_MODEL)
    print(f"  codex replied {resp!r} in {time.time() - t0:.0f}s")

    print(f"STEP 4: waiting {args.wait}s...")
    time.sleep(args.wait)

    print("STEP 5: caption #2 (fails if engine died)...")
    try:
        out2 = cap.caption(frames, "Describe these frames in one sentence.", max_tokens=64)
        print(f"  ok ({len(out2)} chars) — ENGINE ALIVE after codex + wait")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        print("  ENGINE DIED — premature failure confirmed")
        return

    print("STEP 6: frame_inspect_tool on cached DB...")
    import dvd.utils as utils
    from dvd.build_database import frame_inspect_tool, init_single_video_db
    from dvd_backend import _local_get_embeddings

    utils.AzureOpenAIEmbeddingService.get_embeddings = staticmethod(_local_get_embeddings)
    dvd_config.AOAI_EMBEDDING_LARGE_DIM = 384
    dvd_config.VIDEO_FPS = 1.0
    db = init_single_video_db(
        os.path.join(workdir, "captions_tx_fps1_78d8e75e", "captions.json"),
        os.path.join(workdir, "database_tx_fps1_78d8e75e.json"),
        384,
    )
    ans = frame_inspect_tool(
        database=db,
        question="What is shown in this part of the video?",
        time_ranges_hhmmss=[("00:00:10", "00:00:40")],
    )
    print(f"  frame_inspect ok ({len(ans)} chars): {ans[:120]!r}")
    print("ALL STEPS PASSED — engine survives codex + idle; "
          "Phase 0 message was not reproduced here")


if __name__ == "__main__":
    main()
