"""Phase 0: run ONE unchanged DVD baseline QA and save a complete artifact dir.

Does not modify DVD behavior in any way: calls prompt_sensitivity's run_dvd
with default prompts and the same defaults run_video_qas.py uses. Everything
observable (trajectory, answer, prompt texts/hashes, cache paths, timing,
environment, recaption activity) is captured around the call.

Usage:
    conda run -n local_llm_vllm python -m surrogate_rollout.scripts.inspect_dvd_baseline \
        --index 9 --gpu 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROMPT_SENS = os.path.abspath(os.path.join(_HERE, "..", "..", "prompt_sensitivity"))
for p in (_PROMPT_SENS, os.path.join(_PROMPT_SENS, "dvd")):
    if p not in sys.path:
        sys.path.insert(0, p)


def extract_letter(answer: str | None) -> str | None:
    """Same answer parser as prompt_sensitivity/run_video_qas.py."""
    if not answer:
        return None
    m = re.search(r"[A-E]", answer.strip())
    return m.group(0) if m else None


def _snapshot_ckpt(captions_dir: str) -> dict:
    """Count + latest mtime of per-clip caption checkpoints (recaption probe)."""
    ckpt = os.path.join(captions_dir, "ckpt")
    if not os.path.isdir(ckpt):
        return {"exists": False, "n_files": 0, "max_mtime": 0.0}
    files = [os.path.join(ckpt, f) for f in os.listdir(ckpt) if f.endswith(".json")]
    return {
        "exists": True,
        "n_files": len(files),
        "max_mtime": max((os.path.getmtime(f) for f in files), default=0.0),
    }


def _environment_info() -> dict:
    def _cmd(args):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # pragma: no cover
            return f"error: {e}"

    import importlib.metadata as md

    pkgs = {}
    for name in ("gepa", "vllm", "transformers", "torch", "openai", "sentence-transformers", "nano-vectordb"):
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    return {
        "hostname": platform.node(),
        "python": sys.version,
        "executable": sys.executable,
        "packages": pkgs,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _cmd(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv"]),
        "dvd_git_commit": _cmd(["git", "-C", os.path.join(_PROMPT_SENS, "dvd"), "rev-parse", "HEAD"]),
        "timestamp": datetime.now().astimezone().isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="long")
    ap.add_argument("--index", type=int, default=9,
                    help="videomme sample index (default 9 = 0RxMZBLeqRI, cached)")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--out-root", default=os.path.join(_HERE, "..", "runs"))
    args = ap.parse_args()

    from data_provider import get_provider
    from dvd_prompt import run_dvd, get_prompts, reset_prompts

    provider = get_provider("videomme", split=args.split)
    sample = provider[args.index]
    video_id = sample.get("extra", {}).get("videoID") or sample["sample_id"]

    reset_prompts()
    prompts = get_prompts().as_dict()
    cap_hash = hashlib.md5(
        f"{prompts['caption_prompt']}||{prompts['merge_prompt']}".encode()
    ).hexdigest()[:8]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(os.path.join(args.out_root, f"phase0_baseline_{run_id}"))
    os.makedirs(out_dir, exist_ok=True)
    prompts_dir = os.path.join(out_dir, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)

    # Expected cache location (run_dvd derives the same tag internally).
    workdir = os.path.join(_PROMPT_SENS, "dvd", "run_workspace", video_id)
    subtitle_path = sample.get("extra", {}).get("subtitle_path")
    tag = f"{'tx' if subtitle_path else 'notx'}_fps1_{cap_hash}"
    captions_dir = os.path.join(workdir, f"captions_{tag}")
    db_path = os.path.join(workdir, f"database_{tag}.json")

    ckpt_before = _snapshot_ckpt(captions_dir)
    db_mtime_before = os.path.getmtime(db_path) if os.path.exists(db_path) else None

    with open(os.path.join(out_dir, "environment.json"), "w") as f:
        json.dump(_environment_info(), f, indent=2)
    for name, text in prompts.items():
        with open(os.path.join(prompts_dir, f"{name}.txt"), "w") as f:
            f.write(text)

    t0 = time.time()
    error = None
    out = None
    try:
        out = run_dvd(sample, gpu=args.gpu, tool_calling_model=args.model,
                      use_openai_tools=True)
    except Exception as e:  # keep the artifact even on failure
        import traceback
        error = {"error": str(e), "traceback": traceback.format_exc()}
    wall_seconds = time.time() - t0

    ckpt_after = _snapshot_ckpt(captions_dir)
    recaptioned = (
        ckpt_after["n_files"] != ckpt_before["n_files"]
        or ckpt_after["max_mtime"] > ckpt_before["max_mtime"]
    )

    messages = (out or {}).get("messages")
    answer = (out or {}).get("answer")
    pred = extract_letter(answer)
    gold = sample.get("answer")

    tool_calls = []
    for i, m in enumerate(messages or []):
        for tc in (m.get("tool_calls") or []):
            tool_calls.append({"message_index": i, "id": tc.get("id"),
                               "function": tc.get("function")})
        if m.get("role") == "tool":
            tool_calls.append({"message_index": i, "tool_result_for": m.get("tool_call_id"),
                               "name": m.get("name"),
                               "content_chars": len(m.get("content") or "")})

    with open(os.path.join(out_dir, "trajectory.jsonl"), "w") as f:
        for m in messages or []:
            f.write(json.dumps(m, default=str) + "\n")
    with open(os.path.join(out_dir, "tool_calls.json"), "w") as f:
        json.dump(tool_calls, f, indent=2, default=str)

    summary = {
        "run_id": run_id,
        "phase": "phase0_baseline",
        "dataset": {"benchmark": "videomme", "split": args.split, "index": args.index},
        "video_id": video_id,
        "question_full": (out or {}).get("question"),
        "question_raw": sample["question"],
        "options": sample.get("options"),
        "gold": gold,
        "raw_answer": answer,
        "parsed_answer": pred,
        "score": float(pred is not None and pred == gold),
        "error": error,
        "prompt_hashes": {
            "cap_hash_tag": cap_hash,
            **{k: hashlib.md5(v.encode()).hexdigest() for k, v in prompts.items()},
        },
        "models": {
            "orchestrator_tool_calling": args.model,
            "captioner": "Qwen2.5-VL (vLLM, local)",
            "embeddings": "BAAI/bge-small-en-v1.5 (local)",
            "text_fallback": "codex CLI gpt-5.5",
        },
        "paths": {
            "video_path": sample["video_path"],
            "subtitle_path": subtitle_path,
            "captions_path": (out or {}).get("captions_path"),
            "expected_captions_dir": captions_dir,
            "database_path": db_path,
            "workdir": workdir,
        },
        "cache": {
            "ckpt_before": ckpt_before,
            "ckpt_after": ckpt_after,
            "recaptioning_occurred": recaptioned,
            "db_existed_before": db_mtime_before is not None,
        },
        "timing": {"wall_seconds": wall_seconds},
        "counts": {
            "n_messages": len(messages or []),
            "n_tool_call_records": len(tool_calls),
        },
        "notes": "Baseline run through unchanged run_dvd; no DVD code modified. "
                 "Token usage not captured (run_dvd does not expose it; Phase 1 adds it).",
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"ARTIFACT_DIR {out_dir}")
    print(json.dumps({k: summary[k] for k in
                      ("video_id", "gold", "parsed_answer", "score", "timing", "error")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
