"""Manual prompt control surface for the DVD agent (component 3, DVD side).

Every static prompt DVD used to hard-code now lives in dvd/dvd/prompts.py as a
field of a single `DVDPrompts` registry. This file is where you drive it: read
the defaults, override any field, and inspect the effect — without touching the
agent's internal architecture.

Usage
-----
    from dvd_prompt import get_prompts, set_prompts, reset_prompts, show_prompts

    show_prompts()                       # dump every prompt + its field name
    set_prompts(orchestrator_system=...) # override one or more prompts
    reset_prompts()                      # restore originals

The captioner / inference-model backend swap (Qwen caption VLM + Codex CLI
orchestrator) is wired in a later step; this file owns only the prompt surface.
"""

from __future__ import annotations

import os
import sys

# Make the vendored DVD package importable (dvd/dvd/*).
_DVD_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dvd")
if _DVD_ROOT not in sys.path:
    sys.path.insert(0, _DVD_ROOT)

from dvd.prompts import (  # noqa: E402
    DVDPrompts,
    get_prompts,
    reset_prompts,
    set_prompts,
)

__all__ = [
    "DVDPrompts",
    "get_prompts",
    "set_prompts",
    "reset_prompts",
    "show_prompts",
    "prompt_names",
    "run_dvd",
    "CONTROLLED_PROMPTS",
]

# The subset of prompts exposed as direct arguments of run_dvd().
CONTROLLED_PROMPTS = [
    "orchestrator_system",
    "orchestrator_user_template",
    "caption_prompt",
    "merge_prompt",
    "global_browse_user_template",
    "frame_inspect_desc",
    "clip_search_desc",
    "global_browse_desc",
]


def prompt_names() -> list[str]:
    """Names of every overridable prompt field."""
    return list(get_prompts().as_dict().keys())


def show_prompts(*, width: int = 100) -> None:
    """Print every prompt field name and its current value."""
    prompts = get_prompts().as_dict()
    for name, value in prompts.items():
        print("=" * width)
        print(f"# {name}")
        print("-" * width)
        print(value)
    print("=" * width)
    print(f"{len(prompts)} overridable prompts.")


# DVD's benchmark harness (reproduce/run_benchmark.py) appends this to the user
# message for single-choice QA so the agent answers with a bare option letter.
# The core orchestrator prompt stays generic; this is added only for MCQ samples.
DEFAULT_MCQ_INSTRUCTION = (
    "\nSelect the best option that accurately addresses the question.\n"
    "Answer with the option's letter from the given choices directly and only "
    "give the best option."
)


def _format_question(sample: dict, mcq_instruction: str | None = None) -> str:
    """Build a plain-text question (MCQ options + answer-format instruction)."""
    question = sample["question"]
    options = sample.get("options")
    if options:
        question = question + "\n" + "\n".join(options)
        if mcq_instruction:
            question = question + mcq_instruction
    return question


def run_dvd(
    sample: dict,
    *,
    orchestrator_system: str | None = None,
    orchestrator_user_template: str | None = None,
    caption_prompt: str | None = None,
    merge_prompt: str | None = None,
    global_browse_user_template: str | None = None,
    frame_inspect_desc: str | None = None,
    clip_search_desc: str | None = None,
    global_browse_desc: str | None = None,
    inference_model: str = "gpt-5.5",
    tool_calling_model: str = "gpt-4o-mini",
    use_openai_tools: bool = True,
    text_backend: str = "openai",
    use_transcript: bool = True,
    mcq_instruction: str | None = DEFAULT_MCQ_INSTRUCTION,
    sample_fps: float = 1.0,
    max_iterations: int = 15,
    clip_secs: int | None = None,
    tool_vlm_max_frames: int = 16,
    work_root: str | None = None,
    gpu: str | None = None,
) -> dict:
    """Run the DVD agent on one dataset sample with overridable prompts.

    `sample` is a provider sample dict (needs video_path, question, optional
    options, and extra.videoID for the frame cache). Any of the eight prompt
    arguments left as None keeps DVD's original text. Vision runs on Qwen, text
    reasoning uses OpenAI API by default (`text_backend="openai"`) or the
    legacy Codex CLI path (`text_backend="codex"`), embeddings run on local BGE.

    Returns {"answer", "gold", "question", "messages", "captions_path"}.
    """
    import os as _os

    # `gpu` may name one or several GPUs ("1" or "0,1,2"). Multiple GPUs are
    # used data-parallel for captioning (one Qwen worker per GPU); the
    # frame_inspect engine itself always runs on a single GPU (TP=1), since
    # Qwen-7B's 28 heads don't divide by 3.
    gpus = None
    if gpu is not None:
        _os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        gpus = [g.strip() for g in gpu.split(",") if g.strip()]

    # 1. Apply prompt overrides (reset first so a prior run doesn't leak).
    reset_prompts()
    overrides = {
        "orchestrator_system": orchestrator_system,
        "orchestrator_user_template": orchestrator_user_template,
        "caption_prompt": caption_prompt,
        "merge_prompt": merge_prompt,
        "global_browse_user_template": global_browse_user_template,
        "frame_inspect_desc": frame_inspect_desc,
        "clip_search_desc": clip_search_desc,
        "global_browse_desc": global_browse_desc,
    }
    set_prompts(**{k: v for k, v in overrides.items() if v is not None})

    # 2. Swap DVD's backend to configured text + Qwen + BGE.
    import dvd.config as config
    from dvd.dvd_core import DVDCoreAgent
    from dvd.utils import extract_answer
    from dvd_backend import install_backend
    from dvd_captioning import build_captions_json, decode_frames_at_fps

    config.LITE_MODE = False  # FULL mode: Qwen captions + frame_inspect enabled
    from dvd_backend import get_captioner
    install_backend(
        inference_model,
        tool_vlm_max_frames=tool_vlm_max_frames,
        tool_calling_model=tool_calling_model,
        use_openai_tools=use_openai_tools,
        text_backend=text_backend,
        tensor_parallel_size=1,  # frame_inspect engine is single-GPU
    )

    # 3. Working dir + uniform frame sampling. Decode the video at sample_fps
    #    (the frame cache is non-uniform: ~2fps short, ~0.2fps long), naming
    #    frame_n{i}.jpg so frame i is at t = i/sample_fps. fps is baked into the
    #    cache paths so a different rate never reuses stale captions.
    video_id = sample.get("extra", {}).get("videoID") or sample["sample_id"]
    subtitle_path = sample.get("extra", {}).get("subtitle_path") if use_transcript else None

    work_root = work_root or _os.path.join(_DVD_ROOT, "run_workspace")
    workdir = _os.path.join(work_root, video_id)
    fps_tag = f"fps{sample_fps:g}"
    frames_dir = _os.path.join(workdir, f"frames_{fps_tag}")
    decode_frames_at_fps(sample["video_path"], frames_dir, sample_fps)
    # DVD derives frames from <video_file_root>/frames; point it at this dir.
    frames_link = _os.path.join(workdir, "frames")
    if _os.path.islink(frames_link) or _os.path.exists(frames_link):
        if _os.path.islink(frames_link):
            _os.unlink(frames_link)
    if not _os.path.exists(frames_link):
        _os.symlink(frames_dir, frames_link)

    # Captions depend on the caption-stage prompts too, so hash them into the
    # cache path. Variants that change caption_prompt/merge_prompt re-caption;
    # variants that only change orchestrator/tool prompts reuse captions.
    import hashlib as _hashlib
    _p = get_prompts()
    cap_key = f"{_p.caption_prompt}||{_p.merge_prompt}"
    cap_hash = _hashlib.md5(cap_key.encode()).hexdigest()[:8]
    mode = "tx" if subtitle_path else "notx"
    tag = f"{mode}_{fps_tag}_{cap_hash}"
    captions_dir = _os.path.join(workdir, f"captions_{tag}")
    _os.makedirs(captions_dir, exist_ok=True)

    # 4. Caption clips (Qwen, data-parallel across `gpus`), then set the
    #    effective fps so frame_inspect maps timestamps onto frame indices.
    #    Build the frame_inspect engine after workers exit, before codex merge.
    frames_per_clip = max(1, int(round(sample_fps * (clip_secs or config.CLIP_SECS))))
    captions_path, effective_fps = build_captions_json(
        sample["video_path"], frames_dir, captions_dir, clip_secs=clip_secs,
        subtitle_path=subtitle_path, gpus=gpus, max_frames_per_clip=frames_per_clip,
        max_images=tool_vlm_max_frames, before_merge=get_captioner,
    )
    config.VIDEO_FPS = effective_fps

    # 5. Build the vector DB (BGE) + run the orchestrator (codex).
    video_db_path = _os.path.join(workdir, f"database_{tag}.json")
    question = _format_question(sample, mcq_instruction)
    agent = DVDCoreAgent(video_db_path, captions_path, max_iterations)
    messages = agent.run(question)

    answer = extract_answer(messages[-1]) if messages else None
    return {
        "answer": answer,
        "gold": sample.get("answer"),
        "question": question,
        "messages": messages,
        "captions_path": captions_path,
    }


if __name__ == "__main__":
    show_prompts()
