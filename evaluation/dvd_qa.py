"""The single shared DVD reasoning path for Phase 2 evaluators.

Both FullRolloutEvaluator and SelectiveSurrogateRolloutEvaluator run QA through
`run_dvd_qa` over an already-materialized captions.json view — the only thing
that differs between modes is how that view was produced (CLAUDE.md §8/§10:
never duplicate the agent loop per mode).

This reproduces steps 3-5 of prompt_sensitivity.dvd_prompt.run_dvd (frames
link, VIDEO_FPS, vector DB, DVDCoreAgent) without the captioning step, and with
the vector DB keyed by the exact captions.json content hash (PHASE2_3 §2:
init_single_video_db reuses an existing DB file blindly, so the path must be
unique per caption content). Prompt overrides are reset first: candidate
caption prompts influence captions only, never the reasoning-side prompts.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from contextlib import contextmanager

from surrogate_rollout import config
from surrogate_rollout.cache.caption_cache import captions_content_hash
from surrogate_rollout.evaluation.qa_metrics import score_mcq
from surrogate_rollout.schemas import DVDRunResult, ReferenceSets

for _p in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_BACKEND_INSTALLED = False
_DVD_EMBEDDER_PRELOAD_LOCK = threading.Lock()
_DVD_EMBEDDER_PRELOADED = False
DVD_EMBEDDER_PRELOAD_POLICY_VERSION = "dvd_bge_parent_preload_v1"
_DVD_QA_EXECUTION_LOCK = threading.Lock()
DVD_QA_CONCURRENCY_POLICY_VERSION = "serialized_dvd_qa_execution_v1"


@contextmanager
def serialized_dvd_qa_execution():
    """Isolate DVD's mutable FPS, prompt, and instrumentation globals."""
    with _DVD_QA_EXECUTION_LOCK:
        yield


def preload_dvd_embedder() -> None:
    """Build DVD's shared BGE model once before parallel QA threads start.

    DVD's lazy loader is not synchronized. Concurrent first-use from evidence
    video threads can therefore observe a partially initialized
    SentenceTransformer. Keep the synchronization at our execution boundary
    without modifying the external DVD checkout.
    """
    global _DVD_EMBEDDER_PRELOADED
    if _DVD_EMBEDDER_PRELOADED:
        return
    with _DVD_EMBEDDER_PRELOAD_LOCK:
        if _DVD_EMBEDDER_PRELOADED:
            return
        import dvd_backend

        embedder = dvd_backend._get_embedder()
        if embedder is None:
            raise RuntimeError("DVD BGE preload returned no embedder")
        _DVD_EMBEDDER_PRELOADED = True


def dvd_qa_execution_identity(max_iterations: int) -> dict[str, object]:
    return {
        "policy_version": "dvd_qa_execution_v4_strict_frame_inspect",
        "dvd_max_iterations": int(max_iterations),
        "clip_search_top_k": config.DVD_CLIP_SEARCH_TOP_K,
        "clip_search_policy_version": config.DVD_CLIP_SEARCH_POLICY_VERSION,
        "embedding_preload_policy_version":
            DVD_EMBEDDER_PRELOAD_POLICY_VERSION,
        "qa_concurrency_policy_version": DVD_QA_CONCURRENCY_POLICY_VERSION,
        "frame_inspect_tool_contract_version":
            config.DVD_FRAME_INSPECT_TOOL_CONTRACT_VERSION,
        "frame_inspect_corrective_retry_limit":
            config.DVD_FRAME_INSPECT_CORRECTIVE_RETRY_LIMIT,
    }


def _constrain_clip_search_schema(agent: object, top_k: int) -> None:
    """Tell the tool-calling model the same value enforced at execution."""
    schemas = getattr(agent, "function_schemas", ())
    for schema in schemas:
        function = schema.get("function") or {}
        if function.get("name") != "clip_search_tool":
            continue
        properties = (function.get("parameters") or {}).get("properties") or {}
        field = properties.get("top_k")
        if not isinstance(field, dict):
            raise RuntimeError("DVD clip_search_tool schema is missing top_k")
        field["enum"] = [top_k]
        field["default"] = top_k
        field["description"] = (
            f"Fixed by the execution policy. Always use {top_k}.")
        return
    raise RuntimeError("DVD clip_search_tool schema is missing")


def _constrain_frame_inspect_schema(agent: object) -> None:
    """Match the provider tool contract to DVD's string-only implementation."""
    schemas = getattr(agent, "function_schemas", ())
    for schema in schemas:
        function = schema.get("function") or {}
        if function.get("name") != "frame_inspect_tool":
            continue
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        field = properties.get("time_ranges_hhmmss")
        if not isinstance(field, dict):
            raise RuntimeError(
                "DVD frame_inspect_tool schema is missing time ranges")
        field.clear()
        field.update({
            "type": "array", "minItems": 1,
            "description": (
                "One or more [start, end] pairs. Every endpoint must be an "
                "HH:MM:SS string, for example [[\"00:00:10\", "
                "\"00:00:20\"]]. Numeric seconds are invalid."),
            "items": {
                "type": "array", "minItems": 2, "maxItems": 2,
                "items": {
                    "type": "string",
                    "pattern":
                        r"^[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?$",
                },
            },
        })
        parameters["additionalProperties"] = False
        function["strict"] = True
        return
    raise RuntimeError("DVD frame_inspect_tool schema is missing")


def ensure_backend(
    gpu: str | None = None,
    *,
    tool_calling_model: str = config.ORCHESTRATOR_TOOL_MODEL,
    inference_model: str = config.TEXT_FALLBACK_MODEL,
    use_openai_tools: bool = True,
    text_backend: str = "openai",
    preload_captioner: bool = False,
    preload_embedder: bool = False,
) -> None:
    """Install the codex+Qwen+BGE backend once per process (idempotent).

    Must run before any captioning or QA call. With `preload_captioner` the
    vLLM engine is built immediately; otherwise it stays lazy. Long-running
    drivers that will caption at all MUST preload: building the engine after
    BGE/DB work has spawned thread pools deadlocks vLLM's forked EngineCore in
    futex_wait (observed 2026-07-14; legacy run_dvd always built the engine
    first, via before_merge)."""
    global _BACKEND_INSTALLED
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    if not _BACKEND_INSTALLED:
        import dvd.config as dvd_config
        from dvd_backend import install_backend

        dvd_config.LITE_MODE = False
        install_backend(
            inference_model,
            tool_vlm_max_frames=config.TOOL_VLM_MAX_FRAMES,
            tool_calling_model=tool_calling_model,
            use_openai_tools=use_openai_tools,
            text_backend=text_backend,
            tensor_parallel_size=1,
        )
        _BACKEND_INSTALLED = True
    if preload_captioner:
        from dvd_backend import get_captioner

        get_captioner()  # build the vLLM engine before any thread pools exist
    if preload_embedder:
        preload_dvd_embedder()


def resolve_frames_dir(sample: dict, video_id: str) -> str:
    """Decoded-frame directory for the video: reuse the legacy workspace's
    frames_fps{f} (read-only) when present, else decode into the legacy
    workspace layout once. Frames are prompt-independent, so sharing them
    across candidates is safe."""
    fps_tag = f"fps{config.SAMPLE_FPS:g}"
    legacy = os.path.join(config.DVD_RUN_WORKSPACE, video_id, f"frames_{fps_tag}")
    if os.path.isdir(legacy) and os.listdir(legacy):
        return legacy
    from dvd_captioning import decode_frames_at_fps

    decode_frames_at_fps(sample["video_path"], legacy, config.SAMPLE_FPS)
    return legacy


def prepare_video_workdir(work_root: str, video_id: str, sample: dict) -> str:
    """Create work_root/<video_id> with a `frames` symlink so frame_inspect
    (video_file_root = dirname(dirname(captions.json))) finds decoded frames."""
    workdir = os.path.join(work_root, video_id)
    os.makedirs(workdir, exist_ok=True)
    frames_dir = resolve_frames_dir(sample, video_id)
    link = os.path.join(workdir, "frames")
    if os.path.islink(link):
        if os.readlink(link) != frames_dir:
            os.unlink(link)
            os.symlink(frames_dir, link)
    elif not os.path.exists(link):
        os.symlink(frames_dir, link)
    return workdir


def effective_fps_for(sample: dict, video_id: str) -> float:
    from dvd_captioning import video_duration_seconds

    frames_dir = resolve_frames_dir(sample, video_id)
    n = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    return n / video_duration_seconds(sample["video_path"])


def database_path_for(captions_path: str) -> str:
    """Vector-DB path keyed by the exact captions.json content hash."""
    h = captions_content_hash(captions_path)
    workdir = os.path.dirname(os.path.dirname(captions_path))
    return os.path.join(workdir, f"database_c{h[:16]}.json")


def format_question(sample: dict) -> str:
    from dvd_prompt import DEFAULT_MCQ_INSTRUCTION, _format_question

    return _format_question(sample, DEFAULT_MCQ_INSTRUCTION)


def run_dvd_qa(
    *,
    captions_path: str,
    sample: dict,
    run_dir: str,
    question_id: str,
    database_path: str | None = None,
    max_iterations: int = config.MAX_ITERATIONS,
    gpu: str | None = None,
) -> DVDRunResult:
    """Run one QA through the unchanged DVD agent over `captions_path`.

    Writes the same artifact set as dvd_runner.run_qa_instrumented
    (result.json / trajectory.jsonl / tool_events.jsonl / llm_calls.jsonl /
    references.json). Failures land as machine-readable error records with a
    0.0 score (CLAUDE.md §24)."""
    ensure_backend(gpu)
    import dvd.config as dvd_config
    from dvd.dvd_core import DVDCoreAgent
    from dvd.utils import extract_answer
    from dvd_prompt import reset_prompts

    from surrogate_rollout import instrumentation
    from surrogate_rollout.references.extractor import extract_references

    os.makedirs(run_dir, exist_ok=True)
    video_id = sample.get("extra", {}).get("videoID") or sample["sample_id"]
    # frames link must exist next to the captions dir (frame_inspect resolves
    # video_file_root = dirname(dirname(captions_path)))
    workdir = os.path.dirname(os.path.dirname(captions_path))
    link = os.path.join(workdir, "frames")
    if not os.path.exists(link):
        os.makedirs(workdir, exist_ok=True)
        os.symlink(resolve_frames_dir(sample, video_id), link)

    effective_fps = effective_fps_for(sample, video_id)
    db_path = database_path or database_path_for(captions_path)
    question = format_question(sample)

    with open(captions_path) as f:
        captions = json.load(f)
    clip_keys = [k for k in captions
                 if k not in ("subject_registry", "character_registry")]

    messages: list[dict] = []
    errors: list[dict] = []
    with serialized_dvd_qa_execution():
        reset_prompts()  # reasoning prompts and VIDEO_FPS are DVD globals
        dvd_config.VIDEO_FPS = effective_fps
        recorder = instrumentation.install(
            clip_search_top_k=config.DVD_CLIP_SEARCH_TOP_K)
        t0 = time.time()
        try:
            agent = DVDCoreAgent(db_path, captions_path, max_iterations)
            stored_fps = agent.video_db.get_additional_data().get("fps")
            if not isinstance(stored_fps, (int, float)) or not math.isclose(
                    float(stored_fps), effective_fps,
                    rel_tol=1e-9, abs_tol=1e-12):
                raise RuntimeError(
                    "DVD database FPS identity mismatch: "
                    f"expected={effective_fps!r}, stored={stored_fps!r}")
            _constrain_clip_search_schema(agent, config.DVD_CLIP_SEARCH_TOP_K)
            _constrain_frame_inspect_schema(agent)
            messages = agent.run(question)
        except Exception as e:
            import traceback

            errors.append({"stage": "dvd_qa", "type": type(e).__name__,
                           "error": str(e),
                           "traceback": traceback.format_exc()})
        finally:
            latency = time.time() - t0
            recorder.uninstall()

    try:
        refs = extract_references(messages, recorder.tool_events, clip_keys)
    except Exception as e:
        refs = ReferenceSets()
        errors.append({"stage": "reference_extraction",
                       "type": type(e).__name__, "error": str(e)})

    raw_answer = extract_answer(messages[-1]) if messages else None
    gold = sample.get("answer")
    score, parsed, failure_kind = score_mcq(raw_answer, gold)
    if failure_kind == "parse_failure":
        errors.append({"stage": "answer_parsing", "type": "ParseFailure",
                       "error": f"no option letter in {raw_answer!r}"})

    result = DVDRunResult(
        question_id=question_id,
        video_id=video_id,
        prediction=raw_answer,
        parsed_answer=parsed,
        ground_truth=gold,
        score=score,
        trajectory=messages,
        references=refs,
        total_segments=len(clip_keys),
        token_usage=recorder.token_usage_summary(),
        latency_seconds=latency,
        caption_cache_tag=os.path.basename(os.path.dirname(captions_path)),
        captions_path=captions_path,
        database_path=db_path,
        errors=errors,
    )

    recorder.dump(os.path.join(run_dir, "tool_events.jsonl"),
                  os.path.join(run_dir, "llm_calls.jsonl"))
    with open(os.path.join(run_dir, "trajectory.jsonl"), "w") as f:
        for m in messages:
            f.write(json.dumps(m, default=str) + "\n")
    with open(os.path.join(run_dir, "references.json"), "w") as f:
        json.dump(refs.as_json(), f, indent=2)
    result_json = result.as_json()
    result_json.pop("trajectory")
    result_json["qa_execution_identity"] = dvd_qa_execution_identity(
        max_iterations)
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result_json, f, indent=2, default=str)
    return result
