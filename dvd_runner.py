"""Instrumented single-QA DVD runner producing a DVDRunResult.

Wraps prompt_sensitivity's run_dvd without changing its behavior: the
instrumentation only records (tool sidecar, LLM call log); the agent, prompts,
models, caches, and answer are exactly what an uninstrumented run produces.

Every run writes a self-contained artifact directory:
    result.json          DVDRunResult (references as sorted lists)
    trajectory.jsonl     complete messages
    tool_events.jsonl    sidecar events with vector-DB hits
    llm_calls.jsonl      per-call route/model/latency (+usage when exposed)
    references.json      extracted reference sets + evidence
"""

from __future__ import annotations

import json
import os
import sys
import time

from surrogate_rollout import config
from surrogate_rollout.evaluation.qa_metrics import score_mcq
from surrogate_rollout.schemas import DVDRunResult, ReferenceSets

for _p in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_clip_keys(captions_path: str | None) -> list[str]:
    if not captions_path or not os.path.exists(captions_path):
        return []
    with open(captions_path) as f:
        captions = json.load(f)
    return [k for k in captions
            if k not in ("subject_registry", "character_registry")]


def run_qa_instrumented(
    sample: dict,
    *,
    run_dir: str,
    gpu: str | None = None,
    prompt_overrides: dict | None = None,
    tool_calling_model: str = config.ORCHESTRATOR_TOOL_MODEL,
    use_openai_tools: bool = True,
    question_id: str | None = None,
) -> DVDRunResult:
    """Run one QA through the unchanged DVD pipeline with recording installed.

    `sample` is a provider sample dict. Failures inside DVD are captured as a
    machine-readable error record and a 0.0 score — never re-raised silently,
    never retried with modified inputs (CLAUDE.md §24).
    """
    from dvd_prompt import run_dvd

    from surrogate_rollout import instrumentation
    from surrogate_rollout.references.extractor import extract_references

    os.makedirs(run_dir, exist_ok=True)
    video_id = sample.get("extra", {}).get("videoID") or sample["sample_id"]
    qid = question_id or f"{config.BENCHMARK}/{config.BENCHMARK_SPLIT}/{sample.get('sample_id')}"

    recorder = instrumentation.install()
    t0 = time.time()
    out: dict | None = None
    errors: list[dict] = []
    try:
        out = run_dvd(
            sample,
            gpu=gpu,
            tool_calling_model=tool_calling_model,
            use_openai_tools=use_openai_tools,
            **(prompt_overrides or {}),
        )
    except Exception as e:
        import traceback

        errors.append({
            "stage": "run_dvd",
            "type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
    finally:
        latency = time.time() - t0
        recorder.uninstall()

    messages = (out or {}).get("messages") or []
    captions_path = (out or {}).get("captions_path")
    clip_keys = _load_clip_keys(captions_path)

    try:
        refs = extract_references(messages, recorder.tool_events, clip_keys)
    except Exception as e:
        refs = ReferenceSets()
        errors.append({"stage": "reference_extraction",
                       "type": type(e).__name__, "error": str(e)})

    raw_answer = (out or {}).get("answer")
    gold = sample.get("answer")
    score, parsed, failure_kind = score_mcq(raw_answer, gold)
    if failure_kind == "parse_failure":
        errors.append({"stage": "answer_parsing", "type": "ParseFailure",
                       "error": f"no option letter in {raw_answer!r}"})

    cache_tag = os.path.basename(os.path.dirname(captions_path)) if captions_path else ""
    db_path = None
    if captions_path:
        workdir = os.path.dirname(os.path.dirname(captions_path))
        db_path = os.path.join(
            workdir, f"database_{cache_tag.removeprefix('captions_')}.json")

    result = DVDRunResult(
        question_id=qid,
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
        caption_cache_tag=cache_tag,
        captions_path=captions_path,
        database_path=db_path if db_path and os.path.exists(db_path) else db_path,
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
    result_json.pop("trajectory")  # stored separately as jsonl
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result_json, f, indent=2, default=str)
    return result
