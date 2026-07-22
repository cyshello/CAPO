"""Phase 2 sanity check (PHASE2_3 §13): full vs selective rollouts on a fixed
QA and candidate-prompt set — no optimizer.

For every QA (train split), every frozen candidate prompt, and every policy:

    full_rollout                    ground truth
    trace_only / question_clip / trace_plus_question_clip /
    prompt_delta_clip / trace_plus_prompt_delta_clip
    random_budget_matched           (budget-matched to trace_plus_prompt_delta_clip)

Stage 0 evaluates the BASELINE prompt through FullRolloutEvaluator, which both
regression-checks the captured baseline and produces the frozen baseline
caption snapshot + trajectory the selective policies consume.

Usage:
    conda run -n local_llm_vllm python -m surrogate_rollout.scripts.run_phase2_sanity_check \
        --indices 9,10,11 --gpu 2 \
        --candidate-names "Perceptual only,Entity centric,Dense exhaustive" \
        --work-root runs/phase2_sanity

Results append to <work_root>/results.jsonl (resumable: finished combinations
are skipped) and aggregate to <work_root>/summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from surrogate_rollout import config

for _p in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_CANDIDATES_FILE = os.path.join(
    config.PROMPT_SENS_ROOT, "prompts", "DVD", "captioner_drift", "prompts.json")

SELECTIVE_POLICIES = (
    "trace_only",
    "question_clip",
    "trace_plus_question_clip",
    "prompt_delta_clip",
    "trace_plus_prompt_delta_clip",
    "random_budget_matched",
)


def load_candidates(path: str, names: list[str] | None) -> dict[str, str]:
    with open(path) as f:
        data = json.load(f)
    prompts = data["caption_prompt"] if "caption_prompt" in data else data
    if names:
        missing = [n for n in names if n not in prompts]
        if missing:
            raise SystemExit(f"candidate names not in {path}: {missing}")
        return {n: prompts[n] for n in names}
    return dict(prompts)


def result_key(r: dict) -> tuple:
    return (r["qa_id"], r["candidate_name"], r["rollout_mode"], r["selection_policy"])


def load_done(results_path: str) -> dict[tuple, dict]:
    done: dict[tuple, dict] = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[result_key(r)] = r
    return done


def append_result(results_path: str, record: dict) -> None:
    with open(results_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def summarize(records: list[dict]) -> dict:
    """Aggregate metrics per selection policy against full rollout."""
    full = {(r["qa_id"], r["candidate_name"]): r for r in records
            if r["rollout_mode"] == "full" and r["candidate_name"] != "__baseline__"}
    out: dict = {"full_rollout": {
        "n": len(full),
        "qa_accuracy": (sum(r["score"] for r in full.values()) / len(full)) if full else None,
        "mean_latency": (sum(r["latency_seconds"] for r in full.values()) / len(full))
        if full else None,
    }}
    for policy in SELECTIVE_POLICIES:
        sel = [r for r in records
               if r["selection_policy"] == policy and r["rollout_mode"] == "selective"]
        pairs = [(r, full.get((r["qa_id"], r["candidate_name"]))) for r in sel]
        pairs = [(s, f) for s, f in pairs if f is not None]
        if not pairs:
            continue
        n = len(pairs)
        full_lat = sum(f["latency_seconds"] for _, f in pairs)
        sel_lat = sum(s["latency_seconds"] for s, _ in pairs)
        full_calls = sum(f["caption_call_count"] + f["caption_cache_hits"] for _, f in pairs)
        sel_calls = sum(s["caption_call_count"] for s, _ in pairs)
        out[policy] = {
            "n": n,
            "answer_agreement": sum(
                (s["parsed_answer"] == f["parsed_answer"]) for s, f in pairs) / n,
            "correctness_agreement": sum(
                (s["is_correct"] == f["is_correct"]) for s, f in pairs) / n,
            "selective_qa_accuracy": sum(s["score"] for s, _ in pairs) / n,
            "full_qa_accuracy": sum(f["score"] for _, f in pairs) / n,
            "mean_recaption_fraction": sum(s["recaption_fraction"] for s, _ in pairs) / n,
            "caption_call_reduction": (1 - sel_calls / full_calls) if full_calls else None,
            "latency_reduction": (1 - sel_lat / full_lat) if full_lat else None,
            "fallback_counts": {
                ft: sum(1 for s, _ in pairs if s["fallback_type"] == ft)
                for ft in sorted({s["fallback_type"] for s, _ in pairs})
            },
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", required=True,
                    help="comma-separated provider indices (train split QAs)")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--work-root", default=os.path.join(
        config.RUNS_ROOT, f"phase2_sanity_{time.strftime('%Y%m%d_%H%M%S')}"))
    ap.add_argument("--candidates-file", default=DEFAULT_CANDIDATES_FILE)
    ap.add_argument("--candidate-names", default=None,
                    help="comma-separated candidate prompt names (default: all)")
    ap.add_argument("--policies", default=",".join(SELECTIVE_POLICIES))
    ap.add_argument("--reference-policy", default=config.DEFAULT_TRACE_POLICY_PHASE2)
    ap.add_argument("--max-recaption-fraction", type=float, default=0.5)
    ap.add_argument("--retrieval-top-k", type=int, default=config.RETRIEVAL_TOP_K)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from data_provider import get_provider
    from dvd_prompt import get_prompts

    from surrogate_rollout.evaluation.dvd_qa import ensure_backend
    from surrogate_rollout.evaluation.full_rollout import FullRolloutEvaluator, qa_slug
    from surrogate_rollout.evaluation.rollout_evaluator import (
        EvaluationRequest,
        SelectionBudget,
    )
    from surrogate_rollout.evaluation.selective_rollout import (
        SelectiveSurrogateRolloutEvaluator,
    )

    os.makedirs(args.work_root, exist_ok=True)
    results_path = os.path.join(args.work_root, "results.jsonl")
    done = load_done(results_path)

    # preload: the vLLM engine must exist before BGE/DB threads (fork-safety)
    ensure_backend(args.gpu, preload_captioner=True)
    provider = get_provider(config.BENCHMARK, split=config.BENCHMARK_SPLIT)
    indices = [int(i) for i in args.indices.split(",") if i.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = set(policies) - set(SELECTIVE_POLICIES)
    if unknown:
        raise SystemExit(f"unknown policies: {sorted(unknown)}")

    baseline_prompt = get_prompts().caption_prompt
    candidates = load_candidates(
        args.candidates_file,
        [n.strip() for n in args.candidate_names.split(",")] if args.candidate_names else None)
    budget = SelectionBudget(max_recaption_fraction=args.max_recaption_fraction)

    with open(os.path.join(args.work_root, "run_config.json"), "w") as f:
        json.dump({
            "indices": indices, "policies": policies,
            "reference_policy": args.reference_policy,
            "candidates_file": args.candidates_file,
            "candidate_names": sorted(candidates),
            "budget": vars(args), "seed": args.seed,
        }, f, indent=2, default=str)

    full_eval = FullRolloutEvaluator()
    # text encoder on CPU: the vLLM engine fills the whole GPU; query embedding
    # is a handful of short strings (image embeddings are precomputed)
    from surrogate_rollout.retrieval.visual_index import SiglipEmbedder

    sel_eval = SelectiveSurrogateRolloutEvaluator(
        embedder=SiglipEmbedder(device="cpu"),
        retrieval_top_k=args.retrieval_top_k)

    def make_request(sample, idx, candidate_prompt, policy,
                     baseline_captions, baseline_traj_dir, matched=None):
        video_id = sample.get("extra", {}).get("videoID") or sample["sample_id"]
        return EvaluationRequest(
            video_id=video_id,
            qa_id=f"{config.BENCHMARK}/{config.BENCHMARK_SPLIT}/{idx}",
            question=sample["question"],
            answer_options=tuple(sample.get("options") or ()),
            ground_truth=sample.get("answer"),
            sample=sample,
            baseline_prompt=baseline_prompt,
            candidate_prompt=candidate_prompt,
            baseline_captions_path=baseline_captions,
            baseline_trajectory_dir=baseline_traj_dir,
            reference_policy=args.reference_policy,
            selection_policy=policy,
            work_root=args.work_root,
            budget=budget,
            gpu=args.gpu,
            budget_matched_count=matched,
            random_seed=args.seed,
        )

    def record(res, candidate_name):
        rec = {"candidate_name": candidate_name, **res.as_json()}
        if result_key(rec) not in done:
            append_result(results_path, rec)
            done[result_key(rec)] = rec
        print(f"[{rec['qa_id']}] {candidate_name} {rec['rollout_mode']}/"
              f"{rec['selection_policy']}: score={rec['score']} "
              f"recaption={rec['recaption_fraction']:.2f} "
              f"fallback={rec['fallback_type']}", flush=True)
        return rec

    # ------------------- stage 0: baseline full rollouts -------------------- #
    baseline_info: dict[int, tuple[str, str]] = {}  # idx -> (captions, traj dir)
    for idx in indices:
        sample = provider[idx]
        qa_id = f"{config.BENCHMARK}/{config.BENCHMARK_SPLIT}/{idx}"
        req = make_request(sample, idx, baseline_prompt, "full_rollout", "", None)
        key = (qa_id, "__baseline__", "full", "full_rollout")
        if key in done:
            rec = done[key]
        else:
            rec = record(full_eval.evaluate(req), "__baseline__")
        baseline_info[idx] = (
            rec["captions_path"],
            os.path.dirname(rec["trajectory_path"]) if rec.get("trajectory_path") else None,
        )

    # ------------------- stage 1: candidate evaluations --------------------- #
    for cname, cprompt in candidates.items():
        for idx in indices:
            sample = provider[idx]
            qa_id = f"{config.BENCHMARK}/{config.BENCHMARK_SPLIT}/{idx}"
            baseline_captions, baseline_traj = baseline_info[idx]

            if (qa_id, cname, "full", "full_rollout") not in done:
                record(full_eval.evaluate(make_request(
                    sample, idx, cprompt, "full_rollout",
                    baseline_captions, baseline_traj)), cname)

            matched = None
            for policy in policies:
                if policy == "random_budget_matched":
                    ref = done.get((qa_id, cname, "selective",
                                    "trace_plus_prompt_delta_clip"))
                    matched = len(ref["selected_clip_ids"]) if ref else None
                key = (qa_id, cname, "selective", policy)
                if key in done:
                    continue
                record(sel_eval.evaluate(make_request(
                    sample, idx, cprompt, policy,
                    baseline_captions, baseline_traj, matched)), cname)

    summary = summarize(list(done.values()))
    with open(os.path.join(args.work_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
