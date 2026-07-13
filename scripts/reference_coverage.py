"""Reference-coverage diagnostic (Phase 1 deliverable).

For every instrumented run and every configured reference policy, report how
many segments the policy selects, as raw count and as fraction of the video's
segments, before and after temporal-neighbor expansion. This is the evidence
for choosing the surrogate reference policy before Phase 2.

Usage:
    python -m surrogate_rollout.scripts.reference_coverage \
        --runs-root runs/phase1_trajectories \
        --out runs/phase1_trajectories/reference_coverage.json
"""

from __future__ import annotations

import argparse
import json
import os

from surrogate_rollout import config
from surrogate_rollout.references.expansion import expand_references


def coverage_for_run(run_dir: str) -> dict | None:
    result_path = os.path.join(run_dir, "result.json")
    if not os.path.exists(result_path):
        return None
    with open(result_path) as f:
        result = json.load(f)
    refs = {k: set(v) for k, v in result["references"].items()
            if isinstance(v, list) and k != "evidence"}
    captions_path = result.get("captions_path")
    if not captions_path or not os.path.exists(captions_path):
        return None
    with open(captions_path) as f:
        clip_keys = [k for k in json.load(f)
                     if k not in ("subject_registry", "character_registry")]
    total = len(clip_keys)

    row = {
        "run_dir": run_dir,
        "question_id": result["question_id"],
        "video_id": result["video_id"],
        "score": result["score"],
        "total_segments": total,
        "set_sizes": {k: len(v) for k, v in refs.items() if k != "evidence"},
        "policies": {},
    }
    for name, policy in config.REFERENCE_POLICIES.items():
        base: set[str] = set()
        for s in policy.base_sets:
            base |= refs.get(s, set())
        base &= set(clip_keys)
        expanded, _ = expand_references(base, clip_keys, radius=policy.neighbor_radius)
        # radius-1 view of the same base, for the "fraction after expansion"
        # column even when the policy itself uses radius 0
        radius1, _ = expand_references(base, clip_keys, radius=1)
        row["policies"][name] = {
            "base_sets": list(policy.base_sets),
            "neighbor_radius": policy.neighbor_radius,
            "selected_count": len(expanded),
            "total_segments": total,
            "selected_fraction": len(expanded) / total if total else None,
            "selected_fraction_radius1": len(radius1) / total if total else None,
        }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for name in sorted(os.listdir(args.runs_root)):
        run_dir = os.path.join(args.runs_root, name)
        if os.path.isdir(run_dir):
            row = coverage_for_run(run_dir)
            if row:
                rows.append(row)

    summary = {"n_runs": len(rows), "policies": {}}
    for pname in config.REFERENCE_POLICIES:
        fr = [r["policies"][pname]["selected_fraction"]
              for r in rows if r["policies"][pname]["selected_fraction"] is not None]
        summary["policies"][pname] = {
            "mean_selected_fraction": sum(fr) / len(fr) if fr else None,
            "min": min(fr) if fr else None,
            "max": max(fr) if fr else None,
        }

    out = {"summary": summary, "per_qa": rows}
    out_path = args.out or os.path.join(args.runs_root, "reference_coverage.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(summary, indent=2))
    for r in rows:
        cells = "  ".join(
            f"{p}={r['policies'][p]['selected_count']}/{r['total_segments']}"
            f"({r['policies'][p]['selected_fraction']:.0%})"
            for p in config.REFERENCE_POLICIES)
        print(f"{r['question_id']}: {cells}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
