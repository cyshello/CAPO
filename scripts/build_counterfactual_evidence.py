"""Build Stage 4.8 counterfactual evidence from saved artifacts only.

Stage 4.7 example:

    python -m surrogate_rollout.scripts.build_counterfactual_evidence \
      --stage4-7-root surrogate_rollout/runs/phase4_stage4_7_smoke_1f24dc1 \
      --output-dir surrogate_rollout/runs/phase4_stage4_8_checkpoint/evidence

Phase 2–3 example filters two conditions from one saved results.jsonl:

    python -m surrogate_rollout.scripts.build_counterfactual_evidence \
      --phase2-results path/to/results.jsonl \
      --baseline-filter '{"candidate_name":"Entity centric","rollout_mode":"full"}' \
      --candidate-filter '{"candidate_name":"Entity centric","selection_policy":"trace_only"}' \
      --output-dir /tmp/evidence
"""

from __future__ import annotations

import argparse
import json

from surrogate_rollout.optimization.evidence_builder import (
    CounterfactualEvidenceBuilder,
    load_normalized_evaluations,
    load_phase2_3_condition,
    load_stage4_7_pair,
    write_evidence_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline Stage 4.8 counterfactual-evidence builder")
    parser.add_argument("--stage4-7-root")
    parser.add_argument("--phase2-results")
    parser.add_argument("--baseline-records")
    parser.add_argument("--candidate-records")
    parser.add_argument("--baseline-filter")
    parser.add_argument("--candidate-filter")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source_modes = sum(bool(value) for value in (
        args.stage4_7_root, args.phase2_results, args.baseline_records))
    if source_modes != 1:
        parser.error("choose exactly one source: --stage4-7-root, "
                     "--phase2-results, or --baseline-records")
    if args.stage4_7_root:
        baseline, candidate = load_stage4_7_pair(args.stage4_7_root)
        source = {"format": "stage4_7", "root": args.stage4_7_root}
    elif args.phase2_results:
        if not args.baseline_filter or not args.candidate_filter:
            parser.error("Phase 2–3 input requires both JSON filter arguments")
        baseline = load_phase2_3_condition(
            args.phase2_results, condition_name="baseline",
            filters=json.loads(args.baseline_filter))
        candidate = load_phase2_3_condition(
            args.phase2_results, condition_name="candidate",
            filters=json.loads(args.candidate_filter))
        source = {"format": "phase2_3", "results": args.phase2_results,
                  "baseline_filter": json.loads(args.baseline_filter),
                  "candidate_filter": json.loads(args.candidate_filter)}
    else:
        if not args.candidate_records:
            parser.error("--baseline-records requires --candidate-records")
        baseline = load_normalized_evaluations(args.baseline_records)
        candidate = load_normalized_evaluations(args.candidate_records)
        source = {"format": "normalized", "baseline": args.baseline_records,
                  "candidate": args.candidate_records}

    evidence = CounterfactualEvidenceBuilder().build(baseline, candidate)
    jsonl_path, summary_path = write_evidence_artifacts(
        evidence, args.output_dir, source=source)
    print(jsonl_path)
    print(summary_path)


if __name__ == "__main__":
    main()
