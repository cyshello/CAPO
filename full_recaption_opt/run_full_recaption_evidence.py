#!/usr/bin/env python3
"""Full-recaption evidence launcher — the META pipeline with a full-recaption runner.

This is the meta-prompt (free-form generator + v5 updater) evidence run with EXACTLY
one change: the prompt delta is applied to EVERY baseline segment of the video
(full recaption) instead of the source-QA-localized subset. It is the baseline
that isolates selective intervention: identical to the meta run in every other
respect (LLM prompt generator, updater, captioner, DVD QA, cohort), so any
difference is attributable to selective-vs-full intervention alone.

Mechanism (one process-local rebind): swap
``PromptDeltaInterventionRunner`` -> ``ParallelFullRecaptionInterventionRunner``
on the evidence entry. The OpenAI free-form prompt generator is left ACTIVE (this
is the meta path, not the static caption path), and the proposer stays localized
so its untruncated request does not overflow — only the recaption apply-scope
grows.

The runner is the full-recaption one with its caption loop fanned out across the
already-running GPU pool; see ``parallel_intervention_runner``. Recaption scope
and episode contents are unchanged — set
``FULL_RECAPTION_MAX_IN_FLIGHT=1`` to get the sequential loop back.

Usage: identical CLI to scripts/run_fresh_prompt_delta_evidence.py (the meta
iteration shell invokes this when FRESH_PROMPT_DELTA_EVIDENCE_ENTRY points here).
"""

from __future__ import annotations

import sys
from pathlib import Path

for _ancestor in (Path(__file__).resolve().parents[1],
                  Path(__file__).resolve().parents[2]):
    if str(_ancestor) not in sys.path:
        sys.path.insert(0, str(_ancestor))

from surrogate_rollout.scripts import run_fresh_prompt_delta_evidence as evidence_entry
from full_recaption_opt.parallel_intervention_runner import (
    ParallelFullRecaptionInterventionRunner,
)


def main(argv=None) -> int:
    # Full recaption: apply the delta to every baseline segment. Nothing else is
    # changed — the meta path's free-form generator and updater are untouched.
    evidence_entry.PromptDeltaInterventionRunner = (
        ParallelFullRecaptionInterventionRunner)
    return evidence_entry.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
