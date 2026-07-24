#!/usr/bin/env python3
"""Full-recaption rollout / evidence launcher.

Wraps ``surrogate_rollout/scripts/run_fresh_prompt_delta_evidence.py`` with two
process-local rebinds and nothing else:

1. ``build_free_form_generator`` -> the static generator (same as
   ``caption_prompt_opt.run_static_evidence``), so the caption prompt is emitted
   verbatim per segment and no prompt-generator model call occurs — the evidence
   policy matches the confirmation policy.
2. ``PromptDeltaInterventionRunner`` -> ``FullRecaptionInterventionRunner``, so
   the prompt delta is applied to every baseline segment (full recaption) instead
   of the source-QA-localized subset.

Both rebinds are in this process only; the incumbent and caption-prompt evidence
runs are separate processes and are unaffected. The proposer is untouched (its
evidence scope stays localized, so its untruncated request does not overflow);
only the recaption apply-scope grows.

Usage: identical CLI to the wrapped entry, e.g.

    python full_recaption_opt/run_full_recaption_evidence.py \
        --prepared-inputs <dir with manifest.json + component_config.json> \
        --parent-meta-prompt caption_prompt_opt/prompts/init_caption_prompt.json \
        --split-manifest <split.json> \
        --output-dir runs/full_recaption/evidence \
        --source-revision <rev> \
        --worker-result-timeout-seconds 2400
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve both `surrogate_rollout` and the sibling opt packages whether this
# package is nested inside surrogate_rollout or beside it.
for _ancestor in (Path(__file__).resolve().parents[1],
                  Path(__file__).resolve().parents[2]):
    if str(_ancestor) not in sys.path:
        sys.path.insert(0, str(_ancestor))

from surrogate_rollout.scripts import run_fresh_prompt_delta_evidence as evidence_entry
from caption_prompt_opt.run_static_evidence import static_free_form_generator
from full_recaption_opt.full_recaption_runner import FullRecaptionInterventionRunner


def main(argv=None) -> int:
    # Static generator: caption prompt verbatim, no prompt-generator model call.
    evidence_entry.build_free_form_generator = static_free_form_generator
    # Full recaption: apply the delta to every baseline segment.
    evidence_entry.PromptDeltaInterventionRunner = FullRecaptionInterventionRunner

    # The evidence entry cross-checks the prepared generator identity against
    # config.PROMPT_GENERATOR_{MODEL,BACKEND}_ID; align to the 'static' identity
    # in this process only (the generator is never called — pure gating marker).
    from surrogate_rollout import config as _sr_config
    _sr_config.PROMPT_GENERATOR_MODEL_ID = "static"
    _sr_config.PROMPT_GENERATOR_BACKEND_ID = "static"

    return evidence_entry.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
