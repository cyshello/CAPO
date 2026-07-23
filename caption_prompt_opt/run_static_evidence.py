#!/usr/bin/env python3
"""Static-generator rollout / evidence launcher.

Wraps ``surrogate_rollout/scripts/run_fresh_prompt_delta_evidence.py`` and
replaces the OpenAI free-form prompt generator with ``StaticInstructionGenerator``
so the **evidence policy matches the confirmation policy**: the caption prompt
is emitted verbatim per segment and **no prompt-generator model call occurs**.
The proposer and everything else in the evidence run are untouched.

Mechanism (zero-touch): the evidence entry resolves ``build_free_form_generator``
from its own module namespace at rollout time. We rebind that name in this
process only; the incumbent evidence run is a separate process and is
unaffected. The builder's internally constructed placeholder generator (built
inside ``from_local_qwen``) is never invoked, because the rollout uses the
reassigned static generator.

Usage: identical CLI to the wrapped entry, e.g.

    python caption_prompt_opt/run_static_evidence.py \
        --prepared-inputs <dir with manifest.json + component_config.json> \
        --parent-meta-prompt caption_prompt_opt/prompts/init_caption_prompt.json \
        --split-manifest <split.json> \
        --output-dir runs/caption_prompt/evidence \
        --source-revision <rev> \
        --worker-result-timeout-seconds 2400
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make both `import surrogate_rollout` and `import caption_prompt_opt` resolve,
# whether caption_prompt_opt lives inside surrogate_rollout (nested) or beside it
# (sibling). Adding both ancestors covers either layout harmlessly.
for _ancestor in (Path(__file__).resolve().parents[1],
                  Path(__file__).resolve().parents[2]):
    if str(_ancestor) not in sys.path:
        sys.path.insert(0, str(_ancestor))

from surrogate_rollout.scripts import run_fresh_prompt_delta_evidence as evidence_entry
from caption_prompt_opt.static_instruction_generator import StaticInstructionGenerator


def static_free_form_generator(
    provider,
    *,
    template_text=None,
    meta_prompt_id=None,
    model_id=None,
    backend_id=None,
    max_tokens=None,
    **_ignored,
):
    """Drop-in for ``build_free_form_generator``: no model call, caption verbatim.

    ``provider``/``model_id``/``max_tokens`` are accepted for signature
    compatibility and ignored — the caption prompt (``template_text``) is the
    instruction for every segment.
    """
    if not template_text:
        raise ValueError(
            "static rollout generator requires the caption prompt text "
            "(parent.text); refusing to fall back to a model generator")
    return StaticInstructionGenerator(
        template_text=template_text,
        meta_prompt_id=meta_prompt_id,
        model_id="static",
        backend_id=backend_id,
        max_tokens=(1 if max_tokens is None else int(max_tokens)),
    )


def main(argv=None) -> int:
    # Rebind the symbol the evidence entry uses when it assigns
    # builder.free_form_generator (its module-local import of
    # build_free_form_generator). Process-local; incumbent run untouched.
    evidence_entry.build_free_form_generator = static_free_form_generator

    # The evidence entry cross-checks the prepared component_config's generator
    # identity against config.PROMPT_GENERATOR_{MODEL,BACKEND}_ID and refuses to
    # start on a mismatch. config.PROMPT_GENERATOR_BACKEND_ID is hardcoded (not
    # env-overridable), so for the static path we align config to the prepared
    # 'static' identity here, in this process only. The generator is never
    # called, so these values are pure provenance/gating markers.
    from surrogate_rollout import config as _sr_config
    _sr_config.PROMPT_GENERATOR_MODEL_ID = "static"
    _sr_config.PROMPT_GENERATOR_BACKEND_ID = "static"

    return evidence_entry.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
