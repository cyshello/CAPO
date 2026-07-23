"""Confirmation evaluator that scores the caption prompt directly.

The only behavioral difference from ``DVDMetaPromptConfirmationEvaluator`` is
the per-version generator: instead of building an OpenAI free-form generator
that expands a meta-prompt into a per-segment instruction, it builds a
``StaticInstructionGenerator`` that emits the candidate/parent caption prompt
verbatim. Everything else (paired protocol, caching, promotion) is inherited.

A distinct ``policy_version`` keeps this evaluator's configuration identity —
and therefore its cache keys — disjoint from a concurrent meta-prompt run.
"""

from __future__ import annotations

from surrogate_rollout.optimization.dvd_meta_prompt_confirmation import (
    DVDMetaPromptConfirmationEvaluator,
)
from surrogate_rollout.optimization.schemas import MetaPromptVersion

from caption_prompt_opt.static_instruction_generator import (
    StaticInstructionGenerator,
)


class StaticGeneratorConfirmationEvaluator(DVDMetaPromptConfirmationEvaluator):
    """DVD confirmation using the caption prompt itself as the instruction."""

    policy_version = "caption_prompt_confirmation_v1"

    def _generator(
        self, version: MetaPromptVersion,
    ) -> StaticInstructionGenerator:
        return StaticInstructionGenerator(
            template_text=version.text,
            meta_prompt_id=version.meta_prompt_id,
            model_id=self.prompt_generator_model_id,
            backend_id=self.prompt_generator_backend_id,
        )
