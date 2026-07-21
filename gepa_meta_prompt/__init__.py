"""Official-GEPA optimization of the free-form DVD meta-prompt (isolated entry).

This package is an *additive*, self-contained experiment entry. It optimizes
ONLY the free-form caption-instruction meta-prompt text (the
``VLMFreeFormInstructionGenerator`` template) using the installed ``gepa``
engine — the official reflective, Pareto-based prompt optimizer.

Design boundaries (see CLAUDE.md):

* Nothing in the Phase 0-3 infrastructure or the existing Phase 4 optimization
  modules is modified. The DVD runtime is reused via the existing Checkpoint-G
  factory; captioning and QA are reused as-is.
* The single optimizable component is ``meta_prompt``. Each GEPA training
  example is one video (three QAs); its absolute QA accuracy is the metric.
* Integration is a single :class:`~gepa.core.adapter.GEPAAdapter` implementation
  (:class:`DVDMetaPromptGEPAAdapter`); the ``gepa`` engine owns candidate
  selection, minibatch sampling, Pareto tracking, and the rollout budget.

Entry point: ``scripts/run_gepa_meta_prompt.py``.
"""

from surrogate_rollout.gepa_meta_prompt.dvd_single_video_evaluator import (
    GepaQAResult,
    GepaVideoInstance,
    GepaVideoScore,
    caption_and_score_video,
)
from surrogate_rollout.gepa_meta_prompt.gepa_adapter import (
    META_PROMPT_COMPONENT,
    DVDMetaPromptGEPAAdapter,
)
from surrogate_rollout.gepa_meta_prompt.reflection import (
    OpenAICompatibleReflectionMutator,
    ReflectionMutator,
    render_video_feedback,
)

__all__ = [
    "GepaQAResult",
    "GepaVideoInstance",
    "GepaVideoScore",
    "caption_and_score_video",
    "META_PROMPT_COMPONENT",
    "DVDMetaPromptGEPAAdapter",
    "OpenAICompatibleReflectionMutator",
    "ReflectionMutator",
    "render_video_feedback",
]
