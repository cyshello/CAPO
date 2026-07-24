"""Full-recaption training path — a baseline for the selective-intervention method.

This package is a thin, read-only variant of ``caption_prompt_opt``. It reuses
every component of the caption-prompt meta-prompt training pipeline unchanged and
differs in exactly one respect: during the evidence phase the prompt delta is
applied to EVERY baseline segment of the video (full recaption) instead of the
selective, source-QA-localized subset.

Nothing under ``surrogate_rollout`` or ``caption_prompt_opt`` is modified. The
only new logic is:

- ``full_recaption_runner.FullRecaptionInterventionRunner`` — a subclass of the
  incumbent intervention runner that widens the recaption apply-scope to all
  baseline segments and relaxes the source-QA localization invariant for that
  run only. The proposer's evidence scope is left localized on purpose, so the
  (untruncated) proposer request does not overflow; only the recaption apply set
  grows, and the longer feedback that results is absorbed by the existing
  ``TruncatingEpisodeFeedbackProvider``.
- ``run_full_recaption_evidence`` — the evidence launcher: same static-generator
  rebind as ``caption_prompt_opt.run_static_evidence`` plus a process-local swap
  of the intervention runner class.

The confirmation phase already recaptions the whole video, so it is reused as-is.
"""
