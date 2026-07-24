"""Full-recaption baseline for the selective-intervention method.

This package is the META prompt-delta pipeline (LLM free-form prompt generator +
v5 updater) run with EXACTLY one change: during the evidence phase the prompt
delta is applied to EVERY baseline segment of the video (full recaption) instead
of the selective, source-QA-localized subset. It is the baseline that isolates
selective intervention — identical to a meta run in every other respect
(generator, updater, captioner, DVD QA, optimizer, cohort), so any difference is
attributable to selective-vs-full intervention alone.

Nothing under ``surrogate_rollout`` is modified. The only new logic is:

- ``full_recaption_runner.FullRecaptionInterventionRunner`` — a subclass of the
  incumbent intervention runner that widens the recaption apply-scope to all
  baseline segments and relaxes the source-QA localization invariant for that
  run only. The proposer's evidence scope is left localized on purpose, so its
  untruncated request does not overflow; only the recaption apply set grows.
- ``run_full_recaption_evidence`` — the evidence launcher: one process-local swap
  of the intervention runner class, delegating everything else to the meta
  evidence entry (the free-form generator stays active).

The launcher ``scripts/run_full_recaption_kiter.sh`` reuses the meta K-iteration
driver verbatim (only setting FRESH_PROMPT_DELTA_EVIDENCE_ENTRY and a distinct
experiment label), so a full-recaption run inherits identical settings to a meta
run. The confirmation phase already recaptions the whole video, so it is reused
as-is.
"""
