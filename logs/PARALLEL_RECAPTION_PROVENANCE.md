# Full-recaption run 20260725_080528 — code that actually ran

The run records `source_revision = 35cedcf7a256e9fb7a2a1a511acc3459540d74fd`.
From the restart on **2026-07-26 ~14:08 KST** onward that is not the whole
truth: the working tree carried an uncommitted change, deliberately left
uncommitted.

## Restart history on 2026-07-26

- **14:08** — relaunched on GPUs 0,1,2,4,5,6,7 with the parallel runner. Resumed
  in place: four baselines, five episodes and ~196 cached captions reused,
  nothing recomputed. Episode 006 completed. Measured on the fresh captions:
  max 7 concurrent, typically 5–6.
- **14:17** — the user cut the allocation to **GPUs 0,1,2,4,5**. `worker_gpus` is
  in the baseline input fingerprint, so the evidence root was quarantined as
  `runs/fresh_prompt_delta_iteration_20260725_080529_evidence.gpu7_20260726_141733`
  (6 episodes, 4 baselines preserved, not deleted) and the run relaunched on the
  five-GPU list. Baselines rebuild from the caption cache; intervention captions
  are regenerated because the intervention identity hash moves with the
  fingerprint.

## Why it was not committed

`--source-revision` is folded into the baseline input fingerprint
(`optimization/baseline_phase.py:322` → `input_fingerprint` → `run_id`).
Committing moves `git rev-parse HEAD`, changes the fingerprint, and invalidates
the four completed baseline videos (~1126 caption calls) plus the five completed
intervention episodes. The change alters *when* captions are issued, not what
they are, so re-deriving that evidence would have bought nothing.

Decision made by the user on 2026-07-26 after being shown both options.

## What changed

- `full_recaption_opt/parallel_intervention_runner.py` (new) —
  `ParallelFullRecaptionInterventionRunner`. The full-recaption caption loop is
  fanned out across the already-running GPU pool via probe → parallel → replay,
  leaving the parent `run()` in `surrogate_rollout` untouched.
- `full_recaption_opt/run_full_recaption_evidence.py` — rebinds the evidence
  entry to the parallel runner instead of `FullRecaptionInterventionRunner`.
- `tests/test_parallel_full_recaption_intervention.py` (new) — 8 tests.

Verbatim copies as of the restart:

- `logs/parallel_recaption_20260726.patch` (`git diff` of tracked files)
- `logs/parallel_intervention_runner.py.asrun`
- `logs/test_parallel_full_recaption_intervention.py.asrun`

## Why the evidence is unaffected

- Captions are decoded greedily (`config.CAPTION_DECODING`: `temperature 0.0`,
  `top_p 1.0`), and each pool worker still serves one request at a time, so a
  caption does not depend on which GPU took it or on what ran beside it.
- The intervention reads its history from the baseline's
  `frozen_histories_path` (`fresh_prompt_delta_evidence.py:1457`), so segments
  were never sequentially dependent — only the caller was serial.
- Caption kwargs, including the intervention identity hash that keys the caption
  cache, are computed by the parent's own code in both passes, not restated by
  the adapter.
- The replay pass returns the real result objects, so `model_call_count` and
  `caption_seconds` stay the counts a sequential run would have recorded.

## Episodes 001–005

Captioned sequentially before the restart. Identical in content to what the
parallel path would have produced, for the reasons above.

## To restore the sequential loop

`FULL_RECAPTION_MAX_IN_FLIGHT=1`.

## After the run finishes

Commit the three files above. A later run that starts its own baseline will then
record an accurate revision.
