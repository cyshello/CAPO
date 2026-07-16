# AGENTS.md

## Active specification

Read before modifying Phase 4:

1. `docs/CURRENT_METHOD.md`
2. `docs/IMPLEMENTATION_MIGRATION.md`
3. `docs/RUNBOOK.md`

When an older Phase 4 document conflicts with these files,
`docs/CURRENT_METHOD.md` wins.

## Repository preservation

- Inspect the branch, `git status`, and `git diff` before editing.
- Do not reset or discard existing work.
- Preserve Phase 0–3 behavior.
- Preserve stable segment IDs, mixed caption views, cache isolation, downstream
  DVD reasoning, versioned snapshots, and existing tests.
- Keep legacy caption caches read-only.
- Prefer minimal adapters and migrations over architectural rewrites.
- Do not rename files/classes solely for terminology.

## Active implementation scope

### Train roles and iteration cadence

The frozen top-level split remains 10 train / 10 validation / 10 test videos,
with three QAs per video. Derive Phase 4 roles only inside train:

- evidence pool: the eight `previously_cached` train videos;
- confirmation holdout: the two remaining train videos;
- no separate regression-video subset.

Each optimization iteration selects exactly three unique evidence videos and
runs all three QAs per video. Selection rotates deterministically through the
eight-video evidence pool and prioritizes videos not yet used in the current
coverage cycle. If fewer than three unused videos remain, fill the batch from
the deterministic rotation while keeping it unique. Once all eight evidence
videos have appeared, confirmation is due. Reset coverage only after
confirmation accepts the provisional state or rejects it and rolls back to the
last confirmed checkpoint.

Confirmation videos never propose properties or generate optimization
feedback. Validation and test never enter component-update feedback. Use
`correct_to_wrong` flips on the current three evidence videos as the regression
signal. Older separate regression-video and per-iteration confirmation flows
are superseded.

Implement:

- history-aware multi-property VLM inference routing;
- one current-policy full-caption pass over exactly three evidence videos per
  iteration;
- all baseline QAs for every source video;
- multiple property proposals per video;
- frame-only property-conditioned SigLIP retrieval within each source video:
  exact candidate property text against sampled frames, maximum frame pooling,
  deterministic top-M selection;
- one independent property intervention per
  `(source_video, candidate_property)`;
- frozen incumbent history;
- force-add exactly one ephemeral candidate after incumbent routed properties
  only on retrieved segments; the temporary sequence may contain
  `max_selected_properties + 1` entries;
- retain every incumbent property without conflict resolution and fail the
  candidate on prompt-budget overflow or any selected-segment caption failure;
- reuse incumbent captions only for unselected segments;
- parallel property interventions;
- rerun all source-video QAs per intervention;
- correctness-flip-only compact multimodal feedback;
- `S_feedback = S_sim ∩ (S_used ∪ S_usedagain)`;
- one iteration-level codebook/router update;
- acceptance of multiple supported properties per iteration;
- fixed deterministic scaffold.

Do not implement:

- history propagation;
- SLM-only routing;
- per-segment QA rollouts;
- scaffold optimization;
- trust regions;
- broad renaming.

Frozen local history is never a property-retrieval input or retrieval-cache
identity field. History is used only for baseline routing, baseline captioning,
later selective re-captioning, and later multimodal feedback. Retrieval also
must not directly consume QA text, answers, correctness, captions, traces, or
used-segment references.

## Execution policy

The user runs all main experiments manually.

Do not launch:

- benchmark-scale or multi-iteration experiments;
- long GPU jobs;
- paid-API experiments;
- commands that overwrite existing run artifacts.

You may run:

- unit tests;
- deterministic offline dry runs;
- the smallest strictly bounded smoke test when its scope and cost are explicit.

After implementation, update `docs/RUNBOOK.md` with exact copy-paste commands,
required environment variables, output paths, resume/recovery behavior, and
success checks. Do not leave placeholders when declaring completion.

## Required completion report

Report:

- files inspected and changed;
- behavior added or preserved;
- focused and full test commands;
- exact pass/fail counts;
- artifacts created;
- backward-compatibility notes;
- unresolved assumptions;
- exact commands the user should execute next.

Do not execute the user's main command.
