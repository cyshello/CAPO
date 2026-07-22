# AGENTS.md

## Active specification

Read before modifying the fresh prompt-delta path:

1. `docs/CURRENT_METHOD.md`
2. `docs/IMPLEMENTATION_MIGRATION.md`
3. `docs/RUNBOOK.md`

`docs/CURRENT_METHOD.md` wins when older notes conflict.

## Repository preservation

- Inspect the branch, `git status`, and `git diff` before editing.
- Do not reset or discard existing work.
- Preserve stable segment IDs, mixed caption views, cache isolation, downstream
  DVD reasoning, immutable artifact identity, and source non-mutation checks.
- Keep existing run artifacts and caption caches read-only.
- Do not retain unused codebook/router/property/GEPA adapters for compatibility.
- Do not rename files/classes solely for terminology.

## Active implementation scope

The active Phase 4 implementation is fresh prompt-delta only:

- generate per-segment static-meta prompt instructions with OpenAI
  `gpt-4o-mini`;
- compose them with the `replace_body` scaffold;
- caption with local Qwen over original segment frames without passing frozen
  history separately to the captioner;
- propose prompt deltas with independent per-QA provider calls;
- apply each candidate to the source QA localized segments as one mixed view;
- rerun every sibling QA from the same video on that mixed view;
- generate detailed feedback and compact memory from the full sibling outcome;
- update and confirm only the meta-prompt.

Do not reintroduce:

- legacy property/codebook/router optimization;
- GEPA meta-prompt runners;
- rule-based, SLM, or history-aware property routers;
- property-conditioned SigLIP retrieval;
- legacy property intervention adapters;
- deterministic semantic validators or candidate fallbacks.

## Execution policy

The user runs all main experiments manually.

Do not launch:

- benchmark-scale or multi-iteration experiments;
- long GPU jobs;
- paid-API experiments;
- commands that overwrite existing run artifacts.

You may run:

- unit tests;
- Python compile checks;
- CLI `--help`;
- shell syntax checks;
- deterministic file/package inspections.

After implementation, update `docs/RUNBOOK.md` with exact commands, required
environment variables, output paths, resume behavior, and success checks.

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

Do not execute the user's main experiment command.
