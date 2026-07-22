# Implementation Migration

The implementation has been migrated from legacy property/codebook/router Phase
4 to fresh prompt-delta-only Phase 4.

## Kept

- Stable segment IDs and mixed caption views.
- History-aware baseline caption cache identity.
- DVD downstream QA execution.
- Fresh per-QA prompt-delta proposal artifacts.
- Candidate mixed-view sibling QA reruns.
- Episode feedback, compact feedback memory, meta-prompt updater, and paired
  DVD meta-prompt confirmation.
- Existing visual-index utilities used by generic clip retrieval.

## Removed

- `gepa_meta_prompt/`.
- Property proposal, property intervention, property memory, property retrieval,
  codebook updater, router updater, bounded smoke, offline/memory/real fixed
  scaffold iteration modules.
- Routed caption view, offline dry run, rule-based router, SLM scaffold,
  deterministic scaffold, history-aware property router, router validators, and
  structured router policy modules.
- Legacy Phase 4 launcher scripts and tests that imported those modules.

## Active Boundaries

- `captioning/history_aware_baseline.py` supports only
  `routing_mode="free_form_generator"`.
- `prompt_routing/scaffold_applier.py` supports only `policy_type="replace_body"`.
- `prompt_routing/persistence.py` keeps scaffold parsing/stores and atomic
  writes only.
- `prompt_routing/schemas.py` keeps prompt-entry, routing-decision, scaffold,
  composed-prompt, and minimal Phase 4 config records. It no longer defines
  `PromptBankSnapshot`, `RouterPolicySnapshot`, or `RoutingRule`.
- Fresh baseline records use `prompt_delta_source_paths`; old `property_*`
  proposal/retrieval manifest fields are not emitted.

## Validation Policy

Do not add deterministic semantic validators for prompt deltas. Keep only
mechanical integrity checks for malformed provider output, invalid IDs, invalid
scope, artifact identity, hashes, resume, and source non-mutation.
