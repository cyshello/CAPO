# Current Method: Fresh Prompt-Delta Meta-Prompt Optimization

This is the active Phase 4 method. Legacy property/codebook/router and GEPA
paths are removed.

## Runtime

- The initial/current meta-prompt is a `MetaPromptVersion` artifact.
- The active prompt generator is OpenAI `gpt-4o-mini`.
- Generator requests use the static-meta replace-body request schema:
  0.5 FPS sampled frames resized to half resolution, frozen preceding-caption
  history, and the current parent meta-prompt.
- The generated instruction replaces the caption prompt task body through the
  `replace_body` scaffold.
- The local Qwen captioner receives original segment frames and the composed
  prompt. It does not receive frozen history as a separate captioner payload.
- DVD QA and frame-inspection execution remain the downstream evaluator.

## Proposal

Prompt-delta proposals are always per-QA isolated calls.

Each eligible source QA request contains only:

- source QA question, choices, golden answer, baseline answer/correctness;
- source QA trajectory;
- source QA localized segment evidence;
- referenced segment prompts, captions, and bounded histories;
- current parent meta-prompt.

Sibling QA questions, trajectories, outcomes, evidence, and candidates are not
included in the proposer request.

Schema-valid candidates are passed to intervention without semantic validation,
repair calls, deterministic fallbacks, or heuristic gates. Mechanical checks
only verify JSON shape, requested source QA ID, valid segment IDs, source-QA
localized intervention scope, artifact identity, resume hashes, and
non-mutation.

## Intervention

Each candidate jointly recaptions all selected localized segments of its source
QA and creates one candidate mixed view. The same mixed view reruns every sibling
QA from that video. Episodes persist source QA provenance, the proposed delta,
modified segment IDs, mixed-view identity, and all sibling QA transitions.

Single-segment ablations are not part of the active path.

## Feedback And Update

The feedback model sees the candidate mixed view outcome for source and sibling
QAs. Code does not infer semantic benefit/harm labels. Compact memory is
generated as short natural language by the feedback generator and may mention
source improvements and sibling regressions together.

The updater consumes compact/full feedback and proposes a new meta-prompt. Real
confirmation evaluates parent and candidate meta-prompts on the frozen
confirmation videos using identical sampled inputs and independent on-policy
caption histories.

## Removed Legacy

The repository no longer keeps runtime support for:

- property proposals/interventions/retrieval;
- memory-conditioned codebook/router updaters;
- routed caption-view builders and rule/SLM/history-aware property routers;
- GEPA meta-prompt runners;
- legacy property intervention adapters.
