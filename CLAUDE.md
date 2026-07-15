# CLAUDE.md

## Project

This repository studies per-segment caption-prompt routing and prompt-bank
optimization for caption-based long-form video understanding.

The implementation is built on top of the completed Phase 0–3 selective
surrogate-rollout infrastructure.

## Current status

Phases 0–3 are implemented.

The existing implementation provides:

- one shared DVD reasoning path;
- full candidate rollout;
- selective surrogate rollout;
- prompt-versioned caption caching;
- mixed caption views;
- trajectory/reference extraction;
- segment-selection policies;
- fidelity and cost evaluation infrastructure.

See `PHASE2_3_SURROGATE.md` for the completed implementation, invariants,
commands, and tests.

Do not reimplement or broadly refactor this infrastructure.

## Current phase

The next implementation is described in:

- `PHASE4_PROMPT_ROUTING.md`

This phase adds:

1. a prompt bank;
2. counterfactual evidence construction;
3. structured feedback generation;
4. scaffold-based prompt-bank updates;
5. per-segment prompt routing;
6. an optimization loop that invokes the existing rollout evaluators.

## Architecture boundaries

New code should be added primarily under:

prompt_routing/
optimization/

The existing rollout evaluators should be treated as reusable evaluation services.
- Do not duplicate:
- DVD reasoning;
- caption generation wrappers;
- full rollout evaluation;
- selective rollout evaluation;
- mixed-view construction;
- caption cache logic;
- reference extraction;
- segment-selection logic.

## Replaceable policies
The following components must remain independently replaceable.
FeedbackGenerator
Converts counterfactual rollout evidence into structured feedback.
The initial implementation may use a large language model, but the remainder
of the system must depend only on its typed interface.
Alternative feedback policies must be usable without modifying rollout
evaluation or evidence collection.

## ScaffoldApplier
Inference-time composer. Combines the prompt-bank entries selected by the
router with the fixed scaffold contract into one final captioning prompt.
It does not update the prompt bank.
The initial implementation may use a deterministic scaffold or a large
language model.
It must later be replaceable by a small language model without modifying the
optimization loop, feedback generator, or rollout evaluator.

## Update proposers
PromptBankUpdateProposer is a separate optimization-time component that
proposes changes to the prompt bank. RouterUpdateProposer and
ScaffoldUpdateProposer are likewise separate optimization-time components.
None of these compose captioning prompts; they must never be conflated with
ScaffoldApplier.

## PromptRouter
Selects zero or more prompt-bank entries for each video segment.
The initial implementation may use a deterministic or scaffold-based policy.
It must later support an SLM implementation behind the same interface.

## GEPA status
The previous plan used official GEPA as the complete optimizer. That plan is
archived and is not the active implementation target.
Do not integrate GEPA unless explicitly requested in a later phase.
GEPA-related code already present in the repository should not be deleted
unless it blocks the current implementation.

## Modification policy
Inspect existing code before adding abstractions.
Prefer adapters and composition over changes to Phase 0–3 modules.
Keep new public interfaces typed.
Put policy choices in configuration.
Preserve existing run artifacts and cache formats.
Preserve all existing Phase 0–3 tests.
Add focused unit tests for every new policy boundary.
Avoid unrelated cleanup or broad refactoring.
Never overwrite incumbent or baseline caption caches.
Persist prompt-bank versions, evidence, feedback, and update provenance.

## Working procedure
Before editing code:
1. inspect PHASE2_3_SURROGATE.md;
2. inspect the actual Phase 0–3 implementation;
3. inspect PHASE4_PROMPT_ROUTING.md;
4. report the minimal integration points;
5. identify any discrepancy between documentation and code.

After each implementation step, report:
1. files inspected;
2. files changed;
3. behavior added;
4. tests and commands run;
5. unresolved assumptions;
6. the next concrete step.