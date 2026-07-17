# Current Method: Batched Property-Wise Intervention for History-Aware Video Captioning

## Status

This document is the active source of truth for Phase 4. It supersedes older
Phase 4 plans wherever they conflict.

The existing Phase 0–3 caption cache, mixed-caption view, downstream reasoning,
artifact persistence, and versioning infrastructure should be preserved.

### Superseded batch descriptions

Earlier Phase 4 text and runners that use a separate regression-video batch,
evaluate confirmation every iteration, full-caption every train video every
iteration, or optimize the scaffold are superseded. The active data policy is:

- frozen top-level split: 10 train / 10 validation / 10 test videos, three QAs
  per video;
- evidence pool: eight previously inspected/captioned train videos;
- confirmation holdout: the other two train videos;
- no separate regression-video subset;
- exactly three evidence videos and nine baseline QAs per optimization
  iteration;
- validation and test remain outside component-update feedback.

## 1. Core objects

- A **segment** is the minimum captioning unit.
- A **property** is one reusable captioning instruction that can be composed
  with other properties.
- A **codebook** is the versioned bank of active properties.
- A **router** is a lightweight VLM that selects multiple codebook properties
  for one segment.
- **Local history** is the caption history from preceding segments within the
  same configured temporal block.
- A **source video** is the video whose baseline QA evidence generated a
  candidate property.
- A **property intervention** evaluates one candidate property on its source
  video while keeping the incumbent policy and history fixed.

Existing code identifiers containing `prompt_bank`, `prompt_id`, or `prompt`
may remain for compatibility. Research-facing descriptions should use
`property` and `codebook` when entries are composable instructions.

## 2. Inference

For segment \(s_{i,t}\), local history \(h_{i,t}\), and codebook
\(\mathcal P\), the router selects a multi-label property set:

\[
P_{i,t}=R_\phi(s_{i,t},h_{i,t},\mathcal P).
\]

The router receives:

- sampled frames from the current segment;
- local caption history;
- active property IDs, names, and instructions.

The router must not receive:

- the downstream question;
- answer choices or ground truth;
- QA reasoning traces;
- retrieved or consumed segment IDs;
- downstream predictions.

The local Qwen/vLLM router uses a fail-closed JSON Schema decoding constraint.
For the frozen active codebook, `property_ids` is the only permitted output
field, its items are restricted to active property IDs, and its array length is
bounded by the current selection limit. Structured-output fallback is disabled:
if the installed backend cannot enforce the schema, routing fails instead of
silently reverting to unconstrained text generation. This constraint applies
only to routing; ordinary segment caption generation remains unconstrained.

A fixed deterministic composer builds the captioning prompt:

\[
\theta_{i,t}
=
\operatorname{Compose}
\left(
\theta_{\mathrm{base}},
P_{i,t}
\right).
\]

The captioner generates:

\[
c_{i,t}
=
C(s_{i,t},h_{i,t},\theta_{i,t}).
\]

History is restricted to configured temporal blocks. Different blocks may be
processed independently, while segments within one block use preceding
captions as local history.

When multiple caption GPUs are configured, the exact temporal history block is
also the scheduling unit. One persistent worker process owns one Qwen/vLLM
instance on one GPU and processes every segment assigned to a block in temporal
order. Blocks are assigned deterministically and may execute concurrently;
their routing decisions, composed prompts, histories, cache rows, captions, and
registries are merged back in ascending source-segment order. Workers append to
private cache-manifest fragments, and the parent validates and merges those
fragments after every worker succeeds. GPU assignment is execution metadata,
not a caption-cache identity field; the history boundary rule and complete
history-aware semantic identity remain unchanged.

The GPU workers are run-scoped services rather than one-build subprocesses.
After a block batch returns, they remain on their command queues and reuse the
same loaded Qwen/vLLM engines for later evidence videos, selective property
captioning, and confirmation captioning in that process. The parent waits for
task results, not worker process exit. Selective caption calls use the same
worker-private manifest discipline and merge completed registrations in the
parent. Only the explicit run boundary shuts down the vLLM EngineCore and joins
the workers; a bounded smoke performs this cleanup in `finally`.

DVD `frame_inspect_tool` vision calls are raw-VLM tasks on this same pool. While
the pool is active, the parent installs a lightweight `dvd_backend` captioner
proxy and must never lazily initialize another local Qwen/vLLM engine. Without
an active pool, the original standalone `dvd_backend.get_captioner()` behavior
is unchanged.

## 3. One optimization iteration

At iteration \(k\), snapshot the current policy:

\[
\Pi^{(k)}
=
\left(
\mathcal P^{(k)},
R^{(k)},
\theta_{\mathrm{base}}
\right).
\]

Load exactly three unique videos from the eight-video evidence pool:

\[
\mathcal V^{(k)}
=
\{v_1,v_2,v_3\}.
\]

The policy snapshot remains fixed throughout the iteration. Selection rotates
deterministically through the evidence pool and prioritizes videos not yet used
since the last confirmation. If only one or two unused videos remain, fill the
batch from the deterministic rotation without duplicates. Once all eight have
appeared, confirmation is due and no new evidence iteration starts until the
confirmation decision is persisted.

### 3.1 Full-caption the three videos once

For each selected video, run the current router and captioner over the complete
video exactly once:

\[
\mathcal M_i^{(k)}
=
\operatorname{Caption}
\left(
v_i;\Pi^{(k)}
\right).
\]

Persist:

- complete incumbent captions;
- per-segment router decisions;
- per-segment composed prompts;
- sampled frame references;
- local-history snapshots;
- codebook, router, and scaffold versions;
- caption cache keys.

The local-history snapshots created by this full-caption pass are frozen for
the rest of the iteration.

### 3.2 Run all baseline QAs

Run every QA associated with each video using its incumbent caption memory:

\[
(y_{i,q}^{(k)},\tau_{i,q}^{(k)})
=
\operatorname{QA}
\left(
\mathcal M_i^{(k)},q
\right),
\qquad q\in Q_i.
\]

Persist the answer, correctness, reasoning trace, and used segment IDs.
Runtime/tool errors, null predictions, and parse failures are QA execution
failures rather than incorrect answers. Run and persist all three QA attempts,
then fail the video before property proposal if any attempt did not complete
successfully. Only three successfully parsed QA results may enter proposal.

### 3.3 Propose multiple properties per video

Analyze the baseline QA results and relevant visual/caption evidence to propose
zero or more candidate properties for each video:

\[
\mathcal C_i
=
\{p_{i,1},\ldots,p_{i,B_i}\}.
\]

A video may generate multiple candidates because different QAs may expose
different captioning failures.

Each candidate must record its lineage:

- source video ID;
- source QA IDs;
- concise failure evidence;
- current-codebook coverage assessment;
- proposed property instruction.

Candidates are not added to the codebook yet.

Property-proposal model input preserves all three QA outcomes but contains no
source-video ID, question ID, segment ID, provider priority rank, tool-call ID,
or payload-truncation metadata. For each QA it contains the question and answer
choices, ground truth and baseline prediction, at most three bounded sanitized
reasoning events, and at most three actual used-segment evidence items. Each
evidence item pairs one deterministic representative frame with its incumbent
baseline caption; segment lineage remains in a private artifact and is not sent
to the model. Focused cited/inspected/returned segments take priority, with a
deterministic representative sample from consumed segments only when needed.

The current codebook and a source-free output schema complete the multimodal
request. Provider-reported `covered_by_existing_property_ids` are non-binding
hints only and are stored internally as `coverage_hints`. A non-empty hint list
does not reject a proposal. Proposal parsing rejects only deterministic exact
text or active-ID collisions, malformed or duplicate lineage, instance-specific
leakage, and instructions requiring non-visual, external, background, or
historical knowledge. Contradictory coverage claims require an explicit partial
coverage or uncertainty explanation. Source video and all three source QA IDs
are reattached internally after parsing. The text portion and every transformed
image are independently bounded and identity-bound. `request.json` records the
logical model input, `provider_request.json` records the exact secret-free API
body, and `input_identity.json` records private source/segment/frame lineage
that is never sent to the provider.

Semantic coverage is deliberately deferred until after retrieval,
intervention, and correctness-flip feedback. Checkpoint 3D
`coverage_assessment` and `covered_by_property_ids`, rather than proposal hints,
drive the iteration-level `add`, `revise`, `merge`, router-supervision, or
`no_op` decision.

Near-duplicate candidates may be grouped for reporting, but every
property-source-video pair remains independently evaluable.

### 3.4 Property-conditioned segment retrieval

For each candidate property \(p_{i,j}\), retrieve segments only from its source
video:

\[
S_{\mathrm{sim}}^{i,j}
=
\operatorname{TopM}_{t}
\left[
\max_f\;\cos\left(
E_{\mathrm{text}}(p_{i,j}),
E_{\mathrm{visual}}(x_{i,t,f})
\right)
\right].
\]

The exact normalized candidate property text is the only text query. SigLIP
scores every sampled frame in every valid source-video segment; maximum frame
pooling produces the segment score. Select the top `M` by descending score,
ascending segment start, then stable segment ID. Retrieval must not use frozen
history, captions, questions, answer choices, answers, correctness, reasoning
traces, or used-segment references.

`property_retrieval_top_k` defaults to 5. Persist every frame score and every
ranked valid segment, not only the selected segment IDs.

### 3.4.1 Post-intervention execution boundary

`Phase4Config.post_intervention_mode` is a typed three-value boundary:
`qa_only`, `feedback_only`, or `provisional_update`. Production optimization
defaults to `provisional_update`. `qa_only` stops after candidate mixed-view QA
reruns and persisted correctness transitions. `feedback_only` additionally
runs flip-only feedback and property aggregation. `provisional_update`
additionally materializes provisional bank/router artifacts.

The bounded-smoke runner never performs confirmation or changes confirmed or
canonical pointers. The mode is excluded from baseline, proposal, retrieval,
intervention, and QA identities, allowing a later mode to resume those exact
artifacts. Feedback and provisional-update manifests bind their respective
downstream stage identities.

### 3.5 Independent property interventions

Every candidate property is evaluated in a separate counterfactual run that
starts from the same incumbent memory and frozen history.

For each selected segment \(t\in S_{\mathrm{sim}}^{i,j}\), preserve the
incumbent routed property set \(P_{i,t}^{(k)}\) and force-add only the candidate
property:

\[
\widetilde P_{i,t}^{\,j}
=
P_{i,t}^{(k)}
\cup
\{p_{i,j}\}.
\]

Then recaption the selected segment using the frozen incumbent history:

\[
\widetilde c_{i,t}^{\,j}
=
C
\left(
s_{i,t},
h_{i,t}^{(k)},
\operatorname{Compose}
\left(
\theta_{\mathrm{base}},
\widetilde P_{i,t}^{\,j}
\right)
\right).
\]

Unselected segments retain incumbent captions. If any selected segment cannot
be re-captioned or validated, the entire candidate intervention fails
explicitly; selected segments never fall back to incumbent captions.

Build the mixed caption view:

\[
\widetilde{\mathcal M}_{i,j}
=
\operatorname{Replace}
\left(
\mathcal M_i^{(k)},
\{\widetilde c_{i,t}^{\,j}\}_{t\in S_{\mathrm{sim}}^{i,j}}
\right).
\]

Important invariants:

- one candidate property per counterfactual rollout;
- no candidate modifies another candidate's memory;
- all candidates use the same incumbent policy and history;
- candidate properties bypass the router during intervention and are
  force-applied only to their retrieved segments;
- no history propagation occurs after recaptioning;
- no full recaption occurs between candidate interventions.
- the ephemeral sequence appends exactly one candidate after all incumbent
  routed properties and may contain `max_selected_properties + 1` entries;
- incumbent properties are never removed and semantic conflicts are not
  resolved during intervention;
- prompt-budget overflow fails the candidate rather than dropping properties.

All property-source-video interventions may run in parallel. Segment captioning
within one intervention may also run in parallel.

### 3.6 Rerun all QAs of the source video

For each property intervention, rerun every QA associated with the source
video:

\[
(\widetilde y_{i,j,q},\widetilde\tau_{i,j,q})
=
\operatorname{QA}
\left(
\widetilde{\mathcal M}_{i,j},q
\right),
\qquad q\in Q_i.
\]

This measures:

- improvement on the QA that motivated the property;
- transfer to other QAs from the same video;
- collateral regressions.

The downstream reasoning path runs once per property-specific mixed view, not
once per changed segment.

## 4. Interventional feedback

### 4.1 Feedback trigger

Generate optimization feedback only for correctness flips:

- `wrong_to_correct`;
- `correct_to_wrong`.

Keep `correct_to_correct` and `wrong_to_wrong` for analysis, but do not send
them to the feedback model in the initial implementation.

### 4.2 Feedback segment set

For QA \(q\), define:

- \(S_{\mathrm{used}}^{i,q}\): segments used by incumbent reasoning;
- \(S_{\mathrm{usedagain}}^{i,j,q}\): segments used by candidate reasoning.

Only intervened segments used by either reasoning process receive attribution:

\[
S_{\mathrm{feedback}}^{i,j,q}
=
S_{\mathrm{sim}}^{i,j}
\cap
\left(
S_{\mathrm{used}}^{i,q}
\cup
S_{\mathrm{usedagain}}^{i,j,q}
\right).
\]

Similarity retrieval determines what is recaptioned. Before/after reasoning
determines which recaptioned segments are shown to the feedback model.
If this intersection is empty for a correctness flip, persist an explicit
`empty_s_feedback` rejection and do not call the feedback model.

### 4.3 Feedback input

For each correctness flip, provide:

- question and ground-truth answer;
- incumbent and candidate answers;
- relevant incumbent and candidate reasoning excerpts;
- candidate property ID and full instruction text;
- current codebook entries relevant to coverage checking;
- only segments in \(S_{\mathrm{feedback}}^{i,j,q}\);
- for each included segment:
  - sampled frames;
  - concise frozen-history context;
  - incumbent routed properties;
  - force-added candidate property;
  - incumbent and candidate captions.

Oversized sampled frames are deterministically resized and JPEG-compressed
under a persisted bounded transform configuration. Persist source and
transformed hashes and reject only after the configured transform ladder is
exhausted.

Do not send:

- full-video frame sets;
- full-video caption arrays;
- complete raw trajectories;
- artifact paths without loading relevant contents;
- non-flip examples in the feedback batch.

### 4.4 Feedback output

Reuse the existing feedback schema where possible. The feedback should identify:

- whether the candidate property supplied useful information;
- whether it was harmful or unnecessary;
- whether the behavior is already covered by an existing codebook property;
- concise evidence supporting credit or blame.

The evidence text must:

- be one concise sentence;
- include only visual and historical context relevant to the property;
- prefer a generalizable description;
- avoid copying full history, captions, traces, or answers;
- avoid using an answer option as the explanation.

Recommended limit: 30–50 tokens.

### 4.5 Property-level aggregation

Aggregate all QA feedback for one property-source-video intervention:

\[
F_{i,j}
=
\operatorname{Aggregate}_{q\in Q_i}
F_{i,j,q}.
\]

Record:

- helped QA IDs;
- harmed QA IDs;
- positive evidence;
- negative evidence;
- retrieved and attributed segment IDs;
- source-video lineage;
- codebook coverage judgment.

Feedback may recommend `add`, `revise`, `merge`, `retire`,
`router_positive`, `router_negative`, or `no_op`. These are evidence-bearing
recommendations only; aggregation does not apply a component update.

Then aggregate all intervention feedback across the three-video batch:

\[
\mathcal F^{(k)}
=
\bigcup_{i=1}^{3}
\bigcup_{j=1}^{B_i}
F_{i,j}.
\]

### 4.6 Checkpoint 1: compact property-centric memory

Before this checkpoint, cross-iteration state retained the current codebook,
the router's cumulative accepted flip-supervision examples, provisional
lineage, coverage state, and confirmed/provisional pointers. Complete raw QA,
routing, caption, reasoning, proposal, retrieval, intervention, feedback, and
transition artifacts were persisted, but their useful observations were not
converted into a bounded property-level record. In particular, correct-QA
evidence, no-effect interventions, rejected or unpromoted candidate histories,
and the reasons for retaining representative examples were absent from the
next iteration's compact state.

Checkpoint 1 adds an observational sidecar flow without changing proposal,
feedback, codebook-update, or router-update decisions:

```text
immutable raw Phase 4 artifacts
→ property_compact_summary_v1
→ property_memory_v1
```

Raw artifacts remain immutable and authoritative. Every compact example cites
absolute artifact paths and SHA-256 hashes. The compact layer never replaces a
QA result, trajectory, caption, routing decision, intervention result, or
transition artifact.

For a correct, runtime-valid QA, an active property receives credit only when
the evidence establishes the complete chain:

```text
routed property
→ property-related caption information
→ downstream reasoning reused that information
→ correct answer
```

`strong` means the reasoning clearly reused the related caption evidence;
`weak` means the related evidence was present and reused but decisive use is
uncertain; `none` means routing or successful co-occurrence did not establish
the chain. Merely selecting a property never earns credit, and correct-QA
credit cannot create a candidate or a new property. The Checkpoint 1 default
analyzer is deterministic and conservative: it requires an actually used and
routed segment, property-to-caption content overlap, and caption-to-reasoning
content reuse. Exact caption reuse or at least two reused content tokens is
strong; one is weak; all other cases are none.

Every completed candidate intervention receives a deterministic effect summary
from all four transition counts. Any `wrong_to_correct` without a regression is
`positive`; any `correct_to_wrong` without an improvement is `negative`; both
is `mixed`; no flip is `no_effect`. This summary does not change current
intervention or feedback semantics.

One active-property record contains its ID/text, creation reason, intended
behavior, positive/harmful/no-effect examples, positive/negative routing
examples, aliases, revision history, and latest decision. Seed entries lacking
creation artifacts use an explicit `seed_or_legacy` origin; no supporting
evidence is invented. A candidate has a separate temporary memory containing
its proposal rationale, reusable instruction, coverage-related active IDs,
bounded effect examples, and artifact references. It remains separate unless
the existing updater has already produced an `add` or `merge` decision. Passing
that immutable update plan and resulting bank lets the sidecar record the
post-decision promotion; the memory layer cannot cause the decision.

Default per-property limits are three strong and two weak positive examples,
three harmful examples, two no-effect examples, and two positive plus two
negative routing examples. Candidate effect categories are bounded by the
corresponding limits. Selection is deterministic: evidence strength first,
then distinct source videos, then representative-signature diversity, with
recency only as the final tie-breaker. Each build writes a selection audit that
records why every considered example was retained or evicted. Eviction removes
only a compact memory entry; the raw artifact and immutable compact summary
remain available.

`CompactPropertyMemoryRunner` rebuilds from an optional parent
`property_memory_v1` snapshot, the current raw artifacts, the frozen input
codebook, optional already-decided update plan, and the fixed selection policy.
Its manifest binds all source hashes, parent hash, bank hashes, bounds, schema,
and selection policy. Exact matching completion resumes without rebuilding;
partial output, changed source hashes, or an incompatible parent/completed
schema fails closed. Completed legacy Phase 4 artifacts are read as raw sources
and are never reinterpreted or overwritten as memory artifacts.

## 5. Codebook and router update

Update the codebook and router once, after all property interventions in the
iteration finish.

Multiple properties may be accepted in one iteration. There is no
`max_new_entries_per_iteration=1` methodological constraint.

### 5.0 Checkpoint 2: memory-conditioned LLM codebook planning

Before this checkpoint, `aggregate_updates()` deterministically grouped
accepted correctness-flip feedback by normalized candidate text. It selected
`add`, `revise`, `merge`, `retire`, or `no_op` from feedback recommendations,
coverage IDs, positive/negative counts, and a two-video retirement threshold;
`apply_update_plan()` then changed the bank and router together. That path
remains available for legacy deterministic iterations and is not
reinterpreted as an LLM plan.

The new post-feedback checkpoint connects the bounded memory layer inside
`Checkpoint3EOrchestrator`:

```text
completed baseline/proposal/intervention/feedback artifacts
→ resolve property_memory_lineage_pointer_v1
→ CompactPropertyMemoryRunner
→ property_memory_v1 + current intervention summaries
→ memory-conditioned LLM codebook plan
→ deterministic action validation
→ candidate codebook + complete old-to-new ID mapping
```

This checkpoint is deliberately separate from the existing production
bank/router pair commit. It writes a candidate codebook only. It does not
change the router, run confirmation, advance coverage, or mutate confirmed or
active-provisional production pointers. The next router checkpoint must consume
the candidate codebook and ID mapping before any atomic pair commit.

The centralized system prompt is
`optimization/prompts/codebook_updater_v1.txt`, version
`memory_codebook_updater_prompt_v1`. Its request schema is
`memory_codebook_updater_request_v1` and contains only the current codebook,
bounded active-property memories, bounded candidate memories, current compact
intervention summaries, recent no-op/rejection summaries, and audit artifact
references. It never contains raw frames, unbounded captions, or full reasoning
histories.

The strict `memory_codebook_updater_response_v1` response contains zero or more
`add`, `revise`, `merge`, `preserve`, `retire`, or `no_op` actions. Every action
names target property/candidate IDs, optional proposed ID/text, concise
reasoning, supporting memory-example/evidence IDs, behaviors to preserve, and
confidence. The LLM never edits a snapshot directly.

Deterministic validation rejects actions independently when they cite unknown
properties, candidates, examples, or evidence; add without a completed positive
intervention; non-reusable or instance/non-visual knowledge text; revise/merge
that fails to cite and preserve existing positive behavior; invalid merge
lineage or canonical ID; retirement without harmful support across at least two
distinct video-or-iteration groups; conflicting mutations; ID/text collisions;
or malformed schema. Rejections are explicit and do not invalidate otherwise
independent valid actions.

Validated actions materialize `memory_candidate_codebook_v1` and
`property_id_mapping_v1`. The mapping contains every old property ID, mapping
unchanged entries to themselves, merged entries to the canonical ID, and
retired entries to null; candidate promotions are recorded separately. Merge
keeps source IDs and texts as aliases and combines bounded compatible memories.
Revise preserves the existing property ID, origin, positive examples, and
revision history. Candidate memory remains temporary unless a validated `add`
or `merge` promotes it.

Example:

```text
candidate has one positive intervention example
+ related active property has repeated positive memory
→ updater chooses revise instead of add
→ validator preserves existing behavior
→ candidate codebook and ID mapping are produced
```

### 5.1 Existing property already covers the behavior

If an accepted candidate is semantically covered by an existing property:

- do not add a duplicate codebook entry;
- use positive and negative intervention evidence to update router
  applicability for the existing property.

### 5.2 New useful behavior

Add a new property when:

- the current codebook does not express the behavior;
- the intervention has credible positive evidence;
- collateral regressions are acceptable;
- the instruction is reusable beyond the source QA wording.

### 5.3 Revise

Revise an existing property when it captures the correct concept but is too
broad, too narrow, ambiguous, or systematically induces unwanted captions.

### 5.4 Merge

Merge properties or proposals when they express the same reusable behavior.
When similar candidates originate from multiple videos, combine their
interventional evidence before deciding.

### 5.5 Retire

Retire an active property only when harmful or redundant behavior repeats
across sufficiently diverse evidence. A single negative flip should normally
produce negative router supervision rather than retirement.

### 5.6 No-op

Use `no_op` when evidence is weak, noisy, contradictory, or insufficiently
generalizable.

### 5.7 Router update

For each accepted or existing property, update the history-aware VLM router
using compact contextual evidence:

\[
(s_{i,t},h_{i,t})\rightarrow
\text{select or avoid property }p.
\]

Positive examples come from useful interventions. Negative examples come from
harmful or unnecessary interventions.

The updated codebook and router are provisional. The next iteration uses that
provisional policy for a new three-video batch. `correct_to_wrong` flips on the
current source videos are the regression signal; no separate regression-video
subset is reserved.

### 5.8 Coverage-cycle confirmation

After all eight evidence-pool videos have appeared since the last
confirmation, evaluate the accumulated provisional policy on both confirmation
videos. Confirmation videos do not propose properties and do not generate
optimization feedback. Acceptance saves a confirmed checkpoint. Rejection
restores the last confirmed checkpoint. Reset coverage only after that decision
is persisted. Validation and test are not used by this procedure.

Materialize the two videos, six QAs, temporally ordered segments, sampled-frame
paths and content hashes, transcripts, prompts, caption/runtime configuration,
history limits, fixed scaffold/contract versions, and DVD QA configuration as
one immutable confirmation input bundle. Parent and candidate must reference
that exact bundle and runtime hash. Caption both policies independently through
the sequential history-aware builder: they use identical history settings but
each policy's preceding-caption history is produced on-policy from its own
captions. Parent history is never copied into the candidate run.

Confirmation caption caches use the complete history-aware identity plus the
shared bundle hash. A cache hit therefore requires identical source/sampling,
frame bundle, composed prompt, bank/router/scaffold/contract versions, local
history, caption model/backend, decoding, and history configuration. Persist
both policies' per-segment cache keys and paths and fail before DVD QA if the
bundle/runtime references differ or unequal identities resolve to one path.

The initial deterministic acceptance criterion requires all six confirmation
QAs to finish without evaluation errors, zero `correct_to_wrong` transitions,
and candidate mean accuracy greater than or equal to the previous confirmed
policy. Acceptance promotes the bank/router pair atomically. Any failed
criterion restores both exact parent-confirmed versions; a partial component
promotion is invalid.

## 6. Full iteration summary

\[
\boxed{
\begin{aligned}
&\text{snapshot current policy}\\
&\rightarrow \text{select and full-caption three evidence videos once}\\
&\rightarrow \text{run all baseline QAs}\\
&\rightarrow \text{propose multiple properties per video}\\
&\rightarrow \text{property-wise source-video retrieval}\\
&\rightarrow \text{independent selective interventions in parallel}\\
&\rightarrow \text{rerun all source-video QAs}\\
&\rightarrow \text{flip-only interventional feedback}\\
&\rightarrow \text{one provisional iteration-level codebook/router update}\\
&\rightarrow \text{confirm after full eight-video coverage}
\end{aligned}
}
\]

## 7. Active invariants

- The router is a history-aware lightweight VLM.
- The router is query-independent.
- The scaffold/composer is fixed.
- One iteration full-captions exactly three unique evidence videos once under
  the current policy.
- Eight train videos form the evidence pool and the other two form confirmation.
- Coverage prioritizes unused evidence videos and triggers confirmation after
  all eight have appeared.
- Every video runs all of its baseline QAs.
- One video may propose multiple candidate properties.
- Every candidate property is evaluated in its own source-video intervention.
- Property retrieval uses only exact candidate property text and sampled
  segment frames; frozen local history is not a retrieval input.
- Candidate properties are force-added to retrieved segments, not routed as if
  already trained.
- All candidate interventions share frozen incumbent history.
- All candidate interventions are mutually independent and parallelizable.
- All source-video QAs are rerun for every property-specific mixed view.
- Feedback uses
  \(S_{\mathrm{sim}}\cap(S_{\mathrm{used}}\cup S_{\mathrm{usedagain}})\).
- Only correctness flips enter optimization feedback.
- Codebook and router update once after the full batch.
- Multiple properties may be accepted per iteration.
- Iteration updates remain provisional until coverage-cycle confirmation.
- There is no separate regression-video subset; source-video
  `correct_to_wrong` flips provide regression evidence.
- Confirmation videos never generate proposals or optimization feedback.
- Main experiments are executed manually by the user.

Frozen local history is used only for baseline routing, baseline captioning,
later selective re-captioning, and later multimodal feedback.

## 8. Deferred work

Do not implement before the first main result:

- propagating recaptioned history through later segments;
- evaluating every candidate on unrelated videos before codebook acceptance;
- per-segment downstream QA rollouts;
- SLM-only routing or router distillation;
- scaffold optimization;
- trust-region logic;
- broad file/class renaming;
- automatic main-experiment execution by the coding agent.
- feeding compact memory into an LLM codebook updater;
- changing the router prompt or router-update policy to consume memory;
- changing property-proposal semantics based on accumulated memory.
