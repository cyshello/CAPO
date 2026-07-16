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

## 5. Codebook and router update

Update the codebook and router once, after all property interventions in the
iteration finish.

Multiple properties may be accepted in one iteration. There is no
`max_new_entries_per_iteration=1` methodological constraint.

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
