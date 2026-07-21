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
- a configurable number `K` of evidence videos per optimization iteration,
  with `K=3` as the conservative pilot default and three QAs per video;
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

History-aware caption output uses `caption_plain_text_output_contract_v2`.
Qwen is asked for only a non-empty plain-text visual description without a
sentence-count limit. It does not generate the persistent JSON envelope. The harness
validates the exact raw text and deterministically serializes it as
`{"clip_description": text}`. JSON-looking text is not interpreted or repaired;
if non-empty and otherwise valid, it remains literal caption text. Blank text
and deterministic degeneration fail closed.

The output contract text/hash, plain-text parser/normalization version,
repetition-guard version, and decoding policy/hash participate in segment-state,
caption-cache, baseline, intervention, and confirmation identities. Completed
legacy JSON cache artifacts are never reinterpreted or overwritten.

The runtime free-form prompt generator also uses a single-string plain-text
output contract (`free_form_instruction_request_v2_plain_text` and
`free_form_instruction_plain_text_parser_v2`). The model returns only the
non-empty segment-specific captioning instruction. The parser strips surrounding
whitespace and preserves the remaining response literally; it does not parse
JSON, unwrap Markdown fences, extract fields, or repair near-JSON. Its template
text/hash and parser version remain part of the generator and caption-cache
identity, so earlier JSON-instruction runs cannot alias this contract.

Caption parsing uses `caption_plain_text_parse_result_v2` and
`caption_repetition_guard_v1`. Three consecutive canonically identical
sentences are rejected. For outputs of at least 48 word tokens, at least four
occurrences of an eight-token n-gram covering 80 percent or more of all word
tokens are rejected as majority repetition. These thresholds deliberately
avoid rejecting ordinary local repetition. A model call whose raw response is
invalid is followed by at most five retries with identical frames,
transcript, composed property instruction, frozen history, decoding, and
output contract. Thus one segment has at most six generation calls: the
initial attempt plus five retries. Runtime/backend exceptions still propagate
immediately. Every attempt retains its raw output, parse classification,
elapsed time, and attempt index in `history_aware_caption_cache_v5`. The first
valid parse ends the loop; five exhausted retries leave `parsed={}` and the
existing baseline/intervention fail-closed behavior applies.

Caption decoding is `temperature=0`, `max_tokens=1024` newly generated tokens,
and `repetition_penalty=1.05` under
`qwen_caption_plain_text_decoding_v2`. The changed decoding and contract hashes
force a new cache directory, so earlier invalid or valid JSON-caption caches do
not alias the plain-text path. Baseline, selective intervention, and paired
confirmation use the same policy. Only immutable frame/transcript preprocessing
may be shared across the old and new runs; prompts, captions, histories, and DVD
QA evidence are recomputed under the new identity.

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

The persistent Qwen backend disables vLLM's multimodal processor LRU and
prefix cache (`mm_processor_cache_gb=0`, `enable_prefix_caching=False`). In
vLLM 0.11.x the frontend metadata LRU and EngineCore multimodal-feature LRU can
evict at different rates during hundreds of unique image requests. A later
metadata hit can then send no image payload for an EngineCore miss and kill the
EngineCore input thread with `Expected a cached item for mm_hash`. Disabling
both reuse paths makes every request carry its images. Policy
`qwen25_vl_mm_cache_disabled_v1`, both values, and the versioned backend ID are
part of caption/router/run identity; pre-policy incomplete runs fail closed and
must use new output/state roots.

DVD `frame_inspect_tool` vision calls are raw-VLM tasks on this same pool. While
the pool is active, the parent installs a lightweight `dvd_backend` captioner
proxy and must never lazily initialize another local Qwen/vLLM engine. Without
an active pool, the original standalone `dvd_backend.get_captioner()` behavior
is unchanged.

The DVD tool-calling boundary publishes an exact nested JSON schema for
`time_ranges_hhmmss`: a non-empty array of two-item arrays whose endpoints are
`HH:MM:SS` strings. Numeric seconds are never passed to the vendored DVD
implementation. If a backend nevertheless emits invalid arguments, the
instrumentation boundary records the rejected, non-executed call and returns
one deterministic correction message to the agent. A second invalid call fails
closed; there is no argument coercion, broad QA retry, JSON repair, or fallback.
The contract and one-retry limit are part of downstream QA execution identity.

Downstream DVD caption-database retrieval is also fixed across policies.
Every `clip_search_tool` call executes with `top_k=16`, even if the tool-calling
model supplies another value or omits it. The raw requested arguments remain
in the trajectory, while `tool_events.jsonl` records the normalized executed
arguments and the override policy. This setting is separate from Phase 4
property-frame retrieval and is part of QA cache/resume identity.

The parallel fresh prompt-delta proposer uses
`trajectory_grounded_normalized_catalog_v3_localized_inspection`. Prompt-delta
proposals are grounded only in video segments localized by assistant timestamp
citations or non-global frame-inspection calls. A frame-inspection call whose
start and end are within the configured boundary tolerance of the video start
and end is classified as global at the call level; its segments remain audit
provenance but do not enter proposal/intervention scope. These segments indicate
evidence exposure and localization, not causal attribution. The proposer never
uses legacy `explicitly_cited_segments`, `used_segments`, consumed, retrieved,
or returned segments as selection scope. Included histories are
represented losslessly by content-addressed snapshot and history-item catalogs;
segments appear once and QAs retain their ordered
`intervention_candidate_segment_refs` plus stored semantic trajectory evidence.
A QA without localized evidence is recorded as `no_localized_evidence` and does
not trigger a proposal call. The whole-video normalized request is used only
when exact provider token preflight fits. Otherwise it splits deterministically
into one complete request per eligible QA. A single-QA overflow records that QA
as `context_ineligible`, makes no provider call for it, and continues with other
eligible QAs. If none remain, the proposal stage ends normally with
`no_eligible_proposal_evidence`. No caption, history, trajectory, or segment is
truncated or sampled to fit context.
Rejected frame-inspection calls remain in the selection audit as
`invalid_not_executed` and do not localize any segment.

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

Load `K` unique videos from the eight-video evidence pool:

\[
\mathcal V^{(k)}
=
\{v_1,\ldots,v_K\}, \qquad 1\leq K\leq 8.
\]

The policy snapshot remains fixed throughout the iteration. Selection rotates
deterministically through the evidence pool and prioritizes videos not yet used
since the last confirmation. If fewer than `K` unused videos remain, fill the
batch from the deterministic rotation without duplicates. An explicit ordered
list is allowed only after validation against the same evidence pool. Once all
eight have appeared, confirmation is due and no new evidence iteration starts
until the confirmation decision is persisted.

### 3.1 Full-caption the selected videos once

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
exploratory candidate properties for each video:

\[
\mathcal C_i
=
\{p_{i,1},\ldots,p_{i,B_i}\}.
\]

A video may generate multiple candidates because every eligible QA contributes
one or two executable intervention candidates, regardless of baseline
correctness. Incorrect-QA candidates may recover missing or poorly represented
evidence. Correct-QA candidates may preserve useful evidence or improve clarity,
temporal order, robustness, concision, relevance, and suppression of speculative
or repetitive content. The provider identifies each candidate with a source-free
`qa_slot`; private lineage maps it back to exactly one source question and its
baseline correctness. These candidates are hypotheses only; retrieval,
intervention, feedback, and the existing updater still decide whether to add,
revise, merge, preserve, or reject them.

Each candidate must record its lineage:

- a deterministic opaque proposal handle, independent of any model-suggested
  codebook name;
- the model's readable `suggested_property_id` as a non-binding hint;
- source video ID;
- exactly one source QA ID and its baseline correctness for new proposals;
- natural-language `failure_analysis` provenance;
- non-binding current-codebook coverage hints;
- proposed property instruction;
- an atomic applicability contract containing non-empty `when`, one or more
  observable `positive_cues`, optional `negative_cues`, and one or more
  `required_modalities` from `frames`, `transcript`, and `caption_history`.

Candidates are not added to the codebook yet.

Property-proposal model input preserves all three QA outcomes but contains no
source-video ID, question ID, segment ID, provider priority rank, tool-call ID,
or payload-truncation metadata. For each QA it contains the question and answer
choices, ground truth and baseline prediction, at most three bounded sanitized
reasoning events, and the complete time-ordered union of
`explicitly_cited_segments` and `frame_inspected_segments` as private source
provenance. The provider payload identifies intervals by `[start–end]` timestamp
ranges and never by stable segment ID. Explicitly cited intervals are packed
first with fuller normalized `generated_description` and non-duplicate
`transcript`; inspected-only intervals follow as deterministic single-line
`[start–end] description | tx: transcript` summaries, with description and
transcript bounded to 250 and 120 characters. Empty or substantially duplicate
transcripts are omitted. Within each class ordering is deterministic by QA and
timestamp.

The evidence text budget is capped at 200,000 characters and further reduced
to leave room beneath the existing 250,000-character envelope for the task,
schema, QA context, and JSON structure. Once the budget is exhausted, remaining
intervals are omitted from the provider payload only. Their complete original
caption, generated-description/transcript split, stable segment ID, timestamp,
evidence role, representative-frame lineage, hashes, inclusion decision, and
omission reason remain immutable in `input_identity.json`; aggregate and per-QA
included/omitted counts are also written to `evidence_packing.json` and logged.
`returned_segments`, general consumed/used segments, and count-based fallback
sampling do not enter proposer evidence.

The captioner's available evidence contract is current frames, current
transcript, and preceding caption history. A property may request preservation,
integration, or expression of information present in those modalities, but may
not require unavailable tools or inputs. `failure_analysis` is provenance and
is never injected as a caption or routing condition.

The current codebook and a source-free output schema complete the multimodal
request. Provider-reported `covered_by_existing_property_ids` are non-binding
hints only and are stored internally as `coverage_hints`. A non-empty hint list
does not reject a proposal. Pre-intervention parsing rejects only structurally
unusable output: invalid or missing schema fields, empty instructions, unresolved
placeholders, exact duplicates for the same QA, invalid source slots, or an
instruction requiring tools/inputs unavailable to the captioner. It does not
veto correct-sample candidates, lexical QA overlap, active-property similarity,
uncertain benefit, weak generality, or possible repetition. Applicability retains
its strict closed structural schema. `failure_analysis` replaces fixed
failure-type taxonomies. The private source QA ID and baseline correctness are
reattached after parsing. The text portion and every transformed
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

The proposer returns `suggested_property_id`, but orchestration never uses that
name as proposal identity. After parsing, it derives
`candidate_<sha256-prefix>` from the private source-video/baseline lineage and
normalized property text, canonical applicability, source QA, and baseline
correctness under `opaque_candidate_proposal_id_v3`. This handle
is stable for exact resume and differs across source videos. The updater still
chooses the final active property ID and may add, revise, or merge candidates.
Legacy raw artifacts with colliding readable IDs are deterministically migrated
to opaque temporary handles in compact memory while preserving the original ID
as audit lineage and a non-binding suggestion.

The request records a deterministic per-QA proposal requirement. Every eligible
correct or incorrect QA requires one or two intervention candidates. Existing
coverage and uncertain downstream value are recorded for later review rather
than used to suppress a candidate. Strict zero proposal is allowed only when no
QA is eligible because of runtime failure, missing focused evidence, malformed
input, or clearly unreliable annotation. Falling below the required per-QA
candidate count fails the proposal stage rather than silently completing.

`missing_qa_slot_retry_v1` makes this boundary robust to a partial provider
response. After the initial response, orchestration identifies only eligible
`source_qa_slot` values with zero candidates and issues at most two bounded
follow-up requests containing only those QA rows and their original evidence.
Already completed slots and accepted proposal text are never regenerated. Each
retry request, exact provider body, raw response, and row decision is preserved
under `missing_slot_retries/`. Successful rows are combined without rewriting
their instructions. If a required slot remains absent after two retries, the
same cardinality failure remains fail-closed. Exact resume reuses completed
initial and retry calls.

The tested canonical property is not generalized or rewritten before selective
re-captioning. `work_item.json`, `transitions.json`, and `result.json` retain the
original proposal, source QA lineage, baseline correctness, candidate QA outputs,
the four semantic correctness transitions, and explicit `runtime_failure` status.
`property_compact_summary_v3_runtime_validity` carries this bounded provenance
into candidate memory and `memory_codebook_updater_request_v4`.
Only the updater may promote a tested candidate into the active codebook.
Applicability and `failure_analysis` persist through proposal
artifacts and resume, feedback requests, candidate memory, the bounded codebook-
updater request, candidate codebook provenance, and identity hashes. Router-rule
interpretation of applicability is intentionally deferred.

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

The exact normalized candidate property text is the only text query. Before
ranking, intersect the visual-index segment IDs with the stable segment-ID
universe materialized by the frozen baseline caption view. This structural
allowlist uses no caption content: it prevents sampled tail frames or another
index-boundary artifact from creating an `S_sim` segment that cannot replace
an incumbent caption. SigLIP scores sampled frames in the surviving segments;
maximum frame pooling produces the segment score. Select the top `M` only
after this intersection, by descending score, ascending segment start, then
stable segment ID. Retrieval must not use frozen history, caption content,
questions, answer choices, answers, correctness, reasoning traces, or
used-segment references.

`valid_baseline_caption_segment_intersection_v2` defines the retrieval universe
from the actual non-empty keys in the frozen `captions.json`, not from all
segments merely scheduled for captioning. A source segment whose caption parse
failed therefore cannot enter `S_sim` or a mixed-view replacement. The
allowlist preserves source order and its hash/count remain retrieval identity.

`property_retrieval_top_k` defaults to 5. Persist every frame score and every
ranked eligible segment, not only the selected segment IDs. Also persist the
baseline segment-universe hash/count and the visual-index segment IDs excluded
by the intersection. The universe policy and hash participate in retrieval and
resume identity.

The production launcher may place the shared SigLIP embedder on an explicit
dedicated GPU. When `--embedding-gpu` is provided, it must be a valid free
physical GPU outside the persistent Qwen `--gpus` set. SigLIP runs in its own
spawned process with `CUDA_VISIBLE_DEVICES=<physical GPU>` set before child
bootstrap and always addresses that GPU internally as `cuda:0`. The parent DVD
backend's visible-device mutation therefore cannot turn a physical ID into an
invalid logical ordinal. The child process, physical GPU, logical device, PID,
and release state are persisted. The embedding execution mode is resume
identity but not visual-index semantic identity, so a complete
model/sampling/source-compatible frame index remains reusable across CPU/GPU
execution. Requests are serialized through the child connection.

DVD caption-database and query search use a separate CPU BGE model;
`--embedding-gpu` continues to control only property-retrieval SigLIP. The
production launcher synchronously preloads the shared DVD BGE instance in the
parent under a process-local lock before any parallel video-QA wave. This
prevents concurrent first-use from constructing a partial SentenceTransformer.
`dvd_bge_parent_preload_v1` participates in execution and resume identity, and
a preload failure stops the run before the parallel wave.

DVD itself also exposes process-global mutable prompt overrides,
`VIDEO_FPS`, and instrumentation function bindings. Parallel evidence-video
threads therefore do not execute DVD QA tool loops concurrently. Under
`serialized_dvd_qa_execution_v1`, each QA exclusively performs prompt reset,
per-video effective-FPS assignment, database/agent construction, instrumented
tool execution, and instrumentation removal. The generated database FPS is
checked against the effective FPS before the agent runs. Captioning and
property intervention remain GPU-parallel; only downstream DVD QA is serial.
This policy is part of run, cache, and resume identity.

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

### 3.4.2 Human-readable operational stage log

The production launcher appends a thread-safe human-readable operational log
to `<output_dir>/iteration.log` and mirrors every line to stdout. It records the
iteration ordinal, deterministic video waves and GPU assignments, and per-video
stage boundaries for baseline captioning, baseline QA, property proposal,
SigLIP similarity retrieval, intervention recaption, candidate QA, flip-only
feedback, property-memory update, LLM codebook update, LLM router update, and
atomic pair completion. Successful ends include bounded result summaries;
failures include exception type/message; exact resume emits `RESUME` and makes
no model-call claim.

Parallel workers share one logger lock, so each timestamped line is complete
even when video stages overlap. This `.log` is mutable operational telemetry,
not immutable scientific evidence, and is excluded from cache keys and artifact
hash closure. The immutable raw stage artifacts remain the source of truth.

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
within one intervention may also run in parallel. Iteration size `K` and video
parallelism `P` are independent. Selected videos are partitioned into ordered
deterministic waves of at most `P` videos, and each wave position maps to one
of the iteration-scoped persistent GPU workers. Thus `K=8, P=4` runs two
four-video waves without changing selected-video or result ordering.

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

Then aggregate all intervention feedback across the `K`-video batch:

\[
\mathcal F^{(k)}
=
\bigcup_{i=1}^{K}
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

Every candidate intervention receives a deterministic effect summary from the
four semantic transition counts. Any `wrong_to_correct` without a regression is
`positive`; any `correct_to_wrong` without an improvement is `negative`; both
is `mixed`; no flip is `no_effect`. QA runtime failures and intervention
execution failures are instead counted as `runtime_failure` and
`intervention_failure`, retained for reliability review, and never converted
to `wrong_to_wrong`, harmful evidence, or any other semantic effect. A summary
with failures but no valid semantic transition is `unavailable`. This summary
does not change current feedback triggering semantics.

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
`memory_codebook_updater_prompt_v3_deterministic_ids`. Its request schema is
`memory_codebook_updater_request_v4` and contains only the current codebook,
bounded active-property memories, bounded candidate memories, current compact
intervention summaries, recent no-op/rejection summaries, and audit artifact
references. It never contains raw frames, unbounded captions, or full reasoning
histories. The prompt explicitly makes the LLM the semantic decision-maker:
correctness flips are important but not exclusive evidence, and grounding,
temporal coherence, caption improvements without flips, robustness, harmful
side effects, and evidence reliability must be considered holistically.

The strict `memory_codebook_updater_response_v2` response contains zero or more
`add`, `revise`, `merge`, `preserve`, `retire`, or `no_op` actions. Every action
names target property/candidate IDs, optional proposed text, concise
reasoning, supporting memory-example/evidence IDs, behaviors to preserve, and
confidence. It never proposes a new active property ID. A validated `add`
receives `pe_<20-hex-hash>` deterministically from the candidate identity and
normalized proposed text under `deterministic_candidate_property_id_v1`.
`revise` preserves its sole target identity, and the first target of a `merge`
is canonical. The LLM never edits a snapshot directly.

The real provider uses `openai_strict_component_update_v3`: an updater-specific
strict JSON Schema constrains the complete response envelope and every action,
rather than merely requesting a syntactically valid JSON object. The codebook
updater preserves each raw response beneath `attempts/attempt_NNN/` and retries
only strict parse/envelope failures, at most three total attempts. Runtime/API
failures are not converted into plans. `input_identity.json` makes an
interrupted retry exactly resumable: an already persisted raw attempt is parsed
again but never sent to the provider again. Older incomplete directories that
predate this identity fail closed and are not reinterpreted.

`memory_codebook_structural_validation_v2` asks only whether a plan is
well-formed, properly referenced, internally consistent, and executable. It
rejects malformed schema/IDs, unknown properties/candidates/examples/evidence,
missing mutation provenance, impossible action shapes or merge lineage,
conflicting mutations, placeholders, bounds violations, and exact ID/text
collisions. It does not decide whether an intervention is positive enough,
require a correctness flip or a particular evidence polarity, compare token
overlap, require verbatim preservation language, or impose a multi-video retire
threshold. Former semantic text/support heuristics are non-blocking warnings.
Per-action structural errors do not invalidate otherwise independent valid
actions, and the immutable audit records separate the LLM plan, structural
errors, warnings, and final applied plan.

Validated actions materialize `memory_candidate_codebook_v2` and
`property_id_mapping_v2`. The mapping contains every old property ID, mapping
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

### 5.0.1 Checkpoint 3: memory-conditioned router prompt and atomic pair

Before Checkpoint 3, deterministic `apply_update_plan()` could append compact
`supervision_examples` to router configuration, but the real history-aware VLM
router still used one hard-coded instruction prefix. Accumulated supervision
therefore did not define a separately versioned, rendered inference prompt.

Checkpoint 3 consumes only the validated candidate codebook, complete
old-to-new ID mapping, bounded routing memory/current intervention evidence,
and validated codebook actions:

```text
candidate codebook + ID mapping + bounded routing evidence
→ memory_router_updater_prompt_v2_semantic_llm
→ strict memory_router_updater_response_v1
→ structural action validation
→ structured_router_policy_v2_total_examples
→ rendered_router_prompt_v1
→ atomic_provisional_policy_pair_v1
```

The protocol scaffold remains fixed: routing is query-independent; inputs are
current frames and bounded preceding caption history; only active property IDs
may be returned; the selection limit, strict one-key JSON schema, fail-closed
parser, and no-fallback behavior remain unchanged. Editable per-property state
contains `selection_guidance`, `avoidance_guidance`, demonstrated examples,
aliases, and remapped source IDs. The implementation enforces one total budget
of four examples across both stored labels; this is a prompt-size constraint,
not a semantic polarity rule, and deterministic compression balances labels
where possible. `prompt_routing/structured_router_policy.py` renders this
state deterministically. The resulting text is stored in the candidate
`RouterPolicySnapshot.configuration`, and `HistoryAwareVLMRouter` uses that
exact text in later calls. Its schema/renderer versions and SHA-256 enter the
router snapshot, request identity, decision artifact, provisional lineage, and
therefore routing-dependent resume/cache identity.

The centralized router-updater prompt is
`optimization/prompts/router_updater_v1.txt`, version
`memory_router_updater_prompt_v2_semantic_llm`. Actions may set selection or avoidance
guidance, add bounded positive or negative examples, preserve, or no-op. The
LLM interprets complete bounded routing/intervention evidence and makes the
semantic decision. The structural validator rejects unknown candidate-codebook
IDs, unknown memory/evidence IDs, unsupported or conflicting mutations,
empty/placeholder/bounded text, exact duplicates, stale mappings, or
prompt/protocol changes. It does not enforce evidence polarity,
target/evidence semantic equivalence, or reject guidance merely because
ordinary words such as `question` or `answer` appear.
Codebook merges remap old guidance and aliases to one canonical ID;
retirements remove obsolete guidance before LLM actions.

The router provider uses the corresponding updater-specific strict JSON Schema
and the same three-attempt parse-retry boundary. A missing or null
`target_property_id`, missing response version, extra field, or malformed action
is therefore rejected before structural validation. Every raw
attempt and parse error is immutable and exact resume skips already completed
provider calls. By default, zero routing evidence still reaches the LLM so its
explicit no-op judgment is recorded.

`memory_conditioned_llm_router_updater_v4_structural_only` makes the former
zero-evidence shortcut an explicit configuration rather than a hidden semantic
conclusion. The default uses `llm_provider_no_evidence`; an empty action list is
the expected conservative response but is chosen by the LLM. A caller may opt
into `skip_llm_when_no_evidence=True` for cost or legacy compatibility; that
path records `configured_empty_evidence_skip` and
`explicit_skip_llm_when_no_evidence`, then still validates the empty plan,
applies deterministic ID remapping, renders/hashes the prompt, and participates
in the atomic provisional pair. Empty or stale target IDs remain parser or
structural-validator errors rather than being normalized.

Persistence is two-phase. Router request/response/validation/rendering
artifacts may exist without a committed pair, but a codebook is not made active
alone. Only after the candidate bank, candidate router, active ID space,
rendered prompt content/hash, and both updater validation reports agree does
`memory_conditioned_atomic_policy_pair_v1` write one
`atomic_provisional_policy_pair_v1` and the separate
`memory_conditioned_provisional_pair_pointer_v1`. Any provider, validation,
rendering, hashing, or persistence failure records `failure.json`, leaves that
pointer absent, and retains the parent pair. This checkpoint never writes
`confirmed/current.json` or coverage-cycle `active_provisional.json` and does
not run confirmation.

Example:

```text
property is useful across multiple videos
+ one harmful case is context-specific
→ codebook property is preserved
→ router updater adds avoidance guidance
→ next iteration uses the newly rendered router prompt
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
provisional policy for a new `K`-video batch. `correct_to_wrong` flips on the
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
&\rightarrow \text{select and full-caption K evidence videos once}\\
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
- One iteration full-captions `K` unique evidence videos once under the current
  policy; `K=3` is a default pilot setting, not a method constraint.
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
- changing property-proposal semantics based on accumulated memory.
