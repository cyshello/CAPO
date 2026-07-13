# CLAUDE.md

## Project title

**Selective Surrogate Rollout for GEPA-Based Caption-Prompt Optimization in DVD**

---

## 1. Objective

This repository extends DVD with prompt optimization for the **captioner prompt**.

Use the **official GEPA implementation** as the optimizer. GEPA owns the
optimization loop, including reflection, candidate proposal, evolutionary
search, candidate management, Pareto-aware selection, and stopping behavior.

Do not implement a separate custom optimizer.

The research contribution implemented in this repository is a new way to
evaluate GEPA candidates in caption-based long-video agents:

1. **Full rollout evaluation**
   - Re-caption every segment of the evaluation video under each candidate
     captioner prompt.
   - Run the complete DVD agent using those captions.

2. **Selective surrogate rollout evaluation**
   - Identify video segments referenced by the incumbent DVD trajectory.
   - Re-caption only those segments under each candidate captioner prompt.
   - Reuse cached incumbent captions for all other segments.
   - Run the same complete DVD agent over the resulting temporary caption view.

The optimizer, data, models, candidate budget, metric, and seeds must be held
constant. The intended difference between the two conditions is only the
candidate rollout evaluator.

Optional later extension:

3. **Localized multimodal feedback**
   - Give GEPA feedback visual frames sampled from the referenced segments.
   - Compare against text-only trajectory feedback.

Do not implement the multimodal extension until full and surrogate rollout
evaluation are correct and reproducible.

---

## 2. Research questions

### RQ1: Surrogate fidelity

For the same set of GEPA candidate prompts, does selective surrogate rollout
preserve the candidate ordering produced by full rollout?

Primary outcomes:

- Spearman rank correlation;
- Kendall rank correlation;
- pairwise ranking accuracy;
- top-1 agreement;
- top-k recall;
- score estimation error;
- fraction of segments re-captioned;
- captioning time and total time saved.

### RQ2: Optimization outcome

Under the same GEPA budget, does surrogate-rollout GEPA find a captioner prompt
whose full-rollout held-out QA performance is comparable to the prompt found by
full-rollout GEPA?

Primary outcomes:

- final held-out QA accuracy;
- optimization trajectory;
- total segments re-captioned;
- captioning GPU time;
- downstream LLM calls and token use;
- wall-clock time.

### RQ3: Multimodal feedback, optional

Does adding visual frames from trajectory-referenced segments improve GEPA
feedback and final prompt quality over text-only feedback?

This is not part of the first implementation milestone.

---

## 3. Attribution and claim boundaries

Use the following attribution consistently in code comments, experiment names,
and paper drafts:

- **Optimizer:** official GEPA.
- **System being optimized:** the DVD captioner prompt.
- **Our method:** selective surrogate rollout for evaluating GEPA candidates.
- **Optional extension:** localized multimodal feedback.

Do not present the following as contributions:

- generating textual feedback with an LLM;
- reflecting over trajectories;
- proposing prompt revisions;
- maintaining candidate populations;
- Pareto-aware candidate selection;
- iterative prompt optimization.

Those are delegated to GEPA.

A suitable method description is:

> We use GEPA as the underlying reflective prompt optimizer and replace its
> expensive full-video candidate evaluator with a selective surrogate evaluator
> that re-captions only trajectory-referenced segments.

---

## 4. Mandatory use of official GEPA

Use the official `gepa` package and its public API.

The GEPA API may differ across installed versions. Before writing integration
code:

1. inspect the installed package version;
2. inspect the package's official examples and type signatures;
3. identify whether the installed version expects:
   - the universal `optimize_anything` API;
   - a `GEPAAdapter` implementation;
   - a DSPy `GEPA` teleprompter;
   - another current official integration surface;
4. choose the smallest official API that supports:
   - a text candidate representing the captioner prompt;
   - custom evaluation on DVD;
   - per-example scores;
   - actionable textual feedback or side information;
   - access to candidate and optimization logs.

Do not guess GEPA APIs from memory.

Do not vendor, fork, or reimplement GEPA internals unless explicitly requested.

If the official GEPA integration is blocked, stop and report:

- installed GEPA version;
- attempted official API;
- exact incompatibility;
- smallest required repository change.

Do not silently replace GEPA with a homemade best-of-k loop.

---

## 5. Experimental data protocol

Split by **video**, not by individual QA.

Questions from the same video must not appear in different splits because they
share visual content and caption databases.

Use three logical splits:

- `train`: GEPA reflection and prompt evolution;
- `validation`: candidate evaluation and selection during optimization;
- `test`: final frozen-prompt evaluation only.

Never use the test split during GEPA optimization.

If only two sets of 30 QAs currently exist:

- treat the first as `train`;
- treat the second as `validation`;
- reserve another held-out subset for final testing.

Do not call repeatedly accessed candidate-selection data `test`.

Persist split manifests with exact video and question IDs.

Recommended initial mini-batches:

- reflection mini-batch: one video with three QAs;
- candidate evaluation mini-batch: configurable, preferably at least two videos
  or six QAs when affordable.

A one-video, three-QA evaluation batch is acceptable for a smoke test but is too
noisy for the final comparison.

---

## 6. Existing DVD behavior must remain intact

Before making changes:

1. locate the DVD entry point;
2. run one complete baseline QA;
3. save:
   - final prediction;
   - correctness;
   - complete trajectory;
   - tool calls and tool outputs;
   - retrieved clip identifiers;
   - caption files;
   - database/index files;
   - token usage;
   - timing;
4. identify all functions that:
   - generate captions;
   - build caption databases;
   - embed or index captions;
   - retrieve clips;
   - summarize evidence;
   - invoke the orchestrator;
   - parse the final answer.

Do not broadly refactor DVD.

Prefer adapters and thin wrappers around existing functions.

The baseline should still run unchanged after integration.

---

## 7. Core terminology

Use these terms consistently:

- `seed_prompt`: the initial DVD captioner prompt supplied to GEPA.
- `incumbent_prompt`: the currently selected GEPA candidate.
- `candidate_prompt`: a prompt proposed by GEPA.
- `incumbent_caption_cache`: captions generated under the incumbent prompt.
- `full_candidate_cache`: captions for every segment generated under one
  candidate prompt.
- `mixed_candidate_view`: a temporary caption view containing candidate captions
  for selected segments and incumbent captions elsewhere.
- `referenced_segments`: segments demonstrably used or inspected in an incumbent
  DVD trajectory.
- `expanded_reference_set`: referenced segments plus configured temporal
  neighbors or other explicitly logged additions.
- `full_rollout_score`: score obtained using a full candidate caption database.
- `surrogate_rollout_score`: score obtained using a mixed candidate view.

A `mixed_candidate_view` is temporary. It must never be stored as though all its
captions were generated under the candidate prompt.

---

## 8. Suggested repository structure

Adapt this structure to the existing repository. Reuse existing modules where
possible.

```text
surrogate_rollout/
    gepa/
        dvd_evaluator.py
        feedback.py
        integration.py
    rollout/
        base.py
        full.py
        surrogate.py
    references/
        extractor.py
        expansion.py
    cache/
        caption_cache.py
        mixed_view.py
    evaluation/
        qa_metrics.py
        ranking_metrics.py
        cost_metrics.py
    schemas.py
    config.py

scripts/
    inspect_dvd_baseline.py
    compare_rollout_fidelity.py
    run_gepa_full.py
    run_gepa_surrogate.py
    evaluate_frozen_prompts.py
```

Do not duplicate the DVD agent loop in separate full and surrogate scripts.

Both modes must call the same DVD reasoning function.

---

## 9. Required typed records

Use dataclasses, Pydantic models, or equivalent typed records.

```python
@dataclass(frozen=True)
class QAExample:
    video_id: str
    question_id: str
    question: str
    ground_truth: str


@dataclass(frozen=True)
class CaptionCacheKey:
    video_id: str
    segment_id: str
    prompt_hash: str
    caption_model_id: str
    decoding_hash: str
    source_hash: str


@dataclass
class DVDRunResult:
    prediction: str
    parsed_answer: str | None
    score: float
    trajectory: list[dict]
    retrieved_segments: set[str]
    consumed_segments: set[str]
    referenced_segments: set[str]
    token_usage: dict[str, int]
    latency_seconds: float
    errors: list[str]


@dataclass
class CandidateEvaluation:
    prompt: str
    prompt_hash: str
    rollout_mode: Literal["full", "surrogate"]
    aggregate_score: float
    examples: list[DVDRunResult]
    recaptioned_segments: int
    total_segments: int
    caption_cache_hits: int
    caption_cache_misses: int
    caption_seconds: float
    reasoning_seconds: float
    fallback_reasons: list[str]
```

Exact field names may follow repository conventions, but the same information
must be retained.

---

## 10. Evaluator boundary

GEPA must interact with DVD through one evaluator abstraction.

Conceptually:

```python
class DVDCandidateEvaluator(Protocol):
    def evaluate(
        self,
        candidate_prompt: str,
        examples: Sequence[QAExample],
        *,
        incumbent_prompt: str | None,
        reference_map: Mapping[str, set[str]] | None,
    ) -> CandidateEvaluation:
        ...
```

Provide two implementations:

```python
FullRolloutEvaluator
SelectiveSurrogateRolloutEvaluator
```

The GEPA integration layer must not contain separate optimization logic for each
rollout mode.

Switching the evaluator should be a configuration change.

---

## 11. Caption cache correctness

Caption cache isolation is essential.

Each cached caption must be keyed by at least:

- video ID;
- exact segment ID or timestamp range;
- complete captioner-prompt hash;
- caption-model identifier and revision;
- decoding configuration;
- frame/transcript/source hash.

Never key captions only by video ID and segment index.

Candidate evaluation must never overwrite incumbent captions.

Full candidate caches and incumbent caches must be distinguishable by prompt
hash.

Mixed candidate views should ideally be materialized only in memory or in a
clearly temporary run directory.

When GEPA selects a new incumbent prompt:

- do not promote a mixed candidate view into a full cache;
- lazily generate a complete caption cache for each video when that video is
  next needed for reflection or full evaluation;
- ensure that any trajectory used to generate new GEPA feedback is produced from
  captions consistently generated under one prompt.

---

## 12. Reference extraction

The surrogate depends on accurate reference extraction.

Extract references from machine-readable trajectory evidence in this priority
order:

1. segment or clip IDs explicitly passed to tool calls;
2. segment IDs returned by retrieval and subsequently consumed by reasoning;
3. timestamps or database record IDs logged in the trajectory;
4. deterministic timestamp-to-segment mapping.

Do not infer references solely from the final prose answer when exact tool
metadata is available.

Track these sets separately:

- `retrieved_segments`;
- `consumed_segments`;
- `directly_referenced_segments`;
- `expanded_reference_set`.

Save the evidence explaining why each segment was selected.

The default expansion policy should be configurable. A reasonable initial
policy is:

```text
expanded_reference_set =
    consumed_segments
    union directly_referenced_segments
    union temporal_neighbors(radius=1)
```

Do not silently treat all retrieved clips as consumed clips.

If no references can be recovered:

- use a configured full-rollout fallback; or
- mark the example unsupported.

Never silently perform a no-op surrogate evaluation.

---

## 13. Full rollout evaluation

For each GEPA candidate and evaluation batch:

1. collect unique videos in the batch;
2. generate captions for every segment of every video using the candidate
   prompt;
3. rebuild or refresh every caption-dependent DVD index;
4. execute the unchanged DVD reasoning pipeline for all QAs;
5. compute per-example scores and aggregate score;
6. retain trajectories and references;
7. log caption cost, reasoning cost, token use, and timing.

If the incumbent is included in candidate comparison, evaluate it on the same
examples and metric.

---

## 14. Selective surrogate rollout evaluation

References are obtained from an incumbent rollout on the same evaluation
examples.

For each GEPA candidate:

1. group references by video;
2. apply the configured reference expansion policy;
3. generate candidate captions only for expanded referenced segments;
4. load incumbent captions for all unselected segments;
5. create a temporary mixed candidate view;
6. update the caption-dependent index for that temporary view;
7. execute the unchanged DVD reasoning pipeline for all QAs;
8. compute and return the same metrics as full rollout;
9. delete or invalidate the temporary mixed view after scoring.

Formally:

```text
caption_for_candidate_view(segment) =
    caption(candidate_prompt, segment)
        if segment in expanded_reference_set
    caption(incumbent_prompt, segment)
        otherwise
```

Only caption generation is approximated. The downstream DVD rollout should still
run normally.

---

## 15. Known approximation limitation

The incumbent trajectory determines the initial reference set.

A candidate captioner prompt might change an unselected caption enough to alter
retrieval, but the surrogate cannot observe that change.

The first implementation must make this limitation measurable rather than
hiding it.

Support and log:

- direct-reference coverage;
- temporal-neighbor radius;
- recaptioned-segment fraction;
- retrieval-set overlap between full and surrogate rollouts;
- cases where full rollout retrieves a segment absent from the surrogate
  reference set;
- full-rollout fallback frequency.

Optional later mitigation:

- exploration segments;
- retrieval-candidate expansion;
- verifier-based fallback;
- trust-region checks;
- periodic full refresh.

Do not implement all mitigations before the basic fidelity experiment works.

---

## 16. GEPA feedback

GEPA should receive actionable information from DVD evaluation.

Textual feedback may include:

- current captioner prompt;
- question;
- predicted answer;
- ground-truth answer;
- scalar reward or correctness;
- relevant trajectory messages;
- tool calls and outputs;
- referenced captions;
- segment IDs and timestamps;
- parsing or execution errors.

Use the official GEPA mechanism for actionable side information, reflective
datasets, or textual feedback supported by the installed API.

Do not write an independent prompt optimizer around this feedback.

Avoid sending irrelevant raw logs that exceed context limits. Preserve the exact
feedback payload in run artifacts.

---

## 17. GEPA candidate constraints

The optimized text artifact is the captioner prompt.

Every GEPA candidate must preserve:

- required placeholders;
- valid input interpolation;
- output JSON schema expected by DVD;
- required timestamp fields;
- required subject-registry fields;
- parsing compatibility.

Before expensive evaluation, validate candidate structure.

A candidate with invalid placeholders or output schema should receive a
documented failure score and actionable error feedback through GEPA.

Do not manually repair candidates without logging the repair.

---

## 18. First experiment: frozen-candidate surrogate fidelity

Implement this before running complete GEPA optimization.

### Protocol

1. freeze one seed or incumbent prompt;
2. freeze one validation batch;
3. obtain one fixed set of candidate prompts:
   - candidates may be proposed by one short GEPA run;
   - save their exact text and hashes;
4. evaluate every saved candidate with full rollout;
5. evaluate those exact same candidates with surrogate rollout;
6. compare scores and rankings.

Do not let full and surrogate modes generate different candidate sets.

### Required metrics

- raw full and surrogate score for every candidate;
- Spearman correlation;
- Kendall correlation with tie handling;
- pairwise ranking accuracy;
- top-1 agreement;
- top-3 recall when candidate count permits;
- mean absolute score error;
- maximum absolute score error;
- recaptioned segment ratio;
- captioning wall-clock time;
- total wall-clock time;
- cache hit rate;
- downstream LLM calls and tokens.

Small QA batches may produce many tied scores. Preserve raw per-QA outcomes and
state the tie-handling rule.

---

## 19. Second experiment: matched GEPA optimization

Run two optimization conditions:

```text
GEPA + FullRolloutEvaluator
GEPA + SelectiveSurrogateRolloutEvaluator
```

Hold constant:

- seed captioner prompt;
- train and validation manifests;
- example order and batches when the API permits;
- GEPA configuration;
- reflection model;
- candidate/evaluation budget;
- selection strategy;
- merge settings;
- stopping criteria;
- caption model and decoding;
- DVD reasoning model and decoding;
- random seeds;
- QA metric.

Use GEPA's own candidate management and selection.

Do not manually select candidates outside GEPA.

The optimization paths may diverge after different rollout scores. Log the full
candidate lineage and GEPA state for both conditions.

---

## 20. Final evaluation

After optimization, freeze these prompts:

- initial DVD captioner prompt;
- prompt selected by full-rollout GEPA;
- prompt selected by surrogate-rollout GEPA.

Evaluate each frozen prompt on held-out test videos using **full recaptioning of
every segment**.

Do not use surrogate captions for final test evaluation.

The final test compares prompt quality, not approximation quality.

Report:

- QA accuracy and raw correct counts;
- caption token count;
- caption generation time;
- reasoning token use;
- final prompt text;
- prompt lineage;
- total optimization cost before test.

---

## 21. Optional multimodal feedback

Only after RQ1 and RQ2 work:

1. sample frames from referenced segments;
2. provide them through the official GEPA feedback/proposal integration;
3. compare:
   - text-only trajectory feedback;
   - text plus localized visual evidence;
4. use the same GEPA and rollout budget.

Do not provide full long videos to the reflector.

Do not use validation or test frames to generate training feedback outside the
defined optimization protocol.

---

## 22. Logging

Every run must create a self-contained artifact directory.

Suggested contents:

```text
run_config.yaml
environment.json
git_commit.txt
split_manifest.json

prompts/
    seed.txt
    candidates/
    incumbents/

gepa/
    configuration.json
    state/
    logs/

captions/
    cache_manifest.jsonl

trajectories/
    *.jsonl

references/
    *.jsonl

evaluations/
    per_example.jsonl
    per_candidate.jsonl
    ranking_metrics.json
    cost_metrics.json

summary.json
```

Log at minimum:

- exact prompt text and hash;
- parent candidate and iteration;
- GEPA version and configuration;
- all model identifiers;
- dataset IDs;
- random seeds;
- rollout mode;
- selected/reference segment IDs;
- reference-expansion reason;
- number of total and re-captioned segments;
- cache hits and misses;
- per-QA prediction and score;
- parsing errors;
- trajectories and tool calls;
- feedback supplied to GEPA;
- token usage;
- API cost estimate;
- caption GPU time;
- wall-clock time;
- fallback reason.

Never save API keys or secrets.

---

## 23. Tests

Add unit tests for:

1. prompt hashing;
2. caption cache separation across prompt hashes;
3. cache separation across caption-model and decoding settings;
4. extraction of clip IDs from representative DVD trajectories;
5. timestamp-to-segment normalization;
6. temporal-neighbor expansion;
7. replacement of selected captions only;
8. full rollout re-captioning every segment;
9. surrogate rollout re-captioning only the expanded reference set;
10. no mutation of incumbent cache during candidate evaluation;
11. mixed candidate views never being committed as full caches;
12. video-level split disjointness;
13. candidate schema and placeholder validation;
14. identical full and surrogate caption views when every segment is selected;
15. deterministic score aggregation for the same saved outputs.

Add one integration test using one small video and one to three QAs before any
large experiment.

---

## 24. Error handling

- Do not silently change models, prompts, decoding settings, or metrics.
- Do not silently retry with a modified candidate.
- Do not silently fall back from surrogate to full rollout.
- Every fallback must include a machine-readable reason.
- Abort on cache version or prompt-hash mismatch.
- Preserve DVD's current output schema.
- Distinguish answer-parsing failures from semantic QA failures.
- Use one shared answer parser across all conditions.
- Record failed candidates rather than dropping them.

---

## 25. Implementation sequence

Follow this sequence.

### Phase 0: inspect and preserve baseline

- inspect DVD;
- inspect installed official GEPA API;
- run and save one unchanged DVD baseline;
- write a concrete integration plan before editing core logic.

### Phase 1: instrumentation

- standardize QA run records;
- expose trajectories and exact segment IDs;
- add prompt-versioned caption caching;
- save split manifests.

### Phase 2: evaluator abstraction

- implement one shared DVD candidate-evaluation interface;
- implement full rollout;
- implement selective surrogate rollout;
- add cache and reference tests.

### Phase 3: fidelity experiment

- freeze one candidate list;
- run the same candidates in both modes;
- produce ranking, fidelity, and cost reports.

### Phase 4: official GEPA integration

- connect the DVD evaluator to the official GEPA API;
- first run GEPA with full rollout for a tiny smoke test;
- then switch only the evaluator to surrogate rollout.

### Phase 5: matched optimization

- run full-rollout GEPA;
- run surrogate-rollout GEPA;
- save all GEPA states, prompts, trajectories, and costs.

### Phase 6: held-out evaluation

- fully re-caption test videos under each frozen final prompt;
- run DVD;
- produce final comparison.

### Phase 7: optional multimodal feedback

- add referenced-frame extraction;
- run the controlled feedback ablation.

Do not skip directly to a large optimization run.

---

## 26. Coding instructions for Claude

- Inspect the repository before creating new abstractions.
- Use the official GEPA API rather than recreating it.
- Make small, reviewable changes.
- Reuse existing DVD captioning, indexing, retrieval, and agent functions.
- Avoid unrelated cleanup and broad refactors.
- Keep full and surrogate modes behind the same evaluator interface.
- Put experiment choices in configuration files.
- Use type hints for new public interfaces.
- Add concise docstrings for non-obvious behavior.
- Do not hardcode paths, model names, prompts, credentials, or dataset IDs.
- Run focused tests after each phase.
- Preserve existing CLI behavior unless a change is explicitly required.

After each coding step, report:

1. files inspected;
2. files changed;
3. behavior added;
4. tests or commands run;
5. unresolved assumptions;
6. exact next step.

At the beginning of the task, do not edit code immediately. First report:

- current DVD execution flow;
- caption generation and cache locations;
- trajectory/reference availability;
- installed GEPA version and chosen official API;
- proposed minimal integration points.

---

## 27. First requested deliverable

The first deliverable is not a complete optimizer.

Produce a working, reproducible **surrogate-fidelity harness** that:

1. accepts a saved list of captioner prompt candidates;
2. evaluates each candidate through full rollout;
3. evaluates the same candidate through selective surrogate rollout;
4. uses the same DVD reasoning pipeline in both modes;
5. saves raw scores, trajectories, references, timing, and costs;
6. computes ranking-fidelity metrics;
7. passes cache-contamination tests;
8. runs from one documented command.

Only after this harness works should GEPA be allowed to generate and select
candidates online.

---

## 28. Definition of done

### Fidelity milestone

Complete when:

- reference extraction is machine-readable and tested;
- candidate caption caches are prompt-versioned;
- full and surrogate evaluators share one DVD reasoning path;
- the same frozen candidates can be scored by both modes;
- ranking and cost metrics are generated;
- mixed caches cannot contaminate incumbent state.

### Optimization milestone

Complete when:

- official GEPA runs with both evaluator modes;
- both conditions use matched GEPA budgets;
- complete candidate lineages and GEPA states are saved;
- final prompts are frozen;
- held-out test evaluation uses full rollout;
- accuracy, fidelity, and cost results are reproducible;
- no test example influenced GEPA reflection or selection.
