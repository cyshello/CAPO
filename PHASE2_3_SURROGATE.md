# Phase 2 Implementation Plan: Prompt-Delta-Aware Surrogate Rollout

## Current Repository State

Repository:

```text
/home/intern/youngseo/surrogate_rollout
```

Current baseline:

- Branch: `main`
- Baseline commit: `c4ad3eb`
- Existing instrumentation, split manifest, cache protection, reference extraction, and regression tests are already implemented.
- Legacy DVD files must remain untouched.
- Phase 2 candidate evaluation must use an isolated work root.
- Temporary caption databases must be keyed by the exact `captions.json` content hash.
- Legacy caption caches are read-only.
- `returned + frame_inspection` is the current default trace-reference policy candidate.
- Zero-reference QAs must support fallback behavior.

The immediate Phase 2 goal is:

> Compare full candidate-prompt rollouts against selective rollouts that combine reasoning-trace references with prompt-delta-aware CLIP retrieval.

---

## 1. Shared Evaluation Interface

Add a common evaluator interface used by both full and selective rollout evaluation.

Suggested location:

```text
evaluation/rollout_evaluator.py
```

Suggested types:

```python
class RolloutEvaluator(Protocol):
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        ...
```

Minimum `EvaluationRequest` fields:

```python
video_id
qa_id
question
answer_options
ground_truth
baseline_prompt
candidate_prompt
baseline_captions_path
baseline_trajectory_path
reference_policy
selection_policy
work_root
```

Minimum `EvaluationResult` fields:

```python
answer
is_correct
selected_clip_ids
recaptioned_clip_ids
recaption_fraction
selection_sources
fallback_reason
captions_hash
captions_path
database_path
trajectory_path
caption_call_count
latency
```

Implementations:

```text
FullRolloutEvaluator
SelectiveSurrogateRolloutEvaluator
```

---

## 2. FullRolloutEvaluator

Suggested location:

```text
evaluation/full_rollout.py
```

Responsibilities:

1. Load the exact baseline video configuration.
2. Run the captioner with the candidate prompt over every clip.
3. Rebuild the complete `subject_registry`.
4. Snapshot the exact resulting `captions.json`.
5. Compute its content hash.
6. Build a fresh temporary NanoVectorDB keyed by that hash.
7. Run the DVD QA agent.
8. Save the trajectory, final answer, correctness, latency, and caption-call count.
9. Never write into a registered legacy cache.

This evaluator provides the ground-truth candidate-prompt rollout used in sanity checks.

Required regression check:

> Running the baseline prompt through `FullRolloutEvaluator` should reproduce the captured baseline behavior within the expected nondeterminism of the subject-registry merge.

---

## 3. Mixed-View Builder

Suggested location:

```text
mixed_views/builder.py
```

Purpose:

Create a temporary caption view in which selected clips use candidate-prompt captions and all remaining clips reuse the baseline caption snapshot.

Suggested interface:

```python
class MixedViewBuilder:
    def build(
        self,
        baseline_captions_path: Path,
        candidate_captions: dict[str, dict],
        selected_clip_ids: set[str],
        work_root: Path,
    ) -> MixedViewArtifact:
        ...
```

Suggested output:

```python
MixedViewArtifact(
    captions_path=...,
    database_path=...,
    captions_hash=...,
    selected_clip_ids=...,
)
```

Responsibilities:

1. Load the frozen baseline caption snapshot.
2. Replace only the selected per-clip captions.
3. Re-run the registry merge over the resulting mixed caption set.
4. Write the mixed `captions.json` into the Phase 2 work root.
5. Compute the exact content hash.
6. Build a fresh temporary NanoVectorDB keyed by the hash.
7. Return paths and metadata.

Constraints:

- Do not modify the baseline snapshot.
- Do not write into legacy caption caches.
- Keep per-candidate work directories isolated.
- Treat `subject_registry` as derived state.
- Key temporary databases by the full mixed-caption content hash.

---

## 4. Clip Selector Interface

Suggested location:

```text
selection/base.py
```

Suggested interface:

```python
class ClipSelector(Protocol):
    def select(self, context: SelectionContext) -> SelectionResult:
        ...
```

Minimum `SelectionContext` fields:

```python
video_id
qa_id
question
baseline_prompt
candidate_prompt
baseline_trajectory
baseline_captions
visual_index
reference_policy
budget
```

Suggested `SelectionResult`:

```python
SelectionResult(
    clip_ids=[...],
    scores={...},
    sources={clip_id: [...]},
    metadata={...},
)
```

Initial selector implementations:

```text
TraceReferenceSelector
QuestionClipRetriever
PromptDeltaClipRetriever
UnionSelector
RandomBudgetMatchedSelector
```

---

## 5. TraceReferenceSelector

Reuse the existing modules:

```text
references/extractor.py
references/expansion.py
```

The first default policy should be:

```text
returned + frame_inspection
```

Current measured recaption fraction:

```text
mean 17%, range 0–43%
```

Keep the following policies configurable:

```text
all_returned
returned + frame_inspection
explicit citations + frame_inspection
```

Do not add an LLM-based trace selector in Phase 2.

---

## 6. Cached CLIP or SigLIP Visual Index

Suggested locations:

```text
retrieval/visual_index.py
scripts/build_visual_index.py
```

Purpose:

Precompute reusable frame embeddings once per video so prompt candidates can retrieve visually relevant clips without another captioner or multimodal-verifier call.

For the first implementation, use the same sampled frames already consumed by the DVD captioner.

Store, per frame:

```python
video_id
clip_id
timestamp
frame_path_or_id
embedding
```

Suggested artifact structure:

```python
VisualIndexArtifact(
    video_id=...,
    model_id=...,
    frame_sampling_config=...,
    frame_records=[...],
)
```

Required cache-key fields:

```text
video_id
vision_model_id
frame_sampling_config
video_source_hash or frame-source hash
preprocessing version
```

Requirements:

- L2-normalize image embeddings before storage.
- Preserve frame-to-clip mapping.
- Make index creation deterministic.
- Avoid loading the captioner VLM for retrieval.
- Support batched matrix similarity against text embeddings.

---

## 7. Question Query Generator

Suggested location:

```text
retrieval/question_queries.py
```

Purpose:

Convert the downstream QA question into a small set of atomic visual retrieval queries.

Input:

```python
question
answer_options
```

Output example:

```json
{
  "queries": [
    "coaches sitting on chairs",
    "lights attached to chairs",
    "coaches turning around"
  ]
}
```

Requirements:

- Text-only LLM call.
- Structured JSON schema.
- Maximum 4–8 queries.
- Focus on visually retrievable entities, attributes, actions, and text.
- Cache by question, answer options, generator model, and prompt version.

This provides the `question_clip` baseline.

---

## 8. Prompt-Delta Query Generator

Suggested location:

```text
retrieval/prompt_delta_queries.py
```

Purpose:

Extract newly emphasized visual concepts from the difference between the baseline caption prompt and a candidate caption prompt, then combine them with the downstream question.

Input:

```python
question
answer_options
baseline_prompt
candidate_prompt
```

Output example:

```json
{
  "newly_emphasized_concepts": [
    "fine-grained colors",
    "small object attributes",
    "brief actions"
  ],
  "queries": [
    "color of lights attached to coaches' chairs",
    "small visible details on the chairs",
    "coaches turning around in chairs"
  ]
}
```

Requirements:

- Text-only LLM call.
- Fixed JSON schema.
- Maximum 4–8 retrieval queries.
- Output only concepts that can plausibly be retrieved from images.
- Do not pass the entire caption prompt directly into the CLIP text encoder.
- Cache by:
  - question hash,
  - answer-options hash,
  - baseline-prompt hash,
  - candidate-prompt hash,
  - query-generator model,
  - query-generator prompt version.

---

## 9. CLIP Clip Retriever

Suggested location:

```text
retrieval/clip_retriever.py
```

Input:

```python
visual_index
text_queries
top_k
max_fraction
```

Scoring:

```python
frame_score = max(
    cosine_similarity(frame_embedding, query_embedding)
    for query_embedding in query_embeddings
)

clip_score = max(
    frame_score
    for frame in clip
)
```

Return:

```python
SelectionResult(
    clip_ids=[...],
    scores={clip_id: score},
    sources={clip_id: ["question_clip"]},
)
```

For prompt-delta retrieval, use source:

```text
prompt_delta_clip
```

Initial implementation should use simple maximum similarity. Do not add learned fusion or reranking yet.

Optional later alternatives:

```text
mean of top-N frame scores
query-wise reciprocal-rank fusion
temporal smoothing
```

---

## 10. Selection Union and Budget Controller

Suggested locations:

```text
selection/union.py
selection/budget.py
```

Primary selection rule:

```python
selected = trace_references | prompt_delta_clip_references
```

Required configurable policies:

```python
SELECTION_POLICIES = {
    "trace_only": ...,
    "question_clip": ...,
    "trace_plus_question_clip": ...,
    "prompt_delta_clip": ...,
    "trace_plus_prompt_delta_clip": ...,
    "random_budget_matched": ...,
}
```

Initial budget fields:

```python
max_recaption_fraction
max_total_clips
max_clip_retrieval_fraction
neighbor_radius
```

Recommended priority when the union exceeds budget:

1. Explicit frame inspections
2. Strict trace references
3. Other trace-returned clips
4. Prompt-delta CLIP retrieval
5. Question-only CLIP retrieval
6. Temporal neighbors

Preserve per-clip selection provenance in the result.

---

## 11. Zero-Reference Fallback

Existing strict policies produce zero references for some QAs answered through `global_browse`.

Use the following fallback order:

```text
trace references available
    -> trace ∪ configured CLIP retrieval

no trace references
    -> CLIP-only selection

CLIP selection unavailable or empty
    -> full rollout or unsupported, according to configuration
```

Record the exact fallback type:

```text
none
clip_only
full_rollout
unsupported
```

Do not silently convert a zero-reference case into a full rollout without recording it.

---

## 12. SelectiveSurrogateRolloutEvaluator

Suggested location:

```text
evaluation/selective_rollout.py
```

Pipeline:

```text
load baseline trajectory
    -> extract trace references
    -> generate question queries
    -> generate prompt-delta queries
    -> retrieve CLIP-ranked clips
    -> combine selections
    -> enforce recaption budget
    -> recaption selected clips only
    -> build mixed caption view
    -> build fresh DB from mixed captions hash
    -> run DVD QA
    -> save result and trajectory
```

Required behavior:

- Candidate captions must use the existing prompt-aware cache-key logic.
- Baseline captions must remain frozen.
- Candidate-specific temporary work roots must be isolated.
- Every selected clip must record why it was selected.
- Every result must record the full-rollout-equivalent recaption fraction.
- A fallback full rollout must still return through the same result interface.

---

## 13. Sanity-Check Runner

Suggested location:

```text
scripts/run_phase2_sanity_check.py
```

Run the same fixed QA and candidate-prompt set under:

```text
full_rollout
trace_only
question_clip
trace_plus_question_clip
prompt_delta_clip
trace_plus_prompt_delta_clip
random_budget_matched
```

Initially use:

- the existing train split only,
- a small fixed number of videos and QAs,
- 3–5 existing prompt variants,
- deterministic candidate lists,
- no optimizer.

Required output per QA and candidate prompt:

```python
full_answer
selective_answer
full_correct
selective_correct
answer_agreement
correctness_agreement
selected_clip_ids
selection_sources
recaption_fraction
caption_call_count
latency
fallback_type
```

Aggregate metrics:

```text
full-answer agreement
correctness agreement
selective QA accuracy
full-rollout QA accuracy
recaption fraction
caption-call reduction
latency reduction
agreement at fixed recaption budget
```

Primary research check:

> At the same recaption budget, does `trace_plus_prompt_delta_clip` recover the full-rollout QA result better than `trace_only`, `question_clip`, and random selection?

---

## 14. Required Tests

Add at least:

```text
tests/test_full_rollout_matches_baseline.py
tests/test_mixed_view_replaces_only_selected_clips.py
tests/test_mixed_view_database_hash.py
tests/test_visual_index_cache_key.py
tests/test_visual_index_determinism.py
tests/test_question_query_cache.py
tests/test_prompt_delta_query_cache.py
tests/test_clip_retrieval_determinism.py
tests/test_selection_union.py
tests/test_selection_budget.py
tests/test_zero_reference_clip_fallback.py
tests/test_phase2_no_legacy_cache_writes.py
```

Important assertions:

- Only selected per-clip captions change in a mixed view.
- Registry merge does not mutate the frozen baseline snapshot.
- Different mixed caption contents produce different DB keys.
- Different candidate prompts produce separate caption and query-cache entries.
- Selection is deterministic under fixed inputs.
- Full fallback is explicitly recorded.
- Phase 2 writes nothing to registered legacy cache paths.

---

## 15. Recommended Implementation Order

1. Implement `RolloutEvaluator` request/result schemas.
2. Implement `FullRolloutEvaluator`.
3. Implement `MixedViewBuilder`.
4. Implement `SelectiveSurrogateRolloutEvaluator` with trace-only selection.
5. Validate full versus trace-only rollouts.
6. Implement cached visual-index creation.
7. Implement question-only query generation and retrieval.
8. Implement prompt-delta query generation.
9. Implement trace-plus-prompt-delta selection.
10. Add budget-matched random baseline.
11. Run the Phase 2 sanity check.
12. Only after the sanity check, consider iterative support expansion or optimizer integration.

---

## Out of Scope for the Initial Phase 2 Sanity Check

Do not implement yet:

```text
multimodal verifier
learned clip selector
Bayesian optimization
Hyperband scheduler
iterative support expansion
statistical audit verifier
automatic trust-region controller
new DVD prompt optimizer
```

The first implementation should answer only:

> Does prompt-delta-aware retrieval over cached vision-language embeddings improve full-rollout QA recovery over trace-only selective recaptioning at the same recaption cost?
