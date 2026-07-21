# Phase 4 Runbook

## Prompt-delta Checkpoint A: documentation and schemas

Checkpoint A adds only the frozen prompt-delta data contracts in
`optimization/schemas.py` and fixture-only schema tests. It does not add a
runtime mode, persistence store, adapter, feedback generator, updater, CLI, or
model call. No environment variables are required and no run, state, cache, or
other artifact path is created. Re-running either command is side-effect free.

Focused verification:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_prompt_delta_schemas.py
```

Related legacy schema, persistence, router, feedback, free-form generator, and
Phase 2–3 prompt-delta retrieval regression verification:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_phase4_schemas.py \
  tests/test_phase4_persistence.py \
  tests/test_prompt_router.py \
  tests/test_feedback_generator.py \
  tests/test_free_form_instruction.py \
  tests/test_prompt_delta_query_cache.py
```

Success requires every test to pass. These commands must not create or modify
legacy codebook/router snapshots, caption caches, or experiment artifacts.
Checkpoint A itself contains no adapter or later-stage behavior; the separate
Checkpoint B adapter is documented below.

## Prompt-delta Checkpoint B: read-only legacy adapter

Checkpoint B adds only the read-only
`legacy_property_intervention_to_episode(...)` Python adapter. It reads one
completed legacy property-intervention `result.json` plus its exact baseline
`video_complete.json` and returns an in-memory `InterventionEpisode`. It writes
no episode artifact, codebook state, cache, pointer, or runtime configuration.
No environment variables, model credentials, GPU, or output directory are
required.

Focused verification:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_legacy_intervention_adapter.py
```

Checkpoint A and related legacy property regression verification:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_prompt_delta_schemas.py \
  tests/test_checkpoint3b_property_proposal.py \
  tests/test_checkpoint3c_property_intervention.py \
  tests/test_checkpoint1_property_memory.py
```

The repository-local read-only smoke used the following existing saved v1
intervention. It prints only a bounded conversion summary and creates no file:

```bash
PYTHONPATH=/home/intern/youngseo \
conda run -n local_llm_vllm python -c \
'import json; from surrogate_rollout.optimization.legacy_intervention_adapter import legacy_property_intervention_to_episode; episode=legacy_property_intervention_to_episode(intervention_result_path="runs/phase4_memory_k4_isolated_001_output/interventions/7D-gxaie6UI/7D-gxaie6UI/candidate_284d53fde9070f949063_pbad21bc654ed/result.json", baseline_video_manifest_path="runs/phase4_memory_k4_isolated_001_output/baseline_videos/7D-gxaie6UI/baseline/7D-gxaie6UI/video_complete.json", parent_meta_prompt_id="legacy-read-only-smoke-parent"); print(json.dumps({"episode_id": episode.episode_id, "video_id": episode.video_id, "delta_id": episode.prompt_delta.delta_id, "clip_count": len(episode.clips), "qa_count": len(episode.qa_outcomes)}, sort_keys=True))'
```

There is no prompt-delta CLI mode or prompt-delta persistence namespace at
this checkpoint. Checkpoint C's separate in-memory mock path is documented
below.

## Prompt-delta Checkpoint C: deterministic mock feedback

Checkpoint C adds only the in-memory
`DeterministicMockEpisodeFeedbackGenerator`. It accepts an
`InterventionEpisode`, reports deterministic caption-string and QA-correctness
facts, validates every supporting ID against that episode, and returns an
`EpisodeFeedback`. It does not load trajectory payloads or frames, call a
model, propose a meta-prompt change, persist feedback, or modify runtime state.

Focused verification:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_episode_feedback.py
```

Checkpoint A/B, legacy feedback, and related schema/persistence regression
verification:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_prompt_delta_schemas.py \
  tests/test_legacy_intervention_adapter.py \
  tests/test_feedback_generator.py \
  tests/test_checkpoint3d_interventional_feedback.py \
  tests/test_phase4_schemas.py \
  tests/test_phase4_persistence.py
```

The model-free repository test suite can be checked with:

```bash
conda run -n local_llm_vllm python -m pytest -q
```

The repository-local read-only smoke used the existing Checkpoint B saved v1
bundle below. The two path variables are inputs, not output locations. The
command hashes every file under both source artifact directories before and
after conversion, asserts equality, and prints only an in-memory summary:

```bash
LEGACY_INTERVENTION_RESULT=runs/phase4_memory_k4_isolated_001_output/interventions/7D-gxaie6UI/7D-gxaie6UI/candidate_284d53fde9070f949063_pbad21bc654ed/result.json
LEGACY_BASELINE_MANIFEST=runs/phase4_memory_k4_isolated_001_output/baseline_videos/7D-gxaie6UI/baseline/7D-gxaie6UI/video_complete.json
PYTHONPATH=/home/intern/youngseo \
conda run -n local_llm_vllm python -c \
'import hashlib,json,sys; from pathlib import Path; from surrogate_rollout.optimization.episode_feedback import DeterministicMockEpisodeFeedbackGenerator; from surrogate_rollout.optimization.legacy_intervention_adapter import legacy_property_intervention_to_episode; result=Path(sys.argv[1]).resolve(); baseline=Path(sys.argv[2]).resolve(); files=sorted({path.resolve() for root in (result.parent, baseline.parent) for path in root.rglob("*") if path.is_file()}, key=str); aggregate=lambda: hashlib.sha256("".join(str(path)+":"+hashlib.sha256(path.read_bytes()).hexdigest()+"\n" for path in files).encode()).hexdigest(); before=aggregate(); episode=legacy_property_intervention_to_episode(intervention_result_path=str(result), baseline_video_manifest_path=str(baseline), parent_meta_prompt_id="legacy-read-only-smoke-parent"); feedback=DeterministicMockEpisodeFeedbackGenerator().generate(episode); after=aggregate(); assert before == after; print(json.dumps({"aggregate_sha256_before": before, "aggregate_sha256_after": after, "episode_id": episode.episode_id, "feedback_id": feedback.feedback_id, "observation_count": len(feedback.observations), "counterevidence_count": len(feedback.counterevidence)}, sort_keys=True))' \
"$LEGACY_INTERVENTION_RESULT" "$LEGACY_BASELINE_MANIFEST"
```

Success requires equal before/after hashes and stable episode/feedback IDs.
Checkpoint C has no feedback artifact path, resume state, updater, real
feedback provider, or runtime/CLI integration.

## Prompt-delta Checkpoint D1: complete request and LLM policy boundary

Checkpoint D1 resolves the complete saved baseline/intervention QA and
trajectory evidence for one `InterventionEpisode`, constructs one deterministic
text-only request, and exposes a strict one-call Python policy boundary for an
explicitly injected backend. Tests use only a deterministic fake backend. No
real model/API is called, and no request, response, feedback, state, cache, or
run artifact is written.

Focused verification:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_llm_episode_feedback.py
```

Checkpoint A–C and related legacy feedback/schema/persistence regression:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_prompt_delta_schemas.py \
  tests/test_legacy_intervention_adapter.py \
  tests/test_episode_feedback.py \
  tests/test_llm_episode_feedback.py \
  tests/test_feedback_generator.py \
  tests/test_checkpoint3d_interventional_feedback.py \
  tests/test_phase4_schemas.py \
  tests/test_phase4_persistence.py
```

Complete model-free suite:

```bash
conda run -n local_llm_vllm python -m pytest -q
```

The following read-only request-building smoke uses an existing saved v1
intervention. The two variables identify source inputs. The command hashes all
files below both source directories before and after, builds the complete
request in memory, prints bounded statistics, and performs no backend call:

```bash
LEGACY_INTERVENTION_RESULT=runs/phase4_memory_k4_isolated_001_output/interventions/7D-gxaie6UI/7D-gxaie6UI/candidate_284d53fde9070f949063_pbad21bc654ed/result.json
LEGACY_BASELINE_MANIFEST=runs/phase4_memory_k4_isolated_001_output/baseline_videos/7D-gxaie6UI/baseline/7D-gxaie6UI/video_complete.json
PYTHONPATH=/home/intern/youngseo \
conda run -n local_llm_vllm python -c \
'import hashlib,json,sys; from pathlib import Path; from surrogate_rollout.optimization.legacy_intervention_adapter import legacy_property_intervention_to_episode; from surrogate_rollout.optimization.llm_episode_feedback import LegacyEpisodeFeedbackArtifactResolver,build_episode_feedback_request; result=Path(sys.argv[1]).resolve(); baseline=Path(sys.argv[2]).resolve(); files=sorted({path.resolve() for root in (result.parent,baseline.parent) for path in root.rglob("*") if path.is_file()},key=str); aggregate=lambda: hashlib.sha256("".join(str(path)+":"+hashlib.sha256(path.read_bytes()).hexdigest()+"\n" for path in files).encode()).hexdigest(); before=aggregate(); episode=legacy_property_intervention_to_episode(intervention_result_path=str(result),baseline_video_manifest_path=str(baseline),parent_meta_prompt_id="legacy-read-only-smoke-parent"); request=build_episode_feedback_request(episode,artifact_resolver=LegacyEpisodeFeedbackArtifactResolver()); after=aggregate(); assert before==after; stats=request.size_statistics; print(json.dumps({"aggregate_sha256_before":before,"aggregate_sha256_after":after,"episode_id":episode.episode_id,"clip_count":stats.clip_count,"qa_count":stats.qa_count,"history_count":stats.total_history_item_count,"total_request_characters":stats.serialized_request_character_count,"trajectory_characters":stats.trajectory_character_count,"request_payload_hash":request.payload_hash,"unresolved_references":stats.unresolved_reference_count,"context_limit_checked":stats.context_limit_checked},sort_keys=True))' \
"$LEGACY_INTERVENTION_RESULT" "$LEGACY_BASELINE_MANIFEST"
```

No actual LLM invocation, persistence/resume path, updater, or prompt-delta
runtime/CLI mode exists at Checkpoint D1. A backend with a known token context
limit must inject its exact token counter; overflow fails before its callable
is invoked and no truncation is performed. A backend that explicitly reports
an unknown limit (`null`) produces the complete request with
`context_limit_checked=false`.

## Prompt-delta Checkpoint D1.5: compact request representation

Checkpoint D1.5 adds an explicit parallel
`build_compact_episode_feedback_request(...)` representation. The D1 complete
builder remains unchanged and remains opt-in under its existing name. The
compact view catalogs exact history items once, replaces clip-level history
copies with ordered IDs, removes the validated clip-level delta duplicate, and
projects raw trajectory wrappers into executed tool events, retained evidence,
assistant-authored steps, final responses, references, and audit hashes. It
does not truncate, summarize, sample, filter QAs, call a backend, or write an
artifact.

Focused verification:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_compact_episode_feedback.py
```

Checkpoint A–D1.5 and legacy feedback/schema/persistence regression:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_prompt_delta_schemas.py \
  tests/test_legacy_intervention_adapter.py \
  tests/test_episode_feedback.py \
  tests/test_llm_episode_feedback.py \
  tests/test_compact_episode_feedback.py \
  tests/test_feedback_generator.py \
  tests/test_checkpoint3d_interventional_feedback.py \
  tests/test_phase4_schemas.py \
  tests/test_phase4_persistence.py
```

Complete model-free suite:

```bash
conda run -n local_llm_vllm python -m pytest -q
```

The following read-only comparison uses the same saved v1 bundle as D1. It
builds both representations in memory, reconstructs every history snapshot,
hashes all files below both source directories before and after, and performs
no backend call:

```bash
LEGACY_INTERVENTION_RESULT=runs/phase4_memory_k4_isolated_001_output/interventions/7D-gxaie6UI/7D-gxaie6UI/candidate_284d53fde9070f949063_pbad21bc654ed/result.json
LEGACY_BASELINE_MANIFEST=runs/phase4_memory_k4_isolated_001_output/baseline_videos/7D-gxaie6UI/baseline/7D-gxaie6UI/video_complete.json
PYTHONPATH=/home/intern/youngseo \
conda run -n local_llm_vllm python -c \
'import hashlib,json,sys; from pathlib import Path; from surrogate_rollout.optimization.legacy_intervention_adapter import legacy_property_intervention_to_episode; from surrogate_rollout.optimization.llm_episode_feedback import LegacyEpisodeFeedbackArtifactResolver,build_episode_feedback_request,build_compact_episode_feedback_request,reconstruct_compact_history_snapshots; from surrogate_rollout.prompt_routing.schemas import dumps_canonical; result=Path(sys.argv[1]).resolve(); baseline=Path(sys.argv[2]).resolve(); files=sorted({path.resolve() for root in (result.parent,baseline.parent) for path in root.rglob("*") if path.is_file()},key=str); aggregate=lambda: hashlib.sha256("".join(str(path)+":"+hashlib.sha256(path.read_bytes()).hexdigest()+"\n" for path in files).encode()).hexdigest(); before=aggregate(); episode=legacy_property_intervention_to_episode(intervention_result_path=str(result),baseline_video_manifest_path=str(baseline),parent_meta_prompt_id="legacy-read-only-smoke-parent"); resolver=LegacyEpisodeFeedbackArtifactResolver(); complete=build_episode_feedback_request(episode,artifact_resolver=resolver); compact=build_compact_episode_feedback_request(episode,artifact_resolver=resolver); reconstructed=reconstruct_compact_history_snapshots(compact.user_payload); reconstruction_ok=[dumps_canonical(x) for x in reconstructed]==[dumps_canonical(x.history_snapshot) for x in episode.clips]; after=aggregate(); assert before==after and reconstruction_ok; stats=compact.size_statistics; print(json.dumps({"aggregate_sha256_before":before,"aggregate_sha256_after":after,"episode_id":episode.episode_id,"clip_count":stats.clip_count,"qa_count":stats.qa_count,"total_history_occurrences":stats.total_history_item_occurrences,"unique_history_items":stats.unique_history_item_count,"complete_request_characters":stats.complete_request_character_count,"compact_request_characters":stats.compact_request_character_count,"complete_trajectory_characters":complete.size_statistics.trajectory_character_count,"compact_trajectory_characters":stats.compact_trajectory_character_count,"complete_payload_hash":stats.complete_payload_hash,"compact_payload_hash":stats.compact_payload_hash,"unclassified_trajectory_messages":stats.unclassified_trajectory_message_count,"unresolved_references":stats.unresolved_reference_count,"history_reconstruction_success":reconstruction_ok},sort_keys=True))' \
"$LEGACY_INTERVENTION_RESULT" "$LEGACY_BASELINE_MANIFEST"
```

The complete D1 trajectory metric counts serialized raw messages and tool
events only. The D1.5 compact-trajectory metric counts the whole projected
trajectory object, including audit source hashes and complete reference sets;
therefore those two sub-metrics are not expected to be monotonically ordered.
The complete/compact request-character metrics are directly comparable.

## Prompt-delta Checkpoint D1.6: model/audit request separation

Checkpoint D1.6 adds the explicit `request_representation="model_compact"`
view. It retains all semantic clip, cataloged history, QA, executed-tool,
returned-evidence, reference, final-response, and unknown-message content in
`model_payload`; source paths/hashes, projection statistics, and lossless
history-reconstruction metadata remain separately inspectable in
`audit_metadata`. Only canonical `model_payload` is passed to the injected
backend. The default remains `complete`, and D1.5 `compact` remains unchanged.
DVD `references.json` is projected by type: segment-set fields remain ordered
string arrays, while the structured `evidence` object array is preserved as
`reference_evidence` in complete and compact trajectory views and in
model-compact audit metadata. It is not duplicated into the model-facing v3
payload because executed tool events, hits, and reference IDs already retain
the semantic execution evidence. It is never stringified or silently
discarded; its canonical hash is retained in compact audit metadata. For the
same reason, the complete typed `reference_sets` object remains in model-compact
audit metadata while the model-facing trajectory retains the exact referenced
and retrieved segment-ID arrays plus executed tool events and returned evidence.
No model/API call or artifact write is performed by these commands.

Focused verification:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_model_compact_episode_feedback.py
```

Checkpoint A–D1.6 and related legacy regression:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_prompt_delta_schemas.py \
  tests/test_legacy_intervention_adapter.py \
  tests/test_episode_feedback.py \
  tests/test_llm_episode_feedback.py \
  tests/test_compact_episode_feedback.py \
  tests/test_model_compact_episode_feedback.py \
  tests/test_feedback_generator.py \
  tests/test_checkpoint3d_interventional_feedback.py \
  tests/test_phase4_schemas.py \
  tests/test_phase4_persistence.py
```

Complete model-free suite:

```bash
conda run -n local_llm_vllm python -m pytest -q
```

Read-only D1/D1.5/D1.6 comparison over the same saved bundle:

```bash
LEGACY_INTERVENTION_RESULT=runs/phase4_memory_k4_isolated_001_output/interventions/7D-gxaie6UI/7D-gxaie6UI/candidate_284d53fde9070f949063_pbad21bc654ed/result.json
LEGACY_BASELINE_MANIFEST=runs/phase4_memory_k4_isolated_001_output/baseline_videos/7D-gxaie6UI/baseline/7D-gxaie6UI/video_complete.json
PYTHONPATH=/home/intern/youngseo \
conda run -n local_llm_vllm python -c \
'import hashlib,json,sys; from pathlib import Path; from surrogate_rollout.optimization.legacy_intervention_adapter import legacy_property_intervention_to_episode; from surrogate_rollout.optimization.llm_episode_feedback import LegacyEpisodeFeedbackArtifactResolver,build_episode_feedback_request,build_compact_episode_feedback_request,build_model_compact_episode_feedback_request,reconstruct_compact_history_snapshots; from surrogate_rollout.prompt_routing.schemas import dumps_canonical; result=Path(sys.argv[1]).resolve(); baseline=Path(sys.argv[2]).resolve(); files=sorted({path.resolve() for root in (result.parent,baseline.parent) for path in root.rglob("*") if path.is_file()},key=str); aggregate=lambda: hashlib.sha256("".join(str(path)+":"+hashlib.sha256(path.read_bytes()).hexdigest()+"\n" for path in files).encode()).hexdigest(); before=aggregate(); episode=legacy_property_intervention_to_episode(intervention_result_path=str(result),baseline_video_manifest_path=str(baseline),parent_meta_prompt_id="legacy-read-only-smoke-parent"); resolver=LegacyEpisodeFeedbackArtifactResolver(); complete=build_episode_feedback_request(episode,artifact_resolver=resolver); compact=build_compact_episode_feedback_request(episode,artifact_resolver=resolver); model=build_model_compact_episode_feedback_request(episode,artifact_resolver=resolver); reconstructed=reconstruct_compact_history_snapshots(model.model_payload,model.audit_metadata); reconstruction_ok=[dumps_canonical(x) for x in reconstructed]==[dumps_canonical(x.history_snapshot) for x in episode.clips]; after=aggregate(); assert before==after and reconstruction_ok; stats=model.size_statistics; print(json.dumps({"aggregate_sha256_before":before,"aggregate_sha256_after":after,"episode_id":episode.episode_id,"clip_count":stats.clip_count,"qa_count":stats.qa_count,"total_history_occurrences":stats.total_history_item_occurrences,"unique_history_items":stats.unique_history_item_count,"complete_request_characters":stats.complete_request_character_count,"compact_request_characters":stats.compact_request_character_count,"model_compact_request_characters":stats.model_request_character_count,"complete_trajectory_characters":complete.size_statistics.trajectory_character_count,"compact_trajectory_characters":stats.compact_trajectory_character_count,"model_trajectory_characters":stats.model_trajectory_character_count,"complete_payload_hash":stats.complete_payload_hash,"compact_payload_hash":stats.compact_payload_hash,"model_payload_hash":stats.model_payload_hash,"audit_metadata_hash":stats.audit_metadata_hash,"unclassified_trajectory_messages":stats.unclassified_trajectory_message_count,"unresolved_references":stats.unresolved_reference_count,"history_reconstruction_success":reconstruction_ok},sort_keys=True))' \
"$LEGACY_INTERVENTION_RESULT" "$LEGACY_BASELINE_MANIFEST"
```

## Prompt-delta Checkpoint D2: explicit provider and exact token inspection

Checkpoint D2 adds an OpenAI-compatible strict-chat provider adapter without
selecting or calling a provider. Provider identity, model ID, local tokenizer,
tokenizer identity, context limit, output-token reservation, generation
settings, feedback policy version, and response transport are all required.
`prepare_and_measure(...)` builds the D1.6 `model_compact` request, formats the
exact provider messages, and reports system-content, user-content, and complete
chat-template token counts. A generation call is rejected before transport
unless `total_input_tokens + maximum_output_tokens <= context_limit`.

Focused verification:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_episode_feedback_provider.py
```

Checkpoint A–D2 and related legacy regression:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_prompt_delta_schemas.py \
  tests/test_legacy_intervention_adapter.py \
  tests/test_episode_feedback.py \
  tests/test_llm_episode_feedback.py \
  tests/test_compact_episode_feedback.py \
  tests/test_model_compact_episode_feedback.py \
  tests/test_episode_feedback_provider.py \
  tests/test_feedback_generator.py \
  tests/test_checkpoint3d_interventional_feedback.py \
  tests/test_phase4_schemas.py \
  tests/test_phase4_persistence.py
```

Complete model-free suite:

```bash
conda run -n local_llm_vllm python -m pytest -q
```

No D2 model/tokenizer/context configuration is currently committed or present
in the inspected environment. Set all of the following to reviewed exact
values before inspection; the command fails on any missing value. The
tokenizer must already exist locally and expose the exact model chat template.
The script sets `local_files_only=True`, performs no download or provider call,
and verifies the saved source aggregate hash before and after.

```bash
: "${PD_FEEDBACK_PROVIDER:?set exact provider identity}"
: "${PD_FEEDBACK_MODEL_ID:?set exact model ID}"
: "${PD_FEEDBACK_TOKENIZER_PATH:?set existing local tokenizer path}"
: "${PD_FEEDBACK_TOKENIZER_IDENTITY:?set exact tokenizer identity/version}"
: "${PD_FEEDBACK_CONTEXT_LIMIT:?set exact positive context limit}"
: "${PD_FEEDBACK_MAX_OUTPUT_TOKENS:?set exact positive output reservation}"
: "${PD_FEEDBACK_GENERATION_SETTINGS_JSON:?set non-empty JSON settings}"
: "${PD_FEEDBACK_POLICY_VERSION:?set reviewed feedback policy version}"

PYTHONPATH=/home/intern/youngseo \
conda run -n local_llm_vllm python \
  scripts/inspect_episode_feedback_request.py \
  --intervention-result runs/phase4_memory_k4_isolated_001_output/interventions/7D-gxaie6UI/7D-gxaie6UI/candidate_284d53fde9070f949063_pbad21bc654ed/result.json \
  --baseline-manifest runs/phase4_memory_k4_isolated_001_output/baseline_videos/7D-gxaie6UI/baseline/7D-gxaie6UI/video_complete.json \
  --parent-meta-prompt-id legacy-read-only-smoke-parent \
  --provider "$PD_FEEDBACK_PROVIDER" \
  --model-id "$PD_FEEDBACK_MODEL_ID" \
  --tokenizer-path "$PD_FEEDBACK_TOKENIZER_PATH" \
  --tokenizer-identity "$PD_FEEDBACK_TOKENIZER_IDENTITY" \
  --context-limit "$PD_FEEDBACK_CONTEXT_LIMIT" \
  --maximum-output-tokens "$PD_FEEDBACK_MAX_OUTPUT_TOKENS" \
  --generation-settings-json "$PD_FEEDBACK_GENERATION_SETTINGS_JSON" \
  --feedback-policy-version "$PD_FEEDBACK_POLICY_VERSION"
```

### One explicitly operator-run episode-feedback call

The command below is the standalone one-call path for the saved episode used
by the D1–D2 inspections. It is not connected to optimization runtime, updater,
or canonical persistence. It sends only the D1.6 system prompt and
`model_compact` payload. The API key is read only from `OPENAI_API_KEY`.

Choose a new local output directory. The command refuses to overwrite an
existing directory. On success it writes `request.json`, `provider_response.json`,
`raw_response.txt`, `parsed_feedback.json`, `usage.json`, and `manifest.json`
there. On HTTP, transport, or strict-parse failure it performs no retry, repair,
or fallback, returns non-zero, and preserves `raw_error.txt` plus
`manifest.json`. Source artifacts are hashed before and after and are never
written.

```bash
: "${OPENAI_API_KEY:?set the OpenAI API key in the environment}"
: "${EPISODE_FEEDBACK_OUTPUT_DIR:?set a new local output directory}"

PYTHONPATH=/home/intern/youngseo \
conda run -n local_llm_vllm python \
  scripts/run_episode_feedback_once.py \
  --intervention-result runs/phase4_memory_k4_isolated_001_output/interventions/7D-gxaie6UI/7D-gxaie6UI/candidate_284d53fde9070f949063_pbad21bc654ed/result.json \
  --baseline-manifest runs/phase4_memory_k4_isolated_001_output/baseline_videos/7D-gxaie6UI/baseline/7D-gxaie6UI/video_complete.json \
  --parent-meta-prompt-id legacy-read-only-smoke-parent \
  --output-dir "$EPISODE_FEEDBACK_OUTPUT_DIR" \
  --provider openai_api \
  --api-endpoint https://api.openai.com/v1/chat/completions \
  --model-id gpt-4.1-mini-2025-04-14 \
  --request-representation model_compact \
  --context-limit 1047576 \
  --maximum-output-tokens 4096 \
  --temperature 0.0 \
  --feedback-policy-version episode_feedback_request_v3_gpt41mini_v1 \
  --timeout-seconds 600
```

Success requires exit code zero, `status="success"`, both call counts equal to
one, equal source hashes, and a non-empty parsed feedback file. A failed command
must be inspected in its selected output directory; do not rerun it against the
same directory.

## Prompt-delta Checkpoint E1: provisional meta-prompt updater boundary

Checkpoint E1 consumes one immutable parent `MetaPromptVersion` and an ordered,
non-empty sequence of validated `EpisodeFeedback` records. It builds a
deterministic request and returns either a deterministic provisional candidate
identity or an explicit `no_update`. It does not reload episodes, captions, or
trajectories and does not persist, promote, confirm, or activate a candidate.

The intended episode-feedback model is `gpt-4o`, supplied explicitly at the D
provider boundary. E1 deliberately does not select or default an updater model;
an updater backend remains a separate explicit injection.

Focused verification:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_meta_prompt_updater.py
```

Checkpoint A–E1 regression verification:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_prompt_delta_schemas.py \
  tests/test_legacy_intervention_adapter.py \
  tests/test_episode_feedback.py \
  tests/test_llm_episode_feedback.py \
  tests/test_compact_episode_feedback.py \
  tests/test_model_compact_episode_feedback.py \
  tests/test_episode_feedback_provider.py \
  tests/test_meta_prompt_updater.py
```

Complete model-free suite:

```bash
conda run -n local_llm_vllm python -m pytest -q
```

These commands make no model/API call and create no meta-prompt candidate file,
pointer, updater state, confirmation artifact, or runtime configuration.

## Prompt-delta Checkpoint E2: one-call provisional updater artifact

Checkpoint E2 sends only the immutable parent `MetaPromptVersion` and the
ordered, validated `EpisodeFeedback` JSON records to an explicitly selected
updater backend. The updater model is independent of the `gpt-4o` episode-
feedback model and has no repository default. The command makes exactly one
strict-JSON provider attempt with no retry, repair, or fallback. It writes only
to a new operator-selected directory and never changes a current/active
pointer.

Focused and A–E2 regression verification (no API calls):

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_meta_prompt_update_execution.py \
  tests/test_meta_prompt_updater.py

conda run -n local_llm_vllm python -m pytest -q \
  tests/test_prompt_delta_schemas.py \
  tests/test_legacy_intervention_adapter.py \
  tests/test_episode_feedback.py \
  tests/test_llm_episode_feedback.py \
  tests/test_compact_episode_feedback.py \
  tests/test_model_compact_episode_feedback.py \
  tests/test_episode_feedback_provider.py \
  tests/test_meta_prompt_updater.py \
  tests/test_meta_prompt_update_execution.py

conda run -n local_llm_vllm python -m pytest -q
```

Operator-run single updater call. Every environment variable below is required;
in particular, no updater model, temperature, or output-token budget is
inferred from the feedback generator or captioner. Repeat
`--feedback-artifact "$EPISODE_FEEDBACK_ARTIFACT"` in the desired order when
supplying more than one feedback record.

```bash
: "${OPENAI_API_KEY:?export OPENAI_API_KEY for the updater provider}"
: "${PARENT_META_PROMPT_ARTIFACT:?absolute path to one MetaPromptVersion JSON}"
: "${EPISODE_FEEDBACK_ARTIFACT:?absolute path to one validated EpisodeFeedback JSON}"
: "${META_PROMPT_UPDATER_MODEL:?explicit updater model or snapshot ID}"
: "${META_PROMPT_UPDATER_POLICY_VERSION:?reviewed updater policy version}"
: "${META_PROMPT_UPDATER_TEMPERATURE:?explicit updater temperature}"
: "${META_PROMPT_UPDATER_MAX_OUTPUT_TOKENS:?explicit positive output-token budget}"
: "${META_PROMPT_UPDATE_OUTPUT_DIR:?new local output directory}"

conda run --no-capture-output -n local_llm_vllm \
  python scripts/run_meta_prompt_update_once.py \
  --provider openai_api \
  --api-endpoint https://api.openai.com/v1/chat/completions \
  --model-id "$META_PROMPT_UPDATER_MODEL" \
  --parent-meta-prompt "$PARENT_META_PROMPT_ARTIFACT" \
  --feedback-artifact "$EPISODE_FEEDBACK_ARTIFACT" \
  --updater-policy-version "$META_PROMPT_UPDATER_POLICY_VERSION" \
  --temperature "$META_PROMPT_UPDATER_TEMPERATURE" \
  --maximum-output-tokens "$META_PROMPT_UPDATER_MAX_OUTPUT_TOKENS" \
  --candidate-created-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --output-dir "$META_PROMPT_UPDATE_OUTPUT_DIR" \
  --timeout-seconds 600
```

On `update`, the new directory contains `updater_request.json`,
`provider_request.json`, `raw_response.txt`, `provider_response.json`,
`parsed_meta_prompt_update_result.json`, `provisional_meta_prompt.json`,
`usage.json`, `input_manifest.json`, and `run_manifest.json`. On `no_update`,
`no_update.json` replaces `provisional_meta_prompt.json`. Success requires a
zero exit, `run_manifest.json` with `status="succeeded"` and
`provider_call_count=1`, equal before/after source hashes, and a provisional
record whose parent is the input meta-prompt. Failure exits non-zero and writes
`raw_error.txt` plus a failed run manifest when the output directory was
created. Never reuse an existing output directory; inspect it and choose a new
directory for a separately authorized call.

### Grounded five-episode feedback regeneration

The grounded response contract includes a deterministic
`qa_transition_summary` in the model-facing request and requires every
`qa_transition` evidence item to declare exactly one `transition_type`.
Successful one-call runs also write `semantic_eligibility.json`. The updater
batch verifies this sidecar against the canonical parsed-feedback hash, prints
every ineligible input and reason, and refuses to call the updater when fewer
than two eligible records remain. Neither script retries, repairs, falls back,
promotes a candidate, or changes a runtime pointer.

Offline verification (no API calls):

```bash
conda run --no-capture-output -n local_llm_vllm \
  python -m pytest -q \
  tests/test_prompt_delta_schemas.py \
  tests/test_episode_feedback.py \
  tests/test_llm_episode_feedback.py \
  tests/test_compact_episode_feedback.py \
  tests/test_model_compact_episode_feedback.py \
  tests/test_episode_feedback_provider.py \
  tests/test_meta_prompt_updater.py \
  tests/test_meta_prompt_update_execution.py

bash -n scripts/run_selected_episode_feedback_batch.sh
bash -n scripts/run_selected_meta_prompt_update.sh
```

Operator-run commands (the first command performs five authorized provider
calls, one per reviewed production episode; the second performs at most one
updater call):

```bash
cd /home/intern/youngseo/surrogate_rollout
export SELECTED_FEEDBACK_BATCH_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)" && bash /home/intern/youngseo/surrogate_rollout/scripts/run_selected_episode_feedback_batch.sh

cd /home/intern/youngseo/surrogate_rollout
bash /home/intern/youngseo/surrogate_rollout/scripts/run_selected_meta_prompt_update.sh "/home/intern/youngseo/surrogate_rollout/runs/selected_episode_feedback_$SELECTED_FEEDBACK_BATCH_TIMESTAMP"
```

Run both commands in the same shell so the explicit timestamp variable is
preserved. An older ungrounded feedback root has no eligibility sidecars and
is rejected before the updater transport.

## Prompt-delta Checkpoint F: paired confirmation and atomic promotion

`scripts/run_prompt_delta_iteration.py` connects grounded episode feedback,
the ordered meta-prompt updater decision, an independent paired confirmation
set, and atomic promotion/rollback. It does not select a model, decoding
configuration, confirmation sample count, accuracy threshold, regression
guard, cache-reset identity, or evaluator pipeline. Real components must be
supplied together by one reviewed `module:callable` factory returning
`(feedback_generator, updater, confirmation_evaluator)`. The repository does
not currently declare such a real factory, so the command below is the fully
bounded deterministic dry-run and performs no API/model call.

The paired evaluator must echo the exact request identity, ordered cases,
model identity, decoding settings, cache/reset identity, and pipeline identity.
Update and confirmation video/QA identities must be disjoint. A QA flip with
identical parent/candidate captions is recorded as attribution-uncertain no-op
evidence and is neutralized for promotion scoring; it is never counted as
caption-strategy benefit or regression.

Focused and A–F regression tests:

```bash
cd /home/intern/youngseo/surrogate_rollout
PYTHONPATH=/home/intern/youngseo:/home/intern/youngseo/surrogate_rollout conda run --no-capture-output -n local_llm_vllm python -m pytest -q tests/test_prompt_delta_iteration.py tests/test_meta_prompt_updater.py

cd /home/intern/youngseo/surrogate_rollout
PYTHONPATH=/home/intern/youngseo:/home/intern/youngseo/surrogate_rollout conda run --no-capture-output -n local_llm_vllm python -m pytest -q tests/test_prompt_delta_schemas.py tests/test_legacy_intervention_adapter.py tests/test_episode_feedback.py tests/test_llm_episode_feedback.py tests/test_compact_episode_feedback.py tests/test_model_compact_episode_feedback.py tests/test_episode_feedback_provider.py tests/test_meta_prompt_updater.py tests/test_meta_prompt_update_execution.py tests/test_prompt_delta_iteration.py
```

Complete deterministic dry-run with explicit criteria and pair identity:

```bash
cd /home/intern/youngseo/surrogate_rollout
CHECKPOINT_F_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
PYTHONPATH=/home/intern/youngseo:/home/intern/youngseo/surrogate_rollout conda run --no-capture-output -n local_llm_vllm python /home/intern/youngseo/surrogate_rollout/scripts/run_prompt_delta_iteration.py \
  --iteration-id "checkpoint_f_dry_run_$CHECKPOINT_F_TIMESTAMP" \
  --parent-meta-prompt /home/intern/youngseo/surrogate_rollout/tests/fixtures/prompt_delta_iteration/parent.json \
  --update-episode /home/intern/youngseo/surrogate_rollout/tests/fixtures/prompt_delta_iteration/update_episode.json \
  --confirmation-cases /home/intern/youngseo/surrogate_rollout/tests/fixtures/prompt_delta_iteration/confirmation_cases.json \
  --output-dir "/tmp/prompt_delta_checkpoint_f_output_$CHECKPOINT_F_TIMESTAMP" \
  --state-dir "/tmp/prompt_delta_checkpoint_f_state_$CHECKPOINT_F_TIMESTAMP" \
  --candidate-created-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --model-identity fixture-model-not-called \
  --decoding-settings /home/intern/youngseo/surrogate_rollout/tests/fixtures/prompt_delta_iteration/decoding_settings.json \
  --cache-reset-identity fixture-clean-cache-v1 \
  --evaluation-pipeline-identity fixture-paired-pipeline-v1 \
  --minimum-confirmation-samples 2 \
  --minimum-accuracy-delta 0.0 \
  --maximum-correct-to-wrong 0 \
  --require-no-execution-failures true \
  --initialize-parent-pointer \
  --dry-run \
  --mock-candidate-meta-prompt "Inspect current frames and bounded caption history for visible continuity before generating a concise segment instruction." \
  --mock-confirmation-outcomes /home/intern/youngseo/surrogate_rollout/tests/fixtures/prompt_delta_iteration/mock_confirmation_outcomes.json
```

The output is write-once. Repeating the exact command with the same timestamp
resumes the completed result without feedback, updater, or evaluator calls.
Promotion writes the confirmed version beneath the isolated state directory
and atomically replaces only `current_meta_prompt.json`; rollback writes a
rejected candidate while retaining the parent pointer. No legacy codebook or
router state is read or written.

## Prompt-delta Checkpoint G: real DVD paired confirmation pilot

`DVDMetaPromptConfirmationEvaluator` connects the provisional meta-prompt to
the existing history-aware Qwen caption pipeline and `run_dvd_qa`. Parent and
candidate share one immutable two-video/six-QA bundle, frame files,
transcripts, sampling, captioner and DVD model/decoding configuration. They
independently regenerate free-form prompts, on-policy sequential histories,
captions, caption databases, cache identities, and downstream QA runs. The
candidate text is supplied only as the free-form generator meta-prompt; an
empty compatibility bank/router record satisfies the preserved builder API,
but no legacy property is routed or selected.

The operator script freezes the five already-saved production interventions,
derives held-out videos `g1VFfVsZt7w` and `jIx5Zi84Z3Q` from
`split_manifest.json`, and writes resolved model, decoding, history,
transcript, sampling, worker, timeout, context, output-token and promotion
settings into an immutable input manifest before any provider/model stage.
`gpt-4o` feedback is attempted once per episode and the `gpt-4o` updater at
most once. If the updater returns `no_update`, no confirmation runtime starts.
There is no retry, JSON repair, fallback, truncation, sampling, or request
splitting.

Offline verification only:

```bash
cd /home/intern/youngseo/surrogate_rollout
PYTHONPATH=/home/intern/youngseo:/home/intern/youngseo/surrogate_rollout conda run --no-capture-output -n local_llm_vllm python -m pytest -q tests/test_dvd_meta_prompt_confirmation.py tests/test_prompt_delta_iteration.py tests/test_free_form_instruction.py tests/test_checkpoint3a_history_aware_baseline.py
bash -n /home/intern/youngseo/surrogate_rollout/scripts/run_checkpoint_g_pilot.sh
PYTHONPATH=/home/intern/youngseo:/home/intern/youngseo/surrogate_rollout conda run --no-capture-output -n local_llm_vllm python /home/intern/youngseo/surrogate_rollout/scripts/prepare_checkpoint_g_pilot.py --help
PYTHONPATH=/home/intern/youngseo:/home/intern/youngseo/surrogate_rollout conda run --no-capture-output -n local_llm_vllm python /home/intern/youngseo/surrogate_rollout/scripts/run_prompt_delta_iteration.py --help
```

Operator-run pilot (paid API plus local Qwen/DVD work; Codex does not run it):

```bash
cd /home/intern/youngseo/surrogate_rollout
export CHECKPOINT_G_PILOT_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
bash /home/intern/youngseo/surrogate_rollout/scripts/run_checkpoint_g_pilot.sh
```

Resume after interruption in the same shell by preserving the exact timestamp;
completed feedback rows, updater result, paired QA rows, or a completed
iteration are loaded without repeating those calls:

```bash
cd /home/intern/youngseo/surrogate_rollout
: "${CHECKPOINT_G_PILOT_TIMESTAMP:?set this to the original YYYYMMDD_HHMMSS value}"
bash /home/intern/youngseo/surrogate_rollout/scripts/run_checkpoint_g_pilot.sh
```

Inspect the terminal state and paired confirmation (the second file exists only
when the updater returned `update`):

```bash
cd /home/intern/youngseo/surrogate_rollout
jq . "/home/intern/youngseo/surrogate_rollout/runs/checkpoint_g_pilot_${CHECKPOINT_G_PILOT_TIMESTAMP}_output/iteration_result.json"
jq '{aggregate,qa_results}' "/home/intern/youngseo/surrogate_rollout/runs/checkpoint_g_pilot_${CHECKPOINT_G_PILOT_TIMESTAMP}_output/confirmation/dvd_confirmation_manifest.json"
jq . "/home/intern/youngseo/surrogate_rollout/runs/checkpoint_g_pilot_${CHECKPOINT_G_PILOT_TIMESTAMP}_state/current_meta_prompt.json"
```

All four timestamped input/output/state/cache roots are pairwise distinct.
Caption cache entries are content-addressed beneath the isolated cache root;
parent and candidate meta-prompt identities produce distinct router/prompt
cache identities. Immutable decoded frames may be shared, while generated
prompts, histories, captions, caption databases, and QA runs are never shared
between policies. A completed manifest is write-once; a different request with
the same output directory fails rather than overwriting it.

## Active data policy

Older commands or artifacts that use three proposal videos plus separate
confirmation, regression, and final-evaluation train videos describe the
superseded matched pilot.

The frozen split remains:

```text
train:       10 videos / 30 QAs
validation:  10 videos / 30 QAs
test:        10 videos / 30 QAs
```

Checkpoint 2 derives eight `previously_cached` train videos as the evidence
pool and the other two train videos as confirmation. Each production iteration
selects `K` unique evidence videos and runs all `3K` QAs. `K=3` is the default
pilot setting, not a method constraint. Confirmation runs only
after all eight evidence videos have appeared since the last confirmation.
There is no separate regression-video subset.

The main experiment, real feedback calls, component updates, and confirmation
evaluation are not available through Checkpoint 3D. Any older command that
launches those paths is superseded and must not be used as the active method.

Exact active train roles:

```text
Evidence:     0RxMZBLeqRI 7D-gxaie6UI GLW9omJfAdk TGom0uiW130
              pU_yyadYgG8 w0Wmc8C0Eq0 wCkQ138sg6M xKiRmesHWIA
Confirmation: g1VFfVsZt7w jIx5Zi84Z3Q
```

Checkpoint 2 focused tests:

```bash
conda run -n local_llm_vllm python -m pytest \
  tests/test_checkpoint2_train_roles.py \
  tests/test_checkpoint2_baseline_phase.py -q
```

Checkpoint 3B retrieval is authoritative and frame-only:

```text
exact candidate property text + sampled segment frames
→ SigLIP frame cosine scores
→ stable baseline caption-segment intersection
→ maximum frame pooling
→ top-M source-video segments (default M=5)
```

Frozen history is not part of the retrieval request, score, cache identity, or
resume identity. It remains used only by baseline routing, baseline captioning,
later selective re-captioning, and later multimodal feedback. Questions,
answers, correctness, traces, captions, and used segments are also prohibited
as direct retrieval inputs.

The intersection is applied before top-M ranking. `retrieval.json` uses
`property_frame_retrieval_v3` and records `segment_universe_policy`,
`baseline_segment_ids_hash`, `baseline_segment_count`, and
`excluded_visual_index_segment_ids`. Every `s_sim` ID must therefore exist in
the frozen baseline caption view. A changed baseline segment universe conflicts
with an existing retrieval artifact instead of reusing it.

Checkpoint 3B artifacts:

```text
<iteration>/
├── property_proposals/<video_id>/model_artifacts/
│   ├── request.json
│   ├── input_identity.json
│   ├── evidence_packing.json
│   ├── provider_request.json
│   ├── raw_output.txt
│   ├── parsed_output.json
│   ├── rejections.json
│   ├── validation.json
│   └── completed.json
└── property_retrieval/<video_id>/
    ├── manifest.json
    └── <opaque_candidate_property_id>/retrieval.json
```

An exact completed proposal skips the optimization-LLM provider. An exact
retrieval artifact skips the SigLIP text encoder. Conflicting property text,
source video, visual index, baseline segment universe, model/sampling identity,
top-k, pooling, or ranking configuration fails closed rather than overwriting
the artifact.

Focused verification:

```bash
conda run --no-capture-output -n local_llm_vllm \
  python -m pytest -q \
  tests/test_checkpoint3b_property_retrieval.py \
  tests/test_checkpoint2_baseline_phase.py
```

Success checks: filtered `retrieval.json` artifacts report
`segment_universe_policy=valid_baseline_caption_segment_intersection_v2`, and
every value in `s_sim` is a non-empty actual key in the frozen
`captions.json`. Scheduled segments with exhausted caption parsing are excluded.
Existing completed retrieval artifacts remain immutable and are not
reinterpreted.

Checkpoint 3C consumes the frozen baseline and one Checkpoint 3B retrieval
artifact per candidate. Its artifact layout is:

```text
<iteration>/property_interventions/<video_id>/
├── manifest.json
└── <opaque_candidate_property_id>_p<property_text_hash>/
    ├── work_item.json
    ├── composed_prompts.jsonl
    ├── frozen_histories.jsonl
    ├── caption_cache_keys.jsonl
    ├── mixed_view/
    ├── qa/<question_id>/
    ├── transitions.json
    └── result.json
```

Every candidate is independent. The candidate is appended after the incumbent
routed properties only for `S_sim`; its temporary sequence may be one entry
over the router maximum. Exact frozen baseline histories are reused without
candidate-caption propagation. Unselected segments retain incumbent captions.
Any selected-segment caption/validation failure or prompt-budget overflow fails
the complete candidate without incumbent fallback and does not abort siblings.

Completed candidate results resume without captioning or QA calls when the
candidate ID/text, parent baseline, retrieval, composed prompts, frozen
histories, caption configuration, and QA configuration match. Any collision on
those identities fails closed. Mixed-view registries are rebuilt from restored
per-segment incumbent/candidate registries in temporal order.

Checkpoint 3C fixture tests do not invoke a GPU or external model:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint3c_property_intervention.py
```

Checkpoint 3D reads completed Checkpoint 3C artifacts and emits feedback only
for `wrong_to_correct` and `correct_to_wrong`. It computes, in retrieval order:

```text
S_feedback = S_sim ∩ (S_used_before ∪ S_used_after)
```

An empty `S_feedback` is an explicit rejection with no provider call. Unchanged
correctness remains in analysis records but never creates a feedback request.
The default request bounds are five segments, two frames per segment, 64 KiB
per encoded frame, three history items, 1,200 characters per caption, three
reasoning events per side, four relevant codebook entries, and 400,000 total
serialized characters. All limits are configurable through
`FeedbackEvidenceBounds`.

Frames are EXIF-normalized, converted to RGB, resized with deterministic
LANCZOS sampling, and encoded using a fixed JPEG quality ladder and subsampling
policy. `FrameTransformConfig`, the Pillow version, source hash, transformed
hash, dimensions, selected quality, and resize step are persisted in every
frame payload and in run identity. A frame is rejected only if the configured
quality and resize ladder is exhausted while it remains over the byte limit.

```text
<iteration>/property_feedback/
├── manifest.json
└── <video_id>/<opaque_candidate_property_id>_p<property_text_hash>/
    ├── input_identity.json
    ├── qa/<question_id>/
    │   ├── request.json
    │   ├── raw_output.txt
    │   ├── parsed_output.json
    │   └── rejection.json
    └── result.json
```

`result.json` aggregates positive/negative flip counts, source and flip-QA
lineage, attributed segments, accepted evidence references, codebook coverage,
and recommendation evidence. It never applies a recommendation. Exact complete
or per-QA resume performs no provider call; conflicting or partial artifacts
fail closed. Failed property results are immutable and resume by default.
Passing `retry_failed=True` creates an isolated
`retries/retry_001/` artifact tree and never overwrites the original failure.

Checkpoint 3D fixture tests use only mocked feedback and fixture frame payloads:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint3d_interventional_feedback.py
```

Checkpoint 2 baseline artifacts:

```text
<iteration>/
├── policy_snapshot/
├── coverage/{before,after}.json
├── baseline/<video_id>/
│   ├── routing/
│   ├── caption_view/
│   ├── frames.json
│   ├── frozen_histories.jsonl
│   ├── baseline_qas.jsonl
│   ├── qa/<question_id>/
│   └── video_complete.json
├── property_proposals/<video_id>.json
├── iteration_state/provisional.json
└── manifest.json
```

A completed `manifest.json` with the same frozen-input fingerprint returns
without routing, captioning, QA, or proposal calls. Partial resume skips only
videos whose `video_complete.json` fingerprint matches and whose referenced
artifacts all still exist. A conflicting fingerprint or incomplete completion
marker fails closed rather than overwriting results.

## Checkpoint 1 compact property-memory sidecar

Checkpoint 1 is a fixture-tested, model-free sidecar. It does not run
captioning, QA, proposal, retrieval, feedback, the updater, confirmation, or a
production iteration. Construct `CompactPropertyMemoryRunner` and call
`run(...)` only after the referenced baseline and intervention artifacts are
complete. For an updater-decided iteration, pass the already-written update
plan and resulting bank so candidate promotion is recorded after, rather than
decided by, the memory layer.

```python
from surrogate_rollout.optimization.property_memory import (
    CompactPropertyMemoryRunner,
)

memory_result = CompactPropertyMemoryRunner().run(
    iteration_id=iteration_id,
    iteration_ordinal=iteration_ordinal,
    prompt_bank=input_prompt_bank,
    baseline_video_manifest_paths=baseline_video_manifest_paths,
    intervention_manifest_paths=intervention_manifest_paths,
    feedback_manifest_paths=feedback_manifest_paths,
    output_dir=f"{iteration_output}/compact_property_memory",
    parent_memory_path=parent_memory_snapshot_path,  # None for seed bootstrap
    update_plan=completed_update_plan,               # optional
    update_plan_path=completed_update_plan_path,     # optional, hashed lineage
    resulting_prompt_bank=completed_resulting_bank,  # optional
)
```

Artifact layout:

```text
<iteration>/compact_property_memory/
├── manifest.json                         # property_memory_manifest_v1
├── compact_summaries/
│   ├── correct_qa_property_credit.jsonl  # property_compact_summary_v3_runtime_validity
│   └── intervention_effects.jsonl        # property_compact_summary_v3_runtime_validity
└── property_memory/
    ├── snapshot.json                     # property_memory_v1 parent unit
    ├── selection_audit.json
    ├── properties/<property_id>.json
    └── candidates/<video_id>__<candidate_id>.json
```

Defaults are three strong and two weak positives, three harmful, two no-effect,
and two positive/two negative routing examples. Candidate positive, negative,
mixed, and no-effect categories use the corresponding small bounds. Ranking is
strength, distinct video, representative-signature diversity, then recency.
The selection audit explains retention and eviction; evicted compact examples
do not delete raw artifacts or immutable summary rows.

Every example contains source paths and SHA-256 hashes. The manifest binds raw
manifest hashes, parent snapshot hash, input/result bank hashes, optional
update-plan content/path hash, schema versions, bounds, and selection version.
Repeat the exact call for exact resume. A partial output directory, a changed
source or parent, an incompatible memory schema, or a missing/hash-mismatched
completed artifact fails closed. Do not point `parent_memory_path` at a legacy
Phase 4 artifact; legacy artifacts remain raw inputs and are not overwritten.

Focused fixture-only command:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint1_property_memory.py
```

The memory-conditioned LLM codebook updater is the following checkpoint below.

## Checkpoint 2 memory-conditioned LLM codebook updater

Configure the existing orchestrator with `CompactPropertyMemoryRunner` and
`MemoryConditionedLLMCodebookUpdater`, then invoke the post-feedback checkpoint
with already-completed iteration manifests. Tests use a callable mock provider;
no real provider is configured or called by this checkpoint implementation.

```python
orchestrator = Checkpoint3EOrchestrator(
    baseline_runner=baseline_runner,
    intervention_runner=intervention_runner,
    feedback_runner=feedback_runner,
    confirmation_evaluator=confirmation_evaluator,
    property_memory_runner=CompactPropertyMemoryRunner(),
    llm_codebook_updater=MemoryConditionedLLMCodebookUpdater(
        response_provider=reviewed_codebook_provider,
    ),
)

result = orchestrator.run_memory_conditioned_codebook_checkpoint(
    iteration_id=iteration_id,
    iteration_ordinal=iteration_ordinal,
    prompt_bank=input_prompt_bank,
    baseline_video_manifest_paths=baseline_video_manifest_paths,
    intervention_manifest_paths=intervention_manifest_paths,
    feedback_manifest_paths=feedback_manifest_paths,
    output_dir=f"{iteration_output}/memory_codebook_checkpoint",
    state_dir=policy_state_dir,
)
```

Prompt:

```text
optimization/prompts/codebook_updater_v1.txt
memory_codebook_updater_prompt_v3_deterministic_ids
```

Artifacts:

```text
<iteration>/memory_codebook_checkpoint/
├── manifest.json
├── compact_property_memory/
│   ├── manifest.json
│   ├── compact_summaries/
│   └── property_memory/snapshot.json
└── llm_codebook_updater/
    ├── manifest.json
    ├── input_identity.json
    ├── system_prompt.txt
    ├── request.json
    ├── response_attempts.json
    ├── attempts/attempt_NNN/{raw_response.txt,result.json}
    ├── raw_response.txt
    ├── parsed_plan.json
    ├── llm_proposed_plan.json
    ├── validation_report.json
    ├── structural_validation_result.json
    ├── rejected_actions.json
    ├── structural_errors.json
    ├── non_blocking_warnings.json
    ├── applied_plan.json
    ├── final_applied_plan.json
    ├── candidate_codebook.json
    ├── property_id_mapping.json
    └── promoted_property_memory.json

<state_dir>/property_memory/current.json
```

`property_id_mapping.json` contains:

```json
{
  "schema_version": "property_id_mapping_v2",
  "old_to_new_property_ids": {
    "unchanged_or_canonical_id": "same_or_canonical_id",
    "retired_id": null
  },
  "candidate_promotions": {
    "candidate_id": "validated_active_property_id"
  }
}
```

The state pointer tracks only compact-memory lineage for the unchanged active
input bank. It is not `confirmed/current.json` or
`active_provisional.json`. Candidate codebook output is not a production policy
and must not be passed to confirmation before the deferred router updater builds
and validates the matching router.

Focused mock-only command:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint1_property_memory.py \
  tests/test_checkpoint2_memory_codebook_updater.py
```

Real production wiring constrains the complete response with the codebook
strict JSON Schema and permits at most three parse attempts. Check
`response_attempts.json` for `accepted_attempt_index` and the immutable per-call
raw/error records. Repeat the exact command after an interruption: attempts
whose raw response already exists are not called again. A directory from the
older one-shot updater that lacks `input_identity.json` cannot be resumed;
choose a fresh output/state directory while reusing compatible caption/cache
roots.

## Checkpoint 3 memory-conditioned router prompt and atomic pair

After the Checkpoint 2 manifest is complete, configure the same orchestrator
with `MemoryConditionedLLMRouterUpdater` and invoke the router checkpoint. This
is a mock/provider integration boundary; this implementation task does not run
a real provider, GPU, confirmation, or regular production iteration.

```python
orchestrator = Checkpoint3EOrchestrator(
    baseline_runner=baseline_runner,
    intervention_runner=intervention_runner,
    feedback_runner=feedback_runner,
    confirmation_evaluator=confirmation_evaluator,
    llm_router_updater=MemoryConditionedLLMRouterUpdater(
        response_provider=reviewed_router_update_provider,
    ),
)

result = orchestrator.run_memory_conditioned_router_checkpoint(
    iteration_id=iteration_id,
    parent_router_policy=input_router_policy,
    codebook_checkpoint_manifest_path=(
        f"{iteration_output}/memory_codebook_checkpoint/manifest.json"),
    output_dir=f"{iteration_output}/memory_router_checkpoint",
    state_dir=policy_state_dir,
)
```

Prompt and versions:

```text
optimization/prompts/router_updater_v1.txt
memory_router_updater_prompt_v2_semantic_llm
structured_router_policy_v2_total_examples
history_aware_router_prompt_renderer_v2
rendered_router_prompt_v1
```

Artifacts:

```text
<iteration>/memory_router_checkpoint/
├── manifest.json                         # written only after atomic success
├── failure.json                          # failure only; no pair pointer
├── llm_router_updater/
│   ├── manifest.json
│   ├── parent_structured_router_policy.json
│   ├── input_identity.json
│   ├── system_prompt.txt
│   ├── request.json
│   ├── execution.json
│   ├── response_attempts.json
│   ├── attempts/attempt_NNN/{raw_response.txt,result.json}
│   ├── raw_response.txt
│   ├── parsed_plan.json
│   ├── llm_proposed_plan.json
│   ├── validation_report.json
│   ├── structural_validation_result.json
│   ├── rejected_actions.json
│   ├── structural_errors.json
│   ├── non_blocking_warnings.json
│   ├── applied_plan.json
│   ├── final_applied_plan.json
│   ├── structured_router_policy.json
│   ├── rendered_router_prompt.json
│   ├── rendered_router_prompt.txt
│   ├── router_prompt.diff
│   └── candidate_router_policy.json
└── provisional_policy_pair/
    ├── provisional_codebook.json
    ├── provisional_router.json
    └── policy_pair.json                  # atomic_provisional_policy_pair_v1

<state_dir>/memory_conditioned_provisional/current.json
```

The updater accepts only bounded memory/effect IDs and targets active IDs in
the candidate codebook. The LLM, not the validator, decides how evidence
supports selection/avoidance guidance or either stored example label. The
structural validator rejects stale IDs, unknown evidence, malformed or
conflicting writes, exact duplicates, retired guidance, mapping
inconsistencies, protocol or selection-limit changes, and rendered-prompt hash
mismatches. Ordinary `question` or `answer` words and evidence polarity are not
hard rejection rules.

If both `routing_memory_examples` and
`current_compact_routing_evidence` are empty, the default updater still calls
the provider in `execution.mode=llm_provider_no_evidence`; the LLM should
normally return an explicit empty plan. Constructing it with
`skip_llm_when_no_evidence=True` enables the cost/legacy optimization
`configured_empty_evidence_skip` with reason
`explicit_skip_llm_when_no_evidence`. Rendering, hashing, ID-space validation,
and atomic pair persistence still run in both modes. Check
`llm_router_updater/execution.json` or the updater manifest fields
`execution_mode`, `provider_called`, and `deterministic_reason`. An empty
`target_property_id` is never accepted and remains a strict parse failure.

The real router-updater provider uses its own strict response schema and the
same maximum of three parse attempts. Parser failures are auditable in
`response_attempts.json`; API/runtime failures stop immediately. Exact partial
resume reuses persisted raw attempts. Only the explicit no-evidence
optimization records `accepted_attempt_index=0` and consumes zero calls.

Inspect `llm_proposed_plan.json` for the untouched LLM decision,
`structural_validation_result.json` and `structural_errors.json` for blocking
schema/reference/executability failures, `non_blocking_warnings.json` for audit
hints that did not block application, and `final_applied_plan.json` for the
exact applied actions. Runtime and intervention failures remain in the updater
request as reliability-only evidence; they must not become `wrong_to_wrong` or
harmful examples.

The provisional pair references/hashes its parent pair, property memory,
candidate codebook, complete ID mapping, structured policy, rendered prompt,
and both updater plans and validation reports. The final manifest and separate
pointer are written only after both components validate in one property-ID
space. A failure preserves all diagnostic artifacts, does not write the pair
pointer, and cannot resume as success. Exact repeat of a completed invocation
verifies closure and performs no provider call. This path never modifies
`confirmed/current.json` or coverage-cycle `active_provisional.json`.

Focused mock-only command:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint3_memory_router_updater.py
```

### One-video Checkpoint 3 real smoke

The bounded runner now opts into the memory-conditioned path only in
`scripts/run_phase4_bounded_smoke.py`; fixture and legacy callers retain the
deterministic provisional-update path unless they explicitly set
`memory_conditioned_update=True`. The real smoke uses one call each for the
codebook and router updater, then makes exactly one post-commit router call
through the still-active persistent GPU pool to prove that the rendered prompt
is consumed. Worker cleanup is recorded in `worker_cleanup.json`.

Use isolated output/state roots and the compatible read-only caption cache:

```bash
conda run -n local_llm_vllm python -m dotenv run -- python \
  scripts/run_phase4_bounded_smoke.py \
  --post-intervention-mode provisional_update \
  --video-id wCkQ138sg6M --gpu 4 --gpus 4,5 \
  --output-dir runs/phase4_memory_router_smoke_wCkQ138sg6M_v2_output \
  --state-dir runs/phase4_memory_router_smoke_wCkQ138sg6M_v2_state \
  --cache-dir runs/phase4_one_video_smoke_json_cache
```

Repeat that exact command for the resume audit. A successful repeat performs
no proposal, feedback, codebook-updater, router-updater, routing-probe,
captioning, or QA model call. Inspect:

```text
runs/phase4_memory_router_smoke_wCkQ138sg6M_v2_output/
├── mode_manifests/provisional_update.json
├── memory_codebook_checkpoint/
├── memory_router_checkpoint/
│   └── router_prompt_consumption_probe.json
└── worker_cleanup.json
```

`mode_manifests/provisional_update.json` records property-memory, both updater
manifests, the rendered-prompt hash, atomic pair paths, updater call counts,
and the before/after confirmed-pointer hash. The smoke fails if that pointer
changes or if the post-commit router call does not carry the new prompt hash.

## 1. Repository and environment

```bash
cd /home/intern/youngseo/surrogate_rollout
git status
git diff
git log --oneline -5
```

Known test environment:

```bash
conda run -n local_llm_vllm python -m pytest tests
```

Confirm the actual working directory and import path.

The fixture smoke requires no model credentials or GPU. A later real run must
set the GPU and feedback model explicitly; the API key is read by the selected
provider and is never persisted:

```bash
export CUDA_VISIBLE_DEVICES=0
export SR_FEEDBACK_MODEL=gpt-4o
export OPENAI_API_KEY='<set in the shell only>'
```

Do not persist secrets in configs or artifacts.

## 2. Focused tests

The final command must cover:

- configurable-size evidence baseline orchestration;
- per-video multi-property proposals;
- frame-only property-conditioned retrieval;
- independent frozen-history interventions;
- all-QA reruns;
- flip-only feedback;
- multi-property codebook update;
- mixed-view/cache regression.

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint3e_final_iteration.py \
  tests/test_startup_models.py \
  tests/test_bounded_smoke.py \
  tests/test_checkpoint3d_interventional_feedback.py \
  tests/test_checkpoint3c_property_intervention.py \
  tests/test_checkpoint3b_property_proposal.py \
  tests/test_checkpoint3b_property_retrieval.py \
  tests/test_checkpoint3a_history_aware_baseline.py \
  tests/test_checkpoint2_baseline_phase.py \
  tests/test_checkpoint2_train_roles.py
```

## 3. Complete regression suite

```bash
conda run -n local_llm_vllm python -m pytest tests
```

Record the exact pass count.

## 4. Offline dry run

The dry run must build one iteration plan without captioning, QA APIs, or
persistent state mutation.

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint3e_final_iteration.py::test_fixture_end_to_end_accept_reject_rollback_and_resume
```

Confirm that artifacts show:

- the default three source videos;
- multiple candidate properties allowed per video;
- one work item per property-source-video pair;
- isolated intervention output paths;
- frozen policy/history versions;
- no scaffold candidate;
- no main experiment launch.

## 5. One-video bounded smoke

Prerequisite: store `OPENAI_API_KEY` in the repository-local `.env` with file
mode `600`. The commands use `python-dotenv` to inject it only into the child
process; they never print or persist the key. All three commands use evidence video
`0RxMZBLeqRI` and its three frozen train QAs, `max_proposals=1`, retrieval
top-k one, and at most one intervention. Output, state, and cache roots are
separate siblings. Reuse the same three paths when progressing modes.

The first 2026-07-17 pre-schema attempt reached real-model startup validation on GPU
4 but stopped on the first segment because unconstrained Qwen output was not
strict JSON. Its partial immutable roots are
`runs/phase4_one_video_smoke_{output,state}` and must not be reused or
overwritten. Router version `router_v9002` uses vLLM JSON Schema structured
decoding with fallback disabled.

The second 2026-07-17 attempt validated schema-constrained routing over all 184
segments and completed all three baseline QAs. It then stopped before the
OpenAI property-proposal call because the 71,767-character request exceeded the
40,000-character hard limit. Its partial immutable output/state roots are
`runs/phase4_one_video_smoke_json_{output,state}`. Proposer version
`multi_property_proposer_v2` now persists deterministic truncation metadata and
reduces that frozen request to 23,878 characters by retaining three reasoning
events per QA and twelve relevant captions. Use the new `*_json_v2_*`
output/state roots below. Reuse the identity-matched
`runs/phase4_one_video_smoke_json_cache`; completed caption entries are
immutable and avoid regenerating the same 184 captions.

The third 2026-07-17 `qa_only` invocation completed successfully at
`runs/phase4_one_video_smoke_json_v2_output/run_manifest.json`. It validated
184 schema-constrained router calls, 184 caption-cache hits, three baseline QAs,
the 23,783-character persisted proposal request, and zero confirmation calls.
The provider returned one proposal, which the strict proposal policy rejected
as already covered by active `pe_default`; therefore retrieval, intervention,
and candidate-QA work-item lists are valid but empty. This is an allowed
zero-proposal completion, not evidence that the real intervention path ran.

`multi_property_proposer_v3` supersedes that text-only proposal request. Its
model-visible input contains no source-video, question, segment, priority,
tool-call, or truncation identifiers. It sends, per QA, question and
answer/prediction context, three bounded sanitized reasoning events, and three
actual used-segment representative images paired with baseline captions, plus
the current codebook. Offline reconstruction from the v2 baseline produced a
13,388-character text payload and nine bounded image blocks. Use new
`*_multimodal_*` output/state roots below; reuse the identity-matched caption
cache.

`multi_property_proposer_v4` changes coverage handling without weakening
deterministic duplicate or knowledge-grounding checks. The provider field
`covered_by_existing_property_ids` now means possible relation or coverage and
is parsed and persisted as non-binding `coverage_hints`. Exact normalized text
matches and active-ID collisions still reject before retrieval, as do malformed
lineage, instance leakage, and non-visual/external/background/historical
knowledge instructions. Semantic coverage is assessed only from post-
intervention correctness-flip feedback using Checkpoint 3D
`coverage_assessment` and `covered_by_property_ids`. Existing serialized
`covered_by_existing_property_ids` proposal records remain readable through a
legacy alias; completed top-level bounded-smoke manifests retain their normal
exact-resume behavior.

`multi_property_proposer_v5` separates naming from identity. The model returns
`suggested_property_id`; the parser preserves it only as a readable hint and
derives `candidate_<20 hex characters>` under
`opaque_candidate_proposal_id_v1`. That opaque handle is used in retrieval,
intervention, feedback, memory, and updater artifact paths. The same frozen
proposal resumes to the same handle, while identical suggestions from distinct
videos remain distinct. The codebook updater, not the proposer, determines the
eventual active property ID. Start a fresh output/state root for v5; old
completed proposal artifacts are not reinterpreted.

`multi_property_proposer_v6` changes only the QA-evidence selection boundary.
For every QA it includes the complete time-ordered union of
`explicitly_cited_segments` and `frame_inspected_segments`; it no longer mixes
in `returned_segments`, falls back to general consumed/used segments, or stops
after three evidence items. Every selected segment still contributes one
middle source frame and one individually truncated incumbent caption, with
private segment/frame lineage in `input_identity.json`. The request and private
identity schemas are `multimodal_property_proposal_request_v5` and
`multimodal_property_proposal_identity_v2`; the selection policy is
`all_explicitly_cited_and_frame_inspected_segments_v1`. The text envelope is
250,000 characters while per-caption and per-image bounds remain unchanged.
Start a fresh output/state root for v6. Existing v5 proposal artifacts and
completed runs remain immutable and are not resumed under v6 semantics.

`multi_property_proposer_v7` makes each proposal an atomic instruction plus
applicability contract. New provider output contains only
`suggested_property_id`, `property_text`, strict `applicability`
(`when`, `positive_cues`, `negative_cues`, `required_modalities`), non-binding
`covered_by_existing_property_ids`, and natural-language `failure_analysis`.
Allowed modalities are exactly `frames`, `transcript`, and `caption_history`;
topic/genre/source-specific applicability and unavailable external knowledge
are rejected. The proposer returns zero proposals when the observed failure is
not attributable to a repairable caption omission.

For captions containing `\n\nTranscript during this video clip:`, `request.json`
stores the prefix as `baseline_generated_description` and the suffix as
`source_transcript`, subject only to the existing per-field character bound;
the suffix is source evidence rather than generated-caption quality. Versions
are `multimodal_property_proposal_request_v6`,
`multimodal_property_proposal_identity_v3`,
`opaque_candidate_proposal_id_v2`, `multi_property_proposal_artifact_v2`, and
`property_proposer_meta_prompt_v2`. Applicability participates in the opaque
candidate hash and is preserved in parsed proposal, feedback v3, candidate
memory, codebook-updater input, and validated added `PromptEntry` metadata.
Legacy proposal fields load only through compatibility adapters and are not
written in new artifacts. Start a fresh output/state/cache root for v7; v6
completed artifacts fail closed rather than being reinterpreted. Router-rule
generation from applicability is intentionally not part of this change.

`multi_property_proposer_v8` changes proposal cardinality without changing the
candidate object or downstream intervention/update semantics. Every
runtime-valid incorrect QA with focused evidence requires one or two exploratory
candidate hypotheses. The flat video-level response must therefore retain
`N..2N` validator-accepted candidates for `N` eligible incorrect QAs; an empty
response or candidates all removed by deterministic validation fail the proposal
stage. Zero remains valid for all-correct videos and for explicitly invalid
incorrect QAs only: runtime failure, missing evidence, malformed input, or a
clearly unreliable annotation marker. Possible reasoning/retrieval/tool-use
failure and existing-property coverage no longer suppress exploration. Versions
are `multimodal_property_proposal_request_v7`,
`multimodal_property_proposal_identity_v4`,
`multi_property_proposal_artifact_v3`, and
`property_proposer_meta_prompt_v3`. Start a fresh output/state/cache root; older
candidate artifacts remain loadable, but completed v7 proposal calls are not
reinterpreted under v8 request semantics.

`multi_property_proposer_v10` generates executable intervention candidates for
every eligible correct or incorrect QA. Each provider proposal includes a
source-free `source_qa_slot`; private identity maps it to one actual question ID
and baseline correctness. Every eligible slot requires one or two candidates.
Correct-sample candidates may preserve evidence, improve clarity/ordering,
reduce repetition or verbosity, and suppress irrelevant/speculative content.
Incorrect-sample candidates may recover missing or weakly represented evidence.

Pre-intervention validation is structural only. It rejects malformed/missing
fields, empty text, unresolved placeholders, exact same-slot duplicates, invalid
slot lineage, and instructions requiring unavailable captioner tools or inputs.
It does not reject lexical QA overlap, active-codebook similarity, correct-sample
origin, uncertain benefit, weak generality, or possible repetition. Semantic
accept/reject/revise/merge decisions remain downstream. A cardinality failure
still writes `parsed_output.json`, `rejections.json`, and `validation.json`
before failing; no `completed.json` is written.

If the initial response omits an eligible `source_qa_slot`,
`missing_qa_slot_retry_v1` makes at most two calls containing only the missing
QA rows. Inspect `model_artifacts/missing_slot_retries.json` and
`model_artifacts/missing_slot_retries/attempt_NNN/`. Existing candidates are
not regenerated. Exhaustion remains a hard cardinality failure; exact resume
reuses every persisted provider response.

Versions are `multimodal_property_proposal_request_v9`,
`multimodal_property_proposal_identity_v5`,
`opaque_candidate_proposal_id_v3`, `multi_property_proposal_artifact_v5`, and
`property_proposer_meta_prompt_v4`. Intervention provenance uses
`property_intervention_transitions_v3` and `property_intervention_result_v3`;
bounded updater provenance uses `property_compact_summary_v3_runtime_validity`
and `memory_codebook_updater_request_v4`. Start a fresh output/state directory for
v10. Compatible caption caches may be reused, but a completed v9 proposal artifact
must not be reinterpreted or overwritten.

`multi_property_proposer_v11` compacts provider evidence without deleting source
provenance. `request.json` identifies evidence by `[start–end]` timestamp range
rather than stable segment ID. Explicitly cited intervals appear first with
normalized fuller `generated_description` and a transcript only when it is
non-empty and not substantially duplicative. Inspected-only intervals follow as
single-line `[start–end] description | tx: transcript` summaries, bounded to 250
description characters and 120 transcript characters. Ordering is deterministic
by QA and timestamp and no secondary LLM is called.

Evidence text uses at most 200,000 characters and may use less so the existing
250,000-character total envelope retains room for task, schema, QA context, and
JSON structure. Packing stops after cited evidence and then inspected-only
evidence when that budget is exhausted. Inspect:

```text
<proposal_artifact_dir>/
├── request.json             # compact provider-visible intervals
├── input_identity.json      # every original interval and full provenance
└── evidence_packing.json    # included/omitted totals and per-QA counts
```

`input_identity.json` retains the original caption, generated-description and
transcript split, stable segment ID, timestamp, evidence role, frame path and
hashes, plus `included_in_payload` and `omission_reason`. The provider request
contains none of those private stable IDs. Versions are
`multimodal_property_proposal_request_v10`,
`multimodal_property_proposal_identity_v6`,
`multi_property_proposal_artifact_v6`, and
`compact_cited_then_inspected_evidence_v2`. Use a fresh output/state directory;
compatible caption caches remain reusable.

Fixture-only verification (no GPU/API calls):

```bash
conda run --no-capture-output -n local_llm_vllm \
  python -m pytest -q \
  tests/test_checkpoint3b_property_proposal.py \
  tests/test_checkpoint3c_property_intervention.py \
  tests/test_checkpoint3d_interventional_feedback.py \
  tests/test_checkpoint1_property_memory.py \
  tests/test_checkpoint2_memory_codebook_updater.py
```

`flip_only_property_feedback_v3` carries the proposal `coverage_hints`, strict
applicability, and `failure_analysis` into
the bounded feedback request and aggregate artifact as context and lineage.
They do not populate `covered_by_property_ids`; only the post-intervention
feedback response can do that.

History-block-parallel baseline captioning is available through `--gpus`. The
comma-separated list creates at most one run-scoped persistent Qwen/vLLM worker
per GPU;
segments inside each 300-second history block stay sequential, while distinct
blocks run concurrently. `--gpu` selects the primary GPU for DVD QA; selective
caption interventions are sent back to the persistent caption pool so the
parent never loads a competing Qwen instance. Workers remain loaded across
baseline videos, proposals, retrieval, interventions, and confirmation, and
shut down only at the explicit run boundary. Each block resumes from
`history_aware_baseline/parallel_history_blocks/block_<index>/segment_state/`.
Workers write `worker_<index>_cache_manifest.jsonl` fragments, and only the
parent merges them into the configured caption manifest after all workers
succeed. `routing_manifest.json` records the scheduler version, worker/GPU and
block counts, and deterministic merge rule. GPU assignment is not part of the
semantic caption-cache key.

The current worker backend runs vLLM with:

```text
qwen25_vl_mm_cache_disabled_v1
mm_processor_cache_gb=0
enable_prefix_caching=False
```

Do not resume an incomplete run created before this policy using the same
output/state roots. Its EngineCore may have died after the multimodal feature
LRU diverged from the frontend metadata LRU, and already completed failed
interventions remain immutable. Stop that process and launch the same video
list with fresh output/state/cache roots. Reuse a completed caption cache only
when its backend identity matches this policy.

The first multi-GPU attempt on 2026-07-17 completed all 184 segment states but
stalled before the parent merge because the parent joined one-build workers
whose vLLM EngineCore processes remained alive. The run-scoped worker protocol
supersedes that lifecycle: the parent consumes completion messages without a
join, and the bounded runner closes workers in `finally`. Stop the old hung
invocation, then repeat the exact QA-only command below with the same three
roots. All completed block states and caption caches resume; do not delete or
rename them.

That resumed attempt reached QA but exposed a second lifecycle issue: DVD
`frame_inspect_tool` tried to initialize a parent Qwen on GPU 4 while the
persistent worker already occupied it. Two QAs failed with null predictions,
yet the older baseline contract treated them as ordinary incorrect answers and
called the proposer. The current adapter routes raw frame inspection through
the worker pool and fails before proposal on runtime errors, null predictions,
or parse failures. When the exact command below sees the invalid completed QA
mode, it moves only invalid QA/proposal/retrieval/intervention/mode artifacts to
`invalid_attempts/qa_execution_failure_<NNN>/`, leaves the 184 caption artifacts
in place, reruns all three QAs, and regenerates the proposal only after all
three succeed.

The repaired 2026-07-17 resume completed successfully in `qa_only` mode at
`runs/phase4_one_video_smoke_parallel_multimodal_output/run_manifest.json`.
Its merged routing manifest records 184 resumed segments, zero router calls,
and zero caption calls. All three QA executions returned non-null parsed
answers with empty error lists: QAs 9 and 11 were correct, and QA 10 was a
valid incorrect answer. Only then did the v3 proposer regenerate its request
and provider artifacts. The provider returned `pe_historical_context`, which
the strict policy rejected as covered by active `pe_default`, so the completed
mode contains zero accepted proposals and no intervention. The archived failed
attempt remains recoverable under
`invalid_attempts/qa_execution_failure_001/`. Confirmation was disabled and no
production state or coverage pointer was written.

Secure the environment file once:

```bash
chmod 600 .env
```

QA only:

```bash
conda run -n local_llm_vllm python -m dotenv run -- python \
  scripts/run_phase4_bounded_smoke.py \
  --post-intervention-mode qa_only \
  --video-id 0RxMZBLeqRI --gpu 4 --gpus 4,5,6,7 \
  --output-dir runs/phase4_one_video_smoke_parallel_multimodal_output \
  --state-dir runs/phase4_one_video_smoke_parallel_multimodal_state \
  --cache-dir runs/phase4_one_video_smoke_json_cache
```

Continue through flip-only feedback and property aggregation:

```bash
conda run -n local_llm_vllm python -m dotenv run -- python \
  scripts/run_phase4_bounded_smoke.py \
  --post-intervention-mode feedback_only \
  --video-id 0RxMZBLeqRI --gpu 4 --gpus 4,5,6,7 \
  --output-dir runs/phase4_one_video_smoke_parallel_multimodal_output \
  --state-dir runs/phase4_one_video_smoke_parallel_multimodal_state \
  --cache-dir runs/phase4_one_video_smoke_json_cache
```

Continue through isolated provisional bank/router artifacts:

```bash
conda run -n local_llm_vllm python -m dotenv run -- python \
  scripts/run_phase4_bounded_smoke.py \
  --post-intervention-mode provisional_update \
  --video-id 0RxMZBLeqRI --gpu 4 --gpus 4,5,6,7 \
  --output-dir runs/phase4_one_video_smoke_parallel_multimodal_output \
  --state-dir runs/phase4_one_video_smoke_parallel_multimodal_state \
  --cache-dir runs/phase4_one_video_smoke_json_cache
```

Each invocation writes `run_manifest.json` and an immutable mode manifest under
`mode_manifests/`. Later modes reuse completed baseline, proposal, retrieval,
intervention, QA, and feedback artifacts. Repeating a completed mode performs
an exact resume. Calling an earlier mode does not remove or overwrite later
artifacts. `state-dir/provisional_update/` is smoke-local: no coverage-cycle,
confirmed checkpoint, confirmation evaluation, or canonical pointer is used.
The bounded smoke keeps its shared SigLIP image/text retrieval embedder on CPU;
GPUs listed by `--gpus` remain reserved for the persistent Qwen worker pool.

## 6. Production memory-conditioned iteration launcher

The active launcher is `scripts/run_phase4_memory_iteration.py`. It constructs
the latest Checkpoint 3E component path and does not import or wrap the obsolete
Stage 4.13/4.14 launcher:

```text
baseline/intervention/feedback artifacts
→ property_memory_v1
→ LLM codebook plan and candidate codebook/ID mapping
→ LLM router plan and rendered real-router prompt
→ atomic provisional codebook/router pair
```

It never runs confirmation and never writes `confirmed/current.json`.
Production experiments remain manually launched by the user.

CLI selection and scheduling:

- `--num-videos K`: logical iteration size; omitted means `K=3` unless
  `--video-ids` is supplied;
- `--video-ids id1,id2,...`: explicit ordered evidence list; if
  `--num-videos` is also supplied, the count must agree;
- `--max-parallel-videos P`: deterministic video-wave width;
- `--gpus 4,5,6,7`: unique iteration-scoped persistent worker set;
- `--embedding-gpu 3`: optional dedicated SigLIP GPU, which must be available,
  free, and disjoint from `--gpus`; a spawned child exposes physical GPU 3 as
  its private logical `cuda:0`; omission preserves CPU embedding;
- `--selection-seed`: deterministic initial rotation offset, default zero;
- `--dry-run-plan`: save selection, waves, identities, paths, and expected
  stages with zero model calls.

DVD database/query search uses a distinct CPU BGE model. The real launcher
preloads it exactly once before the first parallel video wave;
`--embedding-gpu` does not move this DVD model. A preload error fails before
parallel QA instead of allowing concurrent first-use to create a runtime-invalid
QA. Runs created before `dvd_bge_parent_preload_v1` require fresh output/state
directories; compatible caption and visual-index caches remain reusable.

Baseline and candidate DVD QA calls are intentionally sequential under
`serialized_dvd_qa_execution_v1`, even when `--max-parallel-videos` is greater
than one. Video captioning and property interventions remain parallel. This is
required because upstream DVD stores `VIDEO_FPS`, prompt overrides, and
instrumentation bindings in process-global state. Each QA validates the FPS
stored in its database before any tool call. An FPS mismatch is a runtime
failure and requires a fresh output/state directory rather than reuse of the
bad derived database.

`K` and `P` are independent. For example, `K=8, P=4` produces two ordered
four-video waves. `P` may not exceed the usable worker count. The launcher
rejects duplicate/unavailable/busy GPUs before model startup.

Dry-run plan for the default three-video pilot:

```bash
conda run --no-capture-output -n local_llm_vllm \
  python -m dotenv run -- python -u \
  scripts/run_phase4_memory_iteration.py \
  --dry-run-plan \
  --iteration-id phase4-memory-pilot-k3-plan \
  --num-videos 3 --max-parallel-videos 3 --gpus 4,5,6,7 \
  --embedding-gpu 3 \
  --output-dir runs/phase4_memory_pilot_k3_plan_output \
  --state-dir runs/phase4_memory_pilot_k3_plan_state \
  --cache-dir runs/phase4_memory_pilot_k3_plan_cache
```

Real `K=3` pilot:

```bash
conda run --no-capture-output -n local_llm_vllm \
  python -m dotenv run -- python -u \
  scripts/run_phase4_memory_iteration.py \
  --iteration-id phase4-memory-pilot-k3-iteration-001 \
  --num-videos 3 --max-parallel-videos 3 --gpus 4,5,6,7 \
  --embedding-gpu 3 \
  --output-dir runs/phase4_memory_pilot_k3_output \
  --state-dir runs/phase4_memory_pilot_k3_state \
  --cache-dir runs/phase4_memory_pilot_k3_cache
```

Real `K=5` pilot:

```bash
conda run --no-capture-output -n local_llm_vllm \
  python -m dotenv run -- python -u \
  scripts/run_phase4_memory_iteration.py \
  --iteration-id phase4-memory-pilot-k5-iteration-001 \
  --num-videos 5 --max-parallel-videos 4 --gpus 4,5,6,7 \
  --embedding-gpu 3 \
  --output-dir runs/phase4_memory_pilot_k5_output \
  --state-dir runs/phase4_memory_pilot_k5_state \
  --cache-dir runs/phase4_memory_pilot_k5_cache
```

Full evidence pool in two four-video waves:

```bash
conda run --no-capture-output -n local_llm_vllm \
  python -m dotenv run -- python -u \
  scripts/run_phase4_memory_iteration.py \
  --iteration-id phase4-memory-full-k8-iteration-001 \
  --num-videos 8 --max-parallel-videos 4 --gpus 4,5,6,7 \
  --embedding-gpu 3 \
  --output-dir runs/phase4_memory_full_k8_output \
  --state-dir runs/phase4_memory_full_k8_state \
  --cache-dir runs/phase4_memory_full_k8_cache
```

The examples intentionally use distinct state roots; they are alternatives,
not three sequential calls in one coverage cycle. To resume, repeat the exact
same command. The launcher restores the ordered videos, `K`, parent pair, and
coverage input from `iteration_identity.json` before consulting newer state
pointers. Any changed ordered list, count, seed, split hash, parent identity,
GPU/model/decoding configuration, updater version, or prompt version fails
closed. A completed repeat returns before worker startup and performs no model
or updater call.

Artifacts:

```text
<output_dir>/
├── iteration_identity.json
├── iteration_plan.json
├── iteration.log                           # human-readable live stage timeline
├── startup_models.json                  # real run only
├── baseline_videos/<video_id>/
├── baseline_batch_manifest.json
├── interventions/<video_id>/
├── feedback/<video_id>/
├── memory_codebook_checkpoint/
├── memory_router_checkpoint/
│   └── provisional_policy_pair/
├── worker_cleanup/attempt_<NNN>.json
└── manifest.json                        # only after atomic pair success

<state_dir>/
├── property_memory/current.json
├── memory_conditioned_provisional/current.json
└── production_selection/current.json
```

The `production_worker_cleanup_v2` artifact is written from `finally` on
success, failure, interruption, and completed resume. It records the SigLIP
child PID and release state as well as Qwen workers. After the scientific
`[END] [iteration]` line, cleanup emits explicit `[START] [worker_cleanup]` and
`[END] [worker_cleanup]` lines; only the latter means the launcher is ready to
exit. Persistent-worker command/result queues are explicitly closed and their
feeder threads joined, preventing interpreter shutdown from waiting on implicit
multiprocessing queue finalizers. Router/update failure cannot create the
top-level completed manifest or a codebook-only policy pair. Raw stage
artifacts remain in place for diagnosis and exact stage resume.

Follow a live production iteration with:

```bash
tail -f runs/phase4_memory_k4_isolated_001_output/iteration.log
```

Each line has timestamp, event (`START`, `END`, `ERROR`, `RESUME`, or `PLAN`),
stage, a readable Korean message, and compact JSON details. In particular,
look for `N번 iter 시작`, `video <ids> 병렬화 시작`, and paired boundaries for
`property_proposal`, `similarity_retrieval`, `intervention_recaption`,
`candidate_qa`, `memory_update`, `codebook_update`, and `router_update`.
For process completion, also require the final `worker_cleanup` END line.
The file is operational telemetry and may append on exact resume; use the
referenced immutable JSON/JSONL artifacts for scientific analysis.

All new QA runs deterministically execute DVD `clip_search_tool` with
`top_k=16`. This is `DVD_CLIP_SEARCH_TOP_K`, not
`--property-retrieval-top-k`. If the model requests another value, the raw
trajectory preserves that request and `tool_events.jsonl` records
`requested_args`, executed `args.top_k=16`, and the override policy. Runs made
before this policy are intentionally identity-incompatible with new runs.

The underlying active execution boundary remains
`Checkpoint3EOrchestrator.with_real_confirmation(...)`, which wires
`HistoryAwareDVDConfirmationEvaluator` to the same history-aware builder and
DVD QA path while injecting the property-memory and both LLM updater stages.

Use the same configured history builder for evidence baselines and confirmation:

```python
history_builder = HistoryAwareBaselineCaptionViewBuilder.from_local_qwen()
baseline_runner = BaselinePhaseRunner(
    history_aware_builder=history_builder,
    proposal_policy=proposal_policy,
    property_retrieval_runner=property_retrieval_runner,
)
orchestrator = Checkpoint3EOrchestrator.with_real_confirmation(
    baseline_runner=baseline_runner,
    intervention_runner=intervention_runner,
    feedback_runner=feedback_runner,
    confirmation_kwargs={
        "sample_loader": sample_loader,
        "history_aware_builder": history_builder,
        "base_prompt_template": base_prompt_template,
        "merge_prompt": merge_prompt,
        "sample_source_identity": split_manifest_hash,
        "cache_root": confirmation_cache_root,
        "cache_manifest_path": confirmation_cache_manifest_path,
        "history_block_seconds": 300.0,
        "max_history_captions": 30,
        "dvd_max_iterations": 10,
        "gpu": gpu,
        "downstream_qa_configuration": frozen_dvd_configuration,
    },
)
```

Every real `run()` invocation performs a fail-closed model audit before resume
resolution or any stage call. Standard output begins with one
`STARTUP_MODELS {...}` JSON line, and the same payload is written immutably to
`<output_dir>/startup_models.json`. It names the router, captioner, property
proposer, feedback provider, and downstream QA models. The active real path
requires the local Qwen history-aware router/captioner, OpenAI API property and
feedback providers, and the existing DVD QA runner. A missing model identity,
mock/fixture/stub implementation, provider mismatch, or a changed startup
manifest aborts the run before captioning, proposal, feedback, or QA calls.
Resume prints the same line again and validates it against the saved file.

The fixture-only concrete evaluator command is:

```bash
conda run -n local_llm_vllm python -m pytest -q \
  tests/test_checkpoint3e_confirmation_evaluator.py
```

### 6.1 Coverage-cycle commands

The configured driver must use one stable `state_dir` for every iteration and
one distinct `output_dir` per iteration. These are the exact orchestrator call
contracts; `<configured-driver>` is responsible only for constructing the four
reviewed stage runners and loading the serialized arguments.

First iteration after confirmed checkpoint `C0` (explicit confirmed pair):

```python
result = orchestrator.run(
    iteration_id="cycle-0000-iteration-01",
    roles=roles,
    coverage_state=coverage_state,
    parent_confirmed=confirmed_c0,
    prompt_bank=confirmed_bank_c0,
    router_policy=confirmed_router_c0,
    confirmed_prompt_bank=confirmed_bank_c0,
    confirmed_router_policy=confirmed_router_c0,
    scaffold_policy=fixed_scaffold,
    scaffold_contract=fixed_contract,
    state_dir="<run-root>/policy_state",
    output_dir="<run-root>/iterations/cycle-0000-iteration-01",
    execution_identity=frozen_execution_identity,
    baseline_kwargs=baseline_kwargs,
    intervention_kwargs=intervention_kwargs,
)
```

Subsequent iteration in the same cycle (bank/router omitted deliberately):

```python
result = orchestrator.run(
    iteration_id="cycle-0000-iteration-02",
    roles=roles,
    coverage_state=previous_result.next_coverage_state,
    parent_confirmed=confirmed_c0,
    scaffold_policy=fixed_scaffold,
    scaffold_contract=fixed_contract,
    state_dir="<run-root>/policy_state",
    output_dir="<run-root>/iterations/cycle-0000-iteration-02",
    execution_identity=frozen_execution_identity,
    baseline_kwargs=baseline_kwargs,
    intervention_kwargs=intervention_kwargs,
)
```

The second call resolves the exact bank/router pair referenced by
`coverage_cycles/cycle_0000/active_provisional.json`. It never scans sibling
directories or chooses a file by modification time. Supplying any explicit
bank/router pair while this reference is active fails closed.

Confirmation is not a separate feedback command. Use the same subsequent-call
form for the iteration that completes coverage. The orchestrator evaluates the
latest provisional pair against `C0`, closes the active reference as `accepted`
or `rejected`, resets coverage, and atomically writes `confirmed/current.json`.

Resume command: rerun the exact call for the same `iteration_id`, input
coverage state, `output_dir`, `state_dir`, parent checkpoint, stage identities,
and execution identity. A matching completed manifest returns without stage
calls. For a completed provisional iteration, this also verifies that the
cycle-local reference still names that iteration and its complete lineage.

## 7. Expected artifact layout

```text
<iteration>/
├── manifest.json
├── baseline_stage/
│   ├── policy_snapshot/
│   ├── baseline/<video_id>/
│   ├── property_proposals/
│   ├── property_retrieval/
│   └── manifest.json
├── interventions/<video_id>/
├── feedback/<video_id>/
├── next_state/
│   ├── update_plan.json
│   ├── provisional_bank.json
│   ├── provisional_router.json
│   ├── fixed_scaffold_reference.json
│   └── provisional_state.json
└── confirmation/                  # only at a completed coverage cycle
    ├── evaluation.json
    ├── decision.json
    ├── active_confirmed_checkpoint.json
    ├── history_aware_evaluator/
    │   ├── input_bundle.json
    │   ├── manifest.json
    │   ├── parent/
    │   │   ├── caption_state.json
    │   │   ├── videos/<video_id>/
    │   │   └── qa/<question_id>/
    │   └── candidate/
    │       ├── caption_state.json
    │       ├── videos/<video_id>/
    │       └── qa/<question_id>/
    ├── rollback_bank.json         # rejection only
    └── rollback_router.json       # rejection only

<state_dir>/
├── confirmed/
│   ├── current.json               # atomic canonical confirmed pointer
│   └── checkpoints/<checkpoint_id>/
│       ├── checkpoint.json
│       ├── bank.json
│       └── router.json
└── coverage_cycles/cycle_<NNNN>/
    └── active_provisional.json    # atomic active/accepted/rejected reference
```

At minimum locate:

- iteration policy snapshot;
- complete baseline captions and histories;
- baseline QA outputs and traces;
- per-video property proposals;
- per-property retrieval results;
- per-property mixed caption views;
- all rerun QA outputs;
- QA-level and property-level feedback;
- aggregated codebook/router update;
- next-state versions;
- token, latency, GPU, and cache summaries.

## 8. Resume and recovery

Caption caches produced under earlier JSON-output policies remain immutable.
`caption_plain_text_output_contract_v2`,
`caption_plain_text_parse_result_v2`, `caption_repetition_guard_v1`, and
`qwen_caption_plain_text_decoding_v2` use a distinct cache/resume identity. Do
not copy or edit an earlier `caption.json`. Exact resume is supported only when
the contract, parser, repetition, and decoding identities all match.

Caption parse failures use a fixed five-retry budget. One segment therefore
makes at most six caption-model calls. New artifacts report
`history_aware_caption_cache_v5`, `caption_plain_text_retry_v1`, `attempt_count`,
`retry_count`, `retry_exhausted`, and every attempt's raw output and parse
result. The first valid attempt is used. If all retries fail, the selected
intervention segment still raises `selected_segment_caption_failed`; it is not
silently replaced by an empty caption.

The model emits plain text only. Non-empty text is wrapped deterministically as
`{"clip_description": text}`; even JSON-looking text is preserved literally.
There is no sentence-count limit. Blank, repeated-sentence, and majority
repeated-n-gram outputs fail closed. There is no JSON extraction, truncated-JSON repair,
segment skip, or incumbent-caption fallback. Use a fresh timestamp so prompt,
caption, history, and DVD QA evidence are recomputed in new output/cache roots.

Focused verification:

```bash
conda run --no-capture-output -n local_llm_vllm \
  python -m pytest -q \
  tests/test_checkpoint3a_history_aware_baseline.py \
  tests/test_checkpoint3c_property_intervention.py
```

The current Qwen default is:

```bash
export SR_CAPTION_SUBJECT_REGISTRY_MODE=empty
```

For a later caption model that reliably emits the full registry, start a fresh
compatible run with:

```bash
export SR_CAPTION_SUBJECT_REGISTRY_MODE=optional
```

The mode is part of cache/resume identity. Changing it cannot resume or
reinterpret captions generated under the other mode.

For a caption, inspect the untouched response and parser action together:

```bash
find runs -path '*/history_v1/*/caption.json' -type f -print0 \
  | xargs -0 -n1 jq '{raw_output, parse_result, parsed}'
```

The only normal normalization is `strip_surrounding_whitespace`. Errors such as
`empty_output`, `repeated_sentence`, or
`repeated_ngram_majority` are real caption execution failures.

The top-level resume key hashes the parent confirmed checkpoint, current and
confirmed component snapshots, coverage state, train roles, explicit execution
identity, stage configuration identities, fixed-scaffold setting, and update
thresholds. A matching completed manifest returns before calling baseline,
proposal, retrieval, caption, intervention, feedback, QA, or confirmation
stages. Stage runners retain their own finer-grained resume contracts.

Every final update artifact is write-once. A mismatched top-level manifest,
incomplete completion marker, or conflicting immutable artifact fails closed.
Provisional snapshots never update canonical `confirmed/current.json`.
`active_provisional.json` is a separate cycle-local atomic reference containing
the cycle ID, parent confirmed ID and hashes, exact active bank/router paths,
versions and hashes, ordered lineage, and coverage-state hash. Each provisional
state records the same parent confirmed checkpoint and accumulated lineage.
Confirmation accepts the complete latest bank/router pair or restores the exact
bank and router snapshots already referenced by the parent pointer. Acceptance
or rejection closes the cycle-local reference; the next cycle starts only from
the resulting canonical confirmed pointer.

The confirmation bundle is write-once and shared by parent and candidate. It
fixes video/QA IDs, ordered segments and timestamps, sampled-frame paths and
content hashes, transcripts, prompt text/hashes, model/backend/decoding,
sampling and history configuration, fixed scaffold/contract versions, and DVD
QA configuration. Parent and candidate construct separate sequential on-policy
histories. Their caption cache records may share one physical root, but reuse
requires equality of the complete history-aware key and bundle hash. A changed
frame, sampling setting, model/backend, decoding setting, history setting,
component version, composed prompt, or history resolves to a distinct identity
or fails closed before QA.

## 9. Success criteria

A healthy run satisfies:

- every new history-aware caption artifact reports
  `history_aware_caption_cache_v5`, `caption_plain_text_parse_result_v2`, and
  `caption_plain_text_retry_v1`;
- `clip_description` is non-empty plain text without a sentence-count limit;
- decoding records temperature 0, maximum 1024 new tokens, and repetition
  penalty 1.05;
- current policy is frozen within an iteration;
- each of the selected `K` batch videos has one complete incumbent caption view;
- one video may generate multiple candidate properties;
- every candidate retains source-video and source-QA lineage;
- every property intervention is independent;
- candidate properties are force-applied, not prematurely routed;
- frozen history is reused;
- unselected captions equal incumbent captions;
- selected-segment failures fail the candidate explicitly without incumbent
  fallback;
- all source-video QAs rerun for each property;
- only correctness flips enter optimization feedback;
- feedback evidence is concise;
- multiple supported properties may be accepted;
- codebook/router update cites intervention evidence;
- scaffold version does not change.
## Fresh prompt-delta production iteration (current active parent)

This is the operator-only one-iteration path that creates new evidence rather
than converting a saved legacy property intervention.  The frozen evidence
videos are `0RxMZBLeqRI`, `TGom0uiW130`, and `w0Wmc8C0Eq0`; the confirmation
holdout remains `g1VFfVsZt7w` and `jIx5Zi84Z3Q`.  The other five evidence-pool
videos are recorded as already used by the earlier prompt-delta pilot.  The
fresh path uses the active parent
`runs/checkpoint_g_pilot_20260720_115145_state/versions/meta_prompt_42bb23b19a51450d6a9c.json`.

Run exactly one fresh iteration (paid OpenAI calls and GPU captioning/QA occur
only when the operator runs this command):

```bash
cd /home/intern/youngseo/surrogate_rollout
export FRESH_PROMPT_DELTA_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
bash /home/intern/youngseo/surrogate_rollout/scripts/run_fresh_prompt_delta_iteration.sh
```

After the terminal JSON-caption failure in run `20260720_130647`, do not resume
that timestamp: its completed baseline manifest intentionally retains the
invalid old-policy caption. Start a new plain-text-policy run on the currently
available GPU 7 with:

```bash
cd /home/intern/youngseo/surrogate_rollout
export FRESH_PROMPT_DELTA_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
export FRESH_PROMPT_DELTA_WORKER_GPUS=7
bash /home/intern/youngseo/surrogate_rollout/scripts/run_fresh_prompt_delta_iteration.sh
```

To resume the same immutable run after interruption, reuse the timestamp printed
by the script (or chosen before launch):

```bash
cd /home/intern/youngseo/surrogate_rollout
export FRESH_PROMPT_DELTA_TIMESTAMP="$(find /home/intern/youngseo/surrogate_rollout/runs -maxdepth 1 -type d -name 'fresh_prompt_delta_iteration_*_inputs' -printf '%f\n' | sed -E 's/^fresh_prompt_delta_iteration_([0-9]{8}_[0-9]{6})_inputs$/\1/' | sort | tail -n 1)"
test -n "$FRESH_PROMPT_DELTA_TIMESTAMP"
bash /home/intern/youngseo/surrogate_rollout/scripts/run_fresh_prompt_delta_iteration.sh
```

GPU assignment is execution metadata and may be explicitly overridden without
editing the immutable prepared input. For example, resume a new plain-text run
on GPU 7 by reusing the timestamp printed when that new run started:

```bash
cd /home/intern/youngseo/surrogate_rollout
export FRESH_PROMPT_DELTA_TIMESTAMP="$(find /home/intern/youngseo/surrogate_rollout/runs -maxdepth 1 -type d -name 'fresh_prompt_delta_iteration_*_inputs' -printf '%f\n' | sed -E 's/^fresh_prompt_delta_iteration_([0-9]{8}_[0-9]{6})_inputs$/\1/' | sort | tail -n 1)"
test -n "$FRESH_PROMPT_DELTA_TIMESTAMP"
test "$FRESH_PROMPT_DELTA_TIMESTAMP" != 20260720_130647
FRESH_PROMPT_DELTA_WORKER_GPUS=7 \
bash /home/intern/youngseo/surrogate_rollout/scripts/run_fresh_prompt_delta_iteration.sh
```

The resolved GPU list is persisted in the fresh evidence manifest and is also
passed to a deferred confirmation evaluator. It does not alter caption semantic
cache identities or mutate `component_config.json`.

The resume command resolves the most recent prepared fresh-input directory; set
the same environment variable explicitly instead if resuming an older run. The script
does not retry or repair provider responses. Completed baseline videos,
proposal plans, interventions, feedbacks, updater output, and paired
confirmation are immutable resume boundaries.

The operator command explicitly sets a 900-second result timeout for each
persistent caption-worker request. The parent polls worker liveness once per
second and fails immediately if any configured GPU process exits instead of
waiting the historical 24-hour compatibility default. Worker lifecycle,
block-start/completion, and Python error records are appended to
`runs/fresh_prompt_delta_iteration_${FRESH_PROMPT_DELTA_TIMESTAMP}_evidence/worker_logs/worker_gpu_4.jsonl`
through `worker_gpu_7.jsonl`.
An abrupt process death leaves the last lifecycle event intact; the parent CLI
also atomically records its surfaced exception below the evidence
`failures/` directory. These diagnostics do not alter caption/cache identity,
do not retry work, and do not discard completed segment caches.

Resolved production configuration is written before model startup to
`runs/fresh_prompt_delta_iteration_${FRESH_PROMPT_DELTA_TIMESTAMP}_inputs/component_config.json`.
It explicitly records Qwen2.5-VL-7B caption/prompt generation on GPUs 4–7,
OpenAI `gpt-4o` prompt-delta proposal/episode feedback/updating, DVD
`gpt-4o-mini` tool reasoning with the configured `gpt-5.5` text fallback,
transcript/frame/segment/history settings, context/output budgets, cache paths,
and all provider call limits. The empty compatibility bank/router records used
by the existing caption builder contain no properties or routing rules and are
marked `legacy_property_codebook_or_router_used=false`.

Prompt-delta proposal requests use policy
`fresh_prompt_delta_proposer_gpt4o_localized_inspection_v4` and representation
`trajectory_grounded_normalized_catalog_v3_localized_inspection`. The selection
scope is the ordered union of assistant timestamp citations and frame-inspection
calls classified as localized. A call is global when its requested start and
end are within the explicitly configured 10-second tolerance of the video start
and end. Global inspection segments, legacy `explicitly_cited_segments`,
`used_segments`, consumed, retrieved, and returned provenance remain only in
the private selection audit. These segments indicate evidence exposure and
localization, not causal attribution. Exact histories reconstruct from the
content-addressed catalog, and a shared segment appears once across QAs. A QA
without localized evidence is `no_localized_evidence`; a single-QA payload over
context is `context_ineligible`. Neither makes a provider call, while other QAs
continue. Exact `o200k_base` preflight uses the `128000` context limit and
reserved `4096` output tokens. Context-ineligible details are written beneath
`proposals/localized_trajectory_segments_only_v1/<video_id>/`; no truncation is
performed. When no eligible QA remains, the run records
`no_eligible_proposal_evidence` and exits normally before updating.

Resume the completed `20260721_063937` baseline under the new proposer identity
without rerunning its caption/DVD-QA stage:

```bash
cd /home/intern/youngseo/surrogate_rollout
export FRESH_PROMPT_DELTA_TIMESTAMP=20260721_063937
export FRESH_PROMPT_DELTA_WORKER_GPUS=1,2,3
bash /home/intern/youngseo/surrogate_rollout/scripts/run_fresh_prompt_delta_iteration.sh
```

The original input configuration and oversized failure artifact remain
immutable. The launcher explicitly overlays localized-inspection proposer
policy v4, a 10-second global-boundary tolerance, a nine-call fail-closed
transport ceiling, and up to three per-video
deltas. Compatible completed baseline prompt, caption, and DVD-QA artifacts are
validated and reused read-only. New intervention QA uses the strict
`HH:MM:SS` frame-inspect schema and at most one corrective tool-call retry; the
previous failed intervention remains untouched. New interventions and episodes
are isolated below `dvd_strict_frame_inspect_corrective_retry_v1`. Proposals,
interventions, episodes, feedback, and updater outputs use the new execution
identity. Successful fresh evidence writes the resolved
execution configuration separately under the evidence output and passes that
immutable file to feedback/updating.

Expected stage artifacts:

- inputs: `runs/fresh_prompt_delta_iteration_${FRESH_PROMPT_DELTA_TIMESTAMP}_inputs/manifest.json`,
  `component_config.json`, `confirmation_cases.json`, and
  `paired_decoding_settings.json` in that same directory;
- fresh baseline: `runs/fresh_prompt_delta_iteration_${FRESH_PROMPT_DELTA_TIMESTAMP}_evidence/baseline/manifest.json` plus each selected
  video's generated prompts, frozen histories, captions, and three QA results;
- prompt deltas: the per-video `proposal_plans.json` files below
  `runs/fresh_prompt_delta_iteration_${FRESH_PROMPT_DELTA_TIMESTAMP}_evidence/proposals/localized_trajectory_segments_only_v1/`;
- selective interventions and typed episodes below
  `runs/fresh_prompt_delta_iteration_${FRESH_PROMPT_DELTA_TIMESTAMP}_evidence/interventions/localized_trajectory_segments_only_v1/<video_id>/<delta_id>/dvd_strict_frame_inspect_corrective_retry_v1/`
  and `runs/fresh_prompt_delta_iteration_${FRESH_PROMPT_DELTA_TIMESTAMP}_evidence/episodes/localized_trajectory_segments_only_v1/dvd_strict_frame_inspect_corrective_retry_v1/`;
- fresh evidence completion: `runs/fresh_prompt_delta_iteration_${FRESH_PROMPT_DELTA_TIMESTAMP}_evidence/fresh_evidence_manifest.json`;
- grounded feedback/updater/confirmation:
  `runs/fresh_prompt_delta_iteration_${FRESH_PROMPT_DELTA_TIMESTAMP}_output/feedback/`,
  `updater_result.json`, optional `provisional_meta_prompt.json`, optional
  `confirmation/`, and final `iteration_result.json`;
- lineage/pointer: `runs/fresh_prompt_delta_iteration_${FRESH_PROMPT_DELTA_TIMESTAMP}_state/current_meta_prompt.json`
  and its immutable `versions/` directory.

Completion checks (set the exact run timestamp first):

```bash
cd /home/intern/youngseo/surrogate_rollout
export FRESH_PROMPT_DELTA_TIMESTAMP="$(find /home/intern/youngseo/surrogate_rollout/runs -maxdepth 1 -type d -name 'fresh_prompt_delta_iteration_*_inputs' -printf '%f\n' | sed -E 's/^fresh_prompt_delta_iteration_([0-9]{8}_[0-9]{6})_inputs$/\1/' | sort | tail -n 1)"
test -n "$FRESH_PROMPT_DELTA_TIMESTAMP"
jq -e '(.status == "completed" or .status == "no_eligible_proposal_evidence") and .legacy_property_codebook_or_router_used == false and .source_hashes_before == .source_hashes_after' "/home/intern/youngseo/surrogate_rollout/runs/fresh_prompt_delta_iteration_${FRESH_PROMPT_DELTA_TIMESTAMP}_evidence/fresh_evidence_manifest.json"
if [[ "$(jq -r '.status' "/home/intern/youngseo/surrogate_rollout/runs/fresh_prompt_delta_iteration_${FRESH_PROMPT_DELTA_TIMESTAMP}_evidence/fresh_evidence_manifest.json")" != "no_eligible_proposal_evidence" ]]; then
  jq -e '.status == "no_update" or .status == "promoted" or .status == "rolled_back"' "/home/intern/youngseo/surrogate_rollout/runs/fresh_prompt_delta_iteration_${FRESH_PROMPT_DELTA_TIMESTAMP}_output/iteration_result.json"
  jq . "/home/intern/youngseo/surrogate_rollout/runs/fresh_prompt_delta_iteration_${FRESH_PROMPT_DELTA_TIMESTAMP}_state/current_meta_prompt.json"
fi
```
