# Stage 4.0 — Repository Inspection and Corrected Integration Map

Date: 2026-07-15. Code changes: none (this notes file only, permitted by
PHASE4_PROMPT_ROUTING.md Stage 4.0).

---

## 1. Exact current execution flow (Phase 0–3)

Driver: `scripts/run_phase2_sanity_check.py`

```text
ensure_backend(gpu, preload_captioner=True)        # evaluation/dvd_qa.py:35
    installs codex+Qwen+BGE backend once per process;
    vLLM engine MUST be built before BGE/DB thread pools (fork-safety
    deadlock observed 2026-07-14)

provider[idx] -> sample dict -> EvaluationRequest  # evaluation/rollout_evaluator.py:43
    single `candidate_prompt: str` per request (one prompt for all clips)

FullRolloutEvaluator.evaluate                       # evaluation/full_rollout.py:83
    validate_candidate_prompt                       # captioning/candidate_captions.py:45
    prepare_video_workdir (frames symlink)          # evaluation/dvd_qa.py:88
    caption_clips(sample, candidate_prompt, ...)    # captioning/candidate_captions.py:114
        build_cache_key -> new_candidate_cache_dir  # cache/caption_cache.py:61,190
        assert_writable + register_cache (manifest)
        _prefill_from_legacy (read-only legacy ckpt)
        build_clip_index -> _build_clips            # vendored dvd_captioning.py:100
            clip keys "{start}_{end}", 10 s clips, temporal order
        set_prompts(caption_prompt=..., merge_prompt=...)   # GLOBAL prompt registry
        _pending_tasks(wanted, ckpt)                # renders ONE shared prompt per clip
        _caption_inprocess(pending, ...)            # one Qwen batch; per-clip ckpt JSON
        reset_prompts()
    assemble captions.json (clip order preserved) + merge_fn(subject registries)
    write_captions_json                             # mixed_views/builder.py:52
    run_dvd_qa                                      # evaluation/dvd_qa.py:125
        reset_prompts(); DB path = database_path_for(captions.json content hash)
        DVDCoreAgent(db_path, captions_path).run(question)
        extract_references -> ReferenceSets
        artifacts: result.json / trajectory.jsonl / tool_events.jsonl /
                   llm_calls.jsonl / references.json
    -> EvaluationResult

SelectiveSurrogateRolloutEvaluator.evaluate         # evaluation/selective_rollout.py:148
    baseline trajectory -> selectors (selection/*) -> apply_budget
    caption_clips(clip_ids=selected)                # same cache-key path
    MixedViewBuilder.build                          # mixed_views/builder.py:78
        replaces ONLY selected clips; re-merges registry; content-hash-keyed DB
    run_dvd_qa (same shared reasoning path)
    fallback_type recorded: none / clip_only / full_rollout / unsupported
```

Run artifacts per driver run: `runs/phase2_sanity_<ts>/{run_config.json,
results.jsonl, summary.json, <video_id>/captions_*/, runs/<qa_slug>_<policy>_p<hash8>/}`.
`results.jsonl` rows = `EvaluationResult.as_json()` + `candidate_name`.

## 2. Caption cache key construction

`CaptionCacheKey` (schemas.py:41): video_id, segment_id ("*"), **prompt_hash =
sha256(caption_prompt || merge_prompt)**, caption_model_id, decoding_hash,
source_hash. Cache dir: `caption_caches/<video>/p<hash12>_d<hash8>_s<hash8>/ckpt/<clip>.json`.
Manifest (`cache_manifest.jsonl`) + `assert_writable` guard read-only legacy
caches. Vector DB keyed by exact captions.json content hash
(`database_c<hash16>.json`). **Consequence: a distinct composed prompt gets a
distinct caption cache directory for free — Phase 4 §14.3 keying is already
satisfied if each composed prompt goes through `caption_clips` separately.**

## 3. Per-call prompt support (Stage 4.0 question 7)

- Vendored `_caption_inprocess` **already supports one prompt per clip** in a
  single batch (`caption_batch(files_list, [t["prompt"] for t in pending])`).
- But `_pending_tasks` renders the single global `PROMPTS.caption_prompt` for
  every pending clip, and `caption_clips` sets exactly one prompt per call.
- Therefore the smallest routed-caption adapter is: **group segments by
  composed-prompt hash, call the existing `caption_clips` once per group with
  `clip_ids=<group>`** — no Phase 0–3 edit, correct per-prompt cache keys,
  matches Phase 4 §14 "group segments by composed prompt hash".

## 4. Prompt placeholders and output contract (hard scaffold contract source)

- Required placeholders (enforced by `validate_candidate_prompt`):
  `TRANSCRIPT_PLACEHOLDER`, `CLIP_START_TIME`, `CLIP_END_TIME`
  (captioning/candidate_captions.py:33).
- Required output: JSON with `clip_start_time`, `clip_end_time`,
  `subject_registry` (name/appearance/identity/first_seen), `clip_description`
  (vendored `dvd/prompts.py` `_CAPTION_PROMPT`); parsed via
  `_extract_json(_strip_fences(...))`; parse failure cached as `{}`.
- These form the initial `ScaffoldContract` (required_placeholders,
  output_schema, parser compatibility). Composed prompts must pass
  `validate_candidate_prompt` unchanged.

## 5. Segment materialization order

`_build_clips` walks the video in temporal order; clip keys `"{int(start)}_{int(end)}"`;
`CandidateCaptionSet.clips` and captions.json insertion order preserve it.
Phase 4 `SegmentContext.segment_id` == existing clip key; timestamps derivable
by splitting the key. "Segment" (Phase 4 docs) == "clip" (repo code).

## 6. Smallest integration points for Phase 4

1. **Routed captioning**: new adapter in `prompt_routing/` that takes
   `segment_composed_prompt_map: Mapping[str, ComposedCaptionPrompt]`, groups
   by `prompt_hash`, calls `caption_clips` per group.
2. **Routed view assembly**: new builder (in `prompt_routing/`, composition not
   modification) reusing `mixed_views.builder.caption_entry_from_parsed`,
   `load_clip_registries`, `write_captions_json`, injectable `merge_fn`;
   distinct view dir name (e.g. `captions_routed_<viewhash>`) so a routed view
   can never be mistaken for a single-prompt full cache.
3. **Evaluation**: call `run_dvd_qa` directly over the routed captions.json
   (exact pattern of `FullRolloutEvaluator.evaluate` after view build); DB path
   via `database_path_for`. `EvaluationRequest` stays untouched — a routed
   evaluator adapter produces `EvaluationResult` + a routed sidecar record
   carrying bank/router/scaffold/contract versions, routing decisions, and
   composition traces (EvaluationResult has no version fields; do not extend it
   — preserve existing artifact format).
4. **Evidence builder input**: saved `EvaluationResult.as_json()` rows +
   per-run artifacts (`result.json`, `trajectory.jsonl`, `references.json`)
   + the routed sidecar records. `QAExample` already exists (schemas.py:30)
   and matches the Phase 4 `UpdateValidator` signature.
5. **Backend/bootstrap for any real Phase 4 run**: `ensure_backend(gpu,
   preload_captioner=True)` before BGE/DB work (fork-safety), and the
   `sys.path` insertion of `PROMPT_SENS_ROOT` + `DVD_ROOT` as done in
   `scripts/run_phase2_sanity_check.py`.

## 7. Files likely to change per stage (Stage 4.1 proposal at end)

Phase 0–3 modules requiring **zero** edits for Stages 4.1–4.5. Stage 4.6
adapter is additive (new module calling `caption_clips`). Only plausible later
touch: none identified; `mixed_views/builder.py` helpers are already importable.

## 8. Discrepancies between documentation and code

1. **CLAUDE.md "ScaffoldApplier" section is wrong/stale**: it describes
   converting feedback into *prompt-bank update operations* — that is
   `PromptBankUpdateProposer` per PHASE4 §3.3/§3.4, which explicitly forbids
   conflating the two. PHASE4_PROMPT_ROUTING.md is authoritative.
2. **CLAUDE.md "PromptRouter" says "a prompt-bank entry" (singular)**; PHASE4
   §1/§13 require zero-or-more entries. PHASE4 authoritative.
3. `EvaluationRequest` carries one `candidate_prompt: str`; PHASE4 §14 needs a
   per-segment map — resolved by adapter (above), not by changing the schema.
4. `EvaluationResult` lacks component-version fields required for evidence
   (§10.4) — resolved by routed sidecar record, not schema change.
5. No YAML infrastructure exists (config is pure Python `config.py`); PHASE4
   §17 requires YAML — Phase 4 adds its own typed config + YAML load without
   touching `config.py` defaults.
6. No GEPA implementation code exists in this repo (comment mentions only);
   CLAUDE.md's "GEPA-related code already present" clause is moot.
7. Subject-registry merge (`default_merge_fn` -> codex) is nondeterministic;
   Phase 4 deterministic tests must inject `merge_fn` like existing tests do.
8. Uncommitted working-tree changes exist (CLAUDE.md rewrite,
   PHASE4_PROMPT_ROUTING.md, `preload_captioner` fork-safety fix in
   evaluation/dvd_qa.py, sys.path + CPU-SigLIP fixes in two scripts). Should be
   committed before Stage 4.1 so Phase 4 starts from a clean baseline.
9. Layout conflicts with PHASE4 §8: none. `prompt_routing/` and
   `optimization/` do not exist; proposed test filenames do not collide.
   Top-level `schemas.py` coexists with the planned
   `prompt_routing/schemas.py` / `optimization/schemas.py` (package-qualified
   imports, no shadowing).
10. Tests run via the `local_llm_vllm` conda env
    (`/home/intern/.conda/envs/local_llm_vllm/bin/python -m pytest`), from the
    parent directory of the repo (conftest inserts parent on sys.path;
    package name is `surrogate_rollout`). The `youngseo` env lacks pytest.

## 9. Proposed Stage 4.1 file list (schemas only — pending approval)

```text
prompt_routing/__init__.py
prompt_routing/schemas.py        # PromptEntry, PromptBankSnapshot, RoutingRule,
                                 # RouterPolicySnapshot, SegmentContext,
                                 # RoutingDecision, ScaffoldContract,
                                 # ScaffoldPolicySnapshot, CompositionTrace,
                                 # ComposedCaptionPrompt, version identifiers
prompt_routing/phase4_config.py  # typed Phase4Config (optimize_scaffold=False
                                 # default) + deterministic (de)serialization
tests/test_phase4_schemas.py
```

No persistence, routing logic, scaffold calls, captioning, evidence, feedback,
or optimization in Stage 4.1.
