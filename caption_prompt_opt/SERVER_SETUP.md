# Running caption_prompt_opt on another server (Pro 5000 x4)

Yes — the same "go" pattern the incumbent uses
(`surrogate_rollout/scripts/run_fresh_prompt_delta_iteration.sh`) works here.
The launcher is `caption_prompt_opt/scripts/go_caption_prompt.sh` (prepare →
static evidence → feedback/update/confirm), isolated in its own run roots and
cache identities.

```bash
cd <CAPO checkout>                                  # the surrogate_rollout dir
export SR_CAPTION_MODEL_ID=Qwen/Qwen3.5-9B          # captioner (see caveats!)
export CAPTION_PROMPT_WORKER_GPUS=0,1,2,3           # 4 GPUs
bash caption_prompt_opt/scripts/go_caption_prompt.sh
```

## 1. Code
- `caption_prompt_opt/` now lives **inside** the `surrogate_rollout` checkout
  (`surrogate_rollout/caption_prompt_opt/`). Once it is committed, a normal
  `git clone` of CAPO brings it along — nothing extra to copy. It only **adds**
  files; no existing CAPO file is modified.
- Ensure the checkout includes the vendored DVD stack (`vendor/dvd_stack/`) —
  the captioner and DVD runtime live there.
- The go script auto-detects the layout (nested here, or sibling) and sets
  `PYTHONPATH` so top-level `import caption_prompt_opt` and
  `import surrogate_rollout` both resolve. Override with `SR_PROJECT_ROOT` if
  your checkout is elsewhere.

## 2. Conda env (captioner runtime) — brand-new server

The base bring-up is `docs/PORTABLE_SETUP.md` in the CAPO repo:
`conda create -n local_llm_vllm python=3.11` + `pip install -r requirements.txt`
+ `.env` + sync cohort media (`scripts/sync_cohort_data.sh`).

**One delta for this path:** `requirements.txt` pins `vllm==0.11.2`,
`transformers==4.57.6`, `qwen-vl-utils==0.0.14` — those target **Qwen2.5-VL**.
**Qwen3.5-9B / Qwen3-VL needs newer vLLM (nightly) + transformers.** So after the
pinned install you must upgrade the captioner stack to a combo that recognizes
the model's architecture.

`caption_prompt_opt/scripts/setup_new_server.sh` automates all of it (conda env,
pinned deps, captioner-stack override, `.env`, model download, import check):

```bash
cd <CAPO checkout named surrogate_rollout>
export SR_CAPTION_MODEL_ID=Qwen/Qwen3.5-9B
export OPENAI_API_KEY=sk-...          # only if no .env yet
# pin the verified combo (recommended); defaults just `pip install -U latest`:
export CAPTION_STACK_SPEC='vllm==<ver> transformers==<ver> qwen-vl-utils==<ver>'
export VLLM_PIP_EXTRA='--pre --extra-index-url https://wheels.vllm.ai/nightly'
bash caption_prompt_opt/scripts/setup_new_server.sh
```

It does **not** create GPUs or sync media (data comes from a host that has it);
it prints the remaining data-sync + run steps. Match CUDA to the Pro 5000s.

## 3. Secrets — `surrogate_rollout/.env`
- `OPENAI_API_KEY` (feedback / proposer / updater / DVD text backend — all
  reused, all paid). Add any Azure/codex creds your `SR_DVD_TEXT_BACKEND` needs
  (default `openai`).

## 4. Data + workspace (paths differ on a new box — set them)
- Video-MME frames: `SR_VIDEOMME_DATA_ROOT` / `SR_VIDEOMME_FRAME_CACHE`
  (default `/hub_data3/videomme_data/...` will not exist here).
- DVD workspace: `SR_DVD_RUN_WORKSPACE` (decoded frames land here).
- `surrogate_rollout/split_manifest.json` present (the go script points at it).
- Optional: `SR_RUNS_ROOT`, `SR_CAPTION_CACHE_ROOT`.

## 5. Captioner = Qwen3.5-9B
`SR_CAPTION_MODEL_ID` swaps **both** the vLLM model that is loaded and the
caption-cache key (`captioning/qwen25_vl.py`), and the go script threads it into
`MODEL_IDENTITY`. Pre-download so the local-cache resolver finds it:

```bash
huggingface-cli download Qwen/Qwen3.5-9B
```

### No captioner code change needed
`Qwen25VLCaptioner` is already model-agnostic: it loads
`AutoProcessor.from_pretrained(model_path)`, applies the model's **own** chat
template, and preprocesses via `qwen_vl_utils.process_vision_info`. The file's
own docstring documents driving it with `SR_CAPTION_MODEL_ID=Qwen/Qwen3-VL-...`
on another box — a Qwen3-family VL model is a supported swap, not a rewrite.

### Requirements for the swap
1. **Exact HF repo id.** Set `SR_CAPTION_MODEL_ID` to the *vision* model's exact
   repo id. Verify the id you use is the multimodal one (AutoProcessor + vLLM
   must load it as an image model). Pre-download it (`huggingface-cli download`).
2. **Recent deps.** The env's `transformers` + `qwen_vl_utils` + **nightly vLLM**
   must all recognize the Qwen3-family arch/processor. You already pull vLLM from
   nightly — make sure `qwen_vl_utils`/`transformers` are new enough too.
3. **VRAM.** `dvd_backend.get_captioner()` builds one engine per worker GPU with
   `tensor_parallel_size=1`, `gpu_memory_utilization=0.85`, `max_model_len=12288`.
   Each GPU must hold the **full 9B** (bf16 ≈ 18–20 GB weights + KV). Confirm each
   Pro 5000 has headroom; otherwise lower `max_model_len` / util, or move to
   tensor-parallel (a code change — per-worker TP is fixed at 1).

### Cosmetic only
- `Qwen25VLCaptioner.name = "qwen2.5-vl-7b-instruct"` is a static label; it does
  **not** feed the cache key (the key uses `SR_CAPTION_MODEL_ID`). Harmless.
- Vision-token cache (if enabled) is per-model; the isolated `cache_root` makes
  this a clean rebuild. `image_max_pixels=200704` (~256 tokens/img) was tuned for
  2.5-VL — revisit only if quality/latency looks off.

## 6. GPUs
`CAPTION_PROMPT_WORKER_GPUS=0,1,2,3`. Workers are data-parallel (one captioner
engine per GPU), so 4 GPUs = 4 concurrent caption shards.

## Isolation from any incumbent run
Run roots (`runs/caption_prompt_iteration_<ts>_*`), cache root, and identities
(`cache_reset_identity=caption_prompt_static_clean_<ts>`,
`evaluation_pipeline_identity=dvd_history_aware_static_caption_prompt_paired_v1`)
are all distinct, and rollout caches sit in the reserved `router_v8888_<static>`
namespace. Safe to run beside a meta-prompt job as long as they don't share
`cache_root`.

## Knobs (env overrides)
`CAPTION_PROMPT_TIMESTAMP`, `CAPTION_PROMPT_WORKER_GPUS`,
`CAPTION_PROMPT_SELECTED_VIDEO_IDS`, `CAPTION_PROMPT_PARENT`,
`CAPTION_PROMPT_OPTIMIZER_MODEL_ID` (default `gpt-5-mini`),
`CAPTION_PROMPT_PROMOTION_POLICY`, `CAPTION_PROMPT_CACHE_ROOT`,
`CAPTION_PROMPT_STATE_ROOT`, `CAPTION_PROMPT_MEMORY_BANK_ROOT`.
