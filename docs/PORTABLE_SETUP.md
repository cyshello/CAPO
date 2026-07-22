# Portable setup (vast.ai and other fresh machines)

This repository is self-contained for meta-prompt optimization. Everything the
pipeline imports lives here; only the Video-MME media is synced separately.

## What is in the repository

| Path | What it is |
| --- | --- |
| `optimization/`, `prompt_routing/`, `captioning/`, `evaluation/`, `mixed_views/`, `references/`, `retrieval/`, `selection/`, `cache/` | the harness |
| `optimization/prompts/*.txt`, `optimization/prompts/init_meta_prompt.json` | proposer / feedback / updater system prompts and the parent meta-prompt |
| `vendor/dvd_stack/` | the DVD stack, vendored: `data_provider`, `dvd_prompt`, `dvd_backend`, `dvd_captioning`, `dvd_caption_worker`, `captioner`, `codex_infer`, the `dvd` package, and the Video-MME provider |
| `train_set/*.txt` | frozen evidence / confirmation cohorts |
| `split_manifest.json` | train roles |

`vendor/dvd_stack` used to be an external checkout (`prompt_sensitivity` +
`longVideoPO/providers`). `config.PROMPT_SENS_ROOT` now defaults to the
vendored copy; set `SR_PROMPT_SENS_ROOT` to go back to an external checkout.

## Data that is not in the repository

Only the Video-MME cohort media is needed. The 30 cohort videos
(`train_set/20samples.txt` + `train_set/confirmation_10samples.txt`) are about
**8.4 GB**.

```
<data_root>/videomme_data/Video-MME/
  videomme/test-00000-of-00001.parquet   # ~400 KB, all 2700 QA
  videos/long/<video_id>.mp4             # 8.4 GB for the 30 cohort videos
  subtitles/subtitle/<video_id>.srt      # optional, only with USE_TRANSCRIPT
```

`frame_cache/` (205 GB) and `run_workspace/` (38 GB) are **not** synced —
decoded frames are regenerated on the target machine.

Use `scripts/sync_cohort_data.sh` to copy exactly the cohort files.

## Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `SR_VIDEOMME_DATA_ROOT` | where `Video-MME/` lives | `/hub_data3/videomme_data` |
| `SR_DVD_RUN_WORKSPACE` | decoded frames and DVD scratch state | `vendor/dvd_stack/dvd/run_workspace` |
| `SR_PROJECT_ROOT` | repository root for the shell launchers | derived from the script path |
| `SR_CONDA_ENV` | conda env the launchers run python in | `local_llm_vllm` |
| `SR_PROMPT_SENS_ROOT` | external DVD stack instead of the vendored one | vendored copy |
| `SR_CAPTION_JSON_ATTEMPTS` | registry-JSON attempts before the plain-text fallback | `2` |
| `OPENAI_API_KEY` | proposer / feedback / updater / DVD tool calls | required, read from `.env` |

## Bring-up

```bash
git clone <repo> surrogate_rollout
cd surrogate_rollout

conda create -n local_llm_vllm python=3.11 -y
conda run -n local_llm_vllm pip install -r requirements.txt

printf 'OPENAI_API_KEY=sk-...\n' > .env

export SR_VIDEOMME_DATA_ROOT=/data/videomme_data
export PYTHONPATH="$(dirname "$PWD"):$PWD"

# import check, no GPU and no provider calls
conda run -n local_llm_vllm python -m pytest tests/ -q

# provider check, needs the synced media
conda run -n local_llm_vllm python -c "
import sys
from surrogate_rollout import config
for p in (config.PROMPT_SENS_ROOT, config.DVD_ROOT): sys.path.insert(0, p)
from data_provider import get_provider
p = get_provider(config.BENCHMARK, split=config.BENCHMARK_SPLIT)
print(p, len(p))"
```

The repository must be cloned into a directory named `surrogate_rollout`: the
package imports itself as `surrogate_rollout.*`, and `PYTHONPATH` has to
contain its parent.

## Running one iteration

```bash
env FRESH_PROMPT_DELTA_TIMESTAMP=$(date -u +%Y%m%d_%H%M%S) \
    FRESH_PROMPT_DELTA_WORKER_GPUS=0 \
    FRESH_PROMPT_DELTA_SELECTED_VIDEO_IDS=<comma separated ids> \
    FRESH_PROMPT_DELTA_PREVIOUS_UPDATE_VIDEO_IDS= \
    FRESH_PROMPT_DELTA_EVIDENCE_COHORT_FILE=$PWD/train_set/20samples.txt \
    FRESH_PROMPT_DELTA_CONFIRMATION_COHORT_FILE=$PWD/train_set/confirmation_10samples.txt \
    bash scripts/run_fresh_prompt_delta_iteration.sh
```

Sequential K-iteration runs go through
`scripts/run_prompt_delta_two_iteration_10video_pool.sh` (configurable K; see
`--help`). That launcher still requires exactly five GPU IDs.

## GPU notes

The captioner is Qwen2.5-VL-7B-Instruct on vLLM with
`gpu_memory_utilization=0.85` (`vendor/dvd_stack/dvd_backend.py` and
`vendor/dvd_stack/dvd_caption_worker.py`; the two footprints must stay equal).
On a 24 GB card that is about 20.9 GB, so the GPU has to be otherwise idle.
