Test only the surrogate rollout with Qwen2.5-VL-Instruct.

## Training runs

Three optimization paths, one shared model stack.

| Path | What it optimizes | Prompt-delta scope |
| --- | --- | --- |
| Meta prompt | the meta-prompt driving a per-clip LLM prompt generator | source-QA-localized segments |
| Static prompt | one caption prompt emitted verbatim for every clip | source-QA-localized segments |
| Full recaption | same as meta prompt | **every** segment of the video |

Full recaption is the baseline that isolates selective intervention: identical to
a meta-prompt run in every respect except the delta apply-scope.

### Model stack

`scripts/env/gpt5mini_stack.sh` is the single source for every model, so the
three paths stay comparable. Source it before any launch:

```bash
set -a && source scripts/env/gpt5mini_stack.sh && set +a
```

| Component | Model | Reasoning effort |
| --- | --- | --- |
| Captioner | `Qwen/Qwen3.5-9B` (local vLLM) | — |
| DVD orchestrator | `gpt-5-mini` | `minimal` |
| Prompt generator | `gpt-5-mini` | `minimal` (unused on the static path) |
| Delta proposer | `gpt-5-mini` | provider default |
| Feedback generator | `gpt-5-mini` | `medium` |
| Prompt updater | `gpt-5-mini` | `high` |

DVD falls back to `gpt-4o` for the requests `gpt-5-mini` refuses on prompt policy.

### Host setup (once)

```bash
export SR_CONDA_ENV=capo
export SR_CAPTION_MODEL_ID=Qwen/Qwen3.5-9B
# <root>/Video-MME/{videos/long,videomme,subtitles/subtitle}
export SR_VIDEOMME_DATA_ROOT=/path/to/videomme_data
export PROMPT_DELTA_WORKER_GPUS=5          # one captioning worker per GPU

bash caption_prompt_opt/scripts/setup_training_host.sh all
```

`all` sets up and prints a launch command without starting anything. Do not use
`go`: it launches the static path unconditionally.

### Launch

Export the host variables and the model stack, then pick one. Each command is a
whole sequential K-iteration experiment over `train_set/20samples.txt`.

```bash
export PROMPT_DELTA_ITERATION_TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)   # keep it: resume key

# 1. meta prompt        -- 4 iterations x 5 evidence videos
bash scripts/run_prompt_delta_four_iteration_20video_pool.sh

# 2. static prompt      -- 5 iterations x 4 evidence videos
bash caption_prompt_opt/scripts/run_caption_kiter.sh

# 3. full recaption     -- same preset as (1)
bash full_recaption_opt/scripts/run_full_recaption_kiter.sh
```

Re-running with the same `PROMPT_DELTA_ITERATION_TIMESTAMP` skips completed
iterations and resumes a half-written one. Run one experiment at a time per GPU
set — every captioning worker loads its own copy of the captioner.

The static path defaults to a different schedule and held-out cohort. Align them
when the three are compared directly:

```bash
PROMPT_DELTA_ITERATION_COUNT=4 PROMPT_DELTA_VIDEOS_PER_ITERATION=5 \
PROMPT_DELTA_CONFIRMATION_COHORT_FILE=train_set/confirmation_10samples.txt \
  bash caption_prompt_opt/scripts/run_caption_kiter.sh
```

### Held-out measurement

All three promote every candidate and defer held-out scoring to a queue, so no
run re-captions the confirmation videos on the captioning host.

| Path | Queue |
| --- | --- |
| Meta prompt | `runs/20video_4iteration_measurement_queue` |
| Static prompt | `runs/caption_prompt_measurement_queue` |
| Full recaption | `runs/full_recaption_20video_4iteration_measurement_queue` |

Drain a queue wherever the confirmation media lives:

```bash
python scripts/run_measurement_worker.py \
  --queue-dir <queue> --component-config <run>_inputs/component_config.json
```

## Qwen2.5-VL captioner

Code-level API:

```python
from surrogate_rollout.captioning import Qwen25VLCaptioner

captioner = Qwen25VLCaptioner(max_images_per_prompt=8)
caption = captioner.caption(
    ["frame_000.jpg", "frame_001.jpg", "frame_002.jpg", "frame_003.jpg",
     "frame_004.jpg", "frame_005.jpg", "frame_006.jpg", "frame_007.jpg"],
    "Describe the sequence across these frames.",
    max_tokens=128,
)
```

Smoke test with 8 generated images:

```bash
conda run -n local_llm_vllm python -m surrogate_rollout.scripts.smoke_qwen25vl_captioner --gpu 1
```

Smoke test with a frame directory:

```bash
conda run -n local_llm_vllm python -m surrogate_rollout.scripts.smoke_qwen25vl_captioner \
  --gpu 1 \
  --image-dir /path/to/frames \
  --num-images 8
```

TODO
- run/text VideoARM with different prompts
- cherrypick examples
