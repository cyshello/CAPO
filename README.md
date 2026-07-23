# CAPO — caption-prompt optimization for long-form video understanding

An optimization loop over the *captioning instruction*. Each iteration runs a
DVD question-answering episode on a handful of videos, collects counterfactual
evidence about which caption content the answer actually depended on, turns that
into structured feedback, and rewrites the meta-prompt that produces every
caption. The model weights never move.

## Setting it up on a new host

Pick the caption model, then run one command:

```bash
git clone https://github.com/cyshello/CAPO.git && cd CAPO && \
  SR_CAPTION_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct bash setup_training_host.sh go
```

`go` builds the conda environment, installs the package, downloads the models,
verifies the evidence videos, checks that vLLM can load the caption model on this
hardware, runs the unit tests, and starts the 5-iteration run in the background
with a log path printed. Nothing is spent before the two gates only a human can
pass: the video files and a funded API key. If either is missing it stops and
prints exactly what to supply — including the `rsync` line for the videos.

The run is started under `scripts/watch_prompt_delta_experiment.sh`, so it
survives crashes without anyone watching it. A vLLM engine dying mid-iteration is
a real failure mode, and unattended it would otherwise stop the whole experiment
until a person noticed. Every stage is write-once and skips when its artifact
already exists, so the watcher relaunches the driver on the same timestamp and it
resumes instead of recomputing; it drains any vLLM process still holding the
worker GPUs first, and gives up after `PROMPT_DELTA_MAX_RESTARTS` (20) so a
deterministic failure cannot spin forever. `CAPO_FOREGROUND=1` runs the driver
unwatched in the current shell instead.

`bash setup_training_host.sh all` does the same setup but never launches, and
each stage re-runs on its own:
`check env install models data creds smoke-vllm smoke-tests launch go`.

The checkout directory can be called anything. The package name
`surrogate_rollout` comes from the editable install (`pip install -e .`), not
from the directory.

## Running the loop by hand

```bash
set -a
source scripts/env/gpt5mini_stack.sh   # which hosted models, and at what effort
source scripts/env/training_host.sh    # this machine: GPUs, captioner, schedule
set +a
bash scripts/run_prompt_delta_two_iteration_10video_pool.sh
```

Two profiles, because the two choices are independent: the model stack is an
experiment decision that should travel between hosts unchanged, while GPU
indices, the caption model, and the iteration schedule belong to the machine.
Both files assign defaults only — an exported variable wins over both, which is
what makes the one-liner above work without editing anything.

To resume, re-run with `PROMPT_DELTA_ITERATION_TIMESTAMP` set to an existing
run's timestamp. Completed iterations are skipped and a partial one resumes from
its own artifacts. `CAPO_FOREGROUND=1` keeps the run in the current shell
instead of detaching it.

## What this host does not do

Held-out measurement. Under
`FRESH_PROMPT_DELTA_PROMOTION_POLICY=promote_and_enqueue_measurement_v1` the loop
promotes a candidate and writes the scoring request to a queue directory instead
of scoring it inline, so a measurement that fails — or a worker that is not
running yet — can never stop the optimization. A separate process drains it:

```bash
python scripts/run_measurement_worker.py \
  --queue-dir runs/measurement_queue \
  --component-config <evidence-run>/resolved_component_config.json \
  --worker-gpus 0,1
```

The component config is the `resolved_component_config_path` recorded in an
evidence run's `fresh_evidence_manifest.json`; `--worker-gpus` overrides the GPU
list inside it, which is what lets the worker sit on hardware the loop is not
using.

The two halves share nothing but that directory and can sit on different GPUs or
different machines. The practical consequence for setup: **only the evidence
cohort's videos have to exist on the training host.** The confirmation cohort's
videos are the measurement worker's problem; the training host needs just the
cohort *file*, to derive the held-out case list it enqueues.

## Configuration

`config.py` is the single place experiment choices live, and the ones that vary
per host or per run read from the environment. Its defaults reproduce the
original gpt-4o stack, so an unset environment reproduces the earlier measured
runs byte for byte; the profiles under `scripts/env/` are what move a run off
those defaults. Values that identify the method rather than the host — the split
seed, the embedding model, the decoding policy version — are constants on
purpose: they are part of every cache key. The knobs worth knowing:

| Variable | Effect |
| --- | --- |
| `SR_CAPTION_MODEL_ID` | Local caption model. Part of the cache identity, so two models never share captions. |
| `SR_ORCHESTRATOR_TOOL_MODEL` | The DVD agent's function-calling model. |
| `SR_VIDEOMME_DATA_ROOT` | Where `Video-MME/{videos,videomme}` lives. |
| `PROMPT_DELTA_WORKER_GPUS` | One captioning worker per listed GPU. |
| `PROMPT_DELTA_ITERATION_COUNT` / `_VIDEOS_PER_ITERATION` | The schedule. Their product must fit inside the frozen evidence cohort. |
| `SR_CONDA_ENV` | Environment the run scripts invoke through `conda run`. |

Cohort files under `train_set/` are frozen, deterministic video-ID lists; see
`train_set/README.md` for how they were drawn and why they cannot overlap.

## Captioner API

```python
from surrogate_rollout.captioning import Qwen25VLCaptioner

captioner = Qwen25VLCaptioner(max_images_per_prompt=8)
captions = captioner.caption_batch(
    [["frame_000.jpg", "frame_001.jpg", "frame_002.jpg", "frame_003.jpg"]],
    "Describe the sequence across these frames.",
    max_tokens=128,
)
```

`caption_batch` is the only entry point: one clip per element of the outer list,
one shared prompt or one per clip. Batching is not an optimization here — it is
how the captioning workers keep a GPU busy.

The class name is historical: it serves whatever `SR_CAPTION_MODEL_ID` names.

```bash
python -m surrogate_rollout.scripts.smoke_qwen25vl_captioner --gpu 0
python -m surrogate_rollout.scripts.smoke_qwen25vl_captioner \
  --gpu 0 --image-dir /path/to/frames --num-images 8
```

## Documentation

- `PHASE2_3_SURROGATE.md` — the rollout/caching substrate the loop builds on.
- `PROMPT_DELTA_META_OPTIMIZATION.md` — the current method.
- `docs/RUNBOOK.md` — operating a run.
- `docs/IMPLEMENTATION_MIGRATION.md` — what was removed getting here.
