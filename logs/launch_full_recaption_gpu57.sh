#!/usr/bin/env bash
# Fresh full-recaption run on GPUs 5 and 7: five iterations x four evidence
# videos over the same 20-video pool, held out against the 15-video validation
# cohort -- the shape the meta run it is compared against uses
# (label 20video_5iter_val15). The 20video_4iteration preset defaults to
# 4x5 with a 10-video held-out set, so all three are overridden below.
#
# Replaces the 20260724_203757 run, which was abandoned on 2026-07-25: it had
# no segment-level resume, because the free-form generator re-sampled its
# instruction on every restart and the instruction is part of the caption cache
# key (587 cached segments, zero hits). Branch generator-response-cache
# (commit 5ff3726) caches the generator answer by request identity, so this run
# resumes at the segment instead of the video.
#
# No PROMPT_DELTA_ITERATION_TIMESTAMP here: this is a new experiment root. To
# resume THIS run after a crash, re-run with the timestamp printed in its log
# (run root: runs/prompt_delta_full_recaption_20video_5iter_val15_<ts>). The
# worker GPU list is part of the baseline input fingerprint, so a resume has to
# keep 5,7.
set -euo pipefail

cd /home/seungmin/youngseo/surrogate_rollout

# The gpt-5-mini stack: captioner Qwen/Qwen3.5-9B, generator/QA/optimizer
# gpt-5-mini, per-component reasoning effort.
set -a
source scripts/env/gpt5mini_stack.sh
set +a

export SR_CONDA_ENV=capo
export SR_VIDEOMME_DATA_ROOT=/home/seungmin/youngseo/surrogate_rollout/hub_data_kt/seungmin/VideoMME/Video-MME
export PROMPT_DELTA_WORKER_GPUS=5,7
export FRESH_PROMPT_DELTA_WORKER_RESULT_TIMEOUT_SECONDS=5400

# Experiment shape: 5 x 4 = the same 20 evidence videos, consumed four per
# iteration, plus the 15 held-out videos (train_set/README.md: permutation
# positions 51-65, disjoint from every training cohort).
export PROMPT_DELTA_ITERATION_COUNT=5
export PROMPT_DELTA_VIDEOS_PER_ITERATION=4
export PROMPT_DELTA_EVIDENCE_COHORT_FILE="$PWD/train_set/20samples.txt"
export PROMPT_DELTA_CONFIRMATION_COHORT_FILE="$PWD/train_set/confirmation.txt"
# Label -> run/state/cache/memory dirs and the measurement queue. Mirrors the
# meta run's label with the variant prefix, so the two never share a directory.
export PROMPT_DELTA_EXPERIMENT_LABEL=full_recaption_20video_5iter_val15

# No held-out measurement inside the loop: promotion happens on the evidence
# cohort alone and each candidate's held-out request is written to
# runs/full_recaption_20video_5iter_val15_measurement_queue instead of being
# scored. Nothing drains that queue until a measurement worker is started, so
# the run performs no held-out captioning or QA at all.
export FRESH_PROMPT_DELTA_PROMOTION_POLICY=promote_and_enqueue_measurement_v1

# SR_GENERATOR_CACHE_ROOT is deliberately unset: the evidence entry points it at
# this run's own cache_root before the workers spawn. Export it empty to run
# without generator reuse.

# Transient-transport retry (branch network-transient-retry, commit 840532f,
# which this branch is built on) is active with its defaults: a request that
# fails on DNS/connection/429/5xx is repeated byte-identically for up to 15
# minutes, while a rejected request still fails immediately. It covers the
# generator, the feedback/updater/proposer transport, and the vendored DVD QA
# path. Override with SR_NETWORK_RETRY_DEADLINE_SECONDS if a run needs longer.

# torchcodec's libtorchcodec_core*.so needs the env's own ffmpeg libs on the
# loader path; without this the captioner dies at import.
export LD_LIBRARY_PATH="/home/seungmin/.conda/envs/capo/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_HOME=/usr/local/cuda-12.8

exec bash full_recaption_opt/scripts/run_full_recaption_kiter.sh
