#!/usr/bin/env bash
# Resume of the full-recaption K-iteration run that died 2026-07-25 08:12 KST on
# a transient DNS failure (logs/full_recaption_20260724_203757.log).
#
# Reuses, by keeping PROMPT_DELTA_ITERATION_TIMESTAMP:
#   runs/prompt_delta_full_recaption_20video_4iteration_20260724_203757_cache
#     511 cached caption segments (2 of 5 evidence videos) + their generator
#     instructions; the cache key carries no git revision, so they stay valid.
#   runs/fresh_prompt_delta_iteration_20260724_203757_inputs
#     the prepared iteration-0 inputs, so no proposer work is repeated.
#
# The partial evidence root was moved to
# runs/fresh_prompt_delta_iteration_20260724_203757_evidence.crashed_dns_20260725:
# its baseline fingerprint pins source_revision c40d3c78, and this resume runs at
# f1f7f9c, which drops the updater's provenance-ID screen -- the bug that forced
# a spurious no_update on this very run. The baseline re-runs the two finished
# videos' DVD QA (~$0.3) rather than resume on the buggy revision.
set -euo pipefail

cd /home/seungmin/youngseo/surrogate_rollout

# The gpt-5-mini stack: captioner Qwen/Qwen3.5-9B, generator/QA/optimizer
# gpt-5-mini, per-component reasoning effort. Same file the crashed run used.
set -a
source scripts/env/gpt5mini_stack.sh
set +a

export SR_CONDA_ENV=capo
export SR_VIDEOMME_DATA_ROOT=/home/seungmin/youngseo/surrogate_rollout/hub_data_kt/seungmin/VideoMME/Video-MME
export PROMPT_DELTA_WORKER_GPUS=5
# The resume key: same experiment root, cache, and cache-reset identity.
export PROMPT_DELTA_ITERATION_TIMESTAMP=20260724_203757
export FRESH_PROMPT_DELTA_WORKER_RESULT_TIMEOUT_SECONDS=5400

# torchcodec's libtorchcodec_core*.so needs the env's own ffmpeg libs on the
# loader path; without this the captioner dies at import (the 05:30 attempt).
export LD_LIBRARY_PATH="/home/seungmin/.conda/envs/capo/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_HOME=/usr/local/cuda-12.8

exec bash full_recaption_opt/scripts/run_full_recaption_kiter.sh
