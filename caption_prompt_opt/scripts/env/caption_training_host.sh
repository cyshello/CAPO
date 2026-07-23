# Caption-prompt training-host profile. Self-contained analog of the reference
# scripts/env/training_host.sh, which does NOT exist on the capo-main snapshot.
# Source it (after gpt5mini_stack.sh if that exists) with:
#   set -a && source caption_prompt_opt/scripts/env/caption_training_host.sh && set +a
#
# Every value is a default (:-), so anything already exported wins. It sets the
# 20-video x 5-iteration schedule and a SINGLE experiment label shared by the
# K-loop driver and its watcher (a divergent label would make the watcher watch
# the wrong run directory and restart forever).

# Four GPUs, one captioning worker each.
PROMPT_DELTA_WORKER_GPUS=${PROMPT_DELTA_WORKER_GPUS:-0,1,2,3}

# Captioner. Cache is keyed on this identity; cold start, re-captions every clip.
SR_CAPTION_MODEL_ID=${SR_CAPTION_MODEL_ID:-Qwen/Qwen3.5-9B}

# The schedule: 4 evidence videos per iteration out of the frozen 20-video
# cohort, 5 iterations. Held-out (confirmation) cohort file must exist but its
# videos are never captioned on this host (enqueue policy).
PROMPT_DELTA_ITERATION_COUNT=${PROMPT_DELTA_ITERATION_COUNT:-5}
PROMPT_DELTA_VIDEOS_PER_ITERATION=${PROMPT_DELTA_VIDEOS_PER_ITERATION:-4}
PROMPT_DELTA_EVIDENCE_COHORT_FILE=${PROMPT_DELTA_EVIDENCE_COHORT_FILE:-train_set/20samples.txt}
PROMPT_DELTA_CONFIRMATION_COHORT_FILE=${PROMPT_DELTA_CONFIRMATION_COHORT_FILE:-train_set/confirmation.txt}

# One label shared by run_caption_kiter.sh and watch_caption_experiment.sh.
PROMPT_DELTA_EXPERIMENT_LABEL=${PROMPT_DELTA_EXPERIMENT_LABEL:-caption_20video_5iter}
