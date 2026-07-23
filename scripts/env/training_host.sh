# Training host: the optimization loop only, on a machine that is not the one
# the completed iterations ran on. Source it after the model stack:
#   set -a
#   source scripts/env/gpt5mini_stack.sh
#   source scripts/env/training_host.sh
#   set +a
#
# What this host does NOT do: held-out measurement. The promotion policy
# enqueues those requests instead of running them, so a separate
# scripts/run_measurement_worker.py drains the queue -- possibly on another
# machine entirely. The consequence for setup is that only the evidence cohort's
# videos have to exist here; the confirmation videos do not, though the
# confirmation cohort *file* still does.

# Four GPUs, one captioning worker each.
PROMPT_DELTA_WORKER_GPUS=0,1,2,3

# A different captioner from the 3090 host's Qwen2.5-VL-7B. The caption cache is
# keyed on this identity, so the captions produced here never collide with the
# earlier model's -- and never reuse them either: this host starts cold and
# re-captions every clip.
SR_CAPTION_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct

# The 5-iteration schedule the 3090 host is running: 4 evidence videos per
# iteration out of the frozen 20-video cohort, held-out set of 15.
PROMPT_DELTA_ITERATION_COUNT=5
PROMPT_DELTA_VIDEOS_PER_ITERATION=4
PROMPT_DELTA_EVIDENCE_COHORT_FILE=train_set/20samples.txt
PROMPT_DELTA_CONFIRMATION_COHORT_FILE=train_set/confirmation.txt
PROMPT_DELTA_EXPERIMENT_LABEL=20video_5iter_val15

# Deferred measurement: promote the candidate, write the scoring request to the
# queue, keep going. A measurement failure can then never stop the loop.
FRESH_PROMPT_DELTA_PROMOTION_POLICY=promote_and_enqueue_measurement_v1

# The conda environment the run scripts invoke.
SR_CONDA_ENV=capo
