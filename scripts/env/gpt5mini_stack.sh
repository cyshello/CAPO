# The gpt-5-mini optimization stack. Every value is a default: anything already
# set in the environment wins, so one model can be swapped without editing the
# file. Source this before launching a run:
#   set -a && source scripts/env/gpt5mini_stack.sh && set +a
#
# Captioning stays on the local Qwen model. Everything hosted moves to
# gpt-5-mini, with reasoning effort set per component: the DVD orchestrator is
# prefill-bound and gains nothing from reasoning, the prompt generator writes
# the instruction each caption depends on, and the updater is the one place a
# set of episodes has to become a single general rule.
#
# Measured 2026-07-23 on one segment, three calls each:
#   generator  minimal  3.7s   medium  10.8s   (gpt-4o-mini was ~2s)
#   DVD tool call        2.2s            (gpt-4o was 1.3s)
#   feedback   high    ~100s per episode; medium is roughly a third of that

SR_ORCHESTRATOR_TOOL_MODEL=${SR_ORCHESTRATOR_TOOL_MODEL:-gpt-5-mini}
SR_DVD_REASONING_EFFORT=${SR_DVD_REASONING_EFFORT:-minimal}
# gpt-5-mini rejected one evidence video's transcript as prompt-policy
# violating (400 invalid_prompt), which recurs on every retry. gpt-4o accepts
# it, so it answers the requests the reasoning model refuses. Codex is no
# longer reachable as a fallback -- its ChatGPT-account quota is exhausted.
SR_ORCHESTRATOR_TOOL_FALLBACK_MODEL=${SR_ORCHESTRATOR_TOOL_FALLBACK_MODEL:-gpt-4o}

FRESH_PROMPT_DELTA_GENERATOR_MODEL_ID=${FRESH_PROMPT_DELTA_GENERATOR_MODEL_ID:-gpt-5-mini}
# The evidence runner cross-checks the prepared config against config.py, so
# the generator identity has to be set on both sides or it refuses to start.
SR_PROMPT_GENERATOR_MODEL_ID=${SR_PROMPT_GENERATOR_MODEL_ID:-gpt-5-mini}
SR_GENERATOR_REASONING_EFFORT=${SR_GENERATOR_REASONING_EFFORT:-minimal}
# Reasoning is billed and capped as output. At 512 the generator returned an
# empty completion every time, which is a hard failure once per segment; 4096
# leaves room for the reasoning and the instruction. The generator refuses to
# start below 2048 with an effort set, so this is not silently reducible.
FRESH_PROMPT_DELTA_GENERATOR_MAX_OUTPUT_TOKENS=${FRESH_PROMPT_DELTA_GENERATOR_MAX_OUTPUT_TOKENS:-4096}

FRESH_PROMPT_DELTA_OPTIMIZER_MODEL_ID=${FRESH_PROMPT_DELTA_OPTIMIZER_MODEL_ID:-gpt-5-mini}
SR_FEEDBACK_REASONING_EFFORT=${SR_FEEDBACK_REASONING_EFFORT:-medium}
SR_UPDATER_REASONING_EFFORT=${SR_UPDATER_REASONING_EFFORT:-high}
FRESH_PROMPT_DELTA_OPTIMIZER_MAX_OUTPUT_TOKENS=${FRESH_PROMPT_DELTA_OPTIMIZER_MAX_OUTPUT_TOKENS:-32000}

# The minimal feedback payload and the structural tool-evidence compaction.
SR_EPISODE_FEEDBACK_VIEW=${SR_EPISODE_FEEDBACK_VIEW:-1}
SR_COMPACT_TOOL_EVIDENCE=${SR_COMPACT_TOOL_EVIDENCE:-1}
