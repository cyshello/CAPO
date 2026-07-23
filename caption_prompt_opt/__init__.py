"""Independent caption-prompt optimization path.

This package runs the *same* Checkpoint F/G optimization loop as
``surrogate_rollout.optimization``, but optimizes the caption prompt (the task
body the captioner receives) directly instead of the meta-prompt that a
prompt-generator LLM consumes.

It adds no files inside ``surrogate_rollout`` and imports that package
read-only, so it can run concurrently with a live meta-prompt optimization as
long as its artifact directories, cache root, and identities are disjoint (see
README.md). Feedback and proposer are reused verbatim; only the updater's
system prompt and the per-segment generator differ.
"""
