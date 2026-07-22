"""Central registry of every static prompt used by the DVD agent.

All prompts that were previously hard-coded inside dvd_core.py,
frame_caption.py and build_database.py now live here as fields of a single
mutable `PROMPTS` object. The internal modules read from `PROMPTS` at call
time, so overriding a field before running the agent changes its behaviour
without touching agent architecture.

Placeholder tokens inside the templates are substituted by the internal code
via str.replace / str.format and MUST be preserved when overriding:

    orchestrator_user_template : VIDEO_LENGTH, QUESTION_PLACEHOLDER
    caption_prompt             : TRANSCRIPT_PLACEHOLDER, CLIP_START_TIME, CLIP_END_TIME
    merge_prompt               : REGISTRIES_PLACEHOLDER
    frame_inspect_user_template: {question}
    global_browse_user_template: {question}, {clip_captions}
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields


# --------------------------------------------------------------------------- #
#                        Default (original) prompt text                       #
# --------------------------------------------------------------------------- #

_ORCHESTRATOR_SYSTEM = """You are a helpful assistant who answers multi-step questions by sequentially invoking functions. Follow the THINK → ACT → OBSERVE loop:
  • THOUGHT Reason step-by-step about which function to call next.
  • ACTION   Call exactly one function that moves you closer to the final answer.
  • OBSERVATION Summarize the function's output.
You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls.
Only pass arguments that come verbatim from the user or from earlier function outputs—never invent them. Continue the loop until the user's query is fully resolved, then end your turn with the final answer. If you are uncertain about code structure or video content, use the available tools to inspect rather than guessing. Plan carefully before each call and reflect on every result. Do not rely solely on blind function calls, as this degrades reasoning quality. Timestamps may be formatted as 'HH:MM:SS' or 'MM:SS'."""

_ORCHESTRATOR_USER = """Carefully read the timestamps and narration in the following script, paying attention to the causal order of events, object details and movements, and people's actions and poses.

Here are tools you can use to reveal your reasoning process whenever the provided information is insufficient.

• To get a global information about events and main subjects in the video, use `global_browse_tool`.
• To search without a specific timestamp, use `clip_search_tool`.
• If the retrieved material lacks precise, question-relevant detail (e.g., an unknown name), call `frame_inspect_tool` with a list of time ranges (list[tuple[HH:MM:SS, HH:MM:SS]]).
• Whenever you are uncertain of an answer after searching, inspect frames in the relevant intervals with `frame_inspect_tool`.
• After locating an answer in the script, always make a **CONFIRM** with `frame_inspect_tool` query.


You can first use `global_browse_tool` to a global information about this video, then invoke multiple times of these tools to prgressively find the answer.

Based on your observations and tool outputs, provide a concise answer that directly addresses the question. \n

Total video length: VIDEO_LENGTH seconds.

Question: QUESTION_PLACEHOLDER"""

_FORCE_FINISH_USER = "Please call the `finish` function to finish the task."

# --- captioning (frame_caption.py) ---------------------------------------- #

_CAPTION_SYSTEM = "You are a helpful assistant."

_CAPTION_PROMPT = """There are consecutive frames from a video. Please understand the video clip with the given transcript then output JSON in the template below.

Transcript of current clip:
TRANSCRIPT_PLACEHOLDER

Output ONLY valid JSON (all string values, including timestamps, must be
double-quoted) in the template below:
{
  "clip_start_time": "CLIP_START_TIME",
  "clip_end_time": "CLIP_END_TIME",
  "subject_registry": {
    "<subject_i>": {
      "name": "<fill with short identity if name is unknown>",
      "appearance": ["<appearance description>", "..."],
      "identity": ["<identity description>", "..."],
      "first_seen": "<timestamp>"
    }
  },
  "clip_description": "<smooth and detailed natural narration of the video clip>"
}
"""

_MERGE_PROMPT = """You are given several partial `new_subject_registry` JSON objects extracted from different clips of the *same* video. They may contain duplicated subjects with slightly different IDs or descriptions.

Task:
1. Merge these partial registries into one coherent `subject_registry`.
2. Preserve all unique subjects.
3. If two subjects obviously refer to the same person, merge them
   (keep earliest `first_seen` time and union all fields).

Input (list of JSON objects):
REGISTRIES_PLACEHOLDER

Return *only* the merged `subject_registry` JSON object.
"""

# --- tool VLM prompts (build_database.py) --------------------------------- #

_FRAME_INSPECT_SYSTEM = "You are a helpful assistant to answer questions."

_FRAME_INSPECT_USER = "Carefully watch the video frames. Pay attention to the cause and sequence of events, the detail and movement of objects and the action and pose of persons.\n\nBased on your observations, if you find content that can answer the question. If no relevant content is found within the given time range, return: `Error: Cannot find corresponding result in the given time range.`. \nQuestion: {question}\n"

_GLOBAL_BROWSE_SYSTEM = "You are a knowledgeable assistant specializing in analyzing video content and providing detailed, insightful answers."

_GLOBAL_BROWSE_USER = (
    "Below are descriptions of video clips, each with its corresponding timestamp. "
    "Carefully review the sequence of events, the details and movements of objects, and the actions and poses of people. "
    "Based on these observations, provide a thorough and specific answer to the following question, referencing key events and timestamps as appropriate.\n"
    "Question: {question}\n\n{clip_captions}"
)

# --- tool schema descriptions (function-calling metadata) ----------------- #
# These are sent to the orchestrator as the tools' `description` / parameter
# descriptions and therefore also steer the agent.

_FINISH_DESC = "Call this function after confirming the answer of the user's question, and finish the conversation."
_FINISH_ANSWER_ARG = "Answer to the user's question."

_FRAME_INSPECT_DESC = (
    "Crop the video frames based on the time ranges and ask the model a detailed "
    "question about the cropped video clips.\n"
    "Returns:\n"
    "    str: The model's response to the question. If no relevant content is found "
    "within the time range,\n"
    "         returns an error message: \"Error: Cannot find corresponding result in "
    "the given time range.\""
)
_FRAME_INSPECT_ARG_DATABASE = "The database containing video metadata. Must be an instance of NanoVectorDB."
_FRAME_INSPECT_ARG_QUESTION = "The specific detailed question to ask about the video content during the specified time ranges. No need to add time ranges in the question."
_FRAME_INSPECT_ARG_TIME_RANGES = "A list of tuples containing start and end times in HH:MM:SS format. If the time range is longer than 50 seconds, the function samples 50 evenly distributed frames.  Otherwise, it uses all frames within the specified range."

_CLIP_SEARCH_DESC = (
    "Searches for events in a video clip database based on a given event description "
    "and retrieves the top-k most relevant video clip captions.\n\n"
    "Returns:\n"
    "    str: A formatted string containing the concatenated captions of the searched "
    "video clip scripts.\n\n"
    "Notes:\n"
    "    - This function utilizes the OpenAI Embedding Service to generate embeddings "
    "for the input text.\n"
    "    - Use default values for `top_k` to limit the number of results returned."
)
_CLIP_SEARCH_ARG_DATABASE = "The database object that supports querying with embeddings."
_CLIP_SEARCH_ARG_EVENT = "A textual description of the event to search for."
_CLIP_SEARCH_ARG_TOPK = "The maximum number of top results to retrieve. Just use the default value."

_GLOBAL_BROWSE_DESC = (
    "Analyzes a video database to answer a detailed question by first searching for "
    "relevant video clips and then generating a comprehensive answer based on their "
    "captions.\n\n"
    "Args:\n"
    "    database (NanoVectorDB): The database object that supports querying with "
    "embeddings.\n"
    "    query (str): A textual description or question to search for in the video.\n\n"
    "Returns:\n"
    "    str: A JSON string containing the subject registry and the model's answer to "
    "the query based on the most relevant video clips."
)
_GLOBAL_BROWSE_ARG_DATABASE = "The database object that supports querying with embeddings."
_GLOBAL_BROWSE_ARG_QUERY = "A textual description which will be used to search for relevant video clips in the database."


# --------------------------------------------------------------------------- #
#                              Prompt container                               #
# --------------------------------------------------------------------------- #
@dataclass
class DVDPrompts:
    """Every static prompt the DVD agent uses. Fields are freely overridable."""

    # orchestrator (dvd_core.py)
    orchestrator_system: str = _ORCHESTRATOR_SYSTEM
    orchestrator_user_template: str = _ORCHESTRATOR_USER
    force_finish_user: str = _FORCE_FINISH_USER

    # captioning (frame_caption.py)
    caption_system: str = _CAPTION_SYSTEM
    caption_prompt: str = _CAPTION_PROMPT
    merge_prompt: str = _MERGE_PROMPT

    # tool VLM prompts (build_database.py)
    frame_inspect_system: str = _FRAME_INSPECT_SYSTEM
    frame_inspect_user_template: str = _FRAME_INSPECT_USER
    global_browse_system: str = _GLOBAL_BROWSE_SYSTEM
    global_browse_user_template: str = _GLOBAL_BROWSE_USER

    # tool schema descriptions (function-calling metadata)
    finish_desc: str = _FINISH_DESC
    finish_answer_arg: str = _FINISH_ANSWER_ARG
    frame_inspect_desc: str = _FRAME_INSPECT_DESC
    frame_inspect_arg_database: str = _FRAME_INSPECT_ARG_DATABASE
    frame_inspect_arg_question: str = _FRAME_INSPECT_ARG_QUESTION
    frame_inspect_arg_time_ranges: str = _FRAME_INSPECT_ARG_TIME_RANGES
    clip_search_desc: str = _CLIP_SEARCH_DESC
    clip_search_arg_database: str = _CLIP_SEARCH_ARG_DATABASE
    clip_search_arg_event: str = _CLIP_SEARCH_ARG_EVENT
    clip_search_arg_topk: str = _CLIP_SEARCH_ARG_TOPK
    global_browse_desc: str = _GLOBAL_BROWSE_DESC
    global_browse_arg_database: str = _GLOBAL_BROWSE_ARG_DATABASE
    global_browse_arg_query: str = _GLOBAL_BROWSE_ARG_QUERY

    def override(self, **kwargs) -> "DVDPrompts":
        """Set the given fields in place, validating names. Returns self."""
        valid = {f.name for f in fields(self)}
        for key, value in kwargs.items():
            if key not in valid:
                raise KeyError(
                    f"unknown prompt {key!r}; valid: {sorted(valid)}"
                )
            setattr(self, key, value)
        return self

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# The live, mutable registry the internal modules read from.
PROMPTS = DVDPrompts()

# Pristine copy for reset().
_DEFAULTS = copy.deepcopy(PROMPTS)


def get_prompts() -> DVDPrompts:
    """Return the live prompt registry."""
    return PROMPTS


def set_prompts(**overrides) -> DVDPrompts:
    """Override fields on the live registry. Returns it."""
    return PROMPTS.override(**overrides)


def reset_prompts() -> DVDPrompts:
    """Restore every prompt to its original default text."""
    for f in fields(PROMPTS):
        setattr(PROMPTS, f.name, getattr(_DEFAULTS, f.name))
    return PROMPTS
