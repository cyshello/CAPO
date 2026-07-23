"""A failing DVD tool call must not unwind out of the agent loop.

Upstream `_exec_tool` re-raised everything except `StopException`, so one bad
tool argument ended the question, and with it the intervention, the evidence run
and the whole multi-iteration experiment. The vendored copy reports the failure
back to the agent as a tool message instead.
"""
from __future__ import annotations

import sys

import pytest

from surrogate_rollout import config

for _path in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dvd.dvd_core import DVDCoreAgent, StopException  # noqa: E402


def _agent(tool):
    """A bare agent: `_exec_tool` needs only the map, the db and the appender."""
    agent = DVDCoreAgent.__new__(DVDCoreAgent)
    agent.name_to_function_map = {"frame_inspect_tool": tool}
    agent.video_db = object()
    return agent


def _call(arguments: str = '{"time_ranges_hhmmss": [["00:40:00", "00:41:00"]]}'):
    return {"id": "call-1",
            "function": {"name": "frame_inspect_tool", "arguments": arguments}}


def test_a_raising_tool_becomes_a_tool_message():
    def tool(**_kwargs):
        raise ValueError("One of start time 00:40:00 exceeds video length 1996")

    msgs = []
    _agent(tool)._exec_tool(_call(), msgs)

    assert len(msgs) == 1
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["name"] == "frame_inspect_tool"
    assert "exceeds video length" in msgs[0]["content"]
    assert "Correct the arguments" in msgs[0]["content"]


def test_a_successful_tool_is_unchanged():
    msgs = []
    _agent(lambda **_kwargs: "two people at a table")._exec_tool(_call(), msgs)
    assert msgs[0]["content"] == "two people at a table"


def test_stop_exception_still_ends_the_loop():
    def tool(**_kwargs):
        raise StopException("B")

    with pytest.raises(StopException):
        _agent(tool)._exec_tool(_call(), [])


def test_an_unknown_tool_name_is_reported_not_raised():
    msgs = []
    call = {"id": "call-1",
            "function": {"name": "no_such_tool", "arguments": "{}"}}
    _agent(lambda **_kwargs: "unused")._exec_tool(call, msgs)
    assert "Invalid function name" in msgs[0]["content"]
