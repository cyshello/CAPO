"""A captioner must not think.

Reasoning tokens are billed against the same budget as the caption and the
result is written into a cache that every later comparison reads, so a reasoning
trace stored as a description would corrupt the experiment silently rather than
fail it.
"""

from __future__ import annotations

from surrogate_rollout.captioning.qwen25_vl import (
    Qwen25VLCaptioner,
    _without_reasoning,
)


def _captioner_with_template(template):
    captioner = object.__new__(Qwen25VLCaptioner)
    captioner.processor = type("P", (), {"chat_template": template})()
    return captioner


def test_thinking_is_disabled_when_the_template_supports_it():
    captioner = _captioner_with_template(
        "{% if enable_thinking %}<think>{% endif %}")
    assert captioner._template_kwargs() == {"enable_thinking": False}


def test_no_keyword_for_templates_that_never_heard_of_it():
    # Qwen2.5-VL: passing the keyword would be an unexpected argument.
    captioner = _captioner_with_template("{{ messages }}")
    assert captioner._template_kwargs() == {}


def test_missing_template_is_treated_as_unsupported():
    captioner = object.__new__(Qwen25VLCaptioner)
    captioner.processor = object()
    assert captioner._template_kwargs() == {}


def test_plain_caption_survives_untouched():
    assert _without_reasoning("  A person walks a dog.  ") == "A person walks a dog."


def test_a_closed_reasoning_block_is_dropped():
    text = "<think>The frames show a street.</think>\nA person walks a dog."
    assert _without_reasoning(text) == "A person walks a dog."


def test_an_unclosed_block_yields_no_caption():
    # The budget ran out mid-reasoning: there is no description in the output,
    # and returning the trace would cache it as one.
    assert _without_reasoning("<think>Let me look at frame one and") == ""


def test_a_caption_that_merely_mentions_thinking_is_kept():
    text = "A person appears to be thinking about <think> tags."
    assert _without_reasoning(text) == text
