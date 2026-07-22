"""Registry-JSON caption contract with a description-only plain-text fallback.

The captioner asks for the canonical DVD registry JSON first and falls back to
a plain-text contract once `config.CAPTION_JSON_ATTEMPTS` registry attempts
have failed to parse.
"""
from __future__ import annotations

from surrogate_rollout import config
from surrogate_rollout.captioning.history_aware_baseline import (
    CAPTION_OUTPUT_CONTRACT,
    PLAIN_TEXT_CAPTION_OUTPUT_CONTRACT,
    HistoryAwareSegmentCaptioner,
    _parse_caption_output_with_metadata,
    build_history_snapshot,
)
from test_static_meta_replace_body import (
    _applier, _contract, _context, _decision, _entry,
)


REGISTRY_JSON = (
    '{"clip_start_time": "00:00:30", "clip_end_time": "00:00:40", '
    '"subject_registry": {"chickens": {"name": "chickens", '
    '"appearance": ["brown"], "identity": ["birds"], '
    '"first_seen": "00:00:30"}}, '
    '"clip_description": "Chickens peck at grain in a trough."}'
)


class _ScriptedVLM:
    """Returns a fixed sequence of raw outputs and records every prompt."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def caption(self, frames, prompt, **kwargs):
        self.calls.append(prompt)
        return self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]


def _caption(tmp_path, outputs):
    vlm = _ScriptedVLM(outputs)
    captioner = HistoryAwareSegmentCaptioner(vlm, backend_id="test-backend")
    applier, policy = _applier()
    composed = applier.apply(
        context=_context(), selected_entries=(_entry(),),
        routing_decision=_decision(), scaffold_policy=policy,
        scaffold_contract=_contract())
    history = build_history_snapshot(
        segment_id="30_40", block_seconds=300.0,
        preceding=[{"segment_id": "20_30", "caption": "a previous caption"}],
        max_history_captions=30)
    result = captioner.caption(
        sample={"video_path": "/v/a.mp4", "extra": {}}, video_id="v1",
        segment_id="30_40",
        clip_info={"files": (str(tmp_path / "frame_n000000.jpg"),),
                   "transcript": "clip transcript"},
        composed_prompt=composed, history_snapshot=history,
        merge_prompt="merge", cache_root=str(tmp_path / "cache"),
        cache_manifest_path=str(tmp_path / "cache" / "manifest.jsonl"),
        intervention_identity_hash=None)
    return result, vlm


# ------------------------------- parsing ----------------------------------- #
def test_registry_json_parse_keeps_description_and_registry():
    parsed, meta = _parse_caption_output_with_metadata(REGISTRY_JSON)
    assert meta["parse_path"] == "registry_json"
    assert parsed["clip_description"] == "Chickens peck at grain in a trough."
    assert parsed["subject_registry"]["chickens"]["name"] == "chickens"


def test_markdown_fenced_registry_json_is_parsed():
    parsed, meta = _parse_caption_output_with_metadata(
        f"```json\n{REGISTRY_JSON}\n```")
    assert meta["parse_path"] == "registry_json"
    assert "strip_markdown_fence" in meta["normalizations"]
    assert parsed["subject_registry"]


def test_unparsable_output_becomes_a_plain_text_description():
    parsed, meta = _parse_caption_output_with_metadata("Chickens peck.")
    assert meta["parse_path"] == "plain_text"
    assert parsed == {"clip_description": "Chickens peck.",
                      "subject_registry": {}}


def test_json_without_a_description_is_not_a_registry_parse():
    parsed, meta = _parse_caption_output_with_metadata(
        '{"subject_registry": {"a": {}}}')
    assert meta["parse_path"] == "plain_text"
    assert parsed["subject_registry"] == {}


def test_empty_output_stays_invalid():
    parsed, meta = _parse_caption_output_with_metadata("   ")
    assert parsed == {}
    assert meta["status"] == "invalid"


# ------------------------------ retry policy ------------------------------- #
def test_registry_json_on_the_first_attempt_stops_immediately(tmp_path):
    result, vlm = _caption(tmp_path, [REGISTRY_JSON])
    assert len(vlm.calls) == 1
    assert CAPTION_OUTPUT_CONTRACT in vlm.calls[0]
    assert result.parsed["subject_registry"]
    assert result.retry_count == 0


def test_plain_text_is_retried_as_json_then_accepted_as_plain_text(tmp_path):
    result, vlm = _caption(tmp_path, ["Chickens peck at grain."])
    assert len(vlm.calls) == config.CAPTION_JSON_ATTEMPTS + 1
    for prompt in vlm.calls[:config.CAPTION_JSON_ATTEMPTS]:
        assert CAPTION_OUTPUT_CONTRACT in prompt
    assert PLAIN_TEXT_CAPTION_OUTPUT_CONTRACT in vlm.calls[-1]
    assert result.parsed == {"clip_description": "Chickens peck at grain.",
                             "subject_registry": {}}


def test_registry_json_after_one_failure_skips_the_plain_text_contract(tmp_path):
    result, vlm = _caption(tmp_path, ["not json", REGISTRY_JSON])
    assert len(vlm.calls) == 2
    assert all(CAPTION_OUTPUT_CONTRACT in prompt for prompt in vlm.calls)
    assert result.parsed["subject_registry"]
