"""Static meta-prompt (replace-body) policy + parity with the baseline repo.

The private baseline repo runs the same condition standalone. Both sides must
produce the same generator prompt and the same composed caption prompt, so the
shared meta prompt is pinned by sha here and there. If the baseline repo is
checked out next to this one, the parity test compares the two texts directly;
otherwise the sha pin alone guards the drift.
"""
from __future__ import annotations

import json
import os

import pytest

from surrogate_rollout.prompt_routing import static_meta_replace_body as smrb
from surrogate_rollout.prompt_routing.scaffold_applier import (
    create_scaffold_applier,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptEntry,
    RoutingDecision,
    ScaffoldContract,
    ScaffoldPolicySnapshot,
    SegmentContext,
)

BASE_TEMPLATE = """There are consecutive frames from a video. Please understand the video clip with the given transcript then output JSON in the template below.

Transcript of current clip:
TRANSCRIPT_PLACEHOLDER

Output ONLY valid JSON (all string values, including timestamps, must be
double-quoted) in the template below:
{
  "clip_start_time": "CLIP_START_TIME",
  "clip_end_time": "CLIP_END_TIME",
  "clip_description": "<narration>"
}
"""

BASELINE_REPO_MODULE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "baseline_lvbench", "static_meta_prompt.py")


def _history_json():
    return json.dumps({
        "schema_version": smrb.HISTORY_SCHEMA_VERSION,
        "block_index": 0,
        "block_start_seconds": 0.0,
        "block_end_seconds": 300.0,
        "max_history_captions": 30,
        "preceding_captions": [{"segment_id": "20_30", "caption": "prev"}],
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ------------------------------- pinning ---------------------------------- #
def test_meta_prompt_text_is_pinned():
    from surrogate_rollout.schemas import sha256_text

    assert sha256_text(smrb.META_PROMPT_TEXT) == smrb.META_PROMPT_SHA256
    assert not smrb.META_PROMPT_TEXT.endswith("\n")


def test_meta_prompt_matches_the_baseline_repo_when_present():
    if not os.path.exists(BASELINE_REPO_MODULE):
        pytest.skip("baseline repo not checked out next to this one")
    source = open(BASELINE_REPO_MODULE, encoding="utf-8").read()
    assert smrb.META_PROMPT_SHA256 in source, (
        "baseline repo pins a different meta-prompt sha")
    for name, value in (("META_PROMPT_ID", smrb.META_PROMPT_ID),
                        ("META_PROMPT_VERSION", smrb.META_PROMPT_VERSION),
                        ("REQUEST_SCHEMA_VERSION", smrb.REQUEST_SCHEMA_VERSION),
                        ("PARSER_VERSION", smrb.PARSER_VERSION)):
        assert f'{name} = "{value}"' in source, f"{name} drifted"
    assert "GENERATOR_SAMPLE_FPS = 0.5" in source
    assert "GENERATOR_IMAGE_SCALE = 0.5" in source


# ------------------------------ frame policy ------------------------------- #
def test_generator_sees_every_other_frame():
    frames = [f"/f/frame_n{i:06d}.jpg" for i in range(10)]
    selected = smrb.select_generator_frames(frames, caption_fps=1.0)
    assert selected == frames[::2]
    assert len(selected) == 5


def test_frame_subsample_never_interpolates():
    frames = ["/f/a.jpg", "/f/b.jpg"]
    assert smrb.select_generator_frames(
        frames, caption_fps=1.0, generator_fps=4.0) == frames
    assert smrb.select_generator_frames([], caption_fps=1.0) == []


def test_frames_are_halved(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = tmp_path / "frame_n000000.jpg"
    cv2.imwrite(str(source), np.zeros((360, 640, 3), dtype=np.uint8))

    out = smrb.downscale_frames([str(source)], str(tmp_path / "small"))
    assert cv2.imread(out[0]).shape[:2] == (180, 320)


# ---------------------------- generator request ---------------------------- #
def test_request_is_query_independent():
    request = smrb.build_generator_request(
        video_id="v1", segment_id="30_40",
        frame_references=["/f/a.jpg"], serialized_history=_history_json())
    assert request["schema_version"] == smrb.REQUEST_SCHEMA_VERSION
    assert request["generator_identity"]["sample_fps"] == 0.5
    assert request["generator_identity"]["image_scale"] == 0.5
    serialized = smrb.dumps_canonical(request).casefold()
    for forbidden in ("question", "option", "answer", "ground_truth", "score",
                      "correct", "transcript"):
        assert forbidden not in serialized


def test_request_requires_frames_and_history():
    with pytest.raises(smrb.StaticMetaPromptError):
        smrb.build_generator_request(
            video_id="v1", segment_id="30_40", frame_references=[],
            serialized_history=_history_json())
    with pytest.raises(smrb.StaticMetaPromptError):
        smrb.build_generator_request(
            video_id="v1", segment_id="30_40", frame_references=["/f/a.jpg"],
            serialized_history="")


def test_rendered_prompt_is_meta_prompt_then_canonical_json():
    request = smrb.build_generator_request(
        video_id="v1", segment_id="30_40", frame_references=["/f/a.jpg"],
        serialized_history=_history_json())
    rendered = smrb.render_generator_prompt(request)
    assert rendered.startswith(smrb.META_PROMPT_TEXT + "\n")
    assert json.loads(rendered[len(smrb.META_PROMPT_TEXT) + 1:]) == request


def test_parser_rejects_empty_and_preserves_text():
    assert smrb.parse_generated_instruction("  do x  ") == "do x"
    with pytest.raises(smrb.StaticMetaPromptError):
        smrb.parse_generated_instruction("   ")


# ------------------------------ composition -------------------------------- #
def test_generated_text_owns_the_task_section():
    composed, path = smrb.compose_caption_prompt(
        BASE_TEMPLATE, "Track both speakers by their coats.")
    assert path == "replace_body"
    assert composed.startswith("Track both speakers by their coats.")
    assert "There are consecutive frames from a video" not in composed
    assert composed.index(smrb.TRANSCRIPT_HEADER) < composed.index(
        smrb.JSON_CONTRACT_ANCHOR)
    _, contract = smrb.split_contract_tail(BASE_TEMPLATE)
    assert composed.endswith(contract)
    for placeholder in smrb.REQUIRED_PLACEHOLDERS:
        assert composed.count(placeholder) == 1


def test_composition_rejects_placeholder_smuggling_and_empty_text():
    with pytest.raises(smrb.StaticMetaPromptError):
        smrb.compose_caption_prompt(BASE_TEMPLATE, "use CLIP_START_TIME")
    with pytest.raises(smrb.StaticMetaPromptError):
        smrb.compose_caption_prompt(BASE_TEMPLATE, "  ")
    with pytest.raises(smrb.StaticMetaPromptError):
        smrb.compose_caption_prompt("no contract here", "x")


# --------------------------- scaffold applier ------------------------------ #
def _applier():
    policy = ScaffoldPolicySnapshot(
        scaffold_version="scaffold_v0001", policy_type="replace_body",
        configuration={})
    return create_scaffold_applier(
        policy, base_prompt_template=BASE_TEMPLATE), policy


def _entry(prompt_id="free_form_generated", text="Track both speakers."):
    from surrogate_rollout.schemas import sha256_text

    return PromptEntry(prompt_id=prompt_id, prompt_text=text,
                       prompt_hash=sha256_text(text), name="generated",
                       description="", tags=("free_form_generator",),
                       created_by="test", provenance={})


def _context():
    return SegmentContext(
        video_id="v1", segment_id="30_40", timestamp_start=30.0,
        timestamp_end=40.0,
        segment_features={"frame_references": ("/f/a.jpg",)},
        history_summary=_history_json(), metadata={})


def _decision(prompt_ids=("free_form_generated",)):
    return RoutingDecision(
        video_id="v1", segment_id="30_40", bank_version="bank_v0001",
        router_version="router_v8888_abcdef12",
        selected_prompt_ids=tuple(prompt_ids),
        prompt_scores={pid: 1.0 for pid in prompt_ids},
        selection_order=tuple(prompt_ids), decision_payload={})


def _contract():
    return ScaffoldContract(
        contract_version="contract_v0001",
        required_placeholders=smrb.REQUIRED_PLACEHOLDERS,
        max_prompt_tokens=4096)


def test_applier_is_selected_by_policy_type():
    applier, _ = _applier()
    assert applier.policy_type == "replace_body"


def test_applier_composes_like_the_baseline_repo():
    applier, policy = _applier()
    entry = _entry()
    composed = applier.apply(
        context=_context(), selected_entries=(entry,),
        routing_decision=_decision(), scaffold_policy=policy,
        scaffold_contract=_contract())
    expected, _ = smrb.compose_caption_prompt(BASE_TEMPLATE, entry.prompt_text)
    assert composed.is_valid
    assert composed.prompt_text == expected
    assert composed.composition_trace.preserved_prompt_ids == (entry.prompt_id,)


def test_applier_drops_duplicate_entries():
    applier, policy = _applier()
    first = _entry("a", "Same text.")
    second = _entry("b", "Same text.")
    composed = applier.apply(
        context=_context(), selected_entries=(first, second),
        routing_decision=_decision(("a", "b")), scaffold_policy=policy,
        scaffold_contract=_contract())
    assert composed.prompt_text.count("Same text.") == 1
    assert composed.composition_trace.omitted_prompt_ids == ("b",)


def test_applier_requires_a_selection():
    applier, policy = _applier()
    with pytest.raises(ValueError):
        applier.apply(context=_context(), selected_entries=(),
                      routing_decision=_decision(()), scaffold_policy=policy,
                      scaffold_contract=_contract())


def test_contract_breaking_generation_is_invalid_not_a_crash():
    applier, policy = _applier()
    entry = _entry(text="Repeat CLIP_START_TIME in your answer.")
    composed = applier.apply(
        context=_context(), selected_entries=(entry,),
        routing_decision=_decision(), scaffold_policy=policy,
        scaffold_contract=_contract())
    assert not composed.is_valid
    assert composed.validation_errors


# --------------------------- generator factory ----------------------------- #
def test_openai_is_the_default_free_form_provider(monkeypatch):
    from surrogate_rollout.captioning import history_aware_baseline as hab

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    generator = hab.build_free_form_generator(hab.FREE_FORM_PROVIDER_OPENAI)
    identity = generator.configuration_identity
    assert identity["provider"] == "openai"
    assert identity["model"] == "gpt-4o-mini"
    assert identity["composition"] == "replace_body"
    assert identity["template_hash"] == smrb.META_PROMPT_SHA256
    assert identity["generator_sample_fps"] == 0.5
    assert identity["generator_image_scale"] == 0.5


def test_openai_generator_uses_current_parent_and_static_request(monkeypatch):
    from surrogate_rollout.prompt_routing.policies.openai_free_form_generator import (
        OpenAIFreeFormInstructionGenerator,
    )

    generator = OpenAIFreeFormInstructionGenerator(
        api_key="sk-test", model_id="gpt-4o-mini", max_tokens=512,
        template_text="CURRENT PARENT", meta_prompt_id="parent-v2")
    monkeypatch.setattr(
        smrb, "prepare_generator_frames",
        lambda files, caption_fps: ["/f/a.small.jpg"])
    captured = {}

    def complete(frames, prompt):
        captured.update(frames=tuple(frames), prompt=prompt)
        return "Track the visible hand movement."

    monkeypatch.setattr(generator, "_complete", complete)
    generated = generator.generate(_context(), {})
    assert generated.instruction == "Track the visible hand movement."
    assert captured["frames"] == ("/f/a.small.jpg",)
    assert captured["prompt"].startswith("CURRENT PARENT\n")
    request = generator.last_exchange.request
    assert request["schema_version"] == smrb.REQUEST_SCHEMA_VERSION
    assert request["generator_identity"]["meta_prompt_id"] == "parent-v2"
    assert generator.configuration_identity["provider"] == "openai"


def test_unknown_provider_aborts():
    from surrogate_rollout.captioning import history_aware_baseline as hab

    with pytest.raises(ValueError):
        hab.build_free_form_generator("gemini")


# ------------------------ captioner input isolation ------------------------ #
class _RecordingVLM:
    """Captures exactly what the captioner was asked to look at."""

    def __init__(self):
        self.calls = []

    def caption(self, frames, prompt, **kwargs):
        self.calls.append({"frames": tuple(frames), "prompt": prompt})
        return '{"clip_description": "a caption"}'


def _history_snapshot():
    from surrogate_rollout.captioning.history_aware_baseline import (
        build_history_snapshot,
    )

    return build_history_snapshot(
        segment_id="30_40", block_seconds=300.0,
        preceding=[{"segment_id": "20_30", "caption": "a previous caption"}],
        max_history_captions=30)


def _caption_once(tmp_path, *, caption_sees_history):
    from surrogate_rollout.captioning.history_aware_baseline import (
        HistoryAwareSegmentCaptioner,
    )

    vlm = _RecordingVLM()
    captioner = HistoryAwareSegmentCaptioner(
        vlm, backend_id="test-backend",
        caption_sees_history=caption_sees_history)
    entry = _entry()
    applier, policy = _applier()
    composed = applier.apply(
        context=_context(), selected_entries=(entry,),
        routing_decision=_decision(), scaffold_policy=policy,
        scaffold_contract=_contract())
    frames = tuple(str(tmp_path / f"frame_n{i:06d}.jpg") for i in range(10))
    captioner.caption(
        sample={"video_path": "/v/a.mp4", "extra": {}}, video_id="v1",
        segment_id="30_40",
        clip_info={"files": frames, "transcript": "clip transcript"},
        composed_prompt=composed, history_snapshot=_history_snapshot(),
        merge_prompt="merge", cache_root=str(tmp_path / "cache"),
        cache_manifest_path=str(tmp_path / "cache" / "manifest.jsonl"),
        intervention_identity_hash=None)
    return vlm.calls[0], frames


def test_captioner_sees_only_the_composed_prompt_and_its_own_frames(tmp_path):
    call, frames = _caption_once(tmp_path, caption_sees_history=False)
    assert "a previous caption" not in call["prompt"]
    assert "FROZEN_PRECEDING_CAPTION_HISTORY_JSON" not in call["prompt"]
    assert "preceding_captions" not in call["prompt"]
    # every full-resolution frame of the segment, and nothing downscaled
    assert call["frames"] == frames
    assert not any("generator_frames_scale" in path for path in call["frames"])


def test_property_bank_default_keeps_the_historical_caption_prompt(tmp_path):
    call, _ = _caption_once(tmp_path, caption_sees_history=True)
    assert "FROZEN_PRECEDING_CAPTION_HISTORY_JSON" in call["prompt"]


def test_caption_history_visibility_is_part_of_the_captioner_identity():
    from surrogate_rollout.captioning.history_aware_baseline import (
        HistoryAwareSegmentCaptioner,
    )

    seeing = HistoryAwareSegmentCaptioner(_RecordingVLM(), backend_id="b")
    blind = HistoryAwareSegmentCaptioner(
        _RecordingVLM(), backend_id="b", caption_sees_history=False)
    assert seeing.configuration_identity["caption_sees_history"] is True
    assert blind.configuration_identity["caption_sees_history"] is False


def test_free_form_condition_blinds_the_captioner(monkeypatch):
    """from_local_qwen wires the flag off for the active meta-prompt path."""
    from surrogate_rollout.captioning import history_aware_baseline as hab

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        hab, "build_free_form_generator",
        lambda provider, **kwargs: object())
    monkeypatch.setattr(hab, "get_local_qwen_backend", lambda: _RecordingVLM())

    free_form = hab.HistoryAwareBaselineCaptionViewBuilder.from_local_qwen(
        routing_mode="free_form_generator")
    assert free_form.segment_captioner.caption_sees_history is False
