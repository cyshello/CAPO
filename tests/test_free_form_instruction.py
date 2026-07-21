"""Free-form instruction baseline: plain-text parsing, transcript policy, cache
identity separation, and mocked end-to-end integration through the shared
scaffold + captioner + cache infrastructure. No real model inference."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from surrogate_rollout import config
from surrogate_rollout.cache.caption_cache import (
    build_history_aware_cache_key,
    new_history_aware_cache_dir,
)
from surrogate_rollout.captioning.history_aware_baseline import (
    HistoryAwareBaselineCaptionViewBuilder,
    HistoryAwareSegmentCaptioner,
)
from surrogate_rollout.prompt_routing.free_form_instruction_generator import (
    FREE_FORM_PROMPT_ID,
    FREE_FORM_REQUEST_SCHEMA_VERSION,
    FREE_FORM_ROUTING_MODE,
    FreeFormGenerationError,
    GENERATOR_TEMPLATES,
    VLMFreeFormInstructionGenerator,
    build_free_form_selection,
    free_form_router_version,
)
from surrogate_rollout.prompt_routing.free_form_instruction_parser import (
    PARSER_VERSION,
    parse_generated_instruction,
)
from surrogate_rollout.prompt_routing.persistence import (
    prompt_bank_from_json,
    router_policy_from_json,
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.policies.history_aware_vlm_router import (
    HistoryAwareVLMRouter,
)
from surrogate_rollout.prompt_routing.scaffold_applier import (
    create_scaffold_applier,
)
from surrogate_rollout.prompt_routing.schemas import SegmentContext
from surrogate_rollout.schemas import sha256_text

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
#                            plain-text parser                                 #
# --------------------------------------------------------------------------- #
def test_parser_plain_text():
    outcome = parse_generated_instruction(
        "Describe who holds each object.")
    assert outcome.ok
    assert outcome.parser_path == "plain_text"
    assert outcome.instruction == "Describe who holds each object."


def test_parser_json_looking_text_is_not_interpreted():
    outcome = parse_generated_instruction(
        '{"caption_instruction": "Track the red car."}')
    assert outcome.ok
    assert outcome.parser_path == "plain_text"
    assert outcome.instruction == \
        '{"caption_instruction": "Track the red car."}'


def test_parser_markdown_fence_is_not_interpreted():
    outcome = parse_generated_instruction(
        "```\nTrack the red car.\n```")
    assert outcome.ok
    assert outcome.parser_path == "plain_text"
    assert outcome.instruction == "```\nTrack the red car.\n```"


@pytest.mark.parametrize("raw", [None, "", "   \n  "])
def test_parser_empty_is_explicit_failure(raw):
    outcome = parse_generated_instruction(raw)
    assert not outcome.ok
    assert outcome.parser_path == "empty"
    assert outcome.instruction is None
    assert outcome.error


# --------------------------------------------------------------------------- #
#                             fixtures / fakes                                 #
# --------------------------------------------------------------------------- #
def components():
    data = json.loads((
        ROOT / "prompt_routing/fixtures/stage4_7_components.json").read_text())
    bank = prompt_bank_from_json(data["prompt_bank"])
    router = replace(
        router_policy_from_json(data["router_policy"]),
        policy_type="history_aware_vlm",
        configuration={"max_selected_properties": 3, "max_tokens": 64},
    )
    return (
        bank,
        router,
        scaffold_policy_from_json(data["scaffold_policy"]),
        scaffold_contract_from_json(data["scaffold_contract"]),
    )


def context(**overrides):
    values = dict(
        video_id="video-1", segment_id="0_10",
        timestamp_start=0.0, timestamp_end=10.0,
        segment_features={"frame_references": ["frame-0.jpg"]},
        history_summary=json.dumps({"preceding_captions": []}),
        question="SECRET QUESTION MUST NOT LEAK",
        metadata={"answer": "SECRET ANSWER", "used_segments": ["10_20"]},
    )
    values.update(overrides)
    return SegmentContext(**values)


class GeneratorVLM:
    """Fake backend returning a fixed generator reply; records every call."""

    def __init__(self, output="Track object handoffs."):
        self.output = output
        self.calls = []

    def caption(self, images, prompt, **kwargs):
        self.calls.append({"images": tuple(images), "prompt": prompt,
                           "kwargs": kwargs})
        return self.output


class FreeFormMockVLM:
    """One shared backend serving generator AND caption calls (integration)."""

    def __init__(self):
        self.calls = []
        self.caption_number = 0

    def caption(self, images, prompt, **kwargs):
        kind = ("generator" if FREE_FORM_REQUEST_SCHEMA_VERSION in prompt
                else "caption")
        self.calls.append({"kind": kind, "images": tuple(images),
                           "prompt": prompt, "kwargs": kwargs})
        if kind == "generator":
            return f"Generated instruction {len(self.calls)}"
        self.caption_number += 1
        return f"Generated caption {self.caption_number}."


# --------------------------------------------------------------------------- #
#                        generation and QA-leak guard                          #
# --------------------------------------------------------------------------- #
def test_generate_builds_query_independent_request(monkeypatch):
    monkeypatch.setattr(config, "USE_TRANSCRIPT", True)
    vlm = GeneratorVLM()
    generator = VLMFreeFormInstructionGenerator(vlm, backend_id="test-backend")
    generated = generator.generate(context(), {"transcript": "hello there"})

    assert generated.instruction == "Track object handoffs."
    assert generated.parser_path == "plain_text"
    prompt = vlm.calls[0]["prompt"]
    assert "SECRET QUESTION" not in prompt
    assert "SECRET ANSWER" not in prompt
    assert "used_segments" not in prompt
    # Identical context to the router: transcript is never fed to the generator,
    # even when the clip carries one and the global policy is enabled.
    assert "segment_transcript" not in prompt
    assert "hello there" not in prompt
    assert vlm.calls[0]["images"] == ("frame-0.jpg",)
    assert vlm.calls[0]["kwargs"]["max_tokens"] == generator.max_tokens
    assert generator.last_exchange is not None
    assert generator.last_exchange.request_hash == generated.request_hash


def test_generator_uses_versioned_plain_text_contract():
    vlm = GeneratorVLM()
    generator = VLMFreeFormInstructionGenerator(vlm, backend_id="test-backend")
    generated = generator.generate(context(), {})

    assert generated.parser_path == "plain_text"
    assert FREE_FORM_REQUEST_SCHEMA_VERSION == \
        "free_form_instruction_request_v2_plain_text"
    assert PARSER_VERSION == "free_form_instruction_plain_text_parser_v2"
    assert generator.template_version == "v2_plain_text"
    assert "plain text" in generator.template
    assert "strict JSON" not in generator.template
    assert generator.configuration_identity["parser_version"] == PARSER_VERSION


def test_generate_raises_on_unusable_output():
    generator = VLMFreeFormInstructionGenerator(GeneratorVLM(output="   "))
    with pytest.raises(FreeFormGenerationError, match="unusable"):
        generator.generate(context(), {})


# --------------------------------------------------------------------------- #
#                       request identity sensitivity                           #
# --------------------------------------------------------------------------- #
def _request_hash(monkeypatch, *, ctx=None, template_version="v2_plain_text"):
    generator = VLMFreeFormInstructionGenerator(
        GeneratorVLM(), template_version=template_version)
    generated = generator.generate(ctx or context(), {})
    return generated.request_hash


def test_changing_frames_changes_request_identity(monkeypatch):
    base = _request_hash(monkeypatch)
    changed = _request_hash(monkeypatch, ctx=context(
        segment_features={"frame_references": ["frame-OTHER.jpg"]}))
    assert base != changed


def test_changing_history_changes_request_identity(monkeypatch):
    base = _request_hash(monkeypatch)
    changed = _request_hash(monkeypatch, ctx=context(
        history_summary=json.dumps({"preceding_captions": [
            {"segment_id": "0_10", "caption": "earlier"}]})))
    assert base != changed


def test_transcript_never_enters_request_identity(monkeypatch):
    """The generator ignores the clip transcript entirely (router parity), so
    two different transcripts yield the SAME request identity."""
    monkeypatch.setattr(config, "USE_TRANSCRIPT", True)

    def hash_for(transcript):
        generator = VLMFreeFormInstructionGenerator(GeneratorVLM())
        return generator.generate(context(), {"transcript": transcript}).request_hash

    assert hash_for("hello") == hash_for("goodbye")


def test_changing_template_version_changes_identity(monkeypatch):
    templates = dict(GENERATOR_TEMPLATES)
    templates["v2-test"] = templates["v2_plain_text"] + "\nEXTRA RULE."
    monkeypatch.setattr(
        "surrogate_rollout.prompt_routing.free_form_instruction_generator."
        "GENERATOR_TEMPLATES", templates)
    base = _request_hash(monkeypatch)
    changed = _request_hash(monkeypatch, template_version="v2-test")
    assert base != changed
    g1 = VLMFreeFormInstructionGenerator(GeneratorVLM())
    g2 = VLMFreeFormInstructionGenerator(
        GeneratorVLM(), template_version="v2-test")
    assert g1.identity_hash != g2.identity_hash
    assert free_form_router_version(g1.identity_hash) != \
        free_form_router_version(g2.identity_hash)


# --------------------------------------------------------------------------- #
#                       cache identity separation                              #
# --------------------------------------------------------------------------- #
def _cache_key(tmp_path, *, router_version, composed_text):
    video = tmp_path / "video.mp4"
    if not video.exists():
        video.write_bytes(b"x")
    return build_history_aware_cache_key(
        video_id="video-1", video_path=str(video),
        caption_prompt=composed_text, merge_prompt="merge",
        subtitle_path=None, segment_id="0_10",
        history_hash=sha256_text("history"),
        composed_prompt_hash=sha256_text(composed_text),
        bank_version="bank_v0002", router_version=router_version,
        scaffold_version="scaffold_v0002", contract_version="contract_v0002",
        backend_id="test-backend", history_config_hash=sha256_text("hcfg"),
    )


def test_free_form_cache_keys_do_not_collide_with_property_bank(tmp_path):
    """Even with IDENTICAL composed text, the free-form reserved router-version
    namespace keeps cache identity and directory distinct."""
    generator = VLMFreeFormInstructionGenerator(GeneratorVLM())
    ff_version = free_form_router_version(generator.identity_hash)
    text = "BASE Additional routed caption instructions:\n- same wording"
    property_key = _cache_key(tmp_path, router_version="router_v9002",
                              composed_text=text)
    free_form_key = _cache_key(tmp_path, router_version=ff_version,
                               composed_text=text)
    assert property_key != free_form_key
    assert new_history_aware_cache_dir(property_key, str(tmp_path)) != \
        new_history_aware_cache_dir(free_form_key, str(tmp_path))


def test_changing_generated_instruction_changes_caption_cache_identity(tmp_path):
    generator = VLMFreeFormInstructionGenerator(GeneratorVLM())
    version = free_form_router_version(generator.identity_hash)
    key_a = _cache_key(tmp_path, router_version=version,
                       composed_text="BASE\n- instruction A")
    key_b = _cache_key(tmp_path, router_version=version,
                       composed_text="BASE\n- instruction B")
    assert key_a.composed_prompt_hash != key_b.composed_prompt_hash
    assert new_history_aware_cache_dir(key_a, str(tmp_path)) != \
        new_history_aware_cache_dir(key_b, str(tmp_path))


# --------------------------------------------------------------------------- #
#                    synthetic selection for the scaffold                      #
# --------------------------------------------------------------------------- #
def test_build_free_form_selection_composes_through_shared_scaffold(monkeypatch):
    monkeypatch.setattr(config, "USE_TRANSCRIPT", True)
    bank, _, scaffold_policy, contract = components()
    generator = VLMFreeFormInstructionGenerator(GeneratorVLM())
    ctx = context()
    decision, entries, exchange = build_free_form_selection(
        generator=generator, context=ctx,
        clip_info={"transcript": "hello"}, prompt_bank=bank)

    assert decision.selected_prompt_ids == (FREE_FORM_PROMPT_ID,)
    assert decision.router_version == \
        free_form_router_version(generator.identity_hash)
    assert decision.bank_version == bank.bank_version
    payload = dict(decision.decision_payload)
    assert payload["mode"] == FREE_FORM_ROUTING_MODE
    assert payload["parser_path"] == "plain_text"
    assert payload["generated_instruction"] == "Track object handoffs."
    assert payload["generated_instruction_hash"] == \
        sha256_text("Track object handoffs.")
    assert payload["raw_generator_response"]
    assert "transcript_used" not in payload
    assert exchange is not None and exchange.raw_output

    base = ("BASE TRANSCRIPT_PLACEHOLDER CLIP_START_TIME CLIP_END_TIME\n"
            "ROUTED_INSTRUCTIONS_PLACEHOLDER")
    composed = create_scaffold_applier(
        scaffold_policy, base_prompt_template=base).apply(
            context=ctx, selected_entries=entries, routing_decision=decision,
            scaffold_policy=scaffold_policy, scaffold_contract=contract)
    assert composed.is_valid
    assert "Track object handoffs." in composed.prompt_text
    assert composed.router_version == decision.router_version


# --------------------------------------------------------------------------- #
#                    mocked end-to-end builder integration                     #
# --------------------------------------------------------------------------- #
def _build(builder, tmp_path, bank, policy, scaffold, contract):
    clips = [
        ("0_10", {"files": [tmp_path / "f0.jpg"], "transcript": "zero"}),
        ("10_20", {"files": [tmp_path / "f1.jpg"], "transcript": "one"}),
    ]
    sample = {
        "sample_id": "sample",
        "video_path": str(tmp_path / "video.mp4"),
        "extra": {"videoID": "video-1", "subtitle_path": None},
    }
    return builder.build(
        sample=sample, clip_index=clips, prompt_bank=bank,
        router_policy=policy, scaffold_policy=scaffold,
        scaffold_contract=contract,
        base_prompt_template=(
            "TRANSCRIPT_PLACEHOLDER CLIP_START_TIME CLIP_END_TIME "
            "ROUTED_INSTRUCTIONS_PLACEHOLDER"),
        merge_prompt="merge", work_root=str(tmp_path / "work"),
        history_block_seconds=300, max_history_captions=30,
        candidate_cache_root=str(tmp_path / "cache"),
        cache_manifest_path=str(tmp_path / "cache_manifest.jsonl"),
    )


def test_free_form_mode_end_to_end_mocked(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USE_TRANSCRIPT", True)
    bank, policy, scaffold, contract = components()
    vlm = FreeFormMockVLM()
    builder = HistoryAwareBaselineCaptionViewBuilder(
        router=HistoryAwareVLMRouter(vlm),
        segment_captioner=HistoryAwareSegmentCaptioner(
            vlm, backend_id="test-backend"),
        free_form_generator=VLMFreeFormInstructionGenerator(
            vlm, backend_id="test-backend"),
        merge_fn=lambda values: {},
    )
    artifact = _build(builder, tmp_path, bank, policy, scaffold, contract)

    # generator + captioner alternate; the router itself is never called
    assert [call["kind"] for call in vlm.calls] == \
        ["generator", "caption", "generator", "caption"]
    assert artifact.caption_call_count == 2
    assert artifact.router_call_count == 2  # generator counted as routing stage

    # generated instruction appears verbatim in the final caption prompt
    generated_1 = "Generated instruction 1"
    assert generated_1 in vlm.calls[1]["prompt"]

    # sidecar decision records carry the full audit trail
    decisions = [json.loads(line) for line in Path(
        json.loads(Path(artifact.routing_manifest_path).read_text())
        ["routing_decisions_path"]).read_text().splitlines()]
    for record in decisions:
        payload = record["decision_payload"]
        assert payload["mode"] == FREE_FORM_ROUTING_MODE
        assert payload["parser_path"] == "plain_text"
        assert payload["raw_generator_response"]
        assert payload["generated_instruction_hash"] == \
            sha256_text(payload["generated_instruction"])
        assert record["router_version"].startswith("router_v8888_")

    # caption cache keys live in the reserved free-form namespace
    cache_rows = [json.loads(line) for line in Path(
        json.loads(Path(artifact.routing_manifest_path).read_text())
        ["caption_cache_keys_path"]).read_text().splitlines()]
    for row in cache_rows:
        assert row["cache_key"]["router_version"].startswith("router_v8888_")

    # resume: a second identical build performs zero model calls
    before = len(vlm.calls)
    resumed = _build(builder, tmp_path, bank, policy, scaffold, contract)
    assert len(vlm.calls) == before
    assert resumed.resumed_segment_count == 2


def test_property_bank_mode_still_routes_through_router(tmp_path):
    """Default mode: no generator constructed, router drives selection —
    guards against the free-form branch changing existing behavior."""
    bank, policy, scaffold, contract = components()

    class PropertyMockVLM:
        def __init__(self):
            self.calls = []

        def caption(self, images, prompt, **kwargs):
            kind = ("router" if "history_aware_vlm_router_request_v1" in prompt
                    else "caption")
            self.calls.append({"kind": kind})
            if kind == "router":
                return json.dumps({"property_ids": ["pe_temporal"]})
            return "A concise clip caption."

    vlm = PropertyMockVLM()
    builder = HistoryAwareBaselineCaptionViewBuilder(
        router=HistoryAwareVLMRouter(vlm),
        segment_captioner=HistoryAwareSegmentCaptioner(
            vlm, backend_id="test-backend"),
        merge_fn=lambda values: {},
    )
    assert builder.free_form_generator is None
    artifact = _build(builder, tmp_path, bank, policy, scaffold, contract)
    assert [call["kind"] for call in vlm.calls] == \
        ["router", "caption", "router", "caption"]
    decisions = [json.loads(line) for line in Path(
        json.loads(Path(artifact.routing_manifest_path).read_text())
        ["routing_decisions_path"]).read_text().splitlines()]
    for record in decisions:
        assert record["selected_prompt_ids"] == ["pe_temporal"]
        assert "mode" not in record["decision_payload"]
