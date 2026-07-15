"""Stage 4.6 routed caption-view adapter tests.

Captioning and registry merging are synthetic.  No model, DVD reasoning,
vector database, evaluation, or optimization call occurs in this module.
"""

import json
from pathlib import Path

import pytest

from surrogate_rollout.cache.caption_cache import prompt_hash as cache_prompt_hash
from surrogate_rollout.captioning.candidate_captions import CandidateCaptionSet
from surrogate_rollout.prompt_routing.routed_caption_view import (
    ROUTED_PROMPT_VIEW,
    RoutedCaptionViewBuilder,
    RoutedCaptionViewError,
)
from surrogate_rollout.prompt_routing.schemas import (
    ComposedCaptionPrompt,
    CompositionTrace,
)
from surrogate_rollout.schemas import sha256_text

CLIPS = ("0_10", "10_20", "20_30", "30_40")
VIDEO_ID = "video_1"
MERGE_PROMPT = "merge prompt"
BASE = (
    "Clip: CLIP_START_TIME to CLIP_END_TIME\n"
    "Transcript: TRANSCRIPT_PLACEHOLDER\n"
)


def composed(segment_id, specialization="temporal", *, valid=True, text=None):
    prompt_text = text if text is not None else \
        BASE + f"Instruction: {specialization}."
    selected = (f"pe_{specialization}",)
    trace = CompositionTrace(
        selected_prompt_ids=selected,
        preserved_prompt_ids=selected,
        ordering_decisions=("router_selection_order preserved",),
    )
    errors = () if valid else ("synthetic contract failure",)
    return ComposedCaptionPrompt(
        video_id=VIDEO_ID,
        segment_id=segment_id,
        bank_version="bank_v0001",
        router_version="router_v0001",
        scaffold_version="scaffold_v0001",
        contract_version="contract_v0001",
        selected_prompt_ids=selected,
        prompt_text=prompt_text,
        prompt_hash=sha256_text(prompt_text),
        composition_trace=trace,
        validation_errors=errors,
        is_valid=valid,
    )


class FakeCaptioner:
    def __init__(self, clips=CLIPS):
        self.clips = tuple(clips)
        self.calls = []

    def __call__(self, *, sample, candidate_prompt, merge_prompt, clip_ids,
                 cache_root):
        ordered_requested = tuple(k for k in self.clips if k in clip_ids)
        self.calls.append({
            "prompt": candidate_prompt,
            "clip_ids": set(clip_ids),
            "cache_root": cache_root,
        })
        strong_hash = cache_prompt_hash(candidate_prompt, merge_prompt)
        # Reverse insertion order deliberately; the builder must restore DVD
        # temporal order rather than inheriting group/dict order.
        parsed = {
            key: {
                "clip_description": f"caption {key}",
                "subject_registry": {"segment_id": key},
            }
            for key in reversed(ordered_requested)
        }
        return CandidateCaptionSet(
            video_id=VIDEO_ID,
            prompt_hash=strong_hash,
            ckpt_dir=f"/candidate_cache/{strong_hash[:12]}/ckpt",
            clips=list(self.clips),
            parsed=parsed,
            caption_call_count=len(clip_ids),
            cache_hits=0,
            caption_seconds=0.25,
            cache_key={
                "video_id": VIDEO_ID,
                "segment_id": "*",
                "prompt_hash": strong_hash,
                "caption_model_id": "synthetic",
                "decoding_hash": "d" * 64,
                "source_hash": "s" * 64,
            },
        )


def clip_index(sample, video_id):
    assert video_id == VIDEO_ID
    return [(key, {}) for key in CLIPS]


class MergeRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, registries):
        self.calls.append(list(registries))
        return {"ordered_segments": [r["segment_id"] for r in registries]}


def build(tmp_path, prompt_map, *, captioner=None, merger=None,
          candidate_cache_root=None):
    captioner = captioner or FakeCaptioner()
    merger = merger or MergeRecorder()
    artifact = RoutedCaptionViewBuilder(
        caption_fn=captioner,
        clip_index_fn=clip_index,
        merge_fn=merger,
    ).build(
        sample={"sample_id": VIDEO_ID, "video_path": "unused.mp4"},
        segment_composed_prompt_map=prompt_map,
        work_root=str(tmp_path / "work"),
        merge_prompt=MERGE_PROMPT,
        candidate_cache_root=(str(candidate_cache_root)
                              if candidate_cache_root else None),
    )
    return artifact, captioner, merger


def same_prompt_map():
    return {key: composed(key, "temporal") for key in CLIPS}


def test_all_same_prompt_matches_existing_single_prompt_behavior(tmp_path):
    artifact, captioner, merger = build(tmp_path, same_prompt_map())
    assert len(captioner.calls) == 1
    assert captioner.calls[0]["clip_ids"] == set(CLIPS)
    captions = json.loads(Path(artifact.captions_path).read_text())
    assert captions == {
        **{key: {"caption": f"caption {key}"} for key in CLIPS},
        "subject_registry": {"ordered_segments": list(CLIPS)},
    }
    assert len(merger.calls) == 1


def test_multiple_prompt_hashes_are_grouped(tmp_path):
    prompt_map = {
        key: composed(key, "text" if key in {"10_20", "30_40"}
                      else "temporal")
        for key in CLIPS
    }
    artifact, captioner, _ = build(tmp_path, prompt_map)
    assert len(captioner.calls) == 2
    assert {frozenset(call["clip_ids"]) for call in captioner.calls} == {
        frozenset({"0_10", "20_30"}),
        frozenset({"10_20", "30_40"}),
    }
    assert [group.segment_ids for group in artifact.prompt_groups] == [
        ("0_10", "20_30"), ("10_20", "30_40")]


def test_prompt_hash_cache_separation_and_component_provenance(tmp_path):
    prompt_map = {
        key: composed(key, "text" if key == "10_20" else "temporal")
        for key in CLIPS
    }
    artifact, _, _ = build(tmp_path, prompt_map)
    cache_hashes = {g.caption_cache_prompt_hash for g in artifact.prompt_groups}
    composed_hashes = {g.composed_prompt_hash for g in artifact.prompt_groups}
    assert len(cache_hashes) == len(composed_hashes) == 2
    assert all(len(h) == 64 for h in cache_hashes)
    assert [s.segment_id for s in artifact.segments] == list(CLIPS)
    assert all(s.bank_version == "bank_v0001" for s in artifact.segments)
    sidecar = json.loads(Path(artifact.routed_view_path).read_text())
    assert sidecar["segments"][1]["scaffold_version"] == "scaffold_v0001"
    assert sidecar["segments"][1]["composition_trace"][
        "preserved_prompt_ids"] == ["pe_text"]


def test_segment_and_registry_order_restored_before_one_whole_video_merge(tmp_path):
    # Reversed map order and interleaved prompt groups must not affect output.
    prompt_map = {
        key: composed(key, "text" if i % 2 else "temporal")
        for i, key in reversed(list(enumerate(CLIPS)))
    }
    artifact, _, merger = build(tmp_path, prompt_map)
    captions = json.loads(Path(artifact.captions_path).read_text(),
                          object_pairs_hook=dict)
    assert list(captions) == [*CLIPS, "subject_registry"]
    assert len(merger.calls) == 1
    assert [r["segment_id"] for r in merger.calls[0]] == list(CLIPS)


def test_candidate_cache_isolated_and_incumbent_untouched(tmp_path):
    incumbent = tmp_path / "incumbent" / "captions.json"
    incumbent.parent.mkdir()
    incumbent.write_text('{"frozen": true}\n')
    before = incumbent.read_bytes()
    candidate_root = tmp_path / "candidate_cache"
    artifact, captioner, _ = build(
        tmp_path, same_prompt_map(), candidate_cache_root=candidate_root)
    assert incumbent.read_bytes() == before
    assert all(call["cache_root"] == str(candidate_root)
               for call in captioner.calls)
    assert not Path(artifact.captions_path).is_relative_to(incumbent.parent)


def test_overlapping_routed_view_and_candidate_cache_rejected_before_calls(tmp_path):
    captioner = FakeCaptioner()
    candidate_root = tmp_path / "shared_root"
    builder = RoutedCaptionViewBuilder(
        caption_fn=captioner, clip_index_fn=clip_index,
        merge_fn=MergeRecorder())
    with pytest.raises(RoutedCaptionViewError, match="isolated"):
        builder.build(
            sample={"sample_id": VIDEO_ID, "video_path": "unused.mp4"},
            segment_composed_prompt_map=same_prompt_map(),
            work_root=str(candidate_root),
            merge_prompt=MERGE_PROMPT,
            candidate_cache_root=str(candidate_root),
        )
    assert captioner.calls == []


def test_routed_view_never_labeled_as_full_single_prompt_cache(tmp_path):
    artifact, _, _ = build(tmp_path, same_prompt_map())
    assert artifact.artifact_type == ROUTED_PROMPT_VIEW
    assert "captions_routed_" in artifact.captions_path
    assert "captions_full" not in artifact.captions_path
    sidecar = json.loads(Path(artifact.routed_view_path).read_text())
    assert sidecar["artifact_type"] == "routed_prompt_view"


def test_missing_composed_prompt_rejected_before_captioning(tmp_path):
    captioner = FakeCaptioner()
    prompt_map = same_prompt_map()
    prompt_map.pop("20_30")
    with pytest.raises(RoutedCaptionViewError, match="missing=.*20_30"):
        build(tmp_path, prompt_map, captioner=captioner)
    assert captioner.calls == []


@pytest.mark.parametrize("bad_prompt", [
    composed("10_20", valid=False),
    composed("10_20", text="No required placeholders here."),
])
def test_invalid_prompt_contract_rejected_before_captioning(tmp_path, bad_prompt):
    captioner = FakeCaptioner()
    prompt_map = same_prompt_map()
    prompt_map["10_20"] = bad_prompt
    with pytest.raises(RoutedCaptionViewError, match="invalid|contract"):
        build(tmp_path, prompt_map, captioner=captioner)
    assert captioner.calls == []
