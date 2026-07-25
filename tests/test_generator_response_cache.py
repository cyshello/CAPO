"""Generator response reuse: what it may return, and what it must not.

The cache exists so a resumed run regenerates a byte-identical instruction and
the caption cache below it still resolves. The dangerous failure is the mirror
image — one meta prompt reading another's entry — so most of these tests are
about entries staying apart.
"""
from __future__ import annotations

import json

import pytest

from surrogate_rollout import config
from surrogate_rollout.prompt_routing import generator_response_cache as grc
from surrogate_rollout.prompt_routing import static_meta_replace_body as smrb
from surrogate_rollout.prompt_routing.policies.openai_free_form_generator import (
    OpenAIFreeFormInstructionGenerator,
)
from surrogate_rollout.prompt_routing.schemas import SegmentContext


def _key(**over):
    base = dict(request_hash="r" * 64, base_url="https://api.openai.com/v1",
                model_id="gpt-5-mini", backend_id="backend_v1", max_tokens=4096,
                temperature=0.0, reasoning_effort="minimal")
    base.update(over)
    return grc.GeneratorResponseCacheKey(**base)


def _history_json():
    return json.dumps({
        "schema_version": smrb.HISTORY_SCHEMA_VERSION,
        "block_index": 0,
        "block_start_seconds": 0.0,
        "block_end_seconds": 300.0,
        "max_history_captions": 30,
        "preceding_captions": [{"segment_id": "20_30", "caption": "prev"}],
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _context(video_id="v1", segment_id="30_40"):
    return SegmentContext(
        video_id=video_id, segment_id=segment_id, timestamp_start=30.0,
        timestamp_end=40.0,
        segment_features={"frame_references": ("/f/a.jpg",)},
        history_summary=_history_json(), metadata={})


def _generator(monkeypatch, *, template_text="CURRENT PARENT",
               meta_prompt_id="parent-v2", cache_root=None, model_id="gpt-5-mini",
               answers=("First answer.", "Second answer.")):
    """A generator whose provider call is counted and never actually made."""
    generator = OpenAIFreeFormInstructionGenerator(
        api_key="sk-test", model_id=model_id, max_tokens=4096,
        template_text=template_text, meta_prompt_id=meta_prompt_id,
        backend_id="backend_v1", response_cache_root=cache_root)
    monkeypatch.setattr(
        smrb, "prepare_generator_frames",
        lambda files, caption_fps: ["/f/a.small.jpg"])
    calls = []

    def complete(frames, prompt):
        calls.append(prompt)
        return answers[min(len(calls) - 1, len(answers) - 1)]

    monkeypatch.setattr(generator, "_complete", complete)
    return generator, calls


# ------------------------------ key identity ------------------------------- #
def test_key_separates_every_dimension_that_decides_the_answer():
    base = _key()
    assert grc.key_hash(base) == grc.key_hash(_key())
    for changed in (_key(request_hash="q" * 64), _key(model_id="gpt-4o-mini"),
                    _key(base_url="https://example.invalid/v1"),
                    _key(backend_id="other"), _key(max_tokens=512),
                    _key(temperature=1.0), _key(reasoning_effort=None)):
        assert grc.key_hash(changed) != grc.key_hash(base)


def test_store_then_load_round_trips(tmp_path):
    root = str(tmp_path)
    grc.store(root, video_id="v1", segment_id="30_40", key=_key(),
              raw_response="Describe the hands.")
    assert grc.load(root, video_id="v1", segment_id="30_40",
                    key=_key()) == "Describe the hands."


def test_a_different_request_never_reads_another_entry(tmp_path):
    root = str(tmp_path)
    grc.store(root, video_id="v1", segment_id="30_40", key=_key(),
              raw_response="Describe the hands.")
    assert grc.load(root, video_id="v1", segment_id="30_40",
                    key=_key(request_hash="q" * 64)) is None
    assert grc.load(root, video_id="v2", segment_id="30_40",
                    key=_key()) is None
    assert grc.load(root, video_id="v1", segment_id="40_50",
                    key=_key()) is None


def test_entries_are_write_once(tmp_path):
    root = str(tmp_path)
    grc.store(root, video_id="v1", segment_id="30_40", key=_key(),
              raw_response="first")
    grc.store(root, video_id="v1", segment_id="30_40", key=_key(),
              raw_response="second")
    assert grc.load(root, video_id="v1", segment_id="30_40",
                    key=_key()) == "first"


def test_no_root_means_no_reuse_and_no_writes(tmp_path):
    assert grc.load(None, video_id="v1", segment_id="30_40", key=_key()) is None
    assert grc.store(None, video_id="v1", segment_id="30_40", key=_key(),
                     raw_response="x") is None
    assert not list(tmp_path.iterdir())


def test_corrupt_entry_reads_as_a_miss(tmp_path):
    root = str(tmp_path)
    path = grc.entry_path(root, video_id="v1", segment_id="30_40", key=_key())
    (tmp_path / "generator_v1" / "v1" / "30_40").mkdir(parents=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{not json")
    assert grc.load(root, video_id="v1", segment_id="30_40", key=_key()) is None


# ---------------------------- generator behavior ---------------------------- #
def test_second_run_reuses_the_answer_instead_of_paying_again(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    first, first_calls = _generator(monkeypatch, cache_root=root)
    generated = first.generate(_context(), {})
    assert first_calls and generated.cache_hit is False

    # A restart: a new generator object, same request, same run cache root.
    second, second_calls = _generator(monkeypatch, cache_root=root)
    resumed = second.generate(_context(), {})
    assert second_calls == []
    assert resumed.cache_hit is True
    assert resumed.instruction == generated.instruction
    assert resumed.request_hash == generated.request_hash


def test_without_a_cache_root_the_provider_is_called_again(monkeypatch):
    """No root configured: the old behavior, one paid call per generate."""
    monkeypatch.delenv(config.GENERATOR_RESPONSE_CACHE_VARIABLE, raising=False)
    first, _ = _generator(monkeypatch, cache_root=None)
    first.generate(_context(), {})
    second, second_calls = _generator(monkeypatch, cache_root=None)
    resumed = second.generate(_context(), {})
    assert len(second_calls) == 1
    assert resumed.cache_hit is False


def test_a_different_meta_prompt_gets_its_own_entry(tmp_path, monkeypatch):
    """The one failure that would silently void the experiment."""
    root = str(tmp_path)
    parent, _ = _generator(monkeypatch, cache_root=root,
                           template_text="PARENT META PROMPT",
                           answers=("Parent instruction.",))
    parent_generated = parent.generate(_context(), {})

    candidate, candidate_calls = _generator(
        monkeypatch, cache_root=root, template_text="CANDIDATE META PROMPT",
        meta_prompt_id="candidate-v3", answers=("Candidate instruction.",))
    candidate_generated = candidate.generate(_context(), {})

    assert len(candidate_calls) == 1
    assert candidate_generated.cache_hit is False
    assert candidate_generated.instruction != parent_generated.instruction


def test_reasoning_effort_is_part_of_the_entry(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(config, "GENERATOR_REASONING_EFFORT", "minimal")
    low, _ = _generator(monkeypatch, cache_root=root, answers=("Low.",))
    low.generate(_context(), {})

    monkeypatch.setattr(config, "GENERATOR_REASONING_EFFORT", "medium")
    high, high_calls = _generator(monkeypatch, cache_root=root,
                                  answers=("High.",))
    generated = high.generate(_context(), {})
    assert len(high_calls) == 1
    assert generated.instruction == "High."


# ------------------------------- run wiring -------------------------------- #
def test_run_cache_root_configures_reuse_unless_the_operator_set_it(
        tmp_path, monkeypatch):
    monkeypatch.delenv(config.GENERATOR_RESPONSE_CACHE_VARIABLE, raising=False)
    assert grc.configure_default_root(str(tmp_path)) == str(tmp_path)
    assert grc.configure_default_root("/some/other/root") == str(tmp_path)

    monkeypatch.setenv(config.GENERATOR_RESPONSE_CACHE_VARIABLE, "")
    assert grc.configure_default_root(str(tmp_path)) is None


def test_a_run_without_a_cache_root_leaves_reuse_off(monkeypatch):
    monkeypatch.delenv(config.GENERATOR_RESPONSE_CACHE_VARIABLE, raising=False)
    assert grc.configure_default_root(None) is None


if __name__ == "__main__":  # pragma: no cover - convenience
    raise SystemExit(pytest.main([__file__]))
