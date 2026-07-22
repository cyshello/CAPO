import json

import pytest

from surrogate_rollout.retrieval.question_queries import (
    QueryGenerationError,
    generate_question_queries,
)

OPTIONS = ("A. red", "B. blue")


class CountingLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return self.response


def test_cache_hit_skips_llm(tmp_path):
    llm = CountingLLM(json.dumps({"queries": ["red light", "blue chair", "a", "b"]}))
    root = str(tmp_path)
    q1 = generate_question_queries("What color?", OPTIONS, llm_fn=llm, cache_root=root)
    q2 = generate_question_queries("What color?", OPTIONS, llm_fn=llm, cache_root=root)
    assert q1 == q2 == ["red light", "blue chair", "a", "b"]
    assert llm.calls == 1


def test_different_question_new_cache_entry(tmp_path):
    llm = CountingLLM('{"queries": ["x", "y", "z", "w"]}')
    root = str(tmp_path)
    generate_question_queries("q one?", OPTIONS, llm_fn=llm, cache_root=root)
    generate_question_queries("q two?", OPTIONS, llm_fn=llm, cache_root=root)
    assert llm.calls == 2


def test_different_options_new_cache_entry(tmp_path):
    llm = CountingLLM('{"queries": ["x", "y", "z", "w"]}')
    root = str(tmp_path)
    generate_question_queries("q?", OPTIONS, llm_fn=llm, cache_root=root)
    generate_question_queries("q?", ("A. up", "B. down"), llm_fn=llm, cache_root=root)
    assert llm.calls == 2


def test_max_queries_enforced(tmp_path):
    llm = CountingLLM(json.dumps({"queries": [f"q{i}" for i in range(20)]}))
    out = generate_question_queries("q?", OPTIONS, llm_fn=llm,
                                    cache_root=str(tmp_path), max_queries=8)
    assert len(out) == 8


def test_fenced_json_accepted(tmp_path):
    llm = CountingLLM('```json\n{"queries": ["a", "b", "c", "d"]}\n```')
    out = generate_question_queries("q?", OPTIONS, llm_fn=llm, cache_root=str(tmp_path))
    assert out == ["a", "b", "c", "d"]


def test_empty_queries_raise(tmp_path):
    llm = CountingLLM('{"queries": []}')
    with pytest.raises(QueryGenerationError):
        generate_question_queries("q?", OPTIONS, llm_fn=llm, cache_root=str(tmp_path))
