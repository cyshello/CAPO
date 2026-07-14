import json

from surrogate_rollout.retrieval.prompt_delta_queries import generate_prompt_delta_queries

OPTIONS = ("A. red", "B. blue")
RESPONSE = json.dumps({
    "newly_emphasized_concepts": ["colors"],
    "queries": ["red chair light", "small object detail", "a", "b"],
})


class CountingLLM:
    def __init__(self, response=RESPONSE):
        self.response = response
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return self.response


def test_cache_hit_and_payload(tmp_path):
    llm = CountingLLM()
    root = str(tmp_path)
    out1 = generate_prompt_delta_queries("q?", OPTIONS, "base prompt", "cand prompt",
                                         llm_fn=llm, cache_root=root)
    out2 = generate_prompt_delta_queries("q?", OPTIONS, "base prompt", "cand prompt",
                                         llm_fn=llm, cache_root=root)
    assert llm.calls == 1
    assert out1 == out2
    assert out1["newly_emphasized_concepts"] == ["colors"]
    assert len(out1["queries"]) == 4


def test_candidate_prompt_separates_cache_entries(tmp_path):
    llm = CountingLLM()
    root = str(tmp_path)
    generate_prompt_delta_queries("q?", OPTIONS, "base", "candidate A",
                                  llm_fn=llm, cache_root=root)
    generate_prompt_delta_queries("q?", OPTIONS, "base", "candidate B",
                                  llm_fn=llm, cache_root=root)
    assert llm.calls == 2


def test_baseline_prompt_separates_cache_entries(tmp_path):
    llm = CountingLLM()
    root = str(tmp_path)
    generate_prompt_delta_queries("q?", OPTIONS, "base A", "cand",
                                  llm_fn=llm, cache_root=root)
    generate_prompt_delta_queries("q?", OPTIONS, "base B", "cand",
                                  llm_fn=llm, cache_root=root)
    assert llm.calls == 2
