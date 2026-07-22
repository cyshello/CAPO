"""Question -> atomic visual retrieval queries (PHASE2_3 §7).

One text-only LLM call per (question, options), JSON-schema output, cached by
question/options/model/prompt-version hash so repeat runs are deterministic and
free. The full payload (inputs, raw response, parsed queries) is preserved in
the cache file (CLAUDE.md §22).
"""

from __future__ import annotations

import json
import os
from typing import Callable, Sequence

from surrogate_rollout import config
from surrogate_rollout.schemas import sha256_json

QUESTION_QUERY_PROMPT = """You convert a video question into short visual search queries for a CLIP-style image-text retriever.

Question:
{question}

Answer options:
{options}

Rules:
- Output ONLY a JSON object: {{"queries": ["...", ...]}}
- 4 to {max_queries} queries.
- Each query names something visually retrievable from single frames: concrete entities, attributes, colors, actions, on-screen text.
- No reasoning words, no question words, no "the video".
"""


class QueryGenerationError(RuntimeError):
    pass


def _default_llm_fn(prompt: str) -> str:
    import sys

    if config.PROMPT_SENS_ROOT not in sys.path:
        sys.path.insert(0, config.PROMPT_SENS_ROOT)
    from codex_infer import codex_infer

    return codex_infer(prompt, model=config.QUERY_GENERATOR_MODEL)


def _extract_json_obj(text: str) -> dict:
    t = text.strip()
    start = t.find("{")
    if start < 0:
        raise QueryGenerationError(f"no JSON object in query-generator output: {text[:200]}")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise QueryGenerationError(f"unbalanced JSON in query-generator output: {text[:200]}")


def _cache_file(kind: str, key: dict, cache_root: str | None) -> str:
    root = cache_root or config.QUERY_CACHE_ROOT
    return os.path.join(root, kind, f"{sha256_json(key)[:24]}.json")


def _cached_or_generate(kind: str, key: dict, prompt: str,
                        llm_fn: Callable[[str], str] | None,
                        cache_root: str | None,
                        parse: Callable[[dict], dict]) -> dict:
    path = _cache_file(kind, key, cache_root)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["parsed"]
    fn = llm_fn or _default_llm_fn
    raw = fn(prompt)
    parsed = parse(_extract_json_obj(raw))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"key": key, "prompt": prompt, "raw_response": raw,
                   "parsed": parsed}, f, indent=2)
    return parsed


def _clean_queries(queries, max_queries: int) -> list[str]:
    out = [str(q).strip() for q in (queries or []) if str(q).strip()]
    if not out:
        raise QueryGenerationError("query generator returned no queries")
    return out[:max_queries]


def generate_question_queries(
    question: str,
    answer_options: Sequence[str],
    *,
    llm_fn: Callable[[str], str] | None = None,
    model: str = config.QUERY_GENERATOR_MODEL,
    cache_root: str | None = None,
    max_queries: int = config.MAX_RETRIEVAL_QUERIES,
) -> list[str]:
    key = {
        "question": question,
        "answer_options": list(answer_options),
        "model": model,
        "prompt_version": config.QUESTION_QUERY_PROMPT_VERSION,
    }
    prompt = QUESTION_QUERY_PROMPT.format(
        question=question,
        options="\n".join(answer_options) or "(none)",
        max_queries=max_queries,
    )

    def parse(obj: dict) -> dict:
        return {"queries": _clean_queries(obj.get("queries"), max_queries)}

    return _cached_or_generate("question_queries", key, prompt,
                               llm_fn, cache_root, parse)["queries"]
