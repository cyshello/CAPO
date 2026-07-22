"""Reproduce the DVD tool_calls=None crash without any GPU or network.

dvd.utils.call_openai_model_with_tools returns {"content": ..., "tool_calls":
None} when the model answers in prose without calling a tool. DVDCoreAgent.run
then executes `for tool_call in response.get("tool_calls", [])` — the key
exists with value None, so .get returns None and iteration raises
`TypeError: 'NoneType' object is not iterable`. This matches the historical
"'NoneType' object is not iterable" failures in sd_out results.

The repro stubs the orchestrator with exactly that legal return shape and runs
the real DVDCoreAgent over a real cached video DB. No DVD code is modified;
the router stub mimics a prose-only model turn.

Finding (2026-07-14): NOT reproducible on the current tree — the vendored
dvd_core.py (working-tree modification dated 2026-07-13 23:46, not committed)
already guards with `response.get("tool_calls") or []` in both run and
stream_run. The historical sd_out failures predate that local patch. This
script stays as a regression check: it must keep printing "NOT reproduced".

Usage:
    conda run -n local_llm_vllm python -m surrogate_rollout.scripts.repro_tool_calls_none
"""

from __future__ import annotations

import os
import sys

from surrogate_rollout import config

for _p in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    import dvd.config as dvd_config
    import dvd.dvd_core as dvd_core
    import dvd.utils as utils
    from dvd.dvd_core import DVDCoreAgent

    # Local BGE embeddings so DB load needs no Azure (same swap dvd_backend does).
    from dvd_backend import _local_get_embeddings

    utils.AzureOpenAIEmbeddingService.get_embeddings = staticmethod(_local_get_embeddings)
    dvd_config.AOAI_EMBEDDING_LARGE_DIM = 384
    dvd_config.LITE_MODE = True  # no frame_inspect engine needed

    def prose_only_router(messages, endpoints, model_name, **kwargs):
        # Exact shape dvd.utils returns when the model does not call a tool.
        return {"content": "The answer is C.", "tool_calls": None}

    dvd_core.call_openai_model_with_tools = prose_only_router

    video = "0RxMZBLeqRI"
    workdir = os.path.join(config.DVD_RUN_WORKSPACE, video)
    agent = DVDCoreAgent(
        os.path.join(workdir, "database_tx_fps1_78d8e75e.json"),
        os.path.join(workdir, "captions_tx_fps1_78d8e75e", "captions.json"),
        max_iterations=2,
    )
    try:
        agent.run("dummy question?")
    except TypeError as e:
        print(f"REPRODUCED: TypeError: {e}")
        print("cause: dvd_core.run iterates response.get('tool_calls', []) "
              "but the key holds None")
        print("minimal fix (dvd/dvd/dvd_core.py, run + stream_run): "
              "for tool_call in (response.get('tool_calls') or []):")
        return
    print("NOT reproduced — agent tolerated tool_calls=None")


if __name__ == "__main__":
    main()
