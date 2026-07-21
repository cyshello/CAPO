#!/usr/bin/env python
"""Run official GEPA optimization of the free-form DVD meta-prompt.

Uses the installed ``gepa`` engine (candidate selection, minibatch sampling,
Pareto tracking, budget) with :class:`DVDMetaPromptGEPAAdapter`. The single
optimizable component is the free-form caption-instruction generator template
(``meta_prompt``). Each training example is one video (three QAs), scored by
absolute QA accuracy. Nothing in the Phase 0-3 rollout stack or the existing
Phase 4 optimization modules is modified — the DVD runtime is reused via the
existing Checkpoint-G factory.

Run configuration (JSON, ``gepa_meta_prompt_run_config_v2``)::

    {
      "schema_version": "gepa_meta_prompt_run_config_v2",
      "seed_prompt_path": "train_set/seed_meta_prompt.txt",
      "output_dir": "runs/gepa_meta_prompt_<...>",
      "bootstrap_component_config_path": "cfg/checkpoint_g_component.json",
      "search": {"max_metric_calls": 300, "reflection_minibatch_size": 3,
                 "seed": 0},
      "reflection": {
        "provider": {"name": "openai_api", "api_endpoint": "...",
                     "api_key_environment_variable": "OPENAI_API_KEY",
                     "timeout_seconds": 120},
        "model_id": "gpt-...", "maximum_output_tokens": 2048,
        "generation_settings": {"temperature": 1.0}
      },
      "trainset": [
        {"video_id": "eqJPDJr_irE", "provider_indices": [0, 1, 2],
         "question_ids": ["eqJPDJr_irE/q0", "...", "..."]}
      ],
      "valset": [ ... optional; defaults to trainset ... ]
    }

``bootstrap_component_config_path`` is an existing Checkpoint-G component config
used only to construct the DVD runtime (captioner, embedder, cache, scaffold);
its ``confirmation_videos`` are irrelevant to this per-video search.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


def _load_json(path: str) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _required(mapping: Mapping[str, Any], name: str, where: str) -> Any:
    if name not in mapping or mapping[name] in (None, ""):
        raise ValueError(f"{where} requires {name!r}")
    return mapping[name]


def _instances(raw, where: str):
    from surrogate_rollout.gepa_meta_prompt.dvd_single_video_evaluator import (
        GepaVideoInstance,
    )
    result = []
    for item in raw:
        result.append(GepaVideoInstance(
            video_id=str(_required(item, "video_id", where)),
            provider_indices=tuple(
                int(v) for v in _required(item, "provider_indices", where)),
            question_ids=tuple(
                str(v) for v in _required(item, "question_ids", where))))
    if not result:
        raise ValueError(f"{where} must be non-empty")
    return result


def _bootstrap_dvd_runtime(component_config_path: str):
    """Construct the DVD evaluator and shared policies via the existing factory."""
    from surrogate_rollout.optimization.checkpoint_g_factory import (
        build_checkpoint_g_components,
    )

    _, _, confirmation = build_checkpoint_g_components(
        SimpleNamespace(component_config=component_config_path))
    # The lazy wrapper defers GPU/DVD construction; force it once here so we can
    # borrow the inner history-aware evaluator and the shared policies.
    dvd_meta = confirmation._factory()
    return {
        "evaluator": dvd_meta.evaluator,
        "bank": dvd_meta.bank,
        "router": dvd_meta.router,
        "scaffold": dvd_meta.scaffold,
        "contract": dvd_meta.contract,
        "generator_model_id": dvd_meta.prompt_generator_model_id,
        "generator_backend_id": dvd_meta.prompt_generator_backend_id,
        "generator_max_tokens": dvd_meta.prompt_generator_max_tokens,
    }


def _build_mutator(reflection_cfg, maximum_calls):
    from surrogate_rollout.optimization.checkpoint_g_factory import (
        _BoundedOpenAITransport,
    )
    from surrogate_rollout.gepa_meta_prompt.reflection import (
        OpenAICompatibleReflectionMutator,
    )

    provider = _required(reflection_cfg, "provider", "reflection")
    if str(_required(provider, "name", "reflection.provider")) != "openai_api":
        raise ValueError("only openai_api reflection provider is wired")
    api_key_name = str(_required(
        provider, "api_key_environment_variable", "reflection.provider"))
    api_key = os.environ.get(api_key_name, "")
    if not api_key:
        raise ValueError(f"{api_key_name} is not set")
    transport = _BoundedOpenAITransport(
        endpoint=str(_required(provider, "api_endpoint", "reflection.provider")),
        api_key=api_key,
        timeout_seconds=int(_required(
            provider, "timeout_seconds", "reflection.provider")),
        maximum_calls=maximum_calls)
    return OpenAICompatibleReflectionMutator(
        model_id=str(_required(reflection_cfg, "model_id", "reflection")),
        generation_settings=_required(
            reflection_cfg, "generation_settings", "reflection"),
        response_transport=transport.request,
        maximum_output_tokens=int(_required(
            reflection_cfg, "maximum_output_tokens", "reflection")))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True,
                        help="path to a gepa_meta_prompt_run_config_v2 JSON")
    args = parser.parse_args(argv)

    import gepa
    from surrogate_rollout.gepa_meta_prompt.gepa_adapter import (
        META_PROMPT_COMPONENT,
        DVDMetaPromptGEPAAdapter,
    )

    cfg = _load_json(args.run_config)
    if cfg.get("schema_version") != "gepa_meta_prompt_run_config_v2":
        raise ValueError("unsupported run-config schema_version")
    seed_text = Path(str(_required(
        cfg, "seed_prompt_path", "run config"))).read_text(encoding="utf-8")
    if not seed_text.strip():
        raise ValueError("seed meta-prompt is empty")
    output_dir = os.path.abspath(str(_required(cfg, "output_dir", "run config")))
    search = _required(cfg, "search", "run config")
    max_metric_calls = int(_required(search, "max_metric_calls", "search"))
    reflection_minibatch_size = int(_required(
        search, "reflection_minibatch_size", "search"))

    trainset = _instances(_required(cfg, "trainset", "run config"), "trainset")
    valset = (_instances(cfg["valset"], "valset")
              if cfg.get("valset") else trainset)

    runtime = _bootstrap_dvd_runtime(str(_required(
        cfg, "bootstrap_component_config_path", "run config")))
    mutator = _build_mutator(
        _required(cfg, "reflection", "run config"),
        # One reflection call per proposal round; bound generously by budget.
        maximum_calls=max_metric_calls)
    adapter = DVDMetaPromptGEPAAdapter(
        evaluator=runtime["evaluator"], bank=runtime["bank"],
        router=runtime["router"], scaffold=runtime["scaffold"],
        contract=runtime["contract"],
        generator_model_id=runtime["generator_model_id"],
        generator_backend_id=runtime["generator_backend_id"],
        generator_max_tokens=runtime["generator_max_tokens"],
        mutator=mutator,
        work_root=os.path.join(output_dir, "caption_states"),
        qa_cache_root=os.path.join(output_dir, "qa_cache"))

    result = gepa.optimize(
        seed_candidate={META_PROMPT_COMPONENT: seed_text},
        trainset=trainset, valset=valset, adapter=adapter,
        reflection_minibatch_size=reflection_minibatch_size,
        max_metric_calls=max_metric_calls,
        seed=int(search.get("seed", 0)),
        display_progress_bar=bool(search.get("display_progress_bar", False)),
        run_dir=os.path.join(output_dir, "gepa_run"))

    best_text = result.best_candidate[META_PROMPT_COMPONENT]
    os.makedirs(output_dir, exist_ok=True)
    Path(os.path.join(output_dir, "best_meta_prompt.txt")).write_text(
        best_text, encoding="utf-8")
    summary = {
        "best_meta_prompt_path": os.path.join(output_dir, "best_meta_prompt.txt"),
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "val_aggregate_subscores": result.val_aggregate_subscores,
        "best_idx": result.best_idx,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
