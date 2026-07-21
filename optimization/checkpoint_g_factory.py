"""Explicit production component factory for the Checkpoint G pilot.

Every provider/runtime value comes from the operator-supplied JSON file.  This
module has no model, path, context-window, threshold, or worker defaults and
performs no retry, repair, fallback, or automatic request splitting.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from surrogate_rollout import config
from surrogate_rollout.captioning.history_aware_baseline import (
    HistoryAwareBaselineCaptionViewBuilder,
)
from surrogate_rollout.evaluation.dvd_qa import ensure_backend
from surrogate_rollout.optimization.confirmation_evaluator import (
    HistoryAwareDVDConfirmationEvaluator,
)
from surrogate_rollout.optimization.dvd_meta_prompt_confirmation import (
    DVDMetaPromptConfirmationEvaluator,
    DVDMetaPromptConfirmationVideo,
    LazyDVDMetaPromptConfirmationEvaluator,
)
from surrogate_rollout.optimization.llm_episode_feedback import (
    LLMEpisodeFeedbackGenerator,
    SavedEpisodeFeedbackArtifactResolver,
)
from surrogate_rollout.optimization.meta_prompt_update_execution import (
    OpenAICompatibleMetaPromptUpdaterBackend,
)
from surrogate_rollout.optimization.meta_prompt_updater import LLMMetaPromptUpdater
from surrogate_rollout.optimization.policies.episode_feedback_provider import (
    ExactProviderInputTokenCount,
    OpenAICompatibleEpisodeFeedbackProviderAdapter,
)
from surrogate_rollout.prompt_routing.persistence import (
    scaffold_contract_from_json,
    scaffold_policy_from_json,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    RouterPolicySnapshot,
    dumps_canonical,
)


class CheckpointGConfigurationError(ValueError):
    pass


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointGConfigurationError(f"{name} must be an object")
    return value


def _required(value: Mapping[str, Any], name: str) -> Any:
    if name not in value or value[name] is None or value[name] == "":
        raise CheckpointGConfigurationError(
            f"component config requires {name!r}")
    return value[name]


def _real_identity(value: Any, name: str) -> str:
    result = str(value)
    if not result or any(marker in result.lower()
                         for marker in ("mock", "fixture", "stub")):
        raise CheckpointGConfigurationError(
            f"{name} must identify a reviewed real backend/model")
    return result


class _BoundedOpenAITransport:
    """Explicit bounded attempts; no retry, repair, or fallback."""

    def __init__(self, *, endpoint: str, api_key: str, timeout_seconds: int,
                 maximum_calls: int):
        if not endpoint or not api_key or timeout_seconds <= 0 or \
                maximum_calls <= 0:
            raise CheckpointGConfigurationError(
                "endpoint, API key, and positive timeout are required")
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.maximum_calls = maximum_calls
        self.call_count = 0

    def request(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.call_count >= self.maximum_calls:
            raise RuntimeError("provider transport call budget exhausted")
        self.call_count += 1
        request = urllib.request.Request(
            self.endpoint, data=dumps_canonical(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI API returned HTTP {exc.code}: {detail}") from exc
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise TypeError("provider response envelope must be an object")
        return value

    def feedback(self, body: Mapping[str, Any]) -> str:
        value = self.request(body)
        try:
            result = value["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("provider response has no message content") from exc
        if not isinstance(result, str):
            raise TypeError("provider message content must be a string")
        return result


def _exact_counter(model_id: str):
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model_id)
    except Exception as exc:
        raise CheckpointGConfigurationError(
            f"exact tokenizer is unavailable for {model_id!r}: {exc}") from exc

    def count(messages):
        system = len(encoding.encode(messages[0]["content"]))
        user = len(encoding.encode(messages[1]["content"]))
        total = 3 + sum(3 + len(encoding.encode(item["role"])) +
                        len(encoding.encode(item["content"]))
                        for item in messages)
        return ExactProviderInputTokenCount(system, user, total)
    return count, encoding.name


def _load_configuration(path: str) -> Mapping[str, Any]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    result = _object(value, "component config")
    if result.get("schema_version") not in {
            "checkpoint_g_component_config_v1",
            "fresh_prompt_delta_component_config_v1"}:
        raise CheckpointGConfigurationError(
            "unsupported Checkpoint G component config schema")
    return result


def build_checkpoint_g_components(args):
    """Return the real feedback, updater, and DVD confirmation components."""
    component_path = getattr(args, "component_config", None)
    if not component_path:
        raise CheckpointGConfigurationError(
            "--component-config must be supplied to the real G factory")
    cfg = _load_configuration(component_path)
    provider = _object(_required(cfg, "provider"), "provider")
    if _required(provider, "name") != "openai_api":
        raise CheckpointGConfigurationError("only explicit openai_api is reviewed")
    api_key_name = str(_required(provider, "api_key_environment_variable"))
    api_key = os.environ.get(api_key_name, "")
    if not api_key:
        raise CheckpointGConfigurationError(f"{api_key_name} is not set")
    endpoint = str(_required(provider, "api_endpoint"))
    timeout = int(_required(provider, "timeout_seconds"))

    feedback_cfg = _object(_required(cfg, "feedback"), "feedback")
    feedback_model = _real_identity(
        _required(feedback_cfg, "model_id"), "feedback.model_id")
    counter, tokenizer_identity = _exact_counter(feedback_model)
    feedback_transport = _BoundedOpenAITransport(
        endpoint=endpoint, api_key=api_key, timeout_seconds=timeout,
        maximum_calls=int(_required(feedback_cfg, "maximum_calls")))
    feedback_backend = OpenAICompatibleEpisodeFeedbackProviderAdapter(
        provider="openai_api",
        model_id=feedback_model,
        tokenizer_identity=tokenizer_identity,
        exact_token_counter=counter,
        context_limit=int(_required(feedback_cfg, "context_limit")),
        maximum_output_tokens=int(_required(
            feedback_cfg, "maximum_output_tokens")),
        generation_settings=_object(
            _required(feedback_cfg, "generation_settings"),
            "feedback.generation_settings"),
        feedback_policy_version=str(_required(
            feedback_cfg, "policy_version")),
        response_transport=feedback_transport.feedback)
    feedback = LLMEpisodeFeedbackGenerator(
        response_provider=feedback_backend,
        artifact_resolver=SavedEpisodeFeedbackArtifactResolver(),
        policy_version=str(_required(feedback_cfg, "policy_version")),
        request_representation="model_compact")

    updater_cfg = _object(_required(cfg, "updater"), "updater")
    updater_model = _real_identity(
        _required(updater_cfg, "model_id"), "updater.model_id")
    updater_transport = _BoundedOpenAITransport(
        endpoint=endpoint, api_key=api_key, timeout_seconds=timeout,
        maximum_calls=1)
    updater_backend = OpenAICompatibleMetaPromptUpdaterBackend(
        provider="openai_api", model_id=updater_model,
        maximum_output_tokens=int(_required(
            updater_cfg, "maximum_output_tokens")),
        generation_settings=_object(
            _required(updater_cfg, "generation_settings"),
            "updater.generation_settings"),
        updater_policy_version=str(_required(updater_cfg, "policy_version")),
        response_transport=updater_transport.request)
    updater = LLMMetaPromptUpdater(
        backend=updater_backend,
        updater_policy_version=str(_required(updater_cfg, "policy_version")))

    runtime = dict(_object(_required(cfg, "runtime"), "runtime"))
    worker_gpus_override = getattr(args, "worker_gpus", None)
    if worker_gpus_override is not None:
        resolved_gpus = tuple(
            value.strip() for value in str(worker_gpus_override).split(",")
            if value.strip())
        if not resolved_gpus or len(set(resolved_gpus)) != len(resolved_gpus):
            raise CheckpointGConfigurationError(
                "worker GPU override must be non-empty and unique")
        runtime["worker_gpus"] = resolved_gpus
    worker_timeout_override = getattr(
        args, "worker_result_timeout_seconds", None)
    if worker_timeout_override is not None:
        if float(worker_timeout_override) <= 0:
            raise CheckpointGConfigurationError(
                "worker result timeout must be positive")
        runtime["worker_result_timeout_seconds"] = float(
            worker_timeout_override)
    for identity_name in (
            "captioner_model_id", "prompt_generator_model_id",
            "prompt_generator_backend_id", "dvd_orchestrator_tool_model",
            "dvd_text_fallback_model", "paired_model_identity",
            "evaluation_pipeline_identity"):
        _real_identity(_required(runtime, identity_name),
                       f"runtime.{identity_name}")
    expected = {
        "captioner_model_id": config.CAPTION_MODEL_ID,
        "prompt_generator_model_id": config.CAPTION_MODEL_ID,
        "dvd_orchestrator_tool_model": config.ORCHESTRATOR_TOOL_MODEL,
        "dvd_text_fallback_model": config.TEXT_FALLBACK_MODEL,
        "use_transcript": config.USE_TRANSCRIPT,
        "sample_fps": config.SAMPLE_FPS,
        "clip_seconds": config.CLIP_SECS,
        "dvd_text_backend": config.DVD_TEXT_BACKEND,
        "dvd_use_openai_tools": config.DVD_USE_OPENAI_TOOLS,
    }
    mismatches = {name: (runtime.get(name), value)
                  for name, value in expected.items()
                  if runtime.get(name) != value}
    if mismatches:
        raise CheckpointGConfigurationError(
            f"runtime config disagrees with active production config: {mismatches}")
    if json.loads(dumps_canonical(_required(runtime, "caption_decoding"))) != \
            json.loads(dumps_canonical(config.CAPTION_DECODING)):
        raise CheckpointGConfigurationError(
            "caption_decoding disagrees with active production config")
    gpus = tuple(str(item) for item in _required(runtime, "worker_gpus"))
    if not gpus or len(set(gpus)) != len(gpus):
        raise CheckpointGConfigurationError("worker_gpus must be unique")
    components_path = Path(str(_required(runtime, "scaffold_components_path")))
    components = json.loads(components_path.read_text(encoding="utf-8"))
    scaffold = scaffold_policy_from_json(components["scaffold_policy"])
    contract = scaffold_contract_from_json(components["scaffold_contract"])
    videos = tuple(DVDMetaPromptConfirmationVideo(
        video_id=str(item["video_id"]),
        provider_indices=tuple(int(v) for v in item["provider_indices"]),
        question_ids=tuple(str(v) for v in item["question_ids"]))
        for item in _required(cfg, "confirmation_videos"))
    neutral_bank = PromptBankSnapshot(
        bank_version="bank_v9999", entries=(), max_selected_entries=1,
        created_by="checkpoint_g_free_form_compatibility",
        provenance={"legacy_codebook_used": False})
    neutral_router = RouterPolicySnapshot(
        router_version="router_v9999", policy_type="free_form_compatibility",
        max_selected_entries=1,
        configuration={"legacy_property_routing_used": False},
        provenance={"checkpoint": "G"})
    def construct_confirmation():
        for path in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
            if path not in os.sys.path:
                os.sys.path.insert(0, path)
        from data_provider import get_provider
        from dvd_prompt import get_prompts
        provider_data = get_provider(
            str(_required(runtime, "benchmark")),
            split=str(_required(runtime, "benchmark_split")))
        ensure_backend(
            gpus[0], preload_captioner=False, preload_embedder=True,
            text_backend=str(_required(runtime, "dvd_text_backend")),
            use_openai_tools=bool(_required(
                runtime, "dvd_use_openai_tools")))
        builder_kwargs = {
            "parallel_gpus": gpus,
            "routing_mode": "free_form_generator",
        }
        if "worker_result_timeout_seconds" in runtime:
            builder_kwargs.update({
                "worker_result_timeout_seconds": float(
                    runtime["worker_result_timeout_seconds"]),
                "worker_log_directory": os.path.join(
                    str(_required(runtime, "cache_root")),
                    "worker_logs", "confirmation"),
            })
        builder = HistoryAwareBaselineCaptionViewBuilder.from_local_qwen(
            **builder_kwargs)
        prompts = get_prompts()
        dvd = HistoryAwareDVDConfirmationEvaluator(
            sample_loader=provider_data.__getitem__,
            history_aware_builder=builder,
            base_prompt_template=prompts.caption_prompt,
            merge_prompt=prompts.merge_prompt,
            sample_source_identity=str(_required(
                runtime, "sample_source_identity")),
            cache_root=str(_required(runtime, "cache_root")),
            cache_manifest_path=str(_required(
                runtime, "cache_manifest_path")),
            history_block_seconds=float(_required(
                runtime, "history_block_seconds")),
            max_history_captions=int(_required(
                runtime, "max_history_captions")),
            dvd_max_iterations=int(_required(runtime, "dvd_max_iterations")),
            gpu=gpus[0], frame_sampling_configuration={
                "sample_fps": runtime["sample_fps"],
                "clip_seconds": runtime["clip_seconds"],
                "segment_boundary_rule": "dvd_build_clips_fixed_duration",
                "frame_order_rule": "sorted_decoded_frame_paths"},
            downstream_qa_configuration={
                "text_backend": runtime["dvd_text_backend"],
                "use_openai_tools": runtime["dvd_use_openai_tools"]})
        return DVDMetaPromptConfirmationEvaluator(
            evaluator=dvd, confirmation_videos=videos,
            compatibility_bank=neutral_bank,
            compatibility_router=neutral_router,
            scaffold_policy=scaffold, scaffold_contract=contract,
            prompt_generator_model_id=str(_required(
                runtime, "prompt_generator_model_id")),
            prompt_generator_backend_id=str(_required(
                runtime, "prompt_generator_backend_id")),
            prompt_generator_max_tokens=int(_required(
                runtime, "prompt_generator_max_tokens")),
            model_identity=str(_required(runtime, "paired_model_identity")),
            decoding_settings=_object(
                _required(runtime, "paired_decoding_settings"),
                "runtime.paired_decoding_settings"),
            cache_reset_identity=str(_required(
                runtime, "cache_reset_identity")),
            evaluation_pipeline_identity=str(_required(
                runtime, "evaluation_pipeline_identity")))

    confirmation = LazyDVDMetaPromptConfirmationEvaluator(
        configuration_identity={
            "policy_version": DVDMetaPromptConfirmationEvaluator.policy_version,
            "runtime": runtime,
            "confirmation_videos": videos,
            "legacy_property_routing_used": False,
            "construction": "deferred_until_update_decision",
        }, factory=construct_confirmation)
    return feedback, updater, confirmation
