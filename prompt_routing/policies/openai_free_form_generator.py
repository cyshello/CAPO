"""Free-form instruction generator on a hosted OpenAI vision model.

Implements the shared `FreeFormPromptGenerator` interface (`generate`,
`configuration_identity`, `identity_hash`, `last_exchange`), so
`build_free_form_selection` and the history-aware builder are untouched.

Why it exists: the local 7B captioner cannot hold the "write an instruction,
not a caption" role. Measured 2026-07-22 on five mid-video segments, its
generated instruction and the resulting caption became byte-identical on four of
them, and the same text repeated across segments — the condition collapsed into
one caption per block. A hosted model keeps the two roles separate.

Paid: normally one vision request per segment (up to three identical transport
attempts under the configured retry policy, plus the transient-transport retries
in `network_retry`, which repeat the same bytes after a DNS/connection/5xx
failure and bill only when the provider actually answers). Frames follow the shared frame
policy — a 0.5 fps subset at half resolution — so a generator request carries
a few small images instead of ten full-size ones at the current 1 FPS decode.
Those images dominate the request's token cost, so the subsample is the lever
that controls it.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import time
from typing import Any, Mapping, Sequence

from surrogate_rollout import config
from surrogate_rollout import network_retry
from surrogate_rollout import token_ledger
from surrogate_rollout.optimization.openai_chat_model_profile import (
    adapt_chat_completions_body,
    is_reasoning_chat_model,
)
from surrogate_rollout.prompt_routing import generator_response_cache as grc
from surrogate_rollout.prompt_routing import static_meta_replace_body as smrb
from surrogate_rollout.prompt_routing.free_form_instruction_generator import (
    FREE_FORM_ROUTING_MODE,
    GeneratedCaptionInstruction,
    GeneratorExchange,
    FreeFormGenerationError,
)
from surrogate_rollout.prompt_routing.schemas import SegmentContext, dumps_canonical
from surrogate_rollout.schemas import sha256_text

DEFAULT_MODEL_ID = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_RETRIES = 2
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 300
# Observed: gpt-5-mini at reasoning_effort="medium" returns an empty completion
# at 512 output tokens and a full instruction at 4096.
REASONING_OUTPUT_TOKEN_FLOOR = 2048
PROVIDER = "openai"


def _data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    with open(path, "rb") as handle:
        payload = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime or 'image/jpeg'};base64,{payload}"


class OpenAIFreeFormInstructionGenerator:
    """Replace-body instruction generator over an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        max_tokens: int = smrb.DEFAULT_GENERATOR_MAX_TOKENS,
        caption_fps: float = 1.0,
        retries: int = DEFAULT_RETRIES,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        template_text: str | None = None,
        meta_prompt_id: str | None = None,
        backend_id: str | None = None,
        response_cache_root: str | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.caption_fps = caption_fps
        self.retries = retries
        self.temperature = temperature
        self.timeout = timeout
        self.template = template_text or smrb.META_PROMPT_TEXT
        if not self.template.strip():
            raise ValueError("template_text must be non-empty")
        self.template_version = smrb.META_PROMPT_VERSION
        self.meta_prompt_id = meta_prompt_id or smrb.META_PROMPT_ID
        self.backend_id = backend_id or f"{PROVIDER}:{model_id}"
        # Explicit root wins; otherwise the run's environment decides, and an
        # unset variable leaves the generator calling the provider every time.
        self.response_cache_root = response_cache_root
        self.last_exchange: GeneratorExchange | None = None
        self.last_response_cache_hit: bool = False

    # ----------------------------- identity ------------------------------- #
    @property
    def configuration_identity(self) -> Mapping[str, Any]:
        return {
            "mode": FREE_FORM_ROUTING_MODE,
            "composition": "replace_body",
            "provider": PROVIDER,
            "model": self.model_id,
            "backend": self.backend_id,
            "base_url": self.base_url,
            "template_version": self.template_version,
            "template_hash": sha256_text(self.template),
            "meta_prompt_id": self.meta_prompt_id,
            "max_tokens": self.max_tokens,
            "parser_version": smrb.PARSER_VERSION,
            "composition_version": smrb.COMPOSITION_VERSION,
            "request_schema_version": smrb.REQUEST_SCHEMA_VERSION,
            "generator_sample_fps": smrb.GENERATOR_SAMPLE_FPS,
            "caption_source_fps": self.caption_fps,
            "generator_image_scale": smrb.GENERATOR_IMAGE_SCALE,
            "real_model": True,
        }

    @property
    def identity_hash(self) -> str:
        return sha256_text(dumps_canonical(self.configuration_identity))

    # ------------------------------ calls --------------------------------- #
    def _post_once(
        self, headers: Mapping[str, str], payload: Mapping[str, Any],
    ) -> str:
        """One vision request. Returns the completion text, possibly empty.

        A transient provider status is raised as `TransientTransportError` so
        `network_retry` repeats the identical request; every other non-200 is a
        rejection of this request and fails immediately.
        """
        import requests

        response = requests.post(
            f"{self.base_url}/chat/completions", headers=dict(headers),
            json=dict(payload), timeout=self.timeout)
        if response.status_code != 200:
            detail = f"HTTP {response.status_code}: {response.text[:300]}"
            if network_retry.is_transient_http_status(response.status_code):
                raise network_retry.TransientTransportError(
                    detail, retry_after_seconds=network_retry.parse_retry_after(
                        response.headers.get("Retry-After")))
            raise FreeFormGenerationError(detail)
        data = response.json()
        # Additive token accounting only (vision + text of this call). Recorded
        # per answered request, so retried attempts that never reached the
        # provider contribute nothing.
        token_ledger.record(
            "prompt_generator", self.model_id, data.get("usage"))
        return (data["choices"][0]["message"]["content"] or "").strip()

    def _complete(self, frames: Sequence[str], prompt: str) -> str:
        if not self.api_key:
            raise FreeFormGenerationError(
                "the OpenAI free-form generator requires OPENAI_API_KEY")
        # Checked here rather than in __init__: callers build a generator with
        # default settings and replace it with the configured one a line later,
        # so construction is not where the budget is known. A reasoning model
        # spends this budget on hidden reasoning before it writes anything, and
        # an empty completion is a hard failure once per segment.
        effort = config.GENERATOR_REASONING_EFFORT
        if effort is not None and is_reasoning_chat_model(self.model_id) and \
                self.max_tokens < REASONING_OUTPUT_TOKEN_FLOOR:
            raise FreeFormGenerationError(
                f"{self.model_id} with reasoning_effort={effort!r} needs at "
                f"least {REASONING_OUTPUT_TOKEN_FLOOR} output tokens; "
                f"{self.max_tokens} leaves nothing for the instruction after "
                "the reasoning")

        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": _data_url(path)}}
            for path in frames
        ]
        content.append({"type": "text", "text": prompt})

        payload = adapt_chat_completions_body({
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }, reasoning_effort=config.GENERATOR_REASONING_EFFORT)
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                # The inner retry covers requests that never got an answer; this
                # loop covers answers that came back unusable (an empty
                # completion), which is a different failure and is charged for.
                text = network_retry.retry_transient(
                    lambda: self._post_once(headers, payload),
                    description=f"{self.model_id} free-form generator request")
                if text:
                    return text
                last_error = FreeFormGenerationError("empty completion")
            except Exception as exc:  # noqa: BLE001 - network/quota/5xx
                last_error = exc
            if attempt < self.retries:
                time.sleep(2 ** attempt)
        raise FreeFormGenerationError(
            f"OpenAI free-form generator failed after {self.retries + 1} "
            f"attempts: {type(last_error).__name__}: {last_error}")

    def generate(
        self,
        context: SegmentContext,
        clip_info: Mapping[str, Any],
    ) -> GeneratedCaptionInstruction:
        frames = context.segment_features.get("frame_references") or ()
        if isinstance(frames, str):
            frames = (frames,)
        frames = [str(item) for item in frames]
        if not frames:
            raise FreeFormGenerationError("current segment frames are required")
        if not context.history_summary:
            raise FreeFormGenerationError(
                "serialized caption history is required")

        generator_frames = smrb.prepare_generator_frames(
            frames, caption_fps=self.caption_fps)
        try:
            request = smrb.build_generator_request(
                video_id=context.video_id, segment_id=context.segment_id,
                frame_references=generator_frames,
                serialized_history=context.history_summary,
                generator_max_tokens=self.max_tokens,
                meta_prompt_id=self.meta_prompt_id,
                meta_prompt_version=self.template_version,
                meta_prompt_text=self.template)
        except smrb.StaticMetaPromptError as exc:
            raise FreeFormGenerationError(str(exc)) from exc
        prompt = smrb.render_generator_prompt(
            request, meta_prompt_text=self.template)
        request_hash = sha256_text(prompt)

        # Reuse is keyed by this exact request plus the provider settings that
        # decide the answer, so a different meta prompt renders a different
        # request and can never read this entry. See generator_response_cache.
        cache_root = (self.response_cache_root
                      or config.generator_response_cache_root())
        cache_key = grc.GeneratorResponseCacheKey(
            request_hash=request_hash,
            base_url=self.base_url,
            model_id=self.model_id,
            backend_id=self.backend_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            reasoning_effort=config.GENERATOR_REASONING_EFFORT,
        )
        started = time.monotonic()
        raw = grc.load(cache_root, video_id=context.video_id,
                       segment_id=context.segment_id, key=cache_key)
        self.last_response_cache_hit = raw is not None
        if raw is None:
            raw = self._complete(generator_frames, prompt)
            grc.store(cache_root, video_id=context.video_id,
                      segment_id=context.segment_id, key=cache_key,
                      raw_response=raw)
        elapsed = time.monotonic() - started

        self.last_exchange = GeneratorExchange(
            request=request, request_hash=request_hash,
            raw_output=str(raw), output_hash=sha256_text(str(raw)))
        try:
            instruction = smrb.parse_generated_instruction(raw)
        except smrb.StaticMetaPromptError as exc:
            raise FreeFormGenerationError(str(exc)) from exc
        return GeneratedCaptionInstruction(
            instruction=instruction,
            raw_response=str(raw),
            model_id=self.model_id,
            template_version=self.template_version,
            parser_path="plain_text",
            request_hash=request_hash,
            generation_seconds=elapsed,
            cache_hit=self.last_response_cache_hit,
        )
