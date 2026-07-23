"""Static caption-instruction generator.

Implements the shared ``FreeFormPromptGenerator`` interface (``generate``,
``configuration_identity``, ``identity_hash``, ``last_exchange``) so that
``build_free_form_selection`` and the history-aware builder are used unchanged.

Unlike ``OpenAIFreeFormInstructionGenerator``, this generator performs **no**
model call. It returns the current versioned caption instruction verbatim for
every segment. That is the whole point of this path: the optimized artifact is
the caption prompt itself, not a meta-prompt that a generator LLM expands.

Cache identity: ``configuration_identity`` carries ``template_hash`` (the
sha256 of the instruction text), so the composed-prompt cache key changes the
moment the optimizer promotes a new caption prompt, and never collides with an
OpenAI-generator run (different ``provider``/``real_model``/``template_hash``).
"""

from __future__ import annotations

from typing import Any, Mapping

from surrogate_rollout.prompt_routing.free_form_instruction_generator import (
    FREE_FORM_ROUTING_MODE,
    GeneratedCaptionInstruction,
    GeneratorExchange,
    FreeFormGenerationError,
)
from surrogate_rollout.prompt_routing.schemas import SegmentContext, dumps_canonical
from surrogate_rollout.schemas import sha256_text

PROVIDER = "static"
COMPOSITION = "replace_body"
TEMPLATE_VERSION = "caption_prompt_static_v1"
REQUEST_SCHEMA_VERSION = "caption_prompt_static_request_v1"
PARSER_PATH = "verbatim"


class StaticInstructionGenerator:
    """Return one fixed caption instruction for every segment.

    ``template_text`` is the versioned caption prompt (``MetaPromptVersion.text``
    under this path). It is emitted verbatim; segment frames and caption history
    are ignored on purpose, so the composed caption prompt depends only on the
    promoted instruction and not on any per-segment generation.
    """

    def __init__(
        self,
        *,
        template_text: str,
        meta_prompt_id: str | None = None,
        model_id: str = "static",
        backend_id: str | None = None,
    ) -> None:
        if not isinstance(template_text, str) or not template_text.strip():
            raise ValueError("template_text (the caption instruction) is required")
        self.template = template_text.strip()
        self.model_id = model_id
        self.template_version = TEMPLATE_VERSION
        self.meta_prompt_id = meta_prompt_id or "caption_prompt_static"
        self.backend_id = backend_id or f"{PROVIDER}:{model_id}"
        self.last_exchange: GeneratorExchange | None = None

    # ----------------------------- identity ------------------------------- #
    @property
    def configuration_identity(self) -> Mapping[str, Any]:
        return {
            "mode": FREE_FORM_ROUTING_MODE,
            "composition": COMPOSITION,
            "provider": PROVIDER,
            "model": self.model_id,
            "backend": self.backend_id,
            "template_version": self.template_version,
            "template_hash": sha256_text(self.template),
            "meta_prompt_id": self.meta_prompt_id,
            "parser_version": PARSER_PATH,
            "request_schema_version": REQUEST_SCHEMA_VERSION,
            "real_model": False,
        }

    @property
    def identity_hash(self) -> str:
        return sha256_text(dumps_canonical(dict(self.configuration_identity)))

    # ------------------------------ calls --------------------------------- #
    def generate(
        self,
        context: SegmentContext,
        clip_info: Mapping[str, Any],
    ) -> GeneratedCaptionInstruction:
        # No frames or history are consulted; the instruction is the promoted
        # caption prompt itself. We still validate that the builder handed us a
        # real segment so a misconfigured routing_mode fails loudly.
        if not getattr(context, "video_id", None) or \
                not getattr(context, "segment_id", None):
            raise FreeFormGenerationError(
                "static generator requires a valid SegmentContext")
        request_hash = sha256_text(self.template)
        self.last_exchange = GeneratorExchange(
            request={
                "provider": PROVIDER,
                "video_id": context.video_id,
                "segment_id": context.segment_id,
                "template_hash": request_hash,
                "request_schema_version": REQUEST_SCHEMA_VERSION,
            },
            request_hash=request_hash,
            raw_output=self.template,
            output_hash=request_hash,
        )
        return GeneratedCaptionInstruction(
            instruction=self.template,
            raw_response=None,
            model_id=self.model_id,
            template_version=self.template_version,
            parser_path=PARSER_PATH,
            request_hash=request_hash,
            generation_seconds=0.0,
        )
