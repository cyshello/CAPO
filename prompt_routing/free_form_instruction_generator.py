"""On-the-fly free-form caption-instruction generation (experimental baseline).

Ablation-only alternative to the property-bank + router path: instead of the
router selecting reusable prompt-bank properties, a multimodal generator
writes ONE segment-specific natural-language captioning instruction from the
exact same query-independent context the router sees (sampled segment frames +
the frozen local caption history). The instruction is then composed through the
SAME shared scaffold applier and captioned by the SAME captioner — nothing
downstream of instruction production changes. The generator, like the router,
never sees the segment transcript; the transcript still reaches the captioner
downstream exactly as in the property-bank path.

This module deliberately does NOT touch PromptRouter: the router semantically
selects property IDs; this component semantically generates prose. They stay
separate abstractions. Integration happens through one small branch in
HistoryAwareBaselineCaptionViewBuilder._build_sequential (opt-in, default off).

QA-leakage guard: the generator request is built only from
SegmentContext.video_id/segment_id/timestamps/frame_references and
history_summary. ``SegmentContext.question`` and ``metadata`` are never
serialized into the request (mirrors the existing router request construction).

Cache identity: the generated instruction becomes part of the composed prompt
(=> composed_prompt_hash / prompt_hash), and the RoutingDecision carries a
reserved free-form router version ``router_v8888_<identity8>`` derived from the
generator configuration identity (mode, model, template version/hash, decoding).
Existing property-router caches therefore can never
resolve against free-form caches: their router_version namespace differs and
free-form decisions/records additionally carry ``mode=free_form_generator``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from surrogate_rollout import config
from surrogate_rollout.prompt_routing.free_form_instruction_parser import (
    PARSER_VERSION,
    parse_generated_instruction,
)
from surrogate_rollout.prompt_routing.schemas import (
    PromptBankSnapshot,
    PromptEntry,
    RoutingDecision,
    SegmentContext,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_text

FREE_FORM_ROUTING_MODE = "free_form_generator"
PROPERTY_BANK_ROUTING_MODE = "property_bank"
FREE_FORM_PROMPT_ID = "free_form_generated"
FREE_FORM_REQUEST_SCHEMA_VERSION = "free_form_instruction_request_v1"
# Reserved router-version number for the free-form baseline. Real router
# lineages are persisted snapshots (e.g. router_v0002 / bootstrap router_v9002);
# 8888 plus the generator-identity content suffix keeps free-form caption cache
# keys outside every property-router cache identity.
FREE_FORM_ROUTER_VERSION_NUMBER = 8888

DEFAULT_GENERATOR_MAX_TOKENS = 192
DEFAULT_TEMPLATE_VERSION = "v1"

GENERATOR_TEMPLATES: Mapping[str, str] = {
    "v1": """You generate a segment-specific instruction for a video captioning model.

Inspect the sampled frames from the current video segment and the preceding
caption history.

Determine which visually grounded details should be described carefully
in this segment so that the resulting caption preserves information useful
for later long-video understanding.

Focus only on information supported by the current visual inputs and context.

Do not answer any downstream question.
Do not assume that you know what question will later be asked.
Do not summarize the entire video.
Do not invent identities, actions, states, relationships, or events.
Avoid generic advice that could apply to every segment.

Produce one concise and actionable captioning instruction for the current segment.
Return only strict JSON: {"caption_instruction": "..."}. Do not use Markdown.""",
}


class FreeFormGenerationError(ValueError):
    """The generator request is incomplete or its output is unusable."""


def free_form_router_version(identity_hash: str) -> str:
    """Reserved-namespace router version carrying the generator identity."""
    suffix = identity_hash[:8].lower()
    return f"router_v{FREE_FORM_ROUTER_VERSION_NUMBER}_{suffix}"


@dataclass(frozen=True)
class GeneratedCaptionInstruction:
    """One generated adaptive instruction plus its reproducibility metadata."""

    instruction: str
    raw_response: str | None
    model_id: str
    template_version: str
    parser_path: str
    request_hash: str
    generation_seconds: float


@dataclass(frozen=True)
class GeneratorExchange:
    """Audit record of one generator call (same shape as RouterExchange)."""

    request: Mapping[str, Any]
    request_hash: str
    raw_output: str
    output_hash: str


class FreeFormPromptGenerator(Protocol):
    def generate(
        self,
        context: SegmentContext,
        clip_info: Mapping[str, Any],
    ) -> GeneratedCaptionInstruction:
        ...


class VLMFreeFormInstructionGenerator:
    """Free-form instruction generator on the shared local VLM backend.

    Reuses the same multimodal backend object as the router/captioner (no
    separate inference stack); calls are unconstrained text generation and the
    output goes through the deterministic parser fallback ladder.
    """

    def __init__(self, vlm: Any, *,
                 max_tokens: int = DEFAULT_GENERATOR_MAX_TOKENS,
                 template_version: str = DEFAULT_TEMPLATE_VERSION,
                 model_id: str | None = None,
                 backend_id: str | None = None) -> None:
        if template_version not in GENERATOR_TEMPLATES:
            raise ValueError(
                f"unknown generator template version {template_version!r} "
                f"(known: {sorted(GENERATOR_TEMPLATES)})")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.vlm = vlm
        self.max_tokens = max_tokens
        self.template_version = template_version
        self.template = GENERATOR_TEMPLATES[template_version]
        self.model_id = model_id or config.CAPTION_MODEL_ID
        self.backend_id = backend_id or (
            f"{type(vlm).__module__}.{type(vlm).__qualname__}")
        self.last_exchange: GeneratorExchange | None = None

    @property
    def configuration_identity(self) -> Mapping[str, Any]:
        return {
            "mode": FREE_FORM_ROUTING_MODE,
            "provider": "local_qwen_vlm",
            "model": self.model_id,
            "backend": self.backend_id,
            "template_version": self.template_version,
            "template_hash": sha256_text(self.template),
            "max_tokens": self.max_tokens,
            "parser_version": PARSER_VERSION,
            "real_model": True,
        }

    @property
    def identity_hash(self) -> str:
        return sha256_text(dumps_canonical(self.configuration_identity))

    @classmethod
    def from_local_qwen(cls, **kwargs) -> "VLMFreeFormInstructionGenerator":
        from surrogate_rollout.prompt_routing.policies.history_aware_vlm_router import (
            get_local_qwen_backend,
        )

        return cls(get_local_qwen_backend(), **kwargs)

    def generate(
        self,
        context: SegmentContext,
        clip_info: Mapping[str, Any],
    ) -> GeneratedCaptionInstruction:
        frames = context.segment_features.get("frame_references") or ()
        if isinstance(frames, str):
            frames = (frames,)
        frames = tuple(str(item) for item in frames)
        if not frames:
            raise FreeFormGenerationError("current segment frames are required")
        if not context.history_summary:
            raise FreeFormGenerationError(
                "serialized caption history is required")
        try:
            history = json.loads(context.history_summary)
        except json.JSONDecodeError as exc:
            raise FreeFormGenerationError(
                "history_summary must be serialized JSON") from exc

        # Query-independent request only, identical context to the router:
        # frames + preceding caption history. No question, no metadata, no QA,
        # and — like the router — no segment transcript.
        request: dict[str, Any] = {
            "schema_version": FREE_FORM_REQUEST_SCHEMA_VERSION,
            "current_segment": {
                "video_id": context.video_id,
                "segment_id": context.segment_id,
                "timestamp_start": context.timestamp_start,
                "timestamp_end": context.timestamp_end,
                "frame_references": frames,
            },
            "preceding_caption_history": history,
            "generator_identity": {
                "template_version": self.template_version,
                "template_hash": sha256_text(self.template),
                "max_tokens": self.max_tokens,
            },
        }
        prompt = self.template + "\n" + dumps_canonical(request)
        request_hash = sha256_text(prompt)

        t0 = time.monotonic()
        raw = self.vlm.caption(frames, prompt, max_tokens=self.max_tokens)
        elapsed = time.monotonic() - t0

        outcome = parse_generated_instruction(raw)
        self.last_exchange = GeneratorExchange(
            request=request, request_hash=request_hash,
            raw_output=str(raw), output_hash=sha256_text(str(raw)))
        if not outcome.ok:
            raise FreeFormGenerationError(
                f"free-form generator output unusable "
                f"({outcome.parser_path}): {outcome.error}")
        return GeneratedCaptionInstruction(
            instruction=str(outcome.instruction),
            raw_response=str(raw),
            model_id=self.model_id,
            template_version=self.template_version,
            parser_path=outcome.parser_path,
            request_hash=request_hash,
            generation_seconds=elapsed,
        )


def build_free_form_selection(
    *,
    generator: Any,
    context: SegmentContext,
    clip_info: Mapping[str, Any],
    prompt_bank: PromptBankSnapshot,
) -> tuple[RoutingDecision, tuple[PromptEntry, ...], GeneratorExchange | None]:
    """Generate one instruction and adapt it to the shared scaffold interface.

    Returns a synthetic single-entry selection: a PromptEntry holding the
    generated instruction plus a RoutingDecision in the reserved free-form
    router-version namespace. The existing scaffold applier composes the entry
    exactly like a routed property instruction; the shared caption contract is
    untouched. The full audit trail (raw response, parser path, hashes) rides in
    decision_payload / entry provenance so the run can be audited without
    regenerating anything.
    """
    generated = generator.generate(context, clip_info)
    instruction_hash = sha256_text(generated.instruction)
    identity_hash = getattr(generator, "identity_hash", None) or sha256_text(
        dumps_canonical(dict(generator.configuration_identity)))
    router_version = free_form_router_version(identity_hash)
    entry = PromptEntry(
        prompt_id=FREE_FORM_PROMPT_ID,
        prompt_text=generated.instruction,
        prompt_hash=instruction_hash,
        name="Free-form generated instruction",
        description=("Segment-specific captioning instruction generated "
                     "on the fly by the free-form baseline generator."),
        tags=("free_form_generator",),
        created_by="free_form_instruction_generator",
        provenance={
            "mode": FREE_FORM_ROUTING_MODE,
            "generator_model": generated.model_id,
            "template_version": generated.template_version,
            "parser_path": generated.parser_path,
            "request_hash": generated.request_hash,
        },
    )
    decision = RoutingDecision(
        video_id=context.video_id,
        segment_id=context.segment_id,
        bank_version=prompt_bank.bank_version,
        router_version=router_version,
        selected_prompt_ids=(FREE_FORM_PROMPT_ID,),
        prompt_scores={FREE_FORM_PROMPT_ID: 1.0},
        selection_order=(FREE_FORM_PROMPT_ID,),
        decision_payload={
            "mode": FREE_FORM_ROUTING_MODE,
            "request_schema_version": FREE_FORM_REQUEST_SCHEMA_VERSION,
            "generator_model": generated.model_id,
            "generator_identity_hash": identity_hash,
            "template_version": generated.template_version,
            "parser_path": generated.parser_path,
            "request_hash": generated.request_hash,
            "raw_generator_response": generated.raw_response,
            "generated_instruction": generated.instruction,
            "generated_instruction_hash": instruction_hash,
            "generation_seconds": generated.generation_seconds,
        },
    )
    exchange = getattr(generator, "last_exchange", None)
    return decision, (entry,), exchange
