"""Provider-neutral adapter for one generated caption instruction.

The active provider owns request construction, frame preprocessing, and output
parsing. This module only adapts its one generated instruction into the shared
PromptEntry/RoutingDecision records consumed by the caption builder.

QA-leakage guard: provider implementations receive SegmentContext and clip
metadata but must serialize only query-independent video context. The active
OpenAI implementation enforces that contract in static_meta_replace_body.py.

Cache identity: the generated instruction becomes part of the composed prompt
(=> composed_prompt_hash / prompt_hash), and the RoutingDecision carries a
reserved free-form router version ``router_v8888_<identity8>`` derived from the
generator configuration identity (mode, model, template version/hash, decoding).
Existing property-router caches therefore can never
resolve against free-form caches: their router_version namespace differs and
free-form decisions/records additionally carry ``mode=free_form_generator``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from surrogate_rollout.prompt_routing.schemas import (
    PromptEntry,
    RoutingDecision,
    SegmentContext,
    dumps_canonical,
)
from surrogate_rollout.schemas import sha256_text

FREE_FORM_ROUTING_MODE = "free_form_generator"
FREE_FORM_PROMPT_ID = "free_form_generated"
# Reserved router-version number for the free-form baseline. Real router
# lineages are persisted snapshots (e.g. router_v0002 / bootstrap router_v9002);
# 8888 plus the generator-identity content suffix keeps free-form caption cache
# keys outside every property-router cache identity.
FREE_FORM_ROUTER_VERSION_NUMBER = 8888


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
    # True when the provider was not called because an identical request was
    # already answered in this run's generator cache. Reporting only: it stays
    # out of the routing decision, whose payload feeds persisted artifacts.
    cache_hit: bool = False


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


def build_free_form_selection(
    *,
    generator: Any,
    context: SegmentContext,
    clip_info: Mapping[str, Any],
    bank_version: str,
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
        bank_version=bank_version,
        router_version=router_version,
        selected_prompt_ids=(FREE_FORM_PROMPT_ID,),
        prompt_scores={FREE_FORM_PROMPT_ID: 1.0},
        selection_order=(FREE_FORM_PROMPT_ID,),
        decision_payload={
            "mode": FREE_FORM_ROUTING_MODE,
            "request_schema_version": generator.configuration_identity[
                "request_schema_version"],
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
