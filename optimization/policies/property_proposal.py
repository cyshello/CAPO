"""Strict real-LLM multi-property proposal policy for one evidence video."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from surrogate_rollout import config
from surrogate_rollout.optimization.property_proposal import (
    CandidatePropertyProposal,
    VideoProposalContext,
)
from surrogate_rollout.optimization.interventional_feedback import (
    FrameTransformConfig,
    load_bounded_frame,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import as_json_dict, dumps_canonical
from surrogate_rollout.schemas import sha256_text


PROPOSAL_POLICY_VERSION = "multi_property_proposer_v4"
REQUEST_SCHEMA_VERSION = "multimodal_property_proposal_request_v3"
INPUT_IDENTITY_SCHEMA_VERSION = "multimodal_property_proposal_identity_v1"
OUTPUT_FIELDS = frozenset({
    "candidate_property_id", "property_text", "motivating_failure_types",
    "covered_by_existing_property_ids", "proposal_rationale",
})


class PropertyProposalError(RuntimeError):
    pass


class PropertyProposalParseError(PropertyProposalError):
    pass


class PropertyProposalConflictError(PropertyProposalError):
    pass


@dataclass(frozen=True)
class ProposalEvidenceBounds:
    max_reasoning_events_per_qa: int = 3
    max_reasoning_event_chars: int = 1200
    max_evidence_segments_per_qa: int = 3
    max_caption_chars: int = 1200
    max_frame_bytes: int = 65536
    max_text_payload_chars: int = 40000

    def __post_init__(self) -> None:
        for name, value in as_json_dict(self).items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"ProposalEvidenceBounds.{name} must be positive")


def normalize_property_text(value: str) -> str:
    if not isinstance(value, str):
        raise PropertyProposalParseError("property_text must be a string")
    return " ".join(value.split()).strip()


def _dedup_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _read_trace(row: Mapping[str, Any]) -> tuple[Any, ...]:
    direct = row.get("reasoning_trace") or row.get("trajectory")
    if isinstance(direct, (list, tuple)):
        return tuple(direct)
    path = row.get("trajectory_path")
    if not path or not os.path.exists(path):
        return ()
    output = []
    with open(path) as f:
        for line in f:
            if line.strip():
                output.append(json.loads(line))
    return tuple(output)


_REASONING_DROP_KEYS = frozenset({
    "id", "tool_call_id", "question_id", "priority_rank",
    "source_video_id", "segment_id", "segment_ids", "database",
    "payload_truncation",
})


def _sanitize_reasoning(value: Any) -> Any:
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if str(key) in _REASONING_DROP_KEYS:
                continue
            if key == "arguments" and isinstance(item, str):
                try:
                    item = json.loads(item)
                except json.JSONDecodeError:
                    pass
            output[str(key)] = _sanitize_reasoning(item)
        return output
    if isinstance(value, (list, tuple)):
        return tuple(_sanitize_reasoning(item) for item in value)
    return value


def _bounded_reasoning(
    row: Mapping[str, Any], bounds: ProposalEvidenceBounds,
) -> tuple[str, ...]:
    rows = tuple(item for item in _read_trace(row)
                 if not (isinstance(item, Mapping)
                         and item.get("role") in ("system", "user")))
    selected = rows[-bounds.max_reasoning_events_per_qa:]
    output = []
    for item in selected:
        text = dumps_canonical(_sanitize_reasoning(item))
        if len(text) > bounds.max_reasoning_event_chars:
            text = text[:bounds.max_reasoning_event_chars] + "…"
        output.append(text)
    return tuple(output)


def _segment_sort_key(segment_id: str) -> tuple[float, str]:
    try:
        return float(segment_id.split("_", 1)[0]), segment_id
    except (TypeError, ValueError):
        return float("inf"), segment_id


def _evenly_sample(values: tuple[str, ...], limit: int) -> tuple[str, ...]:
    if len(values) <= limit:
        return values
    if limit == 1:
        return (values[len(values) // 2],)
    indices = tuple(round(index * (len(values) - 1) / (limit - 1))
                    for index in range(limit))
    return tuple(values[index] for index in indices)


def _selected_used_segments(
    row: Mapping[str, Any], limit: int,
) -> tuple[str, ...]:
    references = row.get("reference_sets") or {}
    focused = []
    for name in ("explicitly_cited_segments", "frame_inspected_segments",
                 "returned_segments"):
        for value in references.get(name) or ():
            segment_id = str(value)
            if segment_id not in focused:
                focused.append(segment_id)
    focused = sorted(focused, key=_segment_sort_key)
    selected = focused[:limit]
    if len(selected) < limit:
        used = tuple(sorted({str(item) for item in
                             (row.get("used_segments") or ())
                             if str(item) not in selected},
                            key=_segment_sort_key))
        selected.extend(_evenly_sample(used, limit - len(selected)))
    return tuple(selected[:limit])


def _caption_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("caption") or "")
    return str(value or "")


def _text_only_request(request: Mapping[str, Any]) -> dict[str, Any]:
    output = json.loads(json.dumps(request))
    for qa in output.get("qas") or ():
        for evidence in qa.get("used_segment_evidence") or ():
            image = evidence.get("representative_image") or {}
            evidence["representative_image"] = {
                "mime_type": image.get("mime_type"),
                "attached_multimodal_image": True,
            }
    return output


def build_proposal_input(
    context: VideoProposalContext,
    *,
    max_proposals: int,
    bounds: ProposalEvidenceBounds,
    frame_loader: Callable[
        [str, int, FrameTransformConfig], Mapping[str, Any]
    ] = load_bounded_frame,
    frame_transform: FrameTransformConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(context.baseline_qa_results) != 3:
        raise PropertyProposalError("proposal request requires all three baseline QAs")
    transform = frame_transform or FrameTransformConfig()
    rows = sorted(
        context.baseline_qa_results,
        key=lambda row: (bool(row.get("is_correct")), str(row.get("question_id"))),
    )
    qas = []
    private_qas = []
    for row in rows:
        evidence_rows = []
        private_evidence = []
        selected_segments = _selected_used_segments(
            row, bounds.max_evidence_segments_per_qa)
        for segment_id in selected_segments:
            frames = tuple(context.frame_references.get(segment_id) or ())
            caption = _caption_text(context.captions.get(segment_id))
            if not frames or not caption:
                raise PropertyProposalError(
                    "used proposal evidence is missing frames or baseline caption")
            frame_path = str(frames[len(frames) // 2])
            encoded = dict(frame_loader(
                frame_path, bounds.max_frame_bytes, transform))
            evidence_rows.append({
                "baseline_caption": (
                    caption[:bounds.max_caption_chars] + "…"
                    if len(caption) > bounds.max_caption_chars else caption),
                "representative_image": {
                    "mime_type": encoded["mime_type"],
                    "base64_data": encoded["base64_data"],
                },
            })
            private_evidence.append({
                "segment_id": segment_id,
                "frame_path": os.path.abspath(frame_path),
                "source_sha256": encoded["source_sha256"],
                "transformed_sha256": encoded["transformed_sha256"],
                "transformed_bytes": encoded["transformed_bytes"],
                "transform_configuration": encoded["transform_configuration"],
            })
        qas.append({
            "question": str(row.get("question") or ""),
            "answer_choices": tuple(row.get("options") or ()),
            "ground_truth": str(row.get("ground_truth") or ""),
            "baseline_prediction": row.get("prediction"),
            "is_correct": bool(row.get("is_correct")),
            "bounded_reasoning": _bounded_reasoning(row, bounds),
            "used_segment_evidence": tuple(evidence_rows),
        })
        private_qas.append({
            "question_id": str(row["question_id"]),
            "selected_segment_evidence": tuple(private_evidence),
        })
    active_codebook = tuple({
        "property_id": entry.prompt_id,
        "property_text": entry.prompt_text,
        "name": entry.name,
        "description": entry.description,
    } for entry in context.prompt_bank.entries if entry.status == "active")
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "task": (
            "Given the questions, ground-truth and baseline predictions, bounded "
            "reasoning, actual used-segment images and baseline captions, and the "
            "current codebook, propose only missing reusable captioning properties. "
            "Prioritize incorrect QAs and avoid instance-specific wording. "
            "covered_by_existing_property_ids are non-binding hints naming active "
            "properties that may be related or may cover the candidate; use an empty "
            "list when uncertain. Do not claim definite non-coverage while also "
            "listing coverage hints unless the rationale explicitly explains partial "
            "coverage or uncertainty. Propose only visually verifiable captioning "
            "instructions; never request external, historical, background, or other "
            "non-visual knowledge. Return only the strict output schema."
        ),
        "max_proposals": max_proposals,
        "qas": tuple(qas),
        "current_codebook": active_codebook,
        "output_schema": {
            "proposals": ({
                "candidate_property_id": "string",
                "property_text": "concise reusable captioning instruction",
                "motivating_failure_types": ["snake_case_failure_type"],
                "covered_by_existing_property_ids": [],
                "proposal_rationale": "brief rationale",
            },),
        },
    }
    text_chars = len(dumps_canonical(_text_only_request(request)))
    if text_chars > bounds.max_text_payload_chars:
        raise PropertyProposalError(
            f"proposal text exceeds max_payload_chars={bounds.max_text_payload_chars}")
    identity = {
        "schema_version": INPUT_IDENTITY_SCHEMA_VERSION,
        "source_video_id": context.video_id,
        "baseline_run_id": context.baseline_run_id,
        "qas": tuple(private_qas),
        "bounds": as_json_dict(bounds),
        "text_payload_chars": text_chars,
        "frame_transform": as_json_dict(transform),
    }
    return request, identity


def build_proposal_request(
    context: VideoProposalContext,
    *,
    max_proposals: int,
    max_trace_events_per_qa: int,
    max_captions: int,
    max_payload_chars: int,
) -> dict[str, Any]:
    request, _ = build_proposal_input(
        context,
        max_proposals=max_proposals,
        bounds=ProposalEvidenceBounds(
            max_reasoning_events_per_qa=min(max_trace_events_per_qa, 3),
            max_evidence_segments_per_qa=min(max_captions, 3),
            max_text_payload_chars=max_payload_chars,
        ),
    )
    return request


def _instance_specific_reason(text: str, context: VideoProposalContext) -> str | None:
    lowered = text.casefold()
    if context.video_id.casefold() in lowered:
        return "mentions source video ID"
    forbidden = (
        r"\bthis video\b", r"\bthe question\b", r"\bground[ -]?truth\b",
        r"\bcorrect answer\b", r"\banswer (?:is|option)\b",
        r"\boption\s+[a-z0-9]\b", r"\b\d{1,2}:\d{2}(?::\d{2})?\b",
        r"\b\d+_\d+\b",
    )
    if any(re.search(pattern, lowered) for pattern in forbidden):
        return "contains instance-specific question/answer/timestamp language"
    for row in context.baseline_qa_results:
        question = normalize_property_text(str(row.get("question") or ""))
        answer = normalize_property_text(str(row.get("ground_truth") or ""))
        if len(question) >= 12 and question.casefold() in lowered:
            return "copies source question"
        if len(answer) >= 3 and re.search(
                rf"(?<!\w){re.escape(answer.casefold())}(?!\w)", lowered):
            return "contains ground-truth answer text"
        for option in row.get("options") or ():
            option_text = normalize_property_text(str(option))
            if len(option_text) >= 3 and re.search(
                    rf"(?<!\w){re.escape(option_text.casefold())}(?!\w)", lowered):
                return "contains source answer-choice text"
    return None


def _non_visual_knowledge_reason(text: str) -> str | None:
    lowered = text.casefold()
    direct = (
        r"\bnon[- ]visual (?:knowledge|information|context|facts?)\b",
        r"\bexternal (?:knowledge|information|context|facts?|sources?)\b",
        r"\bbackground (?:knowledge|information|context|facts?|research)\b",
        r"\bhistorical (?:knowledge|context|background|facts?|research)\b",
        r"\b(?:world|domain|prior) knowledge\b",
        r"\bbeyond (?:what is|the content) (?:visible|shown|depicted)\b",
        r"\boutside (?:the|of the) (?:frames?|video|visual evidence)\b",
    )
    if any(re.search(pattern, lowered) for pattern in direct):
        return "requires non-visual or external/background/historical knowledge"
    knowledge_object = (
        r"(?:external|background|historical|biographical|real[- ]world) "
        r"(?:context|knowledge|information|facts?|details?)"
    )
    if re.search(
            rf"\b(?:add|include|provide|supply|infer|research|explain|use)\b"
            rf".{{0,80}}\b{knowledge_object}\b", lowered):
        return "requires non-visual or external/background/historical knowledge"
    return None


def _coverage_contradiction_reason(
    coverage_hints: tuple[str, ...], rationale: str,
) -> str | None:
    if not coverage_hints:
        return None
    lowered = rationale.casefold()
    definite_noncoverage = (
        r"\bnot covered by (?:any |the )?existing",
        r"\bno existing propert(?:y|ies) cover",
        r"\bnone of (?:the )?existing propert(?:y|ies)",
        r"\bunrelated to (?:all |the )?existing propert(?:y|ies)",
        r"\bentirely missing from (?:the )?(?:current )?codebook",
    )
    uncertainty = (
        r"\bmay\b", r"\bmight\b", r"\bpossibly\b", r"\buncertain\b",
        r"\bpartially\b", r"\bpartial coverage\b", r"\bnot fully covered\b",
        r"\brelated\b", r"\boverlap\b",
    )
    if any(re.search(pattern, lowered) for pattern in definite_noncoverage) and \
            not any(re.search(pattern, lowered) for pattern in uncertainty):
        return "contradictory_coverage_hints_without_uncertainty"
    return None


def parse_proposal_output(
    raw: str,
    context: VideoProposalContext,
    *,
    max_proposals: int,
    max_property_text_chars: int,
    policy_version: str = PROPOSAL_POLICY_VERSION,
) -> tuple[tuple[CandidatePropertyProposal, ...], tuple[dict[str, str], ...]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PropertyProposalParseError("proposal output is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {"proposals"}:
        raise PropertyProposalParseError(
            "proposal output must contain only the proposals key")
    rows = value["proposals"]
    if not isinstance(rows, list) or len(rows) > max_proposals:
        raise PropertyProposalParseError(
            f"proposals must be a list with at most {max_proposals} items")
    source_question_ids = tuple(str(row["question_id"])
                                for row in context.baseline_qa_results)
    active = {entry.prompt_id: entry for entry in context.prompt_bank.entries
              if entry.status == "active"}
    active_by_text = {_dedup_key(entry.prompt_text): entry.prompt_id
                      for entry in active.values()}
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    proposals = []
    rejected = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != OUTPUT_FIELDS:
            raise PropertyProposalParseError(
                f"proposal {index} must contain exactly {sorted(OUTPUT_FIELDS)}")
        candidate_id = row["candidate_property_id"]
        if not isinstance(candidate_id, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", candidate_id):
            raise PropertyProposalParseError(f"proposal {index} has invalid ID")
        text = normalize_property_text(row["property_text"])
        if not text or len(text) > max_property_text_chars:
            raise PropertyProposalParseError(
                f"proposal {candidate_id} property_text is empty or too long")
        failures = row["motivating_failure_types"]
        reported_coverage = row["covered_by_existing_property_ids"]
        rationale = row["proposal_rationale"]
        for name, items in (("motivating_failure_types", failures),
                            ("covered_by_existing_property_ids", reported_coverage)):
            if not isinstance(items, list) or any(
                    not isinstance(item, str) or not item for item in items):
                raise PropertyProposalParseError(
                    f"proposal {candidate_id} {name} must be a string array")
        if not failures or len(failures) != len(set(failures)) or \
                len(reported_coverage) != len(set(reported_coverage)):
            raise PropertyProposalParseError(
                f"proposal {candidate_id} requires unique lineage and failure types")
        if any(not re.fullmatch(r"[a-z][a-z0-9_]*", item)
               for item in failures):
            raise PropertyProposalParseError(
                f"proposal {candidate_id} failure types must be snake_case")
        if not isinstance(rationale, str) or not rationale.strip():
            raise PropertyProposalParseError(
                f"proposal {candidate_id} rationale must be non-empty")
        unknown_coverage = set(reported_coverage) - set(active)
        if unknown_coverage:
            raise PropertyProposalParseError(
                f"proposal {candidate_id} covers unknown properties: "
                f"{sorted(unknown_coverage)}")
        text_key = _dedup_key(text)
        if candidate_id in seen_ids or text_key in seen_text:
            raise PropertyProposalParseError("duplicated proposal ID or property text")
        seen_ids.add(candidate_id)
        seen_text.add(text_key)
        reason = _instance_specific_reason(text, context)
        if reason:
            rejected.append({"candidate_property_id": candidate_id, "reason": reason})
            continue
        reason = _non_visual_knowledge_reason(text)
        if reason:
            rejected.append({"candidate_property_id": candidate_id, "reason": reason})
            continue
        coverage_hints = tuple(reported_coverage)
        reason = _coverage_contradiction_reason(coverage_hints, rationale)
        if reason:
            rejected.append({"candidate_property_id": candidate_id, "reason": reason})
            continue
        if candidate_id in active:
            rejected.append({
                "candidate_property_id": candidate_id,
                "reason": "candidate_property_id_collides_with_active_property",
            })
            continue
        computed_coverage = active_by_text.get(text_key)
        if computed_coverage:
            rejected.append({
                "candidate_property_id": candidate_id,
                "reason": "exact_property_text_match:" + computed_coverage,
            })
            continue
        proposals.append(CandidatePropertyProposal(
            candidate_property_id=candidate_id,
            property_text=text,
            source_video_id=context.video_id,
            source_question_ids=source_question_ids,
            motivating_failure_types=tuple(failures),
            coverage_hints=coverage_hints,
            proposal_rationale=rationale.strip(),
            proposer_policy_version=policy_version,
        ))
    return tuple(proposals), tuple(rejected)


class MultiPropertyProposalPolicy:
    policy_version = PROPOSAL_POLICY_VERSION

    def __init__(
        self,
        *,
        response_provider: Callable[[Any], str],
        artifact_root: str | None = None,
        max_proposals: int = config.MAX_PROPERTY_PROPOSALS_PER_VIDEO,
        max_payload_chars: int = config.PROPERTY_PROPOSAL_MAX_PAYLOAD_CHARS,
        max_trace_events_per_qa: int = config.PROPERTY_PROPOSAL_MAX_TRACE_EVENTS_PER_QA,
        max_captions: int = config.PROPERTY_PROPOSAL_MAX_CAPTIONS,
        max_property_text_chars: int = config.PROPERTY_PROPOSAL_MAX_TEXT_CHARS,
        max_reasoning_event_chars: int = 1200,
        max_evidence_segments_per_qa: int = 3,
        max_caption_chars: int = 1200,
        max_frame_bytes: int = 65536,
        frame_loader: Callable[
            [str, int, FrameTransformConfig], Mapping[str, Any]
        ] = load_bounded_frame,
        frame_transform: FrameTransformConfig | None = None,
    ) -> None:
        if max_proposals < 0:
            raise ValueError("max_proposals must be non-negative")
        self.response_provider = response_provider
        self.artifact_root = artifact_root
        self.max_proposals = max_proposals
        self.max_payload_chars = max_payload_chars
        self.max_trace_events_per_qa = max_trace_events_per_qa
        self.max_captions = max_captions
        self.max_property_text_chars = max_property_text_chars
        self.max_reasoning_event_chars = max_reasoning_event_chars
        self.max_evidence_segments_per_qa = min(
            max_captions, max_evidence_segments_per_qa)
        self.max_caption_chars = max_caption_chars
        self.max_frame_bytes = max_frame_bytes
        self.frame_loader = frame_loader
        self.frame_transform = frame_transform or FrameTransformConfig()

    @property
    def configuration_identity(self) -> dict[str, Any]:
        metadata = getattr(self.response_provider, "metadata", None)
        provider = (metadata() if callable(metadata) else {
            "provider_type": (
                f"{type(self.response_provider).__module__}."
                f"{type(self.response_provider).__qualname__}"),
            "model": getattr(self.response_provider, "model", None),
        })
        return {
            "policy_version": self.policy_version,
            "provider": provider,
            "max_proposals": self.max_proposals,
            "max_payload_chars": self.max_payload_chars,
            "max_trace_events_per_qa": self.max_trace_events_per_qa,
            "max_captions": self.max_captions,
            "max_property_text_chars": self.max_property_text_chars,
            "max_reasoning_event_chars": self.max_reasoning_event_chars,
            "max_evidence_segments_per_qa": self.max_evidence_segments_per_qa,
            "max_caption_chars": self.max_caption_chars,
            "max_frame_bytes": self.max_frame_bytes,
            "frame_transform": as_json_dict(self.frame_transform),
        }

    def propose(self, context: VideoProposalContext):
        artifact_dir = context.proposal_artifact_dir or (
            os.path.join(self.artifact_root, context.video_id)
            if self.artifact_root else None)
        if not artifact_dir:
            raise PropertyProposalError("proposal artifact directory is required")
        bounds = ProposalEvidenceBounds(
            max_reasoning_events_per_qa=min(
                self.max_trace_events_per_qa, 3),
            max_reasoning_event_chars=self.max_reasoning_event_chars,
            max_evidence_segments_per_qa=self.max_evidence_segments_per_qa,
            max_caption_chars=self.max_caption_chars,
            max_frame_bytes=self.max_frame_bytes,
            max_text_payload_chars=self.max_payload_chars,
        )
        request, input_identity = build_proposal_input(
            context,
            max_proposals=self.max_proposals,
            bounds=bounds,
            frame_loader=self.frame_loader,
            frame_transform=self.frame_transform,
        )
        request_text = dumps_canonical(request)
        fingerprint = sha256_text(dumps_canonical({
            "request": request,
            "input_identity": input_identity,
            "configuration": self.configuration_identity,
        }))
        completed_path = os.path.join(artifact_dir, "completed.json")
        if os.path.exists(completed_path):
            with open(completed_path) as f:
                completed = json.load(f)
            if completed.get("input_fingerprint") != fingerprint:
                raise PropertyProposalConflictError(
                    "completed proposal artifact has different frozen input")
            with open(completed["raw_output_path"]) as f:
                raw = f.read()
            proposals, _ = parse_proposal_output(
                raw, context, max_proposals=self.max_proposals,
                max_property_text_chars=self.max_property_text_chars,
                policy_version=self.policy_version)
            return proposals

        os.makedirs(artifact_dir, exist_ok=True)
        request_path = os.path.abspath(os.path.join(artifact_dir, "request.json"))
        identity_path = os.path.abspath(os.path.join(
            artifact_dir, "input_identity.json"))
        provider_request_path = os.path.abspath(os.path.join(
            artifact_dir, "provider_request.json"))
        raw_path = os.path.abspath(os.path.join(artifact_dir, "raw_output.txt"))
        parsed_path = os.path.abspath(os.path.join(artifact_dir, "parsed_output.json"))
        rejected_path = os.path.abspath(os.path.join(artifact_dir, "rejections.json"))
        _atomic_write_text(request_path, request_text + "\n")
        _atomic_write_text(identity_path, dumps_canonical(input_identity) + "\n")
        body_builder = getattr(self.response_provider, "build_request_body", None)
        if callable(body_builder):
            _atomic_write_text(
                provider_request_path,
                json.dumps(body_builder(request), sort_keys=True,
                           ensure_ascii=False, indent=2) + "\n")
        else:
            provider_request_path = None
        provider_input = (request if getattr(
            self.response_provider, "supports_multimodal_request", False)
                          else request_text)
        raw = self.response_provider(provider_input)
        _atomic_write_text(raw_path, raw)
        proposals, rejected = parse_proposal_output(
            raw, context, max_proposals=self.max_proposals,
            max_property_text_chars=self.max_property_text_chars,
            policy_version=self.policy_version)
        _atomic_write_text(parsed_path, json.dumps({
            "proposals": [
                {key: value for key, value in as_json_dict(item).items()
                 if key != "proposer_policy_version"}
                for item in proposals
            ]
        }, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
        _atomic_write_text(rejected_path, json.dumps(
            {"rejections": rejected}, sort_keys=True, ensure_ascii=False,
            indent=2) + "\n")
        _atomic_write_text(completed_path, json.dumps({
            "schema_version": "multi_property_proposal_artifact_v1",
            "status": "completed", "input_fingerprint": fingerprint,
            "request_path": request_path, "raw_output_path": raw_path,
            "input_identity_path": identity_path,
            "provider_request_path": provider_request_path,
            "parsed_output_path": parsed_path,
            "rejections_path": rejected_path,
            "proposal_count": len(proposals),
        }, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
        return proposals


class OpenAIPropertyProposalProvider:
    """Configured optimization-LLM provider; instantiated only for real runs."""

    supports_multimodal_request = True

    def __init__(self, *, model: str = config.FEEDBACK_MODEL,
                 api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self.call_count = 0

    def build_request_body(self, request: Mapping[str, Any]) -> dict[str, Any]:
        text_request = _text_only_request(request)
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                "Return only strict JSON for this property proposal request:\n"
                "Attached images follow in QA and evidence-list order.\n"
                + dumps_canonical(text_request)
            ),
        }]
        for qa in request.get("qas") or ():
            for evidence in qa.get("used_segment_evidence") or ():
                image = evidence.get("representative_image") or {}
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": (f"data:{image['mime_type']};base64,"
                                f"{image['base64_data']}"),
                        "detail": "low",
                    },
                })
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

    def __call__(self, request: Mapping[str, Any]) -> str:
        self.call_count += 1
        body = json.dumps(self.build_request_body(request)).encode()
        api_request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key or self._load_key()}",
                     "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(api_request, timeout=600) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"OpenAI property proposal failed with HTTP {exc.code}: {detail}"
            ) from exc
        return payload["choices"][0]["message"]["content"]

    @staticmethod
    def _load_key() -> str:
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return key
        env_path = os.path.join(config.PROMPT_SENS_ROOT, ".env")
        if os.path.isfile(env_path):
            with open(env_path) as env_file:
                for line in env_file:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        raise RuntimeError("OPENAI_API_KEY is not configured")

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "openai_api", "model": self.model,
            "response_format": "json_object", "multimodal": True,
            "real_model": True,
        }
