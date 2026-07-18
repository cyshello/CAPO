"""Strict real-LLM multi-property proposal policy for one evidence video."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from surrogate_rollout import config
from surrogate_rollout.optimization.property_proposal import (
    CandidatePropertyProposal,
    PropertyApplicability,
    VideoProposalContext,
)
from surrogate_rollout.optimization.interventional_feedback import (
    FrameTransformConfig,
    load_bounded_frame,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import as_json_dict, dumps_canonical
from surrogate_rollout.schemas import sha256_text


LOGGER = logging.getLogger(__name__)


PROPOSAL_POLICY_VERSION = "multi_property_proposer_v11"
REQUEST_SCHEMA_VERSION = "multimodal_property_proposal_request_v10"
INPUT_IDENTITY_SCHEMA_VERSION = "multimodal_property_proposal_identity_v6"
OPAQUE_CANDIDATE_ID_SCHEMA_VERSION = "opaque_candidate_proposal_id_v3"
PROPOSAL_ARTIFACT_SCHEMA_VERSION = "multi_property_proposal_artifact_v6"
PROPOSER_META_PROMPT_VERSION = "property_proposer_meta_prompt_v4"
MISSING_SLOT_RETRY_POLICY_VERSION = "missing_qa_slot_retry_v1"
MISSING_SLOT_RETRY_ARTIFACT_SCHEMA_VERSION = (
    "property_proposal_missing_slot_retry_v1")
EVIDENCE_SELECTION_POLICY_VERSION = "compact_cited_then_inspected_evidence_v2"
EVIDENCE_PACKING_SCHEMA_VERSION = "property_proposal_evidence_packing_v1"
MAX_EVIDENCE_TEXT_BUDGET_CHARS = 200_000
EVIDENCE_TEXT_RESERVE_CHARS = 50_000
EVIDENCE_JSON_HEADROOM_CHARS = 2_048
INSPECTED_DESCRIPTION_CHARS = 250
INSPECTED_TRANSCRIPT_CHARS = 120
OUTPUT_FIELDS = frozenset({
    "source_qa_slot", "suggested_property_id", "property_text", "applicability",
    "covered_by_existing_property_ids", "failure_analysis",
})
PROPOSER_META_PROMPT = (
    "Generate executable captioning-property intervention candidates for every QA "
    "marked eligible, whether its baseline answer is correct or incorrect. Produce at "
    "least one candidate for each eligible source_qa_slot. "
    ""
    "For incorrect samples, hypothesize captioning instructions that could recover "
    "missing, ambiguous, temporally disordered, or poorly represented evidence. For "
    "correct samples, hypothesize instructions that could preserve or strengthen the "
    "captioning behavior that supported the correct answer, or reduce plausible risks "
    "such as omission, ambiguity, unsupported inference, temporal confusion, excessive "
    "detail, or repetition. "
    ""
    "Every candidate must describe an executable captioning behavior, not a downstream "
    "QA or reasoning behavior. The property must tell the captioner what observable "
    "information to detect, distinguish, track, organize, or explicitly qualify. It "
    "must not instruct the model to answer the source question, identify the video's "
    "main idea, explain the significance of an event, judge whether a claim is "
    "historically correct, or justify an answer choice. "
    ""
    "Generalize each candidate beyond its source QA. The property_text must remain "
    "useful when the source question, answer choices, video, and named entities are "
    "removed. Do not include source-specific people, places, historical events, answer "
    "content, or conclusions in property_text. Source-specific details may appear in "
    "why_proposed only as evidence motivating the generalized instruction. "
    ""
    "When the source QA does not reveal an obvious captioning failure, still produce "
    "at least one conservative candidate by extracting the most reusable captioning "
    "behavior that plausibly supported or could safeguard the result. Do not fabricate "
    "missing evidence or claim that the baseline was deficient without support. For "
    "such cases, frame why_proposed as a preservation or robustness hypothesis rather "
    "than as an observed failure. "
    ""
    "Do not decide whether a candidate belongs in the persistent codebook: retrieval, "
    "intervention, and the downstream updater decide whether it is useful, redundant, "
    "harmful, mergeable, or rejectable. Similarity to the current codebook and uncertain "
    "benefit are not sufficient reasons to omit all candidates, but candidates must "
    "still be executable, caption-level, source-independent, and grounded in current "
    "frames, current transcript, or preceding caption history available to the "
    "captioner. Never require unavailable external tools or knowledge. "
    ""
    "Ensure that why_proposed is consistent with source_qa_baseline_correct. When it is "
    "true, do not describe the baseline answer as incorrect or failed. Return only the "
    "strict JSON schema."
)

_EXPLICIT_ZERO_PROPOSAL_REASONS = frozenset({
    "runtime_failure", "missing_evidence", "malformed_input",
    "unreliable_annotation",
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


def opaque_candidate_property_id(
    *, context: VideoProposalContext, property_text: str,
    applicability: PropertyApplicability,
    source_question_id: str | None = None,
    source_qa_baseline_correct: bool | None = None,
) -> str:
    """Return a stable proposal handle, not a proposed final property ID."""
    digest = sha256_text(dumps_canonical({
        "schema_version": OPAQUE_CANDIDATE_ID_SCHEMA_VERSION,
        "source_video_id": context.video_id,
        "baseline_run_id": context.baseline_run_id,
        "source_question_id": source_question_id,
        "source_qa_baseline_correct": source_qa_baseline_correct,
        "normalized_property_text": _dedup_key(property_text),
        "applicability": {
            "when": _dedup_key(applicability.when),
            "positive_cues": tuple(
                _dedup_key(item) for item in applicability.positive_cues),
            "negative_cues": tuple(
                _dedup_key(item) for item in applicability.negative_cues),
            "required_modalities": applicability.required_modalities,
        },
    }))
    return f"candidate_{digest[:20]}"


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


def _selected_used_segments(
    row: Mapping[str, Any],
) -> tuple[str, ...]:
    references = row.get("reference_sets") or {}
    focused = []
    for name in ("explicitly_cited_segments", "frame_inspected_segments"):
        for value in references.get(name) or ():
            segment_id = str(value)
            if segment_id not in focused:
                focused.append(segment_id)
    return tuple(sorted(focused, key=_segment_sort_key))


def _selected_evidence_segments(
    row: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Return cited first, then inspected-only, with deterministic time order."""
    references = row.get("reference_sets") or {}
    cited = {str(value) for value in
             references.get("explicitly_cited_segments") or ()}
    inspected = {str(value) for value in
                 references.get("frame_inspected_segments") or ()}
    return tuple(
        [(segment_id, "explicitly_cited")
         for segment_id in sorted(cited, key=_segment_sort_key)]
        + [(segment_id, "inspected_only")
           for segment_id in sorted(inspected - cited, key=_segment_sort_key)]
    )


def _caption_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("caption") or "")
    return str(value or "")


_TRANSCRIPT_SUFFIX_MARKER = "\n\nTranscript during this video clip:"


def _split_baseline_caption(caption: str) -> tuple[str, str]:
    """Losslessly separate generated text from the known appended source suffix."""
    generated, marker, transcript = caption.partition(_TRANSCRIPT_SUFFIX_MARKER)
    if not marker:
        return caption, ""
    return generated, transcript


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit == 1:
        return "…"
    return value[:limit - 1].rstrip() + "…"


def _normalize_evidence_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _timestamp_range(segment_id: str) -> str:
    match = re.fullmatch(
        r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)", str(segment_id))
    if not match:
        raise PropertyProposalError(
            f"proposal evidence has invalid stable segment ID: {segment_id}")
    return f"[{match.group(1)}–{match.group(2)}]"


def _substantially_duplicates(description: str, transcript: str) -> bool:
    description_key = _dedup_key(description)
    transcript_key = _dedup_key(transcript)
    if not transcript_key:
        return True
    if not description_key:
        return False
    if len(transcript_key) >= 20 and (
            transcript_key in description_key or description_key in transcript_key):
        return True
    description_tokens = set(description_key.split())
    transcript_tokens = set(transcript_key.split())
    union = description_tokens | transcript_tokens
    return bool(union) and len(description_tokens & transcript_tokens) / len(union) >= 0.8


def _serialized_evidence_text_chars(evidence: Mapping[str, Any]) -> int:
    text_only = dict(evidence)
    image = text_only.get("representative_image") or {}
    if image:
        text_only["representative_image"] = {
            "mime_type": image.get("mime_type"),
            "attached_multimodal_image": True,
        }
    return len(dumps_canonical(text_only))


def _evidence_text_budget(max_text_payload_chars: int) -> int:
    reserve = min(EVIDENCE_TEXT_RESERVE_CHARS, max_text_payload_chars // 4)
    return min(
        MAX_EVIDENCE_TEXT_BUDGET_CHARS,
        max(1, max_text_payload_chars - reserve),
    )


def _text_only_request(request: Mapping[str, Any]) -> dict[str, Any]:
    output = json.loads(json.dumps(request))
    for qa in output.get("qas") or ():
        for evidence in qa.get("used_segment_evidence") or ():
            image = evidence.get("representative_image") or {}
            if image:
                evidence["representative_image"] = {
                    "mime_type": image.get("mime_type"),
                    "attached_multimodal_image": True,
                }
    return output


def _qa_runtime_valid(row: Mapping[str, Any]) -> bool:
    if "runtime_valid" in row:
        return row.get("runtime_valid") is True
    if row.get("errors"):
        return False
    prediction = row.get("prediction")
    if prediction is None or not str(prediction).strip():
        return False
    if "parsed_answer" in row:
        parsed = row.get("parsed_answer")
        if parsed is None or not str(parsed).strip():
            return False
    return True


def _qa_proposal_requirement(
    row: Mapping[str, Any], selected_segments: tuple[str, ...],
) -> dict[str, Any]:
    reasons = []
    if not _qa_runtime_valid(row):
        reasons.append("runtime_failure")
    if not str(row.get("question") or "").strip() or \
            not tuple(row.get("options") or ()) or \
            not str(row.get("ground_truth") or "").strip():
        reasons.append("malformed_input")
    if not selected_segments:
        reasons.append("missing_evidence")
    if row.get("annotation_reliable") is False:
        reasons.append("unreliable_annotation")
    explicit_reason = row.get("proposal_invalid_reason")
    if explicit_reason:
        normalized_reason = str(explicit_reason).strip()
        if normalized_reason not in _EXPLICIT_ZERO_PROPOSAL_REASONS:
            raise PropertyProposalError(
                "unsupported proposal_invalid_reason: " + normalized_reason)
        reasons.append(normalized_reason)
    reasons = tuple(dict.fromkeys(reasons))
    if reasons:
        return {
            "status": "ineligible",
            "minimum_proposals": 0,
            "maximum_proposals": 0,
            "zero_proposal_allowed": True,
            "invalid_reasons": reasons,
        }
    return {
        "status": "required_intervention_candidate",
        "minimum_proposals": 1,
        "maximum_proposals": 2,
        "zero_proposal_allowed": False,
        "invalid_reasons": (),
    }


def _proposal_cardinality_requirement(
    context: VideoProposalContext, *, max_proposals: int,
) -> dict[str, Any]:
    per_qa = tuple(_qa_proposal_requirement(
        row, _selected_used_segments(row))
        for row in context.baseline_qa_results)
    required_count = sum(
        item["status"] == "required_intervention_candidate"
        for item in per_qa)
    if required_count > max_proposals:
        raise PropertyProposalError(
            "max_proposals is smaller than the number of eligible QAs")
    return {
        "eligible_qa_count": required_count,
        "minimum_accepted_proposals": required_count,
        "maximum_accepted_proposals": (
            min(max_proposals, required_count * 2)
            if required_count else max_proposals),
        "zero_proposal_allowed": required_count == 0,
        "per_qa": per_qa,
    }


def _validate_proposal_cardinality(
    context: VideoProposalContext,
    proposals: tuple[CandidatePropertyProposal, ...],
    *,
    max_proposals: int,
) -> None:
    requirement = _proposal_cardinality_requirement(
        context, max_proposals=max_proposals)
    expected = {
        str(row["question_id"]): item
        for row, item in zip(context.baseline_qa_results, requirement["per_qa"])
        if item["status"] == "required_intervention_candidate"
    }
    counts = {question_id: 0 for question_id in expected}
    for proposal in proposals:
        if len(proposal.source_question_ids) != 1:
            raise PropertyProposalParseError(
                "new proposals must identify exactly one source QA")
        question_id = proposal.source_question_ids[0]
        if question_id not in counts:
            raise PropertyProposalParseError(
                f"proposal references ineligible source QA: {question_id}")
        counts[question_id] += 1
    invalid = {
        question_id: count for question_id, count in counts.items()
        if not 1 <= count <= 2
    }
    if invalid:
        raise PropertyProposalParseError(
            "every eligible QA requires 1..2 executable intervention candidates; "
            f"counts={invalid}")
    maximum = int(requirement["maximum_accepted_proposals"])
    if len(proposals) > maximum:
        raise PropertyProposalParseError(
            f"proposal count exceeds configured maximum {maximum}")


def _required_qa_slots(request: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(row["qa_slot"]) for row in request.get("qas") or ()
        if (row.get("proposal_requirement") or {}).get("status") ==
        "required_intervention_candidate")


def _missing_qa_slots(
    rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    required_slots: tuple[str, ...],
) -> tuple[str, ...]:
    counts = {slot: 0 for slot in required_slots}
    for row in rows:
        slot = row.get("source_qa_slot")
        if slot in counts:
            counts[str(slot)] += 1
    return tuple(slot for slot in required_slots if counts[slot] == 0)


def _build_missing_slot_retry_request(
    request: Mapping[str, Any], *, missing_slots: tuple[str, ...],
    attempt_index: int, remaining_capacity: int,
) -> dict[str, Any]:
    retry = json.loads(json.dumps(request))
    missing = set(missing_slots)
    retry["qas"] = [
        row for row in retry.get("qas") or () if row.get("qa_slot") in missing]
    retry_maximum = min(remaining_capacity, len(missing_slots) * 2)
    retry["max_proposals"] = retry_maximum
    retry["proposal_cardinality"] = {
        "eligible_qa_count": len(missing_slots),
        "minimum_accepted_proposals": len(missing_slots),
        "maximum_accepted_proposals": retry_maximum,
        "zero_proposal_allowed": False,
    }
    retry["retry"] = {
        "schema_version": MISSING_SLOT_RETRY_ARTIFACT_SCHEMA_VERSION,
        "policy_version": MISSING_SLOT_RETRY_POLICY_VERSION,
        "attempt_index": attempt_index,
        "required_source_qa_slots": list(missing_slots),
    }
    retry["task"] = (
        str(retry["task"])
        + " This is a bounded retry because the previous response omitted "
        + ", ".join(missing_slots)
        + ". Return one or two proposals for every QA included in this retry "
        "request, use only those source_qa_slot values, and do not repeat a QA "
        "slot that is absent from this retry request."
    )
    return retry


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
    evidence_candidates = []
    for qa_index, row in enumerate(rows, start=1):
        qa_slot = f"qa_{qa_index}"
        private_evidence = []
        selected_segments = _selected_used_segments(row)
        references = row.get("reference_sets") or {}
        cited_ids = {str(value) for value in
                     references.get("explicitly_cited_segments") or ()}
        inspected_ids = {str(value) for value in
                         references.get("frame_inspected_segments") or ()}
        for segment_id, evidence_role in _selected_evidence_segments(row):
            frames = tuple(context.frame_references.get(segment_id) or ())
            caption = _caption_text(context.captions.get(segment_id))
            if not frames or not caption:
                raise PropertyProposalError(
                    "used proposal evidence is missing frames or baseline caption")
            frame_path = str(frames[len(frames) // 2])
            encoded = dict(frame_loader(
                frame_path, bounds.max_frame_bytes, transform))
            generated_description, source_transcript = _split_baseline_caption(caption)
            normalized_description = _normalize_evidence_text(
                generated_description)
            normalized_transcript = _normalize_evidence_text(source_transcript)
            if _substantially_duplicates(
                    normalized_description, normalized_transcript):
                payload_transcript = ""
            else:
                payload_transcript = normalized_transcript
            timestamp_range = _timestamp_range(segment_id)
            image = {
                "mime_type": encoded["mime_type"],
                "base64_data": encoded["base64_data"],
            }
            if evidence_role == "explicitly_cited":
                payload_evidence = {
                    "timestamp_range": timestamp_range,
                    "evidence_role": evidence_role,
                    "generated_description": _bounded_text(
                        normalized_description, bounds.max_caption_chars),
                    "representative_image": image,
                }
                if payload_transcript:
                    payload_evidence["transcript"] = _bounded_text(
                        payload_transcript, bounds.max_caption_chars)
            else:
                compact_description = _bounded_text(
                    normalized_description, INSPECTED_DESCRIPTION_CHARS)
                summary = f"{timestamp_range} {compact_description}"
                if payload_transcript:
                    summary += " | tx: " + _bounded_text(
                        payload_transcript, INSPECTED_TRANSCRIPT_CHARS)
                payload_evidence = {
                    "evidence_role": evidence_role,
                    "summary": summary,
                    "representative_image": image,
                }
            private_row = {
                "segment_id": segment_id,
                "timestamp_range": timestamp_range,
                "evidence_role": evidence_role,
                "explicitly_cited": segment_id in cited_ids,
                "frame_inspected": segment_id in inspected_ids,
                "baseline_caption": caption,
                "baseline_generated_description": generated_description,
                "source_transcript": source_transcript,
                "frame_path": os.path.abspath(frame_path),
                "source_sha256": encoded["source_sha256"],
                "transformed_sha256": encoded["transformed_sha256"],
                "transformed_bytes": encoded["transformed_bytes"],
                "transform_configuration": encoded["transform_configuration"],
                "included_in_payload": False,
                "omission_reason": None,
            }
            private_evidence.append(private_row)
            evidence_candidates.append({
                "qa_index": qa_index - 1,
                "qa_slot": qa_slot,
                "segment_id": segment_id,
                "evidence_role": evidence_role,
                "payload": payload_evidence,
                "private": private_row,
            })
        qas.append({
            "qa_slot": qa_slot,
            "question": str(row.get("question") or ""),
            "answer_choices": tuple(row.get("options") or ()),
            "ground_truth": str(row.get("ground_truth") or ""),
            "baseline_prediction": row.get("prediction"),
            "is_correct": bool(row.get("is_correct")),
            "proposal_requirement": _qa_proposal_requirement(
                row, selected_segments),
            "bounded_reasoning": _bounded_reasoning(row, bounds),
            "used_segment_evidence": [],
        })
        private_qas.append({
            "qa_slot": qa_slot,
            "question_id": str(row["question_id"]),
            "baseline_correct": bool(row.get("is_correct")),
            "selected_segment_evidence": tuple(private_evidence),
        })
    active_codebook = tuple({
        "property_id": entry.prompt_id,
        "property_text": entry.prompt_text,
        "name": entry.name,
        "description": entry.description,
    } for entry in context.prompt_bank.entries if entry.status == "active")
    cardinality = _proposal_cardinality_requirement(
        context, max_proposals=max_proposals)
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "task": (
            "Given the questions, ground-truth and baseline predictions, bounded "
            "reasoning, timestamp-ranged cited or frame-inspected evidence, and the "
            "current codebook, propose "
            "executable captioning-property intervention candidates. The captioner can "
            "use only "
            "current frames, the current transcript, and preceding caption history. "
            "A property may preserve, integrate, or express information supported by "
            "those available modalities, but must never request external knowledge, "
            "background research, or historical facts absent from them. Each proposal "
            "must name its required modalities. generated_description is the "
            "captioner's generated description. transcript is source evidence "
            "that may previously have appeared after the raw marker 'Transcript during "
            "this video clip:'; its mere presence does not mean the generated description "
            "preserved its meaning. Compare the two when diagnosing omissions. "
            "For every QA marked required_intervention_candidate, return one or two "
            "proposals carrying that exact source_qa_slot. This requirement applies to "
            "both correct and incorrect baselines. For incorrect samples, propose ways "
            "to recover missing or poorly represented evidence. For correct samples, "
            "propose ways to preserve useful evidence or improve clarity, temporal "
            "ordering, robustness, efficiency, concision, relevance, or suppression of "
            "speculation and repetition. These are executable test hypotheses, not "
            "accepted codebook updates. Do not omit a proposal because it may fail to "
            "improve the answer, seems insufficiently general, resembles an active "
            "property, overlaps lexically with QA text, or may later prove repetitive. "
            "Only the downstream updater may accept, reject, revise, or merge a tested "
            "candidate after reading intervention evidence. "
            "covered_by_existing_property_ids are non-binding hints naming active "
            "properties that may be related or may cover the candidate; use an empty "
            "list when uncertain. failure_analysis should explain the caption weakness "
            "or preservation/improvement opportunity motivating this executable test. "
            "suggested_property_id is only a readable naming "
            "hint and never becomes the proposal or final codebook identity. Return "
            "only the strict output schema."
        ),
        "max_proposals": max_proposals,
        "proposal_cardinality": {
            key: value for key, value in cardinality.items() if key != "per_qa"
        },
        "qas": qas,
        "current_codebook": active_codebook,
        "output_schema": {
            "proposals": ({
                "source_qa_slot": "qa_1",
                "suggested_property_id": "readable non-binding ID suggestion",
                "property_text": "concise reusable captioning instruction",
                "applicability": {
                    "when": "reusable natural-language application condition",
                    "positive_cues": ["observable modality or semantic cue"],
                    "negative_cues": ["observable condition for non-application"],
                    "required_modalities": ["frames"],
                },
                "covered_by_existing_property_ids": [],
                "failure_analysis": "natural-language caption failure analysis",
            },),
        },
    }

    base_text_chars = len(dumps_canonical(_text_only_request(request)))
    if base_text_chars > bounds.max_text_payload_chars:
        raise PropertyProposalError(
            "proposal prompt/schema/QA context exceeds "
            f"max_payload_chars={bounds.max_text_payload_chars} before evidence")
    evidence_budget = min(
        _evidence_text_budget(bounds.max_text_payload_chars),
        max(1, bounds.max_text_payload_chars - base_text_chars
            - EVIDENCE_JSON_HEADROOM_CHARS),
    )
    included_evidence_chars = 0
    included_by_role = {"explicitly_cited": 0, "inspected_only": 0}
    omitted_by_role = {"explicitly_cited": 0, "inspected_only": 0}
    # _selected_evidence_segments already orders each QA by role and timestamp;
    # this stable global ordering ensures every cited interval precedes any
    # inspected-only interval across the complete request.
    evidence_candidates.sort(key=lambda item: (
        0 if item["evidence_role"] == "explicitly_cited" else 1,
        item["qa_index"], _segment_sort_key(item["segment_id"]),
    ))
    for candidate in evidence_candidates:
        payload_evidence = candidate["payload"]
        evidence_chars = _serialized_evidence_text_chars(payload_evidence)
        omission_reason = None
        if included_evidence_chars + evidence_chars > evidence_budget:
            omission_reason = "evidence_text_budget_exhausted"
        else:
            qa_evidence = request["qas"][candidate["qa_index"]][
                "used_segment_evidence"]
            qa_evidence.append(payload_evidence)
            proposed_text_chars = len(dumps_canonical(_text_only_request(request)))
            if proposed_text_chars > bounds.max_text_payload_chars:
                qa_evidence.pop()
                omission_reason = "total_text_payload_limit"
        role = candidate["evidence_role"]
        if omission_reason is None:
            included_evidence_chars += evidence_chars
            included_by_role[role] += 1
            candidate["private"]["included_in_payload"] = True
        else:
            omitted_by_role[role] += 1
            candidate["private"]["omission_reason"] = omission_reason

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
        "evidence_selection_policy": EVIDENCE_SELECTION_POLICY_VERSION,
        "evidence_packing": {
            "schema_version": EVIDENCE_PACKING_SCHEMA_VERSION,
            "ordering": "explicitly_cited_then_inspected_only_by_qa_and_timestamp",
            "evidence_text_budget_chars": evidence_budget,
            "included_evidence_text_chars": included_evidence_chars,
            "original_interval_count": len(evidence_candidates),
            "included_interval_count": sum(included_by_role.values()),
            "omitted_interval_count": sum(omitted_by_role.values()),
            "included_by_role": included_by_role,
            "omitted_by_role": omitted_by_role,
        },
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
    # Retained as a compatibility argument. The compact evidence policy is
    # character-budgeted rather than count-truncated.
    del max_captions
    request, _ = build_proposal_input(
        context,
        max_proposals=max_proposals,
        bounds=ProposalEvidenceBounds(
            max_reasoning_events_per_qa=min(max_trace_events_per_qa, 3),
            max_text_payload_chars=max_payload_chars,
        ),
    )
    return request


def _structurally_unusable_instruction_reason(text: str) -> str | None:
    """Reject only instructions that cannot be executed as a captioner prompt."""
    lowered = text.casefold()
    unresolved = (
        r"\b(?:todo|tbd|placeholder|snake_case_failure_type)\b",
        r"\{\{[^{}]+\}\}", r"<[^<>]+>",
        r"\[(?:insert|fill|replace)[^\]]*\]",
    )
    if any(re.search(pattern, lowered) for pattern in unresolved):
        return "contains unresolved placeholder"
    unavailable = (
        r"\b(?:research|look up|search for|consult)\b.{0,80}"
        r"\b(?:external|online|background|historical|web|internet)\b",
        r"\buse\b.{0,40}\b(?:external|background) knowledge\b",
        r"\bfetch\b.{0,40}\b(?:from the web|from the internet)\b",
    )
    if any(re.search(pattern, lowered) for pattern in unavailable):
        return "requires inputs or tools unavailable to the captioner"
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
    ordered_qas = sorted(
        context.baseline_qa_results,
        key=lambda row: (bool(row.get("is_correct")), str(row.get("question_id"))),
    )
    qa_by_slot = {
        f"qa_{index}": row for index, row in enumerate(ordered_qas, start=1)}
    active = {entry.prompt_id: entry for entry in context.prompt_bank.entries
              if entry.status == "active"}
    seen_signatures: set[str] = set()
    proposals = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != OUTPUT_FIELDS:
            raise PropertyProposalParseError(
                f"proposal {index} must contain exactly {sorted(OUTPUT_FIELDS)}")
        suggested_id = row["suggested_property_id"]
        if not isinstance(suggested_id, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", suggested_id):
            raise PropertyProposalParseError(f"proposal {index} has invalid ID")
        source_qa_slot = row["source_qa_slot"]
        if not isinstance(source_qa_slot, str) or source_qa_slot not in qa_by_slot:
            raise PropertyProposalParseError(
                f"proposal {index} has unknown source_qa_slot")
        source_qa = qa_by_slot[source_qa_slot]
        text = normalize_property_text(row["property_text"])
        if not text or len(text) > max_property_text_chars:
            raise PropertyProposalParseError(
                f"proposal {index} property_text is empty or too long")
        try:
            applicability = PropertyApplicability.from_mapping(row["applicability"])
        except ValueError as exc:
            raise PropertyProposalParseError(
                f"proposal {index} has invalid applicability: {exc}") from exc
        reported_coverage = row["covered_by_existing_property_ids"]
        failure_analysis = row["failure_analysis"]
        if not isinstance(reported_coverage, list) or any(
                not isinstance(item, str) or not item
                for item in reported_coverage):
            raise PropertyProposalParseError(
                f"proposal {index} covered_by_existing_property_ids "
                "must be a string array")
        if len(reported_coverage) != len(set(reported_coverage)):
            raise PropertyProposalParseError(
                f"proposal {index} coverage hints must be unique")
        if not isinstance(failure_analysis, str) or not failure_analysis.strip():
            raise PropertyProposalParseError(
                f"proposal {index} failure_analysis must be non-empty")
        unknown_coverage = set(reported_coverage) - set(active)
        if unknown_coverage:
            raise PropertyProposalParseError(
                f"proposal {index} covers unknown properties: "
                f"{sorted(unknown_coverage)}")
        applicability_texts = (
            applicability.when,
            *applicability.positive_cues,
            *applicability.negative_cues,
        )
        for field_text in (text, *applicability_texts):
            invalid_reason = _structurally_unusable_instruction_reason(field_text)
            if invalid_reason:
                raise PropertyProposalParseError(
                    f"proposal {index} {invalid_reason}")
        coverage_hints = tuple(reported_coverage)
        signature = dumps_canonical({
            "source_qa_slot": source_qa_slot,
            "property_text": text,
            "applicability": as_json_dict(applicability),
        })
        if signature in seen_signatures:
            raise PropertyProposalParseError(
                "exact duplicate proposals within one response")
        seen_signatures.add(signature)
        source_question_id = str(source_qa["question_id"])
        baseline_correct = bool(source_qa.get("is_correct"))
        candidate_id = opaque_candidate_property_id(
            context=context, property_text=text, applicability=applicability,
            source_question_id=source_question_id,
            source_qa_baseline_correct=baseline_correct)
        proposals.append(CandidatePropertyProposal(
            candidate_property_id=candidate_id,
            suggested_property_id=suggested_id,
            property_text=text,
            applicability=applicability,
            source_video_id=context.video_id,
            source_question_ids=(source_question_id,),
            source_qa_baseline_correct=baseline_correct,
            coverage_hints=coverage_hints,
            failure_analysis=failure_analysis.strip(),
            proposer_policy_version=policy_version,
        ))
    return tuple(proposals), ()


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
        # These constructor arguments remain accepted for legacy callers, but v6
        # intentionally has no per-QA evidence-count cap.
        self.max_evidence_segments_per_qa = max_evidence_segments_per_qa
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
            "meta_prompt_version": PROPOSER_META_PROMPT_VERSION,
            "missing_slot_retry_policy_version":
                MISSING_SLOT_RETRY_POLICY_VERSION,
            "missing_slot_max_retries":
                config.PROPERTY_PROPOSAL_MISSING_SLOT_MAX_RETRIES,
            "provider": provider,
            "max_proposals": self.max_proposals,
            "max_payload_chars": self.max_payload_chars,
            "max_trace_events_per_qa": self.max_trace_events_per_qa,
            "evidence_selection_policy": EVIDENCE_SELECTION_POLICY_VERSION,
            "evidence_packing_schema": EVIDENCE_PACKING_SCHEMA_VERSION,
            "max_evidence_text_budget_chars": MAX_EVIDENCE_TEXT_BUDGET_CHARS,
            "evidence_text_reserve_chars": EVIDENCE_TEXT_RESERVE_CHARS,
            "evidence_json_headroom_chars": EVIDENCE_JSON_HEADROOM_CHARS,
            "inspected_description_chars": INSPECTED_DESCRIPTION_CHARS,
            "inspected_transcript_chars": INSPECTED_TRANSCRIPT_CHARS,
            "evidence_segment_limit_per_qa": None,
            "max_property_text_chars": self.max_property_text_chars,
            "max_reasoning_event_chars": self.max_reasoning_event_chars,
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
            _validate_proposal_cardinality(
                context, proposals, max_proposals=self.max_proposals)
            return proposals

        os.makedirs(artifact_dir, exist_ok=True)
        request_path = os.path.abspath(os.path.join(artifact_dir, "request.json"))
        identity_path = os.path.abspath(os.path.join(
            artifact_dir, "input_identity.json"))
        evidence_packing_path = os.path.abspath(os.path.join(
            artifact_dir, "evidence_packing.json"))
        provider_request_path = os.path.abspath(os.path.join(
            artifact_dir, "provider_request.json"))
        raw_path = os.path.abspath(os.path.join(artifact_dir, "raw_output.txt"))
        parsed_path = os.path.abspath(os.path.join(artifact_dir, "parsed_output.json"))
        rejected_path = os.path.abspath(os.path.join(artifact_dir, "rejections.json"))
        validation_path = os.path.abspath(os.path.join(
            artifact_dir, "validation.json"))
        body_builder = getattr(self.response_provider, "build_request_body", None)
        supports_multimodal = bool(getattr(
            self.response_provider, "supports_multimodal_request", False))

        def invoke_provider(
            attempt_request: Mapping[str, Any], *, attempt_dir: str,
        ) -> tuple[str, dict[str, Any]]:
            os.makedirs(attempt_dir, exist_ok=True)
            attempt_request_path = os.path.abspath(os.path.join(
                attempt_dir, "request.json"))
            attempt_provider_path = os.path.abspath(os.path.join(
                attempt_dir, "provider_request.json"))
            attempt_raw_path = os.path.abspath(os.path.join(
                attempt_dir, "raw_output.txt"))
            serialized = dumps_canonical(attempt_request)
            if os.path.exists(attempt_request_path):
                with open(attempt_request_path) as handle:
                    frozen = json.load(handle)
                if dumps_canonical(frozen) != serialized:
                    raise PropertyProposalConflictError(
                        "proposal attempt has different frozen input")
            else:
                _atomic_write_text(attempt_request_path, serialized + "\n")
            if callable(body_builder):
                body = body_builder(attempt_request)
                rendered_body = json.dumps(
                    body, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
                if os.path.exists(attempt_provider_path):
                    with open(attempt_provider_path) as handle:
                        if json.load(handle) != body:
                            raise PropertyProposalConflictError(
                                "proposal provider request identity mismatch")
                else:
                    _atomic_write_text(attempt_provider_path, rendered_body)
            else:
                attempt_provider_path = None
            cache_hit = os.path.exists(attempt_raw_path)
            if cache_hit:
                with open(attempt_raw_path) as handle:
                    attempt_raw = handle.read()
            else:
                provider_input = (
                    attempt_request if supports_multimodal else serialized)
                attempt_raw = self.response_provider(provider_input)
                _atomic_write_text(attempt_raw_path, attempt_raw)
            return attempt_raw, {
                "request_path": attempt_request_path,
                "provider_request_path": attempt_provider_path,
                "raw_output_path": attempt_raw_path,
                "provider_cache_hit": cache_hit,
            }

        _atomic_write_text(identity_path, dumps_canonical(input_identity) + "\n")
        packing_artifact = dict(input_identity["evidence_packing"])
        packing_artifact["per_qa"] = [{
            "qa_slot": qa["qa_slot"],
            "original_interval_count": len(qa["selected_segment_evidence"]),
            "included_interval_count": sum(
                bool(item["included_in_payload"])
                for item in qa["selected_segment_evidence"]),
            "omitted_interval_count": sum(
                not bool(item["included_in_payload"])
                for item in qa["selected_segment_evidence"]),
        } for qa in input_identity["qas"]]
        _atomic_write_text(
            evidence_packing_path,
            json.dumps(packing_artifact, sort_keys=True, ensure_ascii=False,
                       indent=2) + "\n",
        )
        LOGGER.info(
            "property proposer evidence intervals included=%s omitted=%s "
            "(cited=%s/%s inspected_only=%s/%s)",
            packing_artifact["included_interval_count"],
            packing_artifact["omitted_interval_count"],
            packing_artifact["included_by_role"]["explicitly_cited"],
            packing_artifact["omitted_by_role"]["explicitly_cited"],
            packing_artifact["included_by_role"]["inspected_only"],
            packing_artifact["omitted_by_role"]["inspected_only"],
        )
        raw, initial_artifacts = invoke_provider(
            request, attempt_dir=artifact_dir)
        provider_request_path = initial_artifacts["provider_request_path"]
        proposals, rejected = parse_proposal_output(
            raw, context, max_proposals=self.max_proposals,
            max_property_text_chars=self.max_property_text_chars,
            policy_version=self.policy_version)
        combined_rows = list(json.loads(raw)["proposals"])
        required_slots = _required_qa_slots(request)
        retry_records = []
        for attempt_index in range(
                1, config.PROPERTY_PROPOSAL_MISSING_SLOT_MAX_RETRIES + 1):
            missing_slots = _missing_qa_slots(combined_rows, required_slots)
            if not missing_slots:
                break
            remaining_capacity = self.max_proposals - len(combined_rows)
            if remaining_capacity < len(missing_slots):
                break
            retry_request = _build_missing_slot_retry_request(
                request, missing_slots=missing_slots,
                attempt_index=attempt_index,
                remaining_capacity=remaining_capacity)
            retry_dir = os.path.join(
                artifact_dir, "missing_slot_retries",
                f"attempt_{attempt_index:03d}")
            retry_raw, retry_artifacts = invoke_provider(
                retry_request, attempt_dir=retry_dir)
            record = {
                "schema_version": MISSING_SLOT_RETRY_ARTIFACT_SCHEMA_VERSION,
                "policy_version": MISSING_SLOT_RETRY_POLICY_VERSION,
                "attempt_index": attempt_index,
                "requested_source_qa_slots": list(missing_slots),
                **retry_artifacts,
            }
            try:
                retry_proposals, retry_rejected = parse_proposal_output(
                    retry_raw, context,
                    max_proposals=int(retry_request["max_proposals"]),
                    max_property_text_chars=self.max_property_text_chars,
                    policy_version=self.policy_version)
                retry_rows = json.loads(retry_raw)["proposals"]
                accepted_rows = []
                row_rejections = []
                for retry_row in retry_rows:
                    slot = retry_row.get("source_qa_slot")
                    if slot not in missing_slots:
                        row_rejections.append({
                            "source_qa_slot": slot,
                            "reason": "slot was not requested by this retry",
                        })
                        continue
                    try:
                        parse_proposal_output(
                            dumps_canonical({
                                "proposals": [*combined_rows, retry_row]}),
                            context, max_proposals=self.max_proposals,
                            max_property_text_chars=self.max_property_text_chars,
                            policy_version=self.policy_version)
                    except PropertyProposalParseError as exc:
                        row_rejections.append({
                            "source_qa_slot": slot, "reason": str(exc)})
                        continue
                    combined_rows.append(retry_row)
                    accepted_rows.append(retry_row)
                proposals, rejected = parse_proposal_output(
                    dumps_canonical({"proposals": combined_rows}), context,
                    max_proposals=self.max_proposals,
                    max_property_text_chars=self.max_property_text_chars,
                    policy_version=self.policy_version)
                missing_after = _missing_qa_slots(
                    combined_rows, required_slots)
                record.update({
                    "status": "accepted" if accepted_rows else "no_progress",
                    "parsed_proposal_count": len(retry_proposals),
                    "accepted_proposal_count": len(accepted_rows),
                    "parser_rejections": list(retry_rejected),
                    "row_rejections": row_rejections,
                    "missing_source_qa_slots_after": list(missing_after),
                })
            except PropertyProposalParseError as exc:
                record.update({
                    "status": "invalid_response", "reason": str(exc),
                    "accepted_proposal_count": 0,
                    "missing_source_qa_slots_after": list(missing_slots),
                })
            retry_validation_path = os.path.join(retry_dir, "validation.json")
            _atomic_write_text(retry_validation_path, json.dumps(
                record, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
            record["validation_path"] = os.path.abspath(retry_validation_path)
            retry_records.append(record)

        combined_raw_path = os.path.abspath(os.path.join(
            artifact_dir, "combined_output.json"))
        _atomic_write_text(combined_raw_path, dumps_canonical({
            "proposals": combined_rows}) + "\n")
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
        retry_manifest_path = os.path.abspath(os.path.join(
            artifact_dir, "missing_slot_retries.json"))
        _atomic_write_text(retry_manifest_path, json.dumps({
            "schema_version": MISSING_SLOT_RETRY_ARTIFACT_SCHEMA_VERSION,
            "policy_version": MISSING_SLOT_RETRY_POLICY_VERSION,
            "max_retries": config.PROPERTY_PROPOSAL_MISSING_SLOT_MAX_RETRIES,
            "attempts": retry_records,
            "initial_provider_cache_hit":
                initial_artifacts["provider_cache_hit"],
            "final_missing_source_qa_slots": list(_missing_qa_slots(
                combined_rows, required_slots)),
        }, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
        try:
            _validate_proposal_cardinality(
                context, proposals, max_proposals=self.max_proposals)
        except PropertyProposalParseError as exc:
            _atomic_write_text(validation_path, json.dumps({
                "status": "rejected", "reason": str(exc),
                "proposal_cardinality": request["proposal_cardinality"],
                "proposal_count": len(proposals),
                "missing_slot_retry_manifest_path": retry_manifest_path,
            }, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
            raise
        _atomic_write_text(validation_path, json.dumps({
            "status": "accepted",
            "proposal_cardinality": request["proposal_cardinality"],
            "proposal_count": len(proposals),
            "missing_slot_retry_manifest_path": retry_manifest_path,
        }, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
        _atomic_write_text(completed_path, json.dumps({
            "schema_version": PROPOSAL_ARTIFACT_SCHEMA_VERSION,
            "status": "completed", "input_fingerprint": fingerprint,
            "request_path": request_path, "raw_output_path": combined_raw_path,
            "initial_raw_output_path": raw_path,
            "input_identity_path": identity_path,
            "evidence_packing_path": evidence_packing_path,
            "provider_request_path": provider_request_path,
            "parsed_output_path": parsed_path,
            "rejections_path": rejected_path,
            "validation_path": validation_path,
            "missing_slot_retry_manifest_path": retry_manifest_path,
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
            "messages": [
                {"role": "system", "content": PROPOSER_META_PROMPT},
                {"role": "user", "content": content},
            ],
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
