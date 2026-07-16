"""Strict real-LLM multi-property proposal policy for one evidence video."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

from surrogate_rollout import config
from surrogate_rollout.optimization.property_proposal import (
    CandidatePropertyProposal,
    VideoProposalContext,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import as_json_dict, dumps_canonical
from surrogate_rollout.schemas import sha256_text


PROPOSAL_POLICY_VERSION = "multi_property_proposer_v1"
REQUEST_SCHEMA_VERSION = "multi_property_proposal_request_v1"
OUTPUT_FIELDS = frozenset({
    "candidate_property_id", "property_text", "source_video_id",
    "source_question_ids", "motivating_failure_types",
    "covered_by_existing_property_ids", "proposal_rationale",
})


class PropertyProposalError(RuntimeError):
    pass


class PropertyProposalParseError(PropertyProposalError):
    pass


class PropertyProposalConflictError(PropertyProposalError):
    pass


def normalize_property_text(value: str) -> str:
    if not isinstance(value, str):
        raise PropertyProposalParseError("property_text must be a string")
    return " ".join(value.split()).strip()


def _dedup_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _read_trace(row: Mapping[str, Any], limit: int) -> tuple[Any, ...]:
    direct = row.get("reasoning_trace") or row.get("trajectory")
    if isinstance(direct, (list, tuple)):
        return tuple(direct[:limit])
    path = row.get("trajectory_path")
    if not path or not os.path.exists(path):
        return ()
    output = []
    with open(path) as f:
        for line in f:
            if line.strip():
                output.append(json.loads(line))
            if len(output) >= limit:
                break
    return tuple(output)


def _relevant_captions(context: VideoProposalContext,
                       limit: int) -> tuple[dict[str, str], ...]:
    referenced: list[str] = []
    for row in context.baseline_qa_results:
        for segment_id in row.get("used_segments") or ():
            if segment_id not in referenced:
                referenced.append(str(segment_id))
    if not referenced:
        referenced = [str(key) for key in context.captions
                      if key not in ("subject_registry", "character_registry")]
    output = []
    for segment_id in referenced[:limit]:
        value = context.captions.get(segment_id)
        caption = (value.get("caption") if isinstance(value, Mapping) else value)
        if caption:
            output.append({"segment_id": segment_id, "caption": str(caption)})
    return tuple(output)


def build_proposal_request(
    context: VideoProposalContext,
    *,
    max_proposals: int,
    max_trace_events_per_qa: int,
    max_captions: int,
    max_payload_chars: int,
) -> dict[str, Any]:
    if len(context.baseline_qa_results) != 3:
        raise PropertyProposalError("proposal request requires all three baseline QAs")
    rows = sorted(
        context.baseline_qa_results,
        key=lambda row: (bool(row.get("is_correct")), str(row.get("question_id"))),
    )
    qas = []
    for priority, row in enumerate(rows, 1):
        qas.append({
            "priority_rank": priority,
            "question_id": str(row["question_id"]),
            "question": str(row.get("question") or ""),
            "answer_choices": tuple(row.get("options") or ()),
            "gold_answer": str(row.get("ground_truth") or ""),
            "baseline_prediction": row.get("prediction"),
            "is_correct": bool(row.get("is_correct")),
            "reasoning_trace": _read_trace(row, max_trace_events_per_qa),
            "used_segment_references": tuple(row.get("used_segments") or ()),
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
            "Propose zero or more concise reusable captioning properties. "
            "Prioritize incorrect QAs, inspect all three QAs for consistency "
            "and regression risk, and return only the strict output schema. "
            "Never mention an instance-specific video, question, answer, "
            "option, timestamp, or ground truth in property_text."
        ),
        "source_video_id": context.video_id,
        "baseline_run_id": context.baseline_run_id,
        "max_proposals": max_proposals,
        "qas": tuple(qas),
        "relevant_baseline_captions": _relevant_captions(context, max_captions),
        "active_codebook": active_codebook,
        "output_schema": {
            "proposals": ({
                "candidate_property_id": "string",
                "property_text": "concise reusable captioning instruction",
                "source_video_id": context.video_id,
                "source_question_ids": ["question_id"],
                "motivating_failure_types": ["snake_case_failure_type"],
                "covered_by_existing_property_ids": ["active_property_id"],
                "proposal_rationale": "brief rationale",
            },),
        },
    }
    if len(dumps_canonical(request)) > max_payload_chars:
        raise PropertyProposalError(
            f"proposal request exceeds max_payload_chars={max_payload_chars}")
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
    allowed_questions = {str(row["question_id"])
                         for row in context.baseline_qa_results}
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
        question_ids = row["source_question_ids"]
        failures = row["motivating_failure_types"]
        covered = row["covered_by_existing_property_ids"]
        rationale = row["proposal_rationale"]
        for name, items in (("source_question_ids", question_ids),
                            ("motivating_failure_types", failures),
                            ("covered_by_existing_property_ids", covered)):
            if not isinstance(items, list) or any(
                    not isinstance(item, str) or not item for item in items):
                raise PropertyProposalParseError(
                    f"proposal {candidate_id} {name} must be a string array")
        if not question_ids or not failures or \
                len(question_ids) != len(set(question_ids)) or \
                len(failures) != len(set(failures)) or \
                len(covered) != len(set(covered)):
            raise PropertyProposalParseError(
                f"proposal {candidate_id} requires unique lineage and failure types")
        if any(not re.fullmatch(r"[a-z][a-z0-9_]*", item)
               for item in failures):
            raise PropertyProposalParseError(
                f"proposal {candidate_id} failure types must be snake_case")
        if row["source_video_id"] != context.video_id:
            raise PropertyProposalParseError(
                f"proposal {candidate_id} source video mismatch")
        if not set(question_ids) <= allowed_questions:
            raise PropertyProposalParseError(
                f"proposal {candidate_id} references a non-source question")
        if not isinstance(rationale, str) or not rationale.strip():
            raise PropertyProposalParseError(
                f"proposal {candidate_id} rationale must be non-empty")
        unknown_coverage = set(covered) - set(active)
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
        computed_coverage = active_by_text.get(text_key)
        id_coverage = candidate_id if candidate_id in active else None
        coverage = tuple(dict.fromkeys(
            [*covered,
             *([computed_coverage] if computed_coverage else []),
             *([id_coverage] if id_coverage else [])]))
        if coverage:
            rejected.append({
                "candidate_property_id": candidate_id,
                "reason": "covered_by_active_codebook:" + ",".join(coverage),
            })
            continue
        proposals.append(CandidatePropertyProposal(
            candidate_property_id=candidate_id,
            property_text=text,
            source_video_id=context.video_id,
            source_question_ids=tuple(question_ids),
            motivating_failure_types=tuple(failures),
            covered_by_existing_property_ids=coverage,
            proposal_rationale=rationale.strip(),
            proposer_policy_version=policy_version,
        ))
    return tuple(proposals), tuple(rejected)


class MultiPropertyProposalPolicy:
    policy_version = PROPOSAL_POLICY_VERSION

    def __init__(
        self,
        *,
        response_provider: Callable[[str], str],
        artifact_root: str | None = None,
        max_proposals: int = config.MAX_PROPERTY_PROPOSALS_PER_VIDEO,
        max_payload_chars: int = config.PROPERTY_PROPOSAL_MAX_PAYLOAD_CHARS,
        max_trace_events_per_qa: int = config.PROPERTY_PROPOSAL_MAX_TRACE_EVENTS_PER_QA,
        max_captions: int = config.PROPERTY_PROPOSAL_MAX_CAPTIONS,
        max_property_text_chars: int = config.PROPERTY_PROPOSAL_MAX_TEXT_CHARS,
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
        }

    def propose(self, context: VideoProposalContext):
        artifact_dir = context.proposal_artifact_dir or (
            os.path.join(self.artifact_root, context.video_id)
            if self.artifact_root else None)
        if not artifact_dir:
            raise PropertyProposalError("proposal artifact directory is required")
        request = build_proposal_request(
            context, max_proposals=self.max_proposals,
            max_trace_events_per_qa=self.max_trace_events_per_qa,
            max_captions=self.max_captions,
            max_payload_chars=self.max_payload_chars)
        request_text = dumps_canonical(request)
        fingerprint = sha256_text(dumps_canonical({
            "request": request,
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
        raw_path = os.path.abspath(os.path.join(artifact_dir, "raw_output.txt"))
        parsed_path = os.path.abspath(os.path.join(artifact_dir, "parsed_output.json"))
        rejected_path = os.path.abspath(os.path.join(artifact_dir, "rejections.json"))
        _atomic_write_text(request_path, request_text + "\n")
        raw = self.response_provider(request_text)
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
            "parsed_output_path": parsed_path,
            "rejections_path": rejected_path,
            "proposal_count": len(proposals),
        }, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
        return proposals


class OpenAIPropertyProposalProvider:
    """Configured optimization-LLM provider; instantiated only for real runs."""

    def __init__(self, *, model: str = config.FEEDBACK_MODEL,
                 api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self.call_count = 0

    def __call__(self, request: str) -> str:
        self.call_count += 1
        prompt = "Return only strict JSON for this property proposal request:\n" + request
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }).encode()
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
            "response_format": "json_object",
            "real_model": True,
        }
