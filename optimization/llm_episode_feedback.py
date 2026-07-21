"""Checkpoint D1 request construction and strict episode-feedback boundary.

The module is deliberately runtime-only and read-only.  It resolves complete
saved QA/trajectory evidence, constructs one deterministic text request, and
parses one injected backend response.  It does not select evidence, truncate
payloads, call a configured model by itself, retry, or persist anything.
"""

from __future__ import annotations

import json
import hashlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from surrogate_rollout.optimization.episode_feedback import (
    classify_qa_transition,
)
from surrogate_rollout.optimization.schemas import (
    EpisodeFeedback,
    InterventionEpisode,
    episode_feedback_from_json,
    validate_episode_feedback,
)
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json, sha256_text


EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION = "episode_feedback_request_v1"
COMPACT_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION = "episode_feedback_request_v2"
MODEL_COMPACT_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION = \
    "episode_feedback_request_v4_grounded"
MODEL_COMPACT_EPISODE_FEEDBACK_AUDIT_SCHEMA_VERSION = \
    "episode_feedback_request_audit_v1"
EPISODE_FEEDBACK_RESPONSE_SCHEMA_VERSION = "episode_feedback_response_v2_grounded"

QA_TRANSITION_SUMMARY_ORDER = (
    "correct_to_correct", "wrong_to_wrong", "wrong_to_correct",
    "correct_to_wrong",
)


EPISODE_FEEDBACK_SYSTEM_INSTRUCTION = """Analyze one prompt-delta intervention episode and produce evidence-linked feedback for improving a visual- and history-conditioned prompt generator.

The runtime prompt generator can use only:
- current visual frames,
- bounded preceding caption history,
- the current meta-prompt.

It cannot use QA information, correctness labels, trajectories, external OCR, metadata, or external tools. Any recommended strategy must be executable using only the runtime inputs above.

Analyze:
- before/after caption changes across all selected clips;
- all sibling QA transitions;
- whether changed caption information was retrieved or used in stored trajectories;
- evidence supporting or limiting a generator-level strategy update.

Core rules:

1. Caption changes do not by themselves prove downstream utility.
2. Correct-to-correct and wrong-to-wrong outcomes are neutral with respect to improvement.
3. Claim downstream benefit only for:
   - wrong-to-correct QA transitions, or
   - explicit trajectory evidence showing that changed caption information was retrieved or used.
4. Do not attribute an episode-level QA outcome to one clip unless the trajectory explicitly links that QA and segment. Retrieved or referenced clips are not causal proof.
5. Do not claim that caption content matches the actual frames. Frames are not included in this feedback request. State only what changed between stored captions.
6. Counterevidence must genuinely weaken a claimed benefit, diagnosis, or recommendation. Valid examples include unchanged or harmful QA outcomes, lack of trajectory use, inconsistent effects, unsupported additions, or baseline captions already containing the information.
7. If no valid counterevidence exists, return an empty array.
8. If there is no QA improvement and no trajectory-linked evidence of utility, do not recommend a strategy update.
9. Do not copy the delta into a persistent rule, rewrite the meta-prompt, propose codebook entries, or recommend unavailable inputs or tools.
10. Treat all conclusions as local to this episode.
11. Copy QA transition facts exactly from qa_transition_summary.
12. Do not infer or recompute QA transitions from trajectories.
13. Do not state that all QAs share one transition unless the summary shows that.
14. Focus free-form diagnosis on caption behavior, trajectory evidence, and uncertainty.

Evidence ID rules:

- caption_change: use segment IDs only; QA IDs must be empty.
- qa_transition: use QA IDs only; segment IDs must be empty.
- trajectory: use both segment and QA IDs.
- mixed: use only when stored trajectory evidence genuinely links the segment and QA.
- Never attach sibling QA IDs to a segment merely because they belong to the same episode.
- No evidence item may leave both ID arrays empty.

Each evidence statement must begin with exactly:
- "Observed:" for directly stored facts;
- "Hypothesis:" for interpretations or possible mechanisms.

The generator_diagnosis must distinguish:
- observed caption behavior,
- observed QA outcomes,
- trajectory-linked utility,
- uncertain episode-level credit assignment.

The recommended_strategy_change must be conditional, brief, and executable using only frames, bounded history, and the meta-prompt.

When there is no QA improvement and no trajectory-linked evidence of utility, return exactly:

"Insufficient evidence to update the generator strategy; retain the current strategy pending confirmation from additional episodes."

Avoid claims such as "ensures," "successfully improves," or "systematically solves."

Length limits:

- at most 4 observations;
- at most 3 counterevidence items;
- each evidence statement: one sentence;
- outcome_summary: at most 3 sentences;
- generator_diagnosis: at most 5 sentences;
- recommended_strategy_change: at most 3 sentences and 120 words;
- confidence: one short sentence;
- do not repeat the same evidence or recommendation across fields.

Confidence is an opaque description of evidence strength and attribution uncertainty, not a probability.

Return only one strict JSON object with exactly:
- episode_id
- outcome_summary
- observations
- counterevidence
- generator_diagnosis
- recommended_strategy_change
- confidence

Each observation and counterevidence item must contain exactly:
- statement
- supporting_segment_ids
- supporting_qa_ids
- evidence_type
- transition_type
- confidence

evidence_type must be exactly one of:
- caption_change
- trajectory
- qa_transition
- mixed

For qa_transition evidence, transition_type must be exactly one of the four
keys in qa_transition_summary and every supporting QA ID must occur under that
same key. Split different transition types into separate evidence items. For
all other evidence types, transition_type must be null.

Do not return feedback_id. Do not use Markdown fences or trailing prose."""

COMPACT_EPISODE_FEEDBACK_SYSTEM_INSTRUCTION = (
    EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
    + "\n\nHistory items are stored once in history_catalog and referenced by "
      "ordered IDs from each clip. Reconstruct each clip's history "
      "conceptually in the listed order. A repeated ID is a repeated history "
      "occurrence.\n\nTrajectory feedback views omit agent boilerplate and "
      "duplicate transport wrappers, but retain executed tool events, "
      "retrieved evidence, complete reference sets, assistant analysis steps "
      "stored in the trace, unclassified messages, and final responses. Full "
      "raw trajectory references and hashes are provided for audit. Do not "
      "infer that omitted boilerplate was evidence."
)

MODEL_COMPACT_EPISODE_FEEDBACK_SYSTEM_INSTRUCTION = (
    EPISODE_FEEDBACK_SYSTEM_INSTRUCTION
    + "\n\nHistory items are stored once in history_catalog and referenced by "
      "ordered IDs from each clip. Reconstruct each clip's history "
      "conceptually in the listed order. A repeated ID is a repeated history "
      "occurrence.\n\nTrajectory feedback views omit agent boilerplate, "
      "duplicate transport wrappers, filesystem references, hashes, and "
      "projection statistics. They retain executed tool events, exact tool "
      "arguments and returned evidence, complete reference sets, assistant "
      "analysis steps stored in the trace, unclassified messages, and final "
      "responses. Audit metadata is retained separately and is not supplied "
      "as evidence. Do not infer that omitted boilerplate or audit metadata "
      "was evidence."
)


class EpisodeFeedbackRequestError(ValueError):
    """Saved episode evidence cannot be resolved without fabrication."""


class EpisodeFeedbackBackendConfigurationError(ValueError):
    """The injected backend omits required explicit configuration metadata."""


class EpisodeFeedbackParseError(ValueError):
    """Strict response parsing failed while preserving the raw response."""

    def __init__(self, reason: str, *, raw_response: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw_response = raw_response


class EpisodeFeedbackContextOverflowError(ValueError):
    """A complete request exceeds a known backend context limit."""

    def __init__(
        self, *, observed_tokens: int, configured_limit: int,
        clip_count: int, qa_count: int,
        largest_payload_sections: tuple[tuple[str, int], ...],
    ) -> None:
        self.observed_tokens = observed_tokens
        self.configured_limit = configured_limit
        self.clip_count = clip_count
        self.qa_count = qa_count
        self.largest_payload_sections = largest_payload_sections
        super().__init__(
            "episode feedback request exceeds configured context limit: "
            f"observed_tokens={observed_tokens}, limit={configured_limit}, "
            f"clip_count={clip_count}, qa_count={qa_count}, "
            f"largest_payload_sections={largest_payload_sections!r}; "
            "no truncation, sampling, summarization, or splitting was performed")


class EpisodeFeedbackArtifactResolver(Protocol):
    def resolve_qas(
        self, episode: InterventionEpisode,
    ) -> tuple[Mapping[str, Any], ...]:
        ...


class EpisodeFeedbackResponseProvider(Protocol):
    def __call__(self, system_instruction: str, request: str) -> str:
        ...

    def metadata(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class EpisodeFeedbackRequestSize:
    clip_count: int
    qa_count: int
    total_history_item_count: int
    serialized_request_character_count: int
    trajectory_character_count: int
    clip_record_character_count: int
    qa_record_character_count: int
    token_count: int | None
    context_limit_tokens: int | None
    context_limit_checked: bool
    unresolved_reference_count: int
    largest_payload_sections: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class EpisodeFeedbackRequest:
    system_instruction: str
    user_payload: Mapping[str, Any]
    messages: tuple[Mapping[str, str], ...]
    payload_hash: str
    size_statistics: EpisodeFeedbackRequestSize

    @property
    def user_request(self) -> str:
        return self.messages[1]["content"]


@dataclass(frozen=True)
class EpisodeFeedbackInvocationResult:
    request: (
        EpisodeFeedbackRequest
        | CompactEpisodeFeedbackRequest
        | ModelCompactEpisodeFeedbackRequest
    )
    raw_response: str
    feedback: EpisodeFeedback
    policy_version: str
    backend_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CompactEpisodeFeedbackRequestSize:
    clip_count: int
    qa_count: int
    complete_request_character_count: int
    compact_request_character_count: int
    history_character_count_before_deduplication: int
    history_catalog_character_count: int
    history_reference_character_count: int
    unique_history_item_count: int
    total_history_item_occurrences: int
    raw_trajectory_character_count: int
    compact_trajectory_character_count: int
    removed_system_tool_schema_character_count: int | None
    removed_duplicate_wrapper_character_count: int | None
    repeated_prompt_delta_character_count_removed: int
    complete_payload_hash: str
    compact_payload_hash: str
    unclassified_trajectory_message_count: int
    unresolved_reference_count: int
    token_count: int | None


@dataclass(frozen=True)
class CompactEpisodeFeedbackRequest:
    system_instruction: str
    user_payload: Mapping[str, Any]
    messages: tuple[Mapping[str, str], ...]
    payload_hash: str
    size_statistics: CompactEpisodeFeedbackRequestSize

    @property
    def user_request(self) -> str:
        return self.messages[1]["content"]


@dataclass(frozen=True)
class ModelCompactEpisodeFeedbackRequestSize:
    clip_count: int
    qa_count: int
    complete_request_character_count: int
    compact_request_character_count: int
    model_request_character_count: int
    model_payload_character_count: int
    audit_metadata_character_count: int
    history_catalog_character_count: int
    history_reference_character_count: int
    unique_history_item_count: int
    total_history_item_occurrences: int
    raw_trajectory_character_count: int
    compact_trajectory_character_count: int
    model_trajectory_character_count: int
    complete_payload_hash: str
    compact_payload_hash: str
    model_payload_hash: str
    audit_metadata_hash: str
    unclassified_trajectory_message_count: int
    unresolved_reference_count: int
    token_count: int | None


@dataclass(frozen=True)
class ModelCompactEpisodeFeedbackRequest:
    """D1.6 request with model-visible evidence separated from audit data."""

    system_instruction: str
    model_payload: Mapping[str, Any]
    audit_metadata: Mapping[str, Any]
    model_payload_hash: str
    audit_metadata_hash: str
    messages: tuple[Mapping[str, str], ...]
    size_statistics: ModelCompactEpisodeFeedbackRequestSize

    @property
    def user_payload(self) -> Mapping[str, Any]:
        """Compatibility alias for request consumers; contains model data only."""
        return self.model_payload

    @property
    def payload_hash(self) -> str:
        """Feedback identity input; audit-only values are intentionally absent."""
        return self.model_payload_hash

    @property
    def user_request(self) -> str:
        return self.messages[1]["content"]


def _read_json_object(path: str, where: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeFeedbackRequestError(
            f"cannot read {where} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EpisodeFeedbackRequestError(f"{where} must be a JSON object")
    return value


def _read_jsonl_objects(path: str, where: str) -> tuple[dict[str, Any], ...]:
    rows = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise EpisodeFeedbackRequestError(
                        f"{where} row {line_number} must be a JSON object")
                rows.append(value)
    except EpisodeFeedbackRequestError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeFeedbackRequestError(
            f"cannot read {where} at {path}: {exc}") from exc
    return tuple(rows)


def _artifact_path(
    owner_path: str, record: Mapping[str, Any], key: str, where: str,
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise EpisodeFeedbackRequestError(
            f"{where}.{key} must identify an existing artifact")
    path = value if os.path.isabs(value) else os.path.join(
        os.path.dirname(owner_path), value)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise EpisodeFeedbackRequestError(
            f"{where}.{key} does not resolve to an existing artifact: {path}")
    return path


def _index_rows(
    rows: Sequence[Mapping[str, Any]], key: str, where: str,
) -> dict[str, Mapping[str, Any]]:
    output = {}
    for index, row in enumerate(rows):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise EpisodeFeedbackRequestError(
                f"{where}[{index}].{key} must be a non-empty string")
        if value in output:
            raise EpisodeFeedbackRequestError(
                f"duplicate {key} {value!r} in {where}")
        output[value] = row
    return output


def _reference_sets(
    row: Mapping[str, Any], trajectory_ref: str | None,
) -> tuple[dict[str, tuple[str, ...]], tuple[dict[str, Any], ...]]:
    value = row.get("reference_sets")
    sibling_value = None
    if trajectory_ref is not None:
        sibling = os.path.join(os.path.dirname(trajectory_ref), "references.json")
        if os.path.isfile(sibling):
            sibling_value = _read_json_object(sibling, "trajectory references")
    if value is not None and not isinstance(value, Mapping):
        raise EpisodeFeedbackRequestError("QA reference_sets must be an object")
    if sibling_value is not None:
        merged = dict(sibling_value)
        for key, items in (value or {}).items():
            if key in merged and json.loads(dumps_canonical(merged[key])) != \
                    json.loads(dumps_canonical(items)):
                raise EpisodeFeedbackRequestError(
                    f"QA reference_sets.{key} conflicts with references.json")
            merged[key] = items
        value = merged
    if value is None:
        return {}, ()
    output = {}
    evidence: tuple[dict[str, Any], ...] = ()
    for key, items in value.items():
        if key == "evidence":
            if not isinstance(items, (list, tuple)) or any(
                    not isinstance(item, Mapping) for item in items):
                raise EpisodeFeedbackRequestError(
                    "QA reference_sets.evidence must be an object array")
            evidence = tuple(json.loads(dumps_canonical(item)) for item in items)
            continue
        if not isinstance(items, (list, tuple)) or any(
                not isinstance(item, str) or not item for item in items):
            raise EpisodeFeedbackRequestError(
                f"QA reference_sets.{key} must be a string array")
        output[str(key)] = tuple(items)
    return output, evidence


def _same_reference(stored: Any, episode_reference: str | None) -> bool:
    if stored is None or episode_reference is None:
        return stored is None and episode_reference is None
    return isinstance(stored, str) and bool(stored) and \
        os.path.abspath(stored) == os.path.abspath(episode_reference)


def _resolve_trajectory(
    reference: str | None, reference_sets: Mapping[str, tuple[str, ...]],
    reference_evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if reference is None:
        if any(reference_sets.values()) or reference_evidence:
            raise EpisodeFeedbackRequestError(
                "unavailable trajectory conflicts with reference provenance")
        return {
            "availability": "unavailable",
            "content": None,
            "tool_events": [],
            "reference_sets": {},
            "reference_evidence": [],
            "referenced_segment_ids": [],
            "retrieved_segment_ids": [],
        }
    if not isinstance(reference, str) or not reference or not os.path.isfile(reference):
        raise EpisodeFeedbackRequestError(
            f"trajectory reference does not resolve: {reference!r}")
    content = _read_jsonl_objects(reference, "trajectory")
    tool_events_path = os.path.join(os.path.dirname(reference), "tool_events.jsonl")
    tool_events = (_read_jsonl_objects(tool_events_path, "trajectory tool events")
                   if os.path.isfile(tool_events_path) else ())
    return {
        "availability": "available",
        "content": list(content),
        "tool_events": list(tool_events),
        "reference_sets": {key: list(items)
                           for key, items in reference_sets.items()},
        "reference_evidence": [
            json.loads(dumps_canonical(item)) for item in reference_evidence],
        "referenced_segment_ids": list(
            reference_sets.get("explicitly_cited_segments", ())),
        "retrieved_segment_ids": list(
            reference_sets.get("retrieved_segments", ())),
    }


class LegacyEpisodeFeedbackArtifactResolver:
    """Resolve QA metadata and raw trajectories from Checkpoint B run refs."""

    def resolve_qas(
        self, episode: InterventionEpisode,
    ) -> tuple[Mapping[str, Any], ...]:
        baseline_path = os.path.abspath(episode.baseline_run_ref)
        intervention_path = os.path.abspath(episode.intervention_run_ref)
        baseline = _read_json_object(baseline_path, "baseline video manifest")
        intervention = _read_json_object(
            intervention_path, "intervention result")
        if baseline.get("video_id") != episode.video_id or \
                intervention.get("source_video_id") != episode.video_id:
            raise EpisodeFeedbackRequestError(
                "episode video lineage conflicts with saved run references")
        baseline_qas_path = _artifact_path(
            baseline_path, baseline, "baseline_qas_path", "baseline manifest")
        transitions_path = _artifact_path(
            intervention_path, intervention, "transitions_path",
            "intervention result")
        baseline_rows = _index_rows(
            _read_jsonl_objects(baseline_qas_path, "baseline QAs"),
            "question_id", "baseline QAs")
        transitions_value = _read_json_object(
            transitions_path, "intervention transitions").get("qas")
        if not isinstance(transitions_value, list):
            raise EpisodeFeedbackRequestError(
                "intervention transitions.qas must be an array")
        transition_rows = _index_rows(
            transitions_value, "question_id", "intervention transitions.qas")

        output = []
        for outcome in episode.qa_outcomes:
            if outcome.qa_id not in baseline_rows or \
                    outcome.qa_id not in transition_rows:
                raise EpisodeFeedbackRequestError(
                    f"saved QA metadata is missing episode QA {outcome.qa_id!r}")
            before = baseline_rows[outcome.qa_id]
            after = transition_rows[outcome.qa_id]
            question = before.get("question")
            if not isinstance(question, str) or not question:
                raise EpisodeFeedbackRequestError(
                    f"baseline QA {outcome.qa_id!r} has no stored question")
            choices = before.get("options", ())
            if not isinstance(choices, (list, tuple)) or any(
                    not isinstance(item, str) for item in choices):
                raise EpisodeFeedbackRequestError(
                    f"baseline QA {outcome.qa_id!r} options are invalid")
            gold = before.get("ground_truth", after.get("ground_truth"))
            if gold is not None and not isinstance(gold, str):
                raise EpisodeFeedbackRequestError(
                    f"baseline QA {outcome.qa_id!r} ground truth is invalid")
            expected = (
                before.get("prediction"), after.get("candidate_prediction"),
                before.get("is_correct"), after.get("candidate_correct"),
            )
            actual = (
                outcome.baseline_answer, outcome.intervention_answer,
                outcome.baseline_correct, outcome.intervention_correct,
            )
            if expected != actual:
                raise EpisodeFeedbackRequestError(
                    f"saved QA outcome conflicts with episode QA {outcome.qa_id!r}")
            if not _same_reference(
                    before.get("trajectory_path"),
                    outcome.baseline_trajectory_ref) or not _same_reference(
                    after.get("trajectory_path"),
                    outcome.intervention_trajectory_ref):
                raise EpisodeFeedbackRequestError(
                    f"saved trajectory reference conflicts for QA {outcome.qa_id!r}")
            before_refs, before_reference_evidence = _reference_sets(
                before, outcome.baseline_trajectory_ref)
            after_refs, after_reference_evidence = _reference_sets(
                after, outcome.intervention_trajectory_ref)
            output.append({
                "qa_id": outcome.qa_id,
                "is_source_qa": outcome.is_source_qa,
                "question": question,
                "answer_choices": list(choices),
                "gold_answer": gold,
                "baseline_answer": outcome.baseline_answer,
                "intervention_answer": outcome.intervention_answer,
                "baseline_correct": outcome.baseline_correct,
                "intervention_correct": outcome.intervention_correct,
                "transition": classify_qa_transition(outcome),
                "baseline_trajectory": _resolve_trajectory(
                    outcome.baseline_trajectory_ref, before_refs,
                    before_reference_evidence),
                "intervention_trajectory": _resolve_trajectory(
                    outcome.intervention_trajectory_ref, after_refs,
                    after_reference_evidence),
            })
        return tuple(output)


class FreshEpisodeFeedbackArtifactResolver:
    """Resolve QA evidence from a fresh prompt-delta intervention manifest."""

    def resolve_qas(
        self, episode: InterventionEpisode,
    ) -> tuple[Mapping[str, Any], ...]:
        baseline_path = os.path.abspath(episode.baseline_run_ref)
        intervention_path = os.path.abspath(episode.intervention_run_ref)
        baseline = _read_json_object(baseline_path, "baseline video manifest")
        intervention = _read_json_object(
            intervention_path, "fresh intervention manifest")
        if baseline.get("video_id") != episode.video_id or \
                intervention.get("video_id") != episode.video_id or \
                intervention.get("schema_version") != \
                "fresh_prompt_delta_intervention_v1":
            raise EpisodeFeedbackRequestError(
                "fresh episode video/schema lineage conflicts with run references")
        baseline_path_qas = _artifact_path(
            baseline_path, baseline, "baseline_qas_path", "baseline manifest")
        before_rows = _index_rows(
            _read_jsonl_objects(baseline_path_qas, "baseline QAs"),
            "question_id", "baseline QAs")
        after_rows = _index_rows(
            intervention.get("qa_records") or (), "qa_id",
            "fresh intervention qa_records")
        output = []
        for outcome in episode.qa_outcomes:
            before = before_rows.get(outcome.qa_id)
            after = after_rows.get(outcome.qa_id)
            if before is None or after is None:
                raise EpisodeFeedbackRequestError(
                    f"fresh saved QA metadata is missing {outcome.qa_id!r}")
            if (before.get("prediction"), before.get("is_correct"),
                    after.get("intervention_answer"),
                    after.get("intervention_correct")) != (
                    outcome.baseline_answer, outcome.baseline_correct,
                    outcome.intervention_answer, outcome.intervention_correct):
                raise EpisodeFeedbackRequestError(
                    f"fresh QA outcome conflicts for {outcome.qa_id!r}")
            if not _same_reference(before.get("trajectory_path"),
                                   outcome.baseline_trajectory_ref) or not \
                    _same_reference(after.get("intervention_trajectory_path"),
                                    outcome.intervention_trajectory_ref):
                raise EpisodeFeedbackRequestError(
                    f"fresh trajectory reference conflicts for {outcome.qa_id!r}")
            question = before.get("question")
            choices = before.get("options") or ()
            if not isinstance(question, str) or not question or not isinstance(
                    choices, (list, tuple)) or any(
                        not isinstance(item, str) for item in choices):
                raise EpisodeFeedbackRequestError(
                    f"fresh baseline QA metadata is invalid: {outcome.qa_id!r}")
            before_refs, before_reference_evidence = _reference_sets(
                before, outcome.baseline_trajectory_ref)
            after_refs, after_reference_evidence = _reference_sets(
                {}, outcome.intervention_trajectory_ref)
            output.append({
                "qa_id": outcome.qa_id,
                "is_source_qa": outcome.is_source_qa,
                "question": question, "answer_choices": list(choices),
                "gold_answer": before.get("ground_truth"),
                "baseline_answer": outcome.baseline_answer,
                "intervention_answer": outcome.intervention_answer,
                "baseline_correct": outcome.baseline_correct,
                "intervention_correct": outcome.intervention_correct,
                "transition": classify_qa_transition(outcome),
                "baseline_trajectory": _resolve_trajectory(
                    outcome.baseline_trajectory_ref, before_refs,
                    before_reference_evidence),
                "intervention_trajectory": _resolve_trajectory(
                    outcome.intervention_trajectory_ref, after_refs,
                    after_reference_evidence),
            })
        return tuple(output)


class SavedEpisodeFeedbackArtifactResolver:
    """Dispatch without repurposing either legacy or fresh artifact schema."""

    def __init__(self) -> None:
        self.legacy = LegacyEpisodeFeedbackArtifactResolver()
        self.fresh = FreshEpisodeFeedbackArtifactResolver()

    def resolve_qas(self, episode: InterventionEpisode):
        value = _read_json_object(
            os.path.abspath(episode.intervention_run_ref),
            "saved intervention manifest")
        if value.get("schema_version") == "fresh_prompt_delta_intervention_v1":
            return self.fresh.resolve_qas(episode)
        return self.legacy.resolve_qas(episode)


def _history_count(snapshot: Any) -> int:
    if isinstance(snapshot, Mapping) and "history" in snapshot:
        history = snapshot["history"]
        if not isinstance(history, (list, tuple)):
            raise EpisodeFeedbackRequestError(
                "history_snapshot.history must be an array")
        return len(history)
    if isinstance(snapshot, (list, tuple)):
        return len(snapshot)
    return 0


def _trajectory_characters(qas: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for qa in qas:
        for key in ("baseline_trajectory", "intervention_trajectory"):
            trajectory = qa[key]
            if trajectory["availability"] == "available":
                total += len(dumps_canonical(trajectory["content"]))
                total += len(dumps_canonical(trajectory["tool_events"]))
    return total


def build_qa_transition_summary(
    episode: InterventionEpisode,
) -> dict[str, list[str]]:
    """Return canonical stored correctness facts without trajectory inference."""
    summary = {name: [] for name in QA_TRANSITION_SUMMARY_ORDER}
    for outcome in episode.qa_outcomes:
        summary[classify_qa_transition(outcome)].append(outcome.qa_id)
    return summary


def build_episode_feedback_request(
    episode: InterventionEpisode,
    *,
    artifact_resolver: EpisodeFeedbackArtifactResolver,
    token_counter: Callable[[tuple[Mapping[str, str], ...]], int] | None = None,
    context_limit_tokens: int | None = None,
) -> EpisodeFeedbackRequest:
    """Build the complete deterministic request without filtering or writes."""
    if not isinstance(episode, InterventionEpisode):
        raise TypeError("episode must be an InterventionEpisode")
    if artifact_resolver is None:
        raise TypeError("artifact_resolver is required")
    if context_limit_tokens is not None and (
            not isinstance(context_limit_tokens, int) or
            isinstance(context_limit_tokens, bool) or context_limit_tokens <= 0):
        raise EpisodeFeedbackBackendConfigurationError(
            "context_limit_tokens must be a positive integer or None")
    if context_limit_tokens is not None and token_counter is None:
        raise EpisodeFeedbackBackendConfigurationError(
            "a known token context limit requires the backend token counter")

    mismatched_delta_segments = tuple(
        clip.segment_id for clip in episode.clips
        if clip.prompt_delta.instruction != episode.prompt_delta.instruction)
    if mismatched_delta_segments:
        raise EpisodeFeedbackRequestError(
            "clip prompt delta instruction conflicts with authoritative episode "
            f"instruction for segments {mismatched_delta_segments!r}")

    clips = [{
        "segment_id": clip.segment_id,
        "time_range": clip.time_range,
        "history_snapshot": clip.history_snapshot,
        "base_prompt": clip.base_prompt,
        "applied_prompt_delta_instruction": clip.prompt_delta.instruction,
        "baseline_caption": clip.baseline_caption,
        "intervention_caption": clip.intervention_caption,
    } for clip in episode.clips]
    qas = list(artifact_resolver.resolve_qas(episode))
    if len(qas) != len(episode.qa_outcomes):
        raise EpisodeFeedbackRequestError(
            "artifact resolver did not return every episode QA exactly once")
    resolved_qa_ids = tuple(
        item.get("qa_id") if isinstance(item, Mapping) else None for item in qas)
    episode_qa_ids = tuple(item.qa_id for item in episode.qa_outcomes)
    if resolved_qa_ids != episode_qa_ids:
        raise EpisodeFeedbackRequestError(
            "artifact resolver changed the canonical episode QA order or identity")
    payload = {
        "schema_version": EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION,
        "episode": {
            "episode_id": episode.episode_id,
            "video_id": episode.video_id,
            "parent_meta_prompt_id": episode.parent_meta_prompt_id,
            "prompt_delta": {
                "delta_id": episode.prompt_delta.delta_id,
                "instruction": episode.prompt_delta.instruction,
                "source_qa_ids": list(episode.prompt_delta.source_qa_ids),
                "proposer_diagnosis": episode.prompt_delta.proposer_diagnosis,
            },
            "qa_transition_summary": build_qa_transition_summary(episode),
            "clips": clips,
            "qas": qas,
        },
    }
    user_request = dumps_canonical(payload)
    canonical_payload = json.loads(user_request)
    messages = (
        {"role": "system", "content": EPISODE_FEEDBACK_SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_request},
    )
    messages = tuple(messages)
    token_count = token_counter(messages) if token_counter is not None else None
    if token_count is not None and (
            not isinstance(token_count, int) or isinstance(token_count, bool) or
            token_count < 0):
        raise EpisodeFeedbackBackendConfigurationError(
            "backend token counter must return a non-negative integer")
    section_sizes = {
        "episode_metadata": len(dumps_canonical({
            key: value for key, value in payload["episode"].items()
            if key not in ("clips", "qas")})),
        "clips": len(dumps_canonical(clips)),
        "qas": len(dumps_canonical(qas)),
        "system_instruction": len(EPISODE_FEEDBACK_SYSTEM_INSTRUCTION),
    }
    largest_sections = tuple(sorted(
        section_sizes.items(), key=lambda item: (-item[1], item[0])))
    stats = EpisodeFeedbackRequestSize(
        clip_count=len(clips),
        qa_count=len(qas),
        total_history_item_count=sum(
            _history_count(clip.history_snapshot) for clip in episode.clips),
        serialized_request_character_count=len(dumps_canonical(messages)),
        trajectory_character_count=_trajectory_characters(qas),
        clip_record_character_count=sum(
            len(dumps_canonical(item)) for item in clips),
        qa_record_character_count=sum(
            len(dumps_canonical(item)) for item in qas),
        token_count=token_count,
        context_limit_tokens=context_limit_tokens,
        context_limit_checked=(context_limit_tokens is not None),
        unresolved_reference_count=0,
        largest_payload_sections=largest_sections,
    )
    if context_limit_tokens is not None and token_count is not None and \
            token_count > context_limit_tokens:
        raise EpisodeFeedbackContextOverflowError(
            observed_tokens=token_count,
            configured_limit=context_limit_tokens,
            clip_count=len(clips), qa_count=len(qas),
            largest_payload_sections=largest_sections)
    return EpisodeFeedbackRequest(
        system_instruction=EPISODE_FEEDBACK_SYSTEM_INSTRUCTION,
        user_payload=canonical_payload,
        messages=messages,
        payload_hash=sha256_json(canonical_payload),
        size_statistics=stats,
    )


_SEQUENTIAL_HISTORY_DERIVED_FIELDS = {
    "history", "preceding_captions", "preceding_segment_ids",
    "serialized_history", "history_hash",
}
_SEQUENTIAL_HISTORY_SERIALIZED_FIELDS = (
    "schema_version", "block_index", "block_start_seconds",
    "block_end_seconds", "max_history_captions", "preceding_captions",
)


def _plain_json(value: Any) -> Any:
    return json.loads(dumps_canonical(value))


def _history_item_id(item: Mapping[str, Any]) -> str:
    segment_id = item.get("segment_id")
    if isinstance(segment_id, str) and segment_id:
        return segment_id
    return "history_item_" + sha256_json(_plain_json(item))[:20]


def _compact_histories(
    episode: InterventionEpisode,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    catalog = []
    content_by_id: dict[str, Any] = {}
    clips = []
    before_chars = 0
    reference_chars = 0
    for clip in episode.clips:
        snapshot = _plain_json(clip.history_snapshot)
        if not isinstance(snapshot, dict):
            raise EpisodeFeedbackRequestError(
                f"clip {clip.segment_id!r} history snapshot must be an object")
        history = snapshot.get("history")
        if not isinstance(history, list) or any(
                not isinstance(item, dict) for item in history):
            raise EpisodeFeedbackRequestError(
                f"clip {clip.segment_id!r} history must be an object array")
        source = snapshot.get("source")
        if source == "sequential_history_aware_baseline":
            if snapshot.get("preceding_captions") != history:
                raise EpisodeFeedbackRequestError(
                    f"clip {clip.segment_id!r} preceding captions conflict with history")
            expected_ids = [item.get("segment_id") for item in history]
            if any(not isinstance(item, str) or not item for item in expected_ids) or \
                    snapshot.get("preceding_segment_ids") != expected_ids:
                raise EpisodeFeedbackRequestError(
                    f"clip {clip.segment_id!r} preceding segment IDs conflict")
            try:
                serialized_value = {
                    key: (history if key == "preceding_captions" else snapshot[key])
                    for key in _SEQUENTIAL_HISTORY_SERIALIZED_FIELDS
                }
            except KeyError as exc:
                raise EpisodeFeedbackRequestError(
                    f"clip {clip.segment_id!r} history snapshot is missing {exc.args[0]!r}") \
                    from exc
            expected_serialized = dumps_canonical(serialized_value)
            if snapshot.get("serialized_history") != expected_serialized or \
                    snapshot.get("history_hash") != sha256_text(expected_serialized):
                raise EpisodeFeedbackRequestError(
                    f"clip {clip.segment_id!r} serialized history invariant failed")
            projection = "sequential_history_aware_baseline_v1"
            metadata = {
                key: value for key, value in snapshot.items()
                if key not in _SEQUENTIAL_HISTORY_DERIVED_FIELDS
            }
        elif source == "frozen_incumbent_caption_view":
            expected_ids = [item.get("segment_id") for item in history]
            if snapshot.get("preceding_segment_ids") != expected_ids:
                raise EpisodeFeedbackRequestError(
                    f"clip {clip.segment_id!r} preceding segment IDs conflict")
            projection = "frozen_incumbent_caption_view_v1"
            metadata = {
                key: value for key, value in snapshot.items()
                if key not in {"history", "preceding_segment_ids"}
            }
        else:
            raise EpisodeFeedbackRequestError(
                f"clip {clip.segment_id!r} uses unsupported history source {source!r}")

        history_ids = []
        for item in history:
            item_id = _history_item_id(item)
            content = _plain_json(item)
            previous = content_by_id.get(item_id)
            if previous is not None and dumps_canonical(previous) != \
                    dumps_canonical(content):
                raise EpisodeFeedbackRequestError(
                    f"history item ID collision for {item_id!r}")
            if previous is None:
                content_by_id[item_id] = content
                catalog.append({"history_item_id": item_id, "content": content})
            history_ids.append(item_id)
        before_chars += len(dumps_canonical(history))
        reference_chars += len(dumps_canonical(history_ids))
        clips.append({
            "segment_id": clip.segment_id,
            "time_range": _plain_json(clip.time_range),
            "history_snapshot_projection": projection,
            "history_snapshot_metadata": metadata,
            "history_item_ids": history_ids,
            "base_prompt": clip.base_prompt,
            "baseline_caption": clip.baseline_caption,
            "intervention_caption": clip.intervention_caption,
        })
    return catalog, clips, before_chars, reference_chars


def reconstruct_compact_history_snapshots(
    compact_payload: Mapping[str, Any],
    audit_metadata: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Losslessly reconstruct histories from D1.5 or split D1.6 data."""
    try:
        episode = compact_payload["episode"]
        catalog_rows = episode["history_catalog"]
        clips = episode["clips"]
    except (KeyError, TypeError) as exc:
        raise EpisodeFeedbackRequestError(
            "compact payload is missing history catalog structure") from exc
    if not isinstance(catalog_rows, list) or not isinstance(clips, list):
        raise EpisodeFeedbackRequestError(
            "compact history catalog and clips must be arrays")
    catalog = {}
    for index, row in enumerate(catalog_rows):
        if not isinstance(row, Mapping) or set(row) != {
                "history_item_id", "content"}:
            raise EpisodeFeedbackRequestError(
                f"history_catalog[{index}] has invalid fields")
        item_id = row["history_item_id"]
        if not isinstance(item_id, str) or not item_id or item_id in catalog:
            raise EpisodeFeedbackRequestError(
                f"history_catalog[{index}] has invalid or duplicate ID")
        catalog[item_id] = _plain_json(row["content"])

    audit_rows = None
    if audit_metadata is not None:
        try:
            audit_rows = audit_metadata["history_reconstruction"]
        except (KeyError, TypeError) as exc:
            raise EpisodeFeedbackRequestError(
                "audit metadata is missing history reconstruction data") from exc
        if not isinstance(audit_rows, list) or len(audit_rows) != len(clips):
            raise EpisodeFeedbackRequestError(
                "audit history reconstruction rows must match compact clips")

    output = []
    for index, clip in enumerate(clips):
        if not isinstance(clip, Mapping):
            raise EpisodeFeedbackRequestError(f"clips[{index}] must be an object")
        item_ids = clip.get("history_item_ids")
        if not isinstance(item_ids, list) or any(
                not isinstance(item, str) or item not in catalog
                for item in item_ids):
            raise EpisodeFeedbackRequestError(
                f"clips[{index}] has an invalid history item sequence")
        history = [_plain_json(catalog[item]) for item in item_ids]
        if "history_snapshot_metadata" in clip or \
                "history_snapshot_projection" in clip:
            metadata_value = clip.get("history_snapshot_metadata")
            projection = clip.get("history_snapshot_projection")
        elif audit_rows is not None:
            audit_row = audit_rows[index]
            if not isinstance(audit_row, Mapping) or \
                    audit_row.get("segment_id") != clip.get("segment_id"):
                raise EpisodeFeedbackRequestError(
                    f"audit history row {index} does not match compact clip")
            metadata_value = audit_row.get("history_snapshot_metadata")
            projection = audit_row.get("history_snapshot_projection")
        else:
            raise EpisodeFeedbackRequestError(
                f"clips[{index}] requires audit history reconstruction data")
        metadata = _plain_json(metadata_value)
        if not isinstance(metadata, dict):
            raise EpisodeFeedbackRequestError(
                f"clips[{index}] history metadata must be an object")
        snapshot = dict(metadata)
        if projection == "sequential_history_aware_baseline_v1":
            snapshot["preceding_captions"] = history
            snapshot["preceding_segment_ids"] = [
                item["segment_id"] for item in history]
            snapshot["history"] = history
            serialized_value = {
                key: (history if key == "preceding_captions" else snapshot[key])
                for key in _SEQUENTIAL_HISTORY_SERIALIZED_FIELDS
            }
            serialized = dumps_canonical(serialized_value)
            snapshot["serialized_history"] = serialized
            snapshot["history_hash"] = sha256_text(serialized)
        elif projection == "frozen_incumbent_caption_view_v1":
            snapshot["preceding_segment_ids"] = [
                item["segment_id"] for item in history]
            snapshot["history"] = history
        else:
            raise EpisodeFeedbackRequestError(
                f"clips[{index}] has unknown history projection {projection!r}")
        output.append(snapshot)
    return tuple(output)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_call_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        raise EpisodeFeedbackRequestError(
            "stored tool-call arguments must be a JSON string")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise EpisodeFeedbackRequestError(
            "stored tool-call arguments are not strict JSON") from exc


def _hit_segment_id(hit: Mapping[str, Any]) -> str | None:
    start = hit.get("time_start_secs")
    end = hit.get("time_end_secs")
    if not isinstance(start, (int, float)) or isinstance(start, bool) or \
            not isinstance(end, (int, float)) or isinstance(end, bool):
        return None
    start_text = str(int(start)) if float(start).is_integer() else f"{start:g}"
    end_text = str(int(end)) if float(end).is_integer() else f"{end:g}"
    return f"{start_text}_{end_text}"


def _with_exact_compact_character_count(view: dict[str, Any]) -> dict[str, Any]:
    value = 0
    for _ in range(8):
        view["projection_statistics"]["compact_character_count"] = value
        observed = len(dumps_canonical(view))
        if observed == value:
            return view
        value = observed
    raise EpisodeFeedbackRequestError(
        "compact trajectory character count did not stabilize")


def _compact_trajectory(
    resolved: Mapping[str, Any], reference: str | None,
) -> dict[str, Any]:
    if resolved.get("availability") == "unavailable":
        if reference is not None:
            raise EpisodeFeedbackRequestError(
                "unavailable trajectory conflicts with episode reference")
        return _with_exact_compact_character_count({
            "availability": "unavailable",
            "source": None,
            "assistant_steps": [],
            "tool_events": [],
            "final_response": None,
            "unclassified_messages": [],
            "reference_sets": {},
            "reference_evidence": [],
            "referenced_segment_ids": [],
            "retrieved_segment_ids": [],
            "projection_statistics": {
                "raw_message_count": 0,
                "raw_character_count": 0,
                "included_assistant_step_count": 0,
                "included_tool_event_count": 0,
                "included_final_response_count": 0,
                "excluded_boilerplate_message_count": 0,
                "excluded_duplicate_wrapper_count": 0,
                "unclassified_message_count": 0,
                "compact_character_count": 0,
            },
        })
    if resolved.get("availability") != "available" or reference is None:
        raise EpisodeFeedbackRequestError(
            "trajectory availability conflicts with episode reference")
    reference = os.path.abspath(reference)
    try:
        with open(reference, encoding="utf-8") as handle:
            raw_text = handle.read()
    except OSError as exc:
        raise EpisodeFeedbackRequestError(
            f"cannot read raw trajectory {reference}: {exc}") from exc
    rows = list(_read_jsonl_objects(reference, "trajectory"))
    if _plain_json(rows) != _plain_json(resolved.get("content")):
        raise EpisodeFeedbackRequestError(
            "resolved trajectory content changed before compact projection")
    tool_events_path = os.path.join(os.path.dirname(reference), "tool_events.jsonl")
    if os.path.isfile(tool_events_path):
        events = list(_read_jsonl_objects(
            tool_events_path, "trajectory tool events"))
        if _plain_json(events) != _plain_json(resolved.get("tool_events")):
            raise EpisodeFeedbackRequestError(
                "resolved tool events changed before compact projection")
        tool_events_sha = _sha256_file(tool_events_path)
        tool_events_ref = os.path.abspath(tool_events_path)
    else:
        events = []
        if resolved.get("tool_events"):
            raise EpisodeFeedbackRequestError(
                "resolved tool events have no source artifact")
        tool_events_sha = None
        tool_events_ref = None

    compact_events = []
    for event_index, event in enumerate(events):
        projected = {
            key: _plain_json(value) for key, value in event.items()
            if key != "latency_seconds"
        }
        hits = event.get("hits") or ()
        if not isinstance(hits, (list, tuple)):
            raise EpisodeFeedbackRequestError(
                f"tool event {event_index} hits must be an array")
        projected["event_index"] = event_index
        projected["returned_segment_ids"] = [
            segment_id for segment_id in (
                _hit_segment_id(hit) if isinstance(hit, Mapping) else None
                for hit in hits) if segment_id is not None]
        projected["returned_evidence"] = []
        compact_events.append(projected)

    assistant_steps = []
    unclassified = []
    final_response = None
    call_to_event: dict[str, int] = {}
    duplicate_wrappers = 0
    boilerplate = 0
    used_events: set[int] = set()
    allowed_assistant_keys = {
        "role", "content", "tool_calls", "annotations", "refusal"}
    allowed_tool_keys = {"role", "content", "name", "tool_call_id"}
    for message_index, row in enumerate(rows):
        role = row.get("role")
        if role in ("system", "user") and set(row) <= {"role", "content"}:
            boilerplate += 1
            continue
        if role == "assistant" and set(row) <= allowed_assistant_keys:
            calls = row.get("tool_calls") or ()
            content = row.get("content")
            # OpenAI chat transport metadata is not semantic trace content.
            # Preserve any non-empty/unknown value instead of discarding it.
            if row.get("annotations") not in (None, []) or \
                    row.get("refusal") is not None:
                unclassified.append({
                    "message_index": message_index,
                    "message": _plain_json(row)})
                continue
            if calls:
                if not isinstance(calls, list) or any(
                        not isinstance(call, Mapping) for call in calls):
                    unclassified.append({
                        "message_index": message_index, "message": _plain_json(row)})
                    continue
                parsed_calls = []
                invalid_call = False
                for call in calls:
                    function = call.get("function")
                    call_id = call.get("id")
                    if not isinstance(function, Mapping) or \
                            not isinstance(call_id, str) or not call_id:
                        invalid_call = True
                        break
                    parsed_calls.append((
                        call_id, function.get("name"),
                        _parse_call_arguments(function.get("arguments"))))
                if invalid_call:
                    unclassified.append({
                        "message_index": message_index, "message": _plain_json(row)})
                    continue
                finish_calls = [item for item in parsed_calls
                                if item[1] == "finish"]
                if finish_calls:
                    if len(parsed_calls) != 1 or len(finish_calls) != 1:
                        unclassified.append({
                            "message_index": message_index,
                            "message": _plain_json(row)})
                        continue
                    if final_response is not None:
                        raise EpisodeFeedbackRequestError(
                            "trajectory contains multiple final responses")
                    final_response = {
                        "message_index": message_index,
                        "finish_arguments": finish_calls[0][2],
                        "response_text": content if isinstance(content, str) else None,
                    }
                    continue
                if content is not None and not isinstance(content, str):
                    unclassified.append({
                        "message_index": message_index, "message": _plain_json(row)})
                    continue
                pending_matches = []
                pending_used = set(used_events)
                for call_id, name, arguments in parsed_calls:
                    match = None
                    for event_index, event in enumerate(events):
                        comparable_args = event.get(
                            "requested_args", event.get("args"))
                        if event_index not in pending_used and \
                                event.get("tool") == name and \
                                _plain_json(comparable_args) == \
                                _plain_json(arguments):
                            match = event_index
                            break
                    if match is None:
                        pending_matches = []
                        break
                    pending_used.add(match)
                    pending_matches.append((call_id, match))
                if len(pending_matches) != len(parsed_calls):
                    unclassified.append({
                        "message_index": message_index, "message": _plain_json(row)})
                    continue
                used_events.update(match for _call_id, match in pending_matches)
                call_to_event.update(pending_matches)
                duplicate_wrappers += len(pending_matches)
                if isinstance(content, str) and content:
                    assistant_steps.append({
                        "message_index": message_index, "content": content})
                continue
            if isinstance(content, str):
                assistant_steps.append({
                    "message_index": message_index, "content": content})
                continue
            unclassified.append({
                "message_index": message_index, "message": _plain_json(row)})
            continue
        if role == "tool" and set(row) <= allowed_tool_keys:
            event_index = call_to_event.get(row.get("tool_call_id"))
            if event_index is None or row.get("name") != \
                    events[event_index].get("tool"):
                unclassified.append({
                    "message_index": message_index, "message": _plain_json(row)})
                continue
            compact_events[event_index]["returned_evidence"].append(
                _plain_json(row.get("content")))
            duplicate_wrappers += 1
            continue
        unclassified.append({
            "message_index": message_index, "message": _plain_json(row)})

    reference_sets = _plain_json(resolved.get("reference_sets") or {})
    reference_evidence = _plain_json(
        resolved.get("reference_evidence") or [])
    view = {
        "availability": "available",
        "source": {
            "trajectory_ref": reference,
            "trajectory_sha256": _sha256_file(reference),
            "tool_events_ref": tool_events_ref,
            "tool_events_sha256": tool_events_sha,
            "reference_sets_sha256": sha256_json(reference_sets),
            "reference_evidence_sha256": sha256_json(reference_evidence),
        },
        "assistant_steps": assistant_steps,
        "tool_events": compact_events,
        "final_response": final_response,
        "unclassified_messages": unclassified,
        "reference_sets": reference_sets,
        "reference_evidence": reference_evidence,
        "referenced_segment_ids": _plain_json(
            resolved.get("referenced_segment_ids") or []),
        "retrieved_segment_ids": _plain_json(
            resolved.get("retrieved_segment_ids") or []),
        "projection_statistics": {
            "raw_message_count": len(rows),
            "raw_character_count": len(raw_text),
            "included_assistant_step_count": len(assistant_steps),
            "included_tool_event_count": len(compact_events),
            "included_final_response_count": int(final_response is not None),
            "excluded_boilerplate_message_count": boilerplate,
            "excluded_duplicate_wrapper_count": duplicate_wrappers,
            "unclassified_message_count": len(unclassified),
            "compact_character_count": 0,
        },
    }
    return _with_exact_compact_character_count(view)


def _prompt_delta_field_characters(episode: InterventionEpisode) -> int:
    field = dumps_canonical({
        "applied_prompt_delta_instruction": episode.prompt_delta.instruction})
    return len(episode.clips) * (len(field) - 2 + 1)


def build_compact_episode_feedback_request(
    episode: InterventionEpisode,
    *,
    artifact_resolver: EpisodeFeedbackArtifactResolver,
    token_counter: Callable[[tuple[Mapping[str, str], ...]], int] | None = None,
) -> CompactEpisodeFeedbackRequest:
    """Build the explicit D1.5 normalized view without filtering evidence."""
    complete = build_episode_feedback_request(
        episode, artifact_resolver=artifact_resolver)
    catalog, clips, history_before_chars, history_reference_chars = \
        _compact_histories(episode)
    complete_qas = complete.user_payload["episode"]["qas"]
    qas = []
    for outcome, qa in zip(episode.qa_outcomes, complete_qas):
        compact_qa = {
            key: _plain_json(value) for key, value in qa.items()
            if key not in ("baseline_trajectory", "intervention_trajectory")
        }
        compact_qa["baseline_trajectory"] = _compact_trajectory(
            qa["baseline_trajectory"], outcome.baseline_trajectory_ref)
        compact_qa["intervention_trajectory"] = _compact_trajectory(
            qa["intervention_trajectory"], outcome.intervention_trajectory_ref)
        qas.append(compact_qa)
    payload = {
        "schema_version": COMPACT_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION,
        "episode": {
            "episode_id": episode.episode_id,
            "video_id": episode.video_id,
            "parent_meta_prompt_id": episode.parent_meta_prompt_id,
            "prompt_delta": _plain_json(complete.user_payload[
                "episode"]["prompt_delta"]),
            "qa_transition_summary": _plain_json(complete.user_payload[
                "episode"]["qa_transition_summary"]),
            "history_catalog": catalog,
            "clips": clips,
            "qas": qas,
        },
    }
    user_request = dumps_canonical(payload)
    canonical_payload = json.loads(user_request)
    messages = (
        {"role": "system",
         "content": COMPACT_EPISODE_FEEDBACK_SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_request},
    )
    token_count = token_counter(messages) if token_counter is not None else None
    if token_count is not None and (
            not isinstance(token_count, int) or isinstance(token_count, bool) or
            token_count < 0):
        raise EpisodeFeedbackBackendConfigurationError(
            "backend token counter must return a non-negative integer")
    compact_hash = sha256_json(canonical_payload)
    trajectories = [
        qa[side] for qa in qas
        for side in ("baseline_trajectory", "intervention_trajectory")]
    compact_trajectory_chars = sum(
        len(dumps_canonical(item)) for item in trajectories)
    if compact_trajectory_chars != sum(
            item["projection_statistics"]["compact_character_count"]
            for item in trajectories):
        raise EpisodeFeedbackRequestError(
            "compact trajectory size accounting mismatch")
    stats = CompactEpisodeFeedbackRequestSize(
        clip_count=len(clips), qa_count=len(qas),
        complete_request_character_count=(
            complete.size_statistics.serialized_request_character_count),
        compact_request_character_count=len(dumps_canonical(messages)),
        history_character_count_before_deduplication=history_before_chars,
        history_catalog_character_count=len(dumps_canonical(catalog)),
        history_reference_character_count=history_reference_chars,
        unique_history_item_count=len(catalog),
        total_history_item_occurrences=sum(
            len(item["history_item_ids"]) for item in clips),
        raw_trajectory_character_count=sum(
            item["projection_statistics"]["raw_character_count"]
            for item in trajectories),
        compact_trajectory_character_count=compact_trajectory_chars,
        # Tool definitions share one raw user-message string with QA metadata,
        # and raw wrapper syntax surrounds retained content.  There is no exact
        # independent source span for either measurement, so do not estimate.
        removed_system_tool_schema_character_count=None,
        removed_duplicate_wrapper_character_count=None,
        repeated_prompt_delta_character_count_removed=(
            _prompt_delta_field_characters(episode)),
        complete_payload_hash=complete.payload_hash,
        compact_payload_hash=compact_hash,
        unclassified_trajectory_message_count=sum(
            item["projection_statistics"]["unclassified_message_count"]
            for item in trajectories),
        unresolved_reference_count=0,
        token_count=token_count,
    )
    return CompactEpisodeFeedbackRequest(
        system_instruction=COMPACT_EPISODE_FEEDBACK_SYSTEM_INSTRUCTION,
        user_payload=canonical_payload,
        messages=messages,
        payload_hash=compact_hash,
        size_statistics=stats,
    )


def _model_trajectory_projection(
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate semantic trajectory evidence from v2 audit/debug fields."""
    assistant_steps = []
    assistant_message_indices = []
    for step in trajectory.get("assistant_steps", ()):
        if not isinstance(step, Mapping):
            raise EpisodeFeedbackRequestError(
                "compact trajectory assistant step must be an object")
        assistant_steps.append({
            key: _plain_json(value) for key, value in step.items()
            if key != "message_index"
        })
        assistant_message_indices.append(step.get("message_index"))

    tool_events = []
    tool_event_indices = []
    for event in trajectory.get("tool_events", ()):
        if not isinstance(event, Mapping):
            raise EpisodeFeedbackRequestError(
                "compact trajectory tool event must be an object")
        tool_events.append({
            key: _plain_json(value) for key, value in event.items()
            if key != "event_index"
        })
        tool_event_indices.append(event.get("event_index"))

    final_response = trajectory.get("final_response")
    final_message_index = None
    if final_response is not None:
        if not isinstance(final_response, Mapping):
            raise EpisodeFeedbackRequestError(
                "compact trajectory final response must be an object or null")
        final_message_index = final_response.get("message_index")
        final_response = {
            key: _plain_json(value) for key, value in final_response.items()
            if key != "message_index"
        }

    unclassified_messages = []
    unclassified_message_indices = []
    for row in trajectory.get("unclassified_messages", ()):
        if not isinstance(row, Mapping) or "message" not in row:
            raise EpisodeFeedbackRequestError(
                "compact unclassified trajectory message is invalid")
        unclassified_messages.append(_plain_json(row["message"]))
        unclassified_message_indices.append(row.get("message_index"))

    model_view = {
        "availability": trajectory.get("availability"),
        "assistant_steps": assistant_steps,
        "tool_events": tool_events,
        "final_response": final_response,
        "unclassified_messages": unclassified_messages,
        "referenced_segment_ids": _plain_json(
            trajectory.get("referenced_segment_ids") or []),
        "retrieved_segment_ids": _plain_json(
            trajectory.get("retrieved_segment_ids") or []),
    }
    audit_view = {
        "source": _plain_json(trajectory.get("source")),
        "projection_statistics": _plain_json(
            trajectory.get("projection_statistics") or {}),
        "reference_sets": _plain_json(
            trajectory.get("reference_sets") or {}),
        "reference_evidence": _plain_json(
            trajectory.get("reference_evidence") or []),
        "projection_debug": {
            "assistant_message_indices": assistant_message_indices,
            "tool_event_indices": tool_event_indices,
            "final_response_message_index": final_message_index,
            "unclassified_message_indices": unclassified_message_indices,
        },
    }
    return model_view, audit_view


def build_model_compact_episode_feedback_request(
    episode: InterventionEpisode,
    *,
    artifact_resolver: EpisodeFeedbackArtifactResolver,
    token_counter: Callable[[tuple[Mapping[str, str], ...]], int] | None = None,
) -> ModelCompactEpisodeFeedbackRequest:
    """Build D1.6 model evidence and audit metadata as independent values."""
    compact = build_compact_episode_feedback_request(
        episode, artifact_resolver=artifact_resolver)
    compact_episode = compact.user_payload["episode"]

    history_audit = []
    model_clips = []
    for clip in compact_episode["clips"]:
        history_audit.append({
            "segment_id": clip["segment_id"],
            "history_snapshot_projection": _plain_json(
                clip["history_snapshot_projection"]),
            "history_snapshot_metadata": _plain_json(
                clip["history_snapshot_metadata"]),
        })
        model_clips.append({
            key: _plain_json(value) for key, value in clip.items()
            if key not in (
                "history_snapshot_projection", "history_snapshot_metadata")
        })

    model_qas = []
    trajectory_audit = []
    for qa in compact_episode["qas"]:
        model_qa = {
            key: _plain_json(value) for key, value in qa.items()
            if key not in ("baseline_trajectory", "intervention_trajectory")
        }
        qa_audit = {"qa_id": qa["qa_id"]}
        for side in ("baseline_trajectory", "intervention_trajectory"):
            model_view, audit_view = _model_trajectory_projection(qa[side])
            model_qa[side] = model_view
            qa_audit[side] = audit_view
        model_qas.append(model_qa)
        trajectory_audit.append(qa_audit)

    model_payload = {
        "schema_version": MODEL_COMPACT_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION,
        "episode": {
            "episode_id": compact_episode["episode_id"],
            "video_id": compact_episode["video_id"],
            "parent_meta_prompt_id": compact_episode["parent_meta_prompt_id"],
            "prompt_delta": _plain_json(compact_episode["prompt_delta"]),
            "qa_transition_summary": _plain_json(
                compact_episode["qa_transition_summary"]),
            "history_catalog": _plain_json(compact_episode["history_catalog"]),
            "clips": model_clips,
            "qas": model_qas,
        },
    }
    model_request = dumps_canonical(model_payload)
    canonical_model_payload = json.loads(model_request)

    source_artifacts = {}
    for name, path in (
            ("baseline_run", episode.baseline_run_ref),
            ("intervention_run", episode.intervention_run_ref)):
        absolute_path = os.path.abspath(path)
        if not os.path.isfile(absolute_path):
            raise EpisodeFeedbackRequestError(
                f"{name} reference does not resolve: {absolute_path!r}")
        source_artifacts[name] = {
            "path": absolute_path,
            "sha256": _sha256_file(absolute_path),
        }

    compact_stats = compact.size_statistics
    audit_metadata = {
        "schema_version": MODEL_COMPACT_EPISODE_FEEDBACK_AUDIT_SCHEMA_VERSION,
        "model_schema_version": (
            MODEL_COMPACT_EPISODE_FEEDBACK_REQUEST_SCHEMA_VERSION),
        "episode_id": episode.episode_id,
        "source_artifacts": source_artifacts,
        "history_reconstruction": history_audit,
        "trajectory_projection": trajectory_audit,
        "request_projection": {
            "complete_payload_hash": compact_stats.complete_payload_hash,
            "compact_payload_hash": compact_stats.compact_payload_hash,
            "complete_request_character_count": (
                compact_stats.complete_request_character_count),
            "compact_request_character_count": (
                compact_stats.compact_request_character_count),
            "history_character_count_before_deduplication": (
                compact_stats.history_character_count_before_deduplication),
            "history_catalog_character_count": (
                compact_stats.history_catalog_character_count),
            "history_reference_character_count": (
                compact_stats.history_reference_character_count),
            "raw_trajectory_character_count": (
                compact_stats.raw_trajectory_character_count),
            "compact_trajectory_character_count": (
                compact_stats.compact_trajectory_character_count),
            "removed_system_tool_schema_character_count": (
                compact_stats.removed_system_tool_schema_character_count),
            "removed_duplicate_wrapper_character_count": (
                compact_stats.removed_duplicate_wrapper_character_count),
            "repeated_prompt_delta_character_count_removed": (
                compact_stats.repeated_prompt_delta_character_count_removed),
            "unclassified_trajectory_message_count": (
                compact_stats.unclassified_trajectory_message_count),
            "unresolved_reference_count": (
                compact_stats.unresolved_reference_count),
        },
    }
    canonical_audit_metadata = json.loads(dumps_canonical(audit_metadata))
    model_hash = sha256_json(canonical_model_payload)
    audit_hash = sha256_json(canonical_audit_metadata)
    messages = (
        {"role": "system",
         "content": MODEL_COMPACT_EPISODE_FEEDBACK_SYSTEM_INSTRUCTION},
        {"role": "user", "content": model_request},
    )
    token_count = token_counter(messages) if token_counter is not None else None
    if token_count is not None and (
            not isinstance(token_count, int) or isinstance(token_count, bool) or
            token_count < 0):
        raise EpisodeFeedbackBackendConfigurationError(
            "backend token counter must return a non-negative integer")
    model_trajectories = [
        qa[side] for qa in model_qas
        for side in ("baseline_trajectory", "intervention_trajectory")]
    stats = ModelCompactEpisodeFeedbackRequestSize(
        clip_count=compact_stats.clip_count,
        qa_count=compact_stats.qa_count,
        complete_request_character_count=(
            compact_stats.complete_request_character_count),
        compact_request_character_count=(
            compact_stats.compact_request_character_count),
        model_request_character_count=len(dumps_canonical(messages)),
        model_payload_character_count=len(model_request),
        audit_metadata_character_count=len(dumps_canonical(
            canonical_audit_metadata)),
        history_catalog_character_count=(
            compact_stats.history_catalog_character_count),
        history_reference_character_count=(
            compact_stats.history_reference_character_count),
        unique_history_item_count=compact_stats.unique_history_item_count,
        total_history_item_occurrences=(
            compact_stats.total_history_item_occurrences),
        raw_trajectory_character_count=(
            compact_stats.raw_trajectory_character_count),
        compact_trajectory_character_count=(
            compact_stats.compact_trajectory_character_count),
        model_trajectory_character_count=sum(
            len(dumps_canonical(item)) for item in model_trajectories),
        complete_payload_hash=compact_stats.complete_payload_hash,
        compact_payload_hash=compact_stats.compact_payload_hash,
        model_payload_hash=model_hash,
        audit_metadata_hash=audit_hash,
        unclassified_trajectory_message_count=(
            compact_stats.unclassified_trajectory_message_count),
        unresolved_reference_count=compact_stats.unresolved_reference_count,
        token_count=token_count,
    )
    return ModelCompactEpisodeFeedbackRequest(
        system_instruction=MODEL_COMPACT_EPISODE_FEEDBACK_SYSTEM_INSTRUCTION,
        model_payload=canonical_model_payload,
        audit_metadata=canonical_audit_metadata,
        model_payload_hash=model_hash,
        audit_metadata_hash=audit_hash,
        messages=messages,
        size_statistics=stats,
    )


_RESPONSE_FIELDS = {
    "episode_id", "outcome_summary", "observations", "counterevidence",
    "generator_diagnosis", "recommended_strategy_change", "confidence",
}
_EVIDENCE_FIELDS = {
    "statement", "supporting_segment_ids", "supporting_qa_ids",
    "evidence_type", "transition_type", "confidence",
}


def parse_episode_feedback_response(
    raw_response: Any,
    *,
    episode: InterventionEpisode,
    policy_version: str,
    request_payload_hash: str,
) -> EpisodeFeedback:
    """Accept exactly one JSON object; perform no extraction or repair."""
    try:
        if not isinstance(raw_response, str):
            raise TypeError("response must be a string")
        value = json.loads(raw_response)
        if not isinstance(value, dict) or set(value) != _RESPONSE_FIELDS:
            raise ValueError(
                f"response must contain exactly {sorted(_RESPONSE_FIELDS)}")
        if value.get("episode_id") != episode.episode_id:
            raise ValueError("response episode_id does not match input episode")
        for collection_name in ("observations", "counterevidence"):
            items = value[collection_name]
            if not isinstance(items, list):
                raise TypeError(f"{collection_name} must be an array")
            for index, item in enumerate(items):
                if not isinstance(item, dict) or set(item) != _EVIDENCE_FIELDS:
                    raise ValueError(
                        f"{collection_name}[{index}] does not match strict fields")
                segment_ids = item["supporting_segment_ids"]
                qa_ids = item["supporting_qa_ids"]
                for field_name, identifiers in (
                        ("supporting_segment_ids", segment_ids),
                        ("supporting_qa_ids", qa_ids)):
                    if not isinstance(identifiers, list) or any(
                            not isinstance(identifier, str) or not identifier
                            for identifier in identifiers):
                        raise TypeError(
                            f"{collection_name}[{index}].{field_name} must be "
                            "a non-empty-string array")
                if not segment_ids and not qa_ids:
                    raise ValueError(
                        f"{collection_name}[{index}] has no supporting IDs")
                evidence_type = item["evidence_type"]
                transition_type = item["transition_type"]
                if evidence_type == "caption_change" and not segment_ids:
                    raise ValueError(
                        f"{collection_name}[{index}] caption evidence has no "
                        "supporting segment ID")
                if evidence_type == "qa_transition" and not qa_ids:
                    raise ValueError(
                        f"{collection_name}[{index}] QA evidence has no "
                        "supporting QA ID")
                if evidence_type == "qa_transition":
                    if transition_type not in QA_TRANSITION_SUMMARY_ORDER:
                        raise ValueError(
                            f"{collection_name}[{index}] has invalid or null "
                            "transition_type")
                    transition_by_qa_id = {
                        outcome.qa_id: classify_qa_transition(outcome)
                        for outcome in episode.qa_outcomes
                    }
                    mismatched = {
                        qa_id: transition_by_qa_id.get(qa_id)
                        for qa_id in qa_ids
                        if transition_by_qa_id.get(qa_id) != transition_type
                    }
                    if mismatched:
                        raise ValueError(
                            f"{collection_name}[{index}] transition_type "
                            "conflicts with stored QA outcomes: "
                            f"{mismatched}")
                elif transition_type is not None:
                    raise ValueError(
                        f"{collection_name}[{index}] transition_type must be "
                        "null for non-qa_transition evidence")
                if evidence_type in ("trajectory", "mixed") and (
                        not segment_ids or not qa_ids):
                    raise ValueError(
                        f"{collection_name}[{index}] trajectory-linked evidence "
                        "requires both segment and QA IDs")
        identity = {
            "episode_id": episode.episode_id,
            "feedback_policy_version": policy_version,
            "request_payload_hash": request_payload_hash,
        }
        feedback = episode_feedback_from_json({
            "feedback_id": "episode_feedback_" + sha256_json(identity)[:20],
            **value,
        })
        validate_episode_feedback(feedback, episode)
        restored = episode_feedback_from_json(
            json.loads(dumps_canonical(feedback)))
        if restored != feedback or dumps_canonical(restored) != \
                dumps_canonical(feedback):
            raise ValueError("EpisodeFeedback failed canonical round-trip")
        return feedback
    except EpisodeFeedbackParseError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EpisodeFeedbackParseError(
            f"invalid episode feedback response: {exc}",
            raw_response=raw_response) from exc


def _backend_metadata(
    provider: EpisodeFeedbackResponseProvider,
) -> dict[str, Any]:
    metadata_fn = getattr(provider, "metadata", None)
    if not callable(metadata_fn):
        raise EpisodeFeedbackBackendConfigurationError(
            "response provider must expose metadata()")
    metadata = metadata_fn()
    if not isinstance(metadata, Mapping):
        raise EpisodeFeedbackBackendConfigurationError(
            "response provider metadata must be an object")
    metadata = json.loads(dumps_canonical(metadata))
    if not isinstance(metadata.get("provider"), str) or not metadata["provider"]:
        raise EpisodeFeedbackBackendConfigurationError(
            "backend provider identity is required")
    if not isinstance(metadata.get("model"), str) or not metadata["model"]:
        raise EpisodeFeedbackBackendConfigurationError(
            "backend model identity is required")
    if not isinstance(metadata.get("generation_settings"), Mapping) or not \
            metadata["generation_settings"]:
        raise EpisodeFeedbackBackendConfigurationError(
            "explicit backend generation_settings are required")
    output_limit = metadata.get("output_token_limit")
    if not isinstance(output_limit, int) or isinstance(output_limit, bool) or \
            output_limit <= 0:
        raise EpisodeFeedbackBackendConfigurationError(
            "explicit positive output_token_limit is required")
    if "context_limit_tokens" not in metadata:
        raise EpisodeFeedbackBackendConfigurationError(
            "context_limit_tokens must be explicit, including null when unknown")
    context_limit = metadata["context_limit_tokens"]
    if context_limit is not None and (
            not isinstance(context_limit, int) or isinstance(context_limit, bool)
            or context_limit <= 0):
        raise EpisodeFeedbackBackendConfigurationError(
            "context_limit_tokens must be a positive integer or null")
    return metadata


class LLMEpisodeFeedbackGenerator:
    """One-call injected policy with an inspectable, non-persistent trace."""

    def __init__(
        self,
        *,
        response_provider: EpisodeFeedbackResponseProvider,
        artifact_resolver: EpisodeFeedbackArtifactResolver,
        policy_version: str,
        request_representation: Literal[
            "complete", "compact", "model_compact"] = "complete",
    ) -> None:
        if not callable(response_provider):
            raise EpisodeFeedbackBackendConfigurationError(
                "response_provider must be callable")
        if artifact_resolver is None:
            raise EpisodeFeedbackBackendConfigurationError(
                "artifact_resolver is required")
        if not isinstance(policy_version, str) or not policy_version:
            raise EpisodeFeedbackBackendConfigurationError(
                "policy_version must be an explicit non-empty string")
        if request_representation not in (
                "complete", "compact", "model_compact"):
            raise EpisodeFeedbackBackendConfigurationError(
                "request_representation must be 'complete', 'compact', or "
                "'model_compact'")
        self.response_provider = response_provider
        self.artifact_resolver = artifact_resolver
        self.policy_version = policy_version
        self.request_representation = request_representation

    def generate(self, episode: InterventionEpisode) -> EpisodeFeedback:
        return self.generate_with_trace(episode).feedback

    def generate_with_trace(
        self, episode: InterventionEpisode,
    ) -> EpisodeFeedbackInvocationResult:
        metadata = _backend_metadata(self.response_provider)
        token_counter = getattr(self.response_provider, "count_tokens", None)
        if token_counter is not None and not callable(token_counter):
            raise EpisodeFeedbackBackendConfigurationError(
                "backend count_tokens must be callable")
        preflight = getattr(self.response_provider, "preflight", None)
        if preflight is not None and not callable(preflight):
            raise EpisodeFeedbackBackendConfigurationError(
                "backend preflight must be callable")
        if self.request_representation == "complete":
            request = build_episode_feedback_request(
                episode,
                artifact_resolver=self.artifact_resolver,
                token_counter=token_counter,
                context_limit_tokens=metadata["context_limit_tokens"],
            )
        elif self.request_representation == "compact":
            request = build_compact_episode_feedback_request(
                episode,
                artifact_resolver=self.artifact_resolver,
                token_counter=token_counter,
            )
        else:
            request = build_model_compact_episode_feedback_request(
                episode,
                artifact_resolver=self.artifact_resolver,
                token_counter=token_counter,
            )
        if self.request_representation != "complete" and preflight is None:
            limit = metadata["context_limit_tokens"]
            observed = request.size_statistics.token_count
            if limit is not None and observed is not None and observed > limit:
                sections = request.user_payload["episode"]
                largest = tuple(sorted((
                    ("history_catalog", len(dumps_canonical(
                        sections["history_catalog"]))),
                    ("clips", len(dumps_canonical(sections["clips"]))),
                    ("qas", len(dumps_canonical(sections["qas"]))),
                    ("system_instruction", len(request.system_instruction)),
                ), key=lambda item: (-item[1], item[0])))
                raise EpisodeFeedbackContextOverflowError(
                    observed_tokens=observed, configured_limit=limit,
                    clip_count=len(episode.clips),
                    qa_count=len(episode.qa_outcomes),
                    largest_payload_sections=largest)
        if preflight is not None:
            preflight(request.system_instruction, request.user_request)
        raw = self.response_provider(
            request.system_instruction, request.user_request)
        feedback = parse_episode_feedback_response(
            raw, episode=episode, policy_version=self.policy_version,
            request_payload_hash=request.payload_hash)
        return EpisodeFeedbackInvocationResult(
            request=request,
            raw_response=raw,
            feedback=feedback,
            policy_version=self.policy_version,
            backend_metadata=_backend_metadata(self.response_provider),
        )
