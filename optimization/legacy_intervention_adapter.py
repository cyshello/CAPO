"""Read-only compatibility adapter from saved property interventions.

The legacy intervention is an artifact bundle, not one self-contained JSON
object.  A completed ``result.json`` supplies intervention identity and points
to QA transitions/mixed captions; the matching baseline ``video_complete.json``
supplies incumbent prompts, captions, QA records, and the original proposal.
This module reads that closed bundle and returns one Checkpoint A
``InterventionEpisode``.  It performs no writes and imports no codebook store.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from surrogate_rollout.optimization.property_proposal import (
    CandidatePropertyProposal,
    candidate_property_proposal_from_json,
)
from surrogate_rollout.optimization.schemas import (
    InterventionClipRecord,
    InterventionEpisode,
    PromptDelta,
    QAInterventionOutcome,
)
from surrogate_rollout.prompt_routing.scaffold_applier import read_composed_prompts
from surrogate_rollout.schemas import sha256_json


EPISODE_IDENTITY_VERSION = "legacy_property_intervention_episode_identity_v1"


class LegacyInterventionConversionError(ValueError):
    """The saved artifact bundle cannot be converted without fabrication."""


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path) as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise LegacyInterventionConversionError(
            f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LegacyInterventionConversionError(
            f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: str) -> tuple[dict[str, Any], ...]:
    rows = []
    try:
        with open(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise LegacyInterventionConversionError(
                        f"expected a JSON object at {path}:{line_number}")
                rows.append(value)
    except LegacyInterventionConversionError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise LegacyInterventionConversionError(
            f"cannot read JSONL artifact {path}: {exc}") from exc
    return tuple(rows)


def _required_text(value: Mapping[str, Any], key: str, where: str) -> str:
    if key not in value:
        raise LegacyInterventionConversionError(
            f"{where} is missing required field {key!r}")
    result = value[key]
    if not isinstance(result, str) or not result:
        raise LegacyInterventionConversionError(
            f"{where}.{key} must be a non-empty string")
    return result


def _resolve_artifact_path(
    value: Mapping[str, Any], key: str, where: str, *, base_dir: str,
) -> str:
    reference = _required_text(value, key, where)
    path = (reference if os.path.isabs(reference)
            else os.path.join(base_dir, reference))
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise LegacyInterventionConversionError(
            f"{where}.{key} does not resolve to an existing artifact: {path}")
    return path


def _trajectory_reference(
    value: Mapping[str, Any], key: str, where: str, *, base_dir: str,
) -> str | None:
    if key not in value:
        raise LegacyInterventionConversionError(
            f"{where} is missing trajectory availability field {key!r}")
    reference = value[key]
    if reference is None:
        return None
    if not isinstance(reference, str) or not reference:
        raise LegacyInterventionConversionError(
            f"{where}.{key} must be an existing path or explicit null")
    path = (reference if os.path.isabs(reference)
            else os.path.join(base_dir, reference))
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise LegacyInterventionConversionError(
            f"{where}.{key} trajectory availability is unresolved: {path}")
    return path


def _unique_by_id(
    rows: tuple[dict[str, Any], ...], key: str, where: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        identity = _required_text(row, key, f"{where}[{index}]")
        if identity in output:
            raise LegacyInterventionConversionError(
                f"duplicate {key} {identity!r} in {where}")
        output[identity] = row
    return output


def _segment_time_range(segment_id: str) -> dict[str, float]:
    """Decode the repository's explicit ``{start}_{end}`` stable clip ID.

    This is not an index-based estimate: ``schemas.CaptionCacheKey`` and the
    history-aware caption path define the two values as clip seconds.
    """
    try:
        start, end = (float(value) for value in segment_id.split("_", 1))
    except (TypeError, ValueError) as exc:
        raise LegacyInterventionConversionError(
            f"segment ID does not contain an exact time range: {segment_id!r}") from exc
    if start < 0 or end <= start:
        raise LegacyInterventionConversionError(
            f"segment ID contains an invalid time range: {segment_id!r}")
    return {"start_seconds": start, "end_seconds": end}


def _caption(
    captions: Mapping[str, Any], segment_id: str, where: str,
) -> str:
    if segment_id not in captions:
        raise LegacyInterventionConversionError(
            f"{where} has no caption for selected segment {segment_id!r}")
    value = captions[segment_id]
    if not isinstance(value, Mapping) or "caption" not in value:
        raise LegacyInterventionConversionError(
            f"{where}[{segment_id!r}] is missing required field 'caption'")
    caption = value["caption"]
    if not isinstance(caption, str) or not caption:
        raise LegacyInterventionConversionError(
            f"{where}[{segment_id!r}].caption must be a non-empty string")
    return caption


def _load_proposal(
    baseline: Mapping[str, Any], baseline_dir: str, candidate_id: str,
) -> CandidatePropertyProposal:
    proposal_path = _resolve_artifact_path(
        baseline, "property_proposal_path", "baseline manifest",
        base_dir=baseline_dir)
    proposal_artifact = _read_json(proposal_path)
    raw_proposals = proposal_artifact.get("proposals")
    if not isinstance(raw_proposals, list):
        raise LegacyInterventionConversionError(
            "legacy property proposal artifact has no proposal list")
    matches = [item for item in raw_proposals if isinstance(item, Mapping)
               and (item.get("candidate_property_id") or item.get("property_id"))
               == candidate_id]
    if len(matches) != 1:
        raise LegacyInterventionConversionError(
            f"expected exactly one proposal for stable candidate ID "
            f"{candidate_id!r}, found {len(matches)}")
    try:
        return candidate_property_proposal_from_json(matches[0])
    except (TypeError, ValueError) as exc:
        raise LegacyInterventionConversionError(
            f"legacy candidate proposal is invalid: {exc}") from exc


def _validate_embedded_proposal(
    artifact: Mapping[str, Any], proposal: CandidatePropertyProposal, where: str,
) -> None:
    embedded = artifact.get("original_proposal")
    if embedded is None:
        return
    if not isinstance(embedded, Mapping):
        raise LegacyInterventionConversionError(
            f"{where}.original_proposal must be an object")
    try:
        loaded = candidate_property_proposal_from_json(embedded)
    except (TypeError, ValueError) as exc:
        raise LegacyInterventionConversionError(
            f"{where}.original_proposal is invalid: {exc}") from exc
    if (loaded.candidate_property_id, loaded.property_text,
            loaded.source_question_ids, loaded.failure_analysis) != (
            proposal.candidate_property_id, proposal.property_text,
            proposal.source_question_ids, proposal.failure_analysis):
        raise LegacyInterventionConversionError(
            f"{where}.original_proposal conflicts with baseline proposal artifact")


def _episode_id(
    result: Mapping[str, Any], *, candidate_id: str,
    parent_baseline_identity: str, intervention_identity: str,
) -> str:
    explicit = result.get("episode_id")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise LegacyInterventionConversionError(
                "intervention result episode_id must be a non-empty string")
        return explicit
    identity = {
        "schema_version": EPISODE_IDENTITY_VERSION,
        "candidate_property_id": candidate_id,
        "parent_baseline_identity": parent_baseline_identity,
        "intervention_input_fingerprint": intervention_identity,
    }
    return f"legacy_episode_{sha256_json(identity)[:20]}"


def legacy_property_intervention_to_episode(
    *,
    intervention_result_path: str,
    baseline_video_manifest_path: str,
    parent_meta_prompt_id: str,
) -> InterventionEpisode:
    """Convert one completed saved legacy intervention without writing files."""
    intervention_result_path = os.path.abspath(intervention_result_path)
    baseline_video_manifest_path = os.path.abspath(baseline_video_manifest_path)
    if intervention_result_path == baseline_video_manifest_path:
        raise LegacyInterventionConversionError(
            "baseline and intervention run references must be distinct")
    if not isinstance(parent_meta_prompt_id, str) or not parent_meta_prompt_id:
        raise LegacyInterventionConversionError(
            "parent_meta_prompt_id must be a non-empty explicit value")

    result = _read_json(intervention_result_path)
    baseline = _read_json(baseline_video_manifest_path)
    result_dir = os.path.dirname(intervention_result_path)
    baseline_dir = os.path.dirname(baseline_video_manifest_path)
    if result.get("status") != "completed":
        raise LegacyInterventionConversionError(
            "only a completed legacy intervention can become an episode")

    candidate_id = _required_text(
        result, "candidate_property_id", "intervention result")
    video_id = _required_text(result, "source_video_id", "intervention result")
    if _required_text(baseline, "video_id", "baseline manifest") != video_id:
        raise LegacyInterventionConversionError(
            "baseline and intervention video IDs do not match")
    parent_baseline_identity = _required_text(
        result, "parent_baseline_identity", "intervention result")
    intervention_identity = _required_text(
        result, "input_fingerprint", "intervention result")

    proposal = _load_proposal(baseline, baseline_dir, candidate_id)
    if proposal.source_video_id != video_id:
        raise LegacyInterventionConversionError(
            "candidate proposal source-video lineage does not match intervention")
    if not proposal.property_text:
        raise LegacyInterventionConversionError(
            "candidate executable property instruction is empty")
    if not proposal.source_question_ids:
        raise LegacyInterventionConversionError(
            "candidate proposal has no source QA ID")
    _validate_embedded_proposal(result, proposal, "intervention result")

    prompt_delta = PromptDelta(
        delta_id=candidate_id,
        instruction=proposal.property_text,
        source_qa_ids=proposal.source_question_ids,
        proposer_diagnosis=proposal.failure_analysis,
    )

    selected_value = result.get("selected_segment_ids")
    if not isinstance(selected_value, list) or not selected_value:
        raise LegacyInterventionConversionError(
            "intervention result must contain recaptioned selected_segment_ids")
    if any(not isinstance(item, str) or not item for item in selected_value):
        raise LegacyInterventionConversionError(
            "selected_segment_ids must contain non-empty strings")
    selected = tuple(selected_value)
    if len(selected) != len(set(selected)):
        raise LegacyInterventionConversionError(
            "duplicate segment ID in intervention scope")

    history_path = os.path.join(result_dir, "frozen_histories.jsonl")
    if not os.path.isfile(history_path):
        raise LegacyInterventionConversionError(
            f"selected clips are missing exact frozen histories: {history_path}")
    histories = _unique_by_id(
        _read_jsonl(history_path), "segment_id", "frozen histories")
    if set(histories) != set(selected):
        raise LegacyInterventionConversionError(
            "frozen history segment IDs do not exactly match intervention scope")

    routing_path = _resolve_artifact_path(
        baseline, "routing_manifest_path", "baseline manifest",
        base_dir=baseline_dir)
    routing = _read_json(routing_path)
    composed_path = _resolve_artifact_path(
        routing, "composed_prompts_path", "baseline routing manifest",
        base_dir=os.path.dirname(routing_path))
    try:
        composed_records = read_composed_prompts(composed_path)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise LegacyInterventionConversionError(
            f"cannot load baseline composed prompts: {exc}") from exc
    composed = {}
    for record in composed_records:
        if record.segment_id in composed:
            raise LegacyInterventionConversionError(
                f"duplicate segment ID {record.segment_id!r} in baseline prompts")
        composed[record.segment_id] = record

    baseline_captions_path = _resolve_artifact_path(
        baseline, "captions_path", "baseline manifest", base_dir=baseline_dir)
    baseline_captions = _read_json(baseline_captions_path)
    mixed_captions_path = _resolve_artifact_path(
        result, "mixed_captions_path", "intervention result", base_dir=result_dir)
    mixed_captions = _read_json(mixed_captions_path)

    clips = []
    for segment_id in selected:
        if segment_id not in composed:
            raise LegacyInterventionConversionError(
                f"selected segment {segment_id!r} has no baseline base prompt")
        base_prompt = composed[segment_id].prompt_text
        if not base_prompt:
            raise LegacyInterventionConversionError(
                f"selected segment {segment_id!r} has an empty baseline base prompt")
        clip = InterventionClipRecord(
            segment_id=segment_id,
            time_range=_segment_time_range(segment_id),
            history_snapshot=histories[segment_id],
            base_prompt=base_prompt,
            prompt_delta=prompt_delta,
            baseline_caption=_caption(
                baseline_captions, segment_id, "baseline captions"),
            intervention_caption=_caption(
                mixed_captions, segment_id, "intervention captions"),
        )
        if clip.prompt_delta.instruction != prompt_delta.instruction:
            raise LegacyInterventionConversionError(
                "clip prompt delta does not match episode prompt delta instruction")
        clips.append(clip)

    transitions_path = _resolve_artifact_path(
        result, "transitions_path", "intervention result", base_dir=result_dir)
    transitions = _read_json(transitions_path)
    if _required_text(
            transitions, "candidate_property_id", "transitions") != candidate_id:
        raise LegacyInterventionConversionError(
            "transition candidate ID does not match intervention")
    if _required_text(
            transitions, "source_video_id", "transitions") != video_id:
        raise LegacyInterventionConversionError(
            "transition video ID does not match intervention")
    _validate_embedded_proposal(transitions, proposal, "transitions")
    transition_value = transitions.get("qas")
    if not isinstance(transition_value, list):
        raise LegacyInterventionConversionError(
            "transitions artifact must contain a QA outcome list")
    transition_rows = tuple(transition_value)
    if any(not isinstance(item, dict) for item in transition_rows):
        raise LegacyInterventionConversionError(
            "transitions.qas must contain JSON objects")
    transition_by_id = _unique_by_id(
        transition_rows, "question_id", "transitions.qas")

    baseline_qas_path = _resolve_artifact_path(
        baseline, "baseline_qas_path", "baseline manifest", base_dir=baseline_dir)
    baseline_qa_rows = _read_jsonl(baseline_qas_path)
    baseline_by_id = _unique_by_id(
        baseline_qa_rows, "question_id", "baseline QAs")
    missing_baselines = set(transition_by_id) - set(baseline_by_id)
    if missing_baselines:
        raise LegacyInterventionConversionError(
            f"candidate QA outcomes lack baseline records: {sorted(missing_baselines)}")
    missing_source_outcomes = set(prompt_delta.source_qa_ids) - set(transition_by_id)
    if missing_source_outcomes:
        raise LegacyInterventionConversionError(
            "source QA outcomes are absent; adapter will not fabricate them: "
            f"{sorted(missing_source_outcomes)}")

    qa_outcomes = []
    for qa_id, row in transition_by_id.items():
        baseline_row = baseline_by_id[qa_id]
        for key in (
            "baseline_prediction", "candidate_prediction",
            "baseline_correct", "candidate_correct",
        ):
            if key not in row:
                raise LegacyInterventionConversionError(
                    f"transitions QA {qa_id!r} is missing required field {key!r}")
        if baseline_row.get("prediction") != row["baseline_prediction"] or \
                baseline_row.get("is_correct") != row["baseline_correct"]:
            raise LegacyInterventionConversionError(
                f"baseline QA values conflict for {qa_id!r}")
        qa_outcomes.append(QAInterventionOutcome(
            qa_id=qa_id,
            is_source_qa=qa_id in prompt_delta.source_qa_ids,
            baseline_answer=row["baseline_prediction"],
            intervention_answer=row["candidate_prediction"],
            baseline_correct=row["baseline_correct"],
            intervention_correct=row["candidate_correct"],
            baseline_trajectory_ref=_trajectory_reference(
                baseline_row, "trajectory_path", f"baseline QA {qa_id!r}",
                base_dir=os.path.dirname(baseline_qas_path)),
            intervention_trajectory_ref=_trajectory_reference(
                row, "trajectory_path", f"intervention QA {qa_id!r}",
                base_dir=os.path.dirname(transitions_path)),
        ))

    episode = InterventionEpisode(
        episode_id=_episode_id(
            result, candidate_id=candidate_id,
            parent_baseline_identity=parent_baseline_identity,
            intervention_identity=intervention_identity),
        video_id=video_id,
        parent_meta_prompt_id=parent_meta_prompt_id,
        prompt_delta=prompt_delta,
        clips=tuple(clips),
        qa_outcomes=tuple(qa_outcomes),
        baseline_run_ref=baseline_video_manifest_path,
        intervention_run_ref=intervention_result_path,
    )
    if len({clip.segment_id for clip in episode.clips}) != len(episode.clips):
        raise LegacyInterventionConversionError("duplicate clip ID in episode")
    if len({qa.qa_id for qa in episode.qa_outcomes}) != len(episode.qa_outcomes):
        raise LegacyInterventionConversionError("duplicate QA ID in episode")
    return episode
