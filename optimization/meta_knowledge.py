"""Deduplicated Stage 4.9 meta-knowledge with conservative promotion rules."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Sequence

from surrogate_rollout.optimization.schemas import (
    CounterfactualEvidence,
    FailureAttribution,
    FeedbackBatch,
    MetaKnowledgeItem,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import as_json_dict, dumps_canonical
from surrogate_rollout.schemas import sha256_text


def meta_knowledge_from_json(data: dict) -> MetaKnowledgeItem:
    return MetaKnowledgeItem(
        knowledge_id=data["knowledge_id"],
        knowledge_type=data["knowledge_type"],
        condition=data.get("condition") or {},
        principle=data["principle"],
        positive_support_ids=tuple(data.get("positive_support_ids") or ()),
        negative_support_ids=tuple(data.get("negative_support_ids") or ()),
        distinct_video_ids=tuple(data.get("distinct_video_ids") or ()),
        scope=data["scope"],
        confidence=float(data["confidence"]),
        status=data["status"],
        provenance=data.get("provenance") or {},
    )


def _identity(item: MetaKnowledgeItem) -> dict:
    return {
        "knowledge_type": item.knowledge_type,
        "condition": item.condition,
        "principle": item.principle,
        "scope": item.scope,
    }


def _canonical_id(item: MetaKnowledgeItem) -> str:
    return "knowledge_" + sha256_text(dumps_canonical(_identity(item)))[:24]


def _provenance_sources(*items: MetaKnowledgeItem) -> tuple[dict, ...]:
    by_text = {}
    for item in items:
        value = as_json_dict(item.provenance)
        sources = value.get("source_records")
        if isinstance(sources, list):
            values = sources
        else:
            values = [value]
        for source in values:
            by_text[dumps_canonical(source)] = source
    return tuple(by_text[key] for key in sorted(by_text))


class MetaKnowledgeStore:
    def __init__(
        self,
        path: str | None = None,
        *,
        min_support_for_confirmed: int = 2,
        min_distinct_videos_for_confirmed: int = 2,
        min_distinct_videos_for_global_scaffold: int = 3,
    ) -> None:
        if min_support_for_confirmed < 1:
            raise ValueError("min_support_for_confirmed must be positive")
        self.path = path
        self.min_support_for_confirmed = min_support_for_confirmed
        self.min_distinct_videos_for_confirmed = \
            min_distinct_videos_for_confirmed
        self.min_distinct_videos_for_global_scaffold = \
            min_distinct_videos_for_global_scaffold
        self._items: dict[str, MetaKnowledgeItem] = {}
        if path and os.path.exists(path):
            self.load()

    def list_items(self) -> tuple[MetaKnowledgeItem, ...]:
        return tuple(self._items[key] for key in sorted(self._items))

    def _resolved_status(
        self,
        item: MetaKnowledgeItem,
        positive: tuple[str, ...],
        negative: tuple[str, ...],
        videos: tuple[str, ...],
    ) -> str:
        if item.status in ("rejected", "deprecated"):
            return item.status
        if positive and negative:
            return "candidate"
        video_threshold = (
            self.min_distinct_videos_for_global_scaffold
            if item.scope == "global_scaffold"
            else self.min_distinct_videos_for_confirmed
        )
        if len(positive) >= self.min_support_for_confirmed and \
                len(videos) >= video_threshold:
            return "confirmed"
        return "candidate"

    def upsert(self, item: MetaKnowledgeItem) -> MetaKnowledgeItem:
        if not isinstance(item, MetaKnowledgeItem):
            raise TypeError("item must be MetaKnowledgeItem")
        key = _canonical_id(item)
        previous = self._items.get(key)
        positive = tuple(sorted(set(item.positive_support_ids) |
                                (set(previous.positive_support_ids)
                                 if previous else set())))
        negative = tuple(sorted(set(item.negative_support_ids) |
                                (set(previous.negative_support_ids)
                                 if previous else set())))
        videos = tuple(sorted(set(item.distinct_video_ids) |
                              (set(previous.distinct_video_ids)
                               if previous else set())))
        contradictory = bool(positive and negative)
        confidence_values = [float(item.confidence)]
        if previous:
            confidence_values.append(float(previous.confidence))
        confidence = (min(0.5, min(confidence_values)) if contradictory
                      else max(confidence_values))
        resolved = replace(
            item,
            knowledge_id=key,
            positive_support_ids=positive,
            negative_support_ids=negative,
            distinct_video_ids=videos,
            confidence=confidence,
            status=self._resolved_status(item, positive, negative, videos),
            provenance={
                "source_records": _provenance_sources(
                    *((previous,) if previous else ()), item),
                "contradictory_support": contradictory,
            },
        )
        self._items[key] = resolved
        return resolved

    def ingest(
        self,
        evidence: Sequence[CounterfactualEvidence],
        feedback: FeedbackBatch,
        attributions: Sequence[FailureAttribution],
    ) -> tuple[MetaKnowledgeItem, ...]:
        evidence_by_id = {item.evidence_id: item for item in evidence}
        attribution_by_id = {item.attribution_id: item for item in attributions}
        added = []
        for feedback_item in feedback.items:
            attribution = attribution_by_id.get(feedback_item.attribution_id)
            if attribution is None:
                raise ValueError(
                    f"missing attribution {feedback_item.attribution_id!r}")
            records = [evidence_by_id[value] for value in feedback_item.evidence_ids]
            target = attribution.targets[0]
            if target == "router":
                knowledge_type, scope = "routing_pattern", "routing"
            elif target == "scaffold":
                knowledge_type, scope = "composition_pattern", "global_scaffold"
            else:
                knowledge_type, scope = "failure_pattern", "meta_only"
            principle = (
                feedback_item.desired_behaviors[0]
                if feedback_item.desired_behaviors else
                (feedback_item.failure_modes[0]
                 if feedback_item.failure_modes else "collect_more_evidence")
            )
            prototype = MetaKnowledgeItem(
                knowledge_id="pending",
                knowledge_type=knowledge_type,
                condition={
                    "attribution_target": target,
                    "segment_traits": feedback_item.applicable_segment_traits,
                },
                principle=principle,
                positive_support_ids=tuple(sorted(feedback_item.evidence_ids)),
                negative_support_ids=(),
                distinct_video_ids=tuple(sorted({item.video_id for item in records})),
                scope=scope,
                confidence=attribution.confidence,
                status="candidate",
                provenance={
                    "feedback_id": feedback_item.feedback_id,
                    "attribution_id": attribution.attribution_id,
                },
            )
            added.append(self.upsert(prototype))
        return tuple(added)

    def save(self, path: str | None = None) -> str:
        output_path = path or self.path
        if not output_path:
            raise ValueError("MetaKnowledgeStore.save requires a path")
        _atomic_write_text(output_path, json.dumps(
            [as_json_dict(item) for item in self.list_items()],
            sort_keys=True, ensure_ascii=False, indent=2,
        ) + "\n")
        return output_path

    def load(self, path: str | None = None) -> tuple[MetaKnowledgeItem, ...]:
        input_path = path or self.path
        if not input_path:
            raise ValueError("MetaKnowledgeStore.load requires a path")
        with open(input_path) as f:
            values = json.load(f)
        if not isinstance(values, list):
            raise ValueError("meta-knowledge artifact must be a JSON list")
        self._items = {}
        for value in values:
            item = meta_knowledge_from_json(value)
            self._items[_canonical_id(item)] = replace(
                item, knowledge_id=_canonical_id(item))
        return self.list_items()
