"""Deterministic Stage 4.9 failure attribution from structured feedback."""

from __future__ import annotations

from typing import Protocol, Sequence

from surrogate_rollout.optimization.schemas import (
    CounterfactualEvidence,
    FailureAttribution,
    FeedbackBatch,
)


SCAFFOLD_FAILURE_MODES = {
    "selected_instruction_omitted",
    "selected_instruction_weakened",
    "composition_contradiction",
    "composition_excessive_verbosity",
    "output_contract_obscured",
    "instruction_ordering_failure",
}


class FailureAttributor(Protocol):
    def attribute(
        self,
        evidence: Sequence[CounterfactualEvidence],
        feedback: FeedbackBatch,
    ) -> Sequence[FailureAttribution]:
        ...


class AttributionError(RuntimeError):
    """Feedback references missing or inconsistent evidence."""


class DeterministicFailureAttributor:
    """Uses typed fields only; rationale prose never controls attribution."""

    def attribute(
        self,
        evidence: Sequence[CounterfactualEvidence],
        feedback: FeedbackBatch,
    ) -> tuple[FailureAttribution, ...]:
        evidence_by_id = {item.evidence_id: item for item in evidence}
        if len(evidence_by_id) != len(evidence):
            raise AttributionError("duplicate evidence IDs")
        missing_inputs = set(feedback.input_evidence_ids) - set(evidence_by_id)
        if missing_inputs:
            raise AttributionError(
                f"feedback input references missing evidence: {sorted(missing_inputs)}")
        output = []
        for item in feedback.items:
            undeclared = set(item.evidence_ids) - set(feedback.input_evidence_ids)
            if undeclared:
                raise AttributionError(
                    "feedback item references evidence not declared by the batch: "
                    f"{sorted(undeclared)}")
            missing = set(item.evidence_ids) - set(evidence_by_id)
            if missing:
                raise AttributionError(
                    f"feedback item references missing evidence: {sorted(missing)}")
            referenced = tuple(evidence_by_id[value] for value in item.evidence_ids)
            components = set(item.target_components) - {"insufficient_evidence"}
            facts = [
                f"{record.evidence_id}:{record.outcome_transition}:"
                f"score_delta={record.score_delta}"
                for record in referenced
            ]

            if "scaffold" in components:
                missing_traces = any(
                    record.metadata.get("missing_trace_sides")
                    or not record.metadata.get("baseline_composition_traces")
                    or not record.metadata.get("candidate_composition_traces")
                    for record in referenced
                )
                explicit_scaffold_fact = (
                    bool(set(item.failure_modes) & SCAFFOLD_FAILURE_MODES)
                    and any(record.metadata.get("scaffold_failure_evidence")
                            for record in referenced)
                )
                if missing_traces or not explicit_scaffold_fact:
                    components.remove("scaffold")
                    facts.append(
                        "scaffold attribution withheld: complete traces and an "
                        "explicit composition failure are required")

            if not components:
                targets = ("insufficient_evidence",)
                confidence = min(float(item.confidence), 0.4)
            elif len(components) > 1:
                targets = ("multiple",)
                confidence = float(item.confidence)
            else:
                targets = (next(iter(sorted(components))),)
                confidence = float(item.confidence)

            issues = tuple(item.failure_modes)
            output.append(FailureAttribution(
                attribution_id=item.attribution_id,
                evidence_ids=item.evidence_ids,
                targets=targets,
                prompt_bank_issues=(issues if "prompt_bank" in components else ()),
                router_issues=(issues if "router" in components else ()),
                scaffold_issues=(issues if "scaffold" in components else ()),
                supporting_facts=tuple(facts),
                confidence=confidence,
                rationale=(
                    "The structured evidence does not isolate a supported component."
                    if targets == ("insufficient_evidence",) else
                    "Attribution follows structured feedback targets and saved evidence."
                ),
            ))
        return tuple(output)
