"""Phase 4 optimization data boundaries.

Stage 4.8 provides offline evidence schemas and construction only. Feedback,
attribution, proposals, review, validation, and optimization are intentionally
absent until later stop-gated stages.
"""

from surrogate_rollout.optimization.evidence_builder import (
    CounterfactualEvidenceBuilder,
    EvidenceBuildError,
    SavedEvaluationArtifact,
    load_normalized_evaluations,
    load_phase2_3_condition,
    load_stage4_7_pair,
    read_evidence_jsonl,
    write_evidence_artifacts,
    write_evidence_jsonl,
    write_normalized_evaluations,
)
from surrogate_rollout.optimization.schemas import (
    CaptionDifference,
    CounterfactualEvidence,
)

__all__ = [
    "CaptionDifference",
    "CounterfactualEvidence",
    "CounterfactualEvidenceBuilder",
    "EvidenceBuildError",
    "SavedEvaluationArtifact",
    "load_normalized_evaluations",
    "load_phase2_3_condition",
    "load_stage4_7_pair",
    "read_evidence_jsonl",
    "write_evidence_artifacts",
    "write_evidence_jsonl",
    "write_normalized_evaluations",
]
