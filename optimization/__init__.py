"""Phase 4 optimization data boundaries.

Stage 4.9 provides offline evidence, deterministic feedback, attribution, and
candidate meta-knowledge. Proposals, review, validation, and optimization are
intentionally absent until later stop-gated stages.
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
    FailureAttribution,
    FeedbackBatch,
    FeedbackItem,
    MetaKnowledgeItem,
)
from surrogate_rollout.optimization.failure_attributor import (
    AttributionError,
    DeterministicFailureAttributor,
    FailureAttributor,
)
from surrogate_rollout.optimization.feedback_generator import (
    FeedbackGenerator,
    FeedbackParseError,
    attach_attributions,
    read_feedback_batch,
    write_feedback_batch,
)
from surrogate_rollout.optimization.meta_knowledge import MetaKnowledgeStore
from surrogate_rollout.optimization.policies import (
    LLMFeedbackGenerator,
    MockFeedbackGenerator,
    RealFeedbackDisabledError,
)

__all__ = [
    "CaptionDifference",
    "CounterfactualEvidence",
    "CounterfactualEvidenceBuilder",
    "DeterministicFailureAttributor",
    "EvidenceBuildError",
    "FailureAttribution",
    "FailureAttributor",
    "AttributionError",
    "FeedbackBatch",
    "FeedbackGenerator",
    "FeedbackItem",
    "FeedbackParseError",
    "LLMFeedbackGenerator",
    "MetaKnowledgeItem",
    "MetaKnowledgeStore",
    "MockFeedbackGenerator",
    "RealFeedbackDisabledError",
    "SavedEvaluationArtifact",
    "attach_attributions",
    "load_normalized_evaluations",
    "load_phase2_3_condition",
    "load_stage4_7_pair",
    "read_evidence_jsonl",
    "read_feedback_batch",
    "write_evidence_artifacts",
    "write_evidence_jsonl",
    "write_feedback_batch",
    "write_normalized_evaluations",
]
