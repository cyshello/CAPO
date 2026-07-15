"""Phase 4 optimization data boundaries.

Stage 4.12 connects one deterministic saved-fixture iteration while keeping
captioning, reasoning, model calls, and canonical component commits absent.
"""

from surrogate_rollout.optimization.candidate_preview import (
    CandidatePreview,
    CandidatePreviewBuilder,
    CandidatePreviewError,
)

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
    NoChangeEvaluationResult,
    OptimizationIterationResult,
    PromptBankOperation,
    PromptBankUpdateProposal,
    RouterUpdateProposal,
    ScaffoldUpdateProposal,
    ComponentValidationResult,
    UpdateReview,
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
from surrogate_rollout.optimization.offline_iteration import (
    PromptRoutingOptimizationLoop,
)
from surrogate_rollout.optimization.real_fixed_scaffold_iteration import (
    RealEvaluationResult,
    RealFixedScaffoldIterationRunner,
    evaluate_routed_states,
)
from surrogate_rollout.optimization.prompt_bank_update_proposer import (
    DeterministicPromptBankUpdateProposer,
    PromptBankUpdateProposer,
)
from surrogate_rollout.optimization.router_update_proposer import (
    DeterministicRouterUpdateProposer,
    RouterUpdateProposer,
)
from surrogate_rollout.optimization.scaffold_update_proposer import (
    DeterministicScaffoldUpdateProposer,
    ScaffoldUpdateProposer,
)
from surrogate_rollout.optimization.policies import (
    LLMFeedbackGenerator,
    MockFeedbackGenerator,
    RealFeedbackDisabledError,
)
from surrogate_rollout.optimization.update_reviewer import (
    DeterministicUpdateReviewer,
    UpdateReviewer,
)
from surrogate_rollout.optimization.update_validator import (
    DeterministicUpdateValidator,
    UpdateValidator,
    ValidationThresholds,
)

__all__ = [
    "CaptionDifference",
    "CandidatePreview",
    "CandidatePreviewBuilder",
    "CandidatePreviewError",
    "ComponentValidationResult",
    "CounterfactualEvidence",
    "CounterfactualEvidenceBuilder",
    "DeterministicFailureAttributor",
    "DeterministicPromptBankUpdateProposer",
    "DeterministicRouterUpdateProposer",
    "DeterministicScaffoldUpdateProposer",
    "DeterministicUpdateReviewer",
    "DeterministicUpdateValidator",
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
    "NoChangeEvaluationResult",
    "RealFeedbackDisabledError",
    "RealEvaluationResult",
    "RealFixedScaffoldIterationRunner",
    "PromptBankUpdateProposer",
    "PromptBankOperation",
    "PromptBankUpdateProposal",
    "PromptRoutingOptimizationLoop",
    "OptimizationIterationResult",
    "RouterUpdateProposer",
    "RouterUpdateProposal",
    "ScaffoldUpdateProposer",
    "ScaffoldUpdateProposal",
    "UpdateReview",
    "UpdateReviewer",
    "UpdateValidator",
    "ValidationThresholds",
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
    "evaluate_routed_states",
]
