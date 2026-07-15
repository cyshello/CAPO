"""Phase 4 optimization data boundaries.

Stage 4.10 adds preview-only component proposals to the offline evidence,
feedback, attribution, and candidate meta-knowledge boundaries. Review,
validation, commit, and optimization are intentionally absent.
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
    PromptBankOperation,
    PromptBankUpdateProposal,
    RouterUpdateProposal,
    ScaffoldUpdateProposal,
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

__all__ = [
    "CaptionDifference",
    "CounterfactualEvidence",
    "CounterfactualEvidenceBuilder",
    "DeterministicFailureAttributor",
    "DeterministicPromptBankUpdateProposer",
    "DeterministicRouterUpdateProposer",
    "DeterministicScaffoldUpdateProposer",
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
    "PromptBankUpdateProposer",
    "PromptBankOperation",
    "PromptBankUpdateProposal",
    "RouterUpdateProposer",
    "RouterUpdateProposal",
    "ScaffoldUpdateProposer",
    "ScaffoldUpdateProposal",
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
