import json
from pathlib import Path

import pytest

from surrogate_rollout.optimization.iteration_state import (
    ComponentVersions,
    ConfirmedCheckpointState,
    ProvisionalIterationState,
)
from surrogate_rollout.optimization.property_proposal import (
    CandidatePropertyProposal,
    VideoPropertyProposalRecord,
)
from surrogate_rollout.optimization.train_roles import (
    derive_train_roles,
    initial_coverage,
    reset_after_confirmation,
    select_evidence_batch,
)


ROOT = Path(__file__).parents[1]


def roles():
    return derive_train_roles(json.loads((ROOT / "split_manifest.json").read_text()))


def test_actual_manifest_derives_eight_pinned_evidence_and_two_confirmation():
    value = roles()
    assert [item.video_id for item in value.evidence_videos] == [
        "0RxMZBLeqRI", "7D-gxaie6UI", "GLW9omJfAdk", "TGom0uiW130",
        "pU_yyadYgG8", "w0Wmc8C0Eq0", "wCkQ138sg6M", "xKiRmesHWIA",
    ]
    assert [item.video_id for item in value.confirmation_videos] == [
        "g1VFfVsZt7w", "jIx5Zi84Z3Q",
    ]
    assert all(item.previously_cached for item in value.evidence_videos)
    assert not any(item.previously_cached for item in value.confirmation_videos)
    assert len(value.validation_video_ids) == len(value.test_video_ids) == 10


def test_rotation_prioritizes_unused_and_requires_confirmation_after_full_coverage():
    state = initial_coverage(roles())
    first, state = select_evidence_batch(state)
    second, state = select_evidence_batch(state)
    third, state = select_evidence_batch(state)
    assert first == ("0RxMZBLeqRI", "7D-gxaie6UI", "GLW9omJfAdk")
    assert second == ("TGom0uiW130", "pU_yyadYgG8", "w0Wmc8C0Eq0")
    assert third == ("wCkQ138sg6M", "xKiRmesHWIA", "0RxMZBLeqRI")
    assert len(state.used_since_confirmation) == 8
    assert state.confirmation_due
    with pytest.raises(RuntimeError, match="confirmation is due"):
        select_evidence_batch(state)
    reset = reset_after_confirmation(state)
    assert reset.coverage_cycle == 1
    assert reset.used_since_confirmation == ()
    assert select_evidence_batch(reset)[0] == (
        "7D-gxaie6UI", "GLW9omJfAdk", "TGom0uiW130")


def test_provisional_confirmed_and_zero_or_multiple_proposal_schemas():
    versions = ComponentVersions(
        "bank_v0001", "router_v0001", "scaffold_v0001", "contract_v0001")
    provisional = ProvisionalIterationState(
        iteration_id="iteration_1", coverage_cycle=0,
        selected_video_ids=("v1", "v2", "v3"),
        parent_confirmed_checkpoint_id="confirmed_0",
        input_versions=versions, provisional_versions=versions,
        baseline_manifest_path="baseline/manifest.json",
        property_proposal_paths=("p1", "p2", "p3"),
        coverage_state_before_hash="before", coverage_state_after_hash="after")
    confirmed = ConfirmedCheckpointState(
        checkpoint_id="confirmed_1", coverage_cycle=0,
        component_versions=versions,
        confirmation_video_ids=("c1", "c2"),
        accepted_provisional_iteration_ids=(provisional.iteration_id,),
        confirmation_artifact_path="confirmation/result.json")
    assert provisional.status == "provisional"
    assert confirmed.status == "confirmed"

    empty = VideoPropertyProposalRecord(
        video_id="v1", baseline_run_id="run",
        baseline_qa_ids=("q1", "q2", "q3"), proposals=(),
        proposer_policy_version="fixture_v1")
    proposals = tuple(CandidatePropertyProposal(
        property_id=f"p{index}", source_video_id="v1",
        source_qa_ids=(f"q{index}",), instruction=f"instruction {index}",
        failure_evidence=f"failure {index}", coverage_assessment="missing",
        proposer_policy_version="fixture_v1") for index in (1, 2))
    multiple = VideoPropertyProposalRecord(
        video_id="v1", baseline_run_id="run",
        baseline_qa_ids=("q1", "q2", "q3"), proposals=proposals,
        proposer_policy_version="fixture_v1")
    assert empty.proposals == ()
    assert len(multiple.proposals) == 2
