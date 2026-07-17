"""Checkpoint 2 provisional and confirmed iteration-state schemas."""

from __future__ import annotations

from dataclasses import dataclass

from surrogate_rollout.prompt_routing.schemas import validate_component_version


@dataclass(frozen=True)
class ComponentVersions:
    bank_version: str
    router_version: str
    scaffold_version: str
    contract_version: str

    def __post_init__(self) -> None:
        validate_component_version(
            "bank", self.bank_version, field_name="ComponentVersions.bank_version")
        validate_component_version(
            "router", self.router_version,
            field_name="ComponentVersions.router_version")
        validate_component_version(
            "scaffold", self.scaffold_version,
            field_name="ComponentVersions.scaffold_version")
        validate_component_version(
            "contract", self.contract_version,
            field_name="ComponentVersions.contract_version")


@dataclass(frozen=True)
class ProvisionalIterationState:
    iteration_id: str
    coverage_cycle: int
    selected_video_ids: tuple[str, ...]
    parent_confirmed_checkpoint_id: str
    input_versions: ComponentVersions
    provisional_versions: ComponentVersions
    baseline_manifest_path: str
    property_proposal_paths: tuple[str, ...]
    coverage_state_before_hash: str
    coverage_state_after_hash: str
    property_retrieval_paths: tuple[str, ...] = ()
    accumulated_provisional_iteration_ids: tuple[str, ...] = ()
    status: str = "provisional"

    def __post_init__(self) -> None:
        if not self.iteration_id or not self.parent_confirmed_checkpoint_id:
            raise ValueError("provisional state requires iteration and parent IDs")
        if not self.selected_video_ids or \
                len(self.selected_video_ids) != len(set(self.selected_video_ids)):
            raise ValueError("provisional state requires unique evidence videos")
        if len(self.property_proposal_paths) != len(self.selected_video_ids):
            raise ValueError("provisional state requires one proposal artifact per video")
        if self.accumulated_provisional_iteration_ids and \
                self.iteration_id not in self.accumulated_provisional_iteration_ids:
            raise ValueError(
                "accumulated provisional lineage must include the current iteration")
        if self.status != "provisional":
            raise ValueError("ProvisionalIterationState.status must be provisional")


@dataclass(frozen=True)
class ConfirmedCheckpointState:
    checkpoint_id: str
    coverage_cycle: int
    component_versions: ComponentVersions
    confirmation_video_ids: tuple[str, ...]
    accepted_provisional_iteration_ids: tuple[str, ...]
    confirmation_artifact_path: str
    parent_confirmed_checkpoint_id: str | None = None
    status: str = "confirmed"

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("confirmed checkpoint ID must be non-empty")
        if len(self.confirmation_video_ids) != 2 or \
                len(set(self.confirmation_video_ids)) != 2:
            raise ValueError("confirmed checkpoint requires two confirmation videos")
        if self.status != "confirmed":
            raise ValueError("ConfirmedCheckpointState.status must be confirmed")
