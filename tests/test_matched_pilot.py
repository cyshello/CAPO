"""Boundaries for the bounded matched multi-iteration pilot."""

import json
from pathlib import Path

from surrogate_rollout.scripts.run_phase4_matched_pilot import CONDITIONS, ROLES


ROOT = Path(__file__).parents[1]


def test_pilot_roles_are_disjoint_train_videos_and_exclude_test():
    manifest = json.loads((ROOT / "split_manifest.json").read_text())
    train = {item["video_id"] for item in manifest["splits"]["train"]}
    test = {item["video_id"] for item in manifest["splits"]["test"]}
    role_ids = {
        role: {video_id for video_id, _, _ in records}
        for role, records in ROLES.items()}
    flattened = [video_id for values in role_ids.values() for video_id in values]
    assert len(flattened) == len(set(flattened)) == 6
    assert set(flattened) <= train
    assert set(flattened).isdisjoint(test)
    assert len(role_ids["proposal"]) == 3
    assert len(role_ids["confirmation"]) == 1
    assert len(role_ids["regression"]) == 1
    assert len(role_ids["final_evaluation"]) == 1


def test_pilot_conditions_are_matched_except_scaffold_flag():
    assert CONDITIONS == {
        "fixed_scaffold": False,
        "optimized_scaffold": True,
    }


def test_pilot_does_not_import_canonical_component_store():
    source = (ROOT / "scripts/run_phase4_matched_pilot.py").read_text()
    assert "ComponentSnapshotStore" not in source
    assert "write_current" not in source
    assert "range(1, 3)" in source
    assert "max_new_entries_per_iteration=1" in source
    assert "max_router_operations_per_iteration=1" in source


def test_pilot_uses_openai_feedback_and_keeps_codex_dvd_backend():
    source = (ROOT / "scripts/run_phase4_matched_pilot.py").read_text()
    assert "OpenAIStructuredFeedbackProvider" in source
    assert '"feedback_provider": "openai_api"' in source
    assert 'text_backend="codex"' in source
    assert "use_openai_tools=False" in source
