"""Resuming a multi-iteration experiment replays completed iterations.

Iteration N's recorded active meta-prompt stops being the pointer's value as
soon as iteration N+1 promotes, so the completed-iteration guard has to accept
a pointer that moved forward while still rejecting one that did not come from
this run.
"""

import json
import os

from surrogate_rollout.optimization.prompt_delta_iteration import (
    _pointer_lineage,
)


def _version(state, meta_prompt_id, parent_meta_prompt_id):
    versions = os.path.join(state, "versions")
    os.makedirs(versions, exist_ok=True)
    with open(os.path.join(versions, f"{meta_prompt_id}.json"), "w") as handle:
        json.dump({
            "meta_prompt_id": meta_prompt_id,
            "parent_meta_prompt_id": parent_meta_prompt_id,
            "status": "confirmed",
            "text": "task section",
            "created_at": "2026-07-22T00:00:00Z",
        }, handle)


def _chain_of_three(state):
    _version(state, "meta_prompt_root", None)
    _version(state, "meta_prompt_second", "meta_prompt_root")
    _version(state, "meta_prompt_third", "meta_prompt_second")


def test_lineage_walks_back_to_the_root(tmp_path):
    state = str(tmp_path)
    _chain_of_three(state)
    assert _pointer_lineage(state, "meta_prompt_third") == (
        "meta_prompt_third", "meta_prompt_second", "meta_prompt_root")


def test_earlier_iteration_result_is_an_ancestor_of_a_moved_pointer(tmp_path):
    state = str(tmp_path)
    _chain_of_three(state)
    lineage = _pointer_lineage(state, "meta_prompt_third")
    assert "meta_prompt_second" in lineage
    assert "meta_prompt_root" in lineage


def test_foreign_meta_prompt_is_not_in_the_lineage(tmp_path):
    state = str(tmp_path)
    _chain_of_three(state)
    assert "meta_prompt_elsewhere" not in _pointer_lineage(
        state, "meta_prompt_third")


def test_rewound_pointer_does_not_cover_later_iterations(tmp_path):
    state = str(tmp_path)
    _chain_of_three(state)
    # A pointer rewound to the root must not vouch for a descendant.
    assert "meta_prompt_third" not in _pointer_lineage(
        state, "meta_prompt_root")


def test_missing_version_artifact_stops_the_walk(tmp_path):
    state = str(tmp_path)
    _version(state, "meta_prompt_second", "meta_prompt_root")
    assert _pointer_lineage(state, "meta_prompt_second") == (
        "meta_prompt_second", "meta_prompt_root")


def test_absent_or_malformed_pointer_value_yields_no_lineage(tmp_path):
    state = str(tmp_path)
    assert _pointer_lineage(state, None) == ()
    assert _pointer_lineage(state, "") == ()


def test_cyclic_parents_terminate(tmp_path):
    state = str(tmp_path)
    _version(state, "meta_prompt_a", "meta_prompt_b")
    _version(state, "meta_prompt_b", "meta_prompt_a")
    assert _pointer_lineage(state, "meta_prompt_a") == (
        "meta_prompt_a", "meta_prompt_b")
