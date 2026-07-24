"""Intervention runner that recaptions every baseline segment (full recaption).

The incumbent ``PromptDeltaInterventionRunner.run`` applies the prompt delta only
to ``plan.selected_segment_ids`` — the source-QA-localized subset — and enforces
``selected ⊆ localized candidates`` (fresh_prompt_delta_evidence.py:1428-1431).
For the full-recaption baseline we want the delta applied to the whole video.

Zero-touch strategy (nothing in ``surrogate_rollout`` is edited):

1. Rebuild the plan with ``selected_segment_ids`` = every baseline segment.
2. For the duration of the parent ``run()`` only, widen each selection record's
   ``intervention_candidate_segments`` to that same full set, so the localization
   invariant is satisfied. This field is NOT part of the frame-inspection
   classification hash (``_source_qa_classification_hash`` reads only the cited /
   inspected / call-count fields), so widening it does not change any identity
   the run validates — it purely relaxes the subset check.

The proposer runs in a separate, earlier step with the unmodified selection
function, so its (untruncated) request stays localized and does not overflow;
only the recaption apply-scope grows here.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from surrogate_rollout.optimization import fresh_prompt_delta_evidence as _ev


def _all_baseline_segments(baseline_video_manifest_path: str) -> tuple[str, ...]:
    baseline = _ev._read_json(baseline_video_manifest_path)
    captions = _ev._read_json(baseline["captions_path"])
    # segment_id is "<start>_<end>"; order by start time for a deterministic set.
    return tuple(sorted(
        captions.keys(), key=lambda segment_id: float(segment_id.split("_", 1)[0])))


class FullRecaptionInterventionRunner(_ev.PromptDeltaInterventionRunner):
    """Apply the prompt delta to every baseline segment, not the localized subset."""

    def run(
        self, *, baseline_video_manifest_path: str,
        parent_meta_prompt_id: str, plan: Any, output_directory: str,
    ):
        all_segments = _all_baseline_segments(baseline_video_manifest_path)
        full_plan = dataclasses.replace(
            plan, selected_segment_ids=all_segments)

        original_selection = _ev._qa_segment_selection_records

        def _widened_selection(
            path: str, *, global_inspection_boundary_tolerance_seconds: float,
        ) -> tuple[Mapping[str, Any], ...]:
            rows = original_selection(
                path,
                global_inspection_boundary_tolerance_seconds=
                    global_inspection_boundary_tolerance_seconds)
            # Preserve every field the classification hash reads; only widen the
            # candidate set the localization invariant checks against.
            return tuple(
                {**row, "intervention_candidate_segments": list(all_segments)}
                for row in rows)

        _ev._qa_segment_selection_records = _widened_selection
        try:
            return super().run(
                baseline_video_manifest_path=baseline_video_manifest_path,
                parent_meta_prompt_id=parent_meta_prompt_id,
                plan=full_plan, output_directory=output_directory)
        finally:
            _ev._qa_segment_selection_records = original_selection
