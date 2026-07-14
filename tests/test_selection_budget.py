from surrogate_rollout.evaluation.rollout_evaluator import SelectionBudget
from surrogate_rollout.selection.base import (
    SOURCE_FRAME_INSPECTION,
    SOURCE_NEIGHBOR,
    SOURCE_PROMPT_DELTA,
    SOURCE_QUESTION,
    SOURCE_TRACE_STRICT,
    SelectionResult,
)
from surrogate_rollout.selection.budget import apply_budget


def _result(entries):
    r = SelectionResult()
    for clip_id, source, score in entries:
        r.add(clip_id, source, score)
    return r


def test_within_budget_unchanged():
    r = _result([("0_10", SOURCE_QUESTION, 0.9), ("10_20", SOURCE_NEIGHBOR, None)])
    out = apply_budget(r, SelectionBudget(max_recaption_fraction=0.5), total_clips=10)
    assert set(out.clip_ids) == {"0_10", "10_20"}
    assert out.metadata["budget"]["dropped"] == []


def test_priority_order_kept_first():
    r = _result([
        ("0_10", SOURCE_NEIGHBOR, None),
        ("10_20", SOURCE_QUESTION, 0.99),
        ("20_30", SOURCE_PROMPT_DELTA, 0.5),
        ("30_40", SOURCE_TRACE_STRICT, None),
        ("40_50", SOURCE_FRAME_INSPECTION, None),
    ])
    out = apply_budget(r, SelectionBudget(max_recaption_fraction=0.3), total_clips=10)
    # limit = 3: frame inspection > strict trace > prompt delta
    assert set(out.clip_ids) == {"40_50", "30_40", "20_30"}
    assert set(out.metadata["budget"]["dropped"]) == {"0_10", "10_20"}


def test_score_breaks_ties_within_class():
    r = _result([
        ("0_10", SOURCE_QUESTION, 0.2),
        ("10_20", SOURCE_QUESTION, 0.8),
    ])
    out = apply_budget(r, SelectionBudget(max_recaption_fraction=0.1), total_clips=10)
    assert out.clip_ids == ["10_20"]


def test_provenance_preserved_after_budget():
    r = _result([("0_10", SOURCE_FRAME_INSPECTION, None),
                 ("0_10", SOURCE_QUESTION, 0.3)])
    out = apply_budget(r, SelectionBudget(max_recaption_fraction=0.1), total_clips=10)
    assert set(out.sources["0_10"]) == {SOURCE_FRAME_INSPECTION, SOURCE_QUESTION}


def test_max_total_clips_caps():
    r = _result([(f"{i*10}_{(i+1)*10}", SOURCE_QUESTION, 1.0 - i * 0.1)
                 for i in range(5)])
    out = apply_budget(
        r, SelectionBudget(max_recaption_fraction=1.0, max_total_clips=2),
        total_clips=10)
    assert len(out.clip_ids) == 2
