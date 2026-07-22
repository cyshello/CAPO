from surrogate_rollout.evaluation.rollout_evaluator import SelectionBudget
from surrogate_rollout.selection.base import (
    SOURCE_PROMPT_DELTA,
    SOURCE_TRACE_RETURNED,
    SelectionContext,
    SelectionResult,
    merge_results,
)
from surrogate_rollout.selection.trace import TraceReferenceSelector
from surrogate_rollout.selection.union import (
    SELECTION_POLICIES,
    RandomBudgetMatchedSelector,
    UnionSelector,
    build_selector,
)

CLIPS = tuple(f"{i * 10}_{(i + 1) * 10}" for i in range(10))


def _context(**over):
    base = dict(
        video_id="v1", qa_id="qa1", question="q?", answer_options=("A", "B"),
        baseline_prompt="base", candidate_prompt="cand",
        baseline_trajectory=(), baseline_tool_events=(),
        all_clip_ids=CLIPS, reference_policy="returned_and_frame_inspection",
        budget=SelectionBudget(), visual_index=None,
    )
    base.update(over)
    return SelectionContext(**base)


class StubSelector:
    def __init__(self, clip_ids, source, scores=None):
        self.clip_ids, self.source, self._scores = clip_ids, source, scores or {}

    def select(self, context):
        r = SelectionResult()
        for c in self.clip_ids:
            r.add(c, self.source, self._scores.get(c))
        return r


def test_merge_results_unions_provenance():
    a = SelectionResult()
    a.add("0_10", SOURCE_TRACE_RETURNED)
    b = SelectionResult()
    b.add("0_10", SOURCE_PROMPT_DELTA, 0.7)
    b.add("10_20", SOURCE_PROMPT_DELTA, 0.5)
    merged = merge_results(a, b)
    assert set(merged.clip_ids) == {"0_10", "10_20"}
    assert merged.sources["0_10"] == [SOURCE_TRACE_RETURNED, SOURCE_PROMPT_DELTA]
    assert merged.scores["0_10"] == 0.7


def test_union_selector_combines_components():
    union = UnionSelector([
        StubSelector(["0_10"], SOURCE_TRACE_RETURNED),
        StubSelector(["0_10", "20_30"], SOURCE_PROMPT_DELTA, {"20_30": 0.9}),
    ])
    result = union.select(_context())
    assert set(result.clip_ids) == {"0_10", "20_30"}
    assert set(result.sources["0_10"]) == {SOURCE_TRACE_RETURNED, SOURCE_PROMPT_DELTA}


def test_build_selector_policy_shapes():
    assert isinstance(build_selector("trace_only"), TraceReferenceSelector)
    assert isinstance(build_selector("random_budget_matched"), RandomBudgetMatchedSelector)
    union = build_selector("trace_plus_prompt_delta_clip",
                           embed_texts_fn=lambda t: None)
    assert isinstance(union, UnionSelector)
    assert len(union.selectors) == 2
    for policy in SELECTION_POLICIES:
        assert build_selector(policy, embed_texts_fn=lambda t: None) is not None


def test_random_budget_matched_deterministic_and_sized():
    ctx = _context(budget_matched_count=3, random_seed=7)
    a = RandomBudgetMatchedSelector().select(ctx)
    b = RandomBudgetMatchedSelector().select(ctx)
    assert a.clip_ids == b.clip_ids
    assert len(a.clip_ids) == 3
    # different candidate prompt -> different deterministic draw allowed;
    # same inputs must match exactly (asserted above)
    c = RandomBudgetMatchedSelector().select(_context(
        budget_matched_count=3, random_seed=8))
    assert len(c.clip_ids) == 3
