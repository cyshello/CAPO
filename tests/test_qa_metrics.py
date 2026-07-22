from surrogate_rollout.evaluation.qa_metrics import parse_letter, score_mcq


def test_parse_letter():
    assert parse_letter("C") == "C"
    assert parse_letter("The answer is (B).") == "B"
    assert parse_letter("  d  ") is None  # lowercase not accepted (baseline parity)
    assert parse_letter("") is None
    assert parse_letter(None) is None


def test_score_distinguishes_parse_from_semantic_failure():
    assert score_mcq("C", "C") == (1.0, "C", "")
    assert score_mcq("A", "C") == (0.0, "A", "wrong_answer")
    assert score_mcq("no letter here... 42", "C") == (0.0, None, "parse_failure")
    assert score_mcq(None, "C") == (0.0, None, "parse_failure")
    assert score_mcq("C", None) == (0.0, "C", "no_gold")
