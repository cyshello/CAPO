import pytest

from surrogate_rollout.scripts.make_split_manifest import make_split

VIDEOS = [f"vid{i:03d}" for i in range(60)]
PINNED = ["vid001", "vid002", "vid003", "vid004", "vid005"]


def test_split_sizes_and_pinning():
    s = make_split(VIDEOS, PINNED, per_split=10, seed=0)
    assert len(s["train"]) == len(s["validation"]) == len(s["test"]) == 10
    for v in PINNED:
        assert v in s["train"]
        assert v not in s["validation"]
        assert v not in s["test"]


def test_video_level_disjointness():
    s = make_split(VIDEOS, PINNED, per_split=10, seed=0)
    train, val, test = map(set, (s["train"], s["validation"], s["test"]))
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_deterministic_and_order_independent():
    a = make_split(VIDEOS, PINNED, per_split=10, seed=0)
    b = make_split(list(reversed(VIDEOS)), PINNED, per_split=10, seed=0)
    assert a == b
    c = make_split(VIDEOS, PINNED, per_split=10, seed=1)
    assert a != c


def test_pool_too_small_raises():
    with pytest.raises(ValueError):
        make_split(VIDEOS[:20], PINNED, per_split=10, seed=0)


def test_missing_pinned_video_raises():
    with pytest.raises(ValueError):
        make_split(VIDEOS, ["not_in_dataset"], per_split=10, seed=0)
