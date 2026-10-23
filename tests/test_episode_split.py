import numpy as np
import pytest

from src.data.splits import episode_group_split, extend_episode_group_split


def test_episode_split_has_no_group_leakage_and_is_deterministic():
    groups = np.repeat([f"episode_{i}" for i in range(10)], [2, 3, 4, 2, 5, 3, 2, 4, 3, 2])
    indices_a, names_a = episode_group_split(groups, seed=9)
    indices_b, names_b = episode_group_split(groups, seed=9)
    assert names_a == names_b
    assert indices_a == indices_b
    sets = {name: set(groups[index] for index in indices) for name, indices in indices_a.items()}
    assert sets["train"].isdisjoint(sets["validation"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])
    assert sum(len(value) for value in indices_a.values()) == len(groups)


def test_dagger_groups_are_forced_to_train():
    groups = np.array([f"episode_{i}" for i in range(8)] + ["dagger_iter9|seed=1|episode=0"])
    _, names = episode_group_split(groups, seed=3)
    assert "dagger_iter9|seed=1|episode=0" in names["train"]
    assert "dagger_iter9|seed=1|episode=0" not in names["validation"] + names["test"]


def test_extending_split_never_moves_existing_holdout_groups():
    initial = np.array([f"episode_{i}" for i in range(10)])
    _, names = episode_group_split(initial, seed=3)
    extended_ids = np.r_[initial, [f"new_{i}" for i in range(10)], ["dagger_new|seed=1|episode=0"]]
    _, extended = extend_episode_group_split(extended_ids, names, seed=3)
    for split in ("train", "validation", "test"):
        assert set(names[split]).issubset(set(extended[split]))
    assert "dagger_new|seed=1|episode=0" in extended["train"]


def test_extend_split_allows_only_replaced_forced_train_groups_to_disappear():
    existing = {
        "train": ["regular_train", "dagger_old|seed=1|episode=0"],
        "validation": ["regular_validation"],
        "test": ["regular_test"],
    }
    current = np.asarray([
        "regular_train", "regular_validation", "regular_test",
        "dagger_new|seed=2|episode=0",
    ])
    _, groups = extend_episode_group_split(current, existing, seed=42)
    assert "dagger_old|seed=1|episode=0" not in groups["train"]
    assert "dagger_new|seed=2|episode=0" in groups["train"]


def test_extend_split_rejects_missing_evaluation_group():
    existing = {
        "train": ["regular_train"],
        "validation": ["regular_validation"],
        "test": ["regular_test"],
    }
    current = np.asarray(["regular_train", "regular_test", "dagger_new"])
    with pytest.raises(ValueError, match="non-replaceable"):
        extend_episode_group_split(current, existing, seed=42)
