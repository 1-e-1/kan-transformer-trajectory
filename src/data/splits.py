"""Leakage-safe dataset splits at complete episode/task granularity."""

from __future__ import annotations

import numpy as np


def episode_group_split(group_ids, seed=42, train_ratio=0.7, validation_ratio=0.15, forced_train_prefixes=("dagger_", "stage2_rrt_seed43")):
    group_ids = np.asarray(group_ids)
    unique = np.unique(group_ids)
    if unique.size < 3:
        raise ValueError("At least three independent episode groups are required for train/validation/test")
    forced = np.asarray([group for group in unique if group.startswith(forced_train_prefixes)], dtype=unique.dtype)
    eligible = np.asarray([group for group in unique if not group.startswith(forced_train_prefixes)], dtype=unique.dtype)
    if eligible.size < 3:
        raise ValueError("At least three non-forced episode groups are required")
    shuffled = np.random.default_rng(seed).permutation(eligible)
    train_end = max(1, min(len(shuffled) - 2, int(round(train_ratio * len(shuffled)))))
    validation_count = max(1, int(round(validation_ratio * len(shuffled))))
    validation_end = min(len(shuffled) - 1, train_end + validation_count)
    groups = {
        "train": np.concatenate((forced, shuffled[:train_end])),
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    if any(len(value) == 0 for value in groups.values()):
        raise RuntimeError(f"Empty episode split: {groups}")
    indices = {name: np.flatnonzero(np.isin(group_ids, values)).tolist() for name, values in groups.items()}
    intersections = [set(groups[a]).intersection(groups[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))]
    if any(intersections):
        raise AssertionError("Episode leakage detected")
    return indices, {name: values.tolist() for name, values in groups.items()}


def extend_episode_group_split(group_ids, existing_groups, seed=42, train_ratio=0.7, validation_ratio=0.15, forced_train_prefixes=("dagger_", "stage2_rrt_seed43")):
    """Freeze prior assignments and distribute only newly arrived groups.

    Versioned DAgger/recovery files may replace their internal episode IDs.
    Missing groups are therefore pruned only when they were forced-train groups;
    a missing ordinary train group or any validation/test group remains a hard
    error because accepting it could silently change an evaluation split.
    """
    group_ids = np.asarray(group_ids)
    current = set(np.unique(group_ids).tolist())
    groups = {name: list(values) for name, values in existing_groups.items()}
    assigned = set().union(*(set(values) for values in groups.values()))
    missing_by_split = {
        name: set(values) - current for name, values in groups.items()
    }
    unsafe_missing = set(missing_by_split["validation"]) | set(missing_by_split["test"])
    unsafe_missing |= {
        group for group in missing_by_split["train"]
        if not group.startswith(forced_train_prefixes)
    }
    if unsafe_missing:
        raise ValueError(
            "Previously assigned non-replaceable groups disappeared: "
            f"{sorted(unsafe_missing)[:3]}"
        )
    groups["train"] = [group for group in groups["train"] if group in current]
    assigned = set().union(*(set(values) for values in groups.values()))
    new = sorted(current - assigned)
    forced = [group for group in new if group.startswith(forced_train_prefixes)]
    eligible = [group for group in new if not group.startswith(forced_train_prefixes)]
    groups["train"].extend(forced)
    if eligible:
        shuffled = np.random.default_rng(seed + len(assigned)).permutation(eligible).tolist()
        if len(shuffled) < 3:
            groups["train"].extend(shuffled)
        else:
            train_end = max(1, min(len(shuffled) - 2, int(round(train_ratio * len(shuffled)))))
            validation_count = max(1, int(round(validation_ratio * len(shuffled))))
            validation_end = min(len(shuffled) - 1, train_end + validation_count)
            groups["train"].extend(shuffled[:train_end])
            groups["validation"].extend(shuffled[train_end:validation_end])
            groups["test"].extend(shuffled[validation_end:])
    sets = {name: set(values) for name, values in groups.items()}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise AssertionError("Extended split leakage detected")
    indices = {name: np.flatnonzero(np.isin(group_ids, values)).tolist() for name, values in groups.items()}
    return indices, groups
