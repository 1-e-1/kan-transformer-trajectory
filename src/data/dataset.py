from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


TENSOR_FIELDS = (
    "q_start",
    "q_current",
    "q_goal",
    "delta_q_goal",
    "x_start",
    "x_current",
    "x_goal",
    "delta_x_goal",
    "other_arm_geometry",
    "history_action",
    "history_mask",
    "expert_action",
    "collision_query_action",
    "self_collision_label",
    "inter_collision_label",
    "action_loss_mask",
    "dt",
)


def _resize_history_from_original_episode(data, requested_length: int):
    """Resize history and recover one older window from the same original NPZ.

    The source files store H=64. For H=128, sample ``t`` can recover the
    preceding 64 actions from the stored history of sample ``t-64`` in the
    same seed/episode. Missing or non-sequential provenance stays zero padded.
    """
    history = np.asarray(data["history_action"])
    mask = np.asarray(data["history_mask"], dtype=np.bool_)
    stored_length = history.shape[1]
    if stored_length >= requested_length:
        return history[:, -requested_length:], mask[:, -requested_length:]

    extended = np.zeros(
        (history.shape[0], requested_length, history.shape[2]), dtype=history.dtype
    )
    extended_mask = np.zeros((history.shape[0], requested_length), dtype=np.bool_)
    prefix_length = requested_length - stored_length
    extended[:, prefix_length:] = history
    extended_mask[:, prefix_length:] = mask

    required = {"episode_id", "step_id", "seed"}
    if prefix_length > stored_length or not required.issubset(data.files):
        return extended, extended_mask

    episode_ids = np.asarray(data["episode_id"])
    step_ids = np.asarray(data["step_id"])
    seeds = np.asarray(data["seed"])
    lookup = {
        (int(seed), int(episode), int(step)): index
        for index, (seed, episode, step) in enumerate(zip(seeds, episode_ids, step_ids))
    }
    for index, (seed, episode, step) in enumerate(zip(seeds, episode_ids, step_ids)):
        # A partially filled current window cannot have a contiguous older
        # window, so retain deterministic left padding in that case.
        if not mask[index].all():
            continue
        source = lookup.get((int(seed), int(episode), int(step) - stored_length))
        if source is None:
            continue
        extended[index, :prefix_length] = history[source, -prefix_length:]
        extended_mask[index, :prefix_length] = mask[source, -prefix_length:]
    return extended, extended_mask


class ArmTrajectoryDataset(Dataset):
    """Loads one side only; labels are always 8D, never concatenated to 16D."""

    def __init__(self, paths: str | Path | list[str | Path], limit: int | None = None, history_length: int | None = None):
        if isinstance(paths, (str, Path)):
            path = Path(paths)
            files = sorted(path.glob("*.npz")) if path.is_dir() else [path]
        else:
            files = [Path(path) for path in paths]
        if not files:
            raise FileNotFoundError(f"No dataset files found for {paths}")
        arrays = {field: [] for field in TENSOR_FIELDS}
        self.metadata = []
        group_ids = []
        for file in files:
            with np.load(file, allow_pickle=False) as data:
                # New kinematic replays store native H=128 while legacy files
                # remain H=64.  Keep the historical default H=64 for callers
                # that omit the argument, and normalize each file before the
                # cross-file concatenate so mixed-native-length datasets load.
                effective_history_length = history_length if history_length is not None else 64
                resized_history = _resize_history_from_original_episode(data, effective_history_length)
                for field in TENSOR_FIELDS:
                    if field == "history_action" and resized_history is not None:
                        arrays[field].append(resized_history[0])
                        continue
                    if field == "history_mask" and resized_history is not None:
                        arrays[field].append(resized_history[1])
                        continue
                    if field == "collision_query_action" and field not in data:
                        arrays[field].append(data["expert_action"])
                        continue
                    if field == "action_loss_mask" and field not in data:
                        arrays[field].append(np.ones(data["expert_action"].shape[0], dtype=np.float32))
                        continue
                    if field not in data:
                        raise KeyError(f"{file} missing {field}")
                    arrays[field].append(data[field])
                sample_count = int(data["expert_action"].shape[0])
                episode_ids = data["episode_id"] if "episode_id" in data else np.zeros(sample_count, dtype=np.int64)
                seeds = data["seed"] if "seed" in data else np.zeros(sample_count, dtype=np.int64)
                group_ids.extend([f"{file.name}|seed={int(seed)}|episode={int(episode)}" for seed, episode in zip(seeds, episode_ids)])
                self.metadata.append({"path": str(file), "samples": sample_count})
        self.arrays = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
        # Keep raw NPZ provenance immutable while repairing legacy near-pi SO(3)
        # targets with the current stable logarithm implementation.
        from src.geometry.rotations import pose_error6, rotation_6d_to_matrix

        current_pose = torch.from_numpy(self.arrays["x_current"])
        goal_pose = torch.from_numpy(self.arrays["x_goal"])
        self.arrays["delta_x_goal"] = pose_error6(
            current_pose[:, :3], rotation_6d_to_matrix(current_pose[:, 3:]),
            goal_pose[:, :3], rotation_6d_to_matrix(goal_pose[:, 3:]),
        ).numpy().astype(np.float32)
        self.group_ids = np.asarray(group_ids, dtype=str)
        if history_length is not None:
            current = self.arrays["history_action"].shape[1]
            if current > history_length:
                self.arrays["history_action"] = self.arrays["history_action"][:, -history_length:]
                self.arrays["history_mask"] = self.arrays["history_mask"][:, -history_length:]
            elif current < history_length:
                pad = history_length - current
                self.arrays["history_action"] = np.pad(self.arrays["history_action"], ((0, 0), (pad, 0), (0, 0)))
                self.arrays["history_mask"] = np.pad(self.arrays["history_mask"], ((0, 0), (pad, 0)), constant_values=False)
        if limit is not None:
            self.arrays = {key: value[:limit] for key, value in self.arrays.items()}
            self.group_ids = self.group_ids[:limit]
        lengths = {value.shape[0] for value in self.arrays.values()}
        if len(lengths) != 1:
            raise ValueError(f"Dataset fields have inconsistent lengths: {lengths}")

    def __len__(self):
        return self.arrays["expert_action"].shape[0]

    def __getitem__(self, index):
        item = {}
        for key, values in self.arrays.items():
            value = torch.as_tensor(values[index])
            item[key] = value.bool() if key == "history_mask" else value.float()
        return item
