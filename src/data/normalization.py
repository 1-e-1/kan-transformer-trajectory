"""Train-split-only, side-specific observation normalization."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _stats(values, epsilon=1e-5):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": values.mean(axis=0).astype(np.float32).tolist(), "scale": np.maximum(values.std(axis=0), epsilon).astype(np.float32).tolist()}


def compute_normalization(dataset, indices):
    index = np.asarray(indices, dtype=np.int64)
    arrays = dataset.arrays
    positions = np.concatenate([arrays[key][index, :3] for key in ("x_start", "x_current", "x_goal")], axis=0)
    history = arrays["history_action"][index]
    mask = arrays["history_mask"][index].astype(bool)
    valid_history = history[mask] if mask.any() else np.zeros((1, 8), dtype=np.float32)
    return {
        "position": _stats(positions),
        "delta_q_goal": _stats(arrays["delta_q_goal"][index]),
        "delta_x_goal": _stats(arrays["delta_x_goal"][index]),
        "other_arm_geometry": _stats(arrays["other_arm_geometry"][index].reshape(len(index), -1)),
        "history_action": _stats(valid_history),
    }


class ObservationNormalizer(nn.Module):
    def __init__(self, stats=None):
        super().__init__()
        stats = stats or {}
        dimensions = {"position": 3, "delta_q_goal": 8, "delta_x_goal": 6, "other_arm_geometry": 27, "history_action": 8}
        for name, dimension in dimensions.items():
            values = stats.get(name, {"mean": [0.0] * dimension, "scale": [1.0] * dimension})
            self.register_buffer(f"{name}_mean", torch.tensor(values["mean"], dtype=torch.float32), persistent=False)
            self.register_buffer(f"{name}_scale", torch.tensor(values["scale"], dtype=torch.float32), persistent=False)

    def _normalize(self, value, name):
        return (value - getattr(self, f"{name}_mean")) / getattr(self, f"{name}_scale")

    def forward(self, batch):
        normalized = dict(batch)
        for key in ("x_start", "x_current", "x_goal"):
            normalized[key] = torch.cat((self._normalize(batch[key][..., :3], "position"), batch[key][..., 3:]), dim=-1)
        normalized["delta_q_goal"] = self._normalize(batch["delta_q_goal"], "delta_q_goal")
        normalized["delta_x_goal"] = self._normalize(batch["delta_x_goal"], "delta_x_goal")
        shape = batch["other_arm_geometry"].shape
        normalized["other_arm_geometry"] = self._normalize(batch["other_arm_geometry"].flatten(1), "other_arm_geometry").reshape(shape)
        history = self._normalize(batch["history_action"], "history_action")
        normalized["history_action"] = torch.where(batch["history_mask"].unsqueeze(-1), history, torch.zeros_like(history))
        return normalized
