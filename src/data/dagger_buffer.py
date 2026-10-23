from __future__ import annotations

from collections import deque

import numpy as np


class ActionHistory:
    """Per-arm realized-action FIFO with hard truncation and left padding."""

    def __init__(self, length: int, action_dim: int = 8):
        self.length = length
        self.action_dim = action_dim
        self._data = deque(maxlen=length)

    def clear(self) -> None:
        self._data.clear()

    def append(self, realized_action) -> None:
        action = np.asarray(realized_action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"Expected [{self.action_dim}], got {action.shape}")
        self._data.append(action.copy())

    def padded(self) -> tuple[np.ndarray, np.ndarray]:
        actions = np.zeros((self.length, self.action_dim), dtype=np.float32)
        mask = np.zeros(self.length, dtype=np.bool_)
        if self._data:
            values = np.stack(tuple(self._data))
            actions[-len(values) :] = values
            mask[-len(values) :] = True
        return actions, mask


class DAggerBuffer:
    """Append-only corrected supervised samples from policy-visited states."""

    def __init__(self):
        self.samples: list[dict] = []

    def append(self, observation: dict, expert_action) -> None:
        sample = dict(observation)
        sample["expert_action"] = np.asarray(expert_action, dtype=np.float32)
        self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

