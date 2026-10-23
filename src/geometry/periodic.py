"""Continuous-joint representations and shortest periodic differences."""

from __future__ import annotations

import numpy as np
import torch


def periodic_delta(goal, current):
    """Elementwise shortest signed angle from ``current`` to ``goal`` in [-pi, pi]."""
    delta = goal - current
    if isinstance(delta, torch.Tensor):
        return torch.atan2(torch.sin(delta), torch.cos(delta))
    return np.arctan2(np.sin(delta), np.cos(delta))


def sincos_encode(q):
    """Interleave ``sin(q_i), cos(q_i)`` along the last dimension."""
    if isinstance(q, torch.Tensor):
        return torch.stack((torch.sin(q), torch.cos(q)), dim=-1).flatten(-2)
    return np.stack((np.sin(q), np.cos(q)), axis=-1).reshape(*q.shape[:-1], -1)

