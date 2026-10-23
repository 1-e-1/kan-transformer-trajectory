from __future__ import annotations

import numpy as np

from src.geometry.periodic import periodic_delta
from .expert_interface import ExpertInterface, ExpertPlan


def quintic_blend(tau: np.ndarray) -> np.ndarray:
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


class QuinticTeacher(ExpertInterface):
    """Shortest-periodic quintic Stage-1 expert (collision filtering is external)."""

    def __init__(self, steps: int = 64, dt: float = 1.0 / 30.0, action_delta_max: float | None = None):
        if steps < 2:
            raise ValueError("steps must be >=2")
        self.steps = steps
        self.dt = dt
        self.action_delta_max = action_delta_max

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray, **kwargs) -> ExpertPlan:
        q_start = np.asarray(q_start, dtype=np.float32)
        q_goal = np.asarray(q_goal, dtype=np.float32)
        if q_start.shape != (8,) or q_goal.shape != (8,):
            raise ValueError("q_start and q_goal must have shape [8]")
        shortest = periodic_delta(q_goal, q_start)
        tau = np.linspace(0.0, 1.0, self.steps, dtype=np.float32)
        positions = q_start[None] + quintic_blend(tau)[:, None] * shortest[None]
        actions = np.diff(positions, axis=0)
        if self.action_delta_max is not None and np.max(np.abs(actions)) > self.action_delta_max + 1e-7:
            raise ValueError("Teacher action exceeds action_delta_max; increase steps")
        return ExpertPlan(positions=positions, actions=actions, dt=self.dt, collision_free=True, planner="periodic_quintic")


def sample_joint_uniform(rng: np.random.Generator) -> np.ndarray:
    """Joint-space-uniform endpoint sample; not Cartesian-workspace uniform."""
    return rng.uniform(-np.pi, np.pi, size=8).astype(np.float32)

