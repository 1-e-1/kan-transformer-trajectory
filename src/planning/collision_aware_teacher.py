from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .expert_interface import ExpertInterface, ExpertPlan
from .rrt_connect import PeriodicRRTConnect


class CollisionAwareTeacher(ExpertInterface):
    """Interface boundary for a locally available planner selected after capability probing."""

    def __init__(self, backend=None):
        self.backend = backend

    def plan(self, q_start, q_goal, **kwargs):
        if self.backend is None:
            raise RuntimeError("No validated local collision-aware planner backend is configured")
        return self.backend.plan(q_start, q_goal, **kwargs)


class DifferentiableCollisionCost:
    """Reserved interface; boolean PhysX contacts are intentionally not backpropagated."""

    def __call__(self, predicted_trajectory, scene_geometry):
        raise NotImplementedError("Configure a genuinely differentiable distance/SDF surrogate before use")


@dataclass
class DualExpertPlan:
    left: ExpertPlan
    right: ExpertPlan
    collision_checks: int
    iterations: int
    used_direct_path: bool


class PhysXDualArmTeacher:
    """16D RRT-Connect teacher using live PhysX self/inter collision checks.

    The expert searches the coupled configuration space for safety only.  Its
    trajectory is split into two independent 8D supervised action streams.
    """

    def __init__(self, scene, seed: int = 42, max_iterations: int = 1200):
        self.scene = scene
        self.seed = seed
        self.max_iterations = max_iterations
        self._validity_cache: dict[bytes, bool] = {}

    def _state_valid(self, q: np.ndarray) -> bool:
        key = np.round(q, 5).astype(np.float32).tobytes()
        if key in self._validity_cache:
            return self._validity_cache[key]
        self.scene.teleport(
            torch.as_tensor(q[:8], dtype=torch.float32, device=self.scene.device),
            torch.as_tensor(q[8:], dtype=torch.float32, device=self.scene.device),
            settle_steps=1,
        )
        labels = self.scene.monitor.labels()
        valid = not any(labels.values())
        self._validity_cache[key] = valid
        return valid

    def plan(self, q_start: dict[str, np.ndarray], q_goal: dict[str, np.ndarray]) -> DualExpertPlan:
        self._validity_cache.clear()
        start = np.concatenate((q_start["left"], q_start["right"])).astype(np.float32)
        goal = np.concatenate((q_goal["left"], q_goal["right"])).astype(np.float32)
        for endpoint in (start, goal):
            self.scene.teleport(
                torch.as_tensor(endpoint[:8], dtype=torch.float32, device=self.scene.device),
                torch.as_tensor(endpoint[8:], dtype=torch.float32, device=self.scene.device),
                settle_steps=3,
            )
            labels = self.scene.monitor.labels()
            valid = not any(labels.values())
            self._validity_cache[np.round(endpoint, 5).astype(np.float32).tobytes()] = valid
            if not valid:
                raise ValueError("PhysX settled endpoint audit failed")
        rng = np.random.default_rng(self.seed)

        def informed_sample():
            # Most hard cases need a local detour around a narrow contact region.
            alpha = rng.uniform(0.0, 1.0)
            shortest = (goal - start + np.pi) % (2.0 * np.pi) - np.pi
            corridor = start + alpha * shortest
            if rng.random() < 0.8:
                corridor += rng.normal(0.0, 0.65, size=16)
                return ((corridor + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
            return rng.uniform(-np.pi, np.pi, 16).astype(np.float32)

        planner = PeriodicRRTConnect(
            self._state_valid,
            dimension=16,
            seed=self.seed,
            extension_step=0.5,
            # Check more finely than the 5-degree output/action spacing.  The old
            # 0.18-rad step could skip a narrow contact seen by the hard-task gate
            # and incorrectly relabel a direct-colliding task as a direct path.
            collision_check_step=min(0.04, self.scene.action_delta_max),
            output_step=self.scene.action_delta_max,
            max_iterations=self.max_iterations,
            sampler=informed_sample,
        )
        result = planner.plan(start, goal)
        plans = {}
        for side, slc in (("left", slice(0, 8)), ("right", slice(8, 16))):
            positions = result.positions[:, slc]
            plans[side] = ExpertPlan(
                positions=positions,
                actions=np.diff(positions, axis=0),
                dt=self.scene.control_dt,
                collision_free=True,
                planner="physx_rrt_connect_16d",
            )
        return DualExpertPlan(plans["left"], plans["right"], result.collision_checks, result.iterations, result.used_direct_path)


class PhysXSingleArmTeacher:
    """8D PhysX/RRT teacher with the non-planned arm held at one fixed pose.

    This is the planning authority for the left-only data path.  A configuration
    is rejected for left self-contact, left/right contact, or contact between the
    left arm and any fixed robot/scene body.  Right-only contacts are deliberately
    excluded because the right arm is not an action variable and is pinned at its
    audited hold configuration.
    """

    def __init__(
        self,
        scene,
        side: str = "left",
        hold_q: np.ndarray | None = None,
        seed: int = 42,
        max_iterations: int = 1200,
        output_step: float | None = None,
    ):
        if side not in {"left", "right"}:
            raise ValueError(f"side must be left or right, got {side!r}")
        self.scene = scene
        self.side = side
        self.other = "right" if side == "left" else "left"
        self.hold_q = np.zeros(8, dtype=np.float32) if hold_q is None else np.asarray(hold_q, dtype=np.float32)
        if self.hold_q.shape != (8,):
            raise ValueError("hold_q must have shape [8]")
        self.seed = int(seed)
        self.max_iterations = int(max_iterations)
        self.output_step = float(output_step or min(np.deg2rad(3.5), scene.action_delta_max))
        self._validity_cache: dict[bytes, bool] = {}

    def _teleport(self, q: np.ndarray, settle_steps: int) -> None:
        moving = torch.as_tensor(q, dtype=torch.float32, device=self.scene.device)
        held = torch.as_tensor(self.hold_q, dtype=torch.float32, device=self.scene.device)
        if self.side == "left":
            self.scene.teleport(moving, held, settle_steps=settle_steps)
        else:
            self.scene.teleport(held, moving, settle_steps=settle_steps)

    def labels(self, q: np.ndarray, settle_steps: int = 1) -> tuple[bool, bool]:
        self._teleport(q, settle_steps=settle_steps)
        labels = self.scene.monitor.labels()
        self_collision = bool(labels[f"{self.side}_self"])
        external_collision = bool(labels["inter_arm"] or labels[f"{self.side}_environment"])
        return self_collision, external_collision

    def state_valid(self, q: np.ndarray) -> bool:
        q = np.asarray(q, dtype=np.float32)
        key = np.round(q, 5).tobytes()
        if key not in self._validity_cache:
            # Two polls are required for a deterministic contact transition in
            # the pinned kinematic GPU pipeline; a single poll can lag behind a
            # teleport when leaving or entering a fixed-body contact.
            self._validity_cache[key] = not any(self.labels(q, settle_steps=2))
        return self._validity_cache[key]

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> ExpertPlan:
        self._validity_cache.clear()
        start = np.asarray(q_start, dtype=np.float32)
        goal = np.asarray(q_goal, dtype=np.float32)
        if start.shape != (8,) or goal.shape != (8,):
            raise ValueError("Single-arm endpoints must have shape [8]")
        for name, endpoint in (("start", start), ("goal", goal)):
            labels = self.labels(endpoint, settle_steps=3)
            self._validity_cache[np.round(endpoint, 5).tobytes()] = not any(labels)
            if any(labels):
                raise ValueError(f"Settled {self.side} {name} endpoint audit failed: labels={labels}")

        rng = np.random.default_rng(self.seed)
        shortest = (goal - start + np.pi) % (2.0 * np.pi) - np.pi

        def informed_sample() -> np.ndarray:
            alpha = rng.uniform(0.0, 1.0)
            if rng.random() < 0.8:
                corridor = start + alpha * shortest + rng.normal(0.0, 0.65, size=8)
                return ((corridor + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
            return rng.uniform(-np.pi, np.pi, 8).astype(np.float32)

        planner = PeriodicRRTConnect(
            self.state_valid,
            dimension=8,
            seed=self.seed,
            extension_step=0.5,
            collision_check_step=min(0.04, self.scene.action_delta_max),
            output_step=self.output_step,
            max_iterations=self.max_iterations,
            sampler=informed_sample,
        )
        result = planner.plan(start, goal)
        self.last_result = result
        return ExpertPlan(
            positions=result.positions,
            actions=np.diff(result.positions, axis=0),
            dt=self.scene.control_dt,
            collision_free=True,
            planner="physx_rrt_connect_8d_fixed_other",
        )
