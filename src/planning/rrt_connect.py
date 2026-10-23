"""Dependency-free periodic joint-space RRT-Connect.

The planner is deliberately geometry-agnostic.  A validated state callback (PhysX
in this project) is the only authority on self/inter-arm collision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.geometry.periodic import periodic_delta


StateValidator = Callable[[np.ndarray], bool]
Sampler = Callable[[], np.ndarray]


def wrap_to_pi(q: np.ndarray) -> np.ndarray:
    return ((np.asarray(q, dtype=np.float32) + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def periodic_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(periodic_delta(a, b)))


def interpolate_periodic(a: np.ndarray, b: np.ndarray, max_step: float) -> np.ndarray:
    delta = np.asarray(periodic_delta(b, a), dtype=np.float32)
    # Tolerance avoids an extra segment from float32 round-off at exact limits.
    count = max(1, int(np.ceil((np.max(np.abs(delta)) - 1e-6) / max_step)))
    alpha = np.linspace(0.0, 1.0, count + 1, dtype=np.float32)
    return wrap_to_pi(a[None] + alpha[:, None] * delta[None])


@dataclass
class RRTResult:
    positions: np.ndarray
    collision_checks: int
    iterations: int
    used_direct_path: bool


class _Tree:
    def __init__(self, root: np.ndarray, label: str):
        self.nodes = [root.copy()]
        self.parents = [-1]
        self.label = label

    def nearest(self, q: np.ndarray) -> int:
        distances = [periodic_distance(node, q) for node in self.nodes]
        return int(np.argmin(distances))

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(q.copy())
        self.parents.append(parent)
        return len(self.nodes) - 1

    def trace(self, index: int) -> list[np.ndarray]:
        result = []
        while index >= 0:
            result.append(self.nodes[index])
            index = self.parents[index]
        return list(reversed(result))


class PeriodicRRTConnect:
    """Bidirectional RRT-Connect over [-pi, pi)^D with checked edges."""

    def __init__(
        self,
        state_valid: StateValidator,
        dimension: int,
        seed: int = 42,
        extension_step: float = 0.35,
        collision_check_step: float = 0.12,
        output_step: float = np.deg2rad(5.0),
        max_iterations: int = 1200,
        goal_bias: float = 0.15,
        sampler: Sampler | None = None,
    ):
        self.state_valid = state_valid
        self.dimension = dimension
        self.rng = np.random.default_rng(seed)
        self.extension_step = float(extension_step)
        self.collision_check_step = float(collision_check_step)
        self.output_step = float(output_step)
        self.max_iterations = int(max_iterations)
        self.goal_bias = float(goal_bias)
        self.sampler = sampler
        self.collision_checks = 0

    def _valid(self, q: np.ndarray) -> bool:
        self.collision_checks += 1
        return bool(self.state_valid(wrap_to_pi(q)))

    def _edge_valid(self, a: np.ndarray, b: np.ndarray) -> bool:
        states = interpolate_periodic(a, b, self.collision_check_step)
        return all(self._valid(q) for q in states[1:])

    def _steer(self, source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        delta = np.asarray(periodic_delta(target, source), dtype=np.float32)
        distance = float(np.linalg.norm(delta))
        if distance <= self.extension_step:
            return wrap_to_pi(target), True
        return wrap_to_pi(source + delta * (self.extension_step / distance)), False

    def _extend(self, tree: _Tree, target: np.ndarray) -> tuple[str, int]:
        near_index = tree.nearest(target)
        q_new, reached = self._steer(tree.nodes[near_index], target)
        if not self._edge_valid(tree.nodes[near_index], q_new):
            return "trapped", near_index
        new_index = tree.add(q_new, near_index)
        return ("reached" if reached else "advanced"), new_index

    def _connect(self, tree: _Tree, target: np.ndarray) -> tuple[str, int]:
        while True:
            status, index = self._extend(tree, target)
            if status != "advanced":
                return status, index

    @staticmethod
    def _assemble(a: _Tree, ia: int, b: _Tree, ib: int) -> list[np.ndarray]:
        trace_a, trace_b = a.trace(ia), b.trace(ib)
        if a.label == "start":
            return trace_a + list(reversed(trace_b))[1:]
        return trace_b + list(reversed(trace_a))[1:]

    def _densify(self, sparse: list[np.ndarray]) -> np.ndarray:
        dense = [sparse[0]]
        for begin, end in zip(sparse[:-1], sparse[1:]):
            dense.extend(interpolate_periodic(begin, end, self.output_step)[1:])
        return np.asarray(dense, dtype=np.float32)

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> RRTResult:
        start, goal = wrap_to_pi(q_start), wrap_to_pi(q_goal)
        if start.shape != (self.dimension,) or goal.shape != (self.dimension,):
            raise ValueError(f"Expected [{self.dimension}] endpoints")
        self.collision_checks = 0
        if not self._valid(start) or not self._valid(goal):
            raise ValueError("RRT endpoints must be collision-free")
        if self._edge_valid(start, goal):
            return RRTResult(self._densify([start, goal]), self.collision_checks, 0, True)

        tree_a, tree_b = _Tree(start, "start"), _Tree(goal, "goal")
        for iteration in range(1, self.max_iterations + 1):
            if self.rng.random() < self.goal_bias:
                target = tree_b.nodes[0]
            else:
                target = self.sampler() if self.sampler is not None else self.rng.uniform(-np.pi, np.pi, self.dimension).astype(np.float32)
            status_a, index_a = self._extend(tree_a, target)
            if status_a != "trapped":
                status_b, index_b = self._connect(tree_b, tree_a.nodes[index_a])
                if status_b == "reached":
                    sparse = self._assemble(tree_a, index_a, tree_b, index_b)
                    dense = self._densify(sparse)
                    # A final independent audit protects against reconstruction bugs.
                    if all(self._edge_valid(x, y) for x, y in zip(dense[:-1], dense[1:])):
                        return RRTResult(dense, self.collision_checks, iteration, False)
            tree_a, tree_b = tree_b, tree_a
        raise RuntimeError(f"RRT-Connect failed after {self.max_iterations} iterations")
