"""Collision-path preserving time parameterization for periodic joint paths."""

from __future__ import annotations

import numpy as np

from src.geometry.periodic import periodic_delta
from src.planning.rrt_connect import wrap_to_pi


def _unwrap_path(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 8 or len(positions) < 2:
        raise ValueError("positions must have shape [N>=2, 8]")
    result = [positions[0].astype(np.float64)]
    for target in positions[1:]:
        result.append(result[-1] + np.asarray(periodic_delta(target, wrap_to_pi(result[-1])), dtype=np.float64))
    return np.asarray(result, dtype=np.float64)


def simplify_rrt_polyline(positions: np.ndarray, cosine_threshold: float = 0.999999) -> np.ndarray:
    """Remove only densification points that lie on the same RRT line segment."""
    path = _unwrap_path(positions)
    nonzero = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-8]
    path = path[nonzero]
    if len(path) <= 2:
        return path
    keep = [0]
    for index in range(1, len(path) - 1):
        before = path[index] - path[index - 1]
        after = path[index + 1] - path[index]
        cosine = float(np.dot(before, after) / max(np.linalg.norm(before) * np.linalg.norm(after), 1e-12))
        if cosine < cosine_threshold:
            keep.append(index)
    keep.append(len(path) - 1)
    return path[np.asarray(keep)]


def _linf_segment_lengths(path: np.ndarray) -> np.ndarray:
    """Return path length in the metric induced by per-joint action limits."""
    return np.max(np.abs(np.diff(path, axis=0)), axis=1)


def _sample_polyline(path: np.ndarray, distances: np.ndarray) -> np.ndarray:
    # The actuator constraint is an infinity-norm constraint (every joint has
    # the same per-step cap), so L-infinity arc length is the appropriate
    # clock.  L2 arc length over-stretches simultaneous multi-joint motion by
    # as much as sqrt(8), which made the learned closed loop unnecessarily
    # slow even though every individual joint was far below its limit.
    segment_lengths = _linf_segment_lengths(path)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    result = []
    for distance in distances:
        segment = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(segment_lengths) - 1)
        segment = max(segment, 0)
        length = segment_lengths[segment]
        alpha = 0.0 if length <= 1e-12 else (distance - cumulative[segment]) / length
        result.append(path[segment] + np.clip(alpha, 0.0, 1.0) * (path[segment + 1] - path[segment]))
    return np.asarray(result, dtype=np.float64)


def retime_rrt_path(
    positions: np.ndarray,
    max_action_rad: float = np.deg2rad(3.5),
    terminal_transitions: int = 12,
    terminal_distance_actions: float = 3.0,
    terminal_fraction_limit: float = 0.40,
) -> np.ndarray:
    """Cruise along RRT geometry and smoothly decelerate near the goal.

    This is a kinematic trajectory generator, so forcing zero initial velocity
    only teaches the policy to emit tiny first actions and wastes the online
    control budget.  The clock therefore starts at cruise speed and applies a
    cubic smoothstep velocity taper only over a bounded terminal distance.
    It is parameterized by L-infinity joint-space arc length to match the
    per-joint action cap.  Unlike a fixed minimum transition count, this keeps
    short paths assertive while reserving a fixed precision tail near the goal.
    """
    if terminal_transitions < 2:
        raise ValueError("terminal_transitions must be at least 2")
    if terminal_distance_actions <= 0.0:
        raise ValueError("terminal_distance_actions must be positive")
    if not 0.0 < terminal_fraction_limit < 1.0:
        raise ValueError("terminal_fraction_limit must be in (0, 1)")
    path = simplify_rrt_polyline(positions)
    lengths = _linf_segment_lengths(path)
    total = float(lengths.sum())
    if total <= 1e-10:
        return np.asarray([wrap_to_pi(path[0]), wrap_to_pi(path[-1])], dtype=np.float32)
    terminal_distance = min(
        terminal_fraction_limit * total,
        terminal_distance_actions * max_action_rad,
    )
    cruise_distance = total - terminal_distance
    cruise_transitions = max(1, int(np.ceil(cruise_distance / max_action_rad)))
    cruise_distances = np.linspace(
        0.0, cruise_distance, cruise_transitions + 1, dtype=np.float64
    )

    # Midpoint samples of a cubic smoothstep velocity give a monotonic taper
    # with nonzero finite increments and an exact endpoint after normalization.
    t = (np.arange(terminal_transitions, dtype=np.float64) + 0.5) / terminal_transitions
    terminal_weights = 1.0 - 3.0 * t**2 + 2.0 * t**3
    terminal_increments = terminal_distance * terminal_weights / terminal_weights.sum()
    terminal_distances = cruise_distance + np.cumsum(terminal_increments)
    distances = np.concatenate((cruise_distances, terminal_distances))
    distances[-1] = total
    wrapped = wrap_to_pi(_sample_polyline(path, distances))
    wrapped[0] = wrap_to_pi(positions[0])
    wrapped[-1] = wrap_to_pi(positions[-1])
    actions = np.asarray([
        periodic_delta(b, a) for a, b in zip(wrapped[:-1], wrapped[1:])
    ])
    maximum = float(np.max(np.abs(actions)))
    if maximum > max_action_rad + 1e-6:
        raise RuntimeError(
            f"Retimer violated per-joint action cap: {maximum} > {max_action_rad}"
        )
    return wrapped.astype(np.float32)


def select_training_frames(
    actions: np.ndarray,
    tcp_goal_distances_m: np.ndarray,
    max_gap: int = 4,
    action_novelty_rad: float = np.deg2rad(0.35),
    precision_per_band: int = 8,
) -> np.ndarray:
    """Thin straight repeated controls while retaining turns and precision tails."""
    actions = np.asarray(actions, dtype=np.float32)
    distances = np.asarray(tcp_goal_distances_m, dtype=np.float64)
    if len(actions) != len(distances):
        raise ValueError("actions and tcp_goal_distances_m must have equal length")
    if not len(actions):
        return np.empty(0, dtype=np.int64)
    keep = {0, len(actions) - 1}
    last = 0
    for index in range(1, len(actions) - 1):
        direction_changed = False
        norm_a = float(np.linalg.norm(actions[index - 1]))
        norm_b = float(np.linalg.norm(actions[index]))
        if norm_a > 1e-8 and norm_b > 1e-8:
            direction_changed = float(np.dot(actions[index - 1], actions[index]) / (norm_a * norm_b)) < 0.9995
        novelty = float(np.max(np.abs(actions[index] - actions[last]))) >= action_novelty_rad
        if direction_changed or novelty or index - last >= max_gap:
            keep.add(index)
            last = index

    # Disjoint annuli provide genuine examples at all requested precision scales.
    bands = ((0.010, 0.030), (0.003, 0.010), (0.0, 0.003))
    for low, high in bands:
        candidates = np.flatnonzero((distances >= low) & (distances < high))
        if len(candidates):
            chosen = candidates[np.linspace(0, len(candidates) - 1, min(precision_per_band, len(candidates))).round().astype(int)]
            keep.update(int(index) for index in chosen)
    return np.asarray(sorted(keep), dtype=np.int64)
