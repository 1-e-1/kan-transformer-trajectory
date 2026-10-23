"""Runtime goal switching policy for closed-loop inference."""

from __future__ import annotations

import numpy as np

from src.geometry.periodic import periodic_delta


class GoalManager:
    def __init__(self, q_goal, x_goal, joint_jump_threshold=0.5, position_jump_threshold=0.1):
        self.q_goal = np.asarray(q_goal, dtype=np.float32)
        self.x_goal = np.asarray(x_goal, dtype=np.float32)
        self.joint_jump_threshold = joint_jump_threshold
        self.position_jump_threshold = position_jump_threshold
        self.reset_events = 0

    def update(self, q_goal, x_goal, current_q, current_x, history):
        q_goal = np.asarray(q_goal, dtype=np.float32)
        x_goal = np.asarray(x_goal, dtype=np.float32)
        large_jump = (
            np.max(np.abs(periodic_delta(q_goal, self.q_goal))) > self.joint_jump_threshold
            or np.linalg.norm(x_goal[:3] - self.x_goal[:3]) > self.position_jump_threshold
        )
        self.q_goal, self.x_goal = q_goal, x_goal
        if large_jump:
            history.clear()
            self.reset_events += 1
            return {"goal_reset_event": True, "q_start": np.asarray(current_q, dtype=np.float32), "x_start": np.asarray(current_x, dtype=np.float32)}
        return {"goal_reset_event": False}
