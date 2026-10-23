"""Collision-filtered simultaneous Stage-1 data generation from live Isaac Sim FK."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.geometry.periodic import periodic_delta
from src.geometry.rotations import pose_error6, rotation_6d_to_matrix
from src.planning.quintic_teacher import QuinticTeacher, sample_joint_uniform
from .dagger_buffer import ActionHistory


@dataclass
class SamplingStats:
    sampling_attempts: int = 0
    accepted_episodes: int = 0
    self_collision_rejections: int = 0
    inter_arm_collision_rejections: int = 0
    trajectory_collision_rejections: int = 0

    def report(self) -> dict:
        attempts = max(1, self.sampling_attempts)
        return {
            **self.__dict__,
            "self_collision_rejection_rate": self.self_collision_rejections / attempts,
            "inter_arm_collision_rejection_rate": self.inter_arm_collision_rejections / attempts,
        }


def _cpu(array: torch.Tensor) -> np.ndarray:
    return array.detach().cpu().numpy().astype(np.float32)


def _delta_pose(current_pose9: torch.Tensor, goal_pose9: torch.Tensor) -> torch.Tensor:
    current_rotation = rotation_6d_to_matrix(current_pose9[:, 3:])
    goal_rotation = rotation_6d_to_matrix(goal_pose9[:, 3:])
    return pose_error6(current_pose9[:, :3], current_rotation, goal_pose9[:, :3], goal_rotation)


class Stage1DataGenerator:
    def __init__(self, scene, output_root: str | Path, history_length: int = 64, trajectory_steps: int = 80, seed: int = 42, max_sampling_attempts: int = 200):
        self.scene = scene
        self.output_root = Path(output_root)
        self.history_length = history_length
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.max_sampling_attempts = max_sampling_attempts
        self.teacher = QuinticTeacher(steps=trajectory_steps, dt=scene.control_dt, action_delta_max=scene.action_delta_max)
        self.stats = SamplingStats()

    def _endpoint_is_safe(self, q_left: np.ndarray, q_right: np.ndarray) -> bool:
        self.scene.teleport(torch.from_numpy(q_left), torch.from_numpy(q_right), settle_steps=3)
        labels = self.scene.monitor.labels()
        self_collision = labels["left_self"] or labels["right_self"]
        if self_collision:
            self.stats.self_collision_rejections += 1
        if labels["inter_arm"] or labels["left_environment"] or labels["right_environment"]:
            self.stats.inter_arm_collision_rejections += 1
        return not self_collision and not (
            labels["inter_arm"] or labels["left_environment"] or labels["right_environment"]
        )

    def _sample_safe_pair(self):
        for _ in range(self.max_sampling_attempts):
            self.stats.sampling_attempts += 1
            q_start = {side: sample_joint_uniform(self.rng) for side in ("left", "right")}
            q_goal = {side: sample_joint_uniform(self.rng) for side in ("left", "right")}
            if not self._endpoint_is_safe(q_start["left"], q_start["right"]):
                continue
            if not self._endpoint_is_safe(q_goal["left"], q_goal["right"]):
                continue
            goal = {side: self.scene.state.snapshot(side) for side in ("left", "right")}
            return q_start, q_goal, goal
        raise RuntimeError(f"Exceeded max_sampling_attempts={self.max_sampling_attempts}")

    def _append_sample(self, samples, side, q_start, q_goal, start_snapshot, goal_snapshot, current_snapshot, history, expert_action, episode_id, step_id):
        history_action, history_mask = history.padded()
        current_pose = current_snapshot["tcp_pose9_local"]
        goal_pose = goal_snapshot["tcp_pose9_local"]
        labels = self.scene.monitor.labels()
        sample = {
            "q_start": q_start.astype(np.float32),
            "q_current": _cpu(current_snapshot["q"])[0],
            "q_goal": q_goal.astype(np.float32),
            "delta_q_goal": periodic_delta(q_goal, _cpu(current_snapshot["q"])[0]).astype(np.float32),
            "x_start": _cpu(start_snapshot["tcp_pose9_local"])[0],
            "x_current": _cpu(current_pose)[0],
            "x_goal": _cpu(goal_pose)[0],
            "delta_x_goal": _cpu(_delta_pose(current_pose, goal_pose))[0],
            "other_arm_geometry": _cpu(current_snapshot["other_arm_geometry_local"])[0],
            "history_action": history_action,
            "history_mask": history_mask,
            "expert_action": expert_action.astype(np.float32),
            "self_collision_label": np.float32(labels[f"{side}_self"]),
            "inter_collision_label": np.float32(labels["inter_arm"] or labels[f"{side}_environment"]),
            "episode_id": np.int64(episode_id),
            "step_id": np.int64(step_id),
            "seed": np.int64(self.seed),
            "dt": np.float32(self.scene.control_dt),
        }
        samples[side].append(sample)

    def generate(self, accepted_episodes: int) -> dict:
        all_samples = {"left": [], "right": []}
        while self.stats.accepted_episodes < accepted_episodes:
            q_start, q_goal, goal_snapshot = self._sample_safe_pair()
            plans = {side: self.teacher.plan(q_start[side], q_goal[side]) for side in ("left", "right")}
            self.scene.teleport(torch.from_numpy(q_start["left"]), torch.from_numpy(q_start["right"]), settle_steps=3)
            start_snapshot = {side: self.scene.state.snapshot(side) for side in ("left", "right")}
            histories = {side: ActionHistory(self.history_length) for side in ("left", "right")}
            episode_samples = {"left": [], "right": []}
            collision = False
            for step_id in range(self.teacher.steps - 1):
                current = {side: self.scene.state.snapshot(side) for side in ("left", "right")}
                expert = {}
                for side in ("left", "right"):
                    q_real = _cpu(current[side]["q"])[0]
                    expert[side] = np.clip(periodic_delta(plans[side].positions[step_id + 1], q_real), -self.scene.action_delta_max, self.scene.action_delta_max)
                    self._append_sample(episode_samples, side, q_start[side], q_goal[side], start_snapshot[side], goal_snapshot[side], current[side], histories[side], expert[side], self.stats.accepted_episodes, step_id)
                result = self.scene.step_targets(torch.from_numpy(plans["left"].positions[step_id + 1]), torch.from_numpy(plans["right"].positions[step_id + 1]))
                for side in ("left", "right"):
                    histories[side].append(_cpu(result[side]["actual_delta"])[0])
                if self.scene.monitor.forbidden_contacts() or result["left"]["emergency_stop"] or result["right"]["emergency_stop"]:
                    collision = True
                    break
            if collision:
                self.stats.trajectory_collision_rejections += 1
                continue
            for side in ("left", "right"):
                all_samples[side].extend(episode_samples[side])
            self.stats.accepted_episodes += 1

        outputs = {}
        for side in ("left", "right"):
            side_dir = self.output_root / side
            side_dir.mkdir(parents=True, exist_ok=True)
            output = side_dir / f"stage1_seed{self.seed}_ep{accepted_episodes}.npz"
            keys = all_samples[side][0].keys()
            arrays = {key: np.stack([sample[key] for sample in all_samples[side]]) for key in keys}
            np.savez_compressed(output, **arrays)
            outputs[side] = str(output)
        report = {"outputs": outputs, "samples_per_side": len(all_samples["left"]), **self.stats.report()}
        report_path = self.output_root.parent / "results" / "stage1_generation.json"
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def generate_fixed_tiny(self, q_start: dict[str, np.ndarray], q_goal: dict[str, np.ndarray]) -> dict:
        """Generate one deliberately simple debugging trajectory; never mix it into formal random data."""
        if not self._endpoint_is_safe(q_start["left"], q_start["right"]):
            raise RuntimeError("Fixed tiny start is in forbidden collision")
        if not self._endpoint_is_safe(q_goal["left"], q_goal["right"]):
            raise RuntimeError("Fixed tiny goal is in forbidden collision")
        goal_snapshot = {side: self.scene.state.snapshot(side) for side in ("left", "right")}
        plans = {side: self.teacher.plan(q_start[side], q_goal[side]) for side in ("left", "right")}
        self.scene.teleport(torch.from_numpy(q_start["left"]), torch.from_numpy(q_start["right"]), settle_steps=3)
        start_snapshot = {side: self.scene.state.snapshot(side) for side in ("left", "right")}
        histories = {side: ActionHistory(self.history_length) for side in ("left", "right")}
        samples = {"left": [], "right": []}
        for step_id in range(self.teacher.steps - 1):
            current = {side: self.scene.state.snapshot(side) for side in ("left", "right")}
            for side in ("left", "right"):
                q_real = _cpu(current[side]["q"])[0]
                expert = np.clip(periodic_delta(plans[side].positions[step_id + 1], q_real), -self.scene.action_delta_max, self.scene.action_delta_max)
                self._append_sample(samples, side, q_start[side], q_goal[side], start_snapshot[side], goal_snapshot[side], current[side], histories[side], expert, 0, step_id)
            result = self.scene.step_targets(torch.from_numpy(plans["left"].positions[step_id + 1]), torch.from_numpy(plans["right"].positions[step_id + 1]))
            if self.scene.monitor.forbidden_contacts() or result["left"]["emergency_stop"] or result["right"]["emergency_stop"]:
                raise RuntimeError(f"Fixed tiny trajectory failed safety gate at step {step_id}")
            for side in ("left", "right"):
                histories[side].append(_cpu(result[side]["actual_delta"])[0])
        outputs = {}
        for side in ("left", "right"):
            side_dir = self.output_root / side
            side_dir.mkdir(parents=True, exist_ok=True)
            output = side_dir / f"tiny_fixed_seed{self.seed}.npz"
            arrays = {key: np.stack([sample[key] for sample in samples[side]]) for key in samples[side][0]}
            np.savez_compressed(output, **arrays)
            outputs[side] = str(output)
        return {"outputs": outputs, "samples_per_side": len(samples["left"]), "purpose": "tiny_overfit_debug_only", "formal_sampling_distribution": None}
