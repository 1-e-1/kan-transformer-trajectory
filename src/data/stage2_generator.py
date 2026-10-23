"""PhysX-validated Stage-2 collision-aware expert data generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.geometry.periodic import periodic_delta
from src.planning.collision_aware_teacher import PhysXDualArmTeacher
from src.planning.quintic_teacher import QuinticTeacher, sample_joint_uniform
from .dagger_buffer import ActionHistory
from .generator import Stage1DataGenerator, _cpu


class Stage2DataGenerator(Stage1DataGenerator):
    """Keep endpoint-safe tasks whose direct quintic path collides, then run RRT."""

    def __init__(self, scene, output_root: str | Path, history_length=64, trajectory_steps=80, seed=142, max_sampling_attempts=500, planner_iterations=1600, max_planner_failures=5, require_direct_collision=True, safe_endpoint_pool=None, pool_probability=0.8, pool_noise_std=0.35):
        super().__init__(scene, output_root, history_length, trajectory_steps, seed, max_sampling_attempts)
        self.quintic = QuinticTeacher(trajectory_steps, scene.control_dt, scene.action_delta_max)
        self.collision_teacher = PhysXDualArmTeacher(scene, seed=seed, max_iterations=planner_iterations)
        self.direct_collision_candidates = 0
        self.direct_path_rejections = 0
        self.planner_failures = 0
        self.planner_checks = 0
        self.max_planner_failures = max_planner_failures
        self.require_direct_collision = require_direct_collision
        self.safe_endpoint_pool = safe_endpoint_pool
        self.pool_probability = pool_probability
        self.pool_noise_std = pool_noise_std

    def _sample_configuration_pair(self):
        if self.safe_endpoint_pool is not None and self.rng.random() < self.pool_probability:
            count = len(self.safe_endpoint_pool["left"])
            indices = self.rng.integers(0, count, size=2)
            result = []
            for index in indices:
                pair = {}
                for side in ("left", "right"):
                    q = self.safe_endpoint_pool[side][index] + self.rng.normal(0.0, self.pool_noise_std, 8)
                    pair[side] = ((q + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)
                result.append(pair)
            return result[0], result[1]
        return ({side: sample_joint_uniform(self.rng) for side in ("left", "right")}, {side: sample_joint_uniform(self.rng) for side in ("left", "right")})

    def _quick_endpoint_is_safe(self, q_left, q_right) -> bool:
        self.scene.teleport(torch.from_numpy(q_left), torch.from_numpy(q_right), settle_steps=1)
        labels = self.scene.monitor.labels()
        self_collision = labels["left_self"] or labels["right_self"]
        self.stats.self_collision_rejections += int(self_collision)
        external_collision = labels["inter_arm"] or labels["left_environment"] or labels["right_environment"]
        self.stats.inter_arm_collision_rejections += int(external_collision)
        return not self_collision and not external_collision

    def _direct_collides(self, q_start, q_goal) -> bool:
        plans = {side: self.quintic.plan(q_start[side], q_goal[side]) for side in ("left", "right")}
        for step in range(1, self.quintic.steps):
            self.scene.teleport(
                torch.as_tensor(plans["left"].positions[step], device=self.scene.device),
                torch.as_tensor(plans["right"].positions[step], device=self.scene.device),
                settle_steps=1,
            )
            if self.scene.monitor.forbidden_contacts():
                return True
        return False

    def _sample_difficult_pair(self):
        for _ in range(self.max_sampling_attempts):
            self.stats.sampling_attempts += 1
            if self.stats.sampling_attempts % 10 == 0:
                print(f"stage2_sampling_attempt={self.stats.sampling_attempts}", flush=True)
            q_start, q_goal = self._sample_configuration_pair()
            if not self._quick_endpoint_is_safe(q_start["left"], q_start["right"]):
                continue
            if not self._quick_endpoint_is_safe(q_goal["left"], q_goal["right"]):
                continue
            goal = {side: self.scene.state.snapshot(side) for side in ("left", "right")}
            direct_collides = self._direct_collides(q_start, q_goal) if self.require_direct_collision else False
            if direct_collides or not self.require_direct_collision:
                # Fast one-step checks are only a sampler prefilter.  Re-audit both
                # endpoints with settled PhysX before handing them to the planner.
                if not self._endpoint_is_safe(q_start["left"], q_start["right"]):
                    continue
                if not self._endpoint_is_safe(q_goal["left"], q_goal["right"]):
                    continue
                goal = {side: self.scene.state.snapshot(side) for side in ("left", "right")}
                self.direct_collision_candidates += int(direct_collides)
                return q_start, q_goal, goal
        raise RuntimeError(f"No endpoint-safe/direct-colliding pair in {self.max_sampling_attempts} attempts")

    def generate(self, accepted_episodes: int) -> dict:
        all_samples = {"left": [], "right": []}
        planner_records = []
        while self.stats.accepted_episodes < accepted_episodes:
            q_start, q_goal, goal_snapshot = self._sample_difficult_pair()
            try:
                dual_plan = self.collision_teacher.plan(q_start, q_goal)
            except (RuntimeError, ValueError):
                self.planner_failures += 1
                if self.planner_failures >= self.max_planner_failures:
                    raise RuntimeError(f"Exceeded max_planner_failures={self.max_planner_failures} for seed={self.seed}")
                continue
            if self.require_direct_collision and dual_plan.used_direct_path:
                # The hard-task sampler and RRT use different interpolation grids.
                # A narrow contact can therefore be observed by the first and missed
                # by the second.  Hard-only output has a strict semantic contract:
                # never accept a path the final planner classified as direct.
                self.direct_path_rejections += 1
                continue
            self.planner_checks += dual_plan.collision_checks
            plans = {"left": dual_plan.left, "right": dual_plan.right}
            self.scene.teleport(torch.from_numpy(q_start["left"]), torch.from_numpy(q_start["right"]), settle_steps=3)
            start_snapshot = {side: self.scene.state.snapshot(side) for side in ("left", "right")}
            histories = {side: ActionHistory(self.history_length) for side in ("left", "right")}
            episode_samples = {"left": [], "right": []}
            failed = False
            for step_id in range(len(plans["left"].positions) - 1):
                current = {side: self.scene.state.snapshot(side) for side in ("left", "right")}
                for side in ("left", "right"):
                    q_real = _cpu(current[side]["q"])[0]
                    expert = np.clip(periodic_delta(plans[side].positions[step_id + 1], q_real), -self.scene.action_delta_max, self.scene.action_delta_max)
                    self._append_sample(episode_samples, side, q_start[side], q_goal[side], start_snapshot[side], goal_snapshot[side], current[side], histories[side], expert, self.stats.accepted_episodes, step_id)
                result = self.scene.step_targets(torch.from_numpy(plans["left"].positions[step_id + 1]), torch.from_numpy(plans["right"].positions[step_id + 1]))
                if self.scene.monitor.forbidden_contacts() or result["left"]["emergency_stop"] or result["right"]["emergency_stop"]:
                    failed = True
                    break
                for side in ("left", "right"):
                    histories[side].append(_cpu(result[side]["actual_delta"])[0])
            if failed:
                self.stats.trajectory_collision_rejections += 1
                continue
            for side in ("left", "right"):
                for sample in episode_samples[side]:
                    sample["planner_used_direct_path"] = np.bool_(dual_plan.used_direct_path)
                    sample["planner_iterations"] = np.int64(dual_plan.iterations)
                    sample["simulation_mode"] = np.str_(self.scene.execution_mode)
                all_samples[side].extend(episode_samples[side])
            planner_record = {"iterations": dual_plan.iterations, "collision_checks": dual_plan.collision_checks, "steps": len(dual_plan.left.positions), "used_direct_path": dual_plan.used_direct_path}
            planner_records.append(planner_record)
            self.stats.accepted_episodes += 1

        outputs = {}
        for side in ("left", "right"):
            side_dir = self.output_root / side
            side_dir.mkdir(parents=True, exist_ok=True)
            output = side_dir / f"stage2_rrt_seed{self.seed}_ep{accepted_episodes}.npz"
            arrays = {key: np.stack([sample[key] for sample in all_samples[side]]) for key in all_samples[side][0]}
            np.savez_compressed(output, **arrays)
            outputs[side] = str(output)
        report = {
            "outputs": outputs, "samples_per_side": len(all_samples["left"]),
            "direct_collision_candidates": self.direct_collision_candidates,
            "direct_path_rejections": self.direct_path_rejections,
            "planner_failures": self.planner_failures, "planner_collision_checks": self.planner_checks,
            "planner_records": planner_records, **self.stats.report(),
            "simulation_mode": self.scene.execution_mode,
        }
        report_path = self.output_root.parent / "results" / "stage2_generation.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
