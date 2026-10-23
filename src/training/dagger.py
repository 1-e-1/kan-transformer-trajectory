"""Closed-loop DAgger collection with a PhysX/RRT corrective expert."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.dagger_buffer import ActionHistory
from src.evaluation.rollout import build_observation
from src.geometry.periodic import periodic_delta
from src.planning.collision_aware_teacher import PhysXDualArmTeacher


def _numpy(value):
    return value.detach().cpu().numpy()[0].astype(np.float32)


class DAggerCollector:
    """Query the coupled expert at states actually visited by independent policies."""

    def __init__(self, scene, models, output_root: str | Path, seed=243, planner_iterations=600):
        self.scene = scene
        self.models = models
        self.output_root = Path(output_root)
        self.seed = seed
        self.teacher = PhysXDualArmTeacher(scene, seed=seed, max_iterations=planner_iterations)

    @torch.no_grad()
    def collect(self, task: dict, max_steps=8, iteration=0):
        device = self.scene.device
        self.scene.teleport(torch.as_tensor(task["left"]["q_start"], device=device), torch.as_tensor(task["right"]["q_start"], device=device), settle_steps=4)
        histories = {side: ActionHistory(self.models[side].config.history_length) for side in ("left", "right")}
        samples = {"left": [], "right": []}
        planner_failures = 0
        policy_collision_steps = 0
        for step_id in range(max_steps):
            observations, policy_actions = {}, {}
            current_q = {side: _numpy(self.scene.state.q(side)) for side in ("left", "right")}
            for side in ("left", "right"):
                spec = task[side]
                observations[side] = build_observation(self.scene, side, spec["q_start"], spec["q_goal"], spec["x_start"], spec["x_goal"], histories[side])
                policy_actions[side] = self.models[side](observations[side])["action"][0].clamp(-self.scene.action_delta_max, self.scene.action_delta_max)

            # Counterfactual label: would the policy's proposed simultaneous next
            # state collide? Restore before asking the corrective expert.
            preview_q = {side: current_q[side] + policy_actions[side].cpu().numpy() for side in ("left", "right")}
            self.scene.teleport(torch.from_numpy(preview_q["left"]), torch.from_numpy(preview_q["right"]), settle_steps=2)
            preview_labels = self.scene.monitor.labels()
            policy_collision_steps += int(any(preview_labels.values()))
            self.scene.teleport(torch.from_numpy(current_q["left"]), torch.from_numpy(current_q["right"]), settle_steps=2)

            try:
                expert = self.teacher.plan(current_q, {side: np.asarray(task[side]["q_goal"], dtype=np.float32) for side in ("left", "right")})
            except (RuntimeError, ValueError):
                planner_failures += 1
                break
            self.scene.teleport(torch.from_numpy(current_q["left"]), torch.from_numpy(current_q["right"]), settle_steps=2)
            for side, plan in (("left", expert.left), ("right", expert.right)):
                expert_action = np.clip(periodic_delta(plan.positions[1], current_q[side]), -self.scene.action_delta_max, self.scene.action_delta_max).astype(np.float32)
                obs = observations[side]
                sample = {key: _numpy(value) for key, value in obs.items()}
                sample.update({
                    "expert_action": expert_action,
                    "collision_query_action": policy_actions[side].cpu().numpy().astype(np.float32),
                    "self_collision_label": np.float32(preview_labels[f"{side}_self"]),
                    "inter_collision_label": np.float32(preview_labels["inter_arm"] or preview_labels[f"{side}_environment"]),
                    "episode_id": np.int64(iteration), "step_id": np.int64(step_id),
                    "seed": np.int64(self.seed), "dt": np.float32(self.scene.control_dt),
                })
                samples[side].append(sample)

            result = self.scene.step_deltas(policy_actions["left"], policy_actions["right"])
            # Executed-transition labels are authoritative; dynamics can create a
            # contact that a teleported counterfactual misses.
            for side in ("left", "right"):
                samples[side][-1]["self_collision_label"] = np.float32(result["labels"][f"{side}_self"])
                samples[side][-1]["inter_collision_label"] = np.float32(
                    result["labels"]["inter_arm"] or result["labels"][f"{side}_environment"]
                )
            if any(result["labels"].values()) and not any(preview_labels.values()):
                policy_collision_steps += 1
            for side in ("left", "right"):
                histories[side].append(_numpy(result[side]["actual_delta"]))
            if self.scene.monitor.forbidden_contacts() or result["left"]["emergency_stop"] or result["right"]["emergency_stop"]:
                break

        if not samples["left"]:
            raise RuntimeError("DAgger collected no expert-labeled policy states")
        outputs = {}
        for side in ("left", "right"):
            side_dir = self.output_root / side
            side_dir.mkdir(parents=True, exist_ok=True)
            path = side_dir / f"dagger_iter{iteration}_seed{self.seed}.npz"
            arrays = {key: np.stack([sample[key] for sample in samples[side]]) for key in samples[side][0]}
            np.savez_compressed(path, **arrays)
            outputs[side] = str(path)
        return {"outputs": outputs, "samples_per_side": len(samples["left"]), "planner_failures": planner_failures, "policy_collision_steps": policy_collision_steps}
