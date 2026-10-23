"""Balanced action-conditioned collision probes from PhysX boundary states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data.dagger_buffer import ActionHistory
from src.evaluation.rollout import build_observation
from src.geometry.periodic import periodic_delta


def _cpu(value):
    return value.detach().cpu().numpy()[0].astype(np.float32)


class CollisionCounterfactualGenerator:
    def __init__(self, scene, collision_examples_path, output_root, seed=342):
        self.scene = scene
        self.examples = json.loads(Path(collision_examples_path).read_text(encoding="utf-8"))["found"]
        self.output_root = Path(output_root)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _teleport_labels(self, q):
        self.scene.teleport(torch.from_numpy(q["left"]), torch.from_numpy(q["right"]), settle_steps=2)
        return self.scene.monitor.labels()

    def _sample(self, observation, side, expert_action, query_action, labels, episode_id, step_id):
        sample = {key: _cpu(value) for key, value in observation.items()}
        sample.update({
            "expert_action": expert_action.astype(np.float32),
            "collision_query_action": query_action.astype(np.float32),
            "self_collision_label": np.float32(labels[f"{side}_self"]),
            "inter_collision_label": np.float32(labels["inter_arm"] or labels[f"{side}_environment"]),
            "action_loss_mask": np.float32(0.0),
            "episode_id": np.int64(episode_id), "step_id": np.int64(step_id),
            "seed": np.int64(self.seed), "dt": np.float32(self.scene.control_dt),
        })
        return sample

    def generate(self, positives_per_kind=8, max_attempts_per_kind=160):
        zero = {side: np.zeros(8, dtype=np.float32) for side in ("left", "right")}
        if any(self._teleport_labels(zero).values()):
            raise RuntimeError("Zero reference configuration must be collision-free")
        goal_snapshot = {side: self.scene.state.snapshot(side) for side in ("left", "right")}
        samples = {"left": [], "right": []}
        counts = {kind: 0 for kind in ("left_self", "right_self", "inter_arm")}
        episode_id = 0
        for kind in counts:
            example = self.examples[kind]
            base_unsafe = {side: np.asarray(example[f"q_{side}"], dtype=np.float32) for side in ("left", "right")}
            for attempt in range(max_attempts_per_kind):
                if counts[kind] >= positives_per_kind:
                    break
                jitter_scale = 0.0 if attempt == 0 else 0.08
                unsafe = {side: ((base_unsafe[side] + self.rng.normal(0, jitter_scale, 8) + np.pi) % (2 * np.pi) - np.pi).astype(np.float32) for side in ("left", "right")}
                high_labels = self._teleport_labels(unsafe)
                if not high_labels[kind]:
                    continue
                low_alpha, high_alpha = 0.0, 1.0
                for _ in range(16):
                    alpha = 0.5 * (low_alpha + high_alpha)
                    middle = {side: (alpha * unsafe[side]).astype(np.float32) for side in ("left", "right")}
                    if any(self._teleport_labels(middle).values()):
                        high_alpha = alpha
                    else:
                        low_alpha = alpha
                safe = {side: (low_alpha * unsafe[side]).astype(np.float32) for side in ("left", "right")}
                colliding = {side: (high_alpha * unsafe[side]).astype(np.float32) for side in ("left", "right")}
                if any(self._teleport_labels(safe).values()):
                    continue
                current_snapshot = {side: self.scene.state.snapshot(side) for side in ("left", "right")}
                query = {side: np.asarray(periodic_delta(colliding[side], safe[side]), dtype=np.float32) for side in ("left", "right")}
                positive_labels = self._teleport_labels(colliding)
                if not positive_labels[kind]:
                    continue
                self.scene.teleport(torch.from_numpy(safe["left"]), torch.from_numpy(safe["right"]), settle_steps=2)
                expert = {side: np.clip(periodic_delta(zero[side], safe[side]), -self.scene.action_delta_max, self.scene.action_delta_max).astype(np.float32) for side in ("left", "right")}
                negative_state = {side: safe[side] + expert[side] for side in ("left", "right")}
                negative_labels = self._teleport_labels(negative_state)
                if any(negative_labels.values()):
                    continue
                self.scene.teleport(torch.from_numpy(safe["left"]), torch.from_numpy(safe["right"]), settle_steps=2)
                histories = {side: ActionHistory(64) for side in ("left", "right")}
                for side in ("left", "right"):
                    current_pose = _cpu(current_snapshot[side]["tcp_pose9_local"])
                    observation = build_observation(self.scene, side, safe[side], zero[side], current_pose, _cpu(goal_snapshot[side]["tcp_pose9_local"]), histories[side])
                    samples[side].append(self._sample(observation, side, expert[side], query[side], positive_labels, episode_id, 0))
                    samples[side].append(self._sample(observation, side, expert[side], expert[side], negative_labels, episode_id, 1))
                counts[kind] += 1
                episode_id += 1
            if counts[kind] < positives_per_kind:
                raise RuntimeError(f"Only generated {counts[kind]}/{positives_per_kind} {kind} probes")

        outputs = {}
        for side in ("left", "right"):
            directory = self.output_root / side
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"collision_counterfactual_seed{self.seed}.npz"
            np.savez_compressed(path, **{key: np.stack([sample[key] for sample in samples[side]]) for key in samples[side][0]})
            outputs[side] = str(path)
        return {"outputs": outputs, "positive_boundaries": counts, "samples_per_side": len(samples["left"]), "balanced_positive_negative_pairs": True, "label_source": "settled_physx_counterfactual_transition"}
