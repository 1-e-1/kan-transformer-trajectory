"""Left-arm joint-space collision-boundary data from settled GPU PhysX queries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.dagger_buffer import ActionHistory
from src.evaluation.rollout import build_observation
from src.geometry.periodic import periodic_delta


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()[0].astype(np.float32)


class LeftCollisionBoundaryGenerator:
    """Build equal self/external positive sets with paired safe inward actions.

    The right arm stays at zero.  ``external`` means either left/right-arm contact
    or contact between the left arm and any fixed robot/scene body.  This keeps
    the checkpoint's historical second-logit name compatible while broadening
    its safety semantics.
    """

    def __init__(self, scene, output_root: str | Path, seed: int = 18350, history_length: int = 64):
        self.scene = scene
        self.output_root = Path(output_root)
        self.seed = seed
        self.history_length = history_length
        self.rng = np.random.default_rng(seed)
        self.zero = np.zeros(8, dtype=np.float32)

    @staticmethod
    def _left_labels(labels: dict[str, bool]) -> tuple[bool, bool]:
        self_collision = bool(labels["left_self"])
        external_collision = bool(labels["inter_arm"] or labels["left_environment"])
        return self_collision, external_collision

    def _labels(self, q_left: np.ndarray) -> tuple[bool, bool]:
        self.scene.teleport(
            torch.from_numpy(q_left), torch.from_numpy(self.zero), settle_steps=2
        )
        return self._left_labels(self.scene.monitor.labels())

    def _make_sample(
        self,
        observation: dict[str, torch.Tensor],
        query_action: np.ndarray,
        self_label: bool,
        external_label: bool,
        episode_id: int,
        step_id: int,
    ) -> dict[str, np.ndarray | np.generic]:
        sample = {key: _numpy(value) for key, value in observation.items()}
        sample.update(
            {
                # These counterfactual queries only supervise the auxiliary head.
                "expert_action": np.zeros(8, dtype=np.float32),
                "collision_query_action": query_action.astype(np.float32),
                "self_collision_label": np.float32(self_label),
                "inter_collision_label": np.float32(external_label),
                "action_loss_mask": np.float32(0.0),
                "episode_id": np.int64(episode_id),
                "step_id": np.int64(step_id),
                "seed": np.int64(self.seed),
                "dt": np.float32(self.scene.control_dt),
            }
        )
        return sample

    def _boundary_pair(self, target: int, max_random_attempts: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (safe_q, colliding_q) for one pure target boundary."""
        for _ in range(max_random_attempts):
            unsafe = self.rng.uniform(-np.pi, np.pi, 8).astype(np.float32)
            unsafe_labels = self._labels(unsafe)
            if not unsafe_labels[target] or unsafe_labels[1 - target]:
                continue

            low, high = 0.0, 1.0
            for _ in range(15):
                alpha = 0.5 * (low + high)
                labels = self._labels((alpha * unsafe).astype(np.float32))
                if any(labels):
                    high = alpha
                else:
                    low = alpha

            # Move a small distance to either side so the label is robust to
            # contact tolerances, while staying within a single 5-degree action.
            max_joint = float(np.max(np.abs(unsafe)))
            alpha_margin = min(0.01, self.scene.action_delta_max * 0.20 / max(max_joint, 1e-6))
            safe = (max(0.0, low - alpha_margin) * unsafe).astype(np.float32)
            colliding = (min(1.0, high + alpha_margin) * unsafe).astype(np.float32)
            safe_labels = self._labels(safe)
            collision_labels = self._labels(colliding)
            query = np.asarray(periodic_delta(colliding, safe), dtype=np.float32)
            if any(safe_labels):
                continue
            if not collision_labels[target] or collision_labels[1 - target]:
                continue
            if np.max(np.abs(query)) > self.scene.action_delta_max + 1e-6:
                continue
            return safe, colliding
        return None

    def generate(self, positives_per_kind: int = 96, max_random_attempts: int = 250) -> dict:
        if any(self._labels(self.zero)):
            raise RuntimeError("The all-zero left/right reference must be collision-free")
        goal_snapshot = self.scene.state.snapshot("left")
        goal_pose = _numpy(goal_snapshot["tcp_pose9_local"])
        samples: list[dict] = []
        counts = {"left_self": 0, "left_external": 0}
        episode_id = 0

        for target, name in enumerate(counts):
            while counts[name] < positives_per_kind:
                pair = self._boundary_pair(target, max_random_attempts)
                if pair is None:
                    raise RuntimeError(
                        f"Could not find pure {name} boundary after "
                        f"{max_random_attempts} random trials"
                    )
                safe, colliding = pair
                positive_query = np.asarray(periodic_delta(colliding, safe), dtype=np.float32)
                positive_labels = self._labels(colliding)

                # The paired negative query moves inward toward the known-safe
                # zero configuration with the same per-step action bound.
                inward = np.clip(
                    periodic_delta(self.zero, safe),
                    -self.scene.action_delta_max,
                    self.scene.action_delta_max,
                ).astype(np.float32)
                negative_q = (safe + inward).astype(np.float32)
                negative_labels = self._labels(negative_q)
                if any(negative_labels):
                    continue

                self.scene.teleport(
                    torch.from_numpy(safe), torch.from_numpy(self.zero), settle_steps=2
                )
                current = self.scene.state.snapshot("left")
                current_pose = _numpy(current["tcp_pose9_local"])
                observation = build_observation(
                    self.scene,
                    "left",
                    safe,
                    self.zero,
                    current_pose,
                    goal_pose,
                    ActionHistory(self.history_length),
                )
                samples.append(
                    self._make_sample(
                        observation,
                        positive_query,
                        positive_labels[0],
                        positive_labels[1],
                        episode_id,
                        0,
                    )
                )
                samples.append(
                    self._make_sample(
                        observation,
                        inward,
                        negative_labels[0],
                        negative_labels[1],
                        episode_id,
                        1,
                    )
                )
                counts[name] += 1
                episode_id += 1

        self.output_root.mkdir(parents=True, exist_ok=True)
        output = self.output_root / f"left_collision_boundaries_seed{self.seed}.npz"
        np.savez_compressed(
            output,
            **{key: np.stack([sample[key] for sample in samples]) for key in samples[0]},
        )
        return {
            "output": str(output),
            "seed": self.seed,
            "right_arm_configuration": "all_zero_fixed",
            "positive_boundaries": counts,
            "paired_safe_negatives": sum(counts.values()),
            "samples": len(samples),
            "action_loss_mask": 0.0,
            "external_label_semantics": "inter_arm_or_left_arm_vs_fixed_robot_scene_body",
            "label_source": "settled_gpu_physx_kinematic_configuration_query",
        }
