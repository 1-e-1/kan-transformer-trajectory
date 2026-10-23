"""Isaac Lab adapter for the fixed host articulation and two independent arms."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import warnings

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import SimulationContext
from isaacsim.core.simulation_manager import SimulationManager

from src.geometry.periodic import periodic_delta

from .collision_monitor import CollisionMonitor
from .state_reader import StateReader


LEFT_PREFIX = "/World/Base_01/zbot_8s_new_01"
RIGHT_PREFIX = "/World/Base_02/zbot_8s_new_02"
LEFT_TOOL = "/World/cixi_01"
RIGHT_TOOL = "/World/cixi_00"


class DualArmScene:
    def __init__(
        self,
        usd_path: str | Path,
        device: str = "cuda:0",
        physics_dt: float = 1.0 / 120.0,
        control_decimation: int = 4,
        joint_stiffness: float = 400.0,
        joint_damping: float = 40.0,
        joint_effort_limit: float = 200.0,
        joint_velocity_limit: float = 6.0,
        execution_mode: str = "kinematic",
        pin_kinematic_articulation: bool = False,
        stage_setup: Callable[[object], None] | None = None,
        collision_mode: str = 'impulse',
    ):
        if execution_mode not in {"kinematic", "dynamic"}:
            raise ValueError(f"execution_mode must be 'kinematic' or 'dynamic', got {execution_mode!r}")
        self.usd_path = Path(usd_path).resolve()
        self.control_decimation = control_decimation
        self.execution_mode = execution_mode
        self.pin_kinematic_articulation = pin_kinematic_articulation
        self.collision_mode = collision_mode
        self.action_delta_max = torch.deg2rad(torch.tensor(5.0, device=device)).item()
        self.emergency_velocity_limit = joint_velocity_limit
        if not sim_utils.open_stage(str(self.usd_path)):
            raise RuntimeError(f"Failed to open stage {self.usd_path}")
        while sim_utils.is_stage_loading():
            sim_utils.update_stage()
        if stage_setup is not None:
            # Task-only USD session-layer setup must happen before PhysX builds its
            # rigid-body views.  The callback is opt-in, so all existing callers
            # retain byte-for-byte equivalent scene initialization behavior.
            stage_setup(sim_utils.get_current_stage())
        self.sim = SimulationContext(sim_utils.SimulationCfg(dt=physics_dt, device=device))
        self.device = self.sim.device
        world = sim_utils.get_current_stage().GetPrimAtPath('/World')
        container = world.GetAttribute('robot:articulationContainer').Get() or '/World/unitree_h1'
        root = world.GetAttribute('robot:articulationRoot').Get() or '/pelvis'
        if world.GetAttribute('robot:collisionModelRequiresRetraining').Get():
            warnings.warn('Scene collision geometry changed: old learned collision fields, collision labels and safe-task manifests are not validated for this scene.', RuntimeWarning)
            if str(self.device) == 'cpu':
                warnings.warn('Astronaut scene contact-monitor acceptance is validated on cuda:0 only; CPU contact tensors did not report test collisions. Do not use CPU results as collision-safety evidence.', RuntimeWarning)
        cfg = ArticulationCfg(
            prim_path=container,
            articulation_root_prim_path=root,
            spawn=None,
            actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
        )
        self.articulation = Articulation(cfg)
        self.monitor = CollisionMonitor(sim_utils.get_current_stage(), LEFT_PREFIX, RIGHT_PREFIX, LEFT_TOOL, RIGHT_TOOL, collision_mode=collision_mode)
        self.monitor.install()
        self.sim.reset()
        self.articulation.update(self.sim.get_physics_dt())
        self.state = StateReader(self.articulation, LEFT_PREFIX, RIGHT_PREFIX, LEFT_TOOL, RIGHT_TOOL)
        self._pinned_root_pose = self.articulation.data.root_link_pose_w.clone()
        self._pinned_joint_positions = self.articulation.data.joint_pos.clone()
        arm_body_paths = [
            path for path in self.state.body_paths if path.startswith(LEFT_PREFIX + "/") or path.startswith(RIGHT_PREFIX + "/")
        ]
        fixed_body_paths = [
            path for path in self.state.body_paths
            if path not in arm_body_paths and path not in {LEFT_TOOL, RIGHT_TOOL}
        ]
        self.monitor.initialize_tensor_views(SimulationManager.get_physics_sim_view(), arm_body_paths, fixed_body_paths)
        arm_indices = self.state.joint_indices["left"] + self.state.joint_indices["right"]
        if self.execution_mode == "dynamic":
            self.articulation.write_joint_stiffness_to_sim(joint_stiffness, joint_ids=arm_indices)
            self.articulation.write_joint_damping_to_sim(joint_damping, joint_ids=arm_indices)
            self.articulation.write_joint_effort_limit_to_sim(joint_effort_limit, joint_ids=arm_indices)
            self.articulation.write_joint_velocity_limit_to_sim(joint_velocity_limit, joint_ids=arm_indices)
        self._last_q = {side: self.state.q(side).clone() for side in ("left", "right")}

    @property
    def physics_dt(self) -> float:
        return self.sim.get_physics_dt()

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.control_decimation

    @staticmethod
    def _wrap_to_pi(q: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(q), torch.cos(q))

    def _write_arm_states(self, q_by_side: dict[str, torch.Tensor]) -> None:
        """Write an exact joint configuration; used as the kinematic state transition."""
        if self.pin_kinematic_articulation:
            all_q = self._pinned_joint_positions.clone()
            for side, q in q_by_side.items():
                all_q[:, self.state.joint_indices[side]] = q.to(self.device).reshape(1, 8)
            self.articulation.write_root_pose_to_sim(self._pinned_root_pose)
            self.articulation.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device))
            self.articulation.write_joint_state_to_sim(all_q, torch.zeros_like(all_q))
            self.articulation.set_joint_position_target(all_q)
            self.articulation.write_data_to_sim()
            return
        for side, q in q_by_side.items():
            q = q.to(self.device).reshape(1, 8)
            zero = torch.zeros_like(q)
            indices = self.state.joint_indices[side]
            self.articulation.write_joint_state_to_sim(q, zero, joint_ids=indices)
            self.articulation.set_joint_position_target(q, joint_ids=indices)
        self.articulation.write_data_to_sim()

    def _poll_configuration(self, q_by_side: dict[str, torch.Tensor], steps: int) -> None:
        """Hold q fixed while PhysX supplies FK and contact queries, not dynamics."""
        for _ in range(max(1, steps)):
            self._write_arm_states(q_by_side)
            self.monitor.begin_step()
            self.sim.step(render=False)
            self.articulation.update(self.physics_dt)
            self.monitor.poll()
        # Contact resolution must not become part of the kinematic transition.
        self._write_arm_states(q_by_side)

    def teleport(self, q_left: torch.Tensor, q_right: torch.Tensor, settle_steps: int = 2) -> None:
        if self.execution_mode == "kinematic":
            q_by_side = {
                "left": self._wrap_to_pi(q_left.to(self.device).reshape(1, 8)),
                "right": self._wrap_to_pi(q_right.to(self.device).reshape(1, 8)),
            }
            self._poll_configuration(q_by_side, settle_steps)
            self._last_q = {side: q.clone() for side, q in q_by_side.items()}
            return
        for side, q in (("left", q_left), ("right", q_right)):
            q = q.to(self.device).reshape(1, 8)
            zero = torch.zeros_like(q)
            indices = self.state.joint_indices[side]
            self.articulation.write_joint_state_to_sim(q, zero, joint_ids=indices)
            self.articulation.set_joint_position_target(q, joint_ids=indices)
        self.articulation.write_data_to_sim()
        for _ in range(settle_steps):
            self.monitor.begin_step()
            self.sim.step(render=False)
            self.articulation.update(self.physics_dt)
            self.monitor.poll()
        self._last_q = {side: self.state.q(side).clone() for side in ("left", "right")}

    def step_targets(self, q_left_target: torch.Tensor, q_right_target: torch.Tensor) -> dict:
        if self.execution_mode == "kinematic":
            previous = {side: self.state.q(side).clone() for side in ("left", "right")}
            targets = {
                "left": self._wrap_to_pi(q_left_target.to(self.device).reshape(1, 8)),
                "right": self._wrap_to_pi(q_right_target.to(self.device).reshape(1, 8)),
            }
            deltas = {side: periodic_delta(targets[side], previous[side]) for side in ("left", "right")}
            self._poll_configuration(targets, steps=1)
            result = {"contacts": self.monitor.serializable_contacts(), "labels": self.monitor.labels()}
            for side in ("left", "right"):
                qd = deltas[side] / self.control_dt
                result[side] = {
                    "q": targets[side].clone(),
                    "qd": qd,
                    "actual_delta": deltas[side].clone(),
                    "emergency_stop": bool(torch.any(torch.abs(qd) > self.emergency_velocity_limit).item()),
                }
            self._last_q = {side: target.clone() for side, target in targets.items()}
            result["safety_stop"] = bool(
                self.monitor.forbidden_contacts()
                or result["left"]["emergency_stop"]
                or result["right"]["emergency_stop"]
            )
            return result
        for side, target in (("left", q_left_target), ("right", q_right_target)):
            self.articulation.set_joint_position_target(target.to(self.device).reshape(1, 8), joint_ids=self.state.joint_indices[side])
        for _ in range(self.control_decimation):
            self.monitor.begin_step()
            self.articulation.write_data_to_sim()
            self.sim.step(render=False)
            self.articulation.update(self.physics_dt)
            self.monitor.poll()
        result = {"contacts": self.monitor.serializable_contacts(), "labels": self.monitor.labels()}
        for side in ("left", "right"):
            q = self.state.q(side).clone()
            qd = self.state.qd(side).clone()
            result[side] = {
                "q": q,
                "qd": qd,
                "actual_delta": q - self._last_q[side],
                "emergency_stop": bool(torch.any(torch.abs(qd) > self.emergency_velocity_limit).item()),
            }
            self._last_q[side] = q
        result["safety_stop"] = bool(
            self.monitor.forbidden_contacts()
            or result["left"]["emergency_stop"]
            or result["right"]["emergency_stop"]
        )
        return result

    def step_deltas(self, delta_left: torch.Tensor, delta_right: torch.Tensor) -> dict:
        delta_left = delta_left.to(self.device).clamp(-self.action_delta_max, self.action_delta_max)
        delta_right = delta_right.to(self.device).clamp(-self.action_delta_max, self.action_delta_max)
        return self.step_targets(
            self._wrap_to_pi(self.state.q("left") + delta_left.reshape(1, 8)),
            self._wrap_to_pi(self.state.q("right") + delta_right.reshape(1, 8)),
        )

    def step_left_delta_with_right_target(
        self, delta_left: torch.Tensor, q_right_target: torch.Tensor
    ) -> dict:
        """Apply a bounded left-arm delta while writing an exact right-arm hold target."""
        delta_left = delta_left.to(self.device).clamp(-self.action_delta_max, self.action_delta_max)
        return self.step_targets(
            self._wrap_to_pi(self.state.q("left") + delta_left.reshape(1, 8)),
            self._wrap_to_pi(q_right_target.to(self.device).reshape(1, 8)),
        )
