"""Runtime path resolution and per-arm base-frame observations."""

from __future__ import annotations

import re

import torch

from src.geometry.base_frame import transform_points, validate_rigid_transform, world_to_base
from src.geometry.rotations import pose9, quaternion_wxyz_to_matrix


class StateReader:
    def __init__(self, articulation, left_prefix: str, right_prefix: str, left_tool: str, right_tool: str):
        self.articulation = articulation
        self.prefixes = {"left": left_prefix, "right": right_prefix}
        self.tools = {"left": left_tool, "right": right_tool}
        self.dof_paths = [str(path) for path in articulation.root_physx_view.dof_paths[0]]
        self.body_paths = [str(path) for path in articulation.root_physx_view.link_paths[0]]
        self.joint_indices = {}
        self.body_indices = {path: index for index, path in enumerate(self.body_paths)}
        self.base_paths = {side: prefix + "/a1" for side, prefix in self.prefixes.items()}
        self.other_link_paths = {
            side: [prefix + f"/a{i}" for i in range(1, 9)] + [prefix + "/b8"]
            for side, prefix in self.prefixes.items()
        }
        for side, prefix in self.prefixes.items():
            indices = [i for i, path in enumerate(self.dof_paths) if path.startswith(prefix + "/") and re.search(r"/joint[1-8]$", path)]
            if len(indices) != 8:
                raise RuntimeError(f"Expected 8 {side} DOFs, found {indices}")
            # Physics order is verified by full paths, then ordered semantically joint1..joint8.
            indices.sort(key=lambda i: int(re.search(r"joint([1-8])$", self.dof_paths[i]).group(1)))
            self.joint_indices[side] = indices
        for required in (*self.base_paths.values(), *self.tools.values(), *sum(self.other_link_paths.values(), [])):
            if required not in self.body_indices:
                raise RuntimeError(f"Required rigid body not in PhysX articulation: {required}")

    def pose_world(self, body_path: str) -> torch.Tensor:
        pose = self.articulation.data.body_link_pose_w[:, self.body_indices[body_path]]
        transform = torch.zeros(pose.shape[0], 4, 4, dtype=pose.dtype, device=pose.device)
        transform[:, :3, :3] = quaternion_wxyz_to_matrix(pose[:, 3:7])
        transform[:, :3, 3] = pose[:, :3]
        transform[:, 3, 3] = 1.0
        return transform

    def q(self, side: str) -> torch.Tensor:
        return self.articulation.data.joint_pos[:, self.joint_indices[side]]

    def qd(self, side: str) -> torch.Tensor:
        return self.articulation.data.joint_vel[:, self.joint_indices[side]]

    def tcp_local_transform(self, side: str) -> torch.Tensor:
        base = self.pose_world(self.base_paths[side])
        validate_rigid_transform(base)
        return world_to_base(base, self.pose_world(self.tools[side]))

    def tcp_pose9_local(self, side: str) -> torch.Tensor:
        transform = self.tcp_local_transform(side)
        return pose9(transform[:, :3, 3], transform[:, :3, :3])

    def other_arm_geometry_local(self, side: str) -> torch.Tensor:
        other = "right" if side == "left" else "left"
        points = torch.stack([self.pose_world(path)[:, :3, 3] for path in self.other_link_paths[other]], dim=1)
        base = self.pose_world(self.base_paths[side])
        return transform_points(torch.linalg.inv(base), points)

    def snapshot(self, side: str) -> dict[str, torch.Tensor]:
        return {
            "q": self.q(side).clone(),
            "qd": self.qd(side).clone(),
            "tcp_transform_local": self.tcp_local_transform(side).clone(),
            "tcp_pose9_local": self.tcp_pose9_local(side).clone(),
            "other_arm_geometry_local": self.other_arm_geometry_local(side).clone(),
        }

