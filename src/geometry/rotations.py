"""SO(3) representations used by the supervised model."""

from __future__ import annotations

import torch


def quaternion_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def rotation_matrix_to_6d(rotation: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 6D representation: first two matrix columns."""
    return rotation[..., :, :2].transpose(-1, -2).flatten(-2)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def so3_log_map(rotation: torch.Tensor) -> torch.Tensor:
    """Batched principal SO(3) logarithm, including a stable near-pi branch."""
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    vee = torch.stack(
        (rotation[..., 2, 1] - rotation[..., 1, 2], rotation[..., 0, 2] - rotation[..., 2, 0], rotation[..., 1, 0] - rotation[..., 0, 1]),
        dim=-1,
    )
    theta = torch.atan2(0.5 * torch.linalg.vector_norm(vee, dim=-1), cosine)
    sin_theta = torch.sin(theta)
    ordinary_axis = vee / (2.0 * sin_theta[..., None].clamp_min(1e-7))
    diagonal = rotation.diagonal(dim1=-2, dim2=-1)
    components = torch.sqrt(((diagonal + 1.0) * 0.5).clamp_min(1e-8))
    x, y, z = components.unbind(-1)
    candidate_x = torch.stack((x, (rotation[..., 0, 1] + rotation[..., 1, 0]) / (4 * x.clamp_min(1e-7)), (rotation[..., 0, 2] + rotation[..., 2, 0]) / (4 * x.clamp_min(1e-7))), dim=-1)
    candidate_y = torch.stack(((rotation[..., 0, 1] + rotation[..., 1, 0]) / (4 * y.clamp_min(1e-7)), y, (rotation[..., 1, 2] + rotation[..., 2, 1]) / (4 * y.clamp_min(1e-7))), dim=-1)
    candidate_z = torch.stack(((rotation[..., 0, 2] + rotation[..., 2, 0]) / (4 * z.clamp_min(1e-7)), (rotation[..., 1, 2] + rotation[..., 2, 1]) / (4 * z.clamp_min(1e-7)), z), dim=-1)
    candidates = torch.stack((candidate_x, candidate_y, candidate_z), dim=-2)
    largest = torch.argmax(components, dim=-1)
    gather_index = largest[..., None, None].expand(*largest.shape, 1, 3)
    pi_axis = torch.gather(candidates, -2, gather_index).squeeze(-2)
    pi_axis = torch.nn.functional.normalize(pi_axis, dim=-1)
    sign = torch.where((pi_axis * vee).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    pi_axis = pi_axis * sign
    axis = torch.where((theta > torch.pi - 1e-3)[..., None], pi_axis, ordinary_axis)
    result = theta[..., None] * axis
    small = (0.5 + theta.square() / 12.0)[..., None] * vee
    return torch.where((theta < 1e-4)[..., None], small, result)


def pose9(position: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.cat((position, rotation_matrix_to_6d(rotation)), dim=-1)


def pose_error6(current_position: torch.Tensor, current_rotation: torch.Tensor, goal_position: torch.Tensor, goal_rotation: torch.Tensor) -> torch.Tensor:
    return torch.cat((goal_position - current_position, so3_log_map(current_rotation.transpose(-1, -2) @ goal_rotation)), dim=-1)


def geodesic_distance(rotation_a: torch.Tensor, rotation_b: torch.Tensor) -> torch.Tensor:
    relative = rotation_a.transpose(-1, -2) @ rotation_b
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
