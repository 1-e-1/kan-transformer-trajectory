"""Rigid-frame conversion utilities using full homogeneous transforms.

All transforms follow the column-vector convention: ``p_parent = T_parent_child @ p_child``.
The functions accept NumPy arrays or torch tensors and preserve batch dimensions.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import torch

Array = TypeVar("Array", np.ndarray, torch.Tensor)


def _transpose_last2(x: Array) -> Array:
    return x.transpose(-1, -2) if isinstance(x, torch.Tensor) else np.swapaxes(x, -1, -2)


def _zeros_like_matrix(t: Array) -> Array:
    shape = (*t.shape[:-2], 4, 4)
    if isinstance(t, torch.Tensor):
        return torch.zeros(shape, dtype=t.dtype, device=t.device)
    return np.zeros(shape, dtype=t.dtype)


def invert_transform(t_parent_child: Array) -> Array:
    """Return ``T_child_parent`` for a proper rigid transform ``T_parent_child``."""
    if t_parent_child.shape[-2:] != (4, 4):
        raise ValueError(f"Expected [...,4,4], got {t_parent_child.shape}")
    rotation = t_parent_child[..., :3, :3]
    translation = t_parent_child[..., :3, 3]
    rotation_t = _transpose_last2(rotation)
    out = _zeros_like_matrix(t_parent_child)
    out[..., :3, :3] = rotation_t
    out[..., :3, 3] = -(rotation_t @ translation[..., None])[..., 0]
    out[..., 3, 3] = 1
    return out


def compose_transform(t_a_b: Array, t_b_c: Array) -> Array:
    """Return ``T_a_c = T_a_b @ T_b_c``."""
    return t_a_b @ t_b_c


def world_to_base(t_world_base: Array, t_world_object: Array) -> Array:
    """Express an object pose in a robot base frame with ``inv(T_W_B) @ T_W_X``."""
    return compose_transform(invert_transform(t_world_base), t_world_object)


def transform_points(t_target_source: Array, points_source: Array) -> Array:
    """Transform ``[...,N,3]`` points, including rotation and translation."""
    rotation = t_target_source[..., :3, :3]
    translation = t_target_source[..., :3, 3]
    return points_source @ _transpose_last2(rotation) + translation[..., None, :]


def validate_rigid_transform(transform: Array, atol: float = 1e-5) -> None:
    """Reject scaled/reflected transforms that cannot define an SO(3) base frame."""
    rotation = transform[..., :3, :3]
    if isinstance(transform, torch.Tensor):
        eye = torch.eye(3, dtype=transform.dtype, device=transform.device).expand(rotation.shape)
        if not torch.allclose(_transpose_last2(rotation) @ rotation, eye, atol=atol, rtol=0):
            raise ValueError("Base transform rotation is not orthonormal")
        if not torch.allclose(torch.linalg.det(rotation), torch.ones_like(torch.linalg.det(rotation)), atol=atol, rtol=0):
            raise ValueError("Base transform must be proper (determinant +1), not a reflection")
    else:
        eye = np.broadcast_to(np.eye(3, dtype=rotation.dtype), rotation.shape)
        if not np.allclose(_transpose_last2(rotation) @ rotation, eye, atol=atol, rtol=0):
            raise ValueError("Base transform rotation is not orthonormal")
        if not np.allclose(np.linalg.det(rotation), 1.0, atol=atol, rtol=0):
            raise ValueError("Base transform must be proper (determinant +1), not a reflection")

