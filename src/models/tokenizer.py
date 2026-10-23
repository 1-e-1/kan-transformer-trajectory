"""Semantic token construction for one independent arm policy."""

from __future__ import annotations

import math
from enum import IntEnum

import torch
from torch import nn

from src.geometry.periodic import sincos_encode
from .kan import JointErrorKAN, JointKAN, OtherArmKAN, PoseErrorKAN, PoseKAN
from .mlp import JointErrorMLP, JointMLP, OtherArmMLP, PoseErrorMLP, PoseMLP


class Role(IntEnum):
    START = 0
    CURRENT = 1
    GOAL = 2
    ERROR = 3
    ENVIRONMENT = 4


class Modality(IntEnum):
    JOINT = 0
    POSE = 1
    OTHER_ARM = 2
    HISTORY_ACTION = 3


def relative_sinusoidal_encoding(length: int, d_model: int, device, dtype) -> torch.Tensor:
    """Deterministic temporal encodings for relative steps ``-L,...,-1``."""
    relative_steps = torch.arange(-length, 0, device=device, dtype=dtype).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model))
    encoding = torch.zeros(length, d_model, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(relative_steps * frequencies)
    encoding[:, 1::2] = torch.cos(relative_steps * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


class SemanticTokenizer(nn.Module):
    """Build a STACK of task, environment, and private history tokens."""

    def __init__(
        self,
        history_length: int,
        other_arm_points: int,
        kan_grid_size: int = 8,
        kan_spline_order: int = 3,
        semantic_encoder: str = "kan",
        mlp_hidden_dim: int = 128,
    ):
        super().__init__()
        self.history_length = history_length
        self.other_arm_points = other_arm_points
        self.semantic_encoder = semantic_encoder
        if semantic_encoder == "kan":
            kan_args = {"grid_size": kan_grid_size, "spline_order": kan_spline_order}
            self.joint = JointKAN(**kan_args)
            self.pose = PoseKAN(**kan_args)
            self.joint_error = JointErrorKAN(**kan_args)
            self.pose_error = PoseErrorKAN(**kan_args)
            self.other_arm = OtherArmKAN(other_arm_points, **kan_args)
        elif semantic_encoder == "mlp":
            self.joint = JointMLP(mlp_hidden_dim)
            self.pose = PoseMLP(mlp_hidden_dim)
            self.joint_error = JointErrorMLP(mlp_hidden_dim)
            self.pose_error = PoseErrorMLP(mlp_hidden_dim)
            self.other_arm = OtherArmMLP(other_arm_points, mlp_hidden_dim)
        else:
            raise ValueError(f"Unsupported semantic_encoder: {semantic_encoder}")
        self.history_action = nn.Sequential(nn.Linear(8, 64), nn.LayerNorm(64))
        self.role_embedding = nn.Embedding(len(Role), 64)
        self.modality_embedding = nn.Embedding(len(Modality), 64)
        self.control_query = nn.Parameter(torch.zeros(1, 1, 64))
        nn.init.normal_(self.control_query, std=0.02)

    def _decorate(self, token: torch.Tensor, role: Role, modality: Modality) -> torch.Tensor:
        return token + self.role_embedding.weight[int(role)] + self.modality_embedding.weight[int(modality)]

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        required = ("q_start", "q_current", "q_goal", "delta_q_goal", "x_start", "x_current", "x_goal", "delta_x_goal", "other_arm_geometry", "history_action", "history_mask")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Missing tokenizer inputs: {missing}")
        batch_size = batch["q_current"].shape[0]
        task_tokens = [
            self._decorate(self.joint(sincos_encode(batch["q_start"])), Role.START, Modality.JOINT),
            self._decorate(self.joint(sincos_encode(batch["q_current"])), Role.CURRENT, Modality.JOINT),
            self._decorate(self.joint(sincos_encode(batch["q_goal"])), Role.GOAL, Modality.JOINT),
            self._decorate(self.joint_error(batch["delta_q_goal"]), Role.ERROR, Modality.JOINT),
            self._decorate(self.pose(batch["x_start"]), Role.START, Modality.POSE),
            self._decorate(self.pose(batch["x_current"]), Role.CURRENT, Modality.POSE),
            self._decorate(self.pose(batch["x_goal"]), Role.GOAL, Modality.POSE),
            self._decorate(self.pose_error(batch["delta_x_goal"]), Role.ERROR, Modality.POSE),
        ]
        task = torch.stack(task_tokens, dim=1)
        other = batch["other_arm_geometry"]
        if other.shape[-2:] != (self.other_arm_points, 3):
            raise ValueError(f"Expected other_arm_geometry [B,{self.other_arm_points},3], got {other.shape}")
        other_token = self._decorate(self.other_arm(other.flatten(1)), Role.ENVIRONMENT, Modality.OTHER_ARM).unsqueeze(1)

        history = batch["history_action"]
        mask = batch["history_mask"].to(torch.bool)
        if history.shape[1:] != (self.history_length, 8) or mask.shape[1:] != (self.history_length,):
            raise ValueError(f"Invalid history shapes: {history.shape}, {mask.shape}")
        history_tokens = self._decorate(self.history_action(history), Role.CURRENT, Modality.HISTORY_ACTION)
        history_tokens = history_tokens + relative_sinusoidal_encoding(self.history_length, 64, history.device, history.dtype)

        control = self.control_query.expand(batch_size, -1, -1)
        tokens = torch.cat((control, task, other_token, history_tokens), dim=1)
        non_history = torch.zeros(batch_size, 10, dtype=torch.bool, device=history.device)
        # Transformer convention: True means ignore. Dataset convention: True means valid.
        key_padding_mask = torch.cat((non_history, ~mask), dim=1)
        return tokens, key_padding_mask
