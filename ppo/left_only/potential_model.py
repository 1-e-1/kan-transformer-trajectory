"""V3 supervised self/fixed clearance and collision-potential network."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def periodic_features(q: torch.Tensor, harmonics: Sequence[int]) -> torch.Tensor:
    q = q.reshape(-1, q.shape[-1])
    return torch.cat([
        torch.stack((torch.sin(k * q), torch.cos(k * q)), dim=-1).flatten(-2) for k in harmonics
    ], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width * 2)
        self.linear2 = nn.Linear(width * 2, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.norm(value)
        residual = F.silu(self.linear1(residual))
        residual = self.dropout(self.linear2(residual))
        return value + residual / math.sqrt(2.0)


class PotentialChannel(nn.Module):
    def __init__(self, input_dim: int, width: int, blocks: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(ResidualBlock(width, dropout) for _ in range(blocks))
        self.norm = nn.LayerNorm(width)
        self.clearance = nn.Linear(width, 1)
        self.collision_logit = nn.Linear(width, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.silu(self.input(features))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = F.silu(self.norm(hidden))
        return self.clearance(hidden).squeeze(-1), self.collision_logit(hidden).squeeze(-1)


class LeftOnlyPotentialField(nn.Module):
    """Independent self/fixed channels; the right arm is never an input."""

    def __init__(
        self,
        dof: int = 8,
        harmonics: Sequence[int] = (1, 2, 4, 8),
        hidden_dim: int = 256,
        residual_blocks: int = 5,
        dropout: float = 0.02,
        fully_kan: bool = False,
        dense_kan_grid_size: int = 4,
        dense_kan_spline_order: int = 3,
    ) -> None:
        super().__init__()
        self.dof = int(dof)
        self.harmonics = tuple(int(k) for k in harmonics)
        self.hidden_dim = int(hidden_dim)
        self.residual_blocks = int(residual_blocks)
        self.dropout = float(dropout)
        self.fully_kan=bool(fully_kan)
        self.dense_kan_grid_size=int(dense_kan_grid_size)
        self.dense_kan_spline_order=int(dense_kan_spline_order)
        input_dim = 2 * self.dof * len(self.harmonics)
        self.self_channel = PotentialChannel(input_dim, hidden_dim, residual_blocks, dropout)
        self.fixed_channel = PotentialChannel(input_dim, hidden_dim, residual_blocks, dropout)
        if self.fully_kan:
            import sys
            from pathlib import Path
            sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
            from src.models.full_kan import replace_linears, assert_no_dense
            replace_linears(self,self.dense_kan_grid_size,self.dense_kan_spline_order)
            assert_no_dense(self)

    @classmethod
    def from_architecture(cls, architecture):
        keys=('dof','harmonics','hidden_dim','residual_blocks','dropout','fully_kan','dense_kan_grid_size','dense_kan_spline_order')
        return cls(**{key:architecture[key] for key in keys if key in architecture})

    def forward(self, q_left: torch.Tensor) -> dict[str, torch.Tensor]:
        features = periodic_features(q_left.reshape(-1, self.dof), self.harmonics)
        clearance_self, logit_self = self.self_channel(features)
        clearance_fixed, logit_fixed = self.fixed_channel(features)
        return {
            "clearance_self": clearance_self,
            "clearance_fixed": clearance_fixed,
            "collision_logit_self": logit_self,
            "collision_logit_fixed": logit_fixed,
            "potential_self": F.softplus(logit_self),
            "potential_fixed": F.softplus(logit_fixed),
        }

    def architecture(self) -> dict:
        return {
            "dof": self.dof, "harmonics": list(self.harmonics), "hidden_dim": self.hidden_dim,
            "residual_blocks": self.residual_blocks, "dropout": self.dropout,
            "input_dim": 2 * self.dof * len(self.harmonics),
            "outputs": ["clearance_self", "clearance_fixed", "collision_logit_self", "collision_logit_fixed"],
            "right_arm_input": False,
            "fully_kan": self.fully_kan,
            "dense_kan_grid_size": self.dense_kan_grid_size,
            "dense_kan_spline_order": self.dense_kan_spline_order,
        }
