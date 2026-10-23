"""Fully-connected semantic encoders used as a KAN-free baseline."""

from __future__ import annotations

from torch import nn


class MLPEncoder(nn.Module):
    """Two-layer GELU MLP followed by token-wise LayerNorm."""

    def __init__(self, in_features: int, out_features: int = 64, hidden_features: int = 128):
        super().__init__()
        if hidden_features < 1:
            raise ValueError("hidden_features must be positive")
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        return self.norm(self.mlp(x))


class JointMLP(MLPEncoder):
    def __init__(self, hidden_features: int = 128):
        super().__init__(16, 64, hidden_features)


class PoseMLP(MLPEncoder):
    def __init__(self, hidden_features: int = 128):
        super().__init__(9, 64, hidden_features)


class JointErrorMLP(MLPEncoder):
    def __init__(self, hidden_features: int = 128):
        super().__init__(8, 64, hidden_features)


class PoseErrorMLP(MLPEncoder):
    def __init__(self, hidden_features: int = 128):
        super().__init__(6, 64, hidden_features)


class OtherArmMLP(MLPEncoder):
    def __init__(self, point_count: int, hidden_features: int = 128):
        super().__init__(point_count * 3, 64, hidden_features)
