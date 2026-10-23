"""A compact, dependency-free B-spline Kolmogorov-Arnold Network implementation."""

from __future__ import annotations

import torch
from torch import nn


class BSplineKANLayer(nn.Module):
    """KAN layer with one trainable univariate cubic spline per input-output edge.

    ``y_o = sum_i (w_base[o,i] SiLU(x_i) + sum_n w_spline[o,i,n] B_n(x_i))``.
    Inputs are smoothly mapped into the spline grid with tanh, while the base branch
    retains an unconstrained, differentiable signal.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 8, spline_order: int = 3, base_activation: str = 'silu'):
        super().__init__()
        if grid_size < 2 or spline_order < 1:
            raise ValueError("grid_size must be >=2 and spline_order >=1")
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        if base_activation not in ('silu', 'identity'):
            raise ValueError(base_activation)
        self.base_activation = base_activation
        step = 2.0 / grid_size
        knots = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * step - 1.0
        self.register_buffer("knots", knots)
        self.num_basis = knots.numel() - spline_order - 1
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, self.num_basis))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02 / max(1, self.in_features) ** 0.5)
        fan_in = max(1, self.in_features)
        bound = fan_in**-0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def b_spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate basis functions; output shape is ``[..., in_features, num_basis]``."""
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dim {self.in_features}, got {x.shape}")
        x = torch.tanh(x).unsqueeze(-1)
        knots = self.knots.to(dtype=x.dtype, device=x.device)
        basis = ((x >= knots[:-1]) & (x < knots[1:])).to(x.dtype)
        for degree in range(1, self.spline_order + 1):
            count = knots.numel() - degree - 1
            left_den = knots[degree : degree + count] - knots[:count]
            right_den = knots[degree + 1 : degree + 1 + count] - knots[1 : 1 + count]
            left = (x - knots[:count]) / left_den.clamp_min(torch.finfo(x.dtype).eps)
            right = (knots[degree + 1 : degree + 1 + count] - x) / right_den.clamp_min(torch.finfo(x.dtype).eps)
            basis = left * basis[..., :count] + right * basis[..., 1 : count + 1]
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_input = torch.nn.functional.silu(x) if self.base_activation == 'silu' else x
        base = base_input @ self.base_weight.transpose(0, 1)
        spline = torch.einsum("...in,oin->...o", self.b_spline_basis(x), self.spline_weight)
        return base + spline + self.bias


class KANEncoder(nn.Module):
    def __init__(self, in_features: int, out_features: int = 64, grid_size: int = 8, spline_order: int = 3):
        super().__init__()
        self.kan = BSplineKANLayer(in_features, out_features, grid_size, spline_order)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.kan(x))


class JointKAN(KANEncoder):
    def __init__(self, **kwargs):
        super().__init__(16, 64, **kwargs)


class PoseKAN(KANEncoder):
    def __init__(self, **kwargs):
        super().__init__(9, 64, **kwargs)


class JointErrorKAN(KANEncoder):
    def __init__(self, **kwargs):
        super().__init__(8, 64, **kwargs)


class PoseErrorKAN(KANEncoder):
    def __init__(self, **kwargs):
        super().__init__(6, 64, **kwargs)


class OtherArmKAN(KANEncoder):
    def __init__(self, point_count: int, **kwargs):
        super().__init__(point_count * 3, 64, **kwargs)
