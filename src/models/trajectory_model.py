from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .collision_head import CollisionRiskHead
from .tokenizer import SemanticTokenizer
from .transformer import ControlTransformer
from src.data.normalization import ObservationNormalizer


@dataclass(frozen=True)
class TrajectoryModelConfig:
    history_length: int = 64
    other_arm_points: int = 9
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.0
    kan_grid_size: int = 8
    kan_spline_order: int = 3
    semantic_encoder: str = "kan"
    mlp_hidden_dim: int = 128
    action_delta_max_rad: float = 0.08726646259971647
    collision_aux_heads: bool = True
    normalization: dict | None = None
    fully_kan: bool = False
    dense_kan_grid_size: int = 4
    dense_kan_spline_order: int = 3


class TrajectoryModel(nn.Module):
    """One arm only. Instantiate this class twice; never share its parameters."""

    def __init__(self, config: TrajectoryModelConfig):
        super().__init__()
        self.config = config
        self.normalizer = ObservationNormalizer(config.normalization)
        self.tokenizer = SemanticTokenizer(
            config.history_length,
            config.other_arm_points,
            config.kan_grid_size,
            config.kan_spline_order,
            config.semantic_encoder,
            config.mlp_hidden_dim,
        )
        self.transformer = ControlTransformer(config.d_model, config.nhead, config.num_layers, config.dim_feedforward, config.dropout, config.fully_kan, config.dense_kan_grid_size, config.dense_kan_spline_order)
        self.action_head = nn.Sequential(nn.Linear(config.d_model, 128), nn.GELU(), nn.Linear(128, 8))
        self.collision_head = CollisionRiskHead(config.d_model) if config.collision_aux_heads else None
        if config.fully_kan:
            from .full_kan import replace_linears, assert_no_dense
            replace_linears(self,config.dense_kan_grid_size,config.dense_kan_spline_order)
            assert_no_dense(self)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens, mask = self.tokenizer(self.normalizer(batch))
        control_hidden = self.transformer(tokens, mask)
        action = self.config.action_delta_max_rad * torch.tanh(self.action_head(control_hidden))
        output = {"action": action, "control_hidden": control_hidden}
        if self.collision_head is not None:
            query_action = batch.get("collision_query_action", action)
            collision_logits = self.collision_head(
                control_hidden,
                batch["q_current"],
                query_action / self.config.action_delta_max_rad,
            )
            output["self_collision_logit"] = collision_logits[:, 0]
            # Compatibility name; semantically this is now external collision
            # (other arm or any fixed robot/scene body).
            output["inter_collision_logit"] = collision_logits[:, 1]
        return output

    def load_compatible_state_dict(self, state_dict, allow_semantic_encoder_reinit=False):
        """Warm-load legacy checkpoints while reinitializing shape-changed aux heads."""
        current = self.state_dict()
        compatible = {key: value for key, value in state_dict.items() if key in current and current[key].shape == value.shape}
        skipped = sorted(set(state_dict) - set(compatible))
        result = self.load_state_dict(compatible, strict=False)
        semantic_prefixes = (
            "tokenizer.joint.",
            "tokenizer.pose.",
            "tokenizer.joint_error.",
            "tokenizer.pose_error.",
            "tokenizer.other_arm.",
        )
        unexpected_core = [
            key for key in skipped
            if not key.startswith("collision_head.")
            and not (allow_semantic_encoder_reinit and key.startswith(semantic_prefixes))
        ]
        if unexpected_core:
            raise RuntimeError(f"Incompatible non-auxiliary checkpoint tensors: {unexpected_core}")
        return result, skipped


def assert_models_independent(left: nn.Module, right: nn.Module) -> None:
    if left is right:
        raise AssertionError("Left and right models are the same object")
    left_ptrs = {parameter.data_ptr() for parameter in left.parameters()}
    right_ptrs = {parameter.data_ptr() for parameter in right.parameters()}
    if left_ptrs.intersection(right_ptrs):
        raise AssertionError("Left and right models share parameter storage")
