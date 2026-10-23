from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    action: float = 1.0
    terminal_error_boost: float = 0.0
    terminal_joint_scale: float = 0.25
    terminal_tcp_scale: float = 0.15
    velocity: float = 0.0
    acceleration: float = 0.0
    jerk: float = 0.0
    self_collision: float = 0.1
    inter_collision: float = 0.1


def teacher_preservation_loss(
    student_action: torch.Tensor,
    teacher_action: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    minimum_weight: float = 0.1,
    terminal_joint_scale: float = 0.25,
    terminal_tcp_scale: float = 0.15,
) -> torch.Tensor:
    """Preserve a stable teacher away from the goal while allowing terminal refinement."""
    if not 0.0 <= minimum_weight <= 1.0:
        raise ValueError("minimum_weight must be in [0, 1]")
    action_mask = batch.get(
        "action_loss_mask", torch.ones(student_action.shape[0], device=student_action.device)
    ).reshape(-1, 1)
    preservation_weight = torch.ones_like(action_mask)
    if "delta_q_goal" in batch and "delta_x_goal" in batch:
        joint_rms = torch.linalg.vector_norm(
            batch["delta_q_goal"], dim=-1, keepdim=True
        ) / student_action.shape[-1] ** 0.5
        tcp_position_error = torch.linalg.vector_norm(
            batch["delta_x_goal"][:, :3], dim=-1, keepdim=True
        )
        terminal_score = torch.exp(
            -joint_rms / max(terminal_joint_scale, 1e-6)
            -tcp_position_error / max(terminal_tcp_scale, 1e-6)
        )
        preservation_weight = minimum_weight + (1.0 - minimum_weight) * (
            1.0 - terminal_score
        )
    error = F.smooth_l1_loss(student_action, teacher_action, reduction="none")
    denominator = (
        (action_mask * preservation_weight).sum() * student_action.shape[-1]
    ).clamp_min(1e-6)
    return (error * action_mask * preservation_weight).sum() / denominator


def supervised_loss(output, batch, weights: LossWeights, velocity_limit=3.0, acceleration_limit=20.0, jerk_limit=200.0, self_pos_weight=1.0, inter_pos_weight=1.0):
    pred = output["action"]
    dt = batch["dt"].reshape(-1, 1).clamp_min(1e-6)
    history = batch["history_action"]
    mask = batch["history_mask"]
    prev = torch.where(mask[:, -1:].expand(-1, 8), history[:, -1], torch.zeros_like(pred))
    prev2 = torch.where(mask[:, -2:-1].expand(-1, 8), history[:, -2], prev)
    action_mask = batch.get("action_loss_mask", torch.ones(pred.shape[0], device=pred.device)).reshape(-1, 1)
    denominator = (action_mask.sum() * pred.shape[-1]).clamp_min(1.0)
    def masked_mean(value):
        return (value * action_mask).sum() / denominator
    action_weight = torch.ones_like(action_mask)
    if weights.terminal_error_boost > 0.0 and "delta_q_goal" in batch and "delta_x_goal" in batch:
        joint_rms = torch.linalg.vector_norm(
            batch["delta_q_goal"], dim=-1, keepdim=True
        ) / pred.shape[-1] ** 0.5
        tcp_position_error = torch.linalg.vector_norm(
            batch["delta_x_goal"][:, :3], dim=-1, keepdim=True
        )
        terminal_score = torch.exp(
            -joint_rms / max(weights.terminal_joint_scale, 1e-6)
            -tcp_position_error / max(weights.terminal_tcp_scale, 1e-6)
        )
        action_weight = 1.0 + weights.terminal_error_boost * terminal_score
    weighted_denominator = (
        (action_mask * action_weight).sum() * pred.shape[-1]
    ).clamp_min(1.0)
    action_error = F.smooth_l1_loss(
        pred, batch["expert_action"], reduction="none"
    )
    losses = {
        "action_loss": (action_error * action_mask * action_weight).sum() / weighted_denominator,
        "velocity_loss": masked_mean(((pred / dt) / velocity_limit).square()),
        "acceleration_loss": masked_mean((((pred - prev) / dt.square()) / acceleration_limit).square()),
        "jerk_loss": masked_mean((((pred - 2 * prev + prev2) / dt.pow(3)) / jerk_limit).square()),
    }
    zero = pred.new_zeros(())
    losses["self_collision_aux_loss"] = F.binary_cross_entropy_with_logits(output["self_collision_logit"], batch["self_collision_label"], pos_weight=pred.new_tensor(self_pos_weight)) if "self_collision_logit" in output else zero
    losses["inter_collision_aux_loss"] = F.binary_cross_entropy_with_logits(output["inter_collision_logit"], batch["inter_collision_label"], pos_weight=pred.new_tensor(inter_pos_weight)) if "inter_collision_logit" in output else zero
    losses["total_loss"] = (
        weights.action * losses["action_loss"]
        + weights.velocity * losses["velocity_loss"]
        + weights.acceleration * losses["acceleration_loss"]
        + weights.jerk * losses["jerk_loss"]
        + weights.self_collision * losses["self_collision_aux_loss"]
        + weights.inter_collision * losses["inter_collision_aux_loss"]
    )
    return losses
