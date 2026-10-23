"""Differentiable action projection along a learned joint-space collision boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BoundaryCorrection:
    corrected: bool
    proposed_risk: float
    corrected_risk: float
    boundary_scale: float
    tangent_scale: float


def _risk(model, control_hidden, q_current, action):
    logits = model.collision_head(
        control_hidden, q_current, action / model.config.action_delta_max_rad
    )
    probabilities = torch.sigmoid(logits)
    # Self and external collision are equal hard constraints.
    return probabilities.max(dim=-1).values, probabilities


def boundary_tangent_action(
    model,
    control_hidden: torch.Tensor,
    q_current: torch.Tensor,
    proposed_action: torch.Tensor,
    threshold: float = 0.5,
    bisection_steps: int = 12,
    tangent_gain: float = 0.75,
    inward_margin: float = 0.05,
):
    """Correct a risky action by retaining motion tangent to the learned boundary."""
    if model.collision_head is None:
        raise RuntimeError("Collision-aware projection requires collision_aux_heads=True")
    hidden = control_hidden.detach()
    q = q_current.detach()
    proposed = proposed_action.detach().clamp(
        -model.config.action_delta_max_rad, model.config.action_delta_max_rad
    )
    with torch.enable_grad():
        proposed_risk = _risk(model, hidden, q, proposed)[0]
    if bool(torch.all(proposed_risk <= threshold)):
        value = float(proposed_risk.max().detach().cpu())
        return proposed, BoundaryCorrection(False, value, value, 1.0, 1.0)

    zero = torch.zeros_like(proposed)
    with torch.enable_grad():
        zero_risk = _risk(model, hidden, q, zero)[0]
    # The current configuration was physically safe in the previous frame, so a
    # zero-action high-risk prediction is internally inconsistent.  There is no
    # learned safe anchor from which to construct a boundary; do not freeze the
    # arm on this out-of-distribution prediction.
    if bool(torch.any(zero_risk > threshold)):
        proposed_value = float(proposed_risk.max().detach().cpu())
        return proposed, BoundaryCorrection(False, proposed_value, proposed_value, 1.0, 1.0)

    low = torch.zeros((proposed.shape[0], 1), device=proposed.device)
    high = torch.ones_like(low)
    for _ in range(bisection_steps):
        middle = 0.5 * (low + high)
        with torch.enable_grad():
            middle_risk = _risk(model, hidden, q, proposed * middle)[0].unsqueeze(-1)
        safe = middle_risk <= threshold
        low = torch.where(safe, middle, low)
        high = torch.where(safe, high, middle)
    boundary = (proposed * low).detach().requires_grad_(True)
    with torch.enable_grad():
        boundary_risk = _risk(model, hidden, q, boundary)[0]
        gradient = torch.autograd.grad(boundary_risk.sum(), boundary, retain_graph=False)[0]
    normal = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tangent = proposed - (proposed * normal).sum(dim=-1, keepdim=True) * normal
    inward = inward_margin * model.config.action_delta_max_rad * normal
    base = boundary.detach() - inward

    selected = (proposed * low * (1.0 - inward_margin)).detach()
    selected_risk = _risk(model, hidden, q, selected)[0].detach()
    selected_scale = torch.zeros_like(low)
    for scale in torch.linspace(tangent_gain, 0.0, 9, device=proposed.device):
        candidate = (base + scale * tangent).clamp(
            -model.config.action_delta_max_rad, model.config.action_delta_max_rad
        )
        candidate_risk = _risk(model, hidden, q, candidate)[0].detach()
        accept = (candidate_risk <= threshold) & (selected_scale[:, 0] == 0)
        selected = torch.where(accept.unsqueeze(-1), candidate.detach(), selected)
        selected_risk = torch.where(accept, candidate_risk, selected_risk)
        selected_scale = torch.where(accept.unsqueeze(-1), torch.full_like(selected_scale, scale), selected_scale)
    return selected, BoundaryCorrection(
        True,
        float(proposed_risk.max().detach().cpu()),
        float(selected_risk.max().detach().cpu()),
        float(low.min().detach().cpu()),
        float(selected_scale.max().detach().cpu()),
    )
