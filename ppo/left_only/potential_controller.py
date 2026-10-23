"""Final supervised potential adapter and discrete-CBF safety filter."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from potential_model import LeftOnlyPotentialField



def wrap(q):
    return torch.remainder(q + math.pi, 2.0 * math.pi) - math.pi



class PotentialAdapter(nn.Module):
    def __init__(self, field, thresholds):
        super().__init__(); self.field = field
        self.register_buffer("threshold_self", torch.tensor(float(thresholds["self"])))
        self.register_buffer("threshold_fixed", torch.tensor(float(thresholds["fixed"])))

    def forward(self, q_current, candidate_action):
        q_candidate = wrap(q_current + candidate_action); out = self.field(q_candidate)
        p_self = torch.sigmoid(out["collision_logit_self"]); p_fixed = torch.sigmoid(out["collision_logit_fixed"])
        threshold_logits = torch.stack((torch.logit(self.threshold_self), torch.logit(self.threshold_fixed)))
        logits = torch.stack((out["collision_logit_self"], out["collision_logit_fixed"]), dim=-1)
        calibrated_potential = F.softplus(logits - threshold_logits)
        return {**out, "q_candidate": q_candidate, "probability_self": p_self,
                "probability_fixed": p_fixed, "calibrated_potential": calibrated_potential,
                "unsafe": torch.stack((p_self >= self.threshold_self, p_fixed >= self.threshold_fixed), -1)}

    def gradients(self, q_candidate):
        with torch.enable_grad():
            q = q_candidate.detach().requires_grad_(True)
            out = self.forward(q, torch.zeros_like(q)); potential = out["calibrated_potential"]
            gradients = []
            for channel in range(2):
                gradients.append(torch.autograd.grad(potential[:, channel].sum(), q, retain_graph=channel == 0)[0])
        return out, torch.stack(gradients, dim=1).detach()

    def barrier_values(self, q_current, candidate_action):
        out = self.forward(q_current, candidate_action)
        logits = torch.stack((out["collision_logit_self"], out["collision_logit_fixed"]), dim=-1)
        threshold_logits = torch.stack((torch.logit(self.threshold_self), torch.logit(self.threshold_fixed)))
        return {**out, "barrier": threshold_logits - logits}

    def barriers(self, q_candidate):
        """Return logit-space safety barriers and their joint gradients.

        h > 0 is on the calibrated safe side of the corresponding decision
        boundary.  Using logits avoids the vanishing gradient introduced by a
        saturated sigmoid probability.
        """
        with torch.enable_grad():
            q = q_candidate.detach().requires_grad_(True)
            raw = self.field(q)
            logits = torch.stack((raw["collision_logit_self"], raw["collision_logit_fixed"]), dim=-1)
            threshold_logits = torch.stack((torch.logit(self.threshold_self), torch.logit(self.threshold_fixed)))
            barrier = threshold_logits - logits
            gradients = []
            for channel in range(2):
                gradients.append(torch.autograd.grad(barrier[:, channel].sum(), q, retain_graph=channel == 0)[0])
        probability = torch.sigmoid(logits.detach())
        out = {
            **{key: value.detach() for key, value in raw.items()},
            "barrier": barrier.detach(),
            "probability_self": probability[:, 0],
            "probability_fixed": probability[:, 1],
            "calibrated_potential": F.softplus(logits.detach() - threshold_logits),
            "unsafe": probability >= torch.stack((self.threshold_self, self.threshold_fixed)),
        }
        return out, torch.stack(gradients, dim=1).detach()

    def dominant_barrier(self, q_candidate, margins):
        """One-backward active-set gradient for the most violated channel."""
        with torch.enable_grad():
            q=q_candidate.detach().requires_grad_(True);raw=self.field(q)
            logits=torch.stack((raw["collision_logit_self"],raw["collision_logit_fixed"]),-1)
            threshold_logits=torch.stack((torch.logit(self.threshold_self),torch.logit(self.threshold_fixed)))
            barrier=threshold_logits-logits
            violation=(q.new_tensor(margins)-barrier).clamp_min(0.0)
            channel=violation.argmax(-1)
            selected=barrier.gather(1,channel[:,None]).sum()
            selected_gradient=torch.autograd.grad(selected,q)[0]
        gradients=torch.zeros(q.shape[0],2,q.shape[1],device=q.device,dtype=q.dtype)
        gradients[torch.arange(q.shape[0],device=q.device),channel]=selected_gradient.detach()
        probability=torch.sigmoid(logits.detach())
        out={**{key:value.detach() for key,value in raw.items()},"barrier":barrier.detach(),
             "probability_self":probability[:,0],"probability_fixed":probability[:,1],
             "calibrated_potential":F.softplus(logits.detach()-threshold_logits),
             "unsafe":probability>=torch.stack((self.threshold_self,self.threshold_fixed))}
        return out,gradients


def load_potential(checkpoint_path, device="cuda:0"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True); architecture = checkpoint["architecture"]
    keys = ("dof", "harmonics", "hidden_dim", "residual_blocks", "dropout")
    field = LeftOnlyPotentialField.from_architecture(architecture).to(device)
    field.load_state_dict(checkpoint["model_state"]); field.eval()
    for parameter in field.parameters(): parameter.requires_grad_(False)
    return PotentialAdapter(field, checkpoint["thresholds"]).to(device).eval(), checkpoint


class BarrierProjectionShield:
    """Discrete CBF safety filter using minimum-norm sequential projections.

    The frozen KAN proposal is the optimization center.  If it satisfies both
    logit-space barriers (including configured margins), it is returned exactly.
    Otherwise the action is projected onto the linearized self/fixed safe
    half-spaces and the barriers are relinearized.  The best feasible-progress
    iterate is retained so a poor local gradient cannot increase predicted risk.
    """

    def __init__(
        self,
        adapter,
        action_max=0.0872664626,
        margin_self=1.5,
        margin_fixed=0.5,
        iterations=3,
        projection_passes=2,
        max_correction=0.0523598776,
        regularization=1e-4,
        dominant_only=False,
    ):
        self.adapter = adapter
        self.action_max = float(action_max)
        self.margins = (float(margin_self), float(margin_fixed))
        self.iterations = int(iterations)
        self.projection_passes = int(projection_passes)
        self.max_correction = float(max_correction)
        self.regularization = float(regularization)
        self.dominant_only = bool(dominant_only)

    def config(self):
        return {
            "type": "discrete_cbf_minimum_norm_projection",
            "barrier_space": "threshold_logit_minus_collision_logit",
            "margin_self": self.margins[0],
            "margin_fixed": self.margins[1],
            "iterations": self.iterations,
            "projection_passes": self.projection_passes,
            "action_max_rad_per_step": self.action_max,
            "max_correction_rad_per_step": self.max_correction,
            "regularization": self.regularization,
            "dominant_only": self.dominant_only,
        }

    def initial_state(self, batch, device):
        return {"correction_streak": torch.zeros(batch, device=device)}

    def _violation(self, barrier):
        margins = barrier.new_tensor(self.margins)
        return (margins - barrier).clamp_min(0.0)

    def _bounded_action(self, nominal, proposed):
        correction = proposed - nominal
        norm = torch.linalg.vector_norm(correction, dim=-1, keepdim=True).clamp_min(1e-8)
        scale = (self.max_correction / norm).clamp_max(1.0)
        return (nominal + correction * scale).clamp(-self.action_max, self.action_max)

    def correct(self, q, q_goal, nominal, state):
        del q_goal
        nominal = nominal.detach().clamp(-self.action_max, self.action_max)
        with torch.no_grad():
            initial_out = self.adapter.barrier_values(q, nominal)
        initial_violation = self._violation(initial_out["barrier"])
        proposed_max_violation = initial_violation.max(-1).values.detach()
        proposed_min_slack = (
            initial_out["barrier"] - initial_out["barrier"].new_tensor(self.margins)
        ).min(-1).values.detach()
        active = initial_violation.max(-1).values > 0

        proposed_probability = torch.stack(
            (initial_out["probability_self"], initial_out["probability_fixed"]), dim=-1
        )
        if not active.any():
            new_state = {"correction_streak": torch.zeros_like(state["correction_streak"])}
            metadata = {
                "corrected": torch.zeros_like(active),
                "ppo_candidate_selected": torch.zeros_like(active),
                "candidate_index": torch.full_like(active, -1, dtype=torch.long),
                "proposed_unsafe": active,
                "executed_unsafe": active,
                "proposed_probability": proposed_probability,
                "executed_probability": proposed_probability,
                "proposed_potential": initial_out["calibrated_potential"],
                "executed_potential": initial_out["calibrated_potential"],
                "proposed_barrier": initial_out["barrier"],
                "executed_barrier": initial_out["barrier"],
                "barrier_violation": initial_violation.max(-1).values,
                "proposed_barrier_violation": proposed_max_violation,
                "executed_barrier_violation": initial_violation.max(-1).values,
                "proposed_min_barrier_slack": proposed_min_slack,
            }
            return nominal, new_state, metadata

        # Compute gradients only after the cheap hard gate has fired.
        if self.dominant_only:initial_out,initial_gradients=self.adapter.dominant_barrier(wrap(q+nominal),self.margins)
        else:initial_out, initial_gradients = self.adapter.barriers(wrap(q + nominal))
        initial_violation = self._violation(initial_out["barrier"])

        action = nominal.clone()
        best_action = nominal.clone()
        best_out = initial_out
        best_violation = initial_violation.max(-1).values
        gradients = initial_gradients
        out = initial_out

        for iteration in range(self.iterations):
            violation = self._violation(out["barrier"])
            step = torch.zeros_like(action)
            # Alternating half-space projections solve the two-constraint
            # minimum-norm QP to sufficient accuracy, including conflicting
            # self/fixed normals, without an external QP dependency.
            for _ in range(self.projection_passes):
                for channel in range(2):
                    gradient = gradients[:, channel]
                    residual = violation[:, channel] - (gradient * step).sum(-1)
                    coefficient = residual.clamp_min(0.0) / (
                        gradient.square().sum(-1) + self.regularization
                    )
                    step = step + coefficient[:, None] * gradient
            proposed = self._bounded_action(nominal, action + step)
            action = torch.where(active[:, None], proposed, nominal)
            if iteration + 1 < self.iterations:
                if self.dominant_only:out,gradients=self.adapter.dominant_barrier(wrap(q+action),self.margins)
                else:out, gradients = self.adapter.barriers(wrap(q + action))
            else:
                with torch.no_grad():out = self.adapter.barrier_values(q, action)
            current_violation = self._violation(out["barrier"]).max(-1).values
            improved = current_violation < best_violation - 1e-7
            best_action = torch.where(improved[:, None], action, best_action)
            best_violation = torch.where(improved, current_violation, best_violation)
            if improved.any():
                # Dict tensors all have the same batch dimension.  Keeping the
                # selected output avoids an extra field query for metadata.
                best_out = {
                    key: torch.where(
                        improved.reshape((-1,) + (1,) * (value.ndim - 1)), value, best_out[key]
                    ) if key in best_out and value.shape[:1] == improved.shape else value
                    for key, value in out.items()
                }

        action = torch.where(active[:, None], best_action, nominal)
        with torch.no_grad():
            executed_out = self.adapter(q, action)
        correction = action - nominal
        corrected = torch.linalg.vector_norm(correction, dim=-1) > 1e-7
        new_streak = torch.where(corrected, state["correction_streak"] + 1, torch.zeros_like(state["correction_streak"]))
        new_state = {"correction_streak": new_streak.detach()}
        metadata = {
            "corrected": corrected,
            "ppo_candidate_selected": torch.zeros_like(active),
            "candidate_index": torch.full_like(active, -1, dtype=torch.long),
            "proposed_unsafe": active,
            "executed_unsafe": self._violation(best_out["barrier"]).max(-1).values > 0,
            "proposed_probability": proposed_probability,
            "executed_probability": torch.stack(
                (executed_out["probability_self"], executed_out["probability_fixed"]), dim=-1
            ),
            "proposed_potential": initial_out["calibrated_potential"],
            "executed_potential": executed_out["calibrated_potential"],
            "proposed_barrier": initial_out["barrier"],
            "executed_barrier": best_out["barrier"],
            "barrier_violation": best_violation.detach(),
            "proposed_barrier_violation": proposed_max_violation,
            "executed_barrier_violation": best_violation.detach(),
            "proposed_min_barrier_slack": proposed_min_slack,
        }
        return action.detach(), new_state, metadata
