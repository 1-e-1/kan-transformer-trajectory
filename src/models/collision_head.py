import torch
from torch import nn


class CollisionRiskHead(nn.Module):
    """Action-conditioned next-step self/external contact risk.

    The second output keeps the legacy ``inter_collision_logit`` key for
    checkpoint/API compatibility.  Its current semantics are inter-arm OR
    controlled-arm contact with any fixed robot/scene body.
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        # 8 candidate deltas + periodic sin/cos encoding of all 8 joints.
        self.head = nn.Sequential(
            nn.Linear(8 + 16, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, control_hidden, q_current, candidate_action):
        periodic_q = torch.cat((torch.sin(q_current), torch.cos(q_current)), dim=-1)
        # Deliberately independent of task goal/history: collision occupancy for
        # fixed geometry is a function of current q and the candidate delta.
        return self.head(torch.cat((periodic_q, candidate_action), dim=-1))
