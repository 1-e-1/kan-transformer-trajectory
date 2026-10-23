from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ExpertPlan:
    positions: np.ndarray
    actions: np.ndarray
    dt: float
    collision_free: bool
    planner: str


class ExpertInterface(ABC):
    @abstractmethod
    def plan(self, q_start: np.ndarray, q_goal: np.ndarray, **kwargs) -> ExpertPlan:
        raise NotImplementedError

