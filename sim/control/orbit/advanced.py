from __future__ import annotations

from dataclasses import dataclass

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


@dataclass
class RobustMPCController(Controller):
    fallback: Controller

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        # Placeholder for robust MPC optimization; delegates until optimizer is added.
        return self.fallback.act(belief, t_s, budget_ms)


@dataclass
class StochasticPolicyController(Controller):
    policy_fn: callable

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        return self.policy_fn(belief, t_s, budget_ms)
