from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


@dataclass
class ZeroTorqueController(Controller):
    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        return Command(thrust_eci_km_s2=np.zeros(3), torque_body_nm=np.zeros(3), mode_flags={"mode": "tumble"})
