from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


@dataclass
class ZeroController(Controller):
    simulated_runtime_ms: float = 0.0

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        if self.simulated_runtime_ms > 0.0:
            time.sleep(self.simulated_runtime_ms / 1000.0)
        return Command(thrust_eci_km_s2=np.zeros(3), torque_body_nm=np.zeros(3), mode_flags={"mode": "coast"})
