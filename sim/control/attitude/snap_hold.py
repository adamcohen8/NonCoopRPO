from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


@dataclass
class SnapAndHoldRICAttitudeController(Controller):
    desired_state6_ric: np.ndarray

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        x = np.array(self.desired_state6_ric, dtype=float)
        if x.size != 6:
            raise ValueError("desired_state6_ric must be [yaw_R,roll_I,pitch_C,wx,wy,wz].")
        return Command(
            thrust_eci_km_s2=np.zeros(3),
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "snap_hold_ric",
                "snap_hold_ric_state6": x.tolist(),
            },
        )
