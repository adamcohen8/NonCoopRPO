from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


@dataclass
class SnapAttitudeController(Controller):
    desired_state6: np.ndarray
    one_shot: bool = True
    _done: bool = field(default=False, init=False)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        if self.one_shot and self._done:
            return Command.zero()
        x = np.array(self.desired_state6, dtype=float)
        if x.size != 6:
            raise ValueError("desired_state6 must be [roll,pitch,yaw,wx,wy,wz].")
        self._done = True
        return Command(
            thrust_eci_km_s2=np.zeros(3),
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "snap",
                "snap_attitude_state6": x.tolist(),
            },
        )
