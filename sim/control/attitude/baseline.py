from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


@dataclass
class QuaternionPDController(Controller):
    kp: float = 0.1
    kd: float = 0.05
    max_torque_nm: float = 0.05

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        # Expected state layout: [r(3),v(3),q(4),w(3)] at minimum
        if belief.state.size < 13:
            return Command.zero()
        q = belief.state[6:10]
        w = belief.state[10:13]
        q_err_vec = q[1:4]
        torque = -self.kp * q_err_vec - self.kd * w
        n = np.linalg.norm(torque)
        if n > self.max_torque_nm and n > 0.0:
            torque *= self.max_torque_nm / n
        return Command(thrust_eci_km_s2=np.zeros(3), torque_body_nm=torque, mode_flags={"mode": "quat_pd"})


@dataclass
class SmallAngleLQRController(Controller):
    k_gain: np.ndarray
    max_torque_nm: float = 0.05

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        if belief.state.size < 13:
            return Command.zero()
        x = np.hstack((belief.state[7:10], belief.state[10:13]))
        torque = -self.k_gain @ x
        n = np.linalg.norm(torque)
        if n > self.max_torque_nm and n > 0.0:
            torque *= self.max_torque_nm / n
        return Command(thrust_eci_km_s2=np.zeros(3), torque_body_nm=torque, mode_flags={"mode": "lqr"})
