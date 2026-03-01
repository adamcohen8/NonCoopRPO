from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


@dataclass
class StationkeepingController(Controller):
    target_state: np.ndarray
    kp_pos: float = 1e-5
    kd_vel: float = 5e-4
    max_accel_km_s2: float = 5e-5

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        pos_err = self.target_state[:3] - belief.state[:3]
        vel_err = self.target_state[3:] - belief.state[3:]
        a_cmd = self.kp_pos * pos_err + self.kd_vel * vel_err
        n = np.linalg.norm(a_cmd)
        if n > self.max_accel_km_s2 and n > 0.0:
            a_cmd *= self.max_accel_km_s2 / n
        return Command(thrust_eci_km_s2=a_cmd, torque_body_nm=np.zeros(3), mode_flags={"mode": "stationkeeping"})


@dataclass
class SafetyBarrierController(Controller):
    keep_out_radius_km: float
    kp_barrier: float = 5e-5
    max_accel_km_s2: float = 1e-4

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        r = belief.state[:3]
        norm_r = np.linalg.norm(r)
        if norm_r >= self.keep_out_radius_km:
            return Command.zero()
        direction = r / max(norm_r, 1e-9)
        a_cmd = self.kp_barrier * (self.keep_out_radius_km - norm_r) * direction
        n = np.linalg.norm(a_cmd)
        if n > self.max_accel_km_s2 and n > 0.0:
            a_cmd *= self.max_accel_km_s2 / n
        return Command(thrust_eci_km_s2=a_cmd, torque_body_nm=np.zeros(3), mode_flags={"mode": "barrier"})


@dataclass
class RiskThresholdController(Controller):
    risk_fn: callable
    nominal: Controller
    evasive: Controller
    threshold: float = 0.5

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        risk = float(self.risk_fn(belief, t_s))
        if risk >= self.threshold:
            return self.evasive.act(belief, t_s, budget_ms)
        return self.nominal.act(belief, t_s, budget_ms)
