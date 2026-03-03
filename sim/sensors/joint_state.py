from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import SensorModel
from sim.core.models import Measurement, StateTruth
from sim.utils.quaternion import normalize_quaternion


@dataclass
class JointStateSensor(SensorModel):
    pos_sigma_km: float = 1e-3
    vel_sigma_km_s: float = 1e-5
    quat_sigma: float = 1e-3
    omega_sigma_rad_s: float = 1e-4
    update_cadence_s: float = 1.0
    dropout_prob: float = 0.0
    rng: np.random.Generator = np.random.default_rng(0)
    _last_update_t_s: float = -np.inf

    def measure(self, truth: StateTruth, env: dict, t_s: float) -> Measurement | None:
        if t_s - self._last_update_t_s < self.update_cadence_s:
            return None
        if self.rng.random() < self.dropout_prob:
            return None

        pos = truth.position_eci_km + self.rng.normal(0.0, self.pos_sigma_km, size=3)
        vel = truth.velocity_eci_km_s + self.rng.normal(0.0, self.vel_sigma_km_s, size=3)

        q = truth.attitude_quat_bn.copy()
        q[1:] += self.rng.normal(0.0, self.quat_sigma, size=3)
        q = normalize_quaternion(q)

        w = truth.angular_rate_body_rad_s + self.rng.normal(0.0, self.omega_sigma_rad_s, size=3)

        z = np.hstack((pos, vel, q, w))
        self._last_update_t_s = t_s
        return Measurement(vector=z, t_s=t_s)
