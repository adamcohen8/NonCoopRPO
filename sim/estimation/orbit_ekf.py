from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief
from sim.dynamics.orbit.two_body import propagate_two_body_rk4


@dataclass
class OrbitEKFEstimator(Estimator):
    mu_km3_s2: float
    dt_s: float
    process_noise_diag: np.ndarray
    meas_noise_diag: np.ndarray

    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        x_prev = belief.state
        p_prev = belief.covariance

        x_pred = propagate_two_body_rk4(
            x_eci=x_prev,
            dt_s=self.dt_s,
            mu_km3_s2=self.mu_km3_s2,
            accel_cmd_eci_km_s2=np.zeros(3),
        )
        f = self._numerical_jacobian(x_prev)
        q = np.diag(self.process_noise_diag)
        p_pred = f @ p_prev @ f.T + q

        if measurement is None:
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=t_s)

        h = np.eye(6)
        r = np.diag(self.meas_noise_diag)
        z = measurement.vector[:6] if measurement.vector.size >= 6 else measurement.vector
        if z.size < 6:
            z = np.pad(z, (0, 6 - z.size))
        y = z - h @ x_pred
        s = h @ p_pred @ h.T + r
        k = p_pred @ h.T @ np.linalg.inv(s)
        x_upd = x_pred + k @ y
        p_upd = (np.eye(6) - k @ h) @ p_pred
        return StateBelief(state=x_upd, covariance=p_upd, last_update_t_s=t_s)

    def _numerical_jacobian(self, x: np.ndarray) -> np.ndarray:
        eps = 1e-6
        base = propagate_two_body_rk4(
            x_eci=x,
            dt_s=self.dt_s,
            mu_km3_s2=self.mu_km3_s2,
            accel_cmd_eci_km_s2=np.zeros(3),
        )
        j = np.zeros((6, 6))
        for i in range(6):
            xp = x.copy()
            xp[i] += eps
            yp = propagate_two_body_rk4(
                x_eci=xp,
                dt_s=self.dt_s,
                mu_km3_s2=self.mu_km3_s2,
                accel_cmd_eci_km_s2=np.zeros(3),
            )
            j[:, i] = (yp - base) / eps
        return j
