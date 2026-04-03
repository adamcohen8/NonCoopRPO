from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.dynamics.orbit.accelerations import OrbitContext


@dataclass
class OrbitUKFEstimator(Estimator):
    propagator: OrbitPropagator
    context: OrbitContext
    dt_s: float
    process_noise_diag: np.ndarray
    meas_noise_diag: np.ndarray
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0

    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        n = belief.state.size
        lam = self.alpha**2 * (n + self.kappa) - n
        wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = wm[0] + (1.0 - self.alpha**2 + self.beta)

        sigma = self._sigma_points(belief.state, belief.covariance, lam)
        sigma_pred = np.array(
            [
                self.propagator.propagate(
                    x_eci=s,
                    dt_s=self.dt_s,
                    t_s=belief.last_update_t_s,
                    command_accel_eci_km_s2=np.zeros(3),
                    env={},
                    ctx=self.context,
                )
                for s in sigma
            ]
        )

        x_pred = np.sum(wm[:, None] * sigma_pred, axis=0)
        p_pred = np.diag(self.process_noise_diag)
        for i in range(2 * n + 1):
            dx = sigma_pred[i] - x_pred
            p_pred += wc[i] * np.outer(dx, dx)

        if measurement is None:
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=t_s)
        z = np.asarray(measurement.vector, dtype=float).reshape(-1)
        if z.size < n:
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=t_s)

        h_sigma = sigma_pred
        z_pred = np.sum(wm[:, None] * h_sigma, axis=0)
        r = np.diag(self.meas_noise_diag)
        s_mat = r.copy()
        pxz = np.zeros((n, n))
        for i in range(2 * n + 1):
            dz = h_sigma[i] - z_pred
            dx = sigma_pred[i] - x_pred
            s_mat += wc[i] * np.outer(dz, dz)
            pxz += wc[i] * np.outer(dx, dz)

        try:
            k_gain = np.linalg.solve(s_mat.T, pxz.T).T
        except np.linalg.LinAlgError:
            k_gain = pxz @ np.linalg.pinv(s_mat)
        innovation = z[:n] - z_pred
        x_upd = x_pred + k_gain @ innovation
        i_kh = np.eye(n) - k_gain
        p_upd = i_kh @ p_pred @ i_kh.T + k_gain @ r @ k_gain.T
        p_upd = 0.5 * (p_upd + p_upd.T)
        return StateBelief(state=x_upd, covariance=p_upd, last_update_t_s=t_s)

    def _sigma_points(self, x: np.ndarray, p: np.ndarray, lam: float) -> np.ndarray:
        n = x.size
        c = np.linalg.cholesky((n + lam) * p + 1e-12 * np.eye(n))
        points = [x]
        for i in range(n):
            points.append(x + c[:, i])
            points.append(x - c[:, i])
        return np.array(points)
