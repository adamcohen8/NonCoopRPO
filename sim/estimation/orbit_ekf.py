from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief
from sim.dynamics.orbit.two_body import propagate_two_body_rk4


@dataclass(frozen=True)
class OrbitEKFUpdateDiagnostics:
    measurement_available: bool
    update_applied: bool
    innovation: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))
    innovation_covariance: np.ndarray = field(default_factory=lambda: np.full((6, 6), np.nan))
    nis: float = float("nan")
    predicted_cov_trace: float = float("nan")
    posterior_cov_trace: float = float("nan")


@dataclass
class OrbitEKFEstimator(Estimator):
    mu_km3_s2: float
    dt_s: float
    process_noise_diag: np.ndarray
    meas_noise_diag: np.ndarray
    last_update_diagnostics: OrbitEKFUpdateDiagnostics | None = field(default=None, init=False, repr=False)
    _q: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)), init=False, repr=False)
    _r: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)), init=False, repr=False)
    _h: np.ndarray = field(default_factory=lambda: np.eye(6), init=False, repr=False)
    _i6: np.ndarray = field(default_factory=lambda: np.eye(6), init=False, repr=False)
    _zero_accel: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)

    def __post_init__(self) -> None:
        self.process_noise_diag = np.array(self.process_noise_diag, dtype=float)
        self.meas_noise_diag = np.array(self.meas_noise_diag, dtype=float)
        self._q = np.diag(self.process_noise_diag)
        self._r = np.diag(self.meas_noise_diag)

    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        x_prev = belief.state
        p_prev = belief.covariance

        x_pred = propagate_two_body_rk4(
            x_eci=x_prev,
            dt_s=self.dt_s,
            mu_km3_s2=self.mu_km3_s2,
            accel_cmd_eci_km_s2=self._zero_accel,
        )
        f = self._numerical_jacobian(x_prev, base=x_pred)
        p_pred = f @ p_prev @ f.T + self._q

        if measurement is None:
            self.last_update_diagnostics = OrbitEKFUpdateDiagnostics(
                measurement_available=False,
                update_applied=False,
                predicted_cov_trace=float(np.trace(p_pred)),
                posterior_cov_trace=float(np.trace(p_pred)),
            )
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=t_s)

        z = np.asarray(measurement.vector, dtype=float).reshape(-1)
        if z.size < 6:
            self.last_update_diagnostics = OrbitEKFUpdateDiagnostics(
                measurement_available=True,
                update_applied=False,
                predicted_cov_trace=float(np.trace(p_pred)),
                posterior_cov_trace=float(np.trace(p_pred)),
            )
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=t_s)
        z = z[:6]
        y = z - x_pred
        s = self._h @ p_pred @ self._h.T + self._r
        hp_t = p_pred @ self._h.T
        try:
            k = np.linalg.solve(s.T, hp_t.T).T
            s_y = np.linalg.solve(s, y)
        except np.linalg.LinAlgError:
            s_pinv = np.linalg.pinv(s)
            k = hp_t @ s_pinv
            s_y = s_pinv @ y
        x_upd = x_pred + k @ y
        i_kh = self._i6 - k @ self._h
        p_upd = i_kh @ p_pred @ i_kh.T + k @ self._r @ k.T
        p_upd = 0.5 * (p_upd + p_upd.T)
        nis = float(y.T @ s_y)
        self.last_update_diagnostics = OrbitEKFUpdateDiagnostics(
            measurement_available=True,
            update_applied=True,
            innovation=np.array(y, dtype=float),
            innovation_covariance=np.array(s, dtype=float),
            nis=nis,
            predicted_cov_trace=float(np.trace(p_pred)),
            posterior_cov_trace=float(np.trace(p_upd)),
        )
        return StateBelief(state=x_upd, covariance=p_upd, last_update_t_s=t_s)

    def _numerical_jacobian(self, x: np.ndarray, *, base: np.ndarray | None = None) -> np.ndarray:
        eps = 1e-6
        base_eval = base
        if base_eval is None:
            base_eval = propagate_two_body_rk4(
                x_eci=x,
                dt_s=self.dt_s,
                mu_km3_s2=self.mu_km3_s2,
                accel_cmd_eci_km_s2=self._zero_accel,
            )
        j = np.zeros((6, 6))
        for i in range(6):
            xp = x.copy()
            xp[i] += eps
            yp = propagate_two_body_rk4(
                x_eci=xp,
                dt_s=self.dt_s,
                mu_km3_s2=self.mu_km3_s2,
                accel_cmd_eci_km_s2=self._zero_accel,
            )
            j[:, i] = (yp - base_eval) / eps
        return j
