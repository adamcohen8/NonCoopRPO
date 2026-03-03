from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.utils.quaternion import normalize_quaternion, omega_matrix


@dataclass
class JointStateEstimator(Estimator):
    orbit_estimator: OrbitEKFEstimator
    dt_s: float
    quat_blend: float = 0.35
    omega_blend: float = 0.45
    attitude_process_var: float = 1e-8
    attitude_meas_var: float = 1e-6

    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        if belief.state.size < 13:
            raise ValueError("JointStateEstimator expects 13-state belief [r,v,q,w].")

        x = belief.state
        p = belief.covariance

        orb_belief = StateBelief(state=x[:6], covariance=p[:6, :6], last_update_t_s=belief.last_update_t_s)
        orb_meas = None if measurement is None else Measurement(vector=measurement.vector[:6], t_s=measurement.t_s)
        orb_upd = self.orbit_estimator.update(orb_belief, orb_meas, t_s)

        q_prev = normalize_quaternion(x[6:10])
        w_prev = x[10:13]

        q_pred = normalize_quaternion(q_prev + 0.5 * self.dt_s * (omega_matrix(w_prev) @ q_prev))
        w_pred = w_prev.copy()

        if measurement is not None and measurement.vector.size >= 13:
            q_meas = normalize_quaternion(measurement.vector[6:10])
            if np.dot(q_meas, q_pred) < 0.0:
                q_meas = -q_meas
            w_meas = measurement.vector[10:13]

            q_upd = normalize_quaternion((1.0 - self.quat_blend) * q_pred + self.quat_blend * q_meas)
            w_upd = (1.0 - self.omega_blend) * w_pred + self.omega_blend * w_meas
        else:
            q_upd = q_pred
            w_upd = w_pred

        x_upd = np.hstack((orb_upd.state, q_upd, w_upd))

        p_upd = np.zeros((13, 13))
        p_upd[:6, :6] = orb_upd.covariance
        att_proc = self.attitude_process_var * np.eye(7)
        p_att_prev = p[6:13, 6:13] if p.shape == (13, 13) else np.eye(7) * self.attitude_meas_var
        if measurement is not None and measurement.vector.size >= 13:
            k_att = np.diag([self.quat_blend] * 4 + [self.omega_blend] * 3)
            r_att = self.attitude_meas_var * np.eye(7)
            p_att = (np.eye(7) - k_att) @ (p_att_prev + att_proc) @ (np.eye(7) - k_att).T + k_att @ r_att @ k_att.T
        else:
            p_att = p_att_prev + att_proc
        p_upd[6:13, 6:13] = p_att

        return StateBelief(state=x_upd, covariance=p_upd, last_update_t_s=t_s)
