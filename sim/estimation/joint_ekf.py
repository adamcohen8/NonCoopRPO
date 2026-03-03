from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief
from sim.estimation.attitude_ekf import AttitudeEKFEstimator
from sim.estimation.orbit_ekf import OrbitEKFEstimator


@dataclass
class JointStateEKFEstimator(Estimator):
    orbit_estimator: OrbitEKFEstimator
    attitude_estimator: AttitudeEKFEstimator

    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        if belief.state.size < 13:
            raise ValueError("JointStateEKFEstimator expects 13-state belief [r,v,q,w].")

        x = belief.state
        p = belief.covariance

        orb_belief = StateBelief(state=x[:6], covariance=p[:6, :6], last_update_t_s=belief.last_update_t_s)
        att_belief = StateBelief(state=x[6:13], covariance=p[6:13, 6:13], last_update_t_s=belief.last_update_t_s)

        orb_meas = None
        att_meas = None
        if measurement is not None:
            if measurement.vector.size >= 6:
                orb_meas = Measurement(vector=measurement.vector[:6], t_s=measurement.t_s)
            if measurement.vector.size >= 13:
                att_meas = Measurement(vector=measurement.vector[6:13], t_s=measurement.t_s)

        orb_upd = self.orbit_estimator.update(orb_belief, orb_meas, t_s)
        att_upd = self.attitude_estimator.update(att_belief, att_meas, t_s)

        x_upd = np.hstack((orb_upd.state, att_upd.state))
        p_upd = np.zeros((13, 13))
        p_upd[:6, :6] = orb_upd.covariance
        p_upd[6:13, 6:13] = att_upd.covariance
        return StateBelief(state=x_upd, covariance=p_upd, last_update_t_s=t_s)
