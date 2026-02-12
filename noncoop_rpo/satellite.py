from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .frames import eci_to_rsw_dcm
from .sat_params import SatParams


# policy(t_s, x_other_ric_curv, x_self_eci) -> commanded accel in RIC (km/s^2)
Policy = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class SatState:
    t: float
    x_eci: np.ndarray
    dv_remaining_km_s: float
    dv_used_km_s: float = 0.0


@dataclass
class Satellite:
    params: SatParams
    state: SatState
    policy: Optional[Policy] = None

    @classmethod
    def from_params(cls, params: SatParams, policy: Optional[Policy] = None) -> "Satellite":
        return cls(
            params=params,
            state=SatState(
                t=0.0,
                x_eci=params.initial_eci_state(),
                dv_remaining_km_s=params.propellant_dv_km_s,
            ),
            policy=policy,
        )

    def command_accel_ric(self, t: float, x_other_ric_curv: Optional[np.ndarray], dt_s: float) -> np.ndarray:
        if self.policy is None:
            return np.zeros(3)
        if x_other_ric_curv is None:
            return np.zeros(3)

        u_ric = np.asarray(self.policy(t, x_other_ric_curv, self.state.x_eci), dtype=float)
        if u_ric.shape != (3,):
            raise ValueError("Policy must return a 3-vector acceleration in RIC.")

        norm_u = np.linalg.norm(u_ric)
        if norm_u <= 0.0:
            return np.zeros(3)

        if norm_u < self.params.min_accel_km_s2:
            # Thruster cannot realize commands below minimum thrust: no-fire mode.
            return np.zeros(3)

        if self.params.max_accel_km_s2 > 0.0 and norm_u > self.params.max_accel_km_s2:
            u_ric = (self.params.max_accel_km_s2 / norm_u) * u_ric
            norm_u = self.params.max_accel_km_s2

        if np.isfinite(self.state.dv_remaining_km_s):
            if self.state.dv_remaining_km_s <= 0.0:
                return np.zeros(3)
            dv_req = norm_u * dt_s
            if dv_req > self.state.dv_remaining_km_s and dt_s > 0.0:
                u_ric = (self.state.dv_remaining_km_s / dv_req) * u_ric
                norm_u = np.linalg.norm(u_ric)
                dv_req = self.state.dv_remaining_km_s
            self.state.dv_remaining_km_s = max(0.0, self.state.dv_remaining_km_s - dv_req)
            self.state.dv_used_km_s += dv_req

        return u_ric

    def command_accel_eci(
        self, t: float, x_other_ric_curv: Optional[np.ndarray], host_x_eci: np.ndarray, dt_s: float
    ) -> np.ndarray:
        u_ric = self.command_accel_ric(t, x_other_ric_curv, dt_s)
        rsw = eci_to_rsw_dcm(host_x_eci[0:3], host_x_eci[3:6])
        return rsw @ u_ric
