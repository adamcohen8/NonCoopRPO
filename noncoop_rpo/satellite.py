from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .attitude import AttitudeConstraint, AttitudeRateState, apply_attitude_rate_constraint
from .frames import eci_to_rsw_dcm
from .sat_params import SatParams
from .strategies import PolicyFn, Strategy, StrategyContext, StrategyLike, as_strategy


# Backward-compatibility alias.
Policy = PolicyFn


@dataclass
class SatState:
    t: float
    x_eci: np.ndarray
    dv_remaining_km_s: float
    dv_used_km_s: float = 0.0
    thrust_axis_ric: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    attitude_slew_rate_rad_s: float = 0.0


@dataclass
class Satellite:
    params: SatParams
    state: SatState
    strategy: Optional[Strategy] = None

    @classmethod
    def from_params(
        cls,
        params: SatParams,
        policy: Optional[Policy] = None,
        strategy: Optional[StrategyLike] = None,
    ) -> "Satellite":
        if policy is not None and strategy is not None:
            raise ValueError("Provide at most one of policy or strategy.")
        strategy_obj: Optional[Strategy] = None
        if strategy is not None:
            strategy_obj = as_strategy(strategy)
        elif policy is not None:
            strategy_obj = as_strategy(policy)
        return cls(
            params=params,
            state=SatState(
                t=0.0,
                x_eci=params.initial_eci_state(),
                dv_remaining_km_s=params.propellant_dv_km_s,
                thrust_axis_ric=np.array([1.0, 0.0, 0.0], dtype=float),
                attitude_slew_rate_rad_s=0.0,
            ),
            strategy=strategy_obj,
        )

    @property
    def policy(self) -> Optional[Policy]:
        """
        Backward-compatible access for callable policies.
        """
        if self.strategy is None:
            return None
        return lambda t_s, x_other_ric_curv, x_self_eci: self.strategy.command_accel_ric(
            StrategyContext(
                t_s=t_s,
                dt_s=0.0,
                x_self_eci=x_self_eci,
                x_other_ric_curv=x_other_ric_curv,
            )
        )

    @policy.setter
    def policy(self, value: Optional[Policy]) -> None:
        self.strategy = None if value is None else as_strategy(value)

    def set_strategy(self, strategy: Optional[StrategyLike]) -> None:
        self.strategy = None if strategy is None else as_strategy(strategy)

    def command_accel_ric(self, t: float, x_other_ric_curv: Optional[np.ndarray], dt_s: float) -> np.ndarray:
        if self.strategy is None:
            return np.zeros(3)
        if x_other_ric_curv is None:
            return np.zeros(3)

        context = StrategyContext(
            t_s=t,
            dt_s=dt_s,
            x_self_eci=self.state.x_eci,
            x_other_ric_curv=x_other_ric_curv,
        )
        u_ric = np.asarray(self.strategy.command_accel_ric(context), dtype=float)
        if u_ric.shape != (3,):
            raise ValueError("Strategy must return a 3-vector acceleration in RIC.")

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

        mag = np.linalg.norm(u_ric)
        if mag <= 0.0:
            return np.zeros(3)
        desired_dir = u_ric / mag
        constraint = AttitudeConstraint(
            enabled=self.params.attitude_control_enabled,
            inertia_body_kg_m2=self.params.inertia_body_kg_m2,
            max_torque_nm=self.params.max_torque_nm,
        )
        rate_state = AttitudeRateState(
            thrust_axis=np.asarray(self.state.thrust_axis_ric, dtype=float),
            slew_rate_rad_s=float(self.state.attitude_slew_rate_rad_s),
        )
        next_state = apply_attitude_rate_constraint(desired_dir, rate_state, dt_s, constraint)
        self.state.thrust_axis_ric = next_state.thrust_axis
        self.state.attitude_slew_rate_rad_s = next_state.slew_rate_rad_s
        return mag * next_state.thrust_axis

    def command_accel_eci(
        self, t: float, x_other_ric_curv: Optional[np.ndarray], host_x_eci: np.ndarray, dt_s: float
    ) -> np.ndarray:
        u_ric = self.command_accel_ric(t, x_other_ric_curv, dt_s)
        rsw = eci_to_rsw_dcm(host_x_eci[0:3], host_x_eci[3:6])
        return rsw @ u_ric
