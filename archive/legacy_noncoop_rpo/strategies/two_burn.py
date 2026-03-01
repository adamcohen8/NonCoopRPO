from __future__ import annotations

import numpy as np

from ..frames import eci2hcw, hcw2eci, ric_curv_to_rect
from .base import StrategyContext


class TwoBurnFirstLegStrategy:
    """
    Continuous approximation of the first burn in a 2-burn HCW rendezvous sequence.

    At each step this strategy:
    1) Interprets current relative state in RIC-rect coordinates,
    2) Solves for the ideal impulsive initial relative velocity that would drive
       position to the rendezvous point after `time_of_flight_s`,
    3) Commands acceleration proportional to the required delta-v (non-impulsive).

    This intentionally does not include the terminal (second-burn) velocity match.
    """

    def __init__(
        self,
        n_rad_s: float,
        time_of_flight_s: float = 1800.0,
        burn_time_constant_s: float = 120.0,
        min_sin_nt_abs: float = 1e-4,
        fixed_arrival_time: bool = False,
        coast_after_first_leg: bool = False,
        cross_track_fixed_arrival: bool = True,
        min_time_to_go_s: float = 1.0,
    ):
        if n_rad_s <= 0.0:
            raise ValueError("n_rad_s must be positive.")
        if time_of_flight_s <= 0.0:
            raise ValueError("time_of_flight_s must be positive.")
        if burn_time_constant_s <= 0.0:
            raise ValueError("burn_time_constant_s must be positive.")
        if min_time_to_go_s <= 0.0:
            raise ValueError("min_time_to_go_s must be positive.")
        self.n = float(n_rad_s)
        self.tof = float(time_of_flight_s)
        self.tau = float(burn_time_constant_s)
        self.min_sin_nt_abs = float(min_sin_nt_abs)
        self.fixed_arrival_time = bool(fixed_arrival_time)
        self.coast_after_first_leg = bool(coast_after_first_leg)
        self.cross_track_fixed_arrival = bool(cross_track_fixed_arrival)
        self.min_time_to_go_s = float(min_time_to_go_s)
        self._start_time_s: float | None = None

    def _required_first_burn_velocity(
        self,
        r_rel: np.ndarray,
        time_to_go_planar_s: float,
        time_to_go_cross_s: float,
    ) -> np.ndarray:
        n = self.n
        t_p = max(float(time_to_go_planar_s), self.min_time_to_go_s)
        nt_p = n * t_p
        s_p = np.sin(nt_p)
        c_p = np.cos(nt_p)

        # Planar solve for [vR0, vI0]
        a_planar = np.array(
            [
                [s_p / n, 2.0 * (1.0 - c_p) / n],
                [-2.0 * (1.0 - c_p) / n, (4.0 * s_p - 3.0 * nt_p) / n],
            ],
            dtype=float,
        )
        b_planar = -np.array(
            [
                (4.0 - 3.0 * c_p) * r_rel[0],
                6.0 * (s_p - nt_p) * r_rel[0] + r_rel[1],
            ],
            dtype=float,
        )
        try:
            v_ri = np.linalg.solve(a_planar, b_planar)
        except np.linalg.LinAlgError:
            v_ri, *_ = np.linalg.lstsq(a_planar, b_planar, rcond=None)
            v_ri = np.asarray(v_ri, dtype=float)

        # Cross-track channel is decoupled: z(t) = c z0 + (s/n) vz0
        t_c = max(float(time_to_go_cross_s), self.min_time_to_go_s)
        nt_c = n * t_c
        s_c = np.sin(nt_c)
        c_c = np.cos(nt_c)
        if abs(s_c) < self.min_sin_nt_abs:
            v_c = 0.0
        else:
            v_c = float(-(n * c_c / s_c) * r_rel[2])

        return np.array([float(v_ri[0]), float(v_ri[1]), v_c], dtype=float)

    def command_accel_ric(self, context: StrategyContext) -> np.ndarray:
        if context.x_other_ric_curv is None:
            return np.zeros(3, dtype=float)
        frame = str(context.state.get("relative_state_frame", "ric_curv")).lower()
        if frame != "ric_curv":
            raise ValueError(
                "TwoBurnFirstLegStrategy expects curvilinear RIC relative states (ric_curv). "
                f"Received: {frame}"
            )

        # The observed relative state is "target wrt self" in the self-hosted RIC frame.
        # curv->rect therefore uses the self radius as host radius.
        r_self = float(np.linalg.norm(context.x_self_eci[0:3]))
        x_target_wrt_self_rect = ric_curv_to_rect(
            np.asarray(context.x_other_ric_curv, dtype=float),
            r0=max(r_self, 1e-6),
        )

        # Convert to "self wrt target" rectangular state exactly via ECI.
        x_target_eci = hcw2eci(context.x_self_eci, x_target_wrt_self_rect)
        x_self_wrt_target_rect = eci2hcw(x_target_eci, context.x_self_eci)
        r_rel = x_self_wrt_target_rect[0:3]
        v_rel = x_self_wrt_target_rect[3:6]

        if self._start_time_s is None or context.t_s < self._start_time_s:
            self._start_time_s = float(context.t_s)
        elapsed_s = max(0.0, float(context.t_s) - self._start_time_s)
        if self.coast_after_first_leg and elapsed_s >= self.tof:
            return np.zeros(3, dtype=float)

        time_to_go_fixed = max(self.min_time_to_go_s, self.tof - elapsed_s)
        time_to_go_planar = time_to_go_fixed if self.fixed_arrival_time else self.tof
        time_to_go_cross = time_to_go_fixed if self.cross_track_fixed_arrival else time_to_go_planar

        v_req = self._required_first_burn_velocity(
            r_rel,
            time_to_go_planar_s=time_to_go_planar,
            time_to_go_cross_s=time_to_go_cross,
        )
        dv1 = v_req - v_rel

        # Non-impulsive execution: steer acceleration toward the first-burn dv.
        u_ric = -dv1 / self.tau
        return np.asarray(u_ric, dtype=float)
