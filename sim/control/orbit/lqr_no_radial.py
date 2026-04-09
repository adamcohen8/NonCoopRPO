from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.control.orbit.lqr import HCWLQRController
from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.frames import ric_curv_to_rect, ric_dcm_ir_from_rv


@dataclass
class HCWNoRadialLQRController(Controller):
    mean_motion_rad_s: float
    max_accel_km_s2: float
    design_dt_s: float = 10.0
    ric_curv_state_slice: tuple[int, int] = (0, 6)
    chief_eci_state_slice: tuple[int, int] = (6, 12)
    state_signs: np.ndarray = field(default_factory=lambda: np.ones(6))
    q_weights: np.ndarray = field(default_factory=lambda: np.array([8.66, 8.66, 8.66, 1.33, 1.33, 1.33]) * 1e3)
    r_weights: np.ndarray = field(default_factory=lambda: np.ones(2) * 1.94e13)
    riccati_max_iter: int = 500
    riccati_tol: float = 1e-8
    _ad: np.ndarray = field(init=False, repr=False)
    _bd: np.ndarray = field(init=False, repr=False)
    _k_gain: np.ndarray = field(init=False, repr=False)

    @staticmethod
    def _position_output_indices() -> list[int]:
        return [1, 2]

    @staticmethod
    def _control_axes() -> list[str]:
        return ["I", "C"]

    def linear_system_summary(self) -> dict[str, object]:
        closed_loop = self._ad - self._bd @ self._k_gain
        zeros = []
        for axis_idx, axis_label in enumerate(self._control_axes()):
            out_idx = self._position_output_indices()[axis_idx]
            c_row = np.zeros(self._ad.shape[0], dtype=float)
            c_row[out_idx] = 1.0
            z = HCWLQRController._position_channel_zeros(self._ad, self._bd[:, axis_idx], c_row)
            zeros.append({"axis": axis_label, "zeros": HCWLQRController._complex_pairs(z)})
        return {
            "system_type": "discrete_state_feedback",
            "sample_time_s": float(self.design_dt_s),
            "law_label": HCWLQRController._control_law_label(self.state_signs),
            "control_axes": self._control_axes(),
            "open_loop_poles": HCWLQRController._complex_pairs(np.linalg.eigvals(self._ad)),
            "closed_loop_poles": HCWLQRController._complex_pairs(np.linalg.eigvals(closed_loop)),
            "position_channel_zeros": zeros,
        }

    def __post_init__(self) -> None:
        if self.mean_motion_rad_s <= 0.0:
            raise ValueError("mean_motion_rad_s must be positive.")
        if self.max_accel_km_s2 < 0.0:
            raise ValueError("max_accel_km_s2 must be non-negative.")
        if self.design_dt_s <= 0.0:
            raise ValueError("design_dt_s must be positive.")
        if self.ric_curv_state_slice[1] - self.ric_curv_state_slice[0] != 6:
            raise ValueError("ric_curv_state_slice must select exactly 6 elements.")
        if self.chief_eci_state_slice[1] - self.chief_eci_state_slice[0] != 6:
            raise ValueError("chief_eci_state_slice must select exactly 6 elements.")
        if self.riccati_max_iter <= 0:
            raise ValueError("riccati_max_iter must be positive.")
        if self.riccati_tol <= 0.0:
            raise ValueError("riccati_tol must be positive.")

        signs = np.array(self.state_signs, dtype=float).reshape(-1)
        if signs.size != 6:
            raise ValueError("state_signs must be length-6.")
        signs[signs == 0.0] = 1.0
        self.state_signs = np.sign(signs)

        q = np.array(self.q_weights, dtype=float).reshape(-1)
        if q.size == 1:
            q = np.full(6, float(q[0]))
        if q.size != 6 or np.any(q < 0.0):
            raise ValueError("q_weights must be non-negative scalar or length-6 vector.")

        r = np.array(self.r_weights, dtype=float).reshape(-1)
        if r.size == 1:
            r = np.full(2, float(r[0]))
        if r.size != 2 or np.any(r <= 0.0):
            raise ValueError("r_weights must be positive scalar or length-2 vector.")

        n = self.mean_motion_rad_s
        A = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [3.0 * n * n, 0.0, 0.0, 0.0, 2.0 * n, 0.0],
                [0.0, 0.0, 0.0, -2.0 * n, 0.0, 0.0],
                [0.0, 0.0, -n * n, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        # Only in-track and cross-track accelerations are available.
        B = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )
        ad, bd = HCWLQRController._discretize_zoh_series(A, B, self.design_dt_s)
        self._ad = ad
        self._bd = bd
        Q = np.diag(q)
        R = np.diag(r)
        self._k_gain = HCWLQRController._solve_discrete_lqr(ad, bd, Q, R, self.riccati_max_iter, self.riccati_tol)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        i0, i1 = self.ric_curv_state_slice
        j0, j1 = self.chief_eci_state_slice
        if belief.state.size < max(i1, j1):
            return Command.zero()

        x_curv = np.array(belief.state[i0:i1], dtype=float)
        chief_eci = np.array(belief.state[j0:j1], dtype=float)
        r_chief = chief_eci[0:3]
        v_chief = chief_eci[3:6]
        r0 = float(np.linalg.norm(r_chief))
        if r0 <= 0.0:
            return Command.zero()

        x_rect = ric_curv_to_rect(x_curv, r0_km=r0)
        x_effective = self.state_signs * x_rect
        a_cmd_ic_pre_limit = -self._k_gain @ x_effective
        a_cmd_ic = np.array(a_cmd_ic_pre_limit, dtype=float)
        a_cmd_ric = np.array([0.0, a_cmd_ic[0], a_cmd_ic[1]], dtype=float)
        nrm = float(np.linalg.norm(a_cmd_ric))
        limit_scale = 1.0
        if nrm > self.max_accel_km_s2 > 0.0:
            limit_scale = float(self.max_accel_km_s2 / nrm)
            a_cmd_ric *= limit_scale
            a_cmd_ic *= limit_scale

        c_ir = ric_dcm_ir_from_rv(r_chief, v_chief)
        a_cmd_eci = c_ir @ a_cmd_ric
        return Command(
            thrust_eci_km_s2=a_cmd_eci,
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "hcw_lqr_no_radial",
                "ric_curv_state_slice": [i0, i1],
                "chief_eci_state_slice": [j0, j1],
                "state_signs": self.state_signs.tolist(),
                "accel_ric_km_s2": a_cmd_ric.tolist(),
                "control_axes": ["I", "C"],
                "linear_feedback_debug": HCWLQRController._linear_feedback_debug_payload(
                    control_axes=["I", "C"],
                    k_gain=self._k_gain,
                    x_rect=x_rect,
                    x_effective=x_effective,
                    control_pre_limit=a_cmd_ic_pre_limit,
                    control_post_limit=a_cmd_ic,
                    limit_scale=limit_scale,
                    state_signs=self.state_signs,
                ),
            },
        )
