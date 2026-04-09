from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.frames import ric_curv_to_rect, ric_dcm_ir_from_rv


@dataclass
class HCWLQRController(Controller):
    mean_motion_rad_s: float
    max_accel_km_s2: float
    design_dt_s: float = 10.0
    ric_curv_state_slice: tuple[int, int] = (0, 6)
    chief_eci_state_slice: tuple[int, int] = (6, 12)
    state_signs: np.ndarray = field(default_factory=lambda: np.ones(6))
    q_weights: np.ndarray = field(default_factory=lambda: np.array([8.66, 8.66, 8.66, 1.33, 1.33, 1.33]) * 1e3)
    r_weights: np.ndarray = field(default_factory=lambda: np.ones(3) * 1.94e13)
    riccati_max_iter: int = 500
    riccati_tol: float = 1e-8
    _ad: np.ndarray = field(init=False, repr=False)
    _bd: np.ndarray = field(init=False, repr=False)
    _k_gain: np.ndarray = field(init=False, repr=False)

    @staticmethod
    def _control_law_label(state_signs: np.ndarray) -> str:
        signs = np.array(state_signs, dtype=float).reshape(-1)
        if signs.size > 0 and np.all(signs < 0.0):
            return "Kx"
        if signs.size > 0 and np.all(signs > 0.0):
            return "-Kx"
        return "-K(state_signs .* x)"

    @staticmethod
    def _linear_feedback_debug_payload(
        *,
        control_axes: list[str],
        k_gain: np.ndarray,
        x_rect: np.ndarray,
        x_effective: np.ndarray,
        control_pre_limit: np.ndarray,
        control_post_limit: np.ndarray,
        limit_scale: float,
        state_signs: np.ndarray,
    ) -> dict[str, object]:
        term_contributions_pre_limit = -(np.array(k_gain, dtype=float) * np.array(x_effective, dtype=float)[None, :])
        term_contributions_post_limit = float(limit_scale) * term_contributions_pre_limit
        return {
            "law_label": HCWLQRController._control_law_label(state_signs),
            "frame": "RIC",
            "units": "km_s2",
            "control_axes": list(control_axes),
            "state_labels": ["R", "I", "C", "dR", "dI", "dC"],
            "gain_matrix": np.array(k_gain, dtype=float).tolist(),
            "state_rect": np.array(x_rect, dtype=float).tolist(),
            "state_effective": np.array(x_effective, dtype=float).tolist(),
            "control_pre_limit": np.array(control_pre_limit, dtype=float).tolist(),
            "control_post_limit": np.array(control_post_limit, dtype=float).tolist(),
            "limit_scale": float(limit_scale),
            "term_contributions_pre_limit": term_contributions_pre_limit.tolist(),
            "term_contributions_post_limit": term_contributions_post_limit.tolist(),
        }

    @staticmethod
    def _complex_pairs(values: np.ndarray) -> list[dict[str, float]]:
        arr = np.array(values, dtype=complex).reshape(-1)
        ordered = sorted(arr.tolist(), key=lambda z: (float(np.real(z)), float(np.imag(z))))
        return [{"real": float(np.real(z)), "imag": float(np.imag(z))} for z in ordered]

    @staticmethod
    def _position_channel_zeros(ad: np.ndarray, bd_col: np.ndarray, c_row: np.ndarray) -> np.ndarray:
        ad = np.array(ad, dtype=float)
        b = np.array(bd_col, dtype=float).reshape(-1)
        c = np.array(c_row, dtype=float).reshape(-1)
        n = int(ad.shape[0])
        if ad.shape != (n, n) or b.size != n or c.size != n:
            return np.array([], dtype=complex)

        poles = np.linalg.eigvals(ad)
        den = np.poly(poles)
        eye = np.eye(n, dtype=complex)
        sample_points: list[complex] = []
        for radius in (0.2, 0.45, 0.7, 0.9, 1.15, 1.4, 1.7, 2.1):
            sample_points.extend(
                [
                    complex(radius, 0.0),
                    complex(-radius, 0.0),
                    complex(0.0, radius),
                    complex(0.0, -radius),
                    complex(radius / np.sqrt(2.0), radius / np.sqrt(2.0)),
                    complex(-radius / np.sqrt(2.0), radius / np.sqrt(2.0)),
                ]
            )

        samples: list[tuple[complex, complex]] = []
        for z in sample_points:
            if min(abs(z - pole) for pole in poles) < 1e-6:
                continue
            try:
                gain = complex(c @ np.linalg.solve(z * eye - ad, b))
            except np.linalg.LinAlgError:
                continue
            num_val = gain * np.polyval(den, z)
            samples.append((z, num_val))
            if len(samples) >= n:
                break
        if len(samples) < n:
            return np.array([], dtype=complex)

        vand = np.array([[z ** (n - 1 - idx) for idx in range(n)] for z, _ in samples], dtype=complex)
        vals = np.array([val for _, val in samples], dtype=complex)
        try:
            coeff = np.linalg.solve(vand, vals)
        except np.linalg.LinAlgError:
            return np.array([], dtype=complex)

        scale = max(1.0, float(np.max(np.abs(coeff))))
        coeff = coeff[np.argmax(np.abs(coeff) > 1e-10 * scale) :] if np.any(np.abs(coeff) > 1e-10 * scale) else coeff[-1:]
        if coeff.size <= 1:
            return np.array([], dtype=complex)
        return np.roots(coeff)

    @staticmethod
    def _position_output_indices() -> list[int]:
        return [0, 1, 2]

    @staticmethod
    def _control_axes() -> list[str]:
        return ["R", "I", "C"]

    def linear_system_summary(self) -> dict[str, object]:
        closed_loop = self._ad - self._bd @ self._k_gain
        zeros = []
        for axis_idx, axis_label in enumerate(self._control_axes()):
            out_idx = self._position_output_indices()[axis_idx]
            c_row = np.zeros(self._ad.shape[0], dtype=float)
            c_row[out_idx] = 1.0
            z = self._position_channel_zeros(self._ad, self._bd[:, axis_idx], c_row)
            zeros.append({"axis": axis_label, "zeros": self._complex_pairs(z)})
        return {
            "system_type": "discrete_state_feedback",
            "sample_time_s": float(self.design_dt_s),
            "law_label": self._control_law_label(self.state_signs),
            "control_axes": self._control_axes(),
            "open_loop_poles": self._complex_pairs(np.linalg.eigvals(self._ad)),
            "closed_loop_poles": self._complex_pairs(np.linalg.eigvals(closed_loop)),
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
            r = np.full(3, float(r[0]))
        if r.size != 3 or np.any(r <= 0.0):
            raise ValueError("r_weights must be positive scalar or length-3 vector.")

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
        B = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        ad, bd = self._discretize_zoh_series(A, B, self.design_dt_s)
        self._ad = ad
        self._bd = bd
        Q = np.diag(q)
        R = np.diag(r)
        self._k_gain = self._solve_discrete_lqr(ad, bd, Q, R, self.riccati_max_iter, self.riccati_tol)

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

        # HCW/LQR operates on rectangular RIC relative states.
        x_rect = ric_curv_to_rect(x_curv, r0_km=r0)
        x_effective = self.state_signs * x_rect
        a_cmd_ric_pre_limit = -self._k_gain @ x_effective
        a_cmd_ric = np.array(a_cmd_ric_pre_limit, dtype=float)
        nrm = float(np.linalg.norm(a_cmd_ric_pre_limit))
        limit_scale = 1.0
        if nrm > self.max_accel_km_s2 > 0.0:
            limit_scale = float(self.max_accel_km_s2 / nrm)
            a_cmd_ric *= limit_scale

        c_ir = ric_dcm_ir_from_rv(r_chief, v_chief)
        a_cmd_eci = c_ir @ a_cmd_ric
        return Command(
            thrust_eci_km_s2=a_cmd_eci,
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "hcw_lqr",
                "ric_curv_state_slice": [i0, i1],
                "chief_eci_state_slice": [j0, j1],
                "state_signs": self.state_signs.tolist(),
                "accel_ric_km_s2": a_cmd_ric.tolist(),
                "linear_feedback_debug": self._linear_feedback_debug_payload(
                    control_axes=["R", "I", "C"],
                    k_gain=self._k_gain,
                    x_rect=x_rect,
                    x_effective=x_effective,
                    control_pre_limit=a_cmd_ric_pre_limit,
                    control_post_limit=a_cmd_ric,
                    limit_scale=limit_scale,
                    state_signs=self.state_signs,
                ),
            },
        )

    @staticmethod
    def _discretize_zoh_series(A: np.ndarray, B: np.ndarray, dt: float, terms: int = 30) -> tuple[np.ndarray, np.ndarray]:
        n = A.shape[0]
        I = np.eye(n)

        ad = I.copy()
        Ak = I.copy()
        for k in range(1, terms + 1):
            Ak = Ak @ (A * dt / float(k))
            ad = ad + Ak

        bd = np.zeros_like(B)
        Ak = I.copy()
        for k in range(0, terms):
            coeff = dt / float(k + 1)
            bd = bd + coeff * (Ak @ B)
            Ak = Ak @ (A * dt / float(k + 1))
        return ad, bd

    @staticmethod
    def _solve_discrete_lqr(
        Ad: np.ndarray,
        Bd: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        max_iter: int,
        tol: float,
    ) -> np.ndarray:
        P = Q.copy()
        K = np.zeros((Bd.shape[1], Ad.shape[0]))
        for _ in range(max_iter):
            s = R + Bd.T @ P @ Bd
            K = np.linalg.solve(s, Bd.T @ P @ Ad)
            Pn = Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K + Q
            if np.max(np.abs(Pn - P)) < tol:
                P = Pn
                break
            P = Pn
        return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
