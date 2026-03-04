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
    _k_gain: np.ndarray = field(init=False, repr=False)

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
        x_rect = self.state_signs * x_rect
        a_cmd_ric = -self._k_gain @ x_rect
        nrm = float(np.linalg.norm(a_cmd_ric))
        if nrm > self.max_accel_km_s2 > 0.0:
            a_cmd_ric *= self.max_accel_km_s2 / nrm

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
