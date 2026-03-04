from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


@dataclass
class QuaternionPDController(Controller):
    kp: float = 0.1
    kd: float = 0.05
    max_torque_nm: float = 0.05

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        # Expected state layout: [r(3),v(3),q(4),w(3)] at minimum
        if belief.state.size < 13:
            return Command.zero()
        q = belief.state[6:10]
        w = belief.state[10:13]
        q_err_vec = q[1:4]
        torque = -self.kp * q_err_vec - self.kd * w
        n = np.linalg.norm(torque)
        if n > self.max_torque_nm and n > 0.0:
            torque *= self.max_torque_nm / n
        return Command(thrust_eci_km_s2=np.zeros(3), torque_body_nm=torque, mode_flags={"mode": "quat_pd"})


@dataclass
class SmallAngleLQRController(Controller):
    inertia_kg_m2: np.ndarray
    wheel_axes_body: np.ndarray = field(default_factory=lambda: np.eye(3))
    wheel_torque_limits_nm: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05]))
    desired_attitude_quat_bn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    desired_rate_body_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_weights: np.ndarray = field(default_factory=lambda: np.array([8.0, 8.0, 8.0, 1.0, 1.0, 1.0]))
    r_weights: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    design_dt_s: float = 0.25
    riccati_max_iter: int = 300
    riccati_tol: float = 1e-9
    max_body_torque_nm: float | None = None
    _k_gain: np.ndarray = field(init=False, repr=False)
    _allocation: np.ndarray = field(init=False, repr=False)
    _wheel_axes_3xn: np.ndarray = field(init=False, repr=False)
    _wheel_limits_nm: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        I = np.array(self.inertia_kg_m2, dtype=float)
        if I.shape != (3, 3):
            raise ValueError("inertia_kg_m2 must be 3x3.")
        if np.linalg.det(I) == 0.0:
            raise ValueError("inertia_kg_m2 must be nonsingular.")

        axes = np.array(self.wheel_axes_body, dtype=float)
        if axes.ndim != 2:
            raise ValueError("wheel_axes_body must be a 2D array with shape (3,N) or (N,3).")
        if axes.shape[0] == 3:
            G = axes.copy()
        elif axes.shape[1] == 3:
            G = axes.T.copy()
        else:
            raise ValueError("wheel_axes_body must be shape (3,N) or (N,3).")
        if G.shape[1] < 3:
            raise ValueError("wheel_axes_body must include at least 3 wheel axes.")
        for k in range(G.shape[1]):
            n = float(np.linalg.norm(G[:, k]))
            if n == 0.0:
                raise ValueError("wheel_axes_body contains a zero vector.")
            G[:, k] /= n

        lim = np.array(self.wheel_torque_limits_nm, dtype=float).reshape(-1)
        if lim.size == 1:
            lim = np.full(G.shape[1], float(lim[0]))
        if lim.size != G.shape[1]:
            raise ValueError("wheel_torque_limits_nm must be scalar or length equal to number of wheels.")
        if np.any(lim <= 0.0):
            raise ValueError("wheel_torque_limits_nm must be positive.")

        qd = np.array(self.desired_attitude_quat_bn, dtype=float).reshape(-1)
        if qd.size != 4:
            raise ValueError("desired_attitude_quat_bn must be length-4.")
        wd = np.array(self.desired_rate_body_rad_s, dtype=float).reshape(-1)
        if wd.size != 3:
            raise ValueError("desired_rate_body_rad_s must be length-3.")

        q_weights = np.array(self.q_weights, dtype=float).reshape(-1)
        if q_weights.size == 1:
            q_weights = np.full(6, float(q_weights[0]))
        if q_weights.size != 6 or np.any(q_weights <= 0.0):
            raise ValueError("q_weights must be positive scalar or length-6 vector.")
        r_weights = np.array(self.r_weights, dtype=float).reshape(-1)
        if r_weights.size == 1:
            r_weights = np.full(G.shape[1], float(r_weights[0]))
        if r_weights.size != G.shape[1] or np.any(r_weights <= 0.0):
            raise ValueError("r_weights must be positive scalar or length-N vector.")
        if self.design_dt_s <= 0.0:
            raise ValueError("design_dt_s must be positive.")
        if self.riccati_max_iter <= 0:
            raise ValueError("riccati_max_iter must be positive.")
        if self.riccati_tol <= 0.0:
            raise ValueError("riccati_tol must be positive.")

        A = np.block(
            [
                [np.zeros((3, 3)), 0.5 * np.eye(3)],
                [np.zeros((3, 3)), np.zeros((3, 3))],
            ]
        )
        B = np.vstack((np.zeros((3, G.shape[1])), np.linalg.solve(I, G)))
        Ad = np.eye(6) + self.design_dt_s * A
        Bd = self.design_dt_s * B
        Q = np.diag(q_weights)
        R = np.diag(r_weights)
        P = Q.copy()
        for _ in range(self.riccati_max_iter):
            s = R + Bd.T @ P @ Bd
            K = np.linalg.solve(s, Bd.T @ P @ Ad)
            P_next = Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K + Q
            if np.max(np.abs(P_next - P)) < self.riccati_tol:
                P = P_next
                break
            P = P_next
        self._k_gain = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        self._allocation = np.linalg.pinv(G)
        self._wheel_axes_3xn = G
        self._wheel_limits_nm = lim

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        if belief.state.size < 13:
            return Command.zero()
        q = _normalize_quaternion(belief.state[6:10])
        q_des = _normalize_quaternion(self.desired_attitude_quat_bn)
        w = np.array(belief.state[10:13], dtype=float)
        w_des = np.array(self.desired_rate_body_rad_s, dtype=float)

        q_err = _quat_multiply(_quat_conjugate(q_des), q)
        if q_err[0] < 0.0:
            q_err *= -1.0
        x = np.hstack((q_err[1:4], (w - w_des)))

        wheel_torque_cmd = -self._k_gain @ x
        wheel_torque_cmd = np.clip(wheel_torque_cmd, -self._wheel_limits_nm, self._wheel_limits_nm)
        torque = self._wheel_axes_3xn @ wheel_torque_cmd

        if self.max_body_torque_nm is not None and self.max_body_torque_nm > 0.0:
            n = float(np.linalg.norm(torque))
            if n > self.max_body_torque_nm:
                torque *= self.max_body_torque_nm / n
                wheel_torque_cmd = self._allocation @ torque
                wheel_torque_cmd = np.clip(wheel_torque_cmd, -self._wheel_limits_nm, self._wheel_limits_nm)
                torque = self._wheel_axes_3xn @ wheel_torque_cmd

        return Command(
            thrust_eci_km_s2=np.zeros(3),
            torque_body_nm=torque,
            mode_flags={
                "mode": "lqr",
                "wheel_torque_cmd_nm": wheel_torque_cmd.tolist(),
            },
        )

    def set_target(self, desired_attitude_quat_bn: np.ndarray, desired_rate_body_rad_s: np.ndarray | None = None) -> None:
        q = np.array(desired_attitude_quat_bn, dtype=float).reshape(-1)
        if q.size != 4:
            raise ValueError("desired_attitude_quat_bn must be length-4.")
        self.desired_attitude_quat_bn = _normalize_quaternion(q)
        if desired_rate_body_rad_s is not None:
            w = np.array(desired_rate_body_rad_s, dtype=float).reshape(-1)
            if w.size != 3:
                raise ValueError("desired_rate_body_rad_s must be length-3.")
            self.desired_rate_body_rad_s = w


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    qv = np.array(q, dtype=float).reshape(-1)
    if qv.size != 4:
        raise ValueError("Quaternion must be length-4.")
    n = float(np.linalg.norm(qv))
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return qv / n


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    qn = _normalize_quaternion(q)
    return np.array([qn[0], -qn[1], -qn[2], -qn[3]])


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3 = _normalize_quaternion(q1)
    b0, b1, b2, b3 = _normalize_quaternion(q2)
    return np.array(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ]
    )
