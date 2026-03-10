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
class ReactionWheelPDController(Controller):
    wheel_axes_body: np.ndarray = field(default_factory=lambda: np.eye(3))
    wheel_torque_limits_nm: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05]))
    desired_attitude_quat_bn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    desired_rate_body_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    kp: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.25, 0.25]))
    kd: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 4.0]))
    max_body_torque_nm: float | None = None
    _allocation: np.ndarray = field(init=False, repr=False)
    _wheel_axes_3xn: np.ndarray = field(init=False, repr=False)
    _wheel_limits_nm: np.ndarray = field(init=False, repr=False)
    _kp: np.ndarray = field(init=False, repr=False)
    _kd: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
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
        if lim.size != G.shape[1] or np.any(lim <= 0.0):
            raise ValueError("wheel_torque_limits_nm must be positive scalar or length-N vector.")

        qd = np.array(self.desired_attitude_quat_bn, dtype=float).reshape(-1)
        if qd.size != 4:
            raise ValueError("desired_attitude_quat_bn must be length-4.")
        self.desired_attitude_quat_bn = _normalize_quaternion(qd)
        wd = np.array(self.desired_rate_body_rad_s, dtype=float).reshape(-1)
        if wd.size != 3:
            raise ValueError("desired_rate_body_rad_s must be length-3.")
        self.desired_rate_body_rad_s = wd

        kp = np.array(self.kp, dtype=float).reshape(-1)
        kd = np.array(self.kd, dtype=float).reshape(-1)
        if kp.size == 1:
            kp = np.full(3, float(kp[0]))
        if kd.size == 1:
            kd = np.full(3, float(kd[0]))
        if kp.size != 3 or kd.size != 3 or np.any(kp < 0.0) or np.any(kd < 0.0):
            raise ValueError("kp and kd must be non-negative scalar or length-3 vectors.")

        self._allocation = np.linalg.pinv(G)
        self._wheel_axes_3xn = G
        self._wheel_limits_nm = lim
        self._kp = kp
        self._kd = kd

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
        w_err = w - w_des
        torque_body_cmd = -(self._kp * q_err[1:4]) - (self._kd * w_err)

        wheel_torque_cmd = self._allocation @ torque_body_cmd
        wheel_torque_cmd = np.clip(wheel_torque_cmd, -self._wheel_limits_nm, self._wheel_limits_nm)
        torque = self._wheel_axes_3xn @ wheel_torque_cmd

        if self.max_body_torque_nm is not None and self.max_body_torque_nm > 0.0:
            n = float(np.linalg.norm(torque))
            if n > self.max_body_torque_nm:
                torque *= self.max_body_torque_nm / n
                wheel_torque_cmd = self._allocation @ torque
                wheel_torque_cmd = np.clip(wheel_torque_cmd, -self._wheel_limits_nm, self._wheel_limits_nm)
                torque = self._wheel_axes_3xn @ wheel_torque_cmd

        angle_deg = float(np.degrees(2.0 * np.arccos(np.clip(float(q_err[0]), -1.0, 1.0))))
        return Command(
            thrust_eci_km_s2=np.zeros(3),
            torque_body_nm=torque,
            mode_flags={
                "mode": "rw_pd",
                "attitude_error_deg": angle_deg,
                "wheel_torque_cmd_nm": wheel_torque_cmd.tolist(),
            },
        )


@dataclass
class ReactionWheelPIDController(ReactionWheelPDController):
    ki: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01, 0.01]))
    integral_limit: np.ndarray | None = None
    reset_integral_on_target_update: bool = False
    _ki: np.ndarray = field(init=False, repr=False)
    _integral_limit: np.ndarray | None = field(init=False, repr=False)
    _integral_error: np.ndarray = field(init=False, repr=False)
    _last_t_s: float | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        ki = np.array(self.ki, dtype=float).reshape(-1)
        if ki.size == 1:
            ki = np.full(3, float(ki[0]))
        if ki.size != 3 or np.any(ki < 0.0):
            raise ValueError("ki must be non-negative scalar or length-3 vector.")
        self._ki = ki

        if self.integral_limit is None:
            self._integral_limit = None
        else:
            lim = np.array(self.integral_limit, dtype=float).reshape(-1)
            if lim.size == 1:
                lim = np.full(3, float(lim[0]))
            if lim.size != 3 or np.any(lim <= 0.0):
                raise ValueError("integral_limit must be positive scalar or length-3 vector.")
            self._integral_limit = lim

        self._integral_error = np.zeros(3)
        self._last_t_s = None

    def set_target(self, desired_attitude_quat_bn: np.ndarray, desired_rate_body_rad_s: np.ndarray | None = None) -> None:
        super().set_target(desired_attitude_quat_bn, desired_rate_body_rad_s)
        if self.reset_integral_on_target_update:
            self.reset_integral()

    def reset_integral(self) -> None:
        self._integral_error = np.zeros(3)
        self._last_t_s = None

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
        w_err = w - w_des

        dt = 0.0
        if self._last_t_s is not None:
            dt = max(float(t_s) - float(self._last_t_s), 0.0)
        candidate_i = self._integral_error.copy()
        if dt > 0.0:
            candidate_i = candidate_i + q_err[1:4] * dt
            if self._integral_limit is not None:
                candidate_i = np.clip(candidate_i, -self._integral_limit, self._integral_limit)

        torque_body_cmd = -(self._kp * q_err[1:4]) - (self._kd * w_err) - (self._ki * candidate_i)
        wheel_torque_unsat = self._allocation @ torque_body_cmd
        saturated = bool(np.any(np.abs(wheel_torque_unsat) > self._wheel_limits_nm + 1e-12))

        if saturated and dt > 0.0:
            # Conditional integration anti-windup: skip I-term update while saturated.
            candidate_i = self._integral_error.copy()
            torque_body_cmd = -(self._kp * q_err[1:4]) - (self._kd * w_err) - (self._ki * candidate_i)
            wheel_torque_unsat = self._allocation @ torque_body_cmd

        self._integral_error = candidate_i
        self._last_t_s = float(t_s)

        wheel_torque_cmd = np.clip(wheel_torque_unsat, -self._wheel_limits_nm, self._wheel_limits_nm)
        torque = self._wheel_axes_3xn @ wheel_torque_cmd

        if self.max_body_torque_nm is not None and self.max_body_torque_nm > 0.0:
            n = float(np.linalg.norm(torque))
            if n > self.max_body_torque_nm:
                torque *= self.max_body_torque_nm / n
                wheel_torque_cmd = self._allocation @ torque
                wheel_torque_cmd = np.clip(wheel_torque_cmd, -self._wheel_limits_nm, self._wheel_limits_nm)
                torque = self._wheel_axes_3xn @ wheel_torque_cmd

        angle_deg = float(np.degrees(2.0 * np.arccos(np.clip(float(q_err[0]), -1.0, 1.0))))
        return Command(
            thrust_eci_km_s2=np.zeros(3),
            torque_body_nm=torque,
            mode_flags={
                "mode": "rw_pid",
                "attitude_error_deg": angle_deg,
                "wheel_torque_cmd_nm": wheel_torque_cmd.tolist(),
                "integral_error_body": self._integral_error.tolist(),
            },
        )


@dataclass
class SmallAngleLQRController(Controller):
    inertia_kg_m2: np.ndarray
    wheel_axes_body: np.ndarray = field(default_factory=lambda: np.eye(3))
    wheel_torque_limits_nm: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05]))
    desired_attitude_quat_bn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    desired_rate_body_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_weights: np.ndarray = field(default_factory=lambda: np.array([60.0, 60.0, 60.0, 8.0, 8.0, 8.0]))
    r_weights: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    design_dt_s: float = 0.25
    riccati_max_iter: int = 300
    riccati_tol: float = 1e-9
    max_body_torque_nm: float | None = None
    capture_enabled: bool = True
    capture_angle_deg: float = 25.0
    capture_kp: float = 0.35
    capture_kd: float = 3.5
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
        if self.capture_angle_deg < 0.0:
            raise ValueError("capture_angle_deg must be non-negative.")
        if self.capture_kp < 0.0 or self.capture_kd < 0.0:
            raise ValueError("capture_kp and capture_kd must be non-negative.")

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
        w_err = w - w_des
        x = np.hstack((q_err[1:4], w_err))
        angle_deg = np.degrees(2.0 * np.arccos(np.clip(float(q_err[0]), -1.0, 1.0)))

        if self.capture_enabled and angle_deg > self.capture_angle_deg:
            # Large-angle capture mode: nonlinear quaternion PD in body torque space.
            torque_body_cmd = -self.capture_kp * q_err[1:4] - self.capture_kd * w_err
            wheel_torque_cmd = self._allocation @ torque_body_cmd
            mode = "lqr_capture"
        else:
            wheel_torque_cmd = -self._k_gain @ x
            mode = "lqr_track"

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
                "mode": mode,
                "attitude_error_deg": float(angle_deg),
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

    @classmethod
    def robust_profile(
        cls,
        inertia_kg_m2: np.ndarray,
        wheel_axes_body: np.ndarray,
        wheel_torque_limits_nm: np.ndarray,
        design_dt_s: float,
    ) -> "SmallAngleLQRController":
        """Factory for a more robust capture+track profile across varied initial conditions."""
        return cls(
            inertia_kg_m2=inertia_kg_m2,
            wheel_axes_body=wheel_axes_body,
            wheel_torque_limits_nm=wheel_torque_limits_nm,
            q_weights=np.array([90.0, 90.0, 90.0, 10.0, 10.0, 10.0]),
            r_weights=np.array([1.0, 1.0, 1.0]),
            design_dt_s=design_dt_s,
            capture_enabled=True,
            capture_angle_deg=30.0,
            capture_kp=0.40,
            capture_kd=4.00,
        )


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
