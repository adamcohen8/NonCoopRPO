from dataclasses import dataclass
from typing import Optional

import numpy as np


def _unit(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n < eps:
        return np.zeros(3, dtype=float)
    return vec / n


def _slerp_dirs(a_hat: np.ndarray, b_hat: np.ndarray, frac: float) -> np.ndarray:
    a_u = _unit(a_hat)
    b_u = _unit(b_hat)
    if np.linalg.norm(a_u) < 1e-12:
        return b_u
    if np.linalg.norm(b_u) < 1e-12:
        return a_u
    c = float(np.clip(np.dot(a_u, b_u), -1.0, 1.0))
    if c > 0.999999:
        return _unit((1.0 - frac) * a_u + frac * b_u)
    if c < -0.999999:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(ref, a_u)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        ortho = _unit(np.cross(a_u, ref))
        theta = np.pi * frac
        return _unit(np.cos(theta) * a_u + np.sin(theta) * ortho)
    theta = np.arccos(c)
    s = np.sin(theta)
    w0 = np.sin((1.0 - frac) * theta) / s
    w1 = np.sin(frac * theta) / s
    return _unit(w0 * a_u + w1 * b_u)


@dataclass(frozen=True)
class AttitudeConstraint:
    enabled: bool = False
    inertia_body_kg_m2: Optional[np.ndarray] = None
    max_torque_nm: Optional[np.ndarray] = None

    def validate(self, label: str) -> None:
        if not self.enabled:
            return
        if self.inertia_body_kg_m2 is None or self.max_torque_nm is None:
            raise ValueError(f"{label}: inertia_body_kg_m2 and max_torque_nm are required when enabled.")
        j = np.asarray(self.inertia_body_kg_m2, dtype=float)
        tau = np.asarray(self.max_torque_nm, dtype=float)
        if j.shape != (3, 3):
            raise ValueError(f"{label}: inertia_body_kg_m2 must be a 3x3 matrix.")
        if tau.shape != (3,):
            raise ValueError(f"{label}: max_torque_nm must be a 3-vector.")
        if not np.all(np.isfinite(j)) or not np.all(np.isfinite(tau)):
            raise ValueError(f"{label}: inertia/torque must be finite.")
        if np.any(np.diag(j) <= 0.0):
            raise ValueError(f"{label}: inertia diagonal terms must be positive.")
        if np.any(tau <= 0.0):
            raise ValueError(f"{label}: max_torque_nm entries must be positive.")

    def max_angular_accel_rad_s2(self) -> float:
        if not self.enabled:
            return np.inf
        j = np.asarray(self.inertia_body_kg_m2, dtype=float)
        tau = np.asarray(self.max_torque_nm, dtype=float)
        alpha = tau / np.maximum(np.diag(j), 1e-12)
        return float(np.min(alpha))


@dataclass
class AttitudeRateState:
    thrust_axis: np.ndarray
    slew_rate_rad_s: float = 0.0


def apply_attitude_rate_constraint(
    desired_dir: np.ndarray,
    state: AttitudeRateState,
    dt_s: float,
    constraint: AttitudeConstraint,
) -> AttitudeRateState:
    desired_u = _unit(np.asarray(desired_dir, dtype=float))
    if np.linalg.norm(desired_u) < 1e-12:
        return AttitudeRateState(thrust_axis=state.thrust_axis.copy(), slew_rate_rad_s=state.slew_rate_rad_s)

    if not constraint.enabled:
        return AttitudeRateState(thrust_axis=desired_u, slew_rate_rad_s=0.0)

    current_u = _unit(np.asarray(state.thrust_axis, dtype=float))
    if np.linalg.norm(current_u) < 1e-12:
        current_u = desired_u
    c = float(np.clip(np.dot(current_u, desired_u), -1.0, 1.0))
    angle = float(np.arccos(c))
    if dt_s <= 0.0 or angle <= 1e-12:
        return AttitudeRateState(thrust_axis=desired_u, slew_rate_rad_s=0.0)

    alpha_max = constraint.max_angular_accel_rad_s2()
    omega_next = max(0.0, float(state.slew_rate_rad_s)) + alpha_max * dt_s
    max_angle = omega_next * dt_s
    if angle <= max_angle:
        achieved = desired_u
        omega_out = angle / dt_s
    else:
        frac = max_angle / max(angle, 1e-12)
        achieved = _slerp_dirs(current_u, desired_u, frac)
        omega_out = omega_next

    return AttitudeRateState(thrust_axis=achieved, slew_rate_rad_s=float(omega_out))
