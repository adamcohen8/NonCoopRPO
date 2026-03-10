from __future__ import annotations

import numpy as np

from sim.utils.quaternion import (
    normalize_quaternion,
    omega_matrix,
    quaternion_delta_from_body_rate,
    quaternion_multiply,
)


def rigid_body_derivatives(
    quat_bn: np.ndarray,
    omega_body_rad_s: np.ndarray,
    inertia_kg_m2: np.ndarray,
    torque_body_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_dot = 0.5 * omega_matrix(omega_body_rad_s) @ quat_bn
    Iw = inertia_kg_m2 @ omega_body_rad_s
    omega_dot = np.linalg.solve(inertia_kg_m2, torque_body_nm - np.cross(omega_body_rad_s, Iw))
    return q_dot, omega_dot


def propagate_attitude_euler(
    quat_bn: np.ndarray,
    omega_body_rad_s: np.ndarray,
    inertia_kg_m2: np.ndarray,
    torque_body_nm: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    q_dot, omega_dot = rigid_body_derivatives(quat_bn, omega_body_rad_s, inertia_kg_m2, torque_body_nm)
    q_next = normalize_quaternion(quat_bn + dt_s * q_dot)
    omega_next = omega_body_rad_s + dt_s * omega_dot
    return q_next, omega_next


def propagate_attitude_exponential_map(
    quat_bn: np.ndarray,
    omega_body_rad_s: np.ndarray,
    inertia_kg_m2: np.ndarray,
    torque_body_nm: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Integrate angular-rate dynamics with first-order step.
    _, omega_dot = rigid_body_derivatives(quat_bn, omega_body_rad_s, inertia_kg_m2, torque_body_nm)
    omega_next = omega_body_rad_s + dt_s * omega_dot

    # Use midpoint body rate to build quaternion delta via exponential map.
    omega_mid = omega_body_rad_s + 0.5 * dt_s * omega_dot
    dq = quaternion_delta_from_body_rate(omega_mid, dt_s)
    # q_dot uses Omega(w) @ q with the convention equivalent to q ⊗ [0, w],
    # so the finite update must right-multiply by dq.
    q_next = normalize_quaternion(quaternion_multiply(quat_bn, dq))
    return q_next, omega_next
