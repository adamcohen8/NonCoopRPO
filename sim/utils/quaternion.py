from __future__ import annotations

import numpy as np


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    x = np.array(q, dtype=float).reshape(-1)
    if x.size != 4 or not np.all(np.isfinite(x)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    n = float(np.linalg.norm(x))
    if n <= 0.0 or not np.isfinite(n):
        return np.array([1.0, 0.0, 0.0, 0.0])
    return x / n


def omega_matrix(w_body_rad_s: np.ndarray) -> np.ndarray:
    wx, wy, wz = w_body_rad_s
    return np.array(
        [
            [0.0, -wx, -wy, -wz],
            [wx, 0.0, wz, -wy],
            [wy, -wz, 0.0, wx],
            [wz, wy, -wx, 0.0],
        ]
    )


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3 = normalize_quaternion(np.array(q1, dtype=float))
    b0, b1, b2, b3 = normalize_quaternion(np.array(q2, dtype=float))
    return np.array(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ],
        dtype=float,
    )


def quaternion_delta_from_body_rate(omega_body_rad_s: np.ndarray, dt_s: float) -> np.ndarray:
    w = np.array(omega_body_rad_s, dtype=float).reshape(3)
    if not np.all(np.isfinite(w)) or not np.isfinite(float(dt_s)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    w_norm = float(np.linalg.norm(w))
    if w_norm <= 1e-15 or dt_s == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    half_theta = float(0.5 * w_norm * dt_s)
    if not np.isfinite(half_theta):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    # Keep trig arguments bounded to avoid invalid/overflow warnings for huge rates.
    half_theta = float(np.remainder(half_theta, 2.0 * np.pi))
    axis = w / w_norm
    s = float(np.sin(half_theta))
    c = float(np.cos(half_theta))
    return normalize_quaternion(np.array([c, axis[0] * s, axis[1] * s, axis[2] * s], dtype=float))


def quaternion_to_dcm_bn(q_bn: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = normalize_quaternion(q_bn)
    return np.array(
        [
            [1.0 - 2.0 * (q2**2 + q3**2), 2.0 * (q1 * q2 + q0 * q3), 2.0 * (q1 * q3 - q0 * q2)],
            [2.0 * (q1 * q2 - q0 * q3), 1.0 - 2.0 * (q1**2 + q3**2), 2.0 * (q2 * q3 + q0 * q1)],
            [2.0 * (q1 * q3 + q0 * q2), 2.0 * (q2 * q3 - q0 * q1), 1.0 - 2.0 * (q1**2 + q2**2)],
        ]
    )


def dcm_to_quaternion_bn(c_bn: np.ndarray) -> np.ndarray:
    if c_bn.shape != (3, 3):
        raise ValueError("c_bn must be a 3x3 matrix.")

    # The closed-form extraction below expects the transpose convention relative
    # to quaternion_to_dcm_bn, so solve on C_nb and return q_bn.
    m = c_bn.T
    tr = float(np.trace(m))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q0 = 0.25 * s
        q1 = (m[2, 1] - m[1, 2]) / s
        q2 = (m[0, 2] - m[2, 0]) / s
        q3 = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q0 = (m[2, 1] - m[1, 2]) / s
        q1 = 0.25 * s
        q2 = (m[0, 1] + m[1, 0]) / s
        q3 = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q0 = (m[0, 2] - m[2, 0]) / s
        q1 = (m[0, 1] + m[1, 0]) / s
        q2 = 0.25 * s
        q3 = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q0 = (m[1, 0] - m[0, 1]) / s
        q1 = (m[0, 2] + m[2, 0]) / s
        q2 = (m[1, 2] + m[2, 1]) / s
        q3 = 0.25 * s

    return normalize_quaternion(np.array([q0, q1, q2, q3], dtype=float))
