from __future__ import annotations

import numpy as np


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


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
