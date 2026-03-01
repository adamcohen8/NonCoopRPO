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
