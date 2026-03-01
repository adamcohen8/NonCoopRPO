from __future__ import annotations

import numpy as np

from sim.utils.integration import rk4_step


def two_body_derivative(x: np.ndarray, mu_km3_s2: float, accel_cmd_eci_km_s2: np.ndarray) -> np.ndarray:
    r = x[:3]
    v = x[3:]
    r_norm = np.linalg.norm(r)
    a_gravity = -mu_km3_s2 * r / (r_norm**3)
    a_total = a_gravity + accel_cmd_eci_km_s2
    return np.hstack((v, a_total))


def propagate_two_body_rk4(
    x_eci: np.ndarray,
    dt_s: float,
    mu_km3_s2: float,
    accel_cmd_eci_km_s2: np.ndarray,
) -> np.ndarray:
    return rk4_step(two_body_derivative, x_eci, dt_s, mu_km3_s2, accel_cmd_eci_km_s2)
