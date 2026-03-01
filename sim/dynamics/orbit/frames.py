from __future__ import annotations

import numpy as np

from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S


def eci_to_ecef_rotation(t_s: float) -> np.ndarray:
    theta = EARTH_ROT_RATE_RAD_S * t_s
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def eci_to_ecef(r_eci_km: np.ndarray, t_s: float) -> np.ndarray:
    return eci_to_ecef_rotation(t_s) @ r_eci_km


def ecef_to_eci(r_ecef_km: np.ndarray, t_s: float) -> np.ndarray:
    return eci_to_ecef_rotation(t_s).T @ r_ecef_km
