from __future__ import annotations

import numpy as np

from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S
from sim.dynamics.orbit.epoch import gmst_angle_rad_from_jd


def eci_to_ecef_rotation(t_s: float, jd_utc_start: float | None = None) -> np.ndarray:
    if jd_utc_start is None:
        theta = EARTH_ROT_RATE_RAD_S * t_s
    else:
        theta = gmst_angle_rad_from_jd(float(jd_utc_start) + float(t_s) / 86400.0)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def eci_to_ecef(r_eci_km: np.ndarray, t_s: float, jd_utc_start: float | None = None) -> np.ndarray:
    return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start) @ r_eci_km


def ecef_to_eci(r_ecef_km: np.ndarray, t_s: float, jd_utc_start: float | None = None) -> np.ndarray:
    return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start).T @ r_ecef_km
