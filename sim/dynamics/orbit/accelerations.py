from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.dynamics.orbit.environment import EARTH_J2, EARTH_RADIUS_KM, SOLAR_PRESSURE_N_M2
from sim.dynamics.orbit.frames import eci_to_ecef


@dataclass(frozen=True)
class OrbitContext:
    mu_km3_s2: float
    mass_kg: float
    area_m2: float = 1.0
    cd: float = 2.2
    cr: float = 1.2


def accel_two_body(r_eci_km: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    r = np.linalg.norm(r_eci_km)
    if r == 0.0:
        return np.zeros(3)
    return -mu_km3_s2 * r_eci_km / (r**3)


def accel_j2(r_eci_km: np.ndarray, mu_km3_s2: float, j2: float = EARTH_J2, re_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    x, y, z = r_eci_km
    r2 = float(np.dot(r_eci_km, r_eci_km))
    r = np.sqrt(r2)
    if r == 0.0:
        return np.zeros(3)
    z2 = z * z
    f = 1.5 * j2 * mu_km3_s2 * (re_km**2) / (r**5)
    g = 5.0 * z2 / r2
    return np.array([
        f * x * (g - 1.0),
        f * y * (g - 1.0),
        f * z * (g - 3.0),
    ])


def accel_drag(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    t_s: float,
    mass_kg: float,
    area_m2: float,
    cd: float,
    env: dict,
) -> np.ndarray:
    rho = float(env.get("density_kg_m3", 0.0))
    if rho <= 0.0 or mass_kg <= 0.0:
        return np.zeros(3)
    v_m_s = v_eci_km_s * 1e3
    v_norm = np.linalg.norm(v_m_s)
    if v_norm == 0.0:
        return np.zeros(3)
    a_m_s2 = -0.5 * rho * cd * area_m2 / mass_kg * v_norm * v_m_s
    return a_m_s2 / 1e3


def accel_srp(
    mass_kg: float,
    area_m2: float,
    cr: float,
    env: dict,
) -> np.ndarray:
    if mass_kg <= 0.0:
        return np.zeros(3)
    sun_dir_eci = np.array(env.get("sun_dir_eci", np.array([1.0, 0.0, 0.0])), dtype=float)
    n = np.linalg.norm(sun_dir_eci)
    if n == 0.0:
        return np.zeros(3)
    sun_dir_eci = sun_dir_eci / n
    force_n = SOLAR_PRESSURE_N_M2 * cr * area_m2
    a_m_s2 = force_n / mass_kg
    return -(a_m_s2 / 1e3) * sun_dir_eci


def accel_third_body(r_eci_km: np.ndarray, body_pos_eci_km: np.ndarray, body_mu_km3_s2: float) -> np.ndarray:
    rb = body_pos_eci_km - r_eci_km
    rb_norm = np.linalg.norm(rb)
    b_norm = np.linalg.norm(body_pos_eci_km)
    if rb_norm == 0.0 or b_norm == 0.0:
        return np.zeros(3)
    return body_mu_km3_s2 * (rb / (rb_norm**3) - body_pos_eci_km / (b_norm**3))


def default_density_model(r_eci_km: np.ndarray, t_s: float) -> float:
    r_ecef_km = eci_to_ecef(r_eci_km, t_s)
    alt_km = max(0.0, np.linalg.norm(r_ecef_km) - EARTH_RADIUS_KM)
    if alt_km > 180.0:
        return 0.0
    rho0 = 1.225
    h = 8.5
    return rho0 * np.exp(-(alt_km) / h)
