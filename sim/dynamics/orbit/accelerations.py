from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.dynamics.orbit.atmosphere import density_exponential
from sim.dynamics.orbit.environment import EARTH_J2, EARTH_J3, EARTH_J4, EARTH_RADIUS_KM, EARTH_ROT_RATE_RAD_S, SOLAR_PRESSURE_N_M2


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


def accel_j3(r_eci_km: np.ndarray, mu_km3_s2: float, j3: float = EARTH_J3, re_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    """
    Zonal J3 perturbation acceleration in ECI (km/s^2).

    Uses the standard spherical-harmonic zonal expansion for n=3.
    """
    x, y, z = r_eci_km
    r2 = float(np.dot(r_eci_km, r_eci_km))
    r = np.sqrt(r2)
    if r == 0.0:
        return np.zeros(3)
    s = z / r
    s2 = s * s
    s4 = s2 * s2

    # a_xy = mu*J3*Re^3 * x(y) / r^6 * [ (7/2) s (5 s^2 - 3) ]
    axy_scale = mu_km3_s2 * j3 * (re_km**3) / (r**6)
    axy_factor = 3.5 * s * (5.0 * s2 - 3.0)

    # a_z = mu*J3*Re^3 / r^5 * [ (1/2) (35 s^4 - 30 s^2 + 3) ]
    az_scale = mu_km3_s2 * j3 * (re_km**3) / (r**5)
    az_factor = 0.5 * (35.0 * s4 - 30.0 * s2 + 3.0)

    return np.array(
        [
            axy_scale * x * axy_factor,
            axy_scale * y * axy_factor,
            az_scale * az_factor,
        ]
    )


def accel_j4(r_eci_km: np.ndarray, mu_km3_s2: float, j4: float = EARTH_J4, re_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    """
    Zonal J4 perturbation acceleration in ECI (km/s^2).

    Uses the standard spherical-harmonic zonal expansion for n=4.
    """
    x, y, z = r_eci_km
    r2 = float(np.dot(r_eci_km, r_eci_km))
    r = np.sqrt(r2)
    if r == 0.0:
        return np.zeros(3)
    s = z / r
    s2 = s * s
    s4 = s2 * s2

    # a_xy = mu*J4*Re^4 * x(y) / r^7 * [ (5/8) (63 s^4 - 42 s^2 + 3) ]
    axy_scale = mu_km3_s2 * j4 * (re_km**4) / (r**7)
    axy_factor = 0.625 * (63.0 * s4 - 42.0 * s2 + 3.0)

    # a_z = mu*J4*Re^4 / r^6 * [ (5/8) s (63 s^4 - 70 s^2 + 15) ]
    az_scale = mu_km3_s2 * j4 * (re_km**4) / (r**6)
    az_factor = 0.625 * s * (63.0 * s4 - 70.0 * s2 + 15.0)

    return np.array(
        [
            axy_scale * x * axy_factor,
            axy_scale * y * axy_factor,
            az_scale * az_factor,
        ]
    )


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
    area_eff_m2 = float(env.get("drag_area_m2", area_m2))
    if area_eff_m2 <= 0.0:
        return np.zeros(3)
    # Atmosphere assumed corotating with Earth about inertial z-axis.
    omega_earth_rad_s = np.array([0.0, 0.0, EARTH_ROT_RATE_RAD_S], dtype=float)
    v_atm_eci_km_s = np.cross(omega_earth_rad_s, r_eci_km)
    v_rel_eci_km_s = v_eci_km_s - v_atm_eci_km_s
    v_rel_m_s = v_rel_eci_km_s * 1e3
    v_norm = np.linalg.norm(v_rel_m_s)
    if v_norm == 0.0:
        return np.zeros(3)
    a_m_s2 = -0.5 * rho * cd * area_eff_m2 / mass_kg * v_norm * v_rel_m_s
    return a_m_s2 / 1e3


def accel_srp(
    mass_kg: float,
    area_m2: float,
    cr: float,
    env: dict,
) -> np.ndarray:
    # NOTE: Eclipse/shadowing is not currently modeled here. SRP is applied
    # continuously based on sun_dir_eci when area > 0.
    if mass_kg <= 0.0:
        return np.zeros(3)
    area_eff_m2 = float(env.get("srp_area_m2", area_m2))
    if area_eff_m2 <= 0.0:
        return np.zeros(3)
    sun_dir_eci = np.array(env.get("sun_dir_eci", np.array([1.0, 0.0, 0.0])), dtype=float)
    n = np.linalg.norm(sun_dir_eci)
    if n == 0.0:
        return np.zeros(3)
    sun_dir_eci = sun_dir_eci / n
    force_n = SOLAR_PRESSURE_N_M2 * cr * area_eff_m2
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
    return density_exponential(r_eci_km, t_s)
