from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.dynamics.orbit.atmosphere import density_exponential
from sim.dynamics.orbit.eclipse import resolve_srp_geometry, srp_shadow_factor
from sim.dynamics.orbit.environment import EARTH_J2, EARTH_J3, EARTH_J4, EARTH_RADIUS_KM, EARTH_ROT_RATE_RAD_S, SOLAR_PRESSURE_N_M2

_OMEGA_EARTH_RAD_S = np.array([0.0, 0.0, EARTH_ROT_RATE_RAD_S], dtype=float)

@dataclass(frozen=True)
class OrbitContext:
    mu_km3_s2: float
    mass_kg: float
    area_m2: float = 1.0
    cd: float = 2.2
    cr: float = 1.2


def accel_two_body(r_eci_km: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    r2 = float(np.dot(r_eci_km, r_eci_km))
    if r2 == 0.0:
        return np.zeros(3)
    r = float(np.sqrt(r2))
    return (-mu_km3_s2 / (r * r2)) * r_eci_km


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
    v_atm_eci_km_s = np.array(
        [
            -EARTH_ROT_RATE_RAD_S * float(r_eci_km[1]),
            EARTH_ROT_RATE_RAD_S * float(r_eci_km[0]),
            0.0,
        ],
        dtype=float,
    )
    v_rel_eci_km_s = v_eci_km_s - v_atm_eci_km_s
    v_rel_m_s = v_rel_eci_km_s * 1e3
    v_norm2 = float(np.dot(v_rel_m_s, v_rel_m_s))
    if v_norm2 == 0.0:
        return np.zeros(3)
    v_norm = float(np.sqrt(v_norm2))
    a_m_s2 = -0.5 * rho * cd * area_eff_m2 / mass_kg * v_norm * v_rel_m_s
    return a_m_s2 / 1e3


def accel_srp(
    r_eci_km: np.ndarray,
    mass_kg: float,
    area_m2: float,
    cr: float,
    t_s: float,
    env: dict,
) -> np.ndarray:
    if mass_kg <= 0.0:
        return np.zeros(3)
    area_eff_m2 = float(env.get("srp_area_m2", area_m2))
    if area_eff_m2 <= 0.0:
        return np.zeros(3)
    srp_geometry = env.get("srp_geometry")
    if not isinstance(srp_geometry, dict):
        srp_geometry = resolve_srp_geometry(r_eci_km, t_s, env)

    sun_dir_eci = env.get("srp_sun_dir_eci")
    if sun_dir_eci is None:
        sun_dir_eci = srp_geometry["sun_dir_sc_eci"]
    sun_dir_eci = np.asarray(sun_dir_eci, dtype=float).reshape(3)

    shadow = env.get("srp_shadow_factor")
    if shadow is None:
        shadow = srp_shadow_factor(r_sc_eci_km=r_eci_km, t_s=t_s, env=env, srp_geometry=srp_geometry)
    shadow = float(shadow)
    if shadow <= 0.0:
        return np.zeros(3)

    n2 = float(np.dot(sun_dir_eci, sun_dir_eci))
    if n2 <= 0.0:
        return np.zeros(3)
    if abs(n2 - 1.0) > 1e-12:
        sun_dir_eci = sun_dir_eci / float(np.sqrt(n2))

    distance_scale = float(env.get("srp_distance_scale", srp_geometry.get("distance_scale", 1.0)))
    force_n = SOLAR_PRESSURE_N_M2 * distance_scale * cr * area_eff_m2
    a_m_s2 = force_n / mass_kg
    return -(a_m_s2 / 1e3) * shadow * sun_dir_eci


def accel_third_body(r_eci_km: np.ndarray, body_pos_eci_km: np.ndarray, body_mu_km3_s2: float) -> np.ndarray:
    rb = body_pos_eci_km - r_eci_km
    rb_norm2 = float(np.dot(rb, rb))
    b_norm2 = float(np.dot(body_pos_eci_km, body_pos_eci_km))
    rb_norm = float(np.sqrt(rb_norm2)) if rb_norm2 > 0.0 else 0.0
    b_norm = float(np.sqrt(b_norm2)) if b_norm2 > 0.0 else 0.0
    if rb_norm == 0.0 or b_norm == 0.0:
        return np.zeros(3)
    return body_mu_km3_s2 * (rb / (rb_norm**3) - body_pos_eci_km / (b_norm**3))


def default_density_model(r_eci_km: np.ndarray, t_s: float) -> float:
    return density_exponential(r_eci_km, t_s)
