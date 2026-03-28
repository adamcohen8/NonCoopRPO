from __future__ import annotations

import numpy as np

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM, SUN_RADIUS_KM
from sim.dynamics.orbit.epoch import AU_KM, resolve_sun_moon_positions


def _resolve_sun_position_eci_km(env: dict, t_s: float) -> np.ndarray:
    if "sun_pos_eci_km" in env:
        return np.asarray(env["sun_pos_eci_km"], dtype=float)
    try:
        sun, _ = resolve_sun_moon_positions(env, t_s)
        if np.linalg.norm(sun) > 0.0:
            return np.asarray(sun, dtype=float)
    except RuntimeError:
        pass
    sun_dir = np.asarray(env.get("sun_dir_eci", np.array([1.0, 0.0, 0.0], dtype=float)), dtype=float)
    n = float(np.linalg.norm(sun_dir))
    if n <= 0.0:
        return np.array([AU_KM, 0.0, 0.0], dtype=float)
    return (sun_dir / n) * AU_KM


def resolve_srp_geometry(r_sc_eci_km: np.ndarray, t_s: float, env: dict) -> dict[str, object]:
    r_sc = np.asarray(r_sc_eci_km, dtype=float).reshape(3)
    r_norm2 = float(np.dot(r_sc, r_sc))
    r_norm = float(np.sqrt(r_norm2)) if r_norm2 > 0.0 else 0.0

    r_sun = _resolve_sun_position_eci_km(env, t_s).reshape(3)
    sun_norm2 = float(np.dot(r_sun, r_sun))
    sun_norm = float(np.sqrt(sun_norm2)) if sun_norm2 > 0.0 else 0.0

    rho = r_sun - r_sc
    rho_norm2 = float(np.dot(rho, rho))
    rho_norm = float(np.sqrt(rho_norm2)) if rho_norm2 > 0.0 else 0.0
    if rho_norm > 0.0:
        sun_dir_sc_eci = rho / rho_norm
        distance_scale = float((AU_KM / rho_norm) ** 2)
    else:
        sun_dir_sc_eci = np.zeros(3, dtype=float)
        distance_scale = 1.0

    return {
        "r_sc_eci_km": r_sc,
        "r_sc_norm_km": r_norm,
        "sun_pos_eci_km": r_sun,
        "sun_pos_norm_km": sun_norm,
        "rho_sc_to_sun_km": rho,
        "rho_norm_km": rho_norm,
        "sun_dir_sc_eci": sun_dir_sc_eci,
        "distance_scale": distance_scale,
    }


def srp_shadow_factor(
    r_sc_eci_km: np.ndarray,
    t_s: float,
    env: dict,
    earth_radius_km: float = EARTH_RADIUS_KM,
    sun_radius_km: float = SUN_RADIUS_KM,
    srp_geometry: dict[str, object] | None = None,
) -> float:
    """
    Returns illumination factor in [0, 1] for SRP.

    - 1.0: full sunlight
    - 0.0: full umbra
    - (0,1): penumbra transition
    """
    model = str(env.get("srp_shadow_model", "conical")).lower()
    if model in ("none", "off", "disabled"):
        return 1.0

    geometry = resolve_srp_geometry(r_sc_eci_km, t_s, env) if srp_geometry is None else srp_geometry
    r_sc = np.asarray(geometry["r_sc_eci_km"], dtype=float)
    r_norm = float(geometry["r_sc_norm_km"])
    if r_norm <= earth_radius_km:
        return 0.0

    rho = np.asarray(geometry["rho_sc_to_sun_km"], dtype=float)
    rho_norm = float(geometry["rho_norm_km"])
    if rho_norm <= 0.0:
        return 1.0

    if model in ("cylindrical", "cylinder"):
        r_sun = np.asarray(geometry["sun_pos_eci_km"], dtype=float)
        sun_norm = max(float(geometry["sun_pos_norm_km"]), 1e-12)
        s_hat = r_sun / sun_norm
        r_sc_along_sun = float(np.dot(r_sc, s_hat))
        if r_sc_along_sun >= 0.0:
            return 1.0
        cross_track2 = max(0.0, float(np.dot(r_sc, r_sc)) - r_sc_along_sun * r_sc_along_sun)
        return 0.0 if cross_track2 < earth_radius_km * earth_radius_km else 1.0

    # Conical angular model (umbra + penumbra).
    # Apparent angular radii as seen from spacecraft.
    alpha = float(np.arcsin(np.clip(earth_radius_km / r_norm, -1.0, 1.0)))
    beta = float(np.arcsin(np.clip(sun_radius_km / rho_norm, -1.0, 1.0)))
    u_earth = -r_sc / r_norm
    u_sun = np.asarray(geometry["sun_dir_sc_eci"], dtype=float)
    gamma = float(np.arccos(np.clip(float(np.dot(u_earth, u_sun)), -1.0, 1.0)))

    if gamma >= alpha + beta:
        return 1.0

    # Complete occultation of Sun disk by Earth disk.
    if alpha > beta and gamma <= (alpha - beta):
        return 0.0

    # Rare annular-center case (Earth disk inside Sun disk).
    min_illum = 0.0
    if beta > alpha and gamma <= (beta - alpha):
        min_illum = max(0.0, 1.0 - (alpha * alpha) / (beta * beta))
        return float(min_illum)

    lo = abs(alpha - beta)
    hi = alpha + beta
    if hi <= lo:
        return 1.0
    f = (gamma - lo) / (hi - lo)
    f = float(np.clip(f, 0.0, 1.0))
    if beta > alpha:
        return float(np.clip(min_illum + (1.0 - min_illum) * f, 0.0, 1.0))
    return f
