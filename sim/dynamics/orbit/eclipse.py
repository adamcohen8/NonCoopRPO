from __future__ import annotations

import numpy as np

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM, SUN_RADIUS_KM
from sim.dynamics.orbit.epoch import AU_KM, resolve_sun_moon_positions


def _resolve_sun_position_eci_km(env: dict, t_s: float) -> np.ndarray:
    if "sun_pos_eci_km" in env:
        return np.array(env["sun_pos_eci_km"], dtype=float)
    try:
        sun, _ = resolve_sun_moon_positions(env, t_s)
        if np.linalg.norm(sun) > 0.0:
            return np.array(sun, dtype=float)
    except RuntimeError:
        pass
    sun_dir = np.array(env.get("sun_dir_eci", np.array([1.0, 0.0, 0.0], dtype=float)), dtype=float)
    n = float(np.linalg.norm(sun_dir))
    if n <= 0.0:
        return np.array([AU_KM, 0.0, 0.0], dtype=float)
    return (sun_dir / n) * AU_KM


def srp_shadow_factor(
    r_sc_eci_km: np.ndarray,
    t_s: float,
    env: dict,
    earth_radius_km: float = EARTH_RADIUS_KM,
    sun_radius_km: float = SUN_RADIUS_KM,
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

    r_sc = np.array(r_sc_eci_km, dtype=float).reshape(3)
    r_norm = float(np.linalg.norm(r_sc))
    if r_norm <= earth_radius_km:
        return 0.0

    r_sun = _resolve_sun_position_eci_km(env, t_s)
    rho = r_sun - r_sc  # spacecraft -> sun
    rho_norm = float(np.linalg.norm(rho))
    if rho_norm <= 0.0:
        return 1.0

    if model in ("cylindrical", "cylinder"):
        s_hat = r_sun / max(float(np.linalg.norm(r_sun)), 1e-12)
        if float(np.dot(r_sc, s_hat)) >= 0.0:
            return 1.0
        cross_track = float(np.linalg.norm(r_sc - np.dot(r_sc, s_hat) * s_hat))
        return 0.0 if cross_track < earth_radius_km else 1.0

    # Conical angular model (umbra + penumbra).
    # Apparent angular radii as seen from spacecraft.
    alpha = float(np.arcsin(np.clip(earth_radius_km / r_norm, -1.0, 1.0)))
    beta = float(np.arcsin(np.clip(sun_radius_km / rho_norm, -1.0, 1.0)))
    u_earth = -r_sc / r_norm
    u_sun = rho / rho_norm
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
