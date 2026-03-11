from __future__ import annotations

import numpy as np

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import eci_to_ecef


def ground_track_from_eci_history(
    r_eci_hist_km: np.ndarray,
    t_s: np.ndarray,
    jd_utc_start: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ECI position history into geocentric latitude/longitude ground track.

    Returns:
    - lat_deg: geocentric latitude in degrees [-90, 90]
    - lon_deg: longitude in degrees [-180, 180]
    - alt_km: radial altitude above spherical Earth [km]
    """
    r_hist = np.array(r_eci_hist_km, dtype=float)
    tt = np.array(t_s, dtype=float).reshape(-1)
    if r_hist.ndim != 2 or r_hist.shape[1] != 3:
        raise ValueError("r_eci_hist_km must be shape (N, 3).")
    if r_hist.shape[0] != tt.size:
        raise ValueError("t_s length must match r_eci_hist_km length.")

    n = tt.size
    lat = np.zeros(n)
    lon = np.zeros(n)
    alt = np.zeros(n)
    for k in range(n):
        r_ecef = eci_to_ecef(r_hist[k, :], tt[k], jd_utc_start=jd_utc_start)
        r_norm = float(np.linalg.norm(r_ecef))
        if r_norm <= 0.0:
            lat[k] = 0.0
            lon[k] = 0.0
            alt[k] = -EARTH_RADIUS_KM
            continue
        x, y, z = r_ecef
        lat[k] = np.degrees(np.arcsin(np.clip(z / r_norm, -1.0, 1.0)))
        lon[k] = np.degrees(np.arctan2(y, x))
        alt[k] = r_norm - EARTH_RADIUS_KM

    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon, alt


def split_ground_track_dateline(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    jump_threshold_deg: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert NaNs at large longitude jumps to avoid map lines crossing the plot.
    """
    lon = np.array(lon_deg, dtype=float).reshape(-1)
    lat = np.array(lat_deg, dtype=float).reshape(-1)
    if lon.size != lat.size:
        raise ValueError("lon_deg and lat_deg must have same length.")
    if lon.size == 0:
        return lon, lat
    out_lon = [float(lon[0])]
    out_lat = [float(lat[0])]
    thr = float(abs(jump_threshold_deg))
    for k in range(1, lon.size):
        if abs(float(lon[k] - lon[k - 1])) > thr:
            out_lon.append(np.nan)
            out_lat.append(np.nan)
        out_lon.append(float(lon[k]))
        out_lat.append(float(lat[k]))
    return np.array(out_lon, dtype=float), np.array(out_lat, dtype=float)
