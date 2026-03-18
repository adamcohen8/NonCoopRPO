from __future__ import annotations

import numpy as np


WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_B_KM = WGS84_A_KM * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef_km(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    lat = np.deg2rad(float(lat_deg))
    lon = np.deg2rad(float(lon_deg))
    alt = float(alt_km)
    s = float(np.sin(lat))
    c = float(np.cos(lat))
    n = WGS84_A_KM / np.sqrt(max(1.0 - WGS84_E2 * s * s, 1e-15))
    x = (n + alt) * c * np.cos(lon)
    y = (n + alt) * c * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * s
    return np.array([x, y, z], dtype=float)


def ecef_to_geodetic_deg_km(r_ecef_km: np.ndarray) -> tuple[float, float, float]:
    x, y, z = np.array(r_ecef_km, dtype=float).reshape(3)
    lon = float(np.arctan2(y, x))
    p = float(np.hypot(x, y))
    if p <= 1e-12:
        lat = np.pi / 2.0 if z >= 0.0 else -np.pi / 2.0
        alt = abs(z) - WGS84_B_KM
        return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(alt)

    lat = float(np.arctan2(z, p * (1.0 - WGS84_E2)))
    for _ in range(8):
        s = float(np.sin(lat))
        n = WGS84_A_KM / np.sqrt(max(1.0 - WGS84_E2 * s * s, 1e-15))
        alt = p / max(float(np.cos(lat)), 1e-15) - n
        lat_next = float(np.arctan2(z, p * (1.0 - WGS84_E2 * n / max(n + alt, 1e-15))))
        if abs(lat_next - lat) <= 1e-13:
            lat = lat_next
            break
        lat = lat_next
    s = float(np.sin(lat))
    n = WGS84_A_KM / np.sqrt(max(1.0 - WGS84_E2 * s * s, 1e-15))
    alt = p / max(float(np.cos(lat)), 1e-15) - n
    return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(alt)


def ecef_to_enu_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = np.deg2rad(float(lat_deg))
    lon = np.deg2rad(float(lon_deg))
    slat = float(np.sin(lat))
    clat = float(np.cos(lat))
    slon = float(np.sin(lon))
    clon = float(np.cos(lon))
    return np.array(
        [
            [-slon, clon, 0.0],
            [-slat * clon, -slat * slon, clat],
            [clat * clon, clat * slon, slat],
        ],
        dtype=float,
    )


def enu_to_ecef_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    return ecef_to_enu_rotation(lat_deg, lon_deg).T
