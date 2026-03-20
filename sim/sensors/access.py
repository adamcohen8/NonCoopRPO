from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroundSite:
    lat_rad: float
    lon_rad: float
    min_elevation_rad: float = 0.0


@dataclass(frozen=True)
class AccessConfig:
    update_cadence_s: float = 1.0
    max_range_km: float | None = None
    fov_half_angle_rad: float | None = None
    solid_angle_sr: float | None = None
    require_ground_visibility: bool = False
    ground_site: GroundSite | None = None


class AccessModel:
    def __init__(self, cfg: AccessConfig):
        self.cfg = cfg
        self._last_update_t_s = -np.inf

    def evaluate(
        self,
        observer_eci_km: np.ndarray,
        target_eci_km: np.ndarray,
        t_s: float,
        *,
        boresight_eci: np.ndarray | None = None,
    ) -> tuple[bool, str]:
        if t_s - self._last_update_t_s < self.cfg.update_cadence_s:
            return False, "cadence"

        los = target_eci_km - observer_eci_km
        rng = np.linalg.norm(los)
        if self.cfg.max_range_km is not None and rng > self.cfg.max_range_km:
            return False, "range"

        fov_half_angle_rad = self.cfg.fov_half_angle_rad
        if fov_half_angle_rad is None:
            fov_half_angle_rad = _solid_angle_to_half_angle_rad(self.cfg.solid_angle_sr)
        if fov_half_angle_rad is not None and rng > 0.0:
            if boresight_eci is None:
                boresight = observer_eci_km / max(np.linalg.norm(observer_eci_km), 1e-12)
            else:
                boresight = np.array(boresight_eci, dtype=float).reshape(3)
                bn = float(np.linalg.norm(boresight))
                if bn <= 0.0:
                    return False, "boresight"
                boresight = boresight / bn
            cosang = np.clip(np.dot(boresight, los / rng), -1.0, 1.0)
            if np.arccos(cosang) > fov_half_angle_rad:
                return False, "solid_angle"

        if self.cfg.require_ground_visibility:
            if self.cfg.ground_site is None:
                return False, "ground_site_missing"
            if not _ground_visible(observer_eci_km, target_eci_km):
                return False, "ground_visibility"

        return True, "ok"

    def can_update(
        self,
        observer_eci_km: np.ndarray,
        target_eci_km: np.ndarray,
        t_s: float,
        *,
        boresight_eci: np.ndarray | None = None,
    ) -> bool:
        allowed, _ = self.evaluate(observer_eci_km, target_eci_km, t_s, boresight_eci=boresight_eci)
        if not allowed:
            return False
        self._last_update_t_s = t_s
        return True


def _ground_visible(observer_eci_km: np.ndarray, target_eci_km: np.ndarray) -> bool:
    # Simple Earth occultation check: LOS not intersecting Earth sphere.
    ro = observer_eci_km
    rt = target_eci_km
    d = rt - ro
    denom = np.dot(d, d)
    if denom <= 0.0:
        return True
    tau = -np.dot(ro, d) / denom
    tau = np.clip(tau, 0.0, 1.0)
    closest = ro + tau * d
    return np.linalg.norm(closest) > 6378.137


def _solid_angle_to_half_angle_rad(solid_angle_sr: float | None) -> float | None:
    if solid_angle_sr is None:
        return None
    omega = float(solid_angle_sr)
    if not np.isfinite(omega) or omega <= 0.0:
        return 0.0
    if omega >= (4.0 * np.pi - 1e-12):
        return None
    cos_half = float(np.clip(1.0 - omega / (2.0 * np.pi), -1.0, 1.0))
    return float(np.arccos(cos_half))
