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
    require_ground_visibility: bool = False
    ground_site: GroundSite | None = None


class AccessModel:
    def __init__(self, cfg: AccessConfig):
        self.cfg = cfg
        self._last_update_t_s = -np.inf

    def can_update(self, observer_eci_km: np.ndarray, target_eci_km: np.ndarray, t_s: float) -> bool:
        if t_s - self._last_update_t_s < self.cfg.update_cadence_s:
            return False

        los = target_eci_km - observer_eci_km
        rng = np.linalg.norm(los)
        if self.cfg.max_range_km is not None and rng > self.cfg.max_range_km:
            return False

        if self.cfg.fov_half_angle_rad is not None and rng > 0.0:
            boresight = observer_eci_km / max(np.linalg.norm(observer_eci_km), 1e-12)
            cosang = np.clip(np.dot(boresight, los / rng), -1.0, 1.0)
            if np.arccos(cosang) > self.cfg.fov_half_angle_rad:
                return False

        if self.cfg.require_ground_visibility:
            if self.cfg.ground_site is None:
                return False
            if not _ground_visible(observer_eci_km, target_eci_km):
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
