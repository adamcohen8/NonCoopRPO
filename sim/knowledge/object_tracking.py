from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.models import Measurement, StateBelief, StateTruth
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.sensors.access import AccessConfig, AccessModel


def _line_of_sight_clear(observer_eci_km: np.ndarray, target_eci_km: np.ndarray) -> bool:
    ro = np.array(observer_eci_km, dtype=float)
    rt = np.array(target_eci_km, dtype=float)
    d = rt - ro
    denom = float(np.dot(d, d))
    if denom <= 0.0:
        return True
    tau = float(np.clip(-np.dot(ro, d) / denom, 0.0, 1.0))
    closest = ro + tau * d
    return float(np.linalg.norm(closest)) > EARTH_RADIUS_KM


@dataclass(frozen=True)
class KnowledgeConditionConfig:
    refresh_rate_s: float = 10.0
    max_range_km: float | None = None
    fov_half_angle_rad: float | None = None
    require_line_of_sight: bool = False
    dropout_prob: float = 0.0


@dataclass(frozen=True)
class KnowledgeNoiseConfig:
    pos_sigma_km: np.ndarray = field(default_factory=lambda: np.array([1e-3, 1e-3, 1e-3]))
    vel_sigma_km_s: np.ndarray = field(default_factory=lambda: np.array([1e-5, 1e-5, 1e-5]))
    pos_bias_km: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel_bias_km_s: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        if np.array(self.pos_sigma_km, dtype=float).reshape(-1).size not in (1, 3):
            raise ValueError("pos_sigma_km must be scalar or length-3.")
        if np.array(self.vel_sigma_km_s, dtype=float).reshape(-1).size not in (1, 3):
            raise ValueError("vel_sigma_km_s must be scalar or length-3.")


@dataclass(frozen=True)
class KnowledgeEKFConfig:
    process_noise_diag: np.ndarray = field(default_factory=lambda: np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10]))
    meas_noise_diag: np.ndarray = field(default_factory=lambda: np.array([1e-6, 1e-6, 1e-6, 1e-10, 1e-10, 1e-10]))
    init_cov_diag: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-2]))

    def __post_init__(self) -> None:
        if np.array(self.process_noise_diag, dtype=float).reshape(-1).size != 6:
            raise ValueError("process_noise_diag must be length-6.")
        if np.array(self.meas_noise_diag, dtype=float).reshape(-1).size != 6:
            raise ValueError("meas_noise_diag must be length-6.")
        if np.array(self.init_cov_diag, dtype=float).reshape(-1).size != 6:
            raise ValueError("init_cov_diag must be length-6.")


@dataclass(frozen=True)
class TrackedObjectConfig:
    target_id: str
    conditions: KnowledgeConditionConfig = KnowledgeConditionConfig()
    sensor_noise: KnowledgeNoiseConfig = KnowledgeNoiseConfig()
    estimator: str = "ekf"
    ekf: KnowledgeEKFConfig = KnowledgeEKFConfig()


class _OtherObjectStateSensor:
    def __init__(self, conditions: KnowledgeConditionConfig, noise: KnowledgeNoiseConfig, rng: np.random.Generator):
        self.conditions = conditions
        self.noise = noise
        self.rng = rng
        self.access = AccessModel(
            AccessConfig(
                update_cadence_s=float(conditions.refresh_rate_s),
                max_range_km=conditions.max_range_km,
                fov_half_angle_rad=conditions.fov_half_angle_rad,
            )
        )

    def measure(self, observer_truth: StateTruth, target_truth: StateTruth, t_s: float) -> Measurement | None:
        if not self.access.can_update(observer_truth.position_eci_km, target_truth.position_eci_km, t_s):
            return None
        if self.conditions.require_line_of_sight and not _line_of_sight_clear(observer_truth.position_eci_km, target_truth.position_eci_km):
            return None
        if self.rng.random() < float(self.conditions.dropout_prob):
            return None

        pos_sigma = _expand3(self.noise.pos_sigma_km)
        vel_sigma = _expand3(self.noise.vel_sigma_km_s)
        pos_bias = _expand3(self.noise.pos_bias_km)
        vel_bias = _expand3(self.noise.vel_bias_km_s)
        z_pos = target_truth.position_eci_km + pos_bias + self.rng.normal(0.0, pos_sigma, size=3)
        z_vel = target_truth.velocity_eci_km_s + vel_bias + self.rng.normal(0.0, vel_sigma, size=3)
        return Measurement(vector=np.hstack((z_pos, z_vel)), t_s=t_s)


@dataclass
class _Track:
    target_id: str
    sensor: _OtherObjectStateSensor
    estimator: OrbitEKFEstimator
    init_cov_diag: np.ndarray
    belief: StateBelief | None = None

    def step(self, observer_truth: StateTruth, target_truth: StateTruth, t_s: float) -> StateBelief | None:
        meas = self.sensor.measure(observer_truth, target_truth, t_s)
        if self.belief is None:
            if meas is None:
                return None
            self.belief = StateBelief(state=meas.vector.copy(), covariance=np.diag(self.init_cov_diag), last_update_t_s=t_s)
            return self.belief
        self.belief = self.estimator.update(self.belief, meas, t_s)
        return self.belief


class ObjectKnowledgeBase:
    def __init__(
        self,
        observer_id: str,
        tracked_objects: list[TrackedObjectConfig],
        dt_s: float,
        rng: np.random.Generator | None = None,
        mu_km3_s2: float = EARTH_MU_KM3_S2,
    ):
        self.observer_id = observer_id
        self._rng = np.random.default_rng() if rng is None else rng
        self._tracks: dict[str, _Track] = {}

        for i, cfg in enumerate(tracked_objects):
            if cfg.target_id == observer_id:
                continue
            if cfg.estimator.lower() != "ekf":
                raise ValueError(f"Unsupported estimator '{cfg.estimator}' for target '{cfg.target_id}'.")
            trng = np.random.default_rng(int(self._rng.integers(0, 2**31 - 1)) + i)
            sensor = _OtherObjectStateSensor(cfg.conditions, cfg.sensor_noise, trng)
            ekf = OrbitEKFEstimator(
                mu_km3_s2=mu_km3_s2,
                dt_s=dt_s,
                process_noise_diag=np.array(cfg.ekf.process_noise_diag, dtype=float),
                meas_noise_diag=np.array(cfg.ekf.meas_noise_diag, dtype=float),
            )
            self._tracks[cfg.target_id] = _Track(
                target_id=cfg.target_id,
                sensor=sensor,
                estimator=ekf,
                init_cov_diag=np.array(cfg.ekf.init_cov_diag, dtype=float),
            )

    def target_ids(self) -> list[str]:
        return sorted(self._tracks.keys())

    def update(self, observer_truth: StateTruth, world_truth: dict[str, StateTruth], t_s: float) -> dict[str, StateBelief]:
        out: dict[str, StateBelief] = {}
        for target_id, track in self._tracks.items():
            tgt = world_truth.get(target_id)
            if tgt is None:
                continue
            b = track.step(observer_truth, tgt, t_s)
            if b is not None:
                out[target_id] = b
        return out

    def snapshot(self) -> dict[str, StateBelief]:
        out: dict[str, StateBelief] = {}
        for target_id, track in self._tracks.items():
            if track.belief is not None:
                out[target_id] = track.belief
        return out


def _expand3(v: np.ndarray) -> np.ndarray:
    a = np.array(v, dtype=float).reshape(-1)
    if a.size == 1:
        return np.full(3, float(a[0]))
    if a.size == 3:
        return a
    raise ValueError("Expected scalar or length-3 array.")
