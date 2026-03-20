from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.models import Measurement, StateBelief, StateTruth
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.sensors.access import AccessConfig, AccessModel
from sim.utils.quaternion import quaternion_to_dcm_bn


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
    solid_angle_sr: float | None = None
    require_line_of_sight: bool = False
    dropout_prob: float = 0.0
    sensor_position_body_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sensor_boresight_body: np.ndarray | None = None


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
                solid_angle_sr=conditions.solid_angle_sr,
            )
        )
        self.last_detection_status: str | None = None

    def measure(self, observer_truth: StateTruth, target_truth: StateTruth, t_s: float) -> Measurement | None:
        sensor_position_eci_km, sensor_boresight_eci = self._sensor_pose_eci(observer_truth)
        access_ok, access_reason = self.access.evaluate(
            sensor_position_eci_km,
            target_truth.position_eci_km,
            t_s,
            boresight_eci=sensor_boresight_eci,
        )
        if not access_ok:
            self.last_detection_status = str(access_reason)
            return None
        if self.conditions.require_line_of_sight and not _line_of_sight_clear(sensor_position_eci_km, target_truth.position_eci_km):
            self.last_detection_status = "line_of_sight"
            return None
        if self.rng.random() < float(self.conditions.dropout_prob):
            self.last_detection_status = "dropout"
            return None
        self.access._last_update_t_s = float(t_s)
        self.last_detection_status = "detected"

        pos_sigma = _expand3(self.noise.pos_sigma_km)
        vel_sigma = _expand3(self.noise.vel_sigma_km_s)
        pos_bias = _expand3(self.noise.pos_bias_km)
        vel_bias = _expand3(self.noise.vel_bias_km_s)
        z_pos = target_truth.position_eci_km + pos_bias + self.rng.normal(0.0, pos_sigma, size=3)
        z_vel = target_truth.velocity_eci_km_s + vel_bias + self.rng.normal(0.0, vel_sigma, size=3)
        return Measurement(vector=np.hstack((z_pos, z_vel)), t_s=t_s)

    def _sensor_pose_eci(self, observer_truth: StateTruth) -> tuple[np.ndarray, np.ndarray | None]:
        c_bn = quaternion_to_dcm_bn(observer_truth.attitude_quat_bn)
        pos_body_m = np.array(self.conditions.sensor_position_body_m, dtype=float).reshape(3)
        sensor_position_eci_km = observer_truth.position_eci_km + (c_bn.T @ pos_body_m) / 1e3
        boresight_body = self.conditions.sensor_boresight_body
        if boresight_body is None:
            if float(np.linalg.norm(pos_body_m)) > 1e-12:
                boresight_body = pos_body_m
            else:
                boresight_body = np.array([1.0, 0.0, 0.0], dtype=float)
        b = np.array(boresight_body, dtype=float).reshape(3)
        bn = float(np.linalg.norm(b))
        sensor_boresight_eci = None if bn <= 0.0 else (c_bn.T @ (b / bn))
        return sensor_position_eci_km, sensor_boresight_eci


@dataclass
class _Track:
    target_id: str
    sensor: _OtherObjectStateSensor
    estimator: OrbitEKFEstimator
    init_cov_diag: np.ndarray
    belief: StateBelief | None = None
    step_count: int = 0
    initialization_count: int = 0
    measurement_count: int = 0
    update_count: int = 0
    last_measurement_t_s: float | None = None
    nis_values: list[float] = field(default_factory=list)
    nees_values: list[float] = field(default_factory=list)
    innovation_norm_values: list[float] = field(default_factory=list)
    pos_error_norm_km_values: list[float] = field(default_factory=list)
    vel_error_norm_km_s_values: list[float] = field(default_factory=list)
    track_age_s_values: list[float] = field(default_factory=list)
    detected_count: int = 0
    reacquisition_count: int = 0
    loss_of_detection_count: int = 0
    consecutive_missed_steps: int = 0
    max_consecutive_missed_steps: int = 0
    last_detected: bool = False
    time_since_last_detection_s_values: list[float] = field(default_factory=list)
    detection_status_counts: dict[str, int] = field(default_factory=dict)

    def step(self, observer_truth: StateTruth, target_truth: StateTruth, t_s: float) -> StateBelief | None:
        self.step_count += 1
        meas = self.sensor.measure(observer_truth, target_truth, t_s)
        detect_status = str(self.sensor.last_detection_status or "unknown")
        self.detection_status_counts[detect_status] = int(self.detection_status_counts.get(detect_status, 0)) + 1
        if meas is not None:
            self.measurement_count += 1
            self.last_measurement_t_s = float(t_s)
            self.detected_count += 1
            if not self.last_detected and (self.detected_count > 1):
                self.reacquisition_count += 1
            self.last_detected = True
            self.consecutive_missed_steps = 0
            self.time_since_last_detection_s_values.append(0.0)
        else:
            if self.last_detected:
                self.loss_of_detection_count += 1
            self.last_detected = False
            self.consecutive_missed_steps += 1
            self.max_consecutive_missed_steps = max(self.max_consecutive_missed_steps, self.consecutive_missed_steps)
            if self.last_measurement_t_s is not None:
                self.time_since_last_detection_s_values.append(float(t_s - self.last_measurement_t_s))
        if self.belief is None:
            if meas is None:
                return None
            self.belief = StateBelief(state=meas.vector.copy(), covariance=np.diag(self.init_cov_diag), last_update_t_s=t_s)
            self.initialization_count += 1
            self._record_consistency(target_truth, None, t_s)
            return self.belief
        self.belief = self.estimator.update(self.belief, meas, t_s)
        diag = self.estimator.last_update_diagnostics
        if diag is not None and diag.update_applied:
            self.update_count += 1
        self._record_consistency(target_truth, diag, t_s)
        return self.belief

    def _record_consistency(self, target_truth: StateTruth, diag: object | None, t_s: float) -> None:
        if self.belief is None:
            return
        err = np.array(self.belief.state[:6], dtype=float) - np.hstack((target_truth.position_eci_km, target_truth.velocity_eci_km_s))
        pos_err = err[:3]
        vel_err = err[3:6]
        self.pos_error_norm_km_values.append(float(np.linalg.norm(pos_err)))
        self.vel_error_norm_km_s_values.append(float(np.linalg.norm(vel_err)))
        age_s = float(t_s - self.last_measurement_t_s) if self.last_measurement_t_s is not None else float("nan")
        self.track_age_s_values.append(age_s)
        cov = np.array(self.belief.covariance, dtype=float)
        if cov.shape == (6, 6):
            try:
                nees = float(err.T @ np.linalg.solve(cov, err))
                if np.isfinite(nees):
                    self.nees_values.append(nees)
            except np.linalg.LinAlgError:
                pass
        if diag is None:
            return
        nis = float(getattr(diag, "nis", float("nan")))
        if np.isfinite(nis):
            self.nis_values.append(nis)
        innovation = np.array(getattr(diag, "innovation", np.full(6, np.nan)), dtype=float).reshape(-1)
        if innovation.size:
            innovation_norm = float(np.linalg.norm(innovation))
            if np.isfinite(innovation_norm):
                self.innovation_norm_values.append(innovation_norm)

    def consistency_summary(self) -> dict[str, float | int | None]:
        update_rate = float(self.update_count / max(self.step_count, 1))
        measurement_rate = float(self.measurement_count / max(self.step_count, 1))
        return {
            "step_count": int(self.step_count),
            "initialization_count": int(self.initialization_count),
            "measurement_count": int(self.measurement_count),
            "update_count": int(self.update_count),
            "measurement_rate": measurement_rate,
            "update_rate": update_rate,
            "nis_mean": _safe_stat_mean(self.nis_values),
            "nis_p95": _safe_stat_percentile(self.nis_values, 95.0),
            "nees_mean": _safe_stat_mean(self.nees_values),
            "nees_p95": _safe_stat_percentile(self.nees_values, 95.0),
            "innovation_norm_mean": _safe_stat_mean(self.innovation_norm_values),
            "innovation_norm_p95": _safe_stat_percentile(self.innovation_norm_values, 95.0),
            "pos_error_rms_km": _safe_stat_rms(self.pos_error_norm_km_values),
            "vel_error_rms_km_s": _safe_stat_rms(self.vel_error_norm_km_s_values),
            "track_age_s_mean": _safe_stat_mean(self.track_age_s_values),
            "track_age_s_p95": _safe_stat_percentile(self.track_age_s_values, 95.0),
        }

    def detection_summary(self) -> dict[str, float | int | dict[str, int] | None]:
        detection_rate = float(self.detected_count / max(self.step_count, 1))
        nondetection_rate = float(1.0 - detection_rate)
        return {
            "step_count": int(self.step_count),
            "detected_count": int(self.detected_count),
            "nondetected_count": int(max(self.step_count - self.detected_count, 0)),
            "detection_rate": detection_rate,
            "nondetection_rate": nondetection_rate,
            "reacquisition_count": int(self.reacquisition_count),
            "loss_of_detection_count": int(self.loss_of_detection_count),
            "max_consecutive_missed_steps": int(self.max_consecutive_missed_steps),
            "time_since_last_detection_s_mean": _safe_stat_mean(self.time_since_last_detection_s_values),
            "time_since_last_detection_s_p95": _safe_stat_percentile(self.time_since_last_detection_s_values, 95.0),
            "status_counts": {str(k): int(v) for k, v in sorted(self.detection_status_counts.items())},
        }


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

    def consistency_summary(self) -> dict[str, dict[str, float | int | None]]:
        return {
            str(target_id): track.consistency_summary()
            for target_id, track in sorted(self._tracks.items())
        }

    def detection_summary(self) -> dict[str, dict[str, float | int | dict[str, int] | None]]:
        return {
            str(target_id): track.detection_summary()
            for target_id, track in sorted(self._tracks.items())
        }


def _expand3(v: np.ndarray) -> np.ndarray:
    a = np.array(v, dtype=float).reshape(-1)
    if a.size == 1:
        return np.full(3, float(a[0]))
    if a.size == 3:
        return a
    raise ValueError("Expected scalar or length-3 array.")


def _safe_stat_array(values: list[float]) -> np.ndarray:
    arr = np.array(values, dtype=float)
    return arr[np.isfinite(arr)]


def _safe_stat_mean(values: list[float]) -> float | None:
    arr = _safe_stat_array(values)
    return float(np.mean(arr)) if arr.size else None


def _safe_stat_percentile(values: list[float], pct: float) -> float | None:
    arr = _safe_stat_array(values)
    return float(np.percentile(arr, pct)) if arr.size else None


def _safe_stat_rms(values: list[float]) -> float | None:
    arr = _safe_stat_array(values)
    return float(np.sqrt(np.mean(arr**2))) if arr.size else None
