from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StateTruth:
    position_eci_km: np.ndarray
    velocity_eci_km_s: np.ndarray
    attitude_quat_bn: np.ndarray
    angular_rate_body_rad_s: np.ndarray
    mass_kg: float
    t_s: float

    def copy(self) -> "StateTruth":
        return StateTruth(
            position_eci_km=self.position_eci_km.copy(),
            velocity_eci_km_s=self.velocity_eci_km_s.copy(),
            attitude_quat_bn=self.attitude_quat_bn.copy(),
            angular_rate_body_rad_s=self.angular_rate_body_rad_s.copy(),
            mass_kg=float(self.mass_kg),
            t_s=float(self.t_s),
        )


@dataclass
class StateBelief:
    state: np.ndarray
    covariance: np.ndarray
    last_update_t_s: float


@dataclass
class Command:
    thrust_eci_km_s2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque_body_nm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mode_flags: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def zero() -> "Command":
        return Command()


@dataclass(frozen=True)
class SimConfig:
    dt_s: float
    steps: int
    integrator: str = "rk4"
    realtime_mode: bool = True
    controller_budget_ms: float = 2.0
    rng_seed: int = 0
    initial_jd_utc: float | None = None

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.controller_budget_ms <= 0.0:
            raise ValueError("controller_budget_ms must be positive")
        if self.initial_jd_utc is not None and self.initial_jd_utc <= 0.0:
            raise ValueError("initial_jd_utc must be positive when provided")


@dataclass(frozen=True)
class ObjectConfig:
    object_id: str
    controller_budget_ms: float | None = None

    def budget_ms(self, default_budget_ms: float) -> float:
        return default_budget_ms if self.controller_budget_ms is None else self.controller_budget_ms


@dataclass
class Measurement:
    vector: np.ndarray
    t_s: float


@dataclass
class RuntimeStats:
    controller_runtime_ms: dict[str, list[float]]
    controller_skipped: dict[str, list[bool]]


@dataclass
class SimLog:
    t_s: np.ndarray
    truth_by_object: dict[str, np.ndarray]
    belief_by_object: dict[str, np.ndarray]
    applied_thrust_by_object: dict[str, np.ndarray]
    applied_torque_by_object: dict[str, np.ndarray]
    controller_runtime_ms_by_object: dict[str, np.ndarray]
    controller_skipped_by_object: dict[str, np.ndarray]
    knowledge_by_observer: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    srp_shadow_by_object: dict[str, np.ndarray] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "t_s": self.t_s.tolist(),
            "truth_by_object": {k: v.tolist() for k, v in self.truth_by_object.items()},
            "belief_by_object": {k: v.tolist() for k, v in self.belief_by_object.items()},
            "knowledge_by_observer": {
                obs: {tgt: hist.tolist() for tgt, hist in by_tgt.items()}
                for obs, by_tgt in self.knowledge_by_observer.items()
            },
            "applied_thrust_by_object": {k: v.tolist() for k, v in self.applied_thrust_by_object.items()},
            "applied_torque_by_object": {k: v.tolist() for k, v in self.applied_torque_by_object.items()},
            "controller_runtime_ms_by_object": {
                k: v.tolist() for k, v in self.controller_runtime_ms_by_object.items()
            },
            "controller_skipped_by_object": {
                k: v.astype(int).tolist() for k, v in self.controller_skipped_by_object.items()
            },
            "srp_shadow_by_object": {k: v.tolist() for k, v in self.srp_shadow_by_object.items()},
        }
