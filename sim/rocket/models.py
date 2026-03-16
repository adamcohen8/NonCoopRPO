from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from presets.rockets import RocketStackPreset
from sim.rocket.aero import RocketAeroConfig


@dataclass(frozen=True)
class RocketSimConfig:
    dt_s: float = 0.5
    max_time_s: float = 2000.0
    target_altitude_km: float = 400.0
    target_altitude_tolerance_km: float = 25.0
    target_eccentricity_max: float = 0.02
    insertion_hold_time_s: float = 30.0
    launch_lat_deg: float = 0.0
    launch_lon_deg: float = 0.0
    launch_alt_km: float = 0.0
    launch_azimuth_deg: float = 90.0
    atmosphere_model: str = "ussa1976"
    enable_drag: bool = True
    enable_srp: bool = False
    enable_j2: bool = True
    enable_j3: bool = False
    enable_j4: bool = False
    terminate_on_earth_impact: bool = True
    earth_impact_radius_km: float = 6378.137
    area_ref_m2: float | None = None
    use_stagewise_aero_geometry: bool = True
    cd: float = 0.35
    cr: float = 1.2
    aero: RocketAeroConfig = field(default_factory=RocketAeroConfig)
    atmosphere_env: dict = field(default_factory=dict)
    inertia_kg_m2: np.ndarray = field(default_factory=lambda: np.diag([8.0e5, 8.0e5, 2.0e4]))
    attitude_substep_s: float = 0.02
    attitude_mode: str = "dynamic"  # dynamic | cheater

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        if self.max_time_s <= 0.0:
            raise ValueError("max_time_s must be positive.")
        if self.target_altitude_tolerance_km < 0.0:
            raise ValueError("target_altitude_tolerance_km must be non-negative.")
        if self.target_eccentricity_max < 0.0:
            raise ValueError("target_eccentricity_max must be non-negative.")
        if self.insertion_hold_time_s < 0.0:
            raise ValueError("insertion_hold_time_s must be non-negative.")
        if self.attitude_substep_s <= 0.0:
            raise ValueError("attitude_substep_s must be positive.")
        if self.earth_impact_radius_km <= 0.0:
            raise ValueError("earth_impact_radius_km must be positive.")
        mode = str(self.attitude_mode).strip().lower()
        if mode not in ("dynamic", "cheater"):
            raise ValueError("attitude_mode must be 'dynamic' or 'cheater'.")


@dataclass(frozen=True)
class RocketVehicleConfig:
    stack: RocketStackPreset
    payload_mass_kg: float
    thrust_axis_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    def __post_init__(self) -> None:
        if self.payload_mass_kg < 0.0:
            raise ValueError("payload_mass_kg must be non-negative.")
        axis = np.array(self.thrust_axis_body, dtype=float).reshape(-1)
        if axis.size != 3:
            raise ValueError("thrust_axis_body must be length-3.")
        if np.linalg.norm(axis) <= 0.0:
            raise ValueError("thrust_axis_body cannot be zero.")


@dataclass
class RocketState:
    t_s: float
    position_eci_km: np.ndarray
    velocity_eci_km_s: np.ndarray
    attitude_quat_bn: np.ndarray
    angular_rate_body_rad_s: np.ndarray
    mass_kg: float
    active_stage_index: int
    stage_prop_remaining_kg: np.ndarray
    payload_attached: bool = True

    def copy(self) -> "RocketState":
        return RocketState(
            t_s=float(self.t_s),
            position_eci_km=self.position_eci_km.copy(),
            velocity_eci_km_s=self.velocity_eci_km_s.copy(),
            attitude_quat_bn=self.attitude_quat_bn.copy(),
            angular_rate_body_rad_s=self.angular_rate_body_rad_s.copy(),
            mass_kg=float(self.mass_kg),
            active_stage_index=int(self.active_stage_index),
            stage_prop_remaining_kg=self.stage_prop_remaining_kg.copy(),
            payload_attached=bool(self.payload_attached),
        )


@dataclass(frozen=True)
class GuidanceCommand:
    throttle: float
    attitude_quat_bn_cmd: np.ndarray | None = None
    torque_body_nm_cmd: np.ndarray | None = None


class RocketGuidanceLaw(Protocol):
    def command(self, state: RocketState, sim_cfg: RocketSimConfig, vehicle_cfg: RocketVehicleConfig) -> GuidanceCommand:
        ...


@dataclass
class RocketSimResult:
    time_s: np.ndarray
    position_eci_km: np.ndarray
    velocity_eci_km_s: np.ndarray
    attitude_quat_bn: np.ndarray
    angular_rate_body_rad_s: np.ndarray
    mass_kg: np.ndarray
    active_stage_index: np.ndarray
    throttle_cmd: np.ndarray
    thrust_n: np.ndarray
    altitude_km: np.ndarray
    eccentricity: np.ndarray
    sma_km: np.ndarray
    dynamic_pressure_pa: np.ndarray
    mach: np.ndarray
    alpha_deg: np.ndarray
    beta_deg: np.ndarray
    cd: np.ndarray
    aero_force_n: np.ndarray
    aero_moment_nm: np.ndarray
    inserted: bool
    insertion_time_s: float | None
    terminated_early: bool = False
    termination_reason: str | None = None
    termination_time_s: float | None = None
