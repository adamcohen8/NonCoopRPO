from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from datetime import datetime, timezone

from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig
from sim.dynamics.orbit.epoch import datetime_to_julian_date
from sim.dynamics.orbit.propagator import (
    OrbitPropagator,
    drag_plugin,
    j2_plugin,
    j3_plugin,
    j4_plugin,
    srp_plugin,
    third_body_moon_plugin,
    third_body_sun_plugin,
)

SimulationProfileName = Literal["fast", "ops", "high_fidelity"]


@dataclass(frozen=True)
class SimulationProfile:
    name: SimulationProfileName
    kernel_dt_s: float
    kernel_integrator: str
    controller_budget_ms: float
    realtime_mode: bool
    orbit_integrator: str
    orbit_substep_s: float | None
    orbit_adaptive_atol: float
    orbit_adaptive_rtol: float
    attitude_substep_s: float | None
    atmosphere_model: str


PROFILE_FAST = SimulationProfile(
    name="fast",
    kernel_dt_s=2.0,
    kernel_integrator="rk4",
    controller_budget_ms=1.0,
    realtime_mode=True,
    orbit_integrator="rk4",
    orbit_substep_s=2.0,
    orbit_adaptive_atol=1e-9,
    orbit_adaptive_rtol=1e-7,
    attitude_substep_s=0.05,
    atmosphere_model="exponential",
)

PROFILE_OPS = SimulationProfile(
    name="ops",
    kernel_dt_s=1.0,
    kernel_integrator="rk4",
    controller_budget_ms=2.0,
    realtime_mode=True,
    orbit_integrator="rk4",
    orbit_substep_s=1.0,
    orbit_adaptive_atol=1e-10,
    orbit_adaptive_rtol=1e-8,
    attitude_substep_s=0.01,
    atmosphere_model="ussa1976",
)

PROFILE_HIGH_FIDELITY = SimulationProfile(
    name="high_fidelity",
    kernel_dt_s=0.5,
    kernel_integrator="adaptive",
    controller_budget_ms=3.0,
    realtime_mode=True,
    orbit_integrator="adaptive",
    orbit_substep_s=0.5,
    orbit_adaptive_atol=1e-12,
    orbit_adaptive_rtol=1e-10,
    attitude_substep_s=0.005,
    atmosphere_model="nrlmsise00",
)

_PROFILES: dict[SimulationProfileName, SimulationProfile] = {
    "fast": PROFILE_FAST,
    "ops": PROFILE_OPS,
    "high_fidelity": PROFILE_HIGH_FIDELITY,
}


def profile_choices() -> tuple[str, ...]:
    return tuple(_PROFILES.keys())


def get_simulation_profile(name: str) -> SimulationProfile:
    key = name.strip().lower()
    if key not in _PROFILES:
        choices = ", ".join(profile_choices())
        raise ValueError(f"Unknown profile '{name}'. Expected one of: {choices}.")
    return _PROFILES[key]  # type: ignore[return-value]


def resolve_dt_s(profile_name: str, dt_override_s: float | None = None) -> float:
    if dt_override_s is not None:
        return float(dt_override_s)
    return float(get_simulation_profile(profile_name).kernel_dt_s)


def resolve_steps_for_duration(duration_s: float, profile_name: str, dt_override_s: float | None = None) -> int:
    dt_s = resolve_dt_s(profile_name=profile_name, dt_override_s=dt_override_s)
    return int(np.ceil(float(duration_s) / dt_s))


def default_env_for_profile(profile_name: str) -> dict:
    profile = get_simulation_profile(profile_name)
    jd_now = datetime_to_julian_date(datetime.now(tz=timezone.utc))
    return {
        "atmosphere_model": profile.atmosphere_model,
        "jd_utc_start": jd_now,
        "ephemeris_mode": "analytic_enhanced",
        "sun_dir_eci": np.array([1.0, 0.0, 0.0], dtype=float),
        "sun_pos_eci_km": np.array([149597870.7, 0.0, 0.0], dtype=float),
        "moon_pos_eci_km": np.array([384400.0, 0.0, 0.0], dtype=float),
    }


def default_disturbance_config_for_profile(profile_name: str) -> DisturbanceTorqueConfig:
    name = get_simulation_profile(profile_name).name
    if name == "fast":
        return DisturbanceTorqueConfig(
            use_gravity_gradient=False,
            use_magnetic=False,
            use_drag=False,
            use_srp=False,
        )
    return DisturbanceTorqueConfig(
        use_gravity_gradient=True,
        use_magnetic=True,
        use_drag=True,
        use_srp=True,
    )


def build_orbit_propagator_for_profile(
    profile_name: str,
    *,
    include_j2: bool = False,
    include_j3: bool = False,
    include_j4: bool = False,
    include_drag: bool = False,
    include_srp: bool = False,
    include_third_body_sun: bool = False,
    include_third_body_moon: bool = False,
) -> OrbitPropagator:
    profile = get_simulation_profile(profile_name)
    plugins = []
    if include_j2:
        plugins.append(j2_plugin)
    if include_j3:
        plugins.append(j3_plugin)
    if include_j4:
        plugins.append(j4_plugin)
    if include_drag:
        plugins.append(drag_plugin)
    if include_srp:
        plugins.append(srp_plugin)
    if include_third_body_sun:
        plugins.append(third_body_sun_plugin)
    if include_third_body_moon:
        plugins.append(third_body_moon_plugin)
    return OrbitPropagator(
        integrator=profile.orbit_integrator,
        plugins=plugins,
        adaptive_atol=profile.orbit_adaptive_atol,
        adaptive_rtol=profile.orbit_adaptive_rtol,
    )


def build_default_ops_orbit_propagator(profile_name: str) -> OrbitPropagator:
    name = get_simulation_profile(profile_name).name
    if name == "fast":
        return build_orbit_propagator_for_profile(profile_name)
    if name == "ops":
        return build_orbit_propagator_for_profile(
            profile_name,
            include_j2=True,
            include_j3=True,
            include_j4=True,
            include_drag=True,
            include_srp=True,
            include_third_body_sun=True,
            include_third_body_moon=True,
        )
    return build_orbit_propagator_for_profile(
        profile_name,
        include_j2=True,
        include_j3=True,
        include_j4=True,
        include_drag=True,
        include_srp=True,
        include_third_body_sun=True,
        include_third_body_moon=True,
    )
