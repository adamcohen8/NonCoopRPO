from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .atmosphere import EARTH_EQUATORIAL_RADIUS_KM
from .launch import Rocket
from .sat_params import SatParams
from .satellite import Satellite
from .strategies import StrategyLike


@dataclass(frozen=True)
class RocketPreset:
    name: str
    rocket: Rocket
    description: str = ""


@dataclass(frozen=True)
class SatellitePreset:
    name: str
    sat_params: SatParams
    description: str = ""


RE_KM = EARTH_EQUATORIAL_RADIUS_KM


ROCKET_PRESETS: dict[str, RocketPreset] = {
    "demo_medium_lift": RocketPreset(
        name="demo_medium_lift",
        description="Balanced medium-lift launcher for fast demo iterations.",
        rocket=Rocket(
            isp_s=335.0,
            dry_mass_kg=18000.0,
            fuel_mass_kg=120000.0,
            thrust_newton=2.8e6,
            cd=0.32,
            area_m2=10.0,
            vertical_ascent_time_s=30.0,
            pitch_over_time_s=110.0,
            max_flight_time_s=420.0,
            dt_s=1.0,
        ),
    ),
    "demo_heavy_lift": RocketPreset(
        name="demo_heavy_lift",
        description="Higher-mass launcher with larger delta-v margin.",
        rocket=Rocket(
            isp_s=410.0,
            dry_mass_kg=52000.0,
            fuel_mass_kg=430000.0,
            thrust_newton=5.4e6,
            cd=0.30,
            area_m2=19.0,
            vertical_ascent_time_s=45.0,
            pitch_over_time_s=230.0,
            guidance_gain_energy=9.0,
            guidance_gain_vr=5.5,
            max_flight_time_s=5400.0,
            dt_s=1.0,
        ),
    ),
    "demo_quick_insert": RocketPreset(
        name="demo_quick_insert",
        description="Aggressive ascent profile for short-run test loops.",
        rocket=Rocket(
            isp_s=320.0,
            dry_mass_kg=14000.0,
            fuel_mass_kg=76000.0,
            thrust_newton=2.0e6,
            cd=0.34,
            area_m2=8.5,
            vertical_ascent_time_s=20.0,
            pitch_over_time_s=85.0,
            max_flight_time_s=260.0,
            dt_s=1.0,
        ),
    ),
}


SATELLITE_PRESETS: dict[str, SatellitePreset] = {
    "target_500km_51p6": SatellitePreset(
        name="target_500km_51p6",
        description="Passive target near ISS-like inclination.",
        sat_params=SatParams(
            name="target_500km_51p6",
            coe=np.array([RE_KM + 500.0, 0.001, np.deg2rad(51.6), np.deg2rad(10.0), 0.0, 0.0]),
            propellant_dv_km_s=np.inf,
        ),
    ),
    "escort_505km_51p6": SatellitePreset(
        name="escort_505km_51p6",
        description="Nearby passive companion orbit for multi-agent scenes.",
        sat_params=SatParams(
            name="escort_505km_51p6",
            coe=np.array([RE_KM + 505.0, 0.001, np.deg2rad(51.6), np.deg2rad(12.0), 0.0, 0.2]),
            propellant_dv_km_s=np.inf,
        ),
    ),
    "inspector_rpo": SatellitePreset(
        name="inspector_rpo",
        description="Maneuvering RPO spacecraft with modest acceleration and dv budget.",
        sat_params=SatParams(
            name="inspector_rpo",
            coe=np.array([RE_KM + 500.0, 0.0005, np.deg2rad(51.6), np.deg2rad(10.2), 0.0, 0.05]),
            max_accel_km_s2=2.5e-6,
            min_accel_km_s2=0.0,
            propellant_dv_km_s=0.08,
        ),
    ),
}


def list_rocket_presets() -> list[str]:
    return sorted(ROCKET_PRESETS.keys())


def list_satellite_presets() -> list[str]:
    return sorted(SATELLITE_PRESETS.keys())


def get_rocket_preset(name: str) -> RocketPreset:
    try:
        return ROCKET_PRESETS[name]
    except KeyError as exc:
        options = ", ".join(list_rocket_presets())
        raise KeyError(f"Unknown rocket preset '{name}'. Available: {options}") from exc


def get_satellite_preset(name: str) -> SatellitePreset:
    try:
        return SATELLITE_PRESETS[name]
    except KeyError as exc:
        options = ", ".join(list_satellite_presets())
        raise KeyError(f"Unknown satellite preset '{name}'. Available: {options}") from exc


def create_rocket(preset_name: str, **overrides: object) -> Rocket:
    preset = get_rocket_preset(preset_name)
    return replace(preset.rocket, **overrides)


def create_sat_params(
    preset_name: str,
    *,
    name: Optional[str] = None,
    **overrides: object,
) -> SatParams:
    preset = get_satellite_preset(preset_name)
    params = replace(preset.sat_params)
    if name is not None:
        params = replace(params, name=name)
    if overrides:
        params = replace(params, **overrides)
    return params


def create_satellite(
    preset_name: str,
    *,
    name: Optional[str] = None,
    strategy: Optional[StrategyLike] = None,
    policy: Optional[StrategyLike] = None,
    **overrides: object,
) -> Satellite:
    if strategy is not None and policy is not None:
        raise ValueError("Provide at most one of strategy or policy.")
    sat_params = create_sat_params(preset_name, name=name, **overrides)
    strategy_like = strategy if strategy is not None else policy
    return Satellite.from_params(sat_params, strategy=strategy_like)
