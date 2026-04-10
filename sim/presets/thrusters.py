from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ThrusterMountPreset:
    name: str
    position_body_m: np.ndarray
    thrust_direction_body: np.ndarray


@dataclass(frozen=True)
class ChemicalPropulsionPreset:
    name: str
    max_thrust_n: float
    isp_s: float
    min_impulse_bit_n_s: float
    mount: ThrusterMountPreset


BASIC_CHEMICAL_BOTTOM_Z = ChemicalPropulsionPreset(
    name="Basic Chemical Bottom-Z",
    max_thrust_n=35.0,
    isp_s=220.0,
    min_impulse_bit_n_s=0.7,
    mount=ThrusterMountPreset(
        name="Bottom Z Panel Centerline Mount",
        position_body_m=np.array([0.0, 0.0, -0.50]),
        thrust_direction_body=np.array([0.0, 0.0, 1.0]),
    ),
)


def resolve_thruster_preset_from_specs(specs: dict[str, Any] | None) -> ChemicalPropulsionPreset | None:
    raw = dict(specs or {})
    preset_name = str(raw.get("thruster", "") or "").strip().upper()
    if preset_name in ("BASIC_CHEMICAL_BOTTOM_Z", "BASIC_CHEMICAL_Z_BOTTOM"):
        return BASIC_CHEMICAL_BOTTOM_Z
    return None


def resolve_thruster_max_thrust_n_from_specs(specs: dict[str, Any] | None) -> float | None:
    raw = dict(specs or {})
    explicit = raw.get("max_thrust_n")
    if explicit is not None:
        try:
            val = float(explicit)
        except (TypeError, ValueError):
            val = np.nan
        if np.isfinite(val) and val >= 0.0:
            return float(val)
    preset = resolve_thruster_preset_from_specs(raw)
    if preset is not None:
        return float(preset.max_thrust_n)
    return None


def resolve_thruster_mount_from_specs(specs: dict[str, Any] | None) -> ThrusterMountPreset | None:
    raw = dict(specs or {})
    preset = resolve_thruster_preset_from_specs(raw)
    preset_mount: ThrusterMountPreset | None = None if preset is None else preset.mount

    explicit_direction = raw.get("thruster_direction_body")
    explicit_position = raw.get("thruster_position_body_m")
    if explicit_direction is None and explicit_position is None:
        return preset_mount

    direction_body = (
        np.array(explicit_direction, dtype=float)
        if explicit_direction is not None
        else np.array(preset_mount.thrust_direction_body if preset_mount is not None else [1.0, 0.0, 0.0], dtype=float)
    )
    position_body_m = (
        np.array(explicit_position, dtype=float)
        if explicit_position is not None
        else np.array(preset_mount.position_body_m if preset_mount is not None else [0.0, 0.0, 0.0], dtype=float)
    )
    return ThrusterMountPreset(
        name=str(raw.get("thruster_mount_name", getattr(preset_mount, "name", "Configured Thruster Mount"))),
        position_body_m=position_body_m,
        thrust_direction_body=direction_body,
    )
