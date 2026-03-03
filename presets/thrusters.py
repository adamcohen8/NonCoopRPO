from __future__ import annotations

from dataclasses import dataclass

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
