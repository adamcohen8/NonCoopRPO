from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SatellitePreset:
    name: str
    dry_mass_kg: float
    propellant_mass_kg: float
    bus_size_m: tuple[float, float, float]
    inertia_kg_m2: np.ndarray

    @property
    def wet_mass_kg(self) -> float:
        return self.dry_mass_kg + self.propellant_mass_kg


BASIC_SATELLITE = SatellitePreset(
    name="Basic Satellite",
    dry_mass_kg=260.0,
    propellant_mass_kg=40.0,
    bus_size_m=(1.2, 1.0, 1.0),
    inertia_kg_m2=np.diag([120.0, 100.0, 80.0]),
)
