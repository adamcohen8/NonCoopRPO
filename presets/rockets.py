from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RocketStagePreset:
    name: str
    dry_mass_kg: float
    propellant_mass_kg: float
    max_thrust_n: float
    isp_s: float
    burn_time_s: float
    diameter_m: float
    length_m: float


@dataclass(frozen=True)
class RocketStackPreset:
    name: str
    stages: tuple[RocketStagePreset, ...]

    @property
    def dry_mass_kg(self) -> float:
        return sum(stage.dry_mass_kg for stage in self.stages)

    @property
    def propellant_mass_kg(self) -> float:
        return sum(stage.propellant_mass_kg for stage in self.stages)

    @property
    def liftoff_mass_kg(self) -> float:
        return self.dry_mass_kg + self.propellant_mass_kg


BASIC_1ST_STAGE = RocketStagePreset(
    name="Basic 1st Stage",
    dry_mass_kg=28000.0,
    propellant_mass_kg=360000.0,
    max_thrust_n=7.6e6,
    isp_s=300.0,
    burn_time_s=160.0,
    diameter_m=3.7,
    length_m=42.0,
)

BASIC_2ND_STAGE = RocketStagePreset(
    name="Basic 2nd Stage",
    dry_mass_kg=4500.0,
    propellant_mass_kg=95000.0,
    max_thrust_n=1.0e6,
    isp_s=348.0,
    burn_time_s=380.0,
    diameter_m=3.7,
    length_m=14.0,
)

BASIC_SSTO_ROCKET = RocketStagePreset(
    name="Basic SSTO",
    dry_mass_kg=58000.0,
    propellant_mass_kg=520000.0,
    max_thrust_n=8.8e6,
    isp_s=325.0,
    burn_time_s=480.0,
    diameter_m=4.1,
    length_m=52.0,
)

BASIC_TWO_STAGE_STACK = RocketStackPreset(
    name="Basic Two-Stage Stack",
    stages=(BASIC_1ST_STAGE, BASIC_2ND_STAGE),
)
