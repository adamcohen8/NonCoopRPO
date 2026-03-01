from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .attitude import AttitudeConstraint
from .constants import MU_EARTH_KM3_S2
from .orbital_elements import coe2rv as default_coe2rv


@dataclass(frozen=True)
class SatParams:
    name: str
    mu: float = MU_EARTH_KM3_S2
    max_accel_km_s2: float = 0.0
    min_accel_km_s2: float = 0.0
    propellant_dv_km_s: float = np.inf
    attitude_control_enabled: bool = False
    inertia_body_kg_m2: Optional[np.ndarray] = None
    max_torque_nm: Optional[np.ndarray] = None
    r0_eci_km: Optional[np.ndarray] = None
    v0_eci_km_s: Optional[np.ndarray] = None
    coe: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        has_eci = self.r0_eci_km is not None and self.v0_eci_km_s is not None
        has_coe = self.coe is not None
        if has_eci == has_coe:
            raise ValueError("Provide exactly one initialization method: ECI or COE.")
        if self.max_accel_km_s2 < 0.0:
            raise ValueError("max_accel_km_s2 must be non-negative.")
        if self.min_accel_km_s2 < 0.0:
            raise ValueError("min_accel_km_s2 must be non-negative.")
        if self.min_accel_km_s2 > self.max_accel_km_s2:
            raise ValueError("min_accel_km_s2 cannot exceed max_accel_km_s2.")
        if self.propellant_dv_km_s < 0.0:
            raise ValueError("propellant_dv_km_s must be non-negative.")
        AttitudeConstraint(
            enabled=self.attitude_control_enabled,
            inertia_body_kg_m2=self.inertia_body_kg_m2,
            max_torque_nm=self.max_torque_nm,
        ).validate("SatParams")

    def initial_eci_state(
        self, coe2rv_func: Optional[Callable[..., tuple[np.ndarray, np.ndarray]]] = None
    ) -> np.ndarray:
        if self.r0_eci_km is not None and self.v0_eci_km_s is not None:
            return np.hstack((np.asarray(self.r0_eci_km, dtype=float), np.asarray(self.v0_eci_km_s, dtype=float)))

        elems = np.asarray(self.coe, dtype=float)
        if elems.size != 6:
            raise ValueError("coe must be a 6-vector: [p, ecc, incl, raan, argp, nu].")
        fn = coe2rv_func or default_coe2rv
        r0, v0 = fn(*elems, mu=self.mu)
        return np.hstack((r0, v0))

    @property
    def a_max_km_s2(self) -> float:
        """
        Backward-compatible alias.
        """
        return self.max_accel_km_s2
