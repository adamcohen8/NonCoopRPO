from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .frames import eci2hcw_curv


@dataclass
class KnowledgeModel:
    """
    Observation model for "other satellite as target" in curvilinear RIC.
    """
    noise_std_percent_of_distance: float = 0.0
    detection_range_km: float = np.inf
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.noise_std_percent_of_distance < 0.0:
            raise ValueError("noise_std_percent_of_distance must be non-negative.")
        if self.detection_range_km <= 0.0:
            raise ValueError("detection_range_km must be positive.")
        self._rng = np.random.default_rng(self.seed)

    def observe(self, observer_x_eci: np.ndarray, target_x_eci: np.ndarray) -> tuple[Optional[np.ndarray], bool]:
        """
        Returns measured [R,I,C,dR,dI,dC] in curvilinear RIC where
        the observer is treated as the chaser and target as the target.
        """
        observer_x_eci = np.asarray(observer_x_eci, dtype=float)
        target_x_eci = np.asarray(target_x_eci, dtype=float)
        separation_km = float(np.linalg.norm(target_x_eci[:3] - observer_x_eci[:3]))

        if separation_km > self.detection_range_km:
            return None, False

        x_true = eci2hcw_curv(target_x_eci, observer_x_eci)
        if self.noise_std_percent_of_distance <= 0.0:
            return x_true, True

        sigma_km = (self.noise_std_percent_of_distance / 100.0) * separation_km
        x_meas = x_true.copy()
        # Apply requested distance-scaled noise to position terms.
        x_meas[:3] += self._rng.normal(0.0, sigma_km, size=3)
        return x_meas, True
