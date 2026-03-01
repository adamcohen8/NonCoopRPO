from __future__ import annotations

from dataclasses import dataclass, field

from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief


@dataclass
class AoITrackingEstimator(Estimator):
    base_estimator: Estimator
    age_of_information_s: float = 0.0
    _last_measurement_t_s: float = field(default=-1.0)

    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        if measurement is not None:
            self._last_measurement_t_s = measurement.t_s
        if self._last_measurement_t_s >= 0.0:
            self.age_of_information_s = max(0.0, t_s - self._last_measurement_t_s)
        new_belief = self.base_estimator.update(belief, measurement, t_s)
        return new_belief
