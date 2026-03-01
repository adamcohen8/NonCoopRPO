from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import SensorModel
from sim.core.models import Measurement, StateTruth


@dataclass
class CompositeSensorModel(SensorModel):
    sensors: list[SensorModel]

    def measure(self, truth: StateTruth, env: dict, t_s: float) -> Measurement | None:
        parts = []
        for sensor in self.sensors:
            m = sensor.measure(truth, env, t_s)
            if m is not None:
                parts.append(m.vector)
        if not parts:
            return None
        return Measurement(vector=np.concatenate(parts), t_s=t_s)
