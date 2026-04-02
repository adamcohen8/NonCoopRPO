from __future__ import annotations

from dataclasses import dataclass

from sim.core.models import Command


@dataclass
class CombinedActuator:
    orbital: object
    attitude: object

    def apply(self, command: Command, limits: dict, dt_s: float) -> Command:
        c_att = self.attitude.apply(command, limits, dt_s)
        return self.orbital.apply(c_att, limits, dt_s)
