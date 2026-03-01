from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Actuator
from sim.core.models import Command


@dataclass(frozen=True)
class ActuatorLimits:
    max_accel_km_s2: float
    max_torque_nm: float


@dataclass
class SimpleActuator(Actuator):
    lag_tau_s: float = 0.0
    _last_applied: Command = field(default_factory=Command.zero)

    def apply(self, command: Command, limits: dict, dt_s: float) -> Command:
        lim: ActuatorLimits = limits["actuator"]

        thrust = np.array(command.thrust_eci_km_s2, dtype=float)
        torque = np.array(command.torque_body_nm, dtype=float)

        thrust_norm = np.linalg.norm(thrust)
        if thrust_norm > lim.max_accel_km_s2 and thrust_norm > 0.0:
            thrust = thrust * (lim.max_accel_km_s2 / thrust_norm)

        torque_norm = np.linalg.norm(torque)
        if torque_norm > lim.max_torque_nm and torque_norm > 0.0:
            torque = torque * (lim.max_torque_nm / torque_norm)

        if self.lag_tau_s > 0.0:
            alpha = min(1.0, dt_s / self.lag_tau_s)
            thrust = self._last_applied.thrust_eci_km_s2 + alpha * (thrust - self._last_applied.thrust_eci_km_s2)
            torque = self._last_applied.torque_body_nm + alpha * (torque - self._last_applied.torque_body_nm)

        applied = Command(thrust_eci_km_s2=thrust, torque_body_nm=torque, mode_flags=dict(command.mode_flags))
        self._last_applied = applied
        return applied
