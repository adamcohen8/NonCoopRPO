from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Actuator
from sim.core.models import Command


@dataclass(frozen=True)
class OrbitalActuatorLimits:
    max_accel_km_s2: float
    min_impulse_bit_km_s: float = 0.0
    max_throttle_rate_km_s2_s: float = 1e-6
    isp_s: float = 220.0


@dataclass
class OrbitalActuator(Actuator):
    lag_tau_s: float = 0.0
    _last_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def apply(self, command: Command, limits: dict, dt_s: float) -> Command:
        lim: OrbitalActuatorLimits = limits["orbital"]
        accel = np.array(command.thrust_eci_km_s2, dtype=float)

        norm = np.linalg.norm(accel)
        if norm > lim.max_accel_km_s2 > 0.0:
            accel *= lim.max_accel_km_s2 / norm

        max_delta = lim.max_throttle_rate_km_s2_s * dt_s
        delta = accel - self._last_accel
        delta_norm = np.linalg.norm(delta)
        if delta_norm > max_delta > 0.0:
            accel = self._last_accel + delta * (max_delta / delta_norm)

        if self.lag_tau_s > 0.0:
            alpha = min(1.0, dt_s / self.lag_tau_s)
            accel = self._last_accel + alpha * (accel - self._last_accel)

        dv = float(np.linalg.norm(accel) * dt_s)
        if 0.0 < dv < lim.min_impulse_bit_km_s:
            accel = np.zeros(3)

        self._last_accel = accel
        g0_m_s2 = 9.80665
        thrust_m_s2 = np.linalg.norm(accel) * 1e3
        mdot_kg_s = 0.0 if lim.isp_s <= 0.0 else thrust_m_s2 / (lim.isp_s * g0_m_s2)
        mode_flags = dict(command.mode_flags)
        mode_flags["delta_mass_kg"] = float(mdot_kg_s * dt_s)
        return Command(thrust_eci_km_s2=accel, torque_body_nm=np.array(command.torque_body_nm), mode_flags=mode_flags)
