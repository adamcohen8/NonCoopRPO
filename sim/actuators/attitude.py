from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Actuator
from sim.core.models import Command


@dataclass(frozen=True)
class ReactionWheelLimits:
    max_torque_nm: np.ndarray
    max_momentum_nms: np.ndarray


@dataclass(frozen=True)
class MagnetorquerLimits:
    max_dipole_a_m2: np.ndarray


@dataclass(frozen=True)
class ThrusterPulseLimits:
    max_torque_nm: np.ndarray
    pulse_quantum_s: float = 0.02


@dataclass
class AttitudeActuator(Actuator):
    reaction_wheels: ReactionWheelLimits | None = None
    magnetorquers: MagnetorquerLimits | None = None
    thruster_pulse: ThrusterPulseLimits | None = None
    wheel_momentum_nms: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def apply(self, command: Command, limits: dict, dt_s: float) -> Command:
        torque = np.array(command.torque_body_nm, dtype=float)

        if self.reaction_wheels is not None:
            rw = self.reaction_wheels
            torque = np.clip(torque, -rw.max_torque_nm, rw.max_torque_nm)
            self.wheel_momentum_nms = np.clip(
                self.wheel_momentum_nms + torque * dt_s,
                -rw.max_momentum_nms,
                rw.max_momentum_nms,
            )
            sat = np.abs(self.wheel_momentum_nms) >= rw.max_momentum_nms
            torque[sat] = 0.0

        if self.thruster_pulse is not None:
            tp = self.thruster_pulse
            torque = np.clip(torque, -tp.max_torque_nm, tp.max_torque_nm)
            if tp.pulse_quantum_s > 0.0:
                pulses = np.round(dt_s / tp.pulse_quantum_s)
                scale = 0.0 if pulses <= 0 else pulses * tp.pulse_quantum_s / dt_s
                torque *= scale

        # Magnetorquer coupling to geomagnetic field would require B-field in env.
        # Here we enforce achievable moment command proxy via clamp.
        if self.magnetorquers is not None:
            mt = self.magnetorquers
            torque = np.clip(torque, -np.abs(mt.max_dipole_a_m2), np.abs(mt.max_dipole_a_m2))

        return Command(
            thrust_eci_km_s2=np.array(command.thrust_eci_km_s2, dtype=float),
            torque_body_nm=torque,
            mode_flags=dict(command.mode_flags),
        )
