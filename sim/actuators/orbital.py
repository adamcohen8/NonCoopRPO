from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.core.interfaces import Actuator
from sim.core.models import Command
from sim.utils.quaternion import quaternion_to_dcm_bn


def _unit(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.array(vec, dtype=float).reshape(3)
    mag = float(np.linalg.norm(arr))
    if mag <= eps:
        return np.zeros(3, dtype=float)
    return arr / mag


def attitude_coupled_thrust_eci(
    commanded_accel_eci_km_s2: np.ndarray,
    *,
    attitude_quat_bn: np.ndarray,
    thruster_direction_body: np.ndarray,
) -> np.ndarray:
    accel_cmd = np.array(commanded_accel_eci_km_s2, dtype=float).reshape(3)
    accel_mag = float(np.linalg.norm(accel_cmd))
    if accel_mag <= 0.0:
        return np.zeros(3, dtype=float)
    plume_axis_body = _unit(np.array(thruster_direction_body, dtype=float).reshape(3))
    if float(np.linalg.norm(plume_axis_body)) <= 0.0:
        return accel_cmd
    c_bn = quaternion_to_dcm_bn(np.array(attitude_quat_bn, dtype=float).reshape(4))
    plume_axis_eci = c_bn.T @ plume_axis_body
    # The stored mount axis is the nozzle / plume direction, so vehicle force is opposite it.
    return -accel_mag * plume_axis_eci


def thruster_disturbance_torque_body_nm(
    applied_accel_eci_km_s2: np.ndarray,
    *,
    current_mass_kg: float,
    thruster_direction_body: np.ndarray,
    thruster_position_body_m: np.ndarray,
) -> np.ndarray:
    accel_mag_m_s2 = float(np.linalg.norm(np.array(applied_accel_eci_km_s2, dtype=float).reshape(3)) * 1e3)
    if accel_mag_m_s2 <= 0.0 or current_mass_kg <= 0.0:
        return np.zeros(3, dtype=float)
    plume_axis_body = _unit(np.array(thruster_direction_body, dtype=float).reshape(3))
    if float(np.linalg.norm(plume_axis_body)) <= 0.0:
        return np.zeros(3, dtype=float)
    mount_position_body_m = np.array(thruster_position_body_m, dtype=float).reshape(3)
    force_body_n = -float(current_mass_kg) * accel_mag_m_s2 * plume_axis_body
    return np.cross(mount_position_body_m, force_body_n)


@dataclass(frozen=True)
class OrbitalActuatorLimits:
    max_accel_km_s2: float
    min_impulse_bit_km_s: float = 0.0
    max_throttle_rate_km_s2_s: float = 1e-6
    isp_s: float = 220.0
    thruster_direction_body: np.ndarray | None = None
    thruster_position_body_m: np.ndarray | None = None
    couple_to_attitude: bool = True


@dataclass
class OrbitalActuator(Actuator):
    lag_tau_s: float = 0.0
    _last_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def apply(self, command: Command, limits: dict, dt_s: float) -> Command:
        lim: OrbitalActuatorLimits = limits["orbital"]
        accel_filtered = np.array(command.thrust_eci_km_s2, dtype=float)
        torque_applied = np.array(command.torque_body_nm, dtype=float)
        mode_flags = dict(command.mode_flags)
        thruster_direction_body = mode_flags.get("thruster_direction_body", lim.thruster_direction_body)
        thruster_position_body_m = mode_flags.get("thruster_position_body_m", lim.thruster_position_body_m)
        current_mass_kg = float(mode_flags.get("current_mass_kg", mode_flags.get("mass_kg", 0.0)))

        norm = np.linalg.norm(accel_filtered)
        if norm > lim.max_accel_km_s2 > 0.0:
            accel_filtered *= lim.max_accel_km_s2 / norm

        max_delta = lim.max_throttle_rate_km_s2_s * dt_s
        delta = accel_filtered - self._last_accel
        delta_norm = np.linalg.norm(delta)
        if delta_norm > max_delta > 0.0:
            accel_filtered = self._last_accel + delta * (max_delta / delta_norm)

        if self.lag_tau_s > 0.0:
            alpha = min(1.0, dt_s / self.lag_tau_s)
            accel_filtered = self._last_accel + alpha * (accel_filtered - self._last_accel)

        dv = float(np.linalg.norm(accel_filtered) * dt_s)
        if 0.0 < dv < lim.min_impulse_bit_km_s:
            accel_filtered = np.zeros(3)

        self._last_accel = accel_filtered.copy()
        accel_applied = accel_filtered.copy()
        if bool(lim.couple_to_attitude):
            attitude_quat_bn = mode_flags.get("current_attitude_quat_bn")
            if thruster_direction_body is not None and attitude_quat_bn is not None:
                accel_applied = attitude_coupled_thrust_eci(
                    accel_filtered,
                    attitude_quat_bn=np.array(attitude_quat_bn, dtype=float),
                    thruster_direction_body=np.array(thruster_direction_body, dtype=float),
                )
        g0_m_s2 = 9.80665
        accel_mag_m_s2 = float(np.linalg.norm(accel_applied) * 1e3)
        thrust_n = max(current_mass_kg, 0.0) * accel_mag_m_s2
        mdot_kg_s = 0.0 if lim.isp_s <= 0.0 or thrust_n <= 0.0 else thrust_n / (lim.isp_s * g0_m_s2)
        mode_flags["delta_mass_kg"] = float(mdot_kg_s * dt_s)
        thruster_torque_body_nm = np.zeros(3, dtype=float)
        if thruster_direction_body is not None and thruster_position_body_m is not None:
            thruster_torque_body_nm = thruster_disturbance_torque_body_nm(
                accel_applied,
                current_mass_kg=current_mass_kg,
                thruster_direction_body=np.array(thruster_direction_body, dtype=float),
                thruster_position_body_m=np.array(thruster_position_body_m, dtype=float),
            )
            torque_applied = torque_applied + thruster_torque_body_nm
        mode_flags["thruster_torque_body_nm"] = thruster_torque_body_nm.tolist()
        return Command(thrust_eci_km_s2=accel_applied, torque_body_nm=torque_applied, mode_flags=mode_flags)
