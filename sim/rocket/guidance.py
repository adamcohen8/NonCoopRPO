from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.rocket.models import GuidanceCommand, RocketGuidanceLaw, RocketSimConfig, RocketState, RocketVehicleConfig
from sim.utils.quaternion import dcm_to_quaternion_bn


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _quat_from_body_x_and_hint(x_axis_eci: np.ndarray, z_hint_eci: np.ndarray) -> np.ndarray:
    x_hat = _unit(np.array(x_axis_eci, dtype=float))
    if np.linalg.norm(x_hat) <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    z_hint = _unit(np.array(z_hint_eci, dtype=float))
    y_hat = _unit(np.cross(z_hint, x_hat))
    if np.linalg.norm(y_hat) <= 0.0:
        # fallback if collinear
        y_hat = _unit(np.cross(np.array([0.0, 0.0, 1.0]), x_hat))
        if np.linalg.norm(y_hat) <= 0.0:
            y_hat = np.array([0.0, 1.0, 0.0])
    z_hat = _unit(np.cross(x_hat, y_hat))
    c_bn = np.vstack((x_hat, y_hat, z_hat))  # body rows in inertial components
    return dcm_to_quaternion_bn(c_bn)


@dataclass(frozen=True)
class OpenLoopPitchProgramGuidance(RocketGuidanceLaw):
    """Simple launch guidance: hold vertical, then pitch over and follow velocity direction."""

    vertical_hold_s: float = 10.0
    pitch_start_s: float = 10.0
    pitch_end_s: float = 180.0
    pitch_final_deg: float = 70.0
    max_throttle: float = 1.0
    min_throttle: float = 0.0

    def command(self, state: RocketState, sim_cfg: RocketSimConfig, vehicle_cfg: RocketVehicleConfig) -> GuidanceCommand:
        t = state.t_s
        r_hat = _unit(state.position_eci_km)
        v_hat = _unit(state.velocity_eci_km_s)
        east_hat = _unit(np.cross(np.array([0.0, 0.0, 1.0]), r_hat))
        if np.linalg.norm(east_hat) <= 0.0:
            east_hat = np.array([0.0, 1.0, 0.0])

        if t <= self.vertical_hold_s:
            x_cmd = r_hat
        elif t <= self.pitch_end_s:
            alpha = float(np.clip((t - self.pitch_start_s) / max(self.pitch_end_s - self.pitch_start_s, 1e-9), 0.0, 1.0))
            pitch_rad = np.deg2rad(alpha * self.pitch_final_deg)
            x_cmd = _unit(np.cos(pitch_rad) * r_hat + np.sin(pitch_rad) * east_hat)
        else:
            if np.linalg.norm(v_hat) > 0.0:
                x_cmd = _unit(v_hat)
            else:
                x_cmd = r_hat

        q_cmd = _quat_from_body_x_and_hint(x_cmd, z_hint_eci=r_hat)
        thr = float(np.clip(self.max_throttle, self.min_throttle, self.max_throttle))
        return GuidanceCommand(throttle=thr, attitude_quat_bn_cmd=q_cmd, torque_body_nm_cmd=np.zeros(3))


@dataclass(frozen=True)
class HoldAttitudeGuidance(RocketGuidanceLaw):
    throttle: float = 1.0

    def command(self, state: RocketState, sim_cfg: RocketSimConfig, vehicle_cfg: RocketVehicleConfig) -> GuidanceCommand:
        return GuidanceCommand(throttle=float(np.clip(self.throttle, 0.0, 1.0)), attitude_quat_bn_cmd=None, torque_body_nm_cmd=np.zeros(3))
