from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.dynamics.orbit.atmosphere import atmosphere_state_from_model
from sim.rocket.models import GuidanceCommand, RocketGuidanceLaw, RocketSimConfig, RocketState, RocketVehicleConfig
from sim.utils.quaternion import dcm_to_quaternion_bn, quaternion_to_dcm_bn


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


@dataclass(frozen=True)
class MaxQThrottleLimiterGuidance(RocketGuidanceLaw):
    """Wrap a base guidance law and limit throttle when dynamic pressure exceeds max_q."""

    base_guidance: RocketGuidanceLaw
    max_q_pa: float = 45_000.0
    min_throttle: float = 0.0

    def _estimate_dynamic_pressure_pa(self, state: RocketState, sim_cfg: RocketSimConfig) -> float:
        env = {"atmosphere_model": sim_cfg.atmosphere_model, **dict(sim_cfg.atmosphere_env)}
        atmos = atmosphere_state_from_model(
            model=str(sim_cfg.atmosphere_model).lower(),
            r_eci_km=state.position_eci_km,
            t_s=state.t_s,
            env=env,
        )
        rho = float(max(atmos["density_kg_m3"], 0.0))
        c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn)
        omega_earth = np.array([0.0, 0.0, 7.2921159e-5], dtype=float)
        v_atm_eci_km_s = np.cross(omega_earth, state.position_eci_km)
        v_rel_eci_m_s = (state.velocity_eci_km_s - v_atm_eci_km_s) * 1e3
        v_rel_body_m_s = c_bn @ v_rel_eci_m_s
        speed = float(np.linalg.norm(v_rel_body_m_s))
        return 0.5 * rho * speed * speed

    def command(self, state: RocketState, sim_cfg: RocketSimConfig, vehicle_cfg: RocketVehicleConfig) -> GuidanceCommand:
        cmd = self.base_guidance.command(state, sim_cfg, vehicle_cfg)
        thr_cmd = float(np.clip(cmd.throttle, 0.0, 1.0))
        if self.max_q_pa <= 0.0 or thr_cmd <= 0.0:
            return cmd

        q_now = self._estimate_dynamic_pressure_pa(state=state, sim_cfg=sim_cfg)
        if q_now <= self.max_q_pa:
            return cmd

        scale = float(np.clip(self.max_q_pa / max(q_now, 1e-9), 0.0, 1.0))
        thr_limited = float(np.clip(thr_cmd * scale, self.min_throttle, thr_cmd))
        return GuidanceCommand(
            throttle=thr_limited,
            attitude_quat_bn_cmd=cmd.attitude_quat_bn_cmd,
            torque_body_nm_cmd=cmd.torque_body_nm_cmd,
        )
