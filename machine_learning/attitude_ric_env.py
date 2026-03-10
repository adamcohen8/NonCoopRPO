from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from presets import BASIC_REACTION_WHEEL_TRIAD, BASIC_SATELLITE, build_sim_object_from_presets
from sim.core.models import Command
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import (
    dcm_to_quaternion_bn,
    normalize_quaternion,
    quaternion_delta_from_body_rate,
    quaternion_multiply,
    quaternion_to_dcm_bn,
)


def _rot_x(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, ca, sa], [0.0, -sa, ca]])


def _rot_y(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])


def _rot_z(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])


def _quat_err_deg(q_target: np.ndarray, q_current: np.ndarray) -> float:
    qt = normalize_quaternion(np.array(q_target, dtype=float))
    qc = normalize_quaternion(np.array(q_current, dtype=float))
    qtc = np.array([qt[0], -qt[1], -qt[2], -qt[3]])
    qe = quaternion_multiply(qtc, qc)
    if qe[0] < 0.0:
        qe *= -1.0
    return float(np.rad2deg(2.0 * np.arccos(np.clip(qe[0], -1.0, 1.0))))


@dataclass(frozen=True)
class AttitudeRICRLConfig:
    dt_s: float = 0.5
    attitude_substep_s: float = 0.01
    episode_duration_s: float = 100.0
    seed: int = 0
    init_error_max_deg: float = 120.0
    max_init_rate_ric_rad_s: float = 0.0
    wheel_speed_inertia_kg_m2: float = 0.02
    kp_reward_angle: float = 0.01
    kp_reward_rate: float = 10.0
    kp_reward_wheel_speed: float = 1e-3
    kp_reward_torque: float = 1e-2
    success_angle_deg: float = 2.0
    success_rate_rad_s: float = np.deg2rad(0.2)


class AttitudeRICRLEnv:
    """RL env for 3-axis RIC attitude control with reaction wheels."""

    def __init__(self, cfg: AttitudeRICRLConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.sat = None
        self.target_ric_euler_rad = np.zeros(3)
        self.target_q_br = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.steps_max = max(1, int(np.ceil(cfg.episode_duration_s / cfg.dt_s)))
        self.step_idx = 0
        self._last_err_deg = 0.0

    @property
    def obs_dim(self) -> int:
        return 10  # q_br(4), w_br_ric(3), wheel_speed(3)

    @property
    def action_dim(self) -> int:
        return 3  # wheel commands

    def sample_new_target_for_epoch(self) -> np.ndarray:
        # RIC convention requested by user:
        # yaw about radial (R), roll about in-track (I), pitch about cross-track (C).
        yaw_r = self.rng.uniform(-np.deg2rad(40.0), np.deg2rad(40.0))
        roll_i = self.rng.uniform(-np.deg2rad(40.0), np.deg2rad(40.0))
        pitch_c = self.rng.uniform(-np.deg2rad(40.0), np.deg2rad(40.0))
        self.target_ric_euler_rad = np.array([yaw_r, roll_i, pitch_c], dtype=float)
        c_br = _rot_z(pitch_c) @ _rot_y(roll_i) @ _rot_x(yaw_r)
        self.target_q_br = normalize_quaternion(dcm_to_quaternion_bn(c_br))
        return self.target_ric_euler_rad.copy()

    def reset(self) -> np.ndarray:
        dt_s = float(self.cfg.dt_s)
        self.sat = build_sim_object_from_presets(
            object_id="ric_rl_sat",
            dt_s=dt_s,
            satellite=BASIC_SATELLITE,
            enable_disturbances=False,
            enable_attitude_knowledge=True,
            attitude_substep_s=self.cfg.attitude_substep_s,
        )

        if np.allclose(self.target_q_br, np.array([1.0, 0.0, 0.0, 0.0])):
            self.sample_new_target_for_epoch()

        # Build initial RIC attitude offset around epoch target.
        axis = self.rng.normal(0.0, 1.0, size=3)
        axis /= max(np.linalg.norm(axis), 1e-12)
        ang = self.rng.uniform(0.0, np.deg2rad(self.cfg.init_error_max_deg))
        dq = quaternion_delta_from_body_rate(axis, 2.0 * ang)  # helper expects omega*dt -> set product directly
        q_br0 = normalize_quaternion(quaternion_multiply(dq, self.target_q_br))

        r = self.sat.truth.position_eci_km
        v = self.sat.truth.velocity_eci_km_s
        c_ir = ric_dcm_ir_from_rv(r, v)
        c_ri = c_ir.T
        c_br0 = quaternion_to_dcm_bn(q_br0)
        q_bn0 = normalize_quaternion(dcm_to_quaternion_bn(c_br0 @ c_ri))
        self.sat.truth.attitude_quat_bn = q_bn0

        # Initial RIC rates near zero by user request; map to inertial body rate.
        w_br_ric0 = self.rng.uniform(
            -self.cfg.max_init_rate_ric_rad_s,
            self.cfg.max_init_rate_ric_rad_s,
            size=3,
        )
        h = np.cross(r, v)
        omega_ri_eci = h / max(np.linalg.norm(r) ** 2, 1e-12)
        omega_ri_ric = c_ri @ omega_ri_eci
        w_bi_body0 = c_br0 @ (w_br_ric0 + omega_ri_ric)
        self.sat.truth.angular_rate_body_rad_s = w_bi_body0
        self.sat.belief.state[6:10] = q_bn0
        self.sat.belief.state[10:13] = w_bi_body0

        if hasattr(self.sat.actuator, "attitude") and hasattr(self.sat.actuator.attitude, "wheel_momentum_nms"):
            self.sat.actuator.attitude.wheel_momentum_nms = np.zeros(3)

        self.step_idx = 0
        self._last_err_deg = self._attitude_error_deg()
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        a = np.clip(np.array(action, dtype=float).reshape(3), -1.0, 1.0)
        rw_lim = np.array(self.sat.actuator.attitude.reaction_wheels.max_torque_nm, dtype=float)
        tau_cmd = a * rw_lim

        cmd = Command(thrust_eci_km_s2=np.zeros(3), torque_body_nm=tau_cmd, mode_flags={"mode": "rl_ric_pd"})
        applied = self.sat.actuator.apply(cmd, self.sat.limits, self.cfg.dt_s)
        self.sat.truth = self.sat.dynamics.step(self.sat.truth, applied, env={}, dt_s=self.cfg.dt_s)
        self.step_idx += 1

        # Ideal fast-refresh estimate for now.
        self.sat.belief.state[6:10] = self.sat.truth.attitude_quat_bn
        self.sat.belief.state[10:13] = self.sat.truth.angular_rate_body_rad_s
        self.sat.belief.last_update_t_s = self.sat.truth.t_s

        err_deg = self._attitude_error_deg()
        w_br_ric = self._w_br_ric()
        wheel_speed = self._wheel_speed_rad_s()

        # Dense shaping + improvement reward + terminal penalty.
        reward = (self._last_err_deg - err_deg) * 0.2
        reward -= self.cfg.kp_reward_angle * (err_deg**2)
        reward -= self.cfg.kp_reward_rate * float(np.dot(w_br_ric, w_br_ric))
        reward -= self.cfg.kp_reward_wheel_speed * float(np.dot(wheel_speed, wheel_speed))
        reward -= self.cfg.kp_reward_torque * float(np.dot(a, a))
        self._last_err_deg = err_deg

        done = self.step_idx >= self.steps_max
        success = bool(err_deg <= self.cfg.success_angle_deg and np.linalg.norm(w_br_ric) <= self.cfg.success_rate_rad_s)
        if done:
            reward += -0.1 * err_deg
            if success:
                reward += 10.0

        info = {
            "attitude_error_deg": float(err_deg),
            "rate_norm_rad_s": float(np.linalg.norm(w_br_ric)),
            "wheel_speed_norm_rad_s": float(np.linalg.norm(wheel_speed)),
            "success": success,
            "target_ric_euler_rad": self.target_ric_euler_rad.tolist(),
        }
        return self._obs(), float(reward), bool(done), info

    def _obs(self) -> np.ndarray:
        q_br = self._q_br()
        if q_br[0] < 0.0:
            q_br = -q_br
        return np.hstack((q_br, self._w_br_ric(), self._wheel_speed_rad_s())).astype(np.float32)

    def _q_br(self) -> np.ndarray:
        r = self.sat.truth.position_eci_km
        v = self.sat.truth.velocity_eci_km_s
        c_ir = ric_dcm_ir_from_rv(r, v)
        c_bn = quaternion_to_dcm_bn(self.sat.truth.attitude_quat_bn)
        c_br = c_bn @ c_ir
        return normalize_quaternion(dcm_to_quaternion_bn(c_br))

    def _w_br_ric(self) -> np.ndarray:
        r = self.sat.truth.position_eci_km
        v = self.sat.truth.velocity_eci_km_s
        c_ir = ric_dcm_ir_from_rv(r, v)
        c_ri = c_ir.T
        c_bn = quaternion_to_dcm_bn(self.sat.truth.attitude_quat_bn)
        c_br = c_bn @ c_ir
        h = np.cross(r, v)
        omega_ri_eci = h / max(np.linalg.norm(r) ** 2, 1e-12)
        omega_ri_ric = c_ri @ omega_ri_eci
        omega_bi_body = self.sat.truth.angular_rate_body_rad_s
        return c_br.T @ omega_bi_body - omega_ri_ric

    def _attitude_error_deg(self) -> float:
        return _quat_err_deg(self.target_q_br, self._q_br())

    def _wheel_speed_rad_s(self) -> np.ndarray:
        if hasattr(self.sat.actuator, "attitude") and hasattr(self.sat.actuator.attitude, "wheel_momentum_nms"):
            h = np.array(self.sat.actuator.attitude.wheel_momentum_nms, dtype=float)
            return h / max(float(self.cfg.wheel_speed_inertia_kg_m2), 1e-9)
        return np.zeros(3)
