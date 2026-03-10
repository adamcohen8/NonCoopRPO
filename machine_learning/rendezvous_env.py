from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from presets import BASIC_CHEMICAL_BOTTOM_Z, BASIC_SATELLITE, build_sim_object_from_presets
from sim.core.models import Command
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.knowledge import (
    KnowledgeConditionConfig,
    KnowledgeEKFConfig,
    KnowledgeNoiseConfig,
    ObjectKnowledgeBase,
    TrackedObjectConfig,
)
from sim.utils.quaternion import quaternion_to_dcm_bn


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _ric_rect_to_eci(x_host_eci: np.ndarray, x_rel_rect: np.ndarray) -> np.ndarray:
    r_host = x_host_eci[:3]
    v_host = x_host_eci[3:]
    xr = x_rel_rect[:3]
    xv = x_rel_rect[3:]
    h = np.cross(r_host, v_host)
    in_vec = np.cross(h, r_host)
    rsw = np.column_stack((_unit(r_host), _unit(in_vec), _unit(h)))
    dr = np.linalg.inv(rsw.T) @ xr
    rtemp = np.cross(h, v_host)
    vtemp = np.cross(h, r_host)
    drsw = np.column_stack((v_host / max(np.linalg.norm(r_host), 1e-12), rtemp / max(np.linalg.norm(vtemp), 1e-12), np.zeros(3)))
    frame_mv = np.array(
        [
            xr[0] * (r_host @ v_host) / (max(np.linalg.norm(r_host), 1e-12) ** 2),
            xr[1] * (vtemp @ rtemp) / (max(np.linalg.norm(vtemp), 1e-12) ** 2),
            0.0,
        ]
    )
    dv = np.linalg.inv(rsw.T) @ (xv + frame_mv - (drsw.T @ dr))
    return np.hstack((r_host + dr, v_host + dv))


@dataclass(frozen=True)
class RLRendezvousConfig:
    dt_s: float = 1.0
    orbits_per_episode: int = 1
    capture_radius_m: float = 5.0
    seed: int = 0
    init_rel_ric_rect: np.ndarray = field(default_factory=lambda: np.array([2.0, -8.0, 1.2, 0.0008, -0.0012, 0.0004]))
    knowledge_refresh_s: float = 1.0
    knowledge_max_range_km: float | None = 5000.0
    knowledge_require_los: bool = True
    knowledge_dropout_prob: float = 0.0
    target_pos_sigma_km: np.ndarray = field(default_factory=lambda: np.array([5e-3, 5e-3, 5e-3]))
    target_vel_sigma_km_s: np.ndarray = field(default_factory=lambda: np.array([5e-5, 5e-5, 5e-5]))
    target_ekf_process_diag: np.ndarray = field(default_factory=lambda: np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10]))
    target_ekf_meas_diag: np.ndarray = field(default_factory=lambda: np.array([2.5e-5, 2.5e-5, 2.5e-5, 2.5e-9, 2.5e-9, 2.5e-9]))
    target_ekf_init_cov_diag: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-2]))
    reward_mode: str = "lookahead_terminal_closest"
    lookahead_horizon_s: float = 600.0
    lookahead_dt_s: float = 10.0

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        if self.orbits_per_episode < 1:
            raise ValueError("orbits_per_episode must be >= 1.")
        if self.orbits_per_episode > 10:
            raise ValueError("orbits_per_episode must be <= 10.")


class RLRendezvousEnv:
    """Two-body, no-perturbation training environment for NN chaser control."""

    def __init__(self, config: RLRendezvousConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.capture_radius_km = float(config.capture_radius_m) / 1000.0
        self.max_steps = 1
        self.step_count = 0
        self.episode_duration_s_override: float | None = None

        self.chief = None
        self.chaser = None
        self.knowledge = None
        self._target_est_eci = np.zeros(6)
        self._closest_range_km = np.inf
        self._prev_min_range_km = np.inf
        self._prev_lookahead_closest_km = np.inf
        self._done = False
        self._init_mass = BASIC_SATELLITE.wet_mass_kg
        self._dry_mass = BASIC_SATELLITE.dry_mass_kg

    @property
    def obs_dim(self) -> int:
        return 20

    @property
    def action_dim(self) -> int:
        return 4

    def reset(self) -> np.ndarray:
        dt_s = float(self.cfg.dt_s)
        self.chief = build_sim_object_from_presets(
            object_id="chief_rl",
            dt_s=dt_s,
            satellite=BASIC_SATELLITE,
            enable_disturbances=False,
            enable_attitude_knowledge=False,
        )
        self.chaser = build_sim_object_from_presets(
            object_id="chaser_rl",
            dt_s=dt_s,
            satellite=BASIC_SATELLITE,
            enable_disturbances=False,
            enable_attitude_knowledge=True,
        )

        x_rel0 = np.array(self.cfg.init_rel_ric_rect, dtype=float).reshape(6)
        x_chief = np.hstack((self.chief.truth.position_eci_km, self.chief.truth.velocity_eci_km_s))
        x_chaser = _ric_rect_to_eci(x_chief, x_rel0)
        self.chaser.truth.position_eci_km = x_chaser[:3]
        self.chaser.truth.velocity_eci_km_s = x_chaser[3:]
        self.chaser.belief.state[:6] = x_chaser
        self.chaser.truth.mass_kg = float(self._init_mass)

        self.knowledge = ObjectKnowledgeBase(
            observer_id="chaser_rl",
            dt_s=dt_s,
            mu_km3_s2=EARTH_MU_KM3_S2,
            rng=np.random.default_rng(int(self.rng.integers(0, 2**31 - 1))),
            tracked_objects=[
                TrackedObjectConfig(
                    target_id="chief_rl",
                    conditions=KnowledgeConditionConfig(
                        refresh_rate_s=float(self.cfg.knowledge_refresh_s),
                        max_range_km=self.cfg.knowledge_max_range_km,
                        require_line_of_sight=bool(self.cfg.knowledge_require_los),
                        dropout_prob=float(self.cfg.knowledge_dropout_prob),
                    ),
                    sensor_noise=KnowledgeNoiseConfig(
                        pos_sigma_km=np.array(self.cfg.target_pos_sigma_km, dtype=float),
                        vel_sigma_km_s=np.array(self.cfg.target_vel_sigma_km_s, dtype=float),
                    ),
                    estimator="ekf",
                    ekf=KnowledgeEKFConfig(
                        process_noise_diag=np.array(self.cfg.target_ekf_process_diag, dtype=float),
                        meas_noise_diag=np.array(self.cfg.target_ekf_meas_diag, dtype=float),
                        init_cov_diag=np.array(self.cfg.target_ekf_init_cov_diag, dtype=float),
                    ),
                )
            ],
        )

        self._refresh_estimates(t_s=0.0)
        if self.episode_duration_s_override is not None:
            duration_s = max(dt_s, float(self.episode_duration_s_override))
        else:
            period_s = 2.0 * np.pi * np.sqrt((np.linalg.norm(self.chief.truth.position_eci_km) ** 3) / EARTH_MU_KM3_S2)
            duration_s = period_s * int(self.cfg.orbits_per_episode)
        self.max_steps = int(np.ceil(duration_s / dt_s))
        self.step_count = 0
        self._closest_range_km = self._range_km()
        self._prev_min_range_km = self._closest_range_km
        self._prev_lookahead_closest_km = self._predict_closest_range_km_over_horizon()
        self._done = False
        return self._observation()

    def set_episode_duration_s(self, duration_s: float | None) -> None:
        if duration_s is None:
            self.episode_duration_s_override = None
            return
        self.episode_duration_s_override = max(float(self.cfg.dt_s), float(duration_s))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        if self._done:
            return self._observation(), 0.0, True, self._info()

        a = np.array(action, dtype=float).reshape(4)
        torque_norm = np.clip(a[:3], -1.0, 1.0)
        thrust_on = bool(a[3] > 0.5)
        dt_s = float(self.cfg.dt_s)

        rw_lim = np.array(self.chaser.actuator.attitude.reaction_wheels.max_torque_nm, dtype=float)
        torque_cmd = torque_norm * rw_lim

        thrust_cmd = np.zeros(3)
        if thrust_on and self.chaser.truth.mass_kg > self._dry_mass + 1e-9:
            max_accel = float(self.chaser.limits["orbital"].max_accel_km_s2)
            c_bn = quaternion_to_dcm_bn(self.chaser.truth.attitude_quat_bn)
            axis_eci = c_bn.T @ _unit(BASIC_CHEMICAL_BOTTOM_Z.mount.thrust_direction_body)
            thrust_cmd = -max_accel * axis_eci

        cmd_chaser = Command(thrust_eci_km_s2=thrust_cmd, torque_body_nm=torque_cmd, mode_flags={"mode": "rl"})
        cmd_chief = Command.zero()
        app_chaser = self.chaser.actuator.apply(cmd_chaser, self.chaser.limits, dt_s)
        app_chief = self.chief.actuator.apply(cmd_chief, self.chief.limits, dt_s)
        self.chaser.truth = self.chaser.dynamics.step(self.chaser.truth, app_chaser, env={}, dt_s=dt_s)
        self.chief.truth = self.chief.dynamics.step(self.chief.truth, app_chief, env={}, dt_s=dt_s)
        self.step_count += 1

        t_now = self.chaser.truth.t_s
        self._refresh_estimates(t_s=t_now)
        r_now = self._range_km()
        self._closest_range_km = min(self._closest_range_km, r_now)
        reward = float((self._prev_min_range_km - self._closest_range_km) * 1e3)
        self._prev_min_range_km = self._closest_range_km

        capture = r_now <= self.capture_radius_km
        timeout = self.step_count >= self.max_steps
        self._done = bool(capture or timeout)
        if self.cfg.reward_mode == "terminal_closest":
            reward = 0.0
            if self._done:
                reward = -float(self._closest_range_km * 1e3)
        elif self.cfg.reward_mode == "lookahead_terminal_closest":
            lookahead_closest_km = self._predict_closest_range_km_over_horizon()
            reward = float((self._prev_lookahead_closest_km - lookahead_closest_km) * 1e3)
            self._prev_lookahead_closest_km = lookahead_closest_km
            if self._done:
                reward += -float(lookahead_closest_km * 1e3)
        return self._observation(), reward, self._done, self._info(capture=capture, timeout=timeout)

    def _refresh_estimates(self, t_s: float) -> None:
        world_truth = {"chief_rl": self.chief.truth, "chaser_rl": self.chaser.truth}
        meas = self.chaser.sensor.measure(self.chaser.truth, {"world_truth": world_truth}, t_s=t_s)
        self.chaser.belief = self.chaser.estimator.update(self.chaser.belief, meas, t_s=t_s)
        self.knowledge.update(observer_truth=self.chaser.truth, world_truth=world_truth, t_s=t_s)
        snap = self.knowledge.snapshot().get("chief_rl")
        if snap is not None:
            self._target_est_eci = np.array(snap.state[:6], dtype=float)
        elif np.all(self._target_est_eci == 0.0):
            pos_sig = np.array(self.cfg.target_pos_sigma_km, dtype=float).reshape(-1)
            vel_sig = np.array(self.cfg.target_vel_sigma_km_s, dtype=float).reshape(-1)
            if pos_sig.size == 1:
                pos_sig = np.full(3, float(pos_sig[0]))
            if vel_sig.size == 1:
                vel_sig = np.full(3, float(vel_sig[0]))
            z_pos = self.chief.truth.position_eci_km + self.rng.normal(0.0, pos_sig, size=3)
            z_vel = self.chief.truth.velocity_eci_km_s + self.rng.normal(0.0, vel_sig, size=3)
            self._target_est_eci = np.hstack((z_pos, z_vel))

    def _observation(self) -> np.ndarray:
        own_est = np.array(self.chaser.belief.state[:6], dtype=float)
        att_est = np.array(self.chaser.belief.state[6:13], dtype=float)
        fuel_remaining_kg = max(0.0, float(self.chaser.truth.mass_kg - self._dry_mass))
        return np.hstack((own_est, self._target_est_eci, att_est, np.array([fuel_remaining_kg], dtype=float)))

    def _range_km(self) -> float:
        return float(np.linalg.norm(self.chaser.truth.position_eci_km - self.chief.truth.position_eci_km))

    def _predict_closest_range_km_over_horizon(self) -> float:
        x_chaser = np.hstack((self.chaser.truth.position_eci_km, self.chaser.truth.velocity_eci_km_s))
        x_chief = np.hstack((self.chief.truth.position_eci_km, self.chief.truth.velocity_eci_km_s))
        closest = float(np.linalg.norm(x_chaser[:3] - x_chief[:3]))
        dt_lh = max(float(self.cfg.lookahead_dt_s), float(self.cfg.dt_s))
        steps = max(1, int(np.ceil(float(self.cfg.lookahead_horizon_s) / dt_lh)))
        for _ in range(steps):
            x_chaser = propagate_two_body_rk4(
                x_eci=x_chaser,
                dt_s=dt_lh,
                mu_km3_s2=EARTH_MU_KM3_S2,
                accel_cmd_eci_km_s2=np.zeros(3),
            )
            x_chief = propagate_two_body_rk4(
                x_eci=x_chief,
                dt_s=dt_lh,
                mu_km3_s2=EARTH_MU_KM3_S2,
                accel_cmd_eci_km_s2=np.zeros(3),
            )
            r = float(np.linalg.norm(x_chaser[:3] - x_chief[:3]))
            if r < closest:
                closest = r
        return closest

    def _info(self, capture: bool = False, timeout: bool = False) -> dict:
        return {
            "range_km": self._range_km(),
            "closest_range_km": float(self._closest_range_km),
            "fuel_remaining_kg": max(0.0, float(self.chaser.truth.mass_kg - self._dry_mass)),
            "capture": bool(capture),
            "timeout": bool(timeout),
            "step": int(self.step_count),
        }
