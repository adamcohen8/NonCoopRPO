from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.dynamics.attitude.rigid_body import propagate_attitude_exponential_map
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import ecef_to_eci, eci_to_ecef_rotation
from sim.dynamics.orbit.propagator import OrbitPropagator, drag_plugin, j2_plugin, j3_plugin, j4_plugin, srp_plugin
from sim.rocket.models import (
    GuidanceCommand,
    RocketGuidanceLaw,
    RocketSimConfig,
    RocketSimResult,
    RocketState,
    RocketVehicleConfig,
)
from sim.utils.quaternion import normalize_quaternion, quaternion_to_dcm_bn

G0_M_S2 = 9.80665


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _launch_position_velocity_eci(lat_deg: float, lon_deg: float, alt_km: float, t_s: float) -> tuple[np.ndarray, np.ndarray]:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    r_mag = EARTH_RADIUS_KM + alt_km
    r_ecef = np.array(
        [
            r_mag * np.cos(lat) * np.cos(lon),
            r_mag * np.cos(lat) * np.sin(lon),
            r_mag * np.sin(lat),
        ],
        dtype=float,
    )
    r_eci = ecef_to_eci(r_ecef, t_s)
    # Stationary on launch pad in rotating Earth frame.
    omega = np.array([0.0, 0.0, 7.2921159e-5], dtype=float)
    v_eci = np.cross(omega, r_eci)
    return r_eci, v_eci


def _initial_attitude_quaternion(r_eci_km: np.ndarray, azimuth_deg: float) -> np.ndarray:
    r_hat = _unit(r_eci_km)
    k = np.array([0.0, 0.0, 1.0], dtype=float)
    east = _unit(np.cross(k, r_hat))
    if np.linalg.norm(east) <= 0.0:
        east = np.array([0.0, 1.0, 0.0])
    north = _unit(np.cross(r_hat, east))
    az = np.deg2rad(azimuth_deg)
    # body +X along launch axis, initially near radial with azimuth yaw bias.
    x_b = _unit(np.cos(np.deg2rad(1.0)) * r_hat + np.sin(np.deg2rad(1.0)) * (_unit(np.cos(az) * north + np.sin(az) * east)))
    y_b = _unit(np.cross(k, x_b))
    if np.linalg.norm(y_b) <= 0.0:
        y_b = _unit(np.cross(np.array([0.0, 1.0, 0.0]), x_b))
    z_b = _unit(np.cross(x_b, y_b))
    c_bn = np.vstack((x_b, y_b, z_b))
    from sim.utils.quaternion import dcm_to_quaternion_bn

    return dcm_to_quaternion_bn(c_bn)


def _orbital_elements_basic(r_km: np.ndarray, v_km_s: np.ndarray, mu_km3_s2: float) -> tuple[float, float]:
    r = float(np.linalg.norm(r_km))
    v2 = float(np.dot(v_km_s, v_km_s))
    if r <= 0.0:
        return np.inf, np.inf
    eps = 0.5 * v2 - mu_km3_s2 / r
    if abs(eps) < 1e-14:
        a = np.inf
    else:
        a = -mu_km3_s2 / (2.0 * eps)
    h = np.cross(r_km, v_km_s)
    e_vec = np.cross(v_km_s, h) / mu_km3_s2 - r_km / r
    e = float(np.linalg.norm(e_vec))
    return float(a), e


@dataclass
class RocketAscentSimulator:
    sim_cfg: RocketSimConfig
    vehicle_cfg: RocketVehicleConfig
    guidance: RocketGuidanceLaw

    def __post_init__(self) -> None:
        stages = self.vehicle_cfg.stack.stages
        if len(stages) == 0:
            raise ValueError("vehicle_cfg.stack must contain at least one stage.")
        self._stage_dry = np.array([s.dry_mass_kg for s in stages], dtype=float)
        self._stage_prop0 = np.array([s.propellant_mass_kg for s in stages], dtype=float)
        self._stage_thrust = np.array([s.max_thrust_n for s in stages], dtype=float)
        self._stage_isp = np.array([s.isp_s for s in stages], dtype=float)
        if self.sim_cfg.area_ref_m2 is None:
            d = float(stages[0].diameter_m)
            self._area_ref_m2 = np.pi * 0.25 * d * d
        else:
            self._area_ref_m2 = float(self.sim_cfg.area_ref_m2)

        plugins = []
        if self.sim_cfg.enable_j2:
            plugins.append(j2_plugin)
        if self.sim_cfg.enable_j3:
            plugins.append(j3_plugin)
        if self.sim_cfg.enable_j4:
            plugins.append(j4_plugin)
        if self.sim_cfg.enable_drag:
            plugins.append(drag_plugin)
        if self.sim_cfg.enable_srp:
            plugins.append(srp_plugin)
        self._propagator = OrbitPropagator(integrator="rk4", plugins=plugins)

    def initial_state(self) -> RocketState:
        r0, v0 = _launch_position_velocity_eci(
            lat_deg=self.sim_cfg.launch_lat_deg,
            lon_deg=self.sim_cfg.launch_lon_deg,
            alt_km=self.sim_cfg.launch_alt_km,
            t_s=0.0,
        )
        q0 = _initial_attitude_quaternion(r0, self.sim_cfg.launch_azimuth_deg)
        mass0 = float(np.sum(self._stage_dry + self._stage_prop0) + self.vehicle_cfg.payload_mass_kg)
        return RocketState(
            t_s=0.0,
            position_eci_km=r0,
            velocity_eci_km_s=v0,
            attitude_quat_bn=q0,
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=mass0,
            active_stage_index=0,
            stage_prop_remaining_kg=self._stage_prop0.copy(),
            payload_attached=True,
        )

    def run(self, state0: RocketState | None = None) -> RocketSimResult:
        state = self.initial_state() if state0 is None else state0.copy()
        dt = self.sim_cfg.dt_s
        steps = int(np.ceil(self.sim_cfg.max_time_s / dt))

        t = np.zeros(steps + 1)
        r = np.zeros((steps + 1, 3))
        v = np.zeros((steps + 1, 3))
        q = np.zeros((steps + 1, 4))
        w = np.zeros((steps + 1, 3))
        m = np.zeros(steps + 1)
        stg = np.zeros(steps + 1, dtype=int)
        thr_cmd = np.zeros(steps + 1)
        thrust_n = np.zeros(steps + 1)
        alt = np.zeros(steps + 1)
        ecc = np.zeros(steps + 1)
        sma = np.zeros(steps + 1)

        inserted = False
        insertion_time = None
        insertion_hold_counter = 0.0
        hold_needed = self.sim_cfg.insertion_hold_time_s

        for k in range(steps + 1):
            t[k] = state.t_s
            r[k, :] = state.position_eci_km
            v[k, :] = state.velocity_eci_km_s
            q[k, :] = state.attitude_quat_bn
            w[k, :] = state.angular_rate_body_rad_s
            m[k] = state.mass_kg
            stg[k] = state.active_stage_index
            alt[k] = float(np.linalg.norm(state.position_eci_km) - EARTH_RADIUS_KM)
            a_km, e_k = _orbital_elements_basic(state.position_eci_km, state.velocity_eci_km_s, EARTH_MU_KM3_S2)
            sma[k] = a_km
            ecc[k] = e_k
            if k == steps:
                break

            cmd = self.guidance.command(state, self.sim_cfg, self.vehicle_cfg)
            throttle = _clamp(float(cmd.throttle), 0.0, 1.0)
            thr_cmd[k] = throttle
            state = self._step_once(state, cmd, throttle, dt)
            thrust_n[k] = float(state._last_step_thrust_n) if hasattr(state, "_last_step_thrust_n") else 0.0

            # insertion criterion: altitude near target and low eccentricity while coasting.
            near_alt = abs((np.linalg.norm(state.position_eci_km) - EARTH_RADIUS_KM) - self.sim_cfg.target_altitude_km) <= self.sim_cfg.target_altitude_tolerance_km
            _, e_now = _orbital_elements_basic(state.position_eci_km, state.velocity_eci_km_s, EARTH_MU_KM3_S2)
            low_e = e_now <= self.sim_cfg.target_eccentricity_max
            if near_alt and low_e and state.active_stage_index >= len(self._stage_dry):
                insertion_hold_counter += dt
                if insertion_hold_counter >= hold_needed:
                    inserted = True
                    insertion_time = state.t_s
                    # truncate arrays
                    n = k + 2
                    return RocketSimResult(
                        time_s=t[:n].copy(),
                        position_eci_km=r[:n, :].copy(),
                        velocity_eci_km_s=v[:n, :].copy(),
                        attitude_quat_bn=q[:n, :].copy(),
                        angular_rate_body_rad_s=w[:n, :].copy(),
                        mass_kg=m[:n].copy(),
                        active_stage_index=stg[:n].copy(),
                        throttle_cmd=thr_cmd[:n].copy(),
                        thrust_n=thrust_n[:n].copy(),
                        altitude_km=alt[:n].copy(),
                        eccentricity=ecc[:n].copy(),
                        sma_km=sma[:n].copy(),
                        inserted=inserted,
                        insertion_time_s=insertion_time,
                    )
            else:
                insertion_hold_counter = 0.0

        return RocketSimResult(
            time_s=t,
            position_eci_km=r,
            velocity_eci_km_s=v,
            attitude_quat_bn=q,
            angular_rate_body_rad_s=w,
            mass_kg=m,
            active_stage_index=stg,
            throttle_cmd=thr_cmd,
            thrust_n=thrust_n,
            altitude_km=alt,
            eccentricity=ecc,
            sma_km=sma,
            inserted=inserted,
            insertion_time_s=insertion_time,
        )

    def _step_once(self, state: RocketState, cmd: GuidanceCommand, throttle: float, dt_s: float) -> RocketState:
        s = state.copy()
        stage_i = s.active_stage_index
        thrust_n = 0.0
        dm_prop = 0.0
        stage_separated = False

        if stage_i < len(self._stage_dry):
            prop_left = float(s.stage_prop_remaining_kg[stage_i])
            if prop_left > 0.0 and throttle > 0.0:
                thrust_n = float(throttle * self._stage_thrust[stage_i])
                mdot = thrust_n / max(self._stage_isp[stage_i] * G0_M_S2, 1e-9)
                dm_prop = min(prop_left, mdot * dt_s)
                s.stage_prop_remaining_kg[stage_i] = prop_left - dm_prop
                s.mass_kg = max(0.0, s.mass_kg - dm_prop)

            # stage separation when prop hits empty.
            if s.stage_prop_remaining_kg[stage_i] <= 1e-9:
                s.mass_kg = max(0.0, s.mass_kg - self._stage_dry[stage_i])
                s.active_stage_index += 1
                stage_separated = True

        if cmd.attitude_quat_bn_cmd is not None:
            s.attitude_quat_bn = normalize_quaternion(np.array(cmd.attitude_quat_bn_cmd, dtype=float))

        c_bn = quaternion_to_dcm_bn(s.attitude_quat_bn)
        thrust_axis_body = _unit(np.array(self.vehicle_cfg.thrust_axis_body, dtype=float))
        thrust_axis_eci = c_bn.T @ thrust_axis_body
        accel_thrust_eci_km_s2 = (thrust_n / max(s.mass_kg, 1e-9)) * thrust_axis_eci / 1e3

        x_orbit = np.hstack((s.position_eci_km, s.velocity_eci_km_s))
        env = {"atmosphere_model": self.sim_cfg.atmosphere_model}
        ctx = OrbitContext(
            mu_km3_s2=EARTH_MU_KM3_S2,
            mass_kg=s.mass_kg,
            area_m2=self._area_ref_m2,
            cd=self.sim_cfg.cd,
            cr=self.sim_cfg.cr,
        )
        x_next = self._propagator.propagate(
            x_eci=x_orbit,
            dt_s=dt_s,
            t_s=s.t_s,
            command_accel_eci_km_s2=accel_thrust_eci_km_s2,
            env=env,
            ctx=ctx,
        )

        torque_cmd = np.zeros(3) if cmd.torque_body_nm_cmd is None else np.array(cmd.torque_body_nm_cmd, dtype=float)
        att_h = max(min(self.sim_cfg.attitude_substep_s, dt_s), 1e-4)
        rem = dt_s
        qn = s.attitude_quat_bn.copy()
        wn = s.angular_rate_body_rad_s.copy()
        while rem > 1e-12:
            h = min(att_h, rem)
            qn, wn = propagate_attitude_exponential_map(
                quat_bn=qn,
                omega_body_rad_s=wn,
                inertia_kg_m2=self.sim_cfg.inertia_kg_m2,
                torque_body_nm=torque_cmd,
                dt_s=h,
            )
            rem -= h

        s.position_eci_km = x_next[:3]
        s.velocity_eci_km_s = x_next[3:]
        s.attitude_quat_bn = qn
        s.angular_rate_body_rad_s = wn
        s.t_s = s.t_s + dt_s
        # attach last-step telemetry attribute for logging convenience.
        setattr(s, "_last_step_thrust_n", thrust_n)
        setattr(s, "_last_step_stage_sep", stage_separated)
        return s
