from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from sim.dynamics.attitude.rigid_body import propagate_attitude_exponential_map
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.atmosphere import atmosphere_state_from_model
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import ecef_to_eci, eci_to_ecef_rotation
from sim.dynamics.orbit.propagator import OrbitPropagator, drag_plugin, j2_plugin, j3_plugin, j4_plugin, srp_plugin
from sim.rocket.aero import RocketAeroConfig, compute_aero_loads, compute_aero_state
from sim.rocket.models import (
    GuidanceCommand,
    RocketGuidanceLaw,
    RocketSimConfig,
    RocketSimResult,
    RocketState,
    RocketVehicleConfig,
)
from sim.utils.geodesy import ecef_to_geodetic_deg_km, enu_to_ecef_rotation, geodetic_to_ecef_km
from sim.utils.quaternion import normalize_quaternion, quaternion_to_dcm_bn

G0_M_S2 = 9.80665
P0_SEA_LEVEL_PA = 101325.0


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _launch_position_velocity_eci(lat_deg: float, lon_deg: float, alt_km: float, t_s: float) -> tuple[np.ndarray, np.ndarray]:
    r_ecef = geodetic_to_ecef_km(lat_deg=lat_deg, lon_deg=lon_deg, alt_km=alt_km)
    r_eci = ecef_to_eci(r_ecef, t_s)
    # Stationary on launch pad in rotating Earth frame.
    omega = np.array([0.0, 0.0, 7.2921159e-5], dtype=float)
    v_eci = np.cross(omega, r_eci)
    return r_eci, v_eci


def _geodetic_state_from_eci(r_eci_km: np.ndarray, t_s: float, jd_utc_start: float | None = None) -> tuple[float, float, float]:
    r_ecef = eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start) @ np.array(r_eci_km, dtype=float).reshape(3)
    return ecef_to_geodetic_deg_km(r_ecef)


def _resolve_wind_eci_m_s(
    *,
    position_eci_km: np.ndarray,
    t_s: float,
    sim_cfg: RocketSimConfig,
    state: RocketState | None = None,
) -> np.ndarray:
    lat_deg, lon_deg, alt_km = _geodetic_state_from_eci(
        position_eci_km,
        t_s,
        jd_utc_start=sim_cfg.atmosphere_env.get("jd_utc_start"),
    )
    wind_enu = np.array(sim_cfg.wind_enu_m_s, dtype=float).reshape(3)
    wind_cb = sim_cfg.wind_enu_callable
    if callable(wind_cb):
        wind_enu = wind_enu + np.array(wind_cb(alt_km, lat_deg, lon_deg, t_s, state, sim_cfg), dtype=float).reshape(3)
    wind_ecef = enu_to_ecef_rotation(lat_deg, lon_deg) @ wind_enu
    wind_eci = eci_to_ecef_rotation(t_s, jd_utc_start=sim_cfg.atmosphere_env.get("jd_utc_start")).T @ (wind_ecef / 1e3)
    return wind_eci * 1e3


def _vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ua = _unit(np.array(a, dtype=float).reshape(3))
    ub = _unit(np.array(b, dtype=float).reshape(3))
    if np.linalg.norm(ua) <= 0.0 or np.linalg.norm(ub) <= 0.0:
        return 0.0
    return float(np.rad2deg(np.arccos(np.clip(float(np.dot(ua, ub)), -1.0, 1.0))))


def _limit_vector_cone(v: np.ndarray, axis: np.ndarray, max_angle_rad: float) -> np.ndarray:
    u = _unit(v)
    a = _unit(axis)
    if np.linalg.norm(u) <= 0.0:
        return a
    if max_angle_rad <= 0.0:
        return a
    angle = float(np.arccos(np.clip(float(np.dot(u, a)), -1.0, 1.0)))
    if angle <= max_angle_rad:
        return u
    lateral = u - float(np.dot(u, a)) * a
    lateral = _unit(lateral)
    return _unit(np.cos(max_angle_rad) * a + np.sin(max_angle_rad) * lateral)


def _step_tvc_vector(
    current_body: np.ndarray,
    target_body: np.ndarray,
    nominal_axis_body: np.ndarray,
    dt_s: float,
    sim_cfg: RocketSimConfig,
) -> np.ndarray:
    current = _unit(current_body)
    nominal = _unit(nominal_axis_body)
    target = _limit_vector_cone(target_body, nominal, np.deg2rad(float(sim_cfg.tvc_max_gimbal_deg)))
    alpha = float(1.0 - np.exp(-dt_s / max(sim_cfg.tvc_time_constant_s, 1e-9)))
    blended = _unit((1.0 - alpha) * current + alpha * target)
    if sim_cfg.tvc_rate_limit_deg_s <= 0.0:
        return blended
    max_step = float(np.deg2rad(sim_cfg.tvc_rate_limit_deg_s) * dt_s)
    step_angle = float(np.arccos(np.clip(float(np.dot(current, blended)), -1.0, 1.0)))
    if step_angle <= max_step or step_angle <= 1e-12:
        return blended
    beta = max_step / step_angle
    return _unit((1.0 - beta) * current + beta * blended)


def _stage_engine_perf(stage, pressure_pa: float) -> tuple[float, float]:
    p = float(np.clip(pressure_pa, 0.0, P0_SEA_LEVEL_PA))
    sea_w = p / P0_SEA_LEVEL_PA
    vac_w = 1.0 - sea_w
    thrust_sl = float(stage.sea_level_thrust_n if stage.sea_level_thrust_n is not None else stage.max_thrust_n)
    thrust_vac = float(stage.vacuum_thrust_n if stage.vacuum_thrust_n is not None else stage.max_thrust_n)
    isp_sl = float(stage.sea_level_isp_s if stage.sea_level_isp_s is not None else stage.isp_s)
    isp_vac = float(stage.vacuum_isp_s if stage.vacuum_isp_s is not None else stage.isp_s)
    thrust_n = sea_w * thrust_sl + vac_w * thrust_vac
    isp_s = sea_w * isp_sl + vac_w * isp_vac
    return float(max(thrust_n, 0.0)), float(max(isp_s, 1e-9))


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
        self._stage_area_ref_m2 = np.array([np.pi * 0.25 * float(s.diameter_m) * float(s.diameter_m) for s in stages], dtype=float)
        self._stage_ref_length_m = np.array([float(s.length_m) for s in stages], dtype=float)
        self._base_aero_ref_length_m = float(max(self.sim_cfg.aero.reference_length_m, 1e-9))
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
        if self.sim_cfg.enable_drag and not self.sim_cfg.aero.enabled:
            plugins.append(drag_plugin)
        if self.sim_cfg.enable_srp:
            plugins.append(srp_plugin)
        self._propagator = OrbitPropagator(integrator="rk4", plugins=plugins)

    def _resolve_aero_config_for_stage(self, stage_i: int) -> RocketAeroConfig:
        cfg = self.sim_cfg.aero
        if not cfg.enabled:
            return cfg
        idx = int(np.clip(stage_i, 0, len(self._stage_ref_length_m) - 1))

        # Optional global override keeps area fixed regardless of stage.
        area_m2 = float(self.sim_cfg.area_ref_m2) if self.sim_cfg.area_ref_m2 is not None else float(cfg.reference_area_m2)
        ref_len_m = float(cfg.reference_length_m)
        cp_offset = np.array(cfg.cp_offset_body_m, dtype=float).reshape(3)
        if self.sim_cfg.use_stagewise_aero_geometry:
            if self.sim_cfg.area_ref_m2 is None:
                area_m2 = float(self._stage_area_ref_m2[idx])
            ref_len_m = float(max(self._stage_ref_length_m[idx], 1e-9))
            scale = ref_len_m / self._base_aero_ref_length_m
            cp_offset = cp_offset * scale
        return replace(cfg, reference_area_m2=area_m2, reference_length_m=ref_len_m, cp_offset_body_m=cp_offset)

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
            thrust_vector_body=_unit(np.array(self.vehicle_cfg.thrust_axis_body, dtype=float)),
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
        q_dyn = np.zeros(steps + 1)
        mach = np.zeros(steps + 1)
        lat_deg = np.zeros(steps + 1)
        lon_deg = np.zeros(steps + 1)
        wind_body_m_s = np.zeros((steps + 1, 3))
        tvc_gimbal_deg = np.zeros(steps + 1)
        alpha_deg = np.zeros(steps + 1)
        beta_deg = np.zeros(steps + 1)
        cd = np.zeros(steps + 1)
        aero_force_n = np.zeros(steps + 1)
        aero_moment_nm = np.zeros(steps + 1)

        inserted = False
        insertion_time = None
        terminated_early = False
        termination_reason = None
        termination_time = None
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
            lat_now, lon_now, alt_now = _geodetic_state_from_eci(
                state.position_eci_km,
                state.t_s,
                jd_utc_start=self.sim_cfg.atmosphere_env.get("jd_utc_start"),
            )
            alt[k] = float(alt_now if self.sim_cfg.use_wgs84_geodesy else np.linalg.norm(state.position_eci_km) - EARTH_RADIUS_KM)
            lat_deg[k] = float(lat_now)
            lon_deg[k] = float(lon_now)
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
            q_dyn[k] = float(getattr(state, "_last_step_q_dyn_pa", 0.0))
            mach[k] = float(getattr(state, "_last_step_mach", 0.0))
            wind_body_m_s[k, :] = np.array(getattr(state, "_last_step_wind_body_m_s", np.zeros(3)), dtype=float)
            tvc_gimbal_deg[k] = float(getattr(state, "_last_step_tvc_gimbal_deg", 0.0))
            alpha_deg[k] = float(getattr(state, "_last_step_alpha_deg", 0.0))
            beta_deg[k] = float(getattr(state, "_last_step_beta_deg", 0.0))
            cd[k] = float(getattr(state, "_last_step_cd", 0.0))
            aero_force_n[k] = float(getattr(state, "_last_step_aero_force_n", 0.0))
            aero_moment_nm[k] = float(getattr(state, "_last_step_aero_moment_nm", 0.0))

            if self.sim_cfg.terminate_on_earth_impact:
                impact = float(np.linalg.norm(state.position_eci_km)) <= float(self.sim_cfg.earth_impact_radius_km)
                if self.sim_cfg.use_wgs84_geodesy:
                    _, _, alt_check = _geodetic_state_from_eci(
                        state.position_eci_km,
                        state.t_s,
                        jd_utc_start=self.sim_cfg.atmosphere_env.get("jd_utc_start"),
                    )
                    impact = bool(alt_check <= 0.0)
                if impact:
                    terminated_early = True
                    termination_reason = "earth_impact"
                    termination_time = float(state.t_s)
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
                        latitude_deg=lat_deg[:n].copy(),
                        longitude_deg=lon_deg[:n].copy(),
                        eccentricity=ecc[:n].copy(),
                        sma_km=sma[:n].copy(),
                        dynamic_pressure_pa=q_dyn[:n].copy(),
                        mach=mach[:n].copy(),
                        wind_body_m_s=wind_body_m_s[:n, :].copy(),
                        tvc_gimbal_deg=tvc_gimbal_deg[:n].copy(),
                        alpha_deg=alpha_deg[:n].copy(),
                        beta_deg=beta_deg[:n].copy(),
                        cd=cd[:n].copy(),
                        aero_force_n=aero_force_n[:n].copy(),
                        aero_moment_nm=aero_moment_nm[:n].copy(),
                        inserted=inserted,
                        insertion_time_s=insertion_time,
                        terminated_early=terminated_early,
                        termination_reason=termination_reason,
                        termination_time_s=termination_time,
                    )

            # insertion criterion: altitude near target and low eccentricity while coasting.
            _, _, alt_ins = _geodetic_state_from_eci(
                state.position_eci_km,
                state.t_s,
                jd_utc_start=self.sim_cfg.atmosphere_env.get("jd_utc_start"),
            )
            alt_compare = float(alt_ins if self.sim_cfg.use_wgs84_geodesy else np.linalg.norm(state.position_eci_km) - EARTH_RADIUS_KM)
            near_alt = abs(alt_compare - self.sim_cfg.target_altitude_km) <= self.sim_cfg.target_altitude_tolerance_km
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
                        latitude_deg=lat_deg[:n].copy(),
                        longitude_deg=lon_deg[:n].copy(),
                        eccentricity=ecc[:n].copy(),
                        sma_km=sma[:n].copy(),
                        dynamic_pressure_pa=q_dyn[:n].copy(),
                        mach=mach[:n].copy(),
                        wind_body_m_s=wind_body_m_s[:n, :].copy(),
                        tvc_gimbal_deg=tvc_gimbal_deg[:n].copy(),
                        alpha_deg=alpha_deg[:n].copy(),
                        beta_deg=beta_deg[:n].copy(),
                        cd=cd[:n].copy(),
                        aero_force_n=aero_force_n[:n].copy(),
                        aero_moment_nm=aero_moment_nm[:n].copy(),
                        inserted=inserted,
                        insertion_time_s=insertion_time,
                        terminated_early=terminated_early,
                        termination_reason=termination_reason,
                        termination_time_s=termination_time,
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
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
            eccentricity=ecc,
            sma_km=sma,
            dynamic_pressure_pa=q_dyn,
            mach=mach,
            wind_body_m_s=wind_body_m_s,
            tvc_gimbal_deg=tvc_gimbal_deg,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
            cd=cd,
            aero_force_n=aero_force_n,
            aero_moment_nm=aero_moment_nm,
            inserted=inserted,
            insertion_time_s=insertion_time,
            terminated_early=terminated_early,
            termination_reason=termination_reason,
            termination_time_s=termination_time,
        )

    def step(self, state: RocketState, command: GuidanceCommand, dt_s: float | None = None) -> RocketState:
        """Public single-step API for external scenario orchestrators."""
        h = float(self.sim_cfg.dt_s if dt_s is None else dt_s)
        throttle = _clamp(float(command.throttle), 0.0, 1.0)
        return self._step_once(state=state, cmd=command, throttle=throttle, dt_s=h)

    def _step_once(self, state: RocketState, cmd: GuidanceCommand, throttle: float, dt_s: float) -> RocketState:
        s = state.copy()
        stage_i = s.active_stage_index
        thrust_n = 0.0
        dm_prop = 0.0
        stage_separated = False
        env = {"atmosphere_model": self.sim_cfg.atmosphere_model, **dict(self.sim_cfg.atmosphere_env)}
        if self.sim_cfg.use_wgs84_geodesy:
            env["geodetic_model"] = "wgs84"
        atmos = atmosphere_state_from_model(
            model=str(self.sim_cfg.atmosphere_model).lower(),
            r_eci_km=s.position_eci_km,
            t_s=s.t_s,
            env=env,
        )
        ambient_pressure_pa = float(atmos["pressure_pa"])

        if stage_i < len(self._stage_dry):
            prop_left = float(s.stage_prop_remaining_kg[stage_i])
            if prop_left > 0.0 and throttle > 0.0:
                stage = self.vehicle_cfg.stack.stages[stage_i]
                stage_thrust_n, stage_isp_s = _stage_engine_perf(stage, ambient_pressure_pa)
                thrust_n = float(throttle * stage_thrust_n)
                mdot = thrust_n / max(stage_isp_s * G0_M_S2, 1e-9)
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

        nominal_thrust_axis_body = _unit(np.array(self.vehicle_cfg.thrust_axis_body, dtype=float))
        tvc_target_body = nominal_thrust_axis_body if cmd.thrust_vector_body_cmd is None else _unit(np.array(cmd.thrust_vector_body_cmd, dtype=float))
        s.thrust_vector_body = _step_tvc_vector(
            current_body=s.thrust_vector_body,
            target_body=tvc_target_body,
            nominal_axis_body=nominal_thrust_axis_body,
            dt_s=dt_s,
            sim_cfg=self.sim_cfg,
        )

        c_bn = quaternion_to_dcm_bn(s.attitude_quat_bn)
        thrust_axis_eci = c_bn.T @ s.thrust_vector_body
        accel_thrust_eci_km_s2 = (thrust_n / max(s.mass_kg, 1e-9)) * thrust_axis_eci / 1e3
        torque_aero_body_nm = np.zeros(3)
        accel_aero_eci_km_s2 = np.zeros(3)
        torque_tvc_body_nm = np.cross(
            np.array(self.sim_cfg.tvc_pivot_offset_body_m, dtype=float).reshape(3),
            thrust_n * s.thrust_vector_body,
        )
        last_q_dyn = 0.0
        last_mach = 0.0
        last_wind_body_m_s = np.zeros(3)
        last_tvc_gimbal_deg = _vector_angle_deg(s.thrust_vector_body, nominal_thrust_axis_body)
        last_alpha_deg = 0.0
        last_beta_deg = 0.0
        last_cd = 0.0
        last_aero_force_n = 0.0
        last_aero_moment_nm = 0.0
        if self.sim_cfg.aero.enabled:
            omega_earth = np.array([0.0, 0.0, 7.2921159e-5], dtype=float)
            v_atm_eci_km_s = np.cross(omega_earth, s.position_eci_km)
            wind_eci_m_s = _resolve_wind_eci_m_s(position_eci_km=s.position_eci_km, t_s=s.t_s, sim_cfg=self.sim_cfg, state=s)
            v_rel_eci_m_s = (s.velocity_eci_km_s - v_atm_eci_km_s) * 1e3 - wind_eci_m_s
            v_rel_body_m_s = c_bn @ v_rel_eci_m_s
            aero_state = compute_aero_state(
                rho_kg_m3=float(atmos["density_kg_m3"]),
                pressure_pa=ambient_pressure_pa,
                temperature_k=float(atmos["temperature_k"]),
                sound_speed_m_s=float(atmos["sound_speed_m_s"]),
                v_rel_body_m_s=v_rel_body_m_s,
                alpha_limit_deg=float(self.sim_cfg.aero.alpha_limit_deg),
                beta_limit_deg=float(self.sim_cfg.aero.beta_limit_deg),
            )
            aero_cfg = self._resolve_aero_config_for_stage(stage_i=stage_i)
            loads = compute_aero_loads(v_rel_body_m_s=v_rel_body_m_s, atmos=aero_state, cfg=aero_cfg)
            f_aero_eci_n = c_bn.T @ loads.force_body_n
            accel_aero_eci_km_s2 = f_aero_eci_n / max(s.mass_kg, 1e-9) / 1e3
            torque_aero_body_nm = loads.moment_body_nm
            last_q_dyn = float(loads.state.dynamic_pressure_pa)
            last_mach = float(loads.state.mach)
            last_wind_body_m_s = c_bn @ wind_eci_m_s
            last_alpha_deg = float(np.rad2deg(loads.state.alpha_rad))
            last_beta_deg = float(np.rad2deg(loads.state.beta_rad))
            last_cd = float(-loads.coeff_force_body[0])
            last_aero_force_n = float(np.linalg.norm(loads.force_body_n))
            last_aero_moment_nm = float(np.linalg.norm(loads.moment_body_nm))

        x_orbit = np.hstack((s.position_eci_km, s.velocity_eci_km_s))
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
            command_accel_eci_km_s2=accel_thrust_eci_km_s2 + accel_aero_eci_km_s2,
            env=env,
            ctx=ctx,
        )

        mode = str(self.sim_cfg.attitude_mode).strip().lower()
        if mode == "cheater":
            if cmd.attitude_quat_bn_cmd is not None:
                qn = normalize_quaternion(np.array(cmd.attitude_quat_bn_cmd, dtype=float))
            else:
                qn = s.attitude_quat_bn.copy()
            wn = np.zeros(3, dtype=float)
        else:
            torque_cmd = np.zeros(3) if cmd.torque_body_nm_cmd is None else np.array(cmd.torque_body_nm_cmd, dtype=float)
            torque_cmd = torque_cmd + torque_aero_body_nm + torque_tvc_body_nm
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
        setattr(s, "_last_step_q_dyn_pa", last_q_dyn)
        setattr(s, "_last_step_mach", last_mach)
        setattr(s, "_last_step_wind_body_m_s", last_wind_body_m_s.copy())
        setattr(s, "_last_step_tvc_gimbal_deg", last_tvc_gimbal_deg)
        setattr(s, "_last_step_alpha_deg", last_alpha_deg)
        setattr(s, "_last_step_beta_deg", last_beta_deg)
        setattr(s, "_last_step_cd", last_cd)
        setattr(s, "_last_step_aero_force_n", last_aero_force_n)
        setattr(s, "_last_step_aero_moment_nm", last_aero_moment_nm)
        return s
