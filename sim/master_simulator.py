from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import importlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from presets.rockets import BASIC_TWO_STAGE_STACK
from sim.config import SimulationScenarioConfig, load_simulation_yaml, scenario_config_from_dict, validate_scenario_plugins
from sim.control.attitude.zero_torque import ZeroTorqueController
from sim.control.orbit.zero_controller import ZeroController
from sim.core.models import Command, Measurement, StateBelief, StateTruth
from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.propagator import OrbitPropagator, drag_plugin, j2_plugin, j3_plugin, j4_plugin, srp_plugin
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.knowledge.object_tracking import (
    KnowledgeConditionConfig,
    KnowledgeEKFConfig,
    KnowledgeNoiseConfig,
    ObjectKnowledgeBase,
    TrackedObjectConfig,
)
from sim.rocket import OpenLoopPitchProgramGuidance, RocketAscentSimulator, RocketSimConfig, RocketState, RocketVehicleConfig
from sim.sensors.noisy_own_state import NoisyOwnStateSensor
from sim.utils.io import write_json
from sim.utils.plotting import plot_attitude_tumble, plot_orbit_eci
from sim.utils.plotting_capabilities import (
    animate_ground_track,
    animate_multi_ground_track,
    animate_multi_rectangular_prism_ric_curv,
    plot_body_rates,
    plot_control_commands,
    plot_multi_control_commands,
    plot_multi_ric_2d_projections,
    plot_multi_trajectory_frame,
    plot_quaternion_components,
    plot_ric_2d_projections,
    plot_trajectory_frame,
)
from sim.utils.ground_track import ground_track_from_eci_history
from sim.utils.frames import ric_curv_to_rect, ric_dcm_ir_from_rv, ric_rect_to_curv
from sim.utils.quaternion import quaternion_to_dcm_bn


def _state_truth_to_array(truth: StateTruth) -> np.ndarray:
    return np.hstack(
        (
            truth.position_eci_km,
            truth.velocity_eci_km_s,
            truth.attitude_quat_bn,
            truth.angular_rate_body_rad_s,
            np.array([truth.mass_kg]),
        )
    )


def _array_to_truth(x14: np.ndarray, t_s: float) -> StateTruth:
    return StateTruth(
        position_eci_km=np.array(x14[0:3], dtype=float),
        velocity_eci_km_s=np.array(x14[3:6], dtype=float),
        attitude_quat_bn=np.array(x14[6:10], dtype=float),
        angular_rate_body_rad_s=np.array(x14[10:13], dtype=float),
        mass_kg=float(x14[13]),
        t_s=float(t_s),
    )


def _rocket_state_to_truth(s: RocketState) -> StateTruth:
    return StateTruth(
        position_eci_km=np.array(s.position_eci_km, dtype=float),
        velocity_eci_km_s=np.array(s.velocity_eci_km_s, dtype=float),
        attitude_quat_bn=np.array(s.attitude_quat_bn, dtype=float),
        angular_rate_body_rad_s=np.array(s.angular_rate_body_rad_s, dtype=float),
        mass_kg=float(s.mass_kg),
        t_s=float(s.t_s),
    )


def _module_obj(pointer) -> Any | None:
    if pointer is None:
        return None
    if pointer.module is None:
        return None
    try:
        mod = importlib.import_module(pointer.module)
        if pointer.class_name:
            cls = getattr(mod, pointer.class_name)
            return cls(**dict(pointer.params or {}))
        if pointer.function:
            fn = getattr(mod, pointer.function)
            return fn
        return mod
    except Exception:
        return None


def _deep_set(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = root
    for i, tok in enumerate(parts):
        last = i == len(parts) - 1
        if "[" in tok and tok.endswith("]"):
            key, idx_txt = tok[:-1].split("[", 1)
            idx = int(idx_txt)
            if key:
                cur = cur[key]
            if not isinstance(cur, list):
                raise TypeError(f"'{tok}' is not a list segment in path '{path}'.")
            if last:
                cur[idx] = value
                return
            cur = cur[idx]
            continue
        if last:
            cur[tok] = value
            return
        cur = cur[tok]


def _sample_variation(v, rng: np.random.Generator) -> Any:
    mode = v.mode.lower()
    if mode == "choice":
        if not v.options:
            raise ValueError(f"Variation '{v.parameter_path}' with mode=choice requires options.")
        return v.options[int(rng.integers(0, len(v.options)))]
    if mode == "uniform":
        if v.low is None or v.high is None:
            raise ValueError(f"Variation '{v.parameter_path}' with mode=uniform requires low/high.")
        return float(rng.uniform(v.low, v.high))
    if mode == "normal":
        if v.mean is None or v.std is None:
            raise ValueError(f"Variation '{v.parameter_path}' with mode=normal requires mean/std.")
        return float(rng.normal(v.mean, v.std))
    raise ValueError(f"Unsupported variation mode '{v.mode}'.")


def _combine_commands(orb: Command, att: Command) -> Command:
    return Command(
        thrust_eci_km_s2=np.array(orb.thrust_eci_km_s2, dtype=float),
        torque_body_nm=np.array(att.torque_body_nm, dtype=float),
        mode_flags={**dict(orb.mode_flags or {}), **dict(att.mode_flags or {})},
    )


def _coe_to_rv_eci(
    *,
    a_km: float,
    ecc: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    true_anomaly_deg: float,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> tuple[np.ndarray, np.ndarray]:
    a = float(a_km)
    e = float(ecc)
    if a <= 0.0:
        raise ValueError("COE a_km must be positive.")
    if e < 0.0 or e >= 1.0:
        raise ValueError("COE eccentricity must satisfy 0 <= e < 1 for current support.")

    inc = np.deg2rad(float(inc_deg))
    raan = np.deg2rad(float(raan_deg))
    argp = np.deg2rad(float(argp_deg))
    nu = np.deg2rad(float(true_anomaly_deg))

    p = a * (1.0 - e * e)
    if p <= 0.0:
        raise ValueError("Invalid COE set: semi-latus rectum must be positive.")

    cnu, snu = np.cos(nu), np.sin(nu)
    r_pf = np.array([p * cnu / (1.0 + e * cnu), p * snu / (1.0 + e * cnu), 0.0], dtype=float)
    v_pf = np.sqrt(mu_km3_s2 / p) * np.array([-snu, e + cnu, 0.0], dtype=float)

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    q_pf_to_eci = np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ],
        dtype=float,
    )
    r_eci = q_pf_to_eci @ r_pf
    v_eci = q_pf_to_eci @ v_pf
    return r_eci, v_eci


def _rv_from_initial_state(s0: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "position_eci_km" in s0:
        pos = np.array(s0.get("position_eci_km", [7000.0, 0.0, 0.0]), dtype=float)
        if "velocity_eci_km_s" in s0:
            vel = np.array(s0["velocity_eci_km_s"], dtype=float)
        else:
            spd = float(np.sqrt(EARTH_MU_KM3_S2 / max(np.linalg.norm(pos), EARTH_RADIUS_KM + 1.0)))
            vel = np.array([0.0, spd, 0.0], dtype=float)
        return pos, vel

    coes = s0.get("coes")
    if isinstance(coes, dict):
        d = dict(coes)
        a_km = float(d.get("a_km", d.get("semi_major_axis_km", 7000.0)))
        ecc = float(d.get("ecc", d.get("e", 0.0)))
        inc_deg = float(d.get("inc_deg", d.get("inclination_deg", 0.0)))
        raan_deg = float(d.get("raan_deg", 0.0))
        argp_deg = float(d.get("argp_deg", d.get("arg_periapsis_deg", 0.0)))
        ta_deg = float(d.get("ta_deg", d.get("true_anomaly_deg", 0.0)))
        return _coe_to_rv_eci(
            a_km=a_km,
            ecc=ecc,
            inc_deg=inc_deg,
            raan_deg=raan_deg,
            argp_deg=argp_deg,
            true_anomaly_deg=ta_deg,
        )

    pos = np.array([7000.0, 0.0, 0.0], dtype=float)
    spd = float(np.sqrt(EARTH_MU_KM3_S2 / np.linalg.norm(pos)))
    return pos, np.array([0.0, spd, 0.0], dtype=float)


def _default_truth_from_agent(agent_cfg: Any, t_s: float = 0.0) -> StateTruth:
    s0 = dict(agent_cfg.initial_state or {})
    specs = dict(agent_cfg.specs or {})
    if ("dry_mass_kg" in specs) or ("fuel_mass_kg" in specs):
        dry_mass_kg = float(specs.get("dry_mass_kg", 0.0))
        fuel_mass_kg = float(specs.get("fuel_mass_kg", 0.0))
        if dry_mass_kg < 0.0 or fuel_mass_kg < 0.0:
            raise ValueError("dry_mass_kg and fuel_mass_kg must be non-negative.")
        mass_kg = dry_mass_kg + fuel_mass_kg
    else:
        mass_kg = float(specs.get("mass_kg", 300.0))
    pos, vel = _rv_from_initial_state(s0)
    return StateTruth(
        position_eci_km=pos,
        velocity_eci_km_s=vel,
        attitude_quat_bn=np.array(s0.get("attitude_quat_bn", [1.0, 0.0, 0.0, 0.0]), dtype=float),
        angular_rate_body_rad_s=np.array(s0.get("angular_rate_body_rad_s", [0.0, 0.0, 0.0]), dtype=float),
        mass_kg=mass_kg,
        t_s=t_s,
    )


def _resolve_chaser_relative_ric_init(initial_state: dict[str, Any]) -> tuple[np.ndarray, str] | None:
    s0 = dict(initial_state or {})
    rel_block = s0.get("relative_to_target_ric")
    if isinstance(rel_block, dict):
        frame = str(rel_block.get("frame", "rect")).strip().lower()
        state = np.array(rel_block.get("state", []), dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("chaser.initial_state.relative_to_target_ric.state must be length-6.")
        if frame not in ("rect", "curv"):
            raise ValueError("chaser.initial_state.relative_to_target_ric.frame must be 'rect' or 'curv'.")
        return state, frame

    if "relative_ric_rect" in s0:
        state = np.array(s0.get("relative_ric_rect"), dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("chaser.initial_state.relative_ric_rect must be length-6.")
        return state, "rect"
    if "relative_ric_curv" in s0:
        state = np.array(s0.get("relative_ric_curv"), dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("chaser.initial_state.relative_ric_curv must be length-6.")
        return state, "curv"
    return None


def _apply_chaser_relative_init_from_target(
    *,
    chaser: AgentRuntime,
    target: AgentRuntime,
    initial_state: dict[str, Any],
) -> None:
    rel = _resolve_chaser_relative_ric_init(initial_state)
    if rel is None:
        return
    x_rel, frame = rel
    if chaser.truth is None or target.truth is None:
        return

    r_t = np.array(target.truth.position_eci_km, dtype=float)
    v_t = np.array(target.truth.velocity_eci_km_s, dtype=float)
    r0 = float(np.linalg.norm(r_t))
    if r0 <= 0.0:
        return

    x_rel_rect = ric_curv_to_rect(x_rel, r0_km=r0) if frame == "curv" else np.array(x_rel, dtype=float).reshape(6)
    dr_ric = x_rel_rect[:3]
    dv_ric = x_rel_rect[3:]
    c_ir = ric_dcm_ir_from_rv(r_t, v_t)

    chaser.truth.position_eci_km = r_t + c_ir @ dr_ric
    chaser.truth.velocity_eci_km_s = v_t + c_ir @ dv_ric


def _build_orbit_propagator(cfg: SimulationScenarioConfig) -> OrbitPropagator:
    o = dict(cfg.simulator.dynamics.get("orbit", {}) or {})
    plugins = []
    if bool(o.get("j2", False)):
        plugins.append(j2_plugin)
    if bool(o.get("j3", False)):
        plugins.append(j3_plugin)
    if bool(o.get("j4", False)):
        plugins.append(j4_plugin)
    if bool(o.get("drag", False)):
        plugins.append(drag_plugin)
    if bool(o.get("srp", False)):
        plugins.append(srp_plugin)
    return OrbitPropagator(integrator="rk4", plugins=plugins)


@dataclass
class AgentRuntime:
    object_id: str
    kind: str  # "rocket" | "satellite"
    enabled: bool
    active: bool
    truth: StateTruth | None
    belief: StateBelief | None
    sensor: Any | None
    estimator: Any | None
    orbit_controller: Any | None
    attitude_controller: Any | None
    dynamics: OrbitalAttitudeDynamics | None
    knowledge_base: ObjectKnowledgeBase | None
    bridge: Any | None
    rocket_sim: RocketAscentSimulator | None
    rocket_state: RocketState | None
    rocket_guidance: Any | None
    deploy_source: str | None
    deploy_time_s: float | None
    deploy_dv_body_m_s: np.ndarray | None
    mission_modules: list[Any]
    waiting_for_launch: bool


@dataclass
class _RateLimitedController:
    base: Any
    period_s: float
    _last_eval_t_s: float | None = None
    _last_cmd: Command = field(default_factory=Command.zero, init=False)

    def __post_init__(self) -> None:
        self.period_s = float(max(self.period_s, 1e-9))

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        if self._last_eval_t_s is None or float(t_s) - float(self._last_eval_t_s) >= self.period_s - 1e-12:
            self._last_cmd = self.base.act(belief, t_s, budget_ms)
            self._last_eval_t_s = float(t_s)
        return self._last_cmd

    def __getattr__(self, item: str) -> Any:
        return getattr(self.base, item)


def _create_satellite_runtime(
    object_id: str,
    agent_cfg: Any,
    cfg: SimulationScenarioConfig,
    rng: np.random.Generator,
) -> AgentRuntime:
    truth = _default_truth_from_agent(agent_cfg, t_s=0.0)
    belief = StateBelief(state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)), covariance=np.eye(6) * 1e-4, last_update_t_s=0.0)
    noise = dict((agent_cfg.knowledge or {}).get("sensor_error", {}) or {})
    pos_sigma = float(np.array(noise.get("pos_sigma_km", [0.001])).reshape(-1)[0])
    vel_sigma = float(np.array(noise.get("vel_sigma_km_s", [1e-5])).reshape(-1)[0])
    sensor = NoisyOwnStateSensor(pos_sigma_km=pos_sigma, vel_sigma_km_s=vel_sigma, rng=rng)
    estimator = OrbitEKFEstimator(
        mu_km3_s2=EARTH_MU_KM3_S2,
        dt_s=float(cfg.simulator.dt_s),
        process_noise_diag=np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10]),
        meas_noise_diag=np.array([1e-6, 1e-6, 1e-6, 1e-10, 1e-10, 1e-10]),
    )
    orbit_ctrl_base = _module_obj(agent_cfg.orbit_control) or ZeroController()
    att_ctrl_base = _module_obj(agent_cfg.attitude_control) or ZeroTorqueController()
    orbit_cfg = dict(cfg.simulator.dynamics.get("orbit", {}) or {})
    att_cfg = dict(cfg.simulator.dynamics.get("attitude", {}) or {})
    dist_cfg = dict(att_cfg.get("disturbance_torques", {}) or {})
    orbit_ctrl_period_s = float(max(float(orbit_cfg.get("orbit_substep_s", cfg.simulator.dt_s) or cfg.simulator.dt_s), 1e-9))
    att_ctrl_period_s = float(max(float(att_cfg.get("attitude_substep_s", cfg.simulator.dt_s) or cfg.simulator.dt_s), 1e-9))
    orbit_ctrl = _RateLimitedController(base=orbit_ctrl_base, period_s=orbit_ctrl_period_s)
    att_ctrl = _RateLimitedController(base=att_ctrl_base, period_s=att_ctrl_period_s)
    dmodel = DisturbanceTorqueModel(
        mu_km3_s2=EARTH_MU_KM3_S2,
        inertia_kg_m2=np.diag([120.0, 100.0, 80.0]),
        config=DisturbanceTorqueConfig(
            use_gravity_gradient=bool(dist_cfg.get("gravity_gradient", False)),
            use_magnetic=bool(dist_cfg.get("magnetic", False)),
            use_drag=bool(dist_cfg.get("drag", False)),
            use_srp=bool(dist_cfg.get("srp", False)),
        ),
    )
    dyn = OrbitalAttitudeDynamics(
        mu_km3_s2=EARTH_MU_KM3_S2,
        inertia_kg_m2=np.diag([120.0, 100.0, 80.0]),
        disturbance_model=dmodel if bool(att_cfg.get("enabled", True)) else None,
        orbit_substep_s=float(orbit_cfg["orbit_substep_s"]) if orbit_cfg.get("orbit_substep_s") is not None else None,
        attitude_substep_s=float(att_cfg["attitude_substep_s"]) if att_cfg.get("attitude_substep_s") is not None else None,
        orbit_propagator=_build_orbit_propagator(cfg),
    )
    bridge = _module_obj(agent_cfg.bridge) if (agent_cfg.bridge is not None and agent_cfg.bridge.enabled) else None
    missions = [_module_obj(p) for p in list(agent_cfg.mission_objectives or [])]
    missions = [m for m in missions if m is not None]
    return AgentRuntime(
        object_id=object_id,
        kind="satellite",
        enabled=bool(agent_cfg.enabled),
        active=bool(agent_cfg.enabled),
        truth=truth,
        belief=belief,
        sensor=sensor,
        estimator=estimator,
        orbit_controller=orbit_ctrl,
        attitude_controller=att_ctrl,
        dynamics=dyn,
        knowledge_base=None,
        bridge=bridge,
        rocket_sim=None,
        rocket_state=None,
        rocket_guidance=None,
        deploy_source=str((agent_cfg.initial_state or {}).get("source", "")) or None,
        deploy_time_s=float((agent_cfg.initial_state or {}).get("deploy_time_s", 0.0)),
        deploy_dv_body_m_s=np.array((agent_cfg.initial_state or {}).get("deploy_dv_body_m_s", [0.0, 0.0, 0.0]), dtype=float),
        mission_modules=missions,
        waiting_for_launch=False,
    )


def _create_rocket_runtime(cfg: SimulationScenarioConfig) -> AgentRuntime:
    rc = cfg.rocket
    r_init = dict(rc.initial_state or {})
    sim_cfg = RocketSimConfig(
        dt_s=float(cfg.simulator.dt_s),
        max_time_s=float(cfg.simulator.duration_s),
        launch_lat_deg=float(r_init.get("launch_lat_deg", 0.0)),
        launch_lon_deg=float(r_init.get("launch_lon_deg", 0.0)),
        launch_alt_km=float(r_init.get("launch_alt_km", 0.0)),
        launch_azimuth_deg=float(r_init.get("launch_azimuth_deg", 90.0)),
        atmosphere_model=str(cfg.simulator.dynamics.get("rocket", {}).get("atmosphere_model", "ussa1976")),
        terminate_on_earth_impact=bool(cfg.simulator.termination.get("earth_impact_enabled", True)),
        earth_impact_radius_km=float(cfg.simulator.termination.get("earth_radius_km", 6378.137)),
    )
    vehicle_cfg = RocketVehicleConfig(
        stack=BASIC_TWO_STAGE_STACK,
        payload_mass_kg=float(rc.specs.get("payload_mass_kg", 150.0)),
        thrust_axis_body=np.array(rc.specs.get("thrust_axis_body", [1.0, 0.0, 0.0]), dtype=float),
    )
    guidance = _module_obj(rc.guidance) or OpenLoopPitchProgramGuidance()
    rsim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=guidance)
    rs = rsim.initial_state()
    rt = _rocket_state_to_truth(rs)
    belief = StateBelief(state=np.hstack((rt.position_eci_km, rt.velocity_eci_km_s)), covariance=np.eye(6) * 1e-4, last_update_t_s=0.0)
    bridge = _module_obj(rc.bridge) if (rc.bridge is not None and rc.bridge.enabled) else None
    missions = [_module_obj(p) for p in list(rc.mission_objectives or [])]
    missions = [m for m in missions if m is not None]
    return AgentRuntime(
        object_id="rocket",
        kind="rocket",
        enabled=bool(rc.enabled),
        active=bool(rc.enabled),
        truth=rt,
        belief=belief,
        sensor=None,
        estimator=None,
        orbit_controller=None,
        attitude_controller=None,
        dynamics=None,
        knowledge_base=None,
        bridge=bridge,
        rocket_sim=rsim,
        rocket_state=rs,
        rocket_guidance=guidance,
        deploy_source=None,
        deploy_time_s=None,
        deploy_dv_body_m_s=None,
        mission_modules=missions,
        waiting_for_launch=False,
    )


def _build_knowledge_base(observer_id: str, agent_cfg: Any, dt_s: float, rng: np.random.Generator) -> ObjectKnowledgeBase | None:
    k = dict(agent_cfg.knowledge or {})
    targets = list(k.get("targets", []) or [])
    if not targets:
        return None
    cond = dict(k.get("conditions", {}) or {})
    noise = dict(k.get("sensor_error", {}) or {})
    est = dict(k.get("estimation", {}) or {})
    tr: list[TrackedObjectConfig] = []
    for tgt in targets:
        tr.append(
            TrackedObjectConfig(
                target_id=str(tgt),
                conditions=KnowledgeConditionConfig(
                    refresh_rate_s=float(k.get("refresh_rate_s", dt_s)),
                    max_range_km=cond.get("max_range_km"),
                    fov_half_angle_rad=cond.get("fov_half_angle_rad"),
                    require_line_of_sight=bool(cond.get("require_line_of_sight", False)),
                    dropout_prob=float(cond.get("dropout_prob", 0.0)),
                ),
                sensor_noise=KnowledgeNoiseConfig(
                    pos_sigma_km=np.array(noise.get("pos_sigma_km", [0.01, 0.01, 0.01]), dtype=float),
                    vel_sigma_km_s=np.array(noise.get("vel_sigma_km_s", [1e-4, 1e-4, 1e-4]), dtype=float),
                    pos_bias_km=np.array(noise.get("pos_bias_km", [0.0, 0.0, 0.0]), dtype=float),
                    vel_bias_km_s=np.array(noise.get("vel_bias_km_s", [0.0, 0.0, 0.0]), dtype=float),
                ),
                estimator=str(est.get("type", "ekf")),
                ekf=KnowledgeEKFConfig(),
            )
        )
    return ObjectKnowledgeBase(observer_id=observer_id, tracked_objects=tr, dt_s=dt_s, rng=rng, mu_km3_s2=EARTH_MU_KM3_S2)


def _deploy_from_rocket(agent: AgentRuntime, rocket: AgentRuntime, t_next: float) -> None:
    if agent.kind != "satellite" or agent.active:
        return
    if agent.deploy_source != "rocket_deployment":
        return
    if rocket.rocket_state is None:
        return
    c_bn = quaternion_to_dcm_bn(rocket.rocket_state.attitude_quat_bn)
    dv_body = np.array(agent.deploy_dv_body_m_s if agent.deploy_dv_body_m_s is not None else np.zeros(3), dtype=float)
    dv_eci_km_s = (c_bn.T @ dv_body) / 1e3
    rs = rocket.rocket_state
    m = float(agent.truth.mass_kg) if agent.truth is not None else 200.0
    agent.truth = StateTruth(
        position_eci_km=np.array(rs.position_eci_km, dtype=float),
        velocity_eci_km_s=np.array(rs.velocity_eci_km_s, dtype=float) + dv_eci_km_s,
        attitude_quat_bn=np.array(rs.attitude_quat_bn, dtype=float),
        angular_rate_body_rad_s=np.array(rs.angular_rate_body_rad_s, dtype=float),
        mass_kg=m,
        t_s=t_next,
    )
    agent.belief = StateBelief(state=np.hstack((agent.truth.position_eci_km, agent.truth.velocity_eci_km_s)), covariance=np.eye(6) * 1e-4, last_update_t_s=t_next)
    agent.active = True


def _run_mission_modules(
    *,
    agent: AgentRuntime,
    world_truth: dict[str, StateTruth],
    t_s: float,
    dt_s: float,
    env: dict[str, Any],
    orbit_controller: Any | None = None,
    attitude_controller: Any | None = None,
    orb_belief: StateBelief | None = None,
    att_belief: StateBelief | None = None,
) -> dict[str, Any]:
    if not agent.mission_modules:
        return {}
    own_knowledge = agent.knowledge_base.snapshot() if agent.knowledge_base is not None else {}
    truth = world_truth.get(agent.object_id)
    if truth is None:
        return {}
    out: dict[str, Any] = {}
    for m in agent.mission_modules:
        if not hasattr(m, "update"):
            continue
        try:
            ret = m.update(
                object_id=agent.object_id,
                truth=truth,
                belief=agent.belief,
                own_knowledge=own_knowledge,
                world_truth=world_truth,
                env=env,
                t_s=t_s,
                dt_s=dt_s,
                orbit_controller=orbit_controller,
                attitude_controller=attitude_controller,
                orb_belief=orb_belief,
                att_belief=att_belief,
                rocket_state=agent.rocket_state,
                rocket_vehicle_cfg=(agent.rocket_sim.vehicle_cfg if agent.rocket_sim is not None else None),
            )
        except TypeError:
            ret = m.update(truth=truth, t_s=t_s)
        if isinstance(ret, dict):
            out.update(ret)
    return out


def _plot_outputs(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    thrust_hist: dict[str, np.ndarray],
    knowledge_hist: dict[str, dict[str, np.ndarray]],
    outdir: Path,
) -> dict[str, str]:
    out: dict[str, str] = {}
    if not bool(cfg.outputs.plots.get("enabled", True)):
        return out
    mode = cfg.outputs.mode
    figure_ids = list(cfg.outputs.plots.get("figure_ids", []) or [])
    ric_2d_planes = list(cfg.outputs.plots.get("ric_2d_planes", ["ri", "ic", "rc"]) or ["ri", "ic", "rc"])
    reference_object_id = str(cfg.outputs.plots.get("reference_object_id", "")).strip()
    if reference_object_id and reference_object_id not in truth_hist:
        reference_object_id = ""
    if not reference_object_id and "target" in truth_hist:
        reference_object_id = "target"
    if not reference_object_id and truth_hist:
        reference_object_id = sorted(truth_hist.keys())[0]
    reference_truth = truth_hist.get(reference_object_id) if reference_object_id else None
    ric_truth_hist = (
        {oid: hist for oid, hist in truth_hist.items() if oid != reference_object_id}
        if reference_object_id
        else dict(truth_hist)
    )
    if not figure_ids:
        return out
    for oid, hist in truth_hist.items():
        if not np.any(np.isfinite(hist[:, 0])):
            continue
        if "orbit_eci" in figure_ids:
            p = outdir / f"{oid}_orbit_eci.png"
            plot_orbit_eci(hist, mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_orbit_eci"] = str(p)
        if "attitude" in figure_ids:
            p = outdir / f"{oid}_attitude.png"
            plot_attitude_tumble(t_s=t_s, truth_hist=hist, mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_attitude"] = str(p)

    if "relative_range" in figure_ids:
        ids = list(truth_hist.keys())
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = truth_hist[ids[i]][:, :3]
                b = truth_hist[ids[j]][:, :3]
                mask = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
                if not np.any(mask):
                    continue
                rr = np.linalg.norm(a - b, axis=1)
                ax.plot(t_s[mask], rr[mask], label=f"{ids[i]}-{ids[j]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Range (km)")
        ax.set_title("Relative Range")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        p = outdir / "relative_ranges.png"
        if mode in ("save", "both"):
            fig.savefig(p, dpi=int(cfg.outputs.plots.get("dpi", 150)))
            out["relative_ranges"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    # Multi-agent shared-figure trajectory plots.
    if "trajectory_eci_multi" in figure_ids:
        p = outdir / "trajectory_eci_multi.png"
        plot_multi_trajectory_frame(t_s, truth_hist, frame="eci", mode=mode, out_path=str(p))
        if mode in ("save", "both"):
            out["trajectory_eci_multi"] = str(p)
    if "trajectory_ecef_multi" in figure_ids:
        p = outdir / "trajectory_ecef_multi.png"
        plot_multi_trajectory_frame(t_s, truth_hist, frame="ecef", mode=mode, out_path=str(p))
        if mode in ("save", "both"):
            out["trajectory_ecef_multi"] = str(p)
    if "trajectory_ric_rect_multi" in figure_ids and reference_truth is not None:
        p = outdir / "trajectory_ric_rect_multi.png"
        plot_multi_trajectory_frame(
            t_s,
            ric_truth_hist,
            frame="ric_rect",
            reference_truth_hist=reference_truth,
            mode=mode,
            out_path=str(p),
        )
        if mode in ("save", "both"):
            out["trajectory_ric_rect_multi"] = str(p)
    if "trajectory_ric_curv_multi" in figure_ids and reference_truth is not None:
        p = outdir / "trajectory_ric_curv_multi.png"
        plot_multi_trajectory_frame(
            t_s,
            ric_truth_hist,
            frame="ric_curv",
            reference_truth_hist=reference_truth,
            mode=mode,
            out_path=str(p),
        )
        if mode in ("save", "both"):
            out["trajectory_ric_curv_multi"] = str(p)
    if "trajectory_ric_rect_2d_multi" in figure_ids and reference_truth is not None:
        p = outdir / "trajectory_ric_rect_2d_multi.png"
        plot_multi_ric_2d_projections(
            t_s,
            ric_truth_hist,
            frame="ric_rect",
            reference_truth_hist=reference_truth,
            planes=ric_2d_planes,
            mode=mode,
            out_path=str(p),
        )
        if mode in ("save", "both"):
            out["trajectory_ric_rect_2d_multi"] = str(p)
    if "trajectory_ric_curv_2d_multi" in figure_ids and reference_truth is not None:
        p = outdir / "trajectory_ric_curv_2d_multi.png"
        plot_multi_ric_2d_projections(
            t_s,
            ric_truth_hist,
            frame="ric_curv",
            reference_truth_hist=reference_truth,
            planes=ric_2d_planes,
            mode=mode,
            out_path=str(p),
        )
        if mode in ("save", "both"):
            out["trajectory_ric_curv_2d_multi"] = str(p)

    for oid, hist in truth_hist.items():
        if not np.any(np.isfinite(hist[:, 0])):
            continue
        if "quaternion_eci" in figure_ids:
            p = outdir / f"{oid}_quat_eci.png"
            plot_quaternion_components(t_s, hist, frame="eci", layout="single", mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_quat_eci"] = str(p)
        if "quaternion_ric" in figure_ids:
            p = outdir / f"{oid}_quat_ric.png"
            plot_quaternion_components(t_s, hist, frame="ric", layout="single", mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_quat_ric"] = str(p)
        if "rates_eci" in figure_ids:
            p = outdir / f"{oid}_rates_eci.png"
            plot_body_rates(t_s, hist, frame="eci", layout="subplots", mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_rates_eci"] = str(p)
        if "rates_ric" in figure_ids:
            p = outdir / f"{oid}_rates_ric.png"
            plot_body_rates(t_s, hist, frame="ric", layout="subplots", mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_rates_ric"] = str(p)
        if "trajectory_ecef" in figure_ids:
            p = outdir / f"{oid}_traj_ecef.png"
            plot_trajectory_frame(t_s, hist, frame="ecef", mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_traj_ecef"] = str(p)
        if "trajectory_ric_rect" in figure_ids and reference_truth is not None and oid != reference_object_id:
            p = outdir / f"{oid}_traj_ric_rect.png"
            plot_trajectory_frame(
                t_s,
                hist,
                frame="ric_rect",
                reference_truth_hist=reference_truth,
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_traj_ric_rect"] = str(p)
        if "trajectory_ric_curv" in figure_ids and reference_truth is not None and oid != reference_object_id:
            p = outdir / f"{oid}_traj_ric_curv.png"
            plot_trajectory_frame(
                t_s,
                hist,
                frame="ric_curv",
                reference_truth_hist=reference_truth,
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_traj_ric_curv"] = str(p)
        if "trajectory_ric_rect_2d" in figure_ids and reference_truth is not None and oid != reference_object_id:
            p = outdir / f"{oid}_traj_ric_rect_2d.png"
            plot_ric_2d_projections(
                t_s,
                hist,
                frame="ric_rect",
                reference_truth_hist=reference_truth,
                planes=ric_2d_planes,
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_traj_ric_rect_2d"] = str(p)
        if "trajectory_ric_curv_2d" in figure_ids and reference_truth is not None and oid != reference_object_id:
            p = outdir / f"{oid}_traj_ric_curv_2d.png"
            plot_ric_2d_projections(
                t_s,
                hist,
                frame="ric_curv",
                reference_truth_hist=reference_truth,
                planes=ric_2d_planes,
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_traj_ric_curv_2d"] = str(p)

    thrust_hist_ric: dict[str, np.ndarray] = {}
    if ("control_thrust_ric" in figure_ids) or ("control_thrust_ric_multi" in figure_ids):
        for oid, u in thrust_hist.items():
            hist = truth_hist.get(oid)
            if hist is None or hist.size == 0:
                continue
            n_s = min(u.shape[0], hist.shape[0], t_s.size)
            ur = np.full((u.shape[0], 3), np.nan, dtype=float)
            for k in range(n_s):
                a_eci = np.array(u[k, :], dtype=float)
                rv = np.array(hist[k, 0:6], dtype=float)
                if not (np.all(np.isfinite(a_eci)) and np.all(np.isfinite(rv))):
                    continue
                c_ir = ric_dcm_ir_from_rv(rv[:3], rv[3:6])
                ur[k, :] = c_ir.T @ a_eci
            thrust_hist_ric[oid] = ur

    if "control_thrust" in figure_ids:
        for oid, u in thrust_hist.items():
            if not np.any(np.isfinite(u[:, 0])):
                continue
            p = outdir / f"{oid}_control_thrust.png"
            plot_control_commands(
                t_s,
                u,
                layout="subplots",
                input_labels=["ax", "ay", "az"],
                title=f"Thrust Commands ({oid})",
                y_label="km/s^2",
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_control_thrust"] = str(p)

    if "control_thrust_ric" in figure_ids:
        for oid, u in thrust_hist_ric.items():
            if not np.any(np.isfinite(u[:, 0])):
                continue
            p = outdir / f"{oid}_control_thrust_ric.png"
            plot_control_commands(
                t_s,
                u,
                layout="subplots",
                input_labels=["aR", "aI", "aC"],
                title=f"Thrust Commands RIC ({oid})",
                y_label="km/s^2",
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_control_thrust_ric"] = str(p)

    if "control_thrust_multi" in figure_ids:
        for i_comp, lbl in enumerate(("ax", "ay", "az")):
            p = outdir / f"control_thrust_multi_{lbl}.png"
            plot_multi_control_commands(
                t_s,
                thrust_hist,
                component_index=i_comp,
                title=f"Thrust Command Overlay ({lbl})",
                y_label="km/s^2",
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"control_thrust_multi_{lbl}"] = str(p)

    if "control_thrust_ric_multi" in figure_ids:
        for i_comp, lbl in enumerate(("aR", "aI", "aC")):
            p = outdir / f"control_thrust_ric_multi_{lbl}.png"
            plot_multi_control_commands(
                t_s,
                thrust_hist_ric,
                component_index=i_comp,
                title=f"Thrust Command Overlay RIC ({lbl})",
                y_label="km/s^2",
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"control_thrust_ric_multi_{lbl}"] = str(p)

    if "thrust_alignment_error" in figure_ids:
        import matplotlib.pyplot as plt

        thrust_dir_body = np.array(cfg.outputs.plots.get("thrust_direction_body", [1.0, 0.0, 0.0]), dtype=float).reshape(-1)
        if thrust_dir_body.size != 3:
            thrust_dir_body = np.array([1.0, 0.0, 0.0], dtype=float)
        n_t = float(np.linalg.norm(thrust_dir_body))
        if n_t <= 0.0:
            thrust_dir_body = np.array([1.0, 0.0, 0.0], dtype=float)
            n_t = 1.0
        thrust_dir_body = thrust_dir_body / n_t

        for oid, hist in truth_hist.items():
            u = thrust_hist.get(oid)
            if u is None or hist.size == 0:
                continue
            thrust_norm = np.linalg.norm(np.nan_to_num(u, nan=0.0), axis=1)
            if not np.any(thrust_norm > 1e-15):
                continue
            err_deg = np.full(t_s.shape, np.nan, dtype=float)
            for k in range(min(hist.shape[0], u.shape[0], t_s.size)):
                a_cmd = np.array(u[k, :], dtype=float)
                if not np.all(np.isfinite(a_cmd)):
                    continue
                a_norm = float(np.linalg.norm(a_cmd))
                if a_norm <= 1e-15:
                    continue
                q_bn = np.array(hist[k, 6:10], dtype=float)
                if not np.all(np.isfinite(q_bn)):
                    continue
                c_bn = quaternion_to_dcm_bn(q_bn)
                thrust_axis_eci = c_bn.T @ thrust_dir_body
                burn_dir_eci = -a_cmd / a_norm
                cosang = float(np.clip(np.dot(thrust_axis_eci, burn_dir_eci), -1.0, 1.0))
                if not np.isfinite(cosang):
                    continue
                err_deg[k] = float(np.degrees(np.arccos(cosang)))

            fig, ax = plt.subplots(figsize=(10, 5))
            finite = np.isfinite(err_deg)
            if np.any(finite):
                t_f = np.array(t_s[finite], dtype=float)
                e_f = np.array(err_deg[finite], dtype=float)
                # Burns are often impulsive/sparse, so error samples may be isolated.
                # Always draw markers so the trace is visible even without contiguous segments.
                ax.plot(t_f, e_f, linewidth=1.2, marker="o", markersize=2.5)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No valid burn/alignment samples in this run",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Angle Error (deg)")
            ax.set_title(f"Attitude vs Thrust Vector Error ({oid})")
            ax.grid(True, alpha=0.3)
            p = outdir / f"{oid}_thrust_alignment_error.png"
            if mode in ("save", "both"):
                fig.savefig(p, dpi=int(cfg.outputs.plots.get("dpi", 150)))
                out[f"{oid}_thrust_alignment_error"] = str(p)
            if mode in ("interactive", "both"):
                plt.show(block=False)
            else:
                plt.close(fig)

    if "knowledge_timeline" in figure_ids:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        i = 0
        for obs, by_tgt in knowledge_hist.items():
            for tgt, hist in by_tgt.items():
                known = np.any(np.isfinite(hist), axis=1).astype(float)
                ax.plot(t_s, known + i * 1.2, label=f"{obs}->{tgt}")
                i += 1
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Known (offset)")
        ax.set_title("Knowledge Timeline")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        p = outdir / "knowledge_timeline.png"
        if mode in ("save", "both"):
            fig.savefig(p, dpi=int(cfg.outputs.plots.get("dpi", 150)))
            out["knowledge_timeline"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    return out


def _animate_outputs(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    outdir: Path,
) -> dict[str, str]:
    out: dict[str, str] = {}
    anim_cfg = dict(cfg.outputs.animations or {})
    if not bool(anim_cfg.get("enabled", False)):
        return out

    mode = cfg.outputs.mode
    fps = float(anim_cfg.get("fps", 30.0))
    speed_multiple = float(anim_cfg.get("speed_multiple", 10.0))
    frame_stride = int(anim_cfg.get("frame_stride", 1))
    draw_earth_map = bool(anim_cfg.get("draw_earth_map", True))
    types = list(anim_cfg.get("types", []) or [])
    if not types:
        return out

    if "ground_track_multi" in types:
        p = outdir / "ground_track_multi.mp4"
        animate_multi_ground_track(
            t_s=t_s,
            truth_hist_by_object=truth_hist,
            jd_utc_start=cfg.simulator.initial_jd_utc,
            mode=mode,
            out_path=str(p),
            fps=fps,
            speed_multiple=speed_multiple,
            draw_earth_map=draw_earth_map,
            frame_stride=frame_stride,
        )
        if mode in ("save", "both"):
            out["ground_track_multi"] = str(p)

    if "ground_track" in types:
        for oid, hist in truth_hist.items():
            if hist.size == 0 or not np.any(np.isfinite(hist[:, 0])):
                continue
            lat_deg, lon_deg, _ = ground_track_from_eci_history(
                hist[:, :3],
                t_s=t_s,
                jd_utc_start=cfg.simulator.initial_jd_utc,
            )
            p = outdir / f"{oid}_ground_track.mp4"
            animate_ground_track(
                lon_deg=lon_deg,
                lat_deg=lat_deg,
                t_s=t_s,
                jd_utc_start=cfg.simulator.initial_jd_utc,
                mode=mode,
                out_path=str(p),
                fps=fps,
                speed_multiple=speed_multiple,
                draw_earth_map=draw_earth_map,
                frame_stride=frame_stride,
            )
            if mode in ("save", "both"):
                out[f"{oid}_ground_track"] = str(p)

    if "ric_curv_prism_multi" in types:
        p = outdir / "ric_curv_prism_multi.mp4"
        target_object_id = str(anim_cfg.get("target_object_id", "target"))
        prism_obj_ids = anim_cfg.get("ric_curv_prism_object_ids")
        if not isinstance(prism_obj_ids, list):
            prism_obj_ids = None
        dims_map_raw = anim_cfg.get("ric_curv_prism_dims_m", {})
        dims_map = dict(dims_map_raw) if isinstance(dims_map_raw, dict) else {}
        animate_multi_rectangular_prism_ric_curv(
            t_s=t_s,
            truth_hist_by_object=truth_hist,
            target_object_id=target_object_id,
            object_ids=prism_obj_ids,
            prism_dims_m_by_object=dims_map,
            mode=mode,
            out_path=str(p),
            fps=fps,
            speed_multiple=speed_multiple,
            frame_stride=frame_stride,
        )
        if mode in ("save", "both"):
            out["ric_curv_prism_multi"] = str(p)

    return out


def _run_single_config(
    cfg: SimulationScenarioConfig,
    step_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    dt = float(cfg.simulator.dt_s)
    n = int(np.floor(float(cfg.simulator.duration_s) / dt)) + 1
    t_s = np.arange(n, dtype=float) * dt
    outdir = Path(cfg.outputs.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.metadata.get("seed", 123))
    rng = np.random.default_rng(seed)

    rocket = _create_rocket_runtime(cfg) if cfg.rocket.enabled else None
    chaser = _create_satellite_runtime("chaser", cfg.chaser, cfg, np.random.default_rng(int(rng.integers(0, 2**31 - 1)))) if cfg.chaser.enabled else None
    target = _create_satellite_runtime("target", cfg.target, cfg, np.random.default_rng(int(rng.integers(0, 2**31 - 1)))) if cfg.target.enabled else None
    if chaser is not None and chaser.deploy_source == "rocket_deployment":
        chaser.active = False
    agents: dict[str, AgentRuntime] = {}
    if rocket is not None:
        agents["rocket"] = rocket
    if target is not None:
        agents["target"] = target
    if chaser is not None:
        agents["chaser"] = chaser
    if chaser is not None and target is not None and chaser.deploy_source != "rocket_deployment":
        _apply_chaser_relative_init_from_target(
            chaser=chaser,
            target=target,
            initial_state=dict(cfg.chaser.initial_state or {}),
        )

    for aid, a in agents.items():
        cfg_src = cfg.rocket if aid == "rocket" else (cfg.chaser if aid == "chaser" else cfg.target)
        a.knowledge_base = _build_knowledge_base(
            observer_id=aid,
            agent_cfg=cfg_src,
            dt_s=dt,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )

    truth_hist = {aid: np.full((n, 14), np.nan) for aid in agents.keys()}
    belief_hist = {aid: np.full((n, 6), np.nan) for aid in agents.keys()}
    thrust_hist = {aid: np.full((n, 3), np.nan) for aid in agents.keys()}
    torque_hist = {aid: np.full((n, 3), np.nan) for aid in agents.keys()}
    throttle_hist = {"rocket": np.full(n, np.nan)} if rocket is not None else {}
    knowledge_hist: dict[str, dict[str, np.ndarray]] = {}
    bridge_hist: dict[str, list[dict[str, Any]]] = {aid: [] for aid in agents.keys()}
    for aid, a in agents.items():
        if a.knowledge_base is not None:
            knowledge_hist[aid] = {}
            for tid in a.knowledge_base.target_ids():
                knowledge_hist[aid][tid] = np.full((n, 6), np.nan)

    terminated_early = False
    termination_reason = None
    termination_time_s = None
    termination_object_id = None
    final_index = n - 1

    # Initial logging
    for aid, a in agents.items():
        if not a.active:
            continue
        tr = a.truth if a.kind == "satellite" else _rocket_state_to_truth(a.rocket_state)
        truth_hist[aid][0, :] = _state_truth_to_array(tr)
        if a.belief is not None:
            belief_hist[aid][0, :] = a.belief.state[:6]

    total_steps = max(n - 1, 0)
    if step_callback is not None:
        try:
            step_callback(0, total_steps)
        except Exception:
            pass

    for k in range(n - 1):
        t = float(t_s[k])
        t_next = float(t_s[k + 1])

        if chaser is not None and rocket is not None and (not chaser.active):
            if t_next >= float(chaser.deploy_time_s or 0.0):
                _deploy_from_rocket(chaser, rocket, t_next)

        world_truth: dict[str, StateTruth] = {}
        for aid, a in agents.items():
            if not a.active:
                continue
            world_truth[aid] = a.truth if a.kind == "satellite" else _rocket_state_to_truth(a.rocket_state)

        for aid, a in agents.items():
            if not a.active or a.knowledge_base is None:
                continue
            a.knowledge_base.update(observer_truth=world_truth[aid], world_truth=world_truth, t_s=t_next)
            snap = a.knowledge_base.snapshot()
            for tid, hist in knowledge_hist.get(aid, {}).items():
                b = snap.get(tid)
                if b is not None:
                    hist[k + 1, :] = b.state[:6]
                elif k > 0:
                    hist[k + 1, :] = hist[k, :]

        for aid, a in agents.items():
            if not a.active:
                continue
            tr_now = world_truth[aid]
            env_common = {**dict(cfg.simulator.environment or {}), "world_truth": world_truth}
            mission_out: dict[str, Any] = {}

            if a.kind == "rocket":
                mission_out = _run_mission_modules(
                    agent=a,
                    world_truth=world_truth,
                    t_s=t_next,
                    dt_s=dt,
                    env=env_common,
                )
                launch_auth = bool(mission_out.get("launch_authorized", True))
                a.waiting_for_launch = not launch_auth
                if not launch_auth:
                    a.rocket_state.t_s = float(t_next)
                    a.truth = _rocket_state_to_truth(a.rocket_state)
                    if a.belief is not None:
                        a.belief = StateBelief(
                            state=np.hstack((a.truth.position_eci_km, a.truth.velocity_eci_km_s)),
                            covariance=a.belief.covariance,
                            last_update_t_s=t_next,
                        )
                    throttle_hist["rocket"][k] = 0.0
                    thrust_hist[aid][k + 1, :] = np.zeros(3, dtype=float)
                    torque_hist[aid][k + 1, :] = np.zeros(3, dtype=float)
                else:
                    cmd = a.rocket_guidance.command(a.rocket_state, a.rocket_sim.sim_cfg, a.rocket_sim.vehicle_cfg)
                    if "guidance_throttle" in mission_out:
                        cmd = type(cmd)(
                            throttle=float(mission_out.get("guidance_throttle", cmd.throttle)),
                            attitude_quat_bn_cmd=cmd.attitude_quat_bn_cmd,
                            torque_body_nm_cmd=cmd.torque_body_nm_cmd,
                        )
                    throttle_hist["rocket"][k] = float(np.clip(cmd.throttle, 0.0, 1.0))
                    a.rocket_state = a.rocket_sim.step(a.rocket_state, cmd, dt_s=dt)
                    a.truth = _rocket_state_to_truth(a.rocket_state)
                    if a.belief is not None:
                        a.belief = StateBelief(
                            state=np.hstack((a.truth.position_eci_km, a.truth.velocity_eci_km_s)),
                            covariance=a.belief.covariance,
                            last_update_t_s=t_next,
                        )
                    thrust_n = float(getattr(a.rocket_state, "_last_step_thrust_n", 0.0))
                    axis_eci = quaternion_to_dcm_bn(a.rocket_state.attitude_quat_bn).T @ np.array(a.rocket_sim.vehicle_cfg.thrust_axis_body, dtype=float)
                    accel = (thrust_n / max(a.rocket_state.mass_kg, 1e-9)) * axis_eci / 1e3
                    thrust_hist[aid][k + 1, :] = accel
                    torque_hist[aid][k + 1, :] = np.array(a.rocket_state.angular_rate_body_rad_s, dtype=float) * 0.0
            else:
                orbit_cfg = dict(cfg.simulator.dynamics.get("orbit", {}) or {})
                att_cfg = dict(cfg.simulator.dynamics.get("attitude", {}) or {})
                orbit_substep_s = float(max(float(orbit_cfg.get("orbit_substep_s", dt) or dt), 1e-9))
                attitude_substep_s = float(max(float(att_cfg.get("attitude_substep_s", dt) or dt), 1e-9))
                sim_substep_s = float(min(orbit_substep_s, attitude_substep_s))
                t_inner = float(t)
                tr_inner = tr_now
                cmd = Command.zero()
                while t_inner < t_next - 1e-12:
                    h = float(min(sim_substep_s, t_next - t_inner))
                    t_eval = t_inner + h
                    world_truth_inner = dict(world_truth)
                    world_truth_inner[aid] = tr_inner
                    env_inner_common = {**dict(cfg.simulator.environment or {}), "world_truth": world_truth_inner}
                    meas = a.sensor.measure(truth=tr_inner, env={"world_truth": world_truth_inner}, t_s=t_eval) if a.sensor is not None else None
                    if a.estimator is not None and a.belief is not None:
                        a.belief = a.estimator.update(a.belief, meas, t_eval)
                    elif a.belief is None:
                        a.belief = StateBelief(state=np.hstack((tr_inner.position_eci_km, tr_inner.velocity_eci_km_s)), covariance=np.eye(6) * 1e-4, last_update_t_s=t_eval)
                    orb_belief = a.belief
                    if a.orbit_controller is not None and orb_belief is not None:
                        chief_truth = world_truth_inner.get("target")
                        if chief_truth is not None and aid != "target" and hasattr(a.orbit_controller, "ric_curv_state_slice"):
                            r_c = np.array(chief_truth.position_eci_km, dtype=float)
                            v_c = np.array(chief_truth.velocity_eci_km_s, dtype=float)
                            r_s = np.array(tr_inner.position_eci_km, dtype=float)
                            v_s = np.array(tr_inner.velocity_eci_km_s, dtype=float)
                            c_ir = ric_dcm_ir_from_rv(r_c, v_c)
                            dr_ric = c_ir.T @ (r_s - r_c)
                            dv_ric = c_ir.T @ (v_s - v_c)
                            x_rect = np.hstack((dr_ric, dv_ric))
                            x_curv = ric_rect_to_curv(x_rect, r0_km=float(np.linalg.norm(r_c)))
                            orb_state = np.hstack((x_curv, np.hstack((r_c, v_c))))
                            orb_belief = StateBelief(
                                state=orb_state,
                                covariance=np.eye(12) * 1e-4,
                                last_update_t_s=orb_belief.last_update_t_s,
                            )
                    att_belief = a.belief
                    if att_belief is not None and att_belief.state.size < 13:
                        att_state = np.hstack(
                            (
                                np.array(att_belief.state[:6], dtype=float),
                                np.array(tr_inner.attitude_quat_bn, dtype=float),
                                np.array(tr_inner.angular_rate_body_rad_s, dtype=float),
                            )
                        )
                        att_belief = StateBelief(
                            state=att_state,
                            covariance=att_belief.covariance,
                            last_update_t_s=att_belief.last_update_t_s,
                        )
                    mission_out = _run_mission_modules(
                        agent=a,
                        world_truth=world_truth_inner,
                        t_s=t_eval,
                        dt_s=h,
                        env=env_inner_common,
                        orbit_controller=a.orbit_controller,
                        attitude_controller=a.attitude_controller,
                        orb_belief=orb_belief,
                        att_belief=att_belief,
                    )
                    # Mission module can set attitude targets on compatible controllers.
                    if "desired_attitude_quat_bn" in mission_out and a.attitude_controller is not None:
                        q_des = np.array(mission_out["desired_attitude_quat_bn"], dtype=float).reshape(-1)
                        if q_des.size == 4 and hasattr(a.attitude_controller, "set_target"):
                            try:
                                a.attitude_controller.set_target(q_des)
                            except Exception:
                                pass
                    if "desired_attitude_quat_br" in mission_out and a.attitude_controller is not None:
                        q_des_r = np.array(mission_out["desired_attitude_quat_br"], dtype=float).reshape(-1)
                        if q_des_r.size == 4 and hasattr(a.attitude_controller, "set_target"):
                            try:
                                a.attitude_controller.set_target(q_des_r)
                            except Exception:
                                pass
                    if "desired_ric_euler_rad" in mission_out and a.attitude_controller is not None and hasattr(a.attitude_controller, "set_desired_ric_state"):
                        e = np.array(mission_out["desired_ric_euler_rad"], dtype=float).reshape(-1)
                        if e.size == 3:
                            try:
                                a.attitude_controller.set_desired_ric_state(float(e[0]), float(e[1]), float(e[2]))
                            except Exception:
                                pass
                    use_integrated_cmd = bool(mission_out.get("mission_use_integrated_command", False))
                    c_orb = (
                        a.orbit_controller.act(orb_belief, t_eval, 2.0)
                        if (not use_integrated_cmd) and a.orbit_controller is not None and orb_belief is not None
                        else Command.zero()
                    )
                    c_att = a.attitude_controller.act(att_belief, t_eval, 2.0) if a.attitude_controller is not None and att_belief is not None else Command.zero()
                    if use_integrated_cmd:
                        cmd = Command.zero()
                        if "thrust_eci_km_s2" in mission_out:
                            cmd.thrust_eci_km_s2 = np.array(mission_out["thrust_eci_km_s2"], dtype=float).reshape(3)
                        if "torque_body_nm" in mission_out:
                            cmd.torque_body_nm = np.array(mission_out["torque_body_nm"], dtype=float).reshape(3)
                        if "command_mode_flags" in mission_out and isinstance(mission_out["command_mode_flags"], dict):
                            cmd.mode_flags.update(dict(mission_out["command_mode_flags"]))
                        cmd.mode_flags["mode"] = "mission_integrated"
                        if "mission_mode" in mission_out:
                            cmd.mode_flags["mission_mode"] = mission_out["mission_mode"]
                    else:
                        cmd = _combine_commands(c_orb, c_att)
                        if "thrust_eci_km_s2" in mission_out:
                            cmd.thrust_eci_km_s2 = np.array(mission_out["thrust_eci_km_s2"], dtype=float).reshape(3)
                        if "torque_body_nm" in mission_out:
                            cmd.torque_body_nm = np.array(mission_out["torque_body_nm"], dtype=float).reshape(3)
                    env_inner = {
                        **dict(cfg.simulator.environment or {}),
                        "world_truth": world_truth_inner,
                        "atmosphere_model": cfg.simulator.dynamics.get("rocket", {}).get("atmosphere_model", "ussa1976"),
                    }
                    tr_inner = a.dynamics.step(state=tr_inner, command=cmd, env=env_inner, dt_s=h)
                    t_inner = t_eval

                a.truth = tr_inner
                thrust_hist[aid][k + 1, :] = cmd.thrust_eci_km_s2
                torque_hist[aid][k + 1, :] = cmd.torque_body_nm

            if a.bridge is not None:
                evt = {"t_s": t_next, "object_id": aid}
                if hasattr(a.bridge, "step"):
                    try:
                        ret = a.bridge.step(evt)
                        if ret is not None:
                            evt["bridge"] = ret
                    except Exception as ex:
                        evt["bridge_error"] = str(ex)
                bridge_hist[aid].append(evt)

        if step_callback is not None:
            try:
                step_callback(k + 1, total_steps)
            except Exception:
                pass

        # Log k+1
        for aid, a in agents.items():
            if not a.active:
                continue
            tr = a.truth if a.kind == "satellite" else _rocket_state_to_truth(a.rocket_state)
            truth_hist[aid][k + 1, :] = _state_truth_to_array(tr)
            if a.belief is not None:
                belief_hist[aid][k + 1, :] = a.belief.state[:6]

        if bool(cfg.simulator.termination.get("earth_impact_enabled", True)):
            re = float(cfg.simulator.termination.get("earth_radius_km", EARTH_RADIUS_KM))
            for aid, a in agents.items():
                if not a.active:
                    continue
                if a.kind == "rocket" and a.waiting_for_launch:
                    continue
                tr = a.truth if a.kind == "satellite" else _rocket_state_to_truth(a.rocket_state)
                if float(np.linalg.norm(tr.position_eci_km)) <= re:
                    terminated_early = True
                    termination_reason = "earth_impact"
                    termination_time_s = t_next
                    termination_object_id = aid
                    final_index = k + 1
                    break
            if terminated_early:
                break

    n_used = final_index + 1
    t_out = t_s[:n_used].copy()
    truth_out = {k: v[:n_used, :].copy() for k, v in truth_hist.items()}
    belief_out = {k: v[:n_used, :].copy() for k, v in belief_hist.items()}
    thrust_out = {k: v[:n_used, :].copy() for k, v in thrust_hist.items()}
    torque_out = {k: v[:n_used, :].copy() for k, v in torque_hist.items()}
    knowledge_out = {obs: {tgt: arr[:n_used, :].copy() for tgt, arr in by_tgt.items()} for obs, by_tgt in knowledge_hist.items()}

    plot_outputs = _plot_outputs(
        cfg=cfg,
        t_s=t_out,
        truth_hist=truth_out,
        thrust_hist=thrust_out,
        knowledge_hist=knowledge_out,
        outdir=outdir,
    )
    if cfg.outputs.mode in ("interactive", "both") and bool(cfg.outputs.plots.get("enabled", True)):
        import matplotlib.pyplot as plt

        plt.show()
    animation_outputs = _animate_outputs(
        cfg=cfg,
        t_s=t_out,
        truth_hist=truth_out,
        outdir=outdir,
    )

    thrust_stats: dict[str, dict[str, float | int]] = {}
    for oid, u in thrust_out.items():
        mag = np.linalg.norm(np.nan_to_num(u, nan=0.0), axis=1)
        burn_mask = mag > 1e-15
        thrust_stats[oid] = {
            "burn_samples": int(np.sum(burn_mask)),
            "max_accel_km_s2": float(np.max(mag)) if mag.size else 0.0,
            "total_dv_m_s": float(np.sum(mag) * float(dt) * 1e3),
        }

    summary = {
        "scenario_name": cfg.scenario_name,
        "objects": sorted(list(agents.keys())),
        "samples": int(n_used),
        "dt_s": dt,
        "duration_s": float(t_out[-1]) if t_out.size else 0.0,
        "terminated_early": terminated_early,
        "termination_reason": termination_reason,
        "termination_time_s": termination_time_s,
        "termination_object_id": termination_object_id,
        "thrust_stats": thrust_stats,
        "plot_outputs": plot_outputs,
        "animation_outputs": animation_outputs,
    }

    payload = {
        "summary": summary,
        "time_s": t_out.tolist(),
        "truth_by_object": {k: v.tolist() for k, v in truth_out.items()},
        "belief_by_object": {k: v.tolist() for k, v in belief_out.items()},
        "applied_thrust_by_object": {k: v.tolist() for k, v in thrust_out.items()},
        "applied_torque_by_object": {k: v.tolist() for k, v in torque_out.items()},
        "knowledge_by_observer": {o: {t: a.tolist() for t, a in bt.items()} for o, bt in knowledge_out.items()},
        "bridge_events_by_object": bridge_hist,
        "rocket_throttle_cmd": throttle_hist.get("rocket", np.array([])).tolist() if throttle_hist else [],
    }
    if bool(cfg.outputs.stats.get("save_json", True)):
        write_json(str(outdir / "master_run_summary.json"), summary)
    if bool(cfg.outputs.stats.get("save_full_log", True)):
        write_json(str(outdir / "master_run_log.json"), payload)
    if bool(cfg.outputs.stats.get("print_summary", True)):
        print("Master simulation summary:")
        for k, v in summary.items():
            if k != "plot_outputs":
                print(f"  {k}: {v}")
    return payload


def run_master_simulation(
    config_path: str | Path,
    step_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    cfg = load_simulation_yaml(config_path)
    strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))
    if strict_plugins:
        errs = validate_scenario_plugins(cfg)
        if errs:
            msg = "Plugin validation failed:\n- " + "\n- ".join(errs)
            raise ValueError(msg)
    if not cfg.monte_carlo.enabled:
        out = _run_single_config(cfg, step_callback=step_callback)
        return {
            "config_path": str(Path(config_path).resolve()),
            "scenario_name": cfg.scenario_name,
            "monte_carlo": {"enabled": False},
            "run": out["summary"],
        }

    root = cfg.to_dict()
    outdir = Path(cfg.outputs.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg.monte_carlo.base_seed))
    runs = []
    for i in range(int(cfg.monte_carlo.iterations)):
        cdict = deepcopy(root)
        sampled = {}
        for v in cfg.monte_carlo.variations:
            sv = _sample_variation(v, rng)
            _deep_set(cdict, v.parameter_path, sv)
            sampled[v.parameter_path] = sv
        # Prevent unwanted pop-up windows in MC mode unless explicitly both/save.
        mode = str(cdict.get("outputs", {}).get("mode", "interactive"))
        if mode == "interactive":
            cdict.setdefault("outputs", {})["mode"] = "save"
        cdict.setdefault("outputs", {})["output_dir"] = str(outdir / f"mc_run_{i:04d}")
        ci = scenario_config_from_dict(cdict)
        if strict_plugins:
            errs = validate_scenario_plugins(ci)
            if errs:
                msg = "Plugin validation failed in Monte Carlo iteration {i}:\n- ".format(i=i) + "\n- ".join(errs)
                raise ValueError(msg)
        ro = _run_single_config(ci)
        entry = {"iteration": i, "sampled_parameters": sampled, "summary": ro["summary"]}
        runs.append(entry)
        if bool(cfg.outputs.monte_carlo.get("save_iteration_summaries", False)):
            write_json(str(outdir / f"master_monte_carlo_run_{i:04d}.json"), entry)

    agg = {
        "config_path": str(Path(config_path).resolve()),
        "scenario_name": cfg.scenario_name,
        "monte_carlo": {"enabled": True, "iterations": int(cfg.monte_carlo.iterations), "base_seed": int(cfg.monte_carlo.base_seed)},
        "runs": runs,
    }
    if bool(cfg.outputs.monte_carlo.get("save_aggregate_summary", True)):
        write_json(str(outdir / "master_monte_carlo_summary.json"), agg)
    return agg
