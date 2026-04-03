from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import inspect
import logging
from typing import Any, Callable

import numpy as np

from sim.presets.rockets import BASIC_1ST_STAGE, BASIC_SSTO_ROCKET, BASIC_TWO_STAGE_STACK, RocketStackPreset
from sim.presets.thrusters import BASIC_CHEMICAL_BOTTOM_Z, resolve_thruster_mount_from_specs
from sim.config import SimulationScenarioConfig
from sim.control.attitude.zero_torque import ZeroTorqueController
from sim.control.orbit.zero_controller import ZeroController
from sim.core.models import Command, StateBelief, StateTruth
from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import eci_to_ecef
from sim.dynamics.orbit.propagator import (
    OrbitPropagator,
    drag_plugin,
    j2_plugin,
    j3_plugin,
    j4_plugin,
    spherical_harmonics_plugin,
    srp_plugin,
    third_body_moon_plugin,
    third_body_sun_plugin,
)
from sim.estimation.joint_state import JointStateEstimator
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.knowledge.object_tracking import (
    KnowledgeConditionConfig,
    KnowledgeEKFConfig,
    KnowledgeNoiseConfig,
    ObjectKnowledgeBase,
    TrackedObjectConfig,
)
from sim.rocket import (
    MaxQThrottleLimiterGuidance,
    OpenLoopPitchProgramGuidance,
    OrbitInsertionCutoffGuidance,
    RocketAscentSimulator,
    RocketGuidanceLaw,
    RocketSimConfig,
    RocketState,
    RocketVehicleConfig,
    TVCSteeringGuidance,
)
from sim.rocket.aero import RocketAeroConfig
from sim.sensors.joint_state import JointStateSensor
from sim.sensors.noisy_own_state import NoisyOwnStateSensor
from sim.utils.frames import eci_relative_to_ric_rect, ric_curv_to_rect, ric_rect_state_to_eci, ric_rect_to_curv
from sim.utils.geodesy import ecef_to_geodetic_deg_km
from sim.utils.quaternion import quaternion_to_dcm_bn

logger = logging.getLogger(__name__)


def _module_obj(pointer, *, extra_kwargs: dict[str, Any] | None = None) -> Any | None:
    if pointer is None or pointer.module is None:
        return None
    try:
        mod = importlib.import_module(pointer.module)
        if pointer.class_name:
            cls = getattr(mod, pointer.class_name)
            kwargs = dict(pointer.params or {})
            if extra_kwargs:
                kwargs.update(dict(extra_kwargs))
            return cls(**kwargs)
        if pointer.function:
            return getattr(mod, pointer.function)
        return mod
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.warning("Failed to construct plugin pointer %r: %s", pointer, exc)
        return None


def _compatible_keyword_args(method: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any] | None:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return None

    accepts_var_kwargs = False
    filtered: dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_kwargs = True
            continue
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            return None
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY) and name in kwargs:
            filtered[name] = kwargs[name]

    for name, param in signature.parameters.items():
        if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        if param.default is inspect.Signature.empty and name not in filtered:
            return None

    return dict(kwargs) if accepts_var_kwargs else filtered


def _call_with_compat_kwargs(
    method: Callable[..., Any],
    *,
    primary_kwargs: dict[str, Any],
    fallback_kwargs: dict[str, Any] | None = None,
) -> Any:
    compatible = _compatible_keyword_args(method, primary_kwargs)
    if compatible is not None:
        return method(**compatible)
    if fallback_kwargs is not None:
        compatible = _compatible_keyword_args(method, fallback_kwargs)
        if compatible is not None:
            return method(**compatible)
    return method(**primary_kwargs)


def _to_jsonable_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable_value(v) for v in value]
    return value


def _command_to_dict(cmd: Command) -> dict[str, Any]:
    return {
        "thrust_eci_km_s2": np.array(cmd.thrust_eci_km_s2, dtype=float).tolist(),
        "torque_body_nm": np.array(cmd.torque_body_nm, dtype=float).tolist(),
        "mode_flags": _to_jsonable_value(dict(cmd.mode_flags or {})),
    }


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


def _rocket_state_to_truth(s: RocketState) -> StateTruth:
    return StateTruth(
        position_eci_km=np.array(s.position_eci_km, dtype=float),
        velocity_eci_km_s=np.array(s.velocity_eci_km_s, dtype=float),
        attitude_quat_bn=np.array(s.attitude_quat_bn, dtype=float),
        angular_rate_body_rad_s=np.array(s.angular_rate_body_rad_s, dtype=float),
        mass_kg=float(s.mass_kg),
        t_s=float(s.t_s),
    )


def _truth_state6(truth: StateTruth, out: np.ndarray | None = None) -> np.ndarray:
    state = np.empty(6, dtype=float) if out is None else out
    state[0:3] = truth.position_eci_km
    state[3:6] = truth.velocity_eci_km_s
    return state


def _attitude_state13_from_belief(
    belief: StateBelief,
    truth: StateTruth,
    out: np.ndarray | None = None,
) -> np.ndarray:
    state = np.empty(13, dtype=float) if out is None else out
    state[0:6] = belief.state[:6]
    state[6:10] = truth.attitude_quat_bn
    state[10:13] = truth.angular_rate_body_rad_s
    return state


def _relative_orbit_state12(
    chief_truth: StateTruth,
    deputy_truth: StateTruth,
    out: np.ndarray | None = None,
    deputy_state6: np.ndarray | None = None,
    chief_state6: np.ndarray | None = None,
) -> np.ndarray:
    state = np.empty(12, dtype=float) if out is None else out
    r_c = chief_truth.position_eci_km
    v_c = chief_truth.velocity_eci_km_s
    x_dep_eci = np.empty(6, dtype=float) if deputy_state6 is None else deputy_state6
    x_chief_eci = np.empty(6, dtype=float) if chief_state6 is None else chief_state6
    x_dep_eci[0:3] = deputy_truth.position_eci_km
    x_dep_eci[3:6] = deputy_truth.velocity_eci_km_s
    x_chief_eci[0:3] = r_c
    x_chief_eci[3:6] = v_c
    x_rect = eci_relative_to_ric_rect(x_dep_eci=x_dep_eci, x_chief_eci=x_chief_eci)
    state[0:6] = ric_rect_to_curv(x_rect, r0_km=float(np.linalg.norm(r_c)))
    state[6:9] = r_c
    state[9:12] = v_c
    return state


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
    return q_pf_to_eci @ r_pf, q_pf_to_eci @ v_pf


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
        return _coe_to_rv_eci(
            a_km=float(d.get("a_km", d.get("semi_major_axis_km", 7000.0))),
            ecc=float(d.get("ecc", d.get("e", 0.0))),
            inc_deg=float(d.get("inc_deg", d.get("inclination_deg", 0.0))),
            raan_deg=float(d.get("raan_deg", 0.0)),
            argp_deg=float(d.get("argp_deg", d.get("arg_periapsis_deg", 0.0))),
            true_anomaly_deg=float(d.get("ta_deg", d.get("true_anomaly_deg", 0.0))),
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


def _resolve_satellite_isp_s(specs: dict[str, Any]) -> float:
    if "isp_s" in specs:
        return float(specs.get("isp_s", 0.0))
    if "thruster_isp_s" in specs:
        return float(specs.get("thruster_isp_s", 0.0))
    thr = str(specs.get("thruster", "")).strip().upper()
    if thr in ("BASIC_CHEMICAL_BOTTOM_Z", "BASIC_CHEMICAL_Z_BOTTOM"):
        return float(BASIC_CHEMICAL_BOTTOM_Z.isp_s)
    return 0.0


def _resolve_satellite_inertia_kg_m2(specs: dict[str, Any]) -> np.ndarray:
    mp = dict(specs.get("mass_properties", {}) or {})
    if "inertia_kg_m2" in mp:
        inertia = np.array(mp.get("inertia_kg_m2"), dtype=float)
        if inertia.shape == (3, 3) and np.all(np.isfinite(inertia)):
            return inertia
    return np.diag([120.0, 100.0, 80.0])


def _apply_thruster_mount_defaults(module_obj: Any | None, pointer: Any | None, specs: dict[str, Any]) -> Any | None:
    if module_obj is None:
        return None
    mount = resolve_thruster_mount_from_specs(specs)
    if mount is None:
        return module_obj
    params = dict(getattr(pointer, "params", {}) or {}) if pointer is not None else {}
    if hasattr(module_obj, "thruster_direction_body") and "thruster_direction_body" not in params:
        try:
            module_obj.thruster_direction_body = np.array(mount.thrust_direction_body, dtype=float)
        except (TypeError, ValueError, AttributeError):
            pass
    if hasattr(module_obj, "thruster_position_body_m") and "thruster_position_body_m" not in params:
        try:
            module_obj.thruster_position_body_m = np.array(mount.position_body_m, dtype=float)
        except (TypeError, ValueError, AttributeError):
            pass
    return module_obj


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


@dataclass
class AgentRuntime:
    object_id: str
    kind: str
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
    mission_strategy: Any | None
    mission_execution: Any | None
    rocket_sim: RocketAscentSimulator | None
    rocket_state: RocketState | None
    rocket_guidance: Any | None
    deploy_source: str | None
    deploy_time_s: float | None
    deploy_dv_body_m_s: np.ndarray | None
    mission_modules: list[Any]
    waiting_for_launch: bool
    orbital_isp_s: float | None = None
    dry_mass_kg: float | None = None
    fuel_capacity_kg: float | None = None
    thruster_direction_body: np.ndarray | None = None
    thruster_position_body_m: np.ndarray | None = None


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


def _apply_chaser_relative_init_from_target(
    *,
    chaser: AgentRuntime,
    target: AgentRuntime,
    initial_state: dict[str, Any],
) -> None:
    rel = _resolve_chaser_relative_ric_init(initial_state)
    if rel is None or chaser.truth is None or target.truth is None:
        return
    x_rel, frame = rel
    r_t = np.array(target.truth.position_eci_km, dtype=float)
    v_t = np.array(target.truth.velocity_eci_km_s, dtype=float)
    r0 = float(np.linalg.norm(r_t))
    if r0 <= 0.0:
        return
    x_rel_rect = ric_curv_to_rect(x_rel, r0_km=r0) if frame == "curv" else np.array(x_rel, dtype=float).reshape(6)
    x_chaser_eci = ric_rect_state_to_eci(x_rel_rect, r_t, v_t)
    chaser.truth.position_eci_km = x_chaser_eci[:3]
    chaser.truth.velocity_eci_km_s = x_chaser_eci[3:]


def _build_orbit_propagator(cfg: SimulationScenarioConfig) -> OrbitPropagator:
    orbit = dict(cfg.simulator.dynamics.get("orbit", {}) or {})
    sh = dict(orbit.get("spherical_harmonics", {}) or {})
    plugins = []
    if bool(orbit.get("j2", False)):
        plugins.append(j2_plugin)
    if bool(orbit.get("j3", False)):
        plugins.append(j3_plugin)
    if bool(orbit.get("j4", False)):
        plugins.append(j4_plugin)
    if bool(sh.get("enabled", False)):
        plugins.append(spherical_harmonics_plugin)
    if bool(orbit.get("drag", False)):
        plugins.append(drag_plugin)
    if bool(orbit.get("srp", False)):
        plugins.append(srp_plugin)
    if bool(orbit.get("third_body_sun", False)):
        plugins.append(third_body_sun_plugin)
    if bool(orbit.get("third_body_moon", False)):
        plugins.append(third_body_moon_plugin)
    return OrbitPropagator(
        integrator=str(orbit.get("integrator", "rk4")),
        plugins=plugins,
        adaptive_atol=float(orbit.get("adaptive_atol", 1e-9)),
        adaptive_rtol=float(orbit.get("adaptive_rtol", 1e-7)),
    )


def _create_satellite_runtime(
    object_id: str,
    agent_cfg: Any,
    cfg: SimulationScenarioConfig,
    rng: np.random.Generator,
) -> AgentRuntime:
    truth = _default_truth_from_agent(agent_cfg, t_s=0.0)
    specs = dict(agent_cfg.specs or {})
    inertia_kg_m2 = _resolve_satellite_inertia_kg_m2(specs)
    noise = dict((agent_cfg.knowledge or {}).get("sensor_error", {}) or {})
    pos_sigma = float(np.array(noise.get("pos_sigma_km", [0.001])).reshape(-1)[0])
    vel_sigma = float(np.array(noise.get("vel_sigma_km_s", [1e-5])).reshape(-1)[0])
    orbit_estimator = OrbitEKFEstimator(
        mu_km3_s2=EARTH_MU_KM3_S2,
        dt_s=float(cfg.simulator.dt_s),
        process_noise_diag=np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10]),
        meas_noise_diag=np.array([1e-6, 1e-6, 1e-6, 1e-10, 1e-10, 1e-10]),
    )
    orbit_ctrl_base = _module_obj(agent_cfg.orbit_control) or ZeroController()
    att_ctrl_base = _module_obj(agent_cfg.attitude_control) or ZeroTorqueController()
    orbit_cfg = dict(cfg.simulator.dynamics.get("orbit", {}) or {})
    att_cfg = dict(cfg.simulator.dynamics.get("attitude", {}) or {})
    attitude_enabled = bool(att_cfg.get("enabled", True))
    if attitude_enabled:
        belief = StateBelief(
            state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s, truth.attitude_quat_bn, truth.angular_rate_body_rad_s)),
            covariance=np.eye(13) * 1e-4,
            last_update_t_s=0.0,
        )
        quat_sigma = float(np.array(noise.get("quat_sigma", [1e-3])).reshape(-1)[0])
        omega_sigma = float(np.array(noise.get("omega_sigma_rad_s", [1e-4])).reshape(-1)[0])
        sensor = JointStateSensor(
            pos_sigma_km=pos_sigma,
            vel_sigma_km_s=vel_sigma,
            quat_sigma=quat_sigma,
            omega_sigma_rad_s=omega_sigma,
            update_cadence_s=float(cfg.simulator.dt_s),
            rng=rng,
        )
        estimator = JointStateEstimator(orbit_estimator=orbit_estimator, dt_s=float(cfg.simulator.dt_s))
    else:
        belief = StateBelief(
            state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)),
            covariance=np.eye(6) * 1e-4,
            last_update_t_s=0.0,
        )
        sensor = NoisyOwnStateSensor(pos_sigma_km=pos_sigma, vel_sigma_km_s=vel_sigma, rng=rng)
        estimator = orbit_estimator
    dist_cfg = dict(att_cfg.get("disturbance_torques", {}) or {})
    orbit_ctrl = _RateLimitedController(
        base=orbit_ctrl_base,
        period_s=float(max(float(orbit_cfg.get("orbit_substep_s", cfg.simulator.dt_s) or cfg.simulator.dt_s), 1e-9)),
    )
    att_ctrl = (
        _RateLimitedController(
            base=att_ctrl_base,
            period_s=float(max(float(att_cfg.get("attitude_substep_s", cfg.simulator.dt_s) or cfg.simulator.dt_s), 1e-9)),
        )
        if attitude_enabled
        else None
    )
    disturbance_model = DisturbanceTorqueModel(
        mu_km3_s2=EARTH_MU_KM3_S2,
        inertia_kg_m2=inertia_kg_m2,
        config=DisturbanceTorqueConfig(
            use_gravity_gradient=bool(dist_cfg.get("gravity_gradient", False)),
            use_magnetic=bool(dist_cfg.get("magnetic", False)),
            use_drag=bool(dist_cfg.get("drag", False)),
            use_srp=bool(dist_cfg.get("srp", False)),
        ),
    )
    dynamics = OrbitalAttitudeDynamics(
        mu_km3_s2=EARTH_MU_KM3_S2,
        inertia_kg_m2=inertia_kg_m2,
        disturbance_model=disturbance_model if attitude_enabled else None,
        orbit_substep_s=float(orbit_cfg["orbit_substep_s"]) if orbit_cfg.get("orbit_substep_s") is not None else None,
        attitude_substep_s=float(att_cfg["attitude_substep_s"]) if att_cfg.get("attitude_substep_s") is not None else None,
        propagate_attitude=attitude_enabled,
        orbit_propagator=_build_orbit_propagator(cfg),
    )
    bridge = _module_obj(agent_cfg.bridge) if (agent_cfg.bridge is not None and agent_cfg.bridge.enabled) else None
    mission_strategy_pointer = getattr(agent_cfg, "mission_strategy", None)
    mission_execution_pointer = getattr(agent_cfg, "mission_execution", None)
    mission_strategy = _module_obj(mission_strategy_pointer)
    mission_execution = _apply_thruster_mount_defaults(_module_obj(mission_execution_pointer), mission_execution_pointer, specs)
    mission_modules = [_module_obj(pointer) for pointer in list(agent_cfg.mission_objectives or [])]
    mission_modules = [module for module in mission_modules if module is not None]
    sat_isp_s = _resolve_satellite_isp_s(specs)
    dry_mass_kg = specs.get("dry_mass_kg")
    fuel_capacity_kg = specs.get("fuel_mass_kg")
    thruster_mount = resolve_thruster_mount_from_specs(specs)
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
        dynamics=dynamics,
        knowledge_base=None,
        bridge=bridge,
        mission_strategy=mission_strategy,
        mission_execution=mission_execution,
        rocket_sim=None,
        rocket_state=None,
        rocket_guidance=None,
        deploy_source=str((agent_cfg.initial_state or {}).get("source", "")) or None,
        deploy_time_s=float((agent_cfg.initial_state or {}).get("deploy_time_s", 0.0)),
        deploy_dv_body_m_s=np.array((agent_cfg.initial_state or {}).get("deploy_dv_body_m_s", [0.0, 0.0, 0.0]), dtype=float),
        mission_modules=mission_modules,
        waiting_for_launch=False,
        orbital_isp_s=(None if sat_isp_s <= 0.0 else float(sat_isp_s)),
        dry_mass_kg=(None if dry_mass_kg is None else float(dry_mass_kg)),
        fuel_capacity_kg=(None if fuel_capacity_kg is None else float(fuel_capacity_kg)),
        thruster_direction_body=(None if thruster_mount is None else np.array(thruster_mount.thrust_direction_body, dtype=float)),
        thruster_position_body_m=(None if thruster_mount is None else np.array(thruster_mount.position_body_m, dtype=float)),
    )


def _resolve_rocket_stack(specs: dict[str, Any]) -> RocketStackPreset:
    preset = str(specs.get("preset_stack", "BASIC_TWO_STAGE_STACK")).strip().upper()
    ssto_stack = RocketStackPreset(name="Basic SSTO Stack", stages=(BASIC_SSTO_ROCKET,))
    by_name: dict[str, RocketStackPreset] = {
        "BASIC_TWO_STAGE_STACK": BASIC_TWO_STAGE_STACK,
        "BASIC_SSTO_STACK": ssto_stack,
        "BASIC_SSTO_ROCKET": ssto_stack,
        "BASIC_1ST_STAGE_STACK": RocketStackPreset(name="Basic 1st Stage Stack", stages=(BASIC_1ST_STAGE,)),
    }
    if preset not in by_name:
        valid = ", ".join(sorted(by_name.keys()))
        raise ValueError(f"Unknown rocket.specs.preset_stack '{preset}'. Valid options: {valid}")
    return by_name[preset]


def _build_rocket_guidance(agent_cfg: Any) -> RocketGuidanceLaw:
    base_pointer = getattr(agent_cfg, "base_guidance", None) or getattr(agent_cfg, "guidance", None)
    guidance = _module_obj(base_pointer) or OpenLoopPitchProgramGuidance()
    for modifier_pointer in list(getattr(agent_cfg, "guidance_modifiers", []) or []):
        modifier_obj = _module_obj(modifier_pointer, extra_kwargs={"base_guidance": guidance})
        if modifier_obj is not None:
            guidance = modifier_obj
    return guidance


def _create_rocket_runtime(cfg: SimulationScenarioConfig) -> AgentRuntime:
    rc = cfg.rocket
    r_init = dict(rc.initial_state or {})
    r_specs = dict(rc.specs or {})
    orbit_dyn = dict(cfg.simulator.dynamics.get("orbit", {}) or {})
    att_dyn = dict(cfg.simulator.dynamics.get("attitude", {}) or {})
    rocket_dyn = dict(cfg.simulator.dynamics.get("rocket", {}) or {})
    aero_dyn = dict(rocket_dyn.get("aero", {}) or {})
    atmosphere_env = dict(cfg.simulator.environment.get("atmosphere_env", {}) or {})
    aero_cfg = RocketAeroConfig(
        enabled=bool(rocket_dyn.get("aero_model_enabled", True)),
        reference_area_m2=float(aero_dyn.get("reference_area_m2", 10.0)),
        reference_length_m=float(aero_dyn.get("reference_length_m", 30.0)),
        cp_offset_body_m=np.array(aero_dyn.get("cp_offset_body_m", [0.0, 0.0, 0.0]), dtype=float),
        cd_base=float(aero_dyn.get("cd_base", 0.20)),
        cd_alpha2=float(aero_dyn.get("cd_alpha2", 0.10)),
        cd_supersonic=float(aero_dyn.get("cd_supersonic", 0.28)),
        transonic_peak_cd=float(aero_dyn.get("transonic_peak_cd", 0.22)),
        transonic_mach=float(aero_dyn.get("transonic_mach", 1.0)),
        transonic_width=float(aero_dyn.get("transonic_width", 0.22)),
        cl_alpha_per_rad=float(aero_dyn.get("cl_alpha_per_rad", 0.15)),
        cy_beta_per_rad=float(aero_dyn.get("cy_beta_per_rad", 0.15)),
        cm_alpha_per_rad=float(aero_dyn.get("cm_alpha_per_rad", -0.02)),
        cn_beta_per_rad=float(aero_dyn.get("cn_beta_per_rad", -0.02)),
        cl_roll_per_rad=float(aero_dyn.get("cl_roll_per_rad", -0.01)),
        alpha_limit_deg=float(aero_dyn.get("alpha_limit_deg", 20.0)),
        beta_limit_deg=float(aero_dyn.get("beta_limit_deg", 20.0)),
    )
    sim_cfg = RocketSimConfig(
        dt_s=float(cfg.simulator.dt_s),
        max_time_s=float(cfg.simulator.duration_s),
        target_altitude_km=float(rocket_dyn.get("target_altitude_km", 400.0)),
        target_altitude_tolerance_km=float(rocket_dyn.get("target_altitude_tolerance_km", 25.0)),
        target_eccentricity_max=float(rocket_dyn.get("target_eccentricity_max", 0.02)),
        insertion_hold_time_s=float(rocket_dyn.get("insertion_hold_time_s", 30.0)),
        launch_lat_deg=float(r_init.get("launch_lat_deg", 0.0)),
        launch_lon_deg=float(r_init.get("launch_lon_deg", 0.0)),
        launch_alt_km=float(r_init.get("launch_alt_km", 0.0)),
        launch_azimuth_deg=float(r_init.get("launch_azimuth_deg", 90.0)),
        atmosphere_model=str(rocket_dyn.get("atmosphere_model", "ussa1976")),
        enable_drag=bool(orbit_dyn.get("drag", True)),
        enable_srp=bool(orbit_dyn.get("srp", False)),
        enable_j2=bool(orbit_dyn.get("j2", True)),
        enable_j3=bool(orbit_dyn.get("j3", False)),
        enable_j4=bool(orbit_dyn.get("j4", False)),
        terminate_on_earth_impact=bool(cfg.simulator.termination.get("earth_impact_enabled", True)),
        earth_impact_radius_km=float(cfg.simulator.termination.get("earth_radius_km", 6378.137)),
        area_ref_m2=(None if rocket_dyn.get("area_ref_m2") is None else float(rocket_dyn.get("area_ref_m2"))),
        use_stagewise_aero_geometry=bool(rocket_dyn.get("use_stagewise_aero_geometry", True)),
        cd=float(rocket_dyn.get("cd", 0.35)),
        cr=float(rocket_dyn.get("cr", 1.2)),
        aero=aero_cfg,
        atmosphere_env=atmosphere_env,
        use_wgs84_geodesy=bool(rocket_dyn.get("use_wgs84_geodesy", True)),
        wind_enu_m_s=np.array(rocket_dyn.get("wind_enu_m_s", [0.0, 0.0, 0.0]), dtype=float),
        inertia_kg_m2=np.array(
            (r_specs.get("mass_properties", {}) or {}).get(
                "inertia_kg_m2",
                [[8.0e5, 0.0, 0.0], [0.0, 8.0e5, 0.0], [0.0, 0.0, 2.0e4]],
            ),
            dtype=float,
        ),
        attitude_substep_s=float(rocket_dyn.get("attitude_substep_s", att_dyn.get("attitude_substep_s", 0.02)) or 0.02),
        attitude_mode=str(rocket_dyn.get("attitude_mode", "dynamic")),
        tvc_time_constant_s=float(rocket_dyn.get("tvc_time_constant_s", 0.1)),
        tvc_max_gimbal_deg=float(rocket_dyn.get("tvc_max_gimbal_deg", 6.0)),
        tvc_rate_limit_deg_s=float(rocket_dyn.get("tvc_rate_limit_deg_s", 20.0)),
        tvc_pivot_offset_body_m=np.array(rocket_dyn.get("tvc_pivot_offset_body_m", [0.0, 0.0, 0.0]), dtype=float),
    )
    vehicle_cfg = RocketVehicleConfig(
        stack=_resolve_rocket_stack(dict(rc.specs or {})),
        payload_mass_kg=float(r_specs.get("payload_mass_kg", 150.0)),
        thrust_axis_body=np.array(r_specs.get("thrust_axis_body", [1.0, 0.0, 0.0]), dtype=float),
    )
    guidance = _build_rocket_guidance(rc)
    if bool(rocket_dyn.get("tvc_steering_enabled", False)):
        guidance = TVCSteeringGuidance(base_guidance=guidance, pass_through_attitude=bool(rocket_dyn.get("tvc_pass_through_attitude", True)))
    if bool(rocket_dyn.get("orbit_insertion_cutoff_enabled", False)):
        guidance = OrbitInsertionCutoffGuidance(
            base_guidance=guidance,
            min_cutoff_alt_km=float(rocket_dyn.get("cutoff_min_alt_km", 80.0)),
            min_periapsis_alt_km=float(rocket_dyn.get("cutoff_min_periapsis_alt_km", 120.0)),
            apoapsis_margin_km=float(rocket_dyn.get("cutoff_apoapsis_margin_km", 5.0)),
            energy_margin_km2_s2=float(rocket_dyn.get("cutoff_energy_margin_km2_s2", 0.0)),
            ecc_relax_factor=float(rocket_dyn.get("cutoff_ecc_relax_factor", 2.0)),
            hard_escape_cutoff=bool(rocket_dyn.get("cutoff_hard_escape_enabled", True)),
            near_escape_speed_margin_frac=float(rocket_dyn.get("cutoff_near_escape_speed_margin_frac", 0.03)),
        )
    if bool(rocket_dyn.get("max_q_limiter_enabled", False)):
        guidance = MaxQThrottleLimiterGuidance(
            base_guidance=guidance,
            max_q_pa=float(rocket_dyn.get("max_q_pa", 45000.0)),
            min_throttle=float(rocket_dyn.get("min_throttle", 0.0)),
        )
    rocket_sim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=guidance)
    rocket_state = rocket_sim.initial_state()
    truth = _rocket_state_to_truth(rocket_state)
    belief = StateBelief(
        state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)),
        covariance=np.eye(6) * 1e-4,
        last_update_t_s=0.0,
    )
    bridge = _module_obj(rc.bridge) if (rc.bridge is not None and rc.bridge.enabled) else None
    mission_strategy = _module_obj(getattr(rc, "mission_strategy", None))
    mission_execution = _module_obj(getattr(rc, "mission_execution", None))
    mission_modules = [_module_obj(pointer) for pointer in list(rc.mission_objectives or [])]
    mission_modules = [module for module in mission_modules if module is not None]
    return AgentRuntime(
        object_id="rocket",
        kind="rocket",
        enabled=bool(rc.enabled),
        active=bool(rc.enabled),
        truth=truth,
        belief=belief,
        sensor=None,
        estimator=None,
        orbit_controller=None,
        attitude_controller=None,
        dynamics=None,
        knowledge_base=None,
        bridge=bridge,
        mission_strategy=mission_strategy,
        mission_execution=mission_execution,
        rocket_sim=rocket_sim,
        rocket_state=rocket_state,
        rocket_guidance=guidance,
        deploy_source=None,
        deploy_time_s=None,
        deploy_dv_body_m_s=None,
        mission_modules=mission_modules,
        waiting_for_launch=False,
        orbital_isp_s=None,
        dry_mass_kg=None,
        fuel_capacity_kg=None,
        thruster_direction_body=None,
        thruster_position_body_m=None,
    )


def _build_knowledge_base(observer_id: str, agent_cfg: Any, dt_s: float, rng: np.random.Generator) -> ObjectKnowledgeBase | None:
    knowledge = dict(agent_cfg.knowledge or {})
    targets = list(knowledge.get("targets", []) or [])
    if not targets:
        return None
    conditions = dict(knowledge.get("conditions", {}) or {})
    noise = dict(knowledge.get("sensor_error", {}) or {})
    estimation = dict(knowledge.get("estimation", {}) or {})
    tracked: list[TrackedObjectConfig] = []
    for target_id in targets:
        tracked.append(
            TrackedObjectConfig(
                target_id=str(target_id),
                conditions=KnowledgeConditionConfig(
                    refresh_rate_s=float(knowledge.get("refresh_rate_s", dt_s)),
                    max_range_km=conditions.get("max_range_km"),
                    fov_half_angle_rad=conditions.get("fov_half_angle_rad"),
                    solid_angle_sr=conditions.get("solid_angle_sr"),
                    require_line_of_sight=bool(conditions.get("require_line_of_sight", False)),
                    dropout_prob=float(conditions.get("dropout_prob", 0.0)),
                    sensor_position_body_m=np.array(conditions.get("sensor_position_body_m", [0.0, 0.0, 0.0]), dtype=float),
                    sensor_boresight_body=(
                        np.array(conditions.get("sensor_boresight_body"), dtype=float)
                        if conditions.get("sensor_boresight_body") is not None
                        else None
                    ),
                ),
                sensor_noise=KnowledgeNoiseConfig(
                    pos_sigma_km=np.array(noise.get("pos_sigma_km", [0.01, 0.01, 0.01]), dtype=float),
                    vel_sigma_km_s=np.array(noise.get("vel_sigma_km_s", [1e-4, 1e-4, 1e-4]), dtype=float),
                    pos_bias_km=np.array(noise.get("pos_bias_km", [0.0, 0.0, 0.0]), dtype=float),
                    vel_bias_km_s=np.array(noise.get("vel_bias_km_s", [0.0, 0.0, 0.0]), dtype=float),
                ),
                estimator=str(estimation.get("type", "ekf")),
                ekf=KnowledgeEKFConfig(),
            )
        )
    return ObjectKnowledgeBase(observer_id=observer_id, tracked_objects=tracked, dt_s=dt_s, rng=rng, mu_km3_s2=EARTH_MU_KM3_S2)


def _deploy_from_rocket(agent: AgentRuntime, rocket: AgentRuntime, t_next: float) -> None:
    if agent.kind != "satellite" or agent.active or agent.deploy_source != "rocket_deployment" or rocket.rocket_state is None:
        return
    c_bn = quaternion_to_dcm_bn(rocket.rocket_state.attitude_quat_bn)
    dv_body = np.array(agent.deploy_dv_body_m_s if agent.deploy_dv_body_m_s is not None else np.zeros(3), dtype=float)
    dv_eci_km_s = (c_bn.T @ dv_body) / 1e3
    rs = rocket.rocket_state
    mass_kg = float(agent.truth.mass_kg) if agent.truth is not None else 200.0
    agent.truth = StateTruth(
        position_eci_km=np.array(rs.position_eci_km, dtype=float),
        velocity_eci_km_s=np.array(rs.velocity_eci_km_s, dtype=float) + dv_eci_km_s,
        attitude_quat_bn=np.array(rs.attitude_quat_bn, dtype=float),
        angular_rate_body_rad_s=np.array(rs.angular_rate_body_rad_s, dtype=float),
        mass_kg=mass_kg,
        t_s=t_next,
    )
    if agent.belief is not None and agent.belief.state.size >= 13:
        agent.belief = StateBelief(
            state=np.hstack((agent.truth.position_eci_km, agent.truth.velocity_eci_km_s, agent.truth.attitude_quat_bn, agent.truth.angular_rate_body_rad_s)),
            covariance=np.eye(13) * 1e-4,
            last_update_t_s=t_next,
        )
    else:
        agent.belief = StateBelief(
            state=np.hstack((agent.truth.position_eci_km, agent.truth.velocity_eci_km_s)),
            covariance=np.eye(6) * 1e-4,
            last_update_t_s=t_next,
        )
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
    truth = world_truth.get(agent.object_id)
    if truth is None:
        return {}
    own_knowledge = agent.knowledge_base.snapshot() if agent.knowledge_base is not None else {}
    out: dict[str, Any] = {}
    for module in agent.mission_modules:
        if not hasattr(module, "update"):
            continue
        ret = _call_with_compat_kwargs(
            module.update,
            primary_kwargs={
                "object_id": agent.object_id,
                "truth": truth,
                "belief": agent.belief,
                "own_knowledge": own_knowledge,
                "world_truth": world_truth,
                "env": env,
                "t_s": t_s,
                "dt_s": dt_s,
                "orbit_controller": orbit_controller,
                "attitude_controller": attitude_controller,
                "orb_belief": orb_belief,
                "att_belief": att_belief,
                "rocket_state": agent.rocket_state,
                "rocket_vehicle_cfg": (agent.rocket_sim.vehicle_cfg if agent.rocket_sim is not None else None),
            },
            fallback_kwargs={"truth": truth, "t_s": t_s},
        )
        if isinstance(ret, dict):
            out.update(ret)
    return out


def _run_mission_strategy(
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
    strategy = agent.mission_strategy
    if strategy is None:
        return {}
    truth = world_truth.get(agent.object_id)
    if truth is None:
        return {}
    own_knowledge = agent.knowledge_base.snapshot() if agent.knowledge_base is not None else {}
    for method_name in ("update", "plan", "decide"):
        if not hasattr(strategy, method_name):
            continue
        method = getattr(strategy, method_name)
        ret = _call_with_compat_kwargs(
            method,
            primary_kwargs={
                "object_id": agent.object_id,
                "truth": truth,
                "belief": agent.belief,
                "own_knowledge": own_knowledge,
                "world_truth": world_truth,
                "env": env,
                "t_s": t_s,
                "dt_s": dt_s,
                "orbit_controller": orbit_controller,
                "attitude_controller": attitude_controller,
                "orb_belief": orb_belief,
                "att_belief": att_belief,
                "rocket_state": agent.rocket_state,
                "rocket_vehicle_cfg": (agent.rocket_sim.vehicle_cfg if agent.rocket_sim is not None else None),
                "dry_mass_kg": agent.dry_mass_kg,
                "fuel_capacity_kg": agent.fuel_capacity_kg,
            },
            fallback_kwargs={"truth": truth, "t_s": t_s},
        )
        return ret if isinstance(ret, dict) else {}
    return {}


def _run_mission_execution(
    *,
    agent: AgentRuntime,
    intent: dict[str, Any],
    world_truth: dict[str, StateTruth],
    t_s: float,
    dt_s: float,
    env: dict[str, Any],
    orbit_controller: Any | None = None,
    attitude_controller: Any | None = None,
    orb_belief: StateBelief | None = None,
    att_belief: StateBelief | None = None,
) -> dict[str, Any]:
    execution = intent.get("_mission_execution_override", agent.mission_execution)
    if execution is None:
        return {}
    truth = world_truth.get(agent.object_id)
    if truth is None:
        return {}
    own_knowledge = agent.knowledge_base.snapshot() if agent.knowledge_base is not None else {}
    for method_name in ("update", "execute", "act"):
        if not hasattr(execution, method_name):
            continue
        method = getattr(execution, method_name)
        ret = _call_with_compat_kwargs(
            method,
            primary_kwargs={
                "intent": dict(intent or {}),
                "object_id": agent.object_id,
                "truth": truth,
                "belief": agent.belief,
                "own_knowledge": own_knowledge,
                "world_truth": world_truth,
                "env": env,
                "t_s": t_s,
                "dt_s": dt_s,
                "orbit_controller": orbit_controller,
                "attitude_controller": attitude_controller,
                "orb_belief": orb_belief,
                "att_belief": att_belief,
                "rocket_state": agent.rocket_state,
                "rocket_vehicle_cfg": (agent.rocket_sim.vehicle_cfg if agent.rocket_sim is not None else None),
            },
            fallback_kwargs={"intent": dict(intent or {}), "truth": truth, "t_s": t_s},
        )
        return ret if isinstance(ret, dict) else {}
    return {}


def _rocket_altitude_km(r_eci_km: np.ndarray, t_s: float, sim_cfg: RocketSimConfig) -> float:
    if not bool(getattr(sim_cfg, "use_wgs84_geodesy", False)):
        return float(np.linalg.norm(r_eci_km) - EARTH_RADIUS_KM)
    r_ecef = eci_to_ecef(
        np.array(r_eci_km, dtype=float),
        float(t_s),
        jd_utc_start=(dict(getattr(sim_cfg, "atmosphere_env", {}) or {}).get("jd_utc_start")),
    )
    _, _, alt_km = ecef_to_geodetic_deg_km(r_ecef)
    return float(alt_km)


def _orbital_elements_basic(r_km: np.ndarray, v_km_s: np.ndarray, mu_km3_s2: float = EARTH_MU_KM3_S2) -> tuple[float, float]:
    r = float(np.linalg.norm(r_km))
    v2 = float(np.dot(v_km_s, v_km_s))
    if r <= 0.0:
        return np.inf, np.inf
    eps = 0.5 * v2 - mu_km3_s2 / r
    a = np.inf if abs(eps) < 1e-14 else float(-mu_km3_s2 / (2.0 * eps))
    h = np.cross(r_km, v_km_s)
    e_vec = np.cross(v_km_s, h) / mu_km3_s2 - r_km / r
    return a, float(np.linalg.norm(e_vec))
