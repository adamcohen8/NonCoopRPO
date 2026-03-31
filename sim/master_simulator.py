from __future__ import annotations

from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue as queue_mod
import subprocess
import time
from typing import Any, Callable

import numpy as np

from presets.rockets import BASIC_1ST_STAGE, BASIC_SSTO_ROCKET, BASIC_TWO_STAGE_STACK, RocketStackPreset
from presets.thrusters import BASIC_CHEMICAL_BOTTOM_Z
from sim.config import SimulationScenarioConfig, load_simulation_yaml, scenario_config_from_dict, validate_scenario_plugins
from sim.control.attitude.zero_torque import ZeroTorqueController
from sim.control.orbit.zero_controller import ZeroController
from sim.core.models import Command, Measurement, StateBelief, StateTruth
from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel
from sim.dynamics.attitude.rigid_body import get_attitude_guardrail_stats, reset_attitude_guardrail_stats
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.frames import eci_to_ecef
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
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
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.knowledge.object_tracking import (
    KnowledgeConditionConfig,
    KnowledgeEKFConfig,
    KnowledgeNoiseConfig,
    ObjectKnowledgeBase,
    TrackedObjectConfig,
)
from sim.master_outputs import animate_outputs as _animate_outputs_impl
from sim.master_outputs import plot_outputs as _plot_outputs_impl
from sim.rocket import OpenLoopPitchProgramGuidance, RocketAscentSimulator, RocketSimConfig, RocketState, RocketVehicleConfig, TVCSteeringGuidance
from sim.rocket.aero import RocketAeroConfig
from sim.rocket.guidance import MaxQThrottleLimiterGuidance, OrbitInsertionCutoffGuidance
from sim.sensors.noisy_own_state import NoisyOwnStateSensor
from sim.utils.io import write_json
from sim.utils.ground_track import ground_track_from_eci_history
from sim.utils.geodesy import ecef_to_geodetic_deg_km
from sim.utils.frames import eci_relative_to_ric_rect, ric_curv_to_rect, ric_rect_state_to_eci, ric_rect_to_curv
from sim.utils.figure_size import cap_figsize
from sim.utils.quaternion import quaternion_to_dcm_bn

logger = logging.getLogger(__name__)

_PARALLEL_WORKER_THREAD_ENV_VARS = (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _set_parallel_worker_thread_limits(default_threads: str = "1") -> dict[str, str | None]:
    """Limit native math library threads for spawned MC workers unless the user already set them."""
    previous: dict[str, str | None] = {}
    for name in _PARALLEL_WORKER_THREAD_ENV_VARS:
        previous[name] = os.environ.get(name)
        if previous[name] is None:
            os.environ[name] = str(default_threads)
    return previous


def _restore_env_vars(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _load_plotting_functions() -> dict[str, Any]:
    from sim.utils.plotting import plot_attitude_tumble, plot_orbit_eci
    from sim.utils.plotting_capabilities import (
        animate_ground_track,
        animate_multi_ground_track,
        animate_multi_rectangular_prism_ric_curv,
        animate_side_by_side_rectangular_prism_ric_attitude,
        plot_body_rates,
        plot_control_commands,
        plot_multi_control_commands,
        plot_multi_ric_2d_projections,
        plot_multi_trajectory_frame,
        plot_quaternion_components,
        plot_ric_2d_projections,
        plot_trajectory_frame,
    )

    return {
        "plot_orbit_eci": plot_orbit_eci,
        "plot_attitude_tumble": plot_attitude_tumble,
        "plot_body_rates": plot_body_rates,
        "plot_control_commands": plot_control_commands,
        "plot_multi_control_commands": plot_multi_control_commands,
        "plot_multi_ric_2d_projections": plot_multi_ric_2d_projections,
        "plot_multi_trajectory_frame": plot_multi_trajectory_frame,
        "plot_quaternion_components": plot_quaternion_components,
        "plot_ric_2d_projections": plot_ric_2d_projections,
        "plot_trajectory_frame": plot_trajectory_frame,
        "animate_ground_track": animate_ground_track,
        "animate_multi_ground_track": animate_multi_ground_track,
        "animate_multi_rectangular_prism_ric_curv": animate_multi_rectangular_prism_ric_curv,
        "animate_side_by_side_rectangular_prism_ric_attitude": animate_side_by_side_rectangular_prism_ric_attitude,
    }


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
    x_rect = eci_relative_to_ric_rect(
        x_dep_eci=x_dep_eci,
        x_chief_eci=x_chief_eci,
    )
    state[0:6] = ric_rect_to_curv(x_rect, r0_km=float(np.linalg.norm(r_c)))
    state[6:9] = r_c
    state[9:12] = v_c
    return state


def _quat_error_angle_deg(q_des: np.ndarray, q_cur: np.ndarray) -> float:
    qd = np.array(q_des, dtype=float).reshape(-1)
    qc = np.array(q_cur, dtype=float).reshape(-1)
    if qd.size != 4 or qc.size != 4:
        return float("nan")
    nd = float(np.linalg.norm(qd))
    nc = float(np.linalg.norm(qc))
    if nd <= 0.0 or nc <= 0.0:
        return float("nan")
    qd /= nd
    qc /= nc
    dot = float(np.clip(np.dot(qd, qc), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(abs(dot))))


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


def _module_obj(pointer, *, extra_kwargs: dict[str, Any] | None = None) -> Any | None:
    if pointer is None:
        return None
    if pointer.module is None:
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
            fn = getattr(mod, pointer.function)
            return fn
        return mod
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.warning("Failed to construct plugin pointer %r: %s", pointer, exc)
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


def _closest_approach_from_run_payload(run_output: dict[str, Any]) -> float:
    closest_approach_km = float("nan")
    try:
        tb = dict(run_output.get("truth_by_object", {}) or {})
        tgt = np.array(tb.get("target", []), dtype=float)
        ch = np.array(tb.get("chaser", []), dtype=float)
        if tgt.ndim == 2 and ch.ndim == 2 and tgt.shape[0] > 0 and ch.shape[0] > 0:
            n_rel = int(min(tgt.shape[0], ch.shape[0]))
            dr = ch[:n_rel, :3] - tgt[:n_rel, :3]
            rng_km = np.linalg.norm(dr, axis=1)
            finite = rng_km[np.isfinite(rng_km)]
            if finite.size > 0:
                closest_approach_km = float(np.min(finite))
    except (TypeError, ValueError, KeyError, IndexError):
        closest_approach_km = float("nan")
    return closest_approach_km


def _relative_range_series_from_run_payload(run_output: dict[str, Any]) -> dict[str, np.ndarray] | None:
    try:
        tb = dict(run_output.get("truth_by_object", {}) or {})
        t_s = np.array(run_output.get("time_s", []), dtype=float).reshape(-1)
        tgt = np.array(tb.get("target", []), dtype=float)
        ch = np.array(tb.get("chaser", []), dtype=float)
        if (
            t_s.ndim != 1
            or t_s.size == 0
            or tgt.ndim != 2
            or ch.ndim != 2
            or tgt.shape[0] == 0
            or ch.shape[0] == 0
        ):
            return None
        n_rel = int(min(t_s.size, tgt.shape[0], ch.shape[0]))
        dr = ch[:n_rel, :3] - tgt[:n_rel, :3]
        return {
            "time_s": np.array(t_s[:n_rel], dtype=float),
            "range_km": np.array(np.linalg.norm(dr, axis=1), dtype=float),
        }
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def _mc_initial_relative_ric_curv_samples(
    cfg: SimulationScenarioConfig,
    run_details: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    rel_block = dict((cfg.chaser.initial_state or {}).get("relative_to_target_ric", {}) or {})
    frame = str(rel_block.get("frame", "rect")).strip().lower()
    base_state = np.array(rel_block.get("state", []), dtype=float).reshape(-1)
    if frame != "curv" or base_state.size != 6 or not run_details:
        return {}

    paths = {
        "radial_sep_km": "chaser.initial_state.relative_to_target_ric.state[0]",
        "in_track_sep_km": "chaser.initial_state.relative_to_target_ric.state[1]",
        "cross_track_sep_km": "chaser.initial_state.relative_to_target_ric.state[2]",
        "radial_vel_km_s": "chaser.initial_state.relative_to_target_ric.state[3]",
        "in_track_vel_km_s": "chaser.initial_state.relative_to_target_ric.state[4]",
        "cross_track_vel_km_s": "chaser.initial_state.relative_to_target_ric.state[5]",
    }
    index_by_name = {
        "radial_sep_km": 0,
        "in_track_sep_km": 1,
        "cross_track_sep_km": 2,
        "radial_vel_km_s": 3,
        "in_track_vel_km_s": 4,
        "cross_track_vel_km_s": 5,
    }
    out: dict[str, np.ndarray] = {}
    for name, path in paths.items():
        idx = index_by_name[name]
        vals: list[float] = []
        for rd in run_details:
            sampled = dict(rd.get("sampled_parameters", {}) or {})
            vals.append(float(_safe_float(sampled.get(path), default=float(base_state[idx]))))
        out[name] = np.array(vals, dtype=float)
    return out


def _run_mc_iteration_from_dict(task: dict[str, Any]) -> dict[str, Any]:
    iteration = int(task.get("iteration", 0))
    cdict = dict(task.get("config_dict", {}) or {})
    strict_plugins = bool(task.get("strict_plugins", True))
    progress_queue = task.get("progress_queue")
    emit_every = int(task.get("progress_emit_every", 20) or 20)
    emit_every = max(1, emit_every)
    ci = scenario_config_from_dict(cdict)
    if strict_plugins:
        errs = validate_scenario_plugins(ci)
        if errs:
            msg = "Plugin validation failed in Monte Carlo iteration {i}:\n- ".format(i=iteration) + "\n- ".join(errs)
            raise ValueError(msg)

    last_emit = -10**9

    def _on_step(step: int, total: int) -> None:
        nonlocal last_emit
        if progress_queue is None:
            return
        s = max(int(step), 0)
        t = max(int(total), 0)
        should_emit = (s == 0) or (t > 0 and s >= t) or (s - last_emit >= emit_every)
        if not should_emit:
            return
        last_emit = s
        try:
            progress_queue.put(
                {
                    "event": "step",
                    "pid": int(os.getpid()),
                    "iteration": int(iteration),
                    "step": int(s),
                    "total": int(t),
                }
            )
        except Exception:
            pass

    ro = _run_single_config(ci, step_callback=_on_step if progress_queue is not None else None)
    if progress_queue is not None:
        try:
            progress_queue.put(
                {
                    "event": "done",
                    "pid": int(os.getpid()),
                    "iteration": int(iteration),
                }
            )
        except Exception:
            pass
    return {
        "iteration": iteration,
        "summary": ro["summary"],
        "closest_approach_km": _closest_approach_from_run_payload(ro),
        "relative_range_series": _relative_range_series_from_run_payload(ro),
    }


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


def _orbital_elements_basic(r_km: np.ndarray, v_km_s: np.ndarray, mu_km3_s2: float = EARTH_MU_KM3_S2) -> tuple[float, float]:
    r = float(np.linalg.norm(r_km))
    v2 = float(np.dot(v_km_s, v_km_s))
    if r <= 0.0:
        return np.inf, np.inf
    eps = 0.5 * v2 - mu_km3_s2 / r
    a = np.inf if abs(eps) < 1e-14 else float(-mu_km3_s2 / (2.0 * eps))
    h = np.cross(r_km, v_km_s)
    e_vec = np.cross(v_km_s, h) / mu_km3_s2 - r_km / r
    e = float(np.linalg.norm(e_vec))
    return a, e


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


def _resolve_satellite_isp_s(specs: dict[str, Any]) -> float:
    if "isp_s" in specs:
        return float(specs.get("isp_s", 0.0))
    if "thruster_isp_s" in specs:
        return float(specs.get("thruster_isp_s", 0.0))
    thr = str(specs.get("thruster", "")).strip().upper()
    if thr in ("BASIC_CHEMICAL_BOTTOM_Z", "BASIC_CHEMICAL_Z_BOTTOM"):
        return float(BASIC_CHEMICAL_BOTTOM_Z.isp_s)
    return 0.0


def _satellite_initial_delta_v_budget_m_s(agent_cfg: Any) -> float:
    specs = dict(getattr(agent_cfg, "specs", {}) or {})
    dry_mass_kg = _safe_float(specs.get("dry_mass_kg"))
    fuel_mass_kg = _safe_float(specs.get("fuel_mass_kg"))
    if not (np.isfinite(dry_mass_kg) and np.isfinite(fuel_mass_kg)):
        return float("nan")
    if dry_mass_kg <= 0.0 or fuel_mass_kg < 0.0:
        return float("nan")
    m0_kg = dry_mass_kg + fuel_mass_kg
    if m0_kg <= dry_mass_kg:
        return 0.0
    isp_s = _resolve_satellite_isp_s(specs)
    if isp_s <= 0.0:
        return float("nan")
    return float(isp_s * 9.80665 * np.log(m0_kg / dry_mass_kg))


def _resolve_satellite_inertia_kg_m2(specs: dict[str, Any]) -> np.ndarray:
    mp = dict(specs.get("mass_properties", {}) or {})
    if "inertia_kg_m2" in mp:
        I = np.array(mp.get("inertia_kg_m2"), dtype=float)
        if I.shape == (3, 3) and np.all(np.isfinite(I)):
            return I
    return np.diag([120.0, 100.0, 80.0])


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
    x_chaser_eci = ric_rect_state_to_eci(x_rel_rect, r_t, v_t)
    chaser.truth.position_eci_km = x_chaser_eci[:3]
    chaser.truth.velocity_eci_km_s = x_chaser_eci[3:]


def _build_orbit_propagator(cfg: SimulationScenarioConfig) -> OrbitPropagator:
    o = dict(cfg.simulator.dynamics.get("orbit", {}) or {})
    sh = dict(o.get("spherical_harmonics", {}) or {})
    plugins = []
    if bool(o.get("j2", False)):
        plugins.append(j2_plugin)
    if bool(o.get("j3", False)):
        plugins.append(j3_plugin)
    if bool(o.get("j4", False)):
        plugins.append(j4_plugin)
    if bool(sh.get("enabled", False)):
        plugins.append(spherical_harmonics_plugin)
    if bool(o.get("drag", False)):
        plugins.append(drag_plugin)
    if bool(o.get("srp", False)):
        plugins.append(srp_plugin)
    if bool(o.get("third_body_sun", False)):
        plugins.append(third_body_sun_plugin)
    if bool(o.get("third_body_moon", False)):
        plugins.append(third_body_moon_plugin)
    return OrbitPropagator(
        integrator=str(o.get("integrator", "rk4")),
        plugins=plugins,
        adaptive_atol=float(o.get("adaptive_atol", 1e-9)),
        adaptive_rtol=float(o.get("adaptive_rtol", 1e-7)),
    )


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
    specs = dict(agent_cfg.specs or {})
    inertia_kg_m2 = _resolve_satellite_inertia_kg_m2(specs)
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
    attitude_enabled = bool(att_cfg.get("enabled", True))
    dist_cfg = dict(att_cfg.get("disturbance_torques", {}) or {})
    orbit_ctrl_period_s = float(max(float(orbit_cfg.get("orbit_substep_s", cfg.simulator.dt_s) or cfg.simulator.dt_s), 1e-9))
    att_ctrl_period_s = float(max(float(att_cfg.get("attitude_substep_s", cfg.simulator.dt_s) or cfg.simulator.dt_s), 1e-9))
    orbit_ctrl = _RateLimitedController(base=orbit_ctrl_base, period_s=orbit_ctrl_period_s)
    att_ctrl = _RateLimitedController(base=att_ctrl_base, period_s=att_ctrl_period_s) if attitude_enabled else None
    dmodel = DisturbanceTorqueModel(
        mu_km3_s2=EARTH_MU_KM3_S2,
        inertia_kg_m2=inertia_kg_m2,
        config=DisturbanceTorqueConfig(
            use_gravity_gradient=bool(dist_cfg.get("gravity_gradient", False)),
            use_magnetic=bool(dist_cfg.get("magnetic", False)),
            use_drag=bool(dist_cfg.get("drag", False)),
            use_srp=bool(dist_cfg.get("srp", False)),
        ),
    )
    dyn = OrbitalAttitudeDynamics(
        mu_km3_s2=EARTH_MU_KM3_S2,
        inertia_kg_m2=inertia_kg_m2,
        disturbance_model=dmodel if attitude_enabled else None,
        orbit_substep_s=float(orbit_cfg["orbit_substep_s"]) if orbit_cfg.get("orbit_substep_s") is not None else None,
        attitude_substep_s=float(att_cfg["attitude_substep_s"]) if att_cfg.get("attitude_substep_s") is not None else None,
        propagate_attitude=attitude_enabled,
        orbit_propagator=_build_orbit_propagator(cfg),
    )
    bridge = _module_obj(agent_cfg.bridge) if (agent_cfg.bridge is not None and agent_cfg.bridge.enabled) else None
    mission_strategy = _module_obj(getattr(agent_cfg, "mission_strategy", None))
    mission_execution = _module_obj(getattr(agent_cfg, "mission_execution", None))
    missions = [_module_obj(p) for p in list(agent_cfg.mission_objectives or [])]
    missions = [m for m in missions if m is not None]
    sat_isp_s = _resolve_satellite_isp_s(specs)
    sat_dry_mass_kg: float | None = None
    sat_fuel_capacity_kg: float | None = None
    if "dry_mass_kg" in specs:
        try:
            sat_dry_mass_kg = float(specs.get("dry_mass_kg"))
        except (TypeError, ValueError):
            sat_dry_mass_kg = None
        if sat_dry_mass_kg is not None and (not np.isfinite(sat_dry_mass_kg) or sat_dry_mass_kg < 0.0):
            sat_dry_mass_kg = None
    if "fuel_mass_kg" in specs:
        try:
            sat_fuel_capacity_kg = float(specs.get("fuel_mass_kg"))
        except (TypeError, ValueError):
            sat_fuel_capacity_kg = None
        if sat_fuel_capacity_kg is not None and (not np.isfinite(sat_fuel_capacity_kg) or sat_fuel_capacity_kg < 0.0):
            sat_fuel_capacity_kg = None
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
        mission_strategy=mission_strategy,
        mission_execution=mission_execution,
        rocket_sim=None,
        rocket_state=None,
        rocket_guidance=None,
        deploy_source=str((agent_cfg.initial_state or {}).get("source", "")) or None,
        deploy_time_s=float((agent_cfg.initial_state or {}).get("deploy_time_s", 0.0)),
        deploy_dv_body_m_s=np.array((agent_cfg.initial_state or {}).get("deploy_dv_body_m_s", [0.0, 0.0, 0.0]), dtype=float),
        mission_modules=missions,
        waiting_for_launch=False,
        orbital_isp_s=(None if sat_isp_s <= 0.0 else float(sat_isp_s)),
        dry_mass_kg=sat_dry_mass_kg,
        fuel_capacity_kg=sat_fuel_capacity_kg,
    )


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
        attitude_substep_s=float(
            rocket_dyn.get(
                "attitude_substep_s",
                att_dyn.get("attitude_substep_s", 0.02),
            )
            or 0.02
        ),
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
        guidance = TVCSteeringGuidance(
            base_guidance=guidance,
            pass_through_attitude=bool(rocket_dyn.get("tvc_pass_through_attitude", True)),
        )
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
    rsim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=guidance)
    rs = rsim.initial_state()
    rt = _rocket_state_to_truth(rs)
    belief = StateBelief(state=np.hstack((rt.position_eci_km, rt.velocity_eci_km_s)), covariance=np.eye(6) * 1e-4, last_update_t_s=0.0)
    bridge = _module_obj(rc.bridge) if (rc.bridge is not None and rc.bridge.enabled) else None
    mission_strategy = _module_obj(getattr(rc, "mission_strategy", None))
    mission_execution = _module_obj(getattr(rc, "mission_execution", None))
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
        mission_strategy=mission_strategy,
        mission_execution=mission_execution,
        rocket_sim=rsim,
        rocket_state=rs,
        rocket_guidance=guidance,
        deploy_source=None,
        deploy_time_s=None,
        deploy_dv_body_m_s=None,
        mission_modules=missions,
        waiting_for_launch=False,
        orbital_isp_s=None,
        dry_mass_kg=None,
        fuel_capacity_kg=None,
    )


def _build_rocket_guidance(agent_cfg: Any) -> RocketGuidanceLaw:
    base_pointer = getattr(agent_cfg, "base_guidance", None) or getattr(agent_cfg, "guidance", None)
    guidance = _module_obj(base_pointer) or OpenLoopPitchProgramGuidance()
    for modifier_pointer in list(getattr(agent_cfg, "guidance_modifiers", []) or []):
        modifier_obj = _module_obj(modifier_pointer, extra_kwargs={"base_guidance": guidance})
        if modifier_obj is None:
            continue
        guidance = modifier_obj
    return guidance


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
                    solid_angle_sr=cond.get("solid_angle_sr"),
                    require_line_of_sight=bool(cond.get("require_line_of_sight", False)),
                    dropout_prob=float(cond.get("dropout_prob", 0.0)),
                    sensor_position_body_m=np.array(cond.get("sensor_position_body_m", [0.0, 0.0, 0.0]), dtype=float),
                    sensor_boresight_body=(
                        np.array(cond.get("sensor_boresight_body"), dtype=float)
                        if cond.get("sensor_boresight_body") is not None
                        else None
                    ),
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
    own_knowledge = agent.knowledge_base.snapshot() if agent.knowledge_base is not None else {}
    truth = world_truth.get(agent.object_id)
    if truth is None:
        return {}
    for method_name in ("update", "plan", "decide"):
        if not hasattr(strategy, method_name):
            continue
        method = getattr(strategy, method_name)
        try:
            ret = method(
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
                dry_mass_kg=agent.dry_mass_kg,
                fuel_capacity_kg=agent.fuel_capacity_kg,
            )
        except TypeError:
            ret = method(truth=truth, t_s=t_s)
        if isinstance(ret, dict):
            return ret
        return {}
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
    own_knowledge = agent.knowledge_base.snapshot() if agent.knowledge_base is not None else {}
    truth = world_truth.get(agent.object_id)
    if truth is None:
        return {}
    for method_name in ("update", "execute", "act"):
        if not hasattr(execution, method_name):
            continue
        method = getattr(execution, method_name)
        try:
            ret = method(
                intent=dict(intent or {}),
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
            ret = method(intent=dict(intent or {}), truth=truth, t_s=t_s)
        if isinstance(ret, dict):
            return ret
        return {}
    return {}


def _plot_outputs(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    thrust_hist: dict[str, np.ndarray],
    desired_attitude_hist: dict[str, np.ndarray] | None,
    knowledge_hist: dict[str, dict[str, np.ndarray]],
    rocket_metrics: dict[str, np.ndarray] | None,
    outdir: Path,
) -> dict[str, str]:
    return _plot_outputs_impl(
        cfg=cfg,
        t_s=t_s,
        truth_hist=truth_hist,
        thrust_hist=thrust_hist,
        desired_attitude_hist=desired_attitude_hist,
        knowledge_hist=knowledge_hist,
        rocket_metrics=rocket_metrics,
        outdir=outdir,
        resolve_rocket_stack=_resolve_rocket_stack,
        resolve_satellite_isp_s=_resolve_satellite_isp_s,
    )


def _animate_outputs(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    outdir: Path,
) -> dict[str, str]:
    return _animate_outputs_impl(
        cfg=cfg,
        t_s=t_s,
        truth_hist=truth_hist,
        outdir=outdir,
    )


def _fmt_float(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _quantile_stats(values: list[float] | np.ndarray, quantiles: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        out = {
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
        for q in quantiles:
            out[f"p{int(q)}"] = float("nan")
        return out
    out = {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    for q in quantiles:
        out[f"p{int(q)}"] = float(np.percentile(arr, q))
    return out


def _coerce_numeric_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in value.items():
        fv = _safe_float(v)
        if np.isfinite(fv):
            out[str(k)] = fv
    return out


def _get_git_commit_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return out or None


def _infer_model_profile(root_cfg: dict[str, Any]) -> str:
    metadata = dict(root_cfg.get("metadata", {}) or {})
    simulator = dict(root_cfg.get("simulator", {}) or {})
    dynamics = dict(simulator.get("dynamics", {}) or {})
    environment = dict(simulator.get("environment", {}) or {})
    for src in (metadata, simulator, dynamics, environment):
        for key in ("profile", "profile_name", "fidelity_profile"):
            val = src.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return "custom"


def _assess_mc_run(
    *,
    run_entry: dict[str, Any],
    gates: dict[str, Any],
    success_termination_reasons: set[str],
    require_rocket_insertion: bool,
) -> dict[str, Any]:
    summary = dict(run_entry.get("summary", {}) or {})
    term_reason = summary.get("termination_reason")
    term_reason_txt = str(term_reason) if term_reason is not None else "none"
    terminated_early = bool(summary.get("terminated_early", False))
    closest_approach_km = _safe_float(run_entry.get("closest_approach_km"))
    duration_s = _safe_float(summary.get("duration_s"), default=0.0)
    guardrail_map = dict(summary.get("attitude_guardrail_stats", {}) or {})
    guardrail_events = int(sum(int(v) for v in guardrail_map.values())) if guardrail_map else 0
    thrust_stats = dict(summary.get("thrust_stats", {}) or {})
    total_dv_m_s_by_object = {
        str(oid): _safe_float(dict(ts or {}).get("total_dv_m_s"), default=0.0) for oid, ts in thrust_stats.items()
    }
    total_dv_m_s_total = float(np.sum(np.array(list(total_dv_m_s_by_object.values()), dtype=float))) if total_dv_m_s_by_object else 0.0

    fail_reasons: list[str] = []
    if terminated_early and term_reason_txt not in success_termination_reasons:
        fail_reasons.append(f"terminated_early:{term_reason_txt}")
    if require_rocket_insertion and (not bool(summary.get("rocket_insertion_achieved", False))):
        fail_reasons.append("rocket_insertion_not_achieved")

    min_closest_approach_km = _safe_float(gates.get("min_closest_approach_km"))
    if np.isfinite(min_closest_approach_km) and np.isfinite(closest_approach_km) and closest_approach_km < min_closest_approach_km:
        fail_reasons.append("gate:min_closest_approach_km")

    max_duration_s = _safe_float(gates.get("max_duration_s"))
    if np.isfinite(max_duration_s) and duration_s > max_duration_s:
        fail_reasons.append("gate:max_duration_s")

    max_guardrail_events = _safe_float(gates.get("max_guardrail_events"))
    if np.isfinite(max_guardrail_events) and float(guardrail_events) > max_guardrail_events:
        fail_reasons.append("gate:max_guardrail_events")

    max_total_dv_m_s = _safe_float(gates.get("max_total_dv_m_s"))
    if np.isfinite(max_total_dv_m_s) and total_dv_m_s_total > max_total_dv_m_s:
        fail_reasons.append("gate:max_total_dv_m_s")

    max_dv_by_object = _coerce_numeric_map(gates.get("max_total_dv_m_s_by_object"))
    for oid, dv_limit in max_dv_by_object.items():
        dv = _safe_float(total_dv_m_s_by_object.get(oid), default=0.0)
        if dv > dv_limit:
            fail_reasons.append(f"gate:max_total_dv_m_s_by_object:{oid}")

    return {
        "pass": len(fail_reasons) == 0,
        "fail_reasons": sorted(set(fail_reasons)),
        "duration_s": duration_s,
        "closest_approach_km": closest_approach_km,
        "guardrail_events": guardrail_events,
        "termination_reason": term_reason_txt,
        "terminated_early": terminated_early,
        "rocket_insertion_achieved": bool(summary.get("rocket_insertion_achieved", False)),
        "total_dv_m_s_total": total_dv_m_s_total,
        "total_dv_m_s_by_object": total_dv_m_s_by_object,
    }


def _build_parameter_sensitivity_rankings(run_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not run_details:
        return []
    all_paths: set[str] = set()
    for d in run_details:
        for path in dict(d.get("sampled_parameters", {}) or {}).keys():
            all_paths.add(str(path))
    rankings: list[dict[str, Any]] = []
    pass_arr = np.array([1.0 if bool(d.get("pass", False)) else 0.0 for d in run_details], dtype=float)
    ca_arr = np.array([_safe_float(d.get("closest_approach_km")) for d in run_details], dtype=float)
    dv_arr = np.array([_safe_float(d.get("total_dv_m_s_total"), default=0.0) for d in run_details], dtype=float)

    for path in sorted(all_paths):
        vals: list[float] = []
        ok: list[bool] = []
        for d in run_details:
            sv = dict(d.get("sampled_parameters", {}) or {}).get(path)
            if isinstance(sv, bool):
                vals.append(1.0 if sv else 0.0)
                ok.append(True)
            elif isinstance(sv, (int, float, np.integer, np.floating)):
                vals.append(float(sv))
                ok.append(np.isfinite(float(sv)))
            else:
                vals.append(float("nan"))
                ok.append(False)
        x = np.array(vals, dtype=float)
        finite_x = np.isfinite(x)
        if int(np.sum(finite_x)) < 3:
            continue

        def _abs_corr(y: np.ndarray) -> float:
            finite = finite_x & np.isfinite(y)
            if int(np.sum(finite)) < 3:
                return float("nan")
            x_ok = x[finite]
            y_ok = y[finite]
            if np.allclose(np.std(x_ok), 0.0) or np.allclose(np.std(y_ok), 0.0):
                return float("nan")
            return float(abs(np.corrcoef(x_ok, y_ok)[0, 1]))

        corr_pass = _abs_corr(pass_arr)
        corr_ca = _abs_corr(ca_arr)
        corr_dv = _abs_corr(dv_arr)
        importance = float(np.nanmax(np.array([corr_pass, corr_ca, corr_dv], dtype=float)))
        if not np.isfinite(importance):
            continue
        rankings.append(
            {
                "parameter_path": path,
                "samples": int(np.sum(finite_x)),
                "abs_corr_pass": corr_pass,
                "abs_corr_closest_approach_km": corr_ca,
                "abs_corr_total_dv_m_s": corr_dv,
                "importance_score": importance,
            }
        )
    rankings.sort(key=lambda x: float(x.get("importance_score", 0.0)), reverse=True)
    return rankings


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return raw if isinstance(raw, dict) else None


def _extract_baseline_metrics(payload: dict[str, Any]) -> dict[str, float]:
    commander = dict(payload.get("commander_brief", {}) or {})
    aggregate = dict(payload.get("aggregate_stats", {}) or {})
    p_success = _safe_float(commander.get("p_success"))
    p_fail = _safe_float(commander.get("p_fail"))
    duration_p95 = _safe_float(dict(commander.get("timeline_confidence_bands_s", {}) or {}).get("p95"))
    dv_total_p95 = _safe_float(dict(commander.get("fuel_confidence_bands_total_dv_m_s", {}) or {}).get("p95"))
    min_closest = _safe_float(aggregate.get("closest_approach_km_min"))
    return {
        "p_success": p_success,
        "p_fail": p_fail,
        "duration_s_p95": duration_p95,
        "total_dv_m_s_p95": dv_total_p95,
        "closest_approach_km_min": min_closest,
    }


def _aggregate_knowledge_consistency_from_runs(run_details: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, str, str], list[float]] = {}
    for detail in run_details:
        summary = dict(detail.get("summary", {}) or {})
        by_observer = dict(summary.get("knowledge_consistency_by_observer", {}) or {})
        for observer_id, by_target in by_observer.items():
            for target_id, metrics in dict(by_target or {}).items():
                for metric_name, value in dict(metrics or {}).items():
                    try:
                        v = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(v):
                        buckets.setdefault((str(observer_id), str(target_id), str(metric_name)), []).append(v)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for (observer_id, target_id, metric_name), values in sorted(buckets.items()):
        obs_map = out.setdefault(observer_id, {})
        tgt_map = obs_map.setdefault(target_id, {})
        arr = np.array(values, dtype=float)
        tgt_map[metric_name] = float(np.mean(arr)) if arr.size else float("nan")
    return out


def _aggregate_knowledge_detection_from_runs(run_details: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, str, str], list[float]] = {}
    status_counts: dict[tuple[str, str, str], int] = {}
    for detail in run_details:
        summary = dict(detail.get("summary", {}) or {})
        by_observer = dict(summary.get("knowledge_detection_by_observer", {}) or {})
        for observer_id, by_target in by_observer.items():
            for target_id, metrics in dict(by_target or {}).items():
                for metric_name, value in dict(metrics or {}).items():
                    if metric_name == "status_counts" and isinstance(value, dict):
                        for status, count in value.items():
                            key = (str(observer_id), str(target_id), str(status))
                            status_counts[key] = int(status_counts.get(key, 0)) + int(count)
                        continue
                    try:
                        v = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(v):
                        buckets.setdefault((str(observer_id), str(target_id), str(metric_name)), []).append(v)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (observer_id, target_id, metric_name), values in sorted(buckets.items()):
        obs_map = out.setdefault(observer_id, {})
        tgt_map = obs_map.setdefault(target_id, {})
        arr = np.array(values, dtype=float)
        tgt_map[metric_name] = float(np.mean(arr)) if arr.size else float("nan")
    for (observer_id, target_id, status), count in sorted(status_counts.items()):
        obs_map = out.setdefault(observer_id, {})
        tgt_map = obs_map.setdefault(target_id, {})
        tgt_map.setdefault("status_counts", {})[status] = int(count)
    return out


def _build_baseline_comparison(current_payload: dict[str, Any], baseline_payload: dict[str, Any]) -> dict[str, Any]:
    cur = _extract_baseline_metrics(current_payload)
    base = _extract_baseline_metrics(baseline_payload)
    deltas: dict[str, float] = {}
    for k in sorted(set(cur.keys()) | set(base.keys())):
        cv = _safe_float(cur.get(k))
        bv = _safe_float(base.get(k))
        if np.isfinite(cv) and np.isfinite(bv):
            deltas[k] = float(cv - bv)
    return {
        "baseline_metrics": base,
        "current_metrics": cur,
        "delta_current_minus_baseline": deltas,
    }


def _write_commander_brief_markdown(path: Path, brief: dict[str, Any]) -> None:
    top_fail = list(brief.get("top_failure_modes", []) or [])
    lines = [
        "# Monte Carlo Commander Brief",
        "",
        f"- Scenario: {brief.get('scenario_name', 'unknown')}",
        f"- Runs: {int(brief.get('runs', 0))}",
        f"- P(success): {100.0 * _safe_float(brief.get('p_success'), default=0.0):.1f}%",
        f"- P(fail): {100.0 * _safe_float(brief.get('p_fail'), default=0.0):.1f}%",
        f"- P(keepout violation): {100.0 * _safe_float(brief.get('p_keepout_violation'), default=0.0):.1f}%",
        f"- Worst-case closest approach (km): {_fmt_float(_safe_float(brief.get('worst_case_closest_approach_km'), default=0.0), 3)}",
        "",
        "## Confidence Bands",
    ]
    timeline = dict(brief.get("timeline_confidence_bands_s", {}) or {})
    fuel = dict(brief.get("fuel_confidence_bands_total_dv_m_s", {}) or {})
    lines.extend(
        [
            f"- Timeline (s): P50={_fmt_float(_safe_float(timeline.get('p50'), default=0.0), 1)}, "
            f"P90={_fmt_float(_safe_float(timeline.get('p90'), default=0.0), 1)}, "
            f"P99={_fmt_float(_safe_float(timeline.get('p99'), default=0.0), 1)}",
            f"- Total dV (m/s): P50={_fmt_float(_safe_float(fuel.get('p50'), default=0.0), 2)}, "
            f"P90={_fmt_float(_safe_float(fuel.get('p90'), default=0.0), 2)}, "
            f"P99={_fmt_float(_safe_float(fuel.get('p99'), default=0.0), 2)}",
            "",
            "## Risk Metrics",
        ]
    )
    lines.extend(
        [
            f"- P(catastrophic outcome): {100.0 * _safe_float(brief.get('p_catastrophic_outcome'), default=0.0):.1f}%",
            f"- P(exceed dV budget): {100.0 * _safe_float(brief.get('p_exceed_dv_budget'), default=0.0):.1f}%",
            f"- P(exceed time budget): {100.0 * _safe_float(brief.get('p_exceed_time_budget'), default=0.0):.1f}%",
            "",
            "## Top Failure Modes",
        ]
    )
    if top_fail:
        for row in top_fail:
            reason = str(row.get("reason", "unknown"))
            count = int(row.get("count", 0))
            frac = 100.0 * _safe_float(row.get("rate"), default=0.0)
            lines.append(f"- {reason}: {count} runs ({frac:.1f}%)")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _format_single_run_summary(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("MASTER SIMULATION SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Scenario   : {summary.get('scenario_name', 'unknown')}")
    lines.append(f"Objects    : {', '.join(summary.get('objects', []))}")
    lines.append(f"Samples    : {summary.get('samples', 0)}")
    lines.append(
        f"Timing     : dt={_fmt_float(float(summary.get('dt_s', 0.0)), 3)} s, "
        f"duration={_fmt_float(float(summary.get('duration_s', 0.0)), 1)} s"
    )
    lines.append("-" * 72)
    if bool(summary.get("terminated_early", False)):
        lines.append(
            "Termination: EARLY "
            f"(reason={summary.get('termination_reason')}, "
            f"t={summary.get('termination_time_s')}, "
            f"object={summary.get('termination_object_id')})"
        )
    else:
        lines.append("Termination: nominal (full duration reached)")
    if "rocket_insertion_achieved" in summary:
        ins_ok = bool(summary.get("rocket_insertion_achieved", False))
        ins_t = summary.get("rocket_insertion_time_s")
        if ins_ok:
            lines.append(f"Insertion  : achieved at t={ins_t}")
        else:
            lines.append("Insertion  : not achieved")

    thrust_stats = dict(summary.get("thrust_stats", {}) or {})
    if thrust_stats:
        lines.append("-" * 72)
        lines.append("Thrust Stats")
        lines.append(f"{'Object':<14}{'Burn Samples':>14}{'Max Accel (km/s^2)':>24}{'Total dV (m/s)':>18}")
        for oid in sorted(thrust_stats.keys()):
            s = dict(thrust_stats.get(oid, {}) or {})
            lines.append(
                f"{oid:<14}"
                f"{int(s.get('burn_samples', 0)):>14d}"
                f"{float(s.get('max_accel_km_s2', 0.0)):>24.3e}"
                f"{float(s.get('total_dv_m_s', 0.0)):>18.3f}"
            )

    plot_outputs = dict(summary.get("plot_outputs", {}) or {})
    anim_outputs = dict(summary.get("animation_outputs", {}) or {})
    guardrails = dict(summary.get("attitude_guardrail_stats", {}) or {})
    guardrail_hits = int(sum(int(v) for v in guardrails.values())) if guardrails else 0
    lines.append("-" * 72)
    lines.append(f"Artifacts  : plots={len(plot_outputs)}  animations={len(anim_outputs)}")
    lines.append(f"Guardrails : attitude_events={guardrail_hits}")
    lines.append("=" * 72)
    return "\n".join(lines)


class _SingleRunEngine:
    def __init__(
        self,
        cfg: SimulationScenarioConfig,
        *,
        step_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.active_step_callback = step_callback
        reset_attitude_guardrail_stats()

        self.dt = float(cfg.simulator.dt_s)
        self.n = int(np.floor(float(cfg.simulator.duration_s) / self.dt)) + 1
        self.t_s = np.arange(self.n, dtype=float) * self.dt
        self.outdir = Path(cfg.outputs.output_dir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        seed = int(cfg.metadata.get("seed", 123))
        rng = np.random.default_rng(seed)
        dynamics_cfg = dict(cfg.simulator.dynamics or {})
        orbit_cfg = dict(dynamics_cfg.get("orbit", {}) or {})
        att_cfg = dict(dynamics_cfg.get("attitude", {}) or {})
        self.base_environment = dict(cfg.simulator.environment or {})
        self.attitude_enabled = bool(att_cfg.get("enabled", True))
        orbit_substep_s = float(max(float(orbit_cfg.get("orbit_substep_s", self.dt) or self.dt), 1e-9))
        attitude_substep_s = float(max(float(att_cfg.get("attitude_substep_s", self.dt) or self.dt), 1e-9))
        self.sim_substep_s = float(min(orbit_substep_s, attitude_substep_s)) if self.attitude_enabled else orbit_substep_s
        self.eye6 = np.eye(6) * 1e-4
        self.eye12 = np.eye(12) * 1e-4
        self.zero3 = np.zeros(3, dtype=float)

        self.rocket = _create_rocket_runtime(cfg) if cfg.rocket.enabled else None
        self.chaser = (
            _create_satellite_runtime("chaser", cfg.chaser, cfg, np.random.default_rng(int(rng.integers(0, 2**31 - 1))))
            if cfg.chaser.enabled
            else None
        )
        self.target = (
            _create_satellite_runtime("target", cfg.target, cfg, np.random.default_rng(int(rng.integers(0, 2**31 - 1))))
            if cfg.target.enabled
            else None
        )
        if self.chaser is not None and self.chaser.deploy_source == "rocket_deployment":
            self.chaser.active = False

        self.agents: dict[str, AgentRuntime] = {}
        if self.rocket is not None:
            self.agents["rocket"] = self.rocket
        if self.target is not None:
            self.agents["target"] = self.target
        if self.chaser is not None:
            self.agents["chaser"] = self.chaser

        if self.chaser is not None and self.target is not None and self.chaser.deploy_source != "rocket_deployment":
            _apply_chaser_relative_init_from_target(
                chaser=self.chaser,
                target=self.target,
                initial_state=dict(cfg.chaser.initial_state or {}),
            )

        for aid, agent in self.agents.items():
            cfg_src = cfg.rocket if aid == "rocket" else (cfg.chaser if aid == "chaser" else cfg.target)
            agent.knowledge_base = _build_knowledge_base(
                observer_id=aid,
                agent_cfg=cfg_src,
                dt_s=self.dt,
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )

        self.truth_hist = {aid: np.full((self.n, 14), np.nan) for aid in self.agents.keys()}
        self.belief_hist = {aid: np.full((self.n, 6), np.nan) for aid in self.agents.keys()}
        self.thrust_hist = {aid: np.full((self.n, 3), np.nan) for aid in self.agents.keys()}
        self.torque_hist = {aid: np.full((self.n, 3), np.nan) for aid in self.agents.keys()}
        self.desired_attitude_hist = {aid: np.full((self.n, 4), np.nan) for aid in self.agents.keys()}
        self.throttle_hist = {"rocket": np.full(self.n, np.nan)} if self.rocket is not None else {}
        self.rocket_stage_hist = np.full(self.n, np.nan) if self.rocket is not None else None
        self.rocket_q_dyn_hist = np.full(self.n, np.nan) if self.rocket is not None else None
        self.rocket_mach_hist = np.full(self.n, np.nan) if self.rocket is not None else None
        self.knowledge_hist: dict[str, dict[str, np.ndarray]] = {}
        self.bridge_hist: dict[str, list[dict[str, Any]]] = {aid: [] for aid in self.agents.keys()}
        for aid, agent in self.agents.items():
            if agent.knowledge_base is not None:
                self.knowledge_hist[aid] = {}
                for tid in agent.knowledge_base.target_ids():
                    self.knowledge_hist[aid][tid] = np.full((self.n, 6), np.nan)

        self.terminated_early = False
        self.termination_reason: str | None = None
        self.termination_time_s: float | None = None
        self.termination_object_id: str | None = None
        self.rocket_inserted = False
        self.rocket_insertion_time_s: float | None = None
        self.rocket_insertion_hold_s = 0.0
        self.total_dv_m_s_by_object = {aid: 0.0 for aid in self.agents.keys()}
        self.burn_samples_by_object = {aid: 0 for aid in self.agents.keys()}
        self.max_accel_km_s2_by_object = {aid: 0.0 for aid in self.agents.keys()}
        self.current_index = 0

        for aid, agent in self.agents.items():
            if not agent.active:
                continue
            truth = agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)
            self.truth_hist[aid][0, :] = _state_truth_to_array(truth)
            if agent.belief is not None:
                self.belief_hist[aid][0, :] = agent.belief.state[:6]
            if aid == "rocket" and agent.rocket_state is not None and self.rocket_stage_hist is not None:
                self.rocket_stage_hist[0] = float(agent.rocket_state.active_stage_index)
                if self.rocket_q_dyn_hist is not None:
                    self.rocket_q_dyn_hist[0] = float(getattr(agent.rocket_state, "_last_step_q_dyn_pa", 0.0))
                if self.rocket_mach_hist is not None:
                    self.rocket_mach_hist[0] = float(getattr(agent.rocket_state, "_last_step_mach", 0.0))

        self._emit_step_callback(0)

    @property
    def total_steps(self) -> int:
        return max(self.n - 1, 0)

    @property
    def done(self) -> bool:
        return bool(self.terminated_early or self.current_index >= max(self.n - 1, 0))

    def _emit_step_callback(self, step: int) -> None:
        if self.active_step_callback is None:
            return
        try:
            self.active_step_callback(int(step), self.total_steps)
        except (TypeError, ValueError) as exc:
            logger.warning("Disabling step callback after runtime error: %s", exc)
            self.active_step_callback = None

    def snapshot(self, step_index: int | None = None) -> dict[str, Any]:
        idx = self.current_index if step_index is None else int(step_index)
        if idx < 0 or idx >= self.n:
            raise IndexError(f"step_index {idx} is out of range for {self.n} samples.")
        return {
            "step_index": idx,
            "time_s": float(self.t_s[idx]),
            "truth": {oid: np.array(hist[idx], dtype=float) for oid, hist in self.truth_hist.items()},
            "belief": {oid: np.array(hist[idx], dtype=float) for oid, hist in self.belief_hist.items()},
            "applied_thrust": {oid: np.array(hist[idx], dtype=float) for oid, hist in self.thrust_hist.items()},
            "applied_torque": {oid: np.array(hist[idx], dtype=float) for oid, hist in self.torque_hist.items()},
        }

    def step(self) -> dict[str, Any]:
        if self.done:
            return self.snapshot()

        k = int(self.current_index)
        t = float(self.t_s[k])
        t_next = float(self.t_s[k + 1])

        if self.chaser is not None and self.rocket is not None and (not self.chaser.active):
            if t_next >= float(self.chaser.deploy_time_s or 0.0):
                _deploy_from_rocket(self.chaser, self.rocket, t_next)

        world_truth: dict[str, StateTruth] = {}
        for aid, agent in self.agents.items():
            if not agent.active:
                continue
            world_truth[aid] = agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)

        world_truth_live = dict(world_truth)

        for aid, agent in self.agents.items():
            if not agent.active:
                continue
            tr_now = world_truth_live[aid]
            env_common = {
                **self.base_environment,
                "world_truth": world_truth_live,
                "attitude_disabled": (not self.attitude_enabled),
            }

            if agent.kind == "rocket":
                mission_out = _run_mission_modules(
                    agent=agent,
                    world_truth=world_truth_live,
                    t_s=t_next,
                    dt_s=self.dt,
                    env=env_common,
                )
                strategy_out = _run_mission_strategy(
                    agent=agent,
                    world_truth=world_truth_live,
                    t_s=t_next,
                    dt_s=self.dt,
                    env=env_common,
                )
                mission_out.update(strategy_out)
                execution_out = _run_mission_execution(
                    agent=agent,
                    intent=mission_out,
                    world_truth=world_truth_live,
                    t_s=t_next,
                    dt_s=self.dt,
                    env=env_common,
                )
                mission_out.update(execution_out)
                launch_auth = bool(mission_out.get("launch_authorized", True))
                agent.waiting_for_launch = not launch_auth
                if not launch_auth:
                    agent.rocket_state.t_s = float(t_next)
                    agent.truth = _rocket_state_to_truth(agent.rocket_state)
                    if agent.belief is not None:
                        agent.belief.state[:6] = _truth_state6(agent.truth, agent.belief.state[:6])
                        agent.belief.last_update_t_s = t_next
                    self.throttle_hist["rocket"][k] = 0.0
                    self.thrust_hist[aid][k + 1, :] = self.zero3
                    self.torque_hist[aid][k + 1, :] = self.zero3
                    if self.rocket_stage_hist is not None:
                        self.rocket_stage_hist[k + 1] = float(agent.rocket_state.active_stage_index)
                    if self.rocket_q_dyn_hist is not None:
                        self.rocket_q_dyn_hist[k + 1] = 0.0
                    if self.rocket_mach_hist is not None:
                        self.rocket_mach_hist[k + 1] = 0.0
                else:
                    cmd = agent.rocket_guidance.command(agent.rocket_state, agent.rocket_sim.sim_cfg, agent.rocket_sim.vehicle_cfg)
                    if "guidance_throttle" in mission_out:
                        cmd = type(cmd)(
                            throttle=float(mission_out.get("guidance_throttle", cmd.throttle)),
                            attitude_quat_bn_cmd=cmd.attitude_quat_bn_cmd,
                            torque_body_nm_cmd=cmd.torque_body_nm_cmd,
                        )
                    self.throttle_hist["rocket"][k] = float(np.clip(cmd.throttle, 0.0, 1.0))
                    agent.rocket_state = agent.rocket_sim.step(agent.rocket_state, cmd, dt_s=self.dt)
                    agent.truth = _rocket_state_to_truth(agent.rocket_state)
                    if agent.belief is not None:
                        agent.belief.state[:6] = _truth_state6(agent.truth, agent.belief.state[:6])
                        agent.belief.last_update_t_s = t_next
                    thrust_n = float(getattr(agent.rocket_state, "_last_step_thrust_n", 0.0))
                    axis_eci = quaternion_to_dcm_bn(agent.rocket_state.attitude_quat_bn).T @ np.array(agent.rocket_sim.vehicle_cfg.thrust_axis_body, dtype=float)
                    accel = (thrust_n / max(agent.rocket_state.mass_kg, 1e-9)) * axis_eci / 1e3
                    self.thrust_hist[aid][k + 1, :] = accel
                    self.torque_hist[aid][k + 1, :] = self.zero3
                    accel_mag = float(np.linalg.norm(accel))
                    self.total_dv_m_s_by_object[aid] += accel_mag * self.dt * 1e3
                    self.max_accel_km_s2_by_object[aid] = max(self.max_accel_km_s2_by_object[aid], accel_mag)
                    if accel_mag > 1e-15:
                        self.burn_samples_by_object[aid] += 1
                    if self.rocket_stage_hist is not None:
                        self.rocket_stage_hist[k + 1] = float(agent.rocket_state.active_stage_index)
                    if self.rocket_q_dyn_hist is not None:
                        self.rocket_q_dyn_hist[k + 1] = float(getattr(agent.rocket_state, "_last_step_q_dyn_pa", 0.0))
                    if self.rocket_mach_hist is not None:
                        self.rocket_mach_hist[k + 1] = float(getattr(agent.rocket_state, "_last_step_mach", 0.0))
            else:
                t_inner = float(t)
                tr_inner = tr_now
                accel_time_integral = self.zero3.copy()
                torque_time_integral = self.zero3.copy()
                step_delta_v_m_s = 0.0
                step_max_accel_km_s2 = 0.0
                burned_this_step = False
                world_truth_inner = world_truth_live.copy()
                env_inner_common = {**self.base_environment, "world_truth": world_truth_inner}
                env_sensor = {"world_truth": world_truth_inner}
                env_inner = {
                    **self.base_environment,
                    "world_truth": world_truth_inner,
                    "attitude_disabled": (not self.attitude_enabled),
                }
                orbit_state12_scratch = np.empty(12, dtype=float)
                attitude_state13_scratch = np.empty(13, dtype=float)
                deputy_state6_scratch = np.empty(6, dtype=float)
                chief_state6_scratch = np.empty(6, dtype=float)
                orbit_belief_scratch = StateBelief(state=orbit_state12_scratch, covariance=self.eye12, last_update_t_s=t)
                attitude_belief_scratch = StateBelief(state=attitude_state13_scratch, covariance=self.eye6, last_update_t_s=t)
                while t_inner < t_next - 1e-12:
                    h = float(min(self.sim_substep_s, t_next - t_inner))
                    t_eval = t_inner + h
                    world_truth_inner[aid] = tr_inner
                    meas = agent.sensor.measure(truth=tr_inner, env=env_sensor, t_s=t_eval) if agent.sensor is not None else None
                    if agent.estimator is not None and agent.belief is not None:
                        agent.belief = agent.estimator.update(agent.belief, meas, t_eval)
                    elif agent.belief is None:
                        agent.belief = StateBelief(state=_truth_state6(tr_inner), covariance=self.eye6.copy(), last_update_t_s=t_eval)
                    orb_belief = agent.belief
                    if agent.orbit_controller is not None and orb_belief is not None:
                        chief_truth = world_truth_inner.get("target")
                        if chief_truth is not None and aid != "target" and hasattr(agent.orbit_controller, "ric_curv_state_slice"):
                            orbit_belief_scratch.last_update_t_s = orb_belief.last_update_t_s
                            orbit_belief_scratch.state = _relative_orbit_state12(
                                chief_truth=chief_truth,
                                deputy_truth=tr_inner,
                                out=orbit_state12_scratch,
                                deputy_state6=deputy_state6_scratch,
                                chief_state6=chief_state6_scratch,
                            )
                            orb_belief = orbit_belief_scratch
                    att_belief = agent.belief
                    if self.attitude_enabled and att_belief is not None and att_belief.state.size < 13:
                        attitude_belief_scratch.covariance = att_belief.covariance
                        attitude_belief_scratch.last_update_t_s = att_belief.last_update_t_s
                        attitude_belief_scratch.state = _attitude_state13_from_belief(
                            belief=att_belief,
                            truth=tr_inner,
                            out=attitude_state13_scratch,
                        )
                        att_belief = attitude_belief_scratch
                    if not self.attitude_enabled:
                        att_belief = None
                    mission_out = _run_mission_modules(
                        agent=agent,
                        world_truth=world_truth_inner,
                        t_s=t_eval,
                        dt_s=h,
                        env=env_inner_common,
                        orbit_controller=agent.orbit_controller,
                        attitude_controller=(agent.attitude_controller if self.attitude_enabled else None),
                        orb_belief=orb_belief,
                        att_belief=att_belief,
                    )
                    strategy_out = _run_mission_strategy(
                        agent=agent,
                        world_truth=world_truth_inner,
                        t_s=t_eval,
                        dt_s=h,
                        env=env_inner_common,
                        orbit_controller=agent.orbit_controller,
                        attitude_controller=(agent.attitude_controller if self.attitude_enabled else None),
                        orb_belief=orb_belief,
                        att_belief=att_belief,
                    )
                    mission_out.update(strategy_out)
                    execution_out = _run_mission_execution(
                        agent=agent,
                        intent=mission_out,
                        world_truth=world_truth_inner,
                        t_s=t_eval,
                        dt_s=h,
                        env=env_inner_common,
                        orbit_controller=agent.orbit_controller,
                        attitude_controller=(agent.attitude_controller if self.attitude_enabled else None),
                        orb_belief=orb_belief,
                        att_belief=att_belief,
                    )
                    mission_out.update(execution_out)
                    if self.attitude_enabled and "desired_attitude_quat_bn" in mission_out and agent.attitude_controller is not None:
                        q_des = np.array(mission_out["desired_attitude_quat_bn"], dtype=float).reshape(-1)
                        if q_des.size == 4 and hasattr(agent.attitude_controller, "set_target"):
                            try:
                                agent.attitude_controller.set_target(q_des)
                            except (TypeError, ValueError, AttributeError) as exc:
                                logger.warning("Failed to set desired_attitude_quat_bn on %s controller: %s", aid, exc)
                    if self.attitude_enabled and "desired_attitude_quat_bn" in mission_out:
                        q_des_log = np.array(mission_out["desired_attitude_quat_bn"], dtype=float).reshape(-1)
                        if q_des_log.size == 4 and np.all(np.isfinite(q_des_log)):
                            self.desired_attitude_hist[aid][k + 1, :] = q_des_log
                    if self.attitude_enabled and "desired_attitude_quat_br" in mission_out and agent.attitude_controller is not None:
                        q_des_r = np.array(mission_out["desired_attitude_quat_br"], dtype=float).reshape(-1)
                        if q_des_r.size == 4 and hasattr(agent.attitude_controller, "set_target"):
                            try:
                                agent.attitude_controller.set_target(q_des_r)
                            except (TypeError, ValueError, AttributeError) as exc:
                                logger.warning("Failed to set desired_attitude_quat_br on %s controller: %s", aid, exc)
                    if (
                        self.attitude_enabled
                        and "desired_ric_euler_rad" in mission_out
                        and agent.attitude_controller is not None
                        and hasattr(agent.attitude_controller, "set_desired_ric_state")
                    ):
                        e = np.array(mission_out["desired_ric_euler_rad"], dtype=float).reshape(-1)
                        if e.size == 3:
                            try:
                                agent.attitude_controller.set_desired_ric_state(float(e[0]), float(e[1]), float(e[2]))
                            except (TypeError, ValueError, AttributeError) as exc:
                                logger.warning("Failed to set desired_ric_euler_rad on %s controller: %s", aid, exc)
                    use_integrated_cmd = bool(mission_out.get("mission_use_integrated_command", False))
                    c_orb = (
                        agent.orbit_controller.act(orb_belief, t_eval, 2.0)
                        if (not use_integrated_cmd) and agent.orbit_controller is not None and orb_belief is not None
                        else Command.zero()
                    )
                    c_att = (
                        agent.attitude_controller.act(att_belief, t_eval, 2.0)
                        if self.attitude_enabled and (not use_integrated_cmd) and agent.attitude_controller is not None and att_belief is not None
                        else Command.zero()
                    )
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
                    if not self.attitude_enabled:
                        cmd.torque_body_nm = self.zero3
                    cmd_step = Command(
                        thrust_eci_km_s2=np.array(cmd.thrust_eci_km_s2, dtype=float),
                        torque_body_nm=(self.zero3.copy() if not self.attitude_enabled else np.array(cmd.torque_body_nm, dtype=float)),
                        mode_flags=dict(cmd.mode_flags or {}),
                    )
                    min_mass_kg = 0.0
                    if agent.dry_mass_kg is not None and np.isfinite(float(agent.dry_mass_kg)):
                        min_mass_kg = float(max(float(agent.dry_mass_kg), 0.0))
                    fuel_depleted = bool(tr_inner.mass_kg <= (min_mass_kg + 1e-12))
                    if fuel_depleted:
                        cmd_step.thrust_eci_km_s2 = np.zeros(3, dtype=float)
                        cmd_step.mode_flags["fuel_depleted"] = True
                    cmd_step.mode_flags["min_mass_kg"] = float(min_mass_kg)
                    isp_s = agent.orbital_isp_s
                    if isp_s is not None and float(isp_s) > 0.0 and "delta_mass_kg" not in cmd_step.mode_flags:
                        g0_m_s2 = 9.80665
                        a_mag_m_s2 = float(np.linalg.norm(cmd_step.thrust_eci_km_s2) * 1e3)
                        thrust_n = float(max(tr_inner.mass_kg, 0.0) * a_mag_m_s2)
                        mdot_kg_s = 0.0 if thrust_n <= 0.0 else float(thrust_n / (float(isp_s) * g0_m_s2))
                        delta_mass_kg = float(max(mdot_kg_s, 0.0) * h)
                        available_propellant_kg = float(max(tr_inner.mass_kg - min_mass_kg, 0.0))
                        cmd_step.mode_flags["delta_mass_kg"] = float(min(delta_mass_kg, available_propellant_kg))
                    tr_inner = agent.dynamics.step(state=tr_inner, command=cmd_step, env=env_inner, dt_s=h)
                    applied_thrust = np.array(cmd_step.thrust_eci_km_s2, dtype=float)
                    applied_torque = np.array(cmd_step.torque_body_nm, dtype=float)
                    accel_time_integral += applied_thrust * h
                    torque_time_integral += applied_torque * h
                    accel_mag = float(np.linalg.norm(applied_thrust))
                    step_delta_v_m_s += accel_mag * h * 1e3
                    step_max_accel_km_s2 = max(step_max_accel_km_s2, accel_mag)
                    burned_this_step = burned_this_step or (accel_mag > 1e-15)
                    t_inner = t_eval

                agent.truth = tr_inner
                self.thrust_hist[aid][k + 1, :] = accel_time_integral / self.dt
                self.torque_hist[aid][k + 1, :] = self.zero3 if not self.attitude_enabled else (torque_time_integral / self.dt)
                self.total_dv_m_s_by_object[aid] += step_delta_v_m_s
                self.max_accel_km_s2_by_object[aid] = max(self.max_accel_km_s2_by_object[aid], step_max_accel_km_s2)
                if burned_this_step:
                    self.burn_samples_by_object[aid] += 1

            world_truth_live[aid] = agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)

            if agent.bridge is not None:
                evt = {"t_s": t_next, "object_id": aid}
                if hasattr(agent.bridge, "step"):
                    try:
                        ret = agent.bridge.step(evt)
                        if ret is not None:
                            evt["bridge"] = ret
                    except Exception as ex:
                        evt["bridge_error"] = str(ex)
                self.bridge_hist[aid].append(evt)

        for aid, agent in self.agents.items():
            if not agent.active or agent.knowledge_base is None:
                continue
            observer_truth = world_truth_live.get(aid)
            if observer_truth is None:
                continue
            agent.knowledge_base.update(observer_truth=observer_truth, world_truth=world_truth_live, t_s=t_next)
            snap = agent.knowledge_base.snapshot()
            for tid, hist in self.knowledge_hist.get(aid, {}).items():
                belief = snap.get(tid)
                if belief is not None:
                    hist[k + 1, :] = belief.state[:6]
                elif k > 0:
                    hist[k + 1, :] = hist[k, :]

        for aid, agent in self.agents.items():
            if not agent.active:
                continue
            truth = agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)
            self.truth_hist[aid][k + 1, :] = _state_truth_to_array(truth)
            if agent.belief is not None:
                self.belief_hist[aid][k + 1, :] = agent.belief.state[:6]

        self.current_index = k + 1
        self._emit_step_callback(self.current_index)

        if bool(self.cfg.simulator.termination.get("earth_impact_enabled", True)):
            re = float(self.cfg.simulator.termination.get("earth_radius_km", EARTH_RADIUS_KM))
            for aid, agent in self.agents.items():
                if not agent.active:
                    continue
                if agent.kind == "rocket" and agent.waiting_for_launch:
                    continue
                truth = agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)
                impact = float(np.linalg.norm(truth.position_eci_km)) <= re
                if agent.kind == "rocket" and agent.rocket_sim is not None:
                    impact = bool(_rocket_altitude_km(truth.position_eci_km, truth.t_s, agent.rocket_sim.sim_cfg) <= 0.0)
                if impact:
                    self.terminated_early = True
                    self.termination_reason = "earth_impact"
                    self.termination_time_s = t_next
                    self.termination_object_id = aid
                    return self.snapshot()

        if (
            self.rocket is not None
            and self.rocket.active
            and (not self.rocket.waiting_for_launch)
            and self.rocket.rocket_state is not None
            and self.rocket.rocket_sim is not None
        ):
            rs = self.rocket.rocket_state
            sim_cfg = self.rocket.rocket_sim.sim_cfg
            alt_km = _rocket_altitude_km(rs.position_eci_km, rs.t_s, sim_cfg)
            near_alt = abs(float(alt_km) - float(sim_cfg.target_altitude_km)) <= float(sim_cfg.target_altitude_tolerance_km)
            _, ecc_now = _orbital_elements_basic(np.array(rs.position_eci_km, dtype=float), np.array(rs.velocity_eci_km_s, dtype=float))
            low_e = float(ecc_now) <= float(sim_cfg.target_eccentricity_max)
            stages_done = int(rs.active_stage_index) >= len(self.rocket.rocket_sim.vehicle_cfg.stack.stages)
            if near_alt and low_e and stages_done:
                self.rocket_insertion_hold_s += float(self.dt)
                if (not self.rocket_inserted) and self.rocket_insertion_hold_s >= float(sim_cfg.insertion_hold_time_s):
                    self.rocket_inserted = True
                    self.rocket_insertion_time_s = float(t_next)
            else:
                self.rocket_insertion_hold_s = 0.0

            if self.rocket_inserted and str(self.cfg.simulator.scenario_type).strip().lower() == "rocket_ascent":
                self.terminated_early = True
                self.termination_reason = "rocket_orbit_insertion"
                self.termination_time_s = float(self.rocket_insertion_time_s if self.rocket_insertion_time_s is not None else t_next)
                self.termination_object_id = "rocket"

        return self.snapshot()

    def run(self) -> dict[str, Any]:
        while not self.done:
            self.step()
        return self.build_payload()

    def build_payload(self) -> dict[str, Any]:
        n_used = self.current_index + 1
        t_out = self.t_s[:n_used].copy()
        truth_out = {k: v[:n_used, :].copy() for k, v in self.truth_hist.items()}
        belief_out = {k: v[:n_used, :].copy() for k, v in self.belief_hist.items()}
        thrust_out = {k: v[:n_used, :].copy() for k, v in self.thrust_hist.items()}
        torque_out = {k: v[:n_used, :].copy() for k, v in self.torque_hist.items()}
        desired_attitude_out = {k: v[:n_used, :].copy() for k, v in self.desired_attitude_hist.items()}
        knowledge_out = {
            obs: {tgt: arr[:n_used, :].copy() for tgt, arr in by_tgt.items()}
            for obs, by_tgt in self.knowledge_hist.items()
        }
        rocket_metrics_out: dict[str, np.ndarray] = {}
        if self.rocket is not None:
            if self.rocket_stage_hist is not None:
                rocket_metrics_out["stage_index"] = self.rocket_stage_hist[:n_used].copy()
            if self.rocket_q_dyn_hist is not None:
                rocket_metrics_out["q_dyn_pa"] = self.rocket_q_dyn_hist[:n_used].copy()
            if self.rocket_mach_hist is not None:
                rocket_metrics_out["mach"] = self.rocket_mach_hist[:n_used].copy()
            if "rocket" in self.throttle_hist:
                rocket_metrics_out["throttle_cmd"] = self.throttle_hist["rocket"][:n_used].copy()

        plot_outputs = _plot_outputs(
            cfg=self.cfg,
            t_s=t_out,
            truth_hist=truth_out,
            thrust_hist=thrust_out,
            desired_attitude_hist=desired_attitude_out,
            knowledge_hist=knowledge_out,
            rocket_metrics=rocket_metrics_out if rocket_metrics_out else None,
            outdir=self.outdir,
        )
        if self.cfg.outputs.mode in ("interactive", "both") and bool(self.cfg.outputs.plots.get("enabled", True)):
            try:
                import matplotlib.pyplot as plt

                plt.show()
            except (ImportError, AttributeError) as exc:
                logger.warning("Skipping interactive plot display because Matplotlib is unavailable: %s", exc)
        animation_outputs = _animate_outputs(
            cfg=self.cfg,
            t_s=t_out,
            truth_hist=truth_out,
            outdir=self.outdir,
        )

        thrust_stats: dict[str, dict[str, float | int]] = {}
        for oid in thrust_out.keys():
            thrust_stats[oid] = {
                "burn_samples": int(self.burn_samples_by_object.get(oid, 0)),
                "max_accel_km_s2": float(self.max_accel_km_s2_by_object.get(oid, 0.0)),
                "total_dv_m_s": float(self.total_dv_m_s_by_object.get(oid, 0.0)),
            }

        summary = {
            "scenario_name": self.cfg.scenario_name,
            "objects": sorted(list(self.agents.keys())),
            "samples": int(n_used),
            "dt_s": self.dt,
            "duration_s": float(t_out[-1]) if t_out.size else 0.0,
            "terminated_early": self.terminated_early,
            "termination_reason": self.termination_reason,
            "termination_time_s": self.termination_time_s,
            "termination_object_id": self.termination_object_id,
            "rocket_insertion_achieved": bool(self.rocket_inserted),
            "rocket_insertion_time_s": self.rocket_insertion_time_s,
            "thrust_stats": thrust_stats,
            "attitude_guardrail_stats": get_attitude_guardrail_stats(),
            "knowledge_detection_by_observer": {
                aid: agent.knowledge_base.detection_summary()
                for aid, agent in self.agents.items()
                if agent.knowledge_base is not None
            },
            "knowledge_consistency_by_observer": {
                aid: agent.knowledge_base.consistency_summary()
                for aid, agent in self.agents.items()
                if agent.knowledge_base is not None
            },
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
            "knowledge_detection_by_observer": dict(summary.get("knowledge_detection_by_observer", {}) or {}),
            "knowledge_consistency_by_observer": dict(summary.get("knowledge_consistency_by_observer", {}) or {}),
            "bridge_events_by_object": self.bridge_hist,
            "rocket_throttle_cmd": self.throttle_hist.get("rocket", np.array([])).tolist() if self.throttle_hist else [],
            "rocket_metrics": {k: v.tolist() for k, v in rocket_metrics_out.items()},
        }
        if bool(self.cfg.outputs.stats.get("save_json", True)):
            write_json(str(self.outdir / "master_run_summary.json"), summary)
        if bool(self.cfg.outputs.stats.get("save_full_log", True)):
            write_json(str(self.outdir / "master_run_log.json"), payload)
        if bool(self.cfg.outputs.stats.get("print_summary", True)):
            print(_format_single_run_summary(summary))
        return payload


def _run_single_config(
    cfg: SimulationScenarioConfig,
    step_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    return _SingleRunEngine(cfg, step_callback=step_callback).run()


def _is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_noninteractive_for_automation(cfg: SimulationScenarioConfig) -> SimulationScenarioConfig:
    if not (_is_truthy_env("SIM_AUTOMATION") or _is_truthy_env("CI")):
        return cfg
    root = cfg.to_dict()
    outputs = root.setdefault("outputs", {})
    mode = str(outputs.get("mode", "interactive")).strip().lower()
    if mode == "interactive":
        outputs["mode"] = "save"
    return scenario_config_from_dict(root)


def run_master_simulation(
    config_path: str | Path,
    step_callback: Callable[[int, int], None] | None = None,
    mc_callback: Callable[[int, int], None] | None = None,
    mc_progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    cfg = _coerce_noninteractive_for_automation(load_simulation_yaml(config_path))
    strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))
    if strict_plugins:
        errs = validate_scenario_plugins(cfg)
        if errs:
            msg = "Plugin validation failed:\n- " + "\n- ".join(errs)
            raise ValueError(msg)
    if not cfg.monte_carlo.enabled:
        from sim.api import SimulationConfig, SimulationSession

        session = SimulationSession.from_config(SimulationConfig(cfg, source_path=Path(config_path).resolve()))
        result = session.run(step_callback=step_callback)
        return {
            "config_path": str(Path(config_path).resolve()),
            "scenario_name": cfg.scenario_name,
            "monte_carlo": {"enabled": False},
            "run": result.summary,
        }

    root = cfg.to_dict()
    outdir = Path(cfg.outputs.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    mc_out_cfg = dict(cfg.outputs.monte_carlo or {})
    repo_root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(int(cfg.monte_carlo.base_seed))
    runs = []
    run_details: list[dict[str, Any]] = []
    closest_approach_km_runs: list[float] = []
    duration_runs_s: list[float] = []
    guardrail_event_runs: list[int] = []
    total_dv_runs_m_s: list[float] = []
    relative_range_series_runs: list[dict[str, np.ndarray] | None] = []
    failure_mode_counts: dict[str, int] = {}
    success_termination_reasons = {str(x) for x in (mc_out_cfg.get("success_termination_reasons", ["rocket_orbit_insertion"]) or [])}
    require_rocket_insertion = bool(mc_out_cfg.get("require_rocket_insertion", False))
    gates = dict(mc_out_cfg.get("gates", {}) or {})
    dv_budget_m_s_by_object: dict[str, float] = {}
    if bool(cfg.chaser.enabled):
        dv_chaser = _satellite_initial_delta_v_budget_m_s(cfg.chaser)
        if np.isfinite(dv_chaser):
            dv_budget_m_s_by_object["chaser"] = float(dv_chaser)
    if bool(cfg.target.enabled):
        dv_target = _satellite_initial_delta_v_budget_m_s(cfg.target)
        if np.isfinite(dv_target):
            dv_budget_m_s_by_object["target"] = float(dv_target)
    varies_metadata_seed = any(str(v.parameter_path) == "metadata.seed" for v in cfg.monte_carlo.variations)
    total_iters = int(cfg.monte_carlo.iterations)
    parallel_enabled = bool(cfg.monte_carlo.parallel_enabled)
    max_workers_cfg = int(cfg.monte_carlo.parallel_workers or 0)
    default_workers = max(1, (os.cpu_count() or 1) - 1)
    parallel_workers = max_workers_cfg if max_workers_cfg > 0 else default_workers
    parallel_workers = max(1, min(parallel_workers, total_iters))
    parallel_active = bool(parallel_enabled and total_iters > 1)
    parallel_fallback_reason: str | None = None
    prepared: list[dict[str, Any]] = []
    for i in range(total_iters):
        cdict = deepcopy(root)
        sampled = {}
        for v in cfg.monte_carlo.variations:
            sv = _sample_variation(v, rng)
            _deep_set(cdict, v.parameter_path, sv)
            sampled[v.parameter_path] = sv
        if not varies_metadata_seed:
            md = cdict.setdefault("metadata", {})
            if "seed" not in md:
                md["seed"] = int(cfg.monte_carlo.base_seed) + i
        # Prevent unwanted pop-up windows in MC mode unless explicitly both/save.
        mode = str(cdict.get("outputs", {}).get("mode", "interactive"))
        if mode == "interactive":
            cdict.setdefault("outputs", {})["mode"] = "save"
        cdict.setdefault("outputs", {})["output_dir"] = str(outdir / f"mc_run_{i:04d}")
        prepared.append(
            {
                "iteration": i,
                "sampled_parameters": sampled,
                "config_dict": cdict,
                "seed": int(cdict.get("metadata", {}).get("seed", int(cfg.monte_carlo.base_seed) + i)),
            }
        )

    completed: dict[int, dict[str, Any]] = {}
    if parallel_active:
        manager = None
        progress_queue = None
        thread_env_prev = _set_parallel_worker_thread_limits(default_threads="1")
        try:
            manager = mp.Manager()
            progress_queue = manager.Queue()
            tasks = [
                {
                    "iteration": p["iteration"],
                    "config_dict": p["config_dict"],
                    "strict_plugins": strict_plugins,
                    "progress_queue": progress_queue,
                    "progress_emit_every": int(mc_out_cfg.get("parallel_progress_emit_every_steps", 20) or 20),
                }
                for p in prepared
            ]
            with ProcessPoolExecutor(max_workers=parallel_workers) as ex:
                fut_to_idx = {ex.submit(_run_mc_iteration_from_dict, t): int(t["iteration"]) for t in tasks}
                pending = set(fut_to_idx.keys())
                while pending:
                    done_now, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                    for fut in done_now:
                        idx = fut_to_idx[fut]
                        completed[idx] = fut.result()
                        if mc_callback is not None:
                            try:
                                mc_callback(len(completed), total_iters)
                            except Exception as exc:
                                logger.warning("Disabling Monte Carlo callback after runtime error: %s", exc)
                                mc_callback = None
                    if progress_queue is not None:
                        while True:
                            try:
                                evt = progress_queue.get_nowait()
                            except queue_mod.Empty:
                                break
                            except Exception:
                                break
                            if mc_progress_callback is not None:
                                try:
                                    mc_progress_callback(dict(evt or {}))
                                except Exception as exc:
                                    logger.warning("Disabling Monte Carlo progress callback after runtime error: %s", exc)
                                    mc_progress_callback = None
        except (OSError, PermissionError, NotImplementedError, EOFError) as exc:
            parallel_active = False
            parallel_fallback_reason = f"{type(exc).__name__}: {exc}"
            logger.warning("Parallel Monte Carlo unavailable, falling back to serial execution: %s", exc)
        finally:
            if progress_queue is not None:
                try:
                    while True:
                        evt = progress_queue.get_nowait()
                        if mc_progress_callback is not None:
                            mc_progress_callback(dict(evt or {}))
                except Exception:
                    pass
            if manager is not None:
                try:
                    manager.shutdown()
                except Exception:
                    pass
            _restore_env_vars(thread_env_prev)
    if not parallel_active:
        completed_count = 0
        for p in prepared:
            ci = scenario_config_from_dict(dict(p["config_dict"]))
            if strict_plugins:
                errs = validate_scenario_plugins(ci)
                if errs:
                    msg = "Plugin validation failed in Monte Carlo iteration {i}:\n- ".format(i=int(p["iteration"])) + "\n- ".join(errs)
                    raise ValueError(msg)
            ro = _run_single_config(ci, step_callback=step_callback)
            completed[int(p["iteration"])] = {
                "iteration": int(p["iteration"]),
                "summary": ro["summary"],
                "closest_approach_km": _closest_approach_from_run_payload(ro),
                "relative_range_series": _relative_range_series_from_run_payload(ro),
            }
            completed_count += 1
            if mc_callback is not None:
                try:
                    mc_callback(completed_count, total_iters)
                except Exception as exc:
                    logger.warning("Disabling Monte Carlo callback after runtime error: %s", exc)
                    mc_callback = None

    for p in sorted(prepared, key=lambda x: int(x["iteration"])):
        i = int(p["iteration"])
        cres = dict(completed.get(i, {}) or {})
        ro_summary = dict(cres.get("summary", {}) or {})
        closest_approach_km = _safe_float(cres.get("closest_approach_km"))
        relative_range_series_runs.append(cres.get("relative_range_series"))
        closest_approach_km_runs.append(closest_approach_km)
        assessment = _assess_mc_run(
            run_entry={"summary": ro_summary, "closest_approach_km": closest_approach_km},
            gates=gates,
            success_termination_reasons=success_termination_reasons,
            require_rocket_insertion=require_rocket_insertion,
        )
        duration_runs_s.append(float(assessment["duration_s"]))
        guardrail_event_runs.append(int(assessment["guardrail_events"]))
        total_dv_runs_m_s.append(float(assessment["total_dv_m_s_total"]))
        run_detail = {
            "iteration": i,
            "seed": int(p["seed"]),
            "sampled_parameters": dict(p["sampled_parameters"]),
            "pass": bool(assessment["pass"]),
            "fail_reasons": list(assessment["fail_reasons"]),
            "duration_s": float(assessment["duration_s"]),
            "closest_approach_km": float(assessment["closest_approach_km"]) if np.isfinite(_safe_float(assessment["closest_approach_km"])) else float("nan"),
            "guardrail_events": int(assessment["guardrail_events"]),
            "termination_reason": str(assessment["termination_reason"]),
            "terminated_early": bool(assessment["terminated_early"]),
            "rocket_insertion_achieved": bool(assessment["rocket_insertion_achieved"]),
            "total_dv_m_s_total": float(assessment["total_dv_m_s_total"]),
            "total_dv_m_s_by_object": dict(assessment["total_dv_m_s_by_object"]),
            "delta_v_remaining_m_s_by_object": {},
        }
        dv_rem = dict(run_detail["delta_v_remaining_m_s_by_object"])
        for oid, dv_budget in dv_budget_m_s_by_object.items():
            dv_used = _safe_float(dict(run_detail["total_dv_m_s_by_object"]).get(oid), default=0.0)
            dv_rem[oid] = float(max(float(dv_budget) - max(float(dv_used), 0.0), 0.0))
        run_detail["delta_v_remaining_m_s_by_object"] = dv_rem
        for reason in run_detail["fail_reasons"]:
            failure_mode_counts[str(reason)] = int(failure_mode_counts.get(str(reason), 0) + 1)
        run_details.append(run_detail)
        entry = {
            "iteration": i,
            "sampled_parameters": dict(p["sampled_parameters"]),
            "summary": ro_summary,
            "closest_approach_km": closest_approach_km,
            "assessment": assessment,
        }
        runs.append(entry)
        if bool(cfg.outputs.monte_carlo.get("save_iteration_summaries", False)):
            write_json(str(outdir / f"master_monte_carlo_run_{i:04d}.json"), entry)

    durations_s = np.array([float(dict(r.get("summary", {}) or {}).get("duration_s", 0.0)) for r in runs], dtype=float)
    terminated_early_flags = np.array(
        [1.0 if bool(dict(r.get("summary", {}) or {}).get("terminated_early", False)) else 0.0 for r in runs],
        dtype=float,
    )
    termination_reason_counts: dict[str, int] = {}
    dv_by_object: dict[str, list[float]] = {}
    burn_samples_by_object: dict[str, list[float]] = {}
    for entry in runs:
        s = dict(entry.get("summary", {}) or {})
        term_reason = s.get("termination_reason")
        if term_reason is not None:
            key = str(term_reason)
            termination_reason_counts[key] = int(termination_reason_counts.get(key, 0) + 1)
        thrust_stats = dict(s.get("thrust_stats", {}) or {})
        for oid, ts in thrust_stats.items():
            tsd = dict(ts or {})
            dv_by_object.setdefault(str(oid), []).append(float(tsd.get("total_dv_m_s", 0.0)))
            burn_samples_by_object.setdefault(str(oid), []).append(float(tsd.get("burn_samples", 0.0)))
    dv_remaining_m_s_by_object: dict[str, list[float]] = {}
    for oid in sorted(dv_budget_m_s_by_object.keys()):
        vals: list[float] = []
        for d in run_details:
            rem = _safe_float(dict(d.get("delta_v_remaining_m_s_by_object", {}) or {}).get(oid))
            if np.isfinite(rem):
                vals.append(float(rem))
        if vals:
            dv_remaining_m_s_by_object[oid] = vals

    by_object_stats: dict[str, dict[str, float]] = {}
    all_obj_ids = sorted(set(dv_by_object.keys()) | set(burn_samples_by_object.keys()))
    for oid in all_obj_ids:
        dv_arr = np.array(dv_by_object.get(oid, []), dtype=float)
        b_arr = np.array(burn_samples_by_object.get(oid, []), dtype=float)
        by_object_stats[oid] = {
            "total_dv_m_s_mean": float(np.mean(dv_arr)) if dv_arr.size else 0.0,
            "total_dv_m_s_min": float(np.min(dv_arr)) if dv_arr.size else 0.0,
            "total_dv_m_s_max": float(np.max(dv_arr)) if dv_arr.size else 0.0,
            "total_dv_m_s_p95": float(np.percentile(dv_arr, 95)) if dv_arr.size else 0.0,
            "burn_samples_mean": float(np.mean(b_arr)) if b_arr.size else 0.0,
            "burn_samples_p95": float(np.percentile(b_arr, 95)) if b_arr.size else 0.0,
        }
    ca_arr_full = np.array(closest_approach_km_runs, dtype=float)
    ca_finite = ca_arr_full[np.isfinite(ca_arr_full)]
    duration_arr = np.array(duration_runs_s, dtype=float)
    total_dv_arr = np.array(total_dv_runs_m_s, dtype=float)
    guardrail_arr = np.array(guardrail_event_runs, dtype=float)
    pass_flags = np.array([1.0 if bool(d.get("pass", False)) else 0.0 for d in run_details], dtype=float)
    pass_rate = float(np.mean(pass_flags)) if pass_flags.size else 0.0
    guardrail_violation_flags = np.array([1.0 if int(d.get("guardrail_events", 0)) > 0 else 0.0 for d in run_details], dtype=float)

    aggregate_stats = {
        "duration_s_mean": float(np.mean(durations_s)) if durations_s.size else 0.0,
        "duration_s_min": float(np.min(durations_s)) if durations_s.size else 0.0,
        "duration_s_max": float(np.max(durations_s)) if durations_s.size else 0.0,
        "duration_s_p50": float(np.percentile(durations_s, 50)) if durations_s.size else float("nan"),
        "duration_s_p90": float(np.percentile(durations_s, 90)) if durations_s.size else float("nan"),
        "duration_s_p95": float(np.percentile(durations_s, 95)) if durations_s.size else float("nan"),
        "duration_s_p99": float(np.percentile(durations_s, 99)) if durations_s.size else float("nan"),
        "terminated_early_rate": float(np.mean(terminated_early_flags)) if terminated_early_flags.size else 0.0,
        "closest_approach_km_mean": float(np.mean(ca_finite)) if ca_finite.size else float("nan"),
        "closest_approach_km_min": float(np.min(ca_finite)) if ca_finite.size else float("nan"),
        "closest_approach_km_max": float(np.max(ca_finite)) if ca_finite.size else float("nan"),
        "closest_approach_km_p05": float(np.percentile(ca_finite, 5)) if ca_finite.size else float("nan"),
        "closest_approach_km_p50": float(np.percentile(ca_finite, 50)) if ca_finite.size else float("nan"),
        "closest_approach_km_p95": float(np.percentile(ca_finite, 95)) if ca_finite.size else float("nan"),
        "total_dv_m_s_mean": float(np.mean(total_dv_arr)) if total_dv_arr.size else float("nan"),
        "total_dv_m_s_p50": float(np.percentile(total_dv_arr, 50)) if total_dv_arr.size else float("nan"),
        "total_dv_m_s_p90": float(np.percentile(total_dv_arr, 90)) if total_dv_arr.size else float("nan"),
        "total_dv_m_s_p95": float(np.percentile(total_dv_arr, 95)) if total_dv_arr.size else float("nan"),
        "total_dv_m_s_p99": float(np.percentile(total_dv_arr, 99)) if total_dv_arr.size else float("nan"),
        "guardrail_events_mean": float(np.mean(guardrail_arr)) if guardrail_arr.size else float("nan"),
        "guardrail_events_p95": float(np.percentile(guardrail_arr, 95)) if guardrail_arr.size else float("nan"),
        "pass_rate": pass_rate,
        "fail_rate": 1.0 - pass_rate,
        "guardrail_violation_rate": float(np.mean(guardrail_violation_flags)) if guardrail_violation_flags.size else float("nan"),
        "failure_mode_counts": failure_mode_counts,
        "termination_reason_counts": termination_reason_counts,
        "by_object": by_object_stats,
        "delta_v_budget_m_s_by_object": dict(dv_budget_m_s_by_object),
        "delta_v_remaining_m_s_by_object": {
            oid: _quantile_stats(vals, (50.0, 90.0, 99.0)) for oid, vals in sorted(dv_remaining_m_s_by_object.items())
        },
        "knowledge_detection_by_observer": _aggregate_knowledge_detection_from_runs(run_details),
        "knowledge_consistency_by_observer": _aggregate_knowledge_consistency_from_runs(run_details),
    }

    keepout_threshold = _safe_float(gates.get("min_closest_approach_km"))
    if not np.isfinite(keepout_threshold):
        keepout_threshold = _safe_float(mc_out_cfg.get("keepout_radius_km"))
    p_keepout_violation = float("nan")
    if np.isfinite(keepout_threshold) and ca_finite.size:
        p_keepout_violation = float(np.mean(ca_finite < keepout_threshold))

    catastrophic_failure_reasons = [str(x) for x in (mc_out_cfg.get("catastrophic_failure_reasons", ["terminated_early:earth_impact"]) or [])]
    catastrophic_count = 0
    for rd in run_details:
        reasons = set(str(x) for x in list(rd.get("fail_reasons", []) or []))
        if any(r in reasons for r in catastrophic_failure_reasons):
            catastrophic_count += 1
    p_catastrophic_outcome = float(catastrophic_count / max(len(run_details), 1))

    max_duration_gate = _safe_float(gates.get("max_duration_s"))
    p_exceed_time_budget = float("nan")
    if np.isfinite(max_duration_gate) and duration_arr.size:
        p_exceed_time_budget = float(np.mean(duration_arr > max_duration_gate))

    max_total_dv_gate = _safe_float(gates.get("max_total_dv_m_s"))
    p_exceed_dv_budget = float("nan")
    if np.isfinite(max_total_dv_gate) and total_dv_arr.size:
        p_exceed_dv_budget = float(np.mean(total_dv_arr > max_total_dv_gate))

    if np.isfinite(max_total_dv_gate) and total_dv_arr.size:
        fuel_margin = max_total_dv_gate - total_dv_arr
        fuel_margin_stats = _quantile_stats(fuel_margin, (5.0, 50.0, 95.0))
    else:
        fuel_margin_stats = _quantile_stats([], (5.0, 50.0, 95.0))
    if np.isfinite(max_duration_gate) and duration_arr.size:
        time_margin = max_duration_gate - duration_arr
        time_margin_stats = _quantile_stats(time_margin, (5.0, 50.0, 95.0))
    else:
        time_margin_stats = _quantile_stats([], (5.0, 50.0, 95.0))
    aggregate_stats["p_keepout_violation"] = p_keepout_violation
    aggregate_stats["p_catastrophic_outcome"] = p_catastrophic_outcome
    aggregate_stats["p_exceed_dv_budget"] = p_exceed_dv_budget
    aggregate_stats["p_exceed_time_budget"] = p_exceed_time_budget

    top_failure_modes: list[dict[str, Any]] = []
    if failure_mode_counts:
        sorted_modes = sorted(failure_mode_counts.items(), key=lambda kv: int(kv[1]), reverse=True)
        for reason, cnt in sorted_modes[:3]:
            top_failure_modes.append(
                {
                    "reason": str(reason),
                    "count": int(cnt),
                    "rate": float(int(cnt) / max(len(run_details), 1)),
                }
            )

    cfg_json = json.dumps(root, sort_keys=True, separators=(",", ":"))
    reproducibility = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _get_git_commit_sha(repo_root),
        "config_sha256": hashlib.sha256(cfg_json.encode("utf-8")).hexdigest(),
        "model_profile": _infer_model_profile(root),
        "random_seed_policy": (
            "Per-run metadata.seed set to monte_carlo.base_seed + iteration unless metadata.seed is explicitly varied."
            if not varies_metadata_seed
            else "metadata.seed controlled by monte_carlo variations."
        ),
    }
    sensitivity_rankings = _build_parameter_sensitivity_rankings(run_details)
    top_parameter_drivers = [
        {
            "parameter_path": str(row.get("parameter_path")),
            "importance_score": _safe_float(row.get("importance_score"), default=0.0),
            "abs_corr_pass": _safe_float(row.get("abs_corr_pass")),
            "abs_corr_closest_approach_km": _safe_float(row.get("abs_corr_closest_approach_km")),
            "abs_corr_total_dv_m_s": _safe_float(row.get("abs_corr_total_dv_m_s")),
        }
        for row in sensitivity_rankings[:5]
    ]
    unique_seeds = len(set(int(_safe_float(d.get("seed"), default=-1)) for d in run_details))
    finite_ca_rate = float(np.mean(np.isfinite(ca_arr_full))) if ca_arr_full.size else float("nan")
    analysis_confidence = {
        "runs_executed": int(len(run_details)),
        "run_count_sufficient_for_tail_estimates": bool(len(run_details) >= 100),
        "unique_seed_count": int(unique_seeds),
        "finite_closest_approach_rate": finite_ca_rate,
        "varied_parameter_count": int(len(set(str(v.parameter_path) for v in cfg.monte_carlo.variations))),
        "model_profile": reproducibility.get("model_profile"),
        "git_commit_sha": reproducibility.get("git_commit_sha"),
        "config_sha256": reproducibility.get("config_sha256"),
    }
    commander_brief = {
        "scenario_name": cfg.scenario_name,
        "runs": int(cfg.monte_carlo.iterations),
        "p_success": pass_rate,
        "p_fail": 1.0 - pass_rate,
        "p_keepout_violation": p_keepout_violation,
        "p_catastrophic_outcome": p_catastrophic_outcome,
        "p_exceed_dv_budget": p_exceed_dv_budget,
        "p_exceed_time_budget": p_exceed_time_budget,
        "keepout_threshold_km": keepout_threshold if np.isfinite(keepout_threshold) else None,
        "worst_case_closest_approach_km": float(np.min(ca_finite)) if ca_finite.size else float("nan"),
        "timeline_confidence_bands_s": _quantile_stats(duration_runs_s, (50.0, 90.0, 99.0)),
        "fuel_confidence_bands_total_dv_m_s": _quantile_stats(total_dv_runs_m_s, (50.0, 90.0, 99.0)),
        "fuel_confidence_bands_dv_m_s_by_object": {
            oid: _quantile_stats(dv_by_object.get(oid, []), (50.0, 90.0, 99.0)) for oid in sorted(dv_by_object.keys())
        },
        "delta_v_remaining_confidence_bands_m_s_by_object": {
            oid: _quantile_stats(vals, (50.0, 90.0, 99.0)) for oid, vals in sorted(dv_remaining_m_s_by_object.items())
        },
        "resource_margin": {
            "fuel_margin_m_s_vs_budget": fuel_margin_stats,
            "time_margin_s_vs_budget": time_margin_stats,
        },
        "constraint_violation_summary": {
            "p_guardrail_violation": float(np.mean(guardrail_violation_flags)) if guardrail_violation_flags.size else float("nan"),
            "guardrail_events_per_run": _quantile_stats(guardrail_event_runs, (50.0, 90.0, 99.0)),
            "compute_deadline_overrun_available": False,
            "control_saturation_available": False,
        },
        "top_failure_modes": top_failure_modes,
        "top_parameter_drivers": top_parameter_drivers,
        "analysis_confidence": analysis_confidence,
    }

    analyst_pack = {
        "scenario_name": cfg.scenario_name,
        "reproducibility": reproducibility,
        "gates": gates,
        "run_details": run_details,
        "failure_mode_counts": failure_mode_counts,
        "sensitivity_rankings": sensitivity_rankings,
        "catastrophic_failure_reasons": catastrophic_failure_reasons,
    }

    agg = {
        "config_path": str(Path(config_path).resolve()),
        "scenario_name": cfg.scenario_name,
        "monte_carlo": {
            "enabled": True,
            "iterations": int(cfg.monte_carlo.iterations),
            "base_seed": int(cfg.monte_carlo.base_seed),
            "parallel_enabled": bool(parallel_active),
            "parallel_requested": bool(parallel_enabled and total_iters > 1),
            "parallel_workers": int(parallel_workers if parallel_active else 1),
        },
        "aggregate_stats": aggregate_stats,
        "commander_brief": commander_brief,
        "reproducibility": reproducibility,
        "analyst_pack": analyst_pack,
        "artifacts": {},
        "runs": runs,
    }
    if parallel_fallback_reason is not None:
        agg["monte_carlo"]["parallel_fallback_reason"] = str(parallel_fallback_reason)

    baseline_summary_json = str(mc_out_cfg.get("baseline_summary_json", "")).strip()
    if baseline_summary_json:
        bpath = Path(baseline_summary_json)
        if not bpath.is_absolute():
            bpath = Path(config_path).resolve().parent / bpath
        baseline_payload = _load_json_file(bpath)
        if baseline_payload is not None:
            comparison = _build_baseline_comparison(agg, baseline_payload)
            agg["baseline_comparison"] = comparison
            commander_brief["baseline_comparison"] = comparison
        else:
            agg["baseline_comparison_error"] = f"Unable to load baseline summary: {str(bpath)}"

    save_hist = bool(cfg.outputs.monte_carlo.get("save_histograms", False))
    show_hist = bool(cfg.outputs.monte_carlo.get("display_histograms", False))
    if (save_hist or show_hist) and runs:
        import matplotlib.pyplot as plt

        plot_series: list[tuple[str, np.ndarray]] = [("Duration (s)", durations_s)]
        if ca_finite.size:
            plot_series.append(("Closest Approach (km)", ca_finite))
        for oid in all_obj_ids:
            dv_arr = np.array(dv_by_object.get(oid, []), dtype=float)
            if dv_arr.size:
                plot_series.append((f"{oid} Total dV (m/s)", dv_arr))
        for oid in ("chaser", "target"):
            rem_arr = np.array(dv_remaining_m_s_by_object.get(oid, []), dtype=float)
            if rem_arr.size:
                plot_series.append((f"{oid} dV Remaining (m/s)", rem_arr))
        nplots = len(plot_series)
        if nplots == 6:
            nrows, ncols = 3, 2
        else:
            ncols = min(3, max(1, nplots))
            nrows = int(np.ceil(nplots / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=cap_figsize(5.2 * ncols, 3.8 * nrows), squeeze=False)
        axes_flat = list(np.ravel(axes))
        for ax, (title, arr) in zip(axes_flat, plot_series):
            bins = int(max(5, min(30, np.sqrt(max(arr.size, 1)))))
            ax.hist(arr, bins=bins, alpha=0.85)
            ax.set_title(title)
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[nplots:]:
            ax.set_visible(False)
        fig.tight_layout()
        if save_hist:
            fig.savefig(str(outdir / "master_monte_carlo_histograms.png"), dpi=int(cfg.outputs.plots.get("dpi", 150)))
        if show_hist:
            # Block so the histogram window is actually visible before close.
            plt.show()
        plt.close(fig)

        range_series_available = [s for s in relative_range_series_runs if isinstance(s, dict)]
        if range_series_available:
            fig_rr, ax_rr = plt.subplots(figsize=cap_figsize(10, 6))
            for idx, series in enumerate(relative_range_series_runs):
                if not isinstance(series, dict):
                    continue
                t_rr = np.array(series.get("time_s", []), dtype=float)
                r_rr = np.array(series.get("range_km", []), dtype=float)
                if t_rr.size == 0 or r_rr.size == 0:
                    continue
                ax_rr.plot(t_rr, r_rr, linewidth=1.0, alpha=0.65, label=f"run {idx}")
            ax_rr.set_title("Chaser-Target Relative Range by Iteration")
            ax_rr.set_xlabel("Time (s)")
            ax_rr.set_ylabel("Range (km)")
            ax_rr.grid(True, alpha=0.3)
            fig_rr.tight_layout()
            rr_path = outdir / "master_monte_carlo_relative_range_timeseries.png"
            if save_hist:
                fig_rr.savefig(str(rr_path), dpi=int(cfg.outputs.plots.get("dpi", 150)))
                agg["artifacts"]["relative_range_timeseries_png"] = str(rr_path)
            if show_hist:
                plt.show()
            plt.close(fig_rr)

    save_ops_dashboard = bool(mc_out_cfg.get("save_ops_dashboard", True))
    show_ops_dashboard = bool(mc_out_cfg.get("display_ops_dashboard", False))
    if (save_ops_dashboard or show_ops_dashboard) and run_details:
        import matplotlib.pyplot as plt

        ric_initial_samples = _mc_initial_relative_ric_curv_samples(cfg, run_details)
        pass_color = np.array(["tab:green" if bool(d.get("pass", False)) else "tab:red" for d in run_details], dtype=object)
        idx_arr = np.arange(len(run_details), dtype=float)
        ca_run_arr = np.array([_safe_float(d.get("closest_approach_km")) for d in run_details], dtype=float)
        dur_run_arr = np.array([_safe_float(d.get("duration_s"), default=0.0) for d in run_details], dtype=float)
        dv_run_arr = np.array([_safe_float(d.get("total_dv_m_s_total"), default=0.0) for d in run_details], dtype=float)

        fig, axes = plt.subplots(2, 3, figsize=cap_figsize(14, 8))
        bins = int(max(5, min(30, np.sqrt(max(len(run_details), 1)))))

        finite_ca = ca_run_arr[np.isfinite(ca_run_arr)]
        axes[0, 0].hist(finite_ca, bins=bins, alpha=0.85)
        if np.isfinite(keepout_threshold):
            axes[0, 0].axvline(keepout_threshold, linestyle="--", color="k")
        axes[0, 0].set_title("Closest Approach (km)")
        axes[0, 0].set_ylabel("count")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(dur_run_arr[np.isfinite(dur_run_arr)], bins=bins, alpha=0.85)
        max_duration_s = _safe_float(gates.get("max_duration_s"))
        if np.isfinite(max_duration_s):
            axes[0, 1].axvline(max_duration_s, linestyle="--", color="k")
        axes[0, 1].set_title("Duration (s)")
        axes[0, 1].set_ylabel("count")
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].hist(dv_run_arr[np.isfinite(dv_run_arr)], bins=bins, alpha=0.85, color="tab:orange")
        max_total_dv_m_s = _safe_float(gates.get("max_total_dv_m_s"))
        if np.isfinite(max_total_dv_m_s):
            axes[0, 2].axvline(max_total_dv_m_s, linestyle="--", color="k")
        axes[0, 2].set_title("Total dV (m/s)")
        axes[0, 2].set_ylabel("count")
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].scatter(idx_arr, ca_run_arr, c=pass_color, s=22, alpha=0.9)
        if np.isfinite(keepout_threshold):
            axes[1, 0].axhline(keepout_threshold, linestyle="--", color="k")
        axes[1, 0].set_title("Closest Approach by Run")
        axes[1, 0].set_xlabel("run index")
        axes[1, 0].set_ylabel("km")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].scatter(idx_arr, dv_run_arr, c=pass_color, s=22, alpha=0.9)
        if np.isfinite(max_total_dv_m_s):
            axes[1, 1].axhline(max_total_dv_m_s, linestyle="--", color="k")
        axes[1, 1].set_title("Total dV by Run")
        axes[1, 1].set_xlabel("run index")
        axes[1, 1].set_ylabel("m/s")
        axes[1, 1].grid(True, alpha=0.3)

        top_fail_pairs = sorted(failure_mode_counts.items(), key=lambda kv: int(kv[1]), reverse=True)[:6]
        if top_fail_pairs:
            labels = [k for k, _ in top_fail_pairs]
            vals = [int(v) for _, v in top_fail_pairs]
            axes[1, 2].bar(np.arange(len(vals)), vals, color="tab:red", alpha=0.85)
            axes[1, 2].set_xticks(np.arange(len(vals)))
            axes[1, 2].set_xticklabels(labels, rotation=30, ha="right")
        axes[1, 2].set_title("Failure Mode Counts")
        axes[1, 2].set_ylabel("count")
        axes[1, 2].grid(True, alpha=0.3)

        fig.suptitle("Monte Carlo Ops Dashboard", fontsize=12)
        fig.tight_layout()
        dashboard_path = outdir / "master_monte_carlo_ops_dashboard.png"
        if save_ops_dashboard:
            fig.savefig(str(dashboard_path), dpi=int(cfg.outputs.plots.get("dpi", 150)))
            agg["artifacts"]["ops_dashboard_png"] = str(dashboard_path)
        if show_ops_dashboard:
            plt.show()
        plt.close(fig)

        if ric_initial_samples:
            fig_ic, axes_ic = plt.subplots(3, 2, figsize=cap_figsize(12, 9), squeeze=False)
            scatter_specs = [
                ("radial_sep_km", "Initial Radial Separation (km)"),
                ("radial_vel_km_s", "Initial Radial Velocity (km/s)"),
                ("in_track_sep_km", "Initial In-Track Separation (km)"),
                ("in_track_vel_km_s", "Initial In-Track Velocity (km/s)"),
                ("cross_track_sep_km", "Initial Cross-Track Separation (km)"),
                ("cross_track_vel_km_s", "Initial Cross-Track Velocity (km/s)"),
            ]
            axes_ic_flat = [
                axes_ic[0, 0],
                axes_ic[0, 1],
                axes_ic[1, 0],
                axes_ic[1, 1],
                axes_ic[2, 0],
                axes_ic[2, 1],
            ]
            for ax, (key, xlabel) in zip(axes_ic_flat, scatter_specs):
                x = np.array(ric_initial_samples.get(key, []), dtype=float)
                finite = np.isfinite(x) & np.isfinite(ca_run_arr)
                ax.scatter(x[finite], ca_run_arr[finite], c=pass_color[finite], s=24, alpha=0.85)
                if np.isfinite(keepout_threshold):
                    ax.axhline(keepout_threshold, linestyle="--", color="k")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Closest Approach (km)")
                ax.grid(True, alpha=0.3)
            fig_ic.suptitle("Initial Relative RIC State vs Closest Approach", fontsize=12)
            fig_ic.tight_layout()
            ic_path = outdir / "master_monte_carlo_initial_relative_state_vs_closest_approach.png"
            if save_ops_dashboard:
                fig_ic.savefig(str(ic_path), dpi=int(cfg.outputs.plots.get("dpi", 150)))
                agg["artifacts"]["initial_relative_state_vs_closest_approach_png"] = str(ic_path)
            if show_ops_dashboard:
                plt.show()
            plt.close(fig_ic)

        rem_obj_ids = [oid for oid in ("chaser", "target") if oid in dv_budget_m_s_by_object]
        if rem_obj_ids:
            fig_rem, axes_rem = plt.subplots(2, len(rem_obj_ids), figsize=cap_figsize(5.0 * len(rem_obj_ids), 7.0), squeeze=False)
            for j, oid in enumerate(rem_obj_ids):
                rem_arr = np.array(
                    [
                        _safe_float(dict(d.get("delta_v_remaining_m_s_by_object", {}) or {}).get(oid))
                        for d in run_details
                    ],
                    dtype=float,
                )
                finite_rem = rem_arr[np.isfinite(rem_arr)]
                bins_rem = int(max(5, min(30, np.sqrt(max(finite_rem.size, 1)))))
                axes_rem[0, j].hist(finite_rem, bins=bins_rem, alpha=0.85, color="tab:blue")
                axes_rem[0, j].set_title(f"{oid} dV Remaining (m/s)")
                axes_rem[0, j].set_ylabel("count")
                axes_rem[0, j].grid(True, alpha=0.3)

                axes_rem[1, j].scatter(idx_arr, rem_arr, c=pass_color, s=22, alpha=0.9)
                axes_rem[1, j].set_title(f"{oid} dV Remaining by Run")
                axes_rem[1, j].set_xlabel("run index")
                axes_rem[1, j].set_ylabel("m/s")
                axes_rem[1, j].grid(True, alpha=0.3)
            fig_rem.suptitle("Monte Carlo Delta-V Remaining", fontsize=12)
            fig_rem.tight_layout()
            rem_path = outdir / "master_monte_carlo_delta_v_remaining.png"
            if save_ops_dashboard:
                fig_rem.savefig(str(rem_path), dpi=int(cfg.outputs.plots.get("dpi", 150)))
                agg["artifacts"]["delta_v_remaining_png"] = str(rem_path)
            if show_ops_dashboard:
                plt.show()
            plt.close(fig_rem)

    if bool(cfg.outputs.monte_carlo.get("save_aggregate_summary", True)):
        summary_path = outdir / "master_monte_carlo_summary.json"
        commander_json_path = outdir / "master_monte_carlo_commander_brief.json"
        commander_md_path = outdir / "master_monte_carlo_commander_brief.md"
        analyst_path = outdir / "master_monte_carlo_analyst_pack.json"
        write_json(str(summary_path), agg)
        write_json(str(commander_json_path), commander_brief)
        _write_commander_brief_markdown(commander_md_path, commander_brief)
        write_json(str(analyst_path), analyst_pack)
        agg["artifacts"]["summary_json"] = str(summary_path)
        agg["artifacts"]["commander_brief_json"] = str(commander_json_path)
        agg["artifacts"]["commander_brief_md"] = str(commander_md_path)
        agg["artifacts"]["analyst_pack_json"] = str(analyst_path)
        if bool(mc_out_cfg.get("save_raw_runs", False)):
            runs_path = outdir / "master_monte_carlo_run_details.json"
            write_json(str(runs_path), {"scenario_name": cfg.scenario_name, "run_details": run_details})
            agg["artifacts"]["run_details_json"] = str(runs_path)
    return agg
