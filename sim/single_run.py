from __future__ import annotations

from pathlib import Path
import logging
import os
from time import perf_counter
from typing import Any, Callable

import numpy as np

from sim.actuators.orbital import attitude_coupled_thrust_eci, thruster_disturbance_torque_body_nm
from sim.config import SimulationScenarioConfig, scenario_config_from_dict
from sim.core.models import Command, StateBelief, StateTruth
from sim.dynamics.attitude.rigid_body import get_attitude_guardrail_stats, reset_attitude_guardrail_stats
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.master_outputs import animate_outputs as _animate_outputs_impl
from sim.master_outputs import plot_outputs as _plot_outputs_impl
from sim.runtime_support import (
    AgentRuntime,
    _apply_chaser_relative_init_from_target,
    _attitude_state13_from_belief,
    _build_knowledge_base,
    _combine_commands,
    _command_to_dict,
    _create_rocket_runtime,
    _create_satellite_runtime,
    _deploy_from_rocket,
    _orbital_elements_basic,
    _relative_orbit_state12,
    _resolve_rocket_stack,
    _resolve_satellite_isp_s,
    _rocket_altitude_km,
    _rocket_state_to_truth,
    _run_mission_execution,
    _run_mission_modules,
    _run_mission_strategy,
    _to_jsonable_value,
    _truth_state6,
)
from sim.utils.io import write_json
from sim.utils.quaternion import quaternion_to_dcm_bn

logger = logging.getLogger(__name__)


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
    return _animate_outputs_impl(cfg=cfg, t_s=t_s, truth_hist=truth_hist, outdir=outdir)


def _fmt_float(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def _format_single_run_summary(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    scenario_description = str(summary.get("scenario_description", "") or "").strip()
    lines.append("")
    lines.append("=" * 72)
    lines.append("MASTER SIMULATION SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Scenario   : {summary.get('scenario_name', 'unknown')}")
    if scenario_description:
        lines.append(f"Desc       : {scenario_description}")
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
        lines.append(f"Insertion  : achieved at t={ins_t}" if ins_ok else "Insertion  : not achieved")
    thrust_stats = dict(summary.get("thrust_stats", {}) or {})
    if thrust_stats:
        lines.append("-" * 72)
        lines.append("Thrust Stats")
        lines.append(f"{'Object':<14}{'Burn Samples':>14}{'Max Accel (km/s^2)':>24}{'Total dV (m/s)':>18}")
        for oid in sorted(thrust_stats.keys()):
            stats = dict(thrust_stats.get(oid, {}) or {})
            lines.append(
                f"{oid:<14}"
                f"{int(stats.get('burn_samples', 0)):>14d}"
                f"{float(stats.get('max_accel_km_s2', 0.0)):>24.3e}"
                f"{float(stats.get('total_dv_m_s', 0.0)):>18.3f}"
            )
    plot_outputs = dict(summary.get("plot_outputs", {}) or {})
    anim_outputs = dict(summary.get("animation_outputs", {}) or {})
    guardrails = dict(summary.get("attitude_guardrail_stats", {}) or {})
    lines.append("-" * 72)
    lines.append(f"Artifacts  : plots={len(plot_outputs)}  animations={len(anim_outputs)}")
    lines.append(f"Guardrails : attitude_events={int(sum(int(v) for v in guardrails.values())) if guardrails else 0}")
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
        self.orbit_command_period_s = orbit_substep_s
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
        self.belief_hist = {
            aid: np.full((self.n, int(agent.belief.state.size) if agent.belief is not None else 0), np.nan)
            for aid, agent in self.agents.items()
        }
        self.thrust_hist = {aid: np.full((self.n, 3), np.nan) for aid in self.agents.keys()}
        self.torque_hist = {aid: np.full((self.n, 3), np.nan) for aid in self.agents.keys()}
        self.desired_attitude_hist = {aid: np.full((self.n, 4), np.nan) for aid in self.agents.keys()}
        self.controller_debug_hist: dict[str, list[dict[str, Any]]] = {aid: [] for aid in self.agents.keys()}
        self._last_orbital_command_eval_t_s: dict[str, float | None] = {aid: None for aid in self.agents.keys()}
        self._latched_orbital_thrust_cmd_by_object: dict[str, np.ndarray] = {
            aid: self.zero3.copy() for aid in self.agents.keys()
        }
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
                self._ensure_belief_hist_width(aid, agent.belief.state.size)
                self.belief_hist[aid][0, : agent.belief.state.size] = agent.belief.state
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

    def _ensure_belief_hist_width(self, aid: str, width: int) -> None:
        hist = self.belief_hist[aid]
        if hist.shape[1] >= int(width):
            return
        expanded = np.full((self.n, int(width)), np.nan)
        if hist.shape[1] > 0:
            expanded[:, : hist.shape[1]] = hist
        self.belief_hist[aid] = expanded

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

        world_truth = {
            aid: (agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state))
            for aid, agent in self.agents.items()
            if agent.active
        }
        world_truth_live = dict(world_truth)

        for aid, agent in self.agents.items():
            if not agent.active:
                continue
            tr_now = world_truth_live[aid]
            env_common = {**self.base_environment, "world_truth": world_truth_live, "attitude_disabled": (not self.attitude_enabled)}

            if agent.kind == "rocket":
                mission_out = _run_mission_modules(agent=agent, world_truth=world_truth_live, t_s=t_next, dt_s=self.dt, env=env_common)
                mission_out.update(_run_mission_strategy(agent=agent, world_truth=world_truth_live, t_s=t_next, dt_s=self.dt, env=env_common))
                mission_out.update(_run_mission_execution(agent=agent, intent=mission_out, world_truth=world_truth_live, t_s=t_next, dt_s=self.dt, env=env_common))
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
                env_inner = {**self.base_environment, "world_truth": world_truth_inner, "attitude_disabled": (not self.attitude_enabled)}
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
                        attitude_belief_scratch.state = _attitude_state13_from_belief(belief=att_belief, truth=tr_inner, out=attitude_state13_scratch)
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
                    mission_out.update(
                        _run_mission_strategy(
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
                    )
                    mission_out.update(
                        _run_mission_execution(
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
                    )
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
                    if self.attitude_enabled and "desired_ric_euler_rad" in mission_out and agent.attitude_controller is not None and hasattr(agent.attitude_controller, "set_desired_ric_state"):
                        e = np.array(mission_out["desired_ric_euler_rad"], dtype=float).reshape(-1)
                        if e.size == 3:
                            try:
                                agent.attitude_controller.set_desired_ric_state(float(e[0]), float(e[1]), float(e[2]))
                            except (TypeError, ValueError, AttributeError) as exc:
                                logger.warning("Failed to set desired_ric_euler_rad on %s controller: %s", aid, exc)
                    use_integrated_cmd = bool(mission_out.get("mission_use_integrated_command", False))
                    orbit_runtime_ms = 0.0
                    attitude_runtime_ms = 0.0
                    c_orb = Command.zero()
                    if (not use_integrated_cmd) and agent.orbit_controller is not None and orb_belief is not None:
                        orbit_t0 = perf_counter()
                        c_orb = agent.orbit_controller.act(orb_belief, t_eval, 2.0)
                        orbit_runtime_ms = (perf_counter() - orbit_t0) * 1000.0
                    c_att = Command.zero()
                    if self.attitude_enabled and (not use_integrated_cmd) and agent.attitude_controller is not None and att_belief is not None:
                        attitude_t0 = perf_counter()
                        c_att = agent.attitude_controller.act(att_belief, t_eval, 2.0)
                        attitude_runtime_ms = (perf_counter() - attitude_t0) * 1000.0
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
                    orbital_command_due = (
                        self._last_orbital_command_eval_t_s[aid] is None
                        or float(t_eval) - float(self._last_orbital_command_eval_t_s[aid]) >= self.orbit_command_period_s - 1e-12
                    )
                    if orbital_command_due:
                        self._last_orbital_command_eval_t_s[aid] = float(t_eval)
                        self._latched_orbital_thrust_cmd_by_object[aid] = np.array(cmd.thrust_eci_km_s2, dtype=float).reshape(3)
                    latched_thrust_cmd = np.array(self._latched_orbital_thrust_cmd_by_object[aid], dtype=float).reshape(3)
                    if not self.attitude_enabled:
                        cmd.torque_body_nm = self.zero3
                    cmd_step = Command(
                        thrust_eci_km_s2=latched_thrust_cmd,
                        torque_body_nm=(self.zero3.copy() if not self.attitude_enabled else np.array(cmd.torque_body_nm, dtype=float)),
                        mode_flags=dict(cmd.mode_flags or {}),
                    )
                    cmd_step.mode_flags["orbital_command_updated"] = bool(orbital_command_due)
                    if self._last_orbital_command_eval_t_s[aid] is not None:
                        cmd_step.mode_flags["orbital_command_sample_t_s"] = float(self._last_orbital_command_eval_t_s[aid])
                    cmd_step.mode_flags["current_attitude_quat_bn"] = np.array(tr_inner.attitude_quat_bn, dtype=float)
                    if agent.thruster_direction_body is not None:
                        cmd_step.mode_flags["thruster_direction_body"] = np.array(agent.thruster_direction_body, dtype=float)
                    if agent.thruster_position_body_m is not None:
                        cmd_step.mode_flags["thruster_position_body_m"] = np.array(agent.thruster_position_body_m, dtype=float)
                    if agent.thruster_direction_body is not None:
                        cmd_step.mode_flags["commanded_thrust_eci_km_s2"] = np.array(cmd_step.thrust_eci_km_s2, dtype=float)
                        cmd_step.thrust_eci_km_s2 = attitude_coupled_thrust_eci(
                            cmd_step.thrust_eci_km_s2,
                            attitude_quat_bn=np.array(tr_inner.attitude_quat_bn, dtype=float),
                            thruster_direction_body=np.array(agent.thruster_direction_body, dtype=float),
                        )
                    min_mass_kg = 0.0
                    if agent.dry_mass_kg is not None and np.isfinite(float(agent.dry_mass_kg)):
                        min_mass_kg = float(max(float(agent.dry_mass_kg), 0.0))
                    if bool(tr_inner.mass_kg <= (min_mass_kg + 1e-12)):
                        cmd_step.thrust_eci_km_s2 = np.zeros(3, dtype=float)
                        cmd_step.mode_flags["fuel_depleted"] = True
                    cmd_step.mode_flags["min_mass_kg"] = float(min_mass_kg)
                    thruster_torque_body_nm = np.zeros(3, dtype=float)
                    if (
                        self.attitude_enabled
                        and agent.thruster_direction_body is not None
                        and agent.thruster_position_body_m is not None
                    ):
                        thruster_torque_body_nm = thruster_disturbance_torque_body_nm(
                            cmd_step.thrust_eci_km_s2,
                            current_mass_kg=float(max(tr_inner.mass_kg, 0.0)),
                            thruster_direction_body=np.array(agent.thruster_direction_body, dtype=float),
                            thruster_position_body_m=np.array(agent.thruster_position_body_m, dtype=float),
                        )
                        cmd_step.torque_body_nm = np.array(cmd_step.torque_body_nm, dtype=float) + thruster_torque_body_nm
                    cmd_step.mode_flags["thruster_torque_body_nm"] = thruster_torque_body_nm.tolist()
                    isp_s = agent.orbital_isp_s
                    if isp_s is not None and float(isp_s) > 0.0 and "delta_mass_kg" not in cmd_step.mode_flags:
                        g0_m_s2 = 9.80665
                        a_mag_m_s2 = float(np.linalg.norm(cmd_step.thrust_eci_km_s2) * 1e3)
                        thrust_n = float(max(tr_inner.mass_kg, 0.0) * a_mag_m_s2)
                        mdot_kg_s = 0.0 if thrust_n <= 0.0 else float(thrust_n / (float(isp_s) * g0_m_s2))
                        delta_mass_kg = float(max(mdot_kg_s, 0.0) * h)
                        available_propellant_kg = float(max(tr_inner.mass_kg - min_mass_kg, 0.0))
                        cmd_step.mode_flags["delta_mass_kg"] = float(min(delta_mass_kg, available_propellant_kg))
                    self.controller_debug_hist[aid].append(
                        {
                            "t_s": float(t_eval),
                            "dt_s": float(h),
                            "belief": (np.array(agent.belief.state, dtype=float).tolist() if agent.belief is not None else None),
                            "orbit_belief": (np.array(orb_belief.state, dtype=float).tolist() if orb_belief is not None else None),
                            "attitude_belief": (np.array(att_belief.state, dtype=float).tolist() if att_belief is not None else None),
                            "orbit_controller_runtime_ms": float(orbit_runtime_ms),
                            "attitude_controller_runtime_ms": float(attitude_runtime_ms),
                            "controller_runtime_ms": float(orbit_runtime_ms + attitude_runtime_ms),
                            "command_orbit": _command_to_dict(c_orb),
                            "command_attitude": _command_to_dict(c_att),
                            "command_raw": _command_to_dict(cmd),
                            "command_applied": _command_to_dict(cmd_step),
                            "use_integrated_command": bool(use_integrated_cmd),
                            "mode_flags": _to_jsonable_value(dict(cmd_step.mode_flags or {})),
                        }
                    )
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
                self._ensure_belief_hist_width(aid, agent.belief.state.size)
                self.belief_hist[aid][k + 1, : agent.belief.state.size] = agent.belief.state

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

        if self.rocket is not None and self.rocket.active and (not self.rocket.waiting_for_launch) and self.rocket.rocket_state is not None and self.rocket.rocket_sim is not None:
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
        knowledge_out = {obs: {tgt: arr[:n_used, :].copy() for tgt, arr in by_tgt.items()} for obs, by_tgt in self.knowledge_hist.items()}
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
        animation_outputs = _animate_outputs(cfg=self.cfg, t_s=t_out, truth_hist=truth_out, outdir=self.outdir)

        thrust_stats = {
            oid: {
                "burn_samples": int(self.burn_samples_by_object.get(oid, 0)),
                "max_accel_km_s2": float(self.max_accel_km_s2_by_object.get(oid, 0.0)),
                "total_dv_m_s": float(self.total_dv_m_s_by_object.get(oid, 0.0)),
            }
            for oid in thrust_out.keys()
        }
        summary = {
            "scenario_name": self.cfg.scenario_name,
            "scenario_description": self.cfg.scenario_description,
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
            "desired_attitude_by_object": {k: v.tolist() for k, v in desired_attitude_out.items()},
            "knowledge_by_observer": {o: {t: a.tolist() for t, a in bt.items()} for o, bt in knowledge_out.items()},
            "knowledge_detection_by_observer": dict(summary.get("knowledge_detection_by_observer", {}) or {}),
            "knowledge_consistency_by_observer": dict(summary.get("knowledge_consistency_by_observer", {}) or {}),
            "bridge_events_by_object": self.bridge_hist,
            "controller_debug_by_object": self.controller_debug_hist,
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
