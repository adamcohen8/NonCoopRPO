from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from presets.rockets import BASIC_TWO_STAGE_STACK
from sim.core.models import StateBelief, StateTruth
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.knowledge.object_tracking import (
    KnowledgeConditionConfig,
    KnowledgeEKFConfig,
    KnowledgeNoiseConfig,
    ObjectKnowledgeBase,
    TrackedObjectConfig,
)
from sim.rocket import (
    OpenLoopPitchProgramGuidance,
    RocketAscentSimulator,
    RocketGuidanceLaw,
    RocketSimConfig,
    RocketState,
    RocketVehicleConfig,
)
from sim.utils.quaternion import quaternion_to_dcm_bn

StrategyMode = Literal["coast", "prograde", "retrograde", "knowledge_pursuit", "knowledge_evade"]


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _state_truth_to_vec(truth: StateTruth) -> np.ndarray:
    return np.hstack(
        (
            truth.position_eci_km,
            truth.velocity_eci_km_s,
            truth.attitude_quat_bn,
            truth.angular_rate_body_rad_s,
            np.array([truth.mass_kg]),
        )
    )


def _rocket_to_truth(s: RocketState) -> StateTruth:
    return StateTruth(
        position_eci_km=np.array(s.position_eci_km, dtype=float),
        velocity_eci_km_s=np.array(s.velocity_eci_km_s, dtype=float),
        attitude_quat_bn=np.array(s.attitude_quat_bn, dtype=float),
        angular_rate_body_rad_s=np.array(s.angular_rate_body_rad_s, dtype=float),
        mass_kg=float(s.mass_kg),
        t_s=float(s.t_s),
    )


@dataclass(frozen=True)
class AgentStrategyConfig:
    mode: StrategyMode = "coast"
    max_accel_km_s2: float = 0.0
    target_id: str | None = None


@dataclass(frozen=True)
class KnowledgeGateConfig:
    rocket_starts_tracking_target_at_s: float = 0.0
    target_starts_tracking_rocket_at_s: float = 0.0
    chaser_starts_tracking_target_at_s: float = 0.0


@dataclass(frozen=True)
class ASATPhasedScenarioConfig:
    dt_s: float = 1.0
    duration_s: float = 3600.0
    deploy_time_s: float = 600.0
    chaser_mass_kg: float = 200.0
    chaser_deploy_dv_body_m_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rocket_id: str = "asat_rocket"
    target_id: str = "target"
    chaser_id: str = "chaser"
    atmosphere_model: str = "ussa1976"
    seed: int = 123
    terminate_on_earth_impact: bool = True
    earth_impact_radius_km: float = 6378.137

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive.")
        if self.deploy_time_s < 0.0:
            raise ValueError("deploy_time_s must be non-negative.")
        if self.chaser_mass_kg <= 0.0:
            raise ValueError("chaser_mass_kg must be positive.")
        if np.array(self.chaser_deploy_dv_body_m_s, dtype=float).reshape(-1).size != 3:
            raise ValueError("chaser_deploy_dv_body_m_s must be length-3.")
        if self.earth_impact_radius_km <= 0.0:
            raise ValueError("earth_impact_radius_km must be positive.")


def _default_target_state() -> StateTruth:
    r = np.array([7000.0, 0.0, 0.0], dtype=float)
    v = np.array([0.0, np.sqrt(EARTH_MU_KM3_S2 / np.linalg.norm(r)), 0.0], dtype=float)
    return StateTruth(
        position_eci_km=r,
        velocity_eci_km_s=v,
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3),
        mass_kg=400.0,
        t_s=0.0,
    )


def _default_track(target_id: str, refresh_s: float) -> TrackedObjectConfig:
    return TrackedObjectConfig(
        target_id=target_id,
        conditions=KnowledgeConditionConfig(
            refresh_rate_s=refresh_s,
            max_range_km=50_000.0,
            require_line_of_sight=True,
            dropout_prob=0.0,
        ),
        sensor_noise=KnowledgeNoiseConfig(
            pos_sigma_km=np.array([0.01, 0.01, 0.01]),
            vel_sigma_km_s=np.array([1e-4, 1e-4, 1e-4]),
        ),
        estimator="ekf",
        ekf=KnowledgeEKFConfig(),
    )


def _strategy_accel_eci(
    *,
    self_truth: StateTruth,
    own_knowledge: dict[str, StateBelief],
    cfg: AgentStrategyConfig,
) -> np.ndarray:
    amax = float(max(cfg.max_accel_km_s2, 0.0))
    if amax <= 0.0 or cfg.mode == "coast":
        return np.zeros(3)
    if cfg.mode == "prograde":
        return amax * _unit(self_truth.velocity_eci_km_s)
    if cfg.mode == "retrograde":
        return -amax * _unit(self_truth.velocity_eci_km_s)
    if cfg.mode in ("knowledge_pursuit", "knowledge_evade"):
        if cfg.target_id is None:
            return np.zeros(3)
        b = own_knowledge.get(cfg.target_id)
        if b is None:
            return np.zeros(3)
        rel = np.array(b.state[:3], dtype=float) - self_truth.position_eci_km
        d = _unit(rel)
        if cfg.mode == "knowledge_evade":
            d = -d
        return amax * d
    return np.zeros(3)


def run_asat_phased_engagement(
    scenario_cfg: ASATPhasedScenarioConfig,
    *,
    rocket_sim_cfg: RocketSimConfig | None = None,
    rocket_vehicle_cfg: RocketVehicleConfig | None = None,
    rocket_guidance: RocketGuidanceLaw | None = None,
    target_initial_truth: StateTruth | None = None,
    rocket_strategy: AgentStrategyConfig = AgentStrategyConfig(mode="coast", max_accel_km_s2=0.0),
    target_strategy: AgentStrategyConfig = AgentStrategyConfig(mode="coast", max_accel_km_s2=0.0),
    chaser_strategy: AgentStrategyConfig = AgentStrategyConfig(mode="coast", max_accel_km_s2=0.0),
    gates: KnowledgeGateConfig = KnowledgeGateConfig(),
    rocket_track_cfg: TrackedObjectConfig | None = None,
    target_track_cfg: TrackedObjectConfig | None = None,
    chaser_track_cfg: TrackedObjectConfig | None = None,
) -> dict:
    rng = np.random.default_rng(scenario_cfg.seed)
    dt = float(scenario_cfg.dt_s)
    n = int(np.floor(scenario_cfg.duration_s / dt)) + 1
    t_s = np.arange(n, dtype=float) * dt

    sim_cfg = rocket_sim_cfg or RocketSimConfig(dt_s=dt, max_time_s=scenario_cfg.duration_s, atmosphere_model=scenario_cfg.atmosphere_model)
    vehicle_cfg = rocket_vehicle_cfg or RocketVehicleConfig(stack=BASIC_TWO_STAGE_STACK, payload_mass_kg=150.0)
    guidance = rocket_guidance or OpenLoopPitchProgramGuidance()
    target = (target_initial_truth.copy() if target_initial_truth is not None else _default_target_state())
    target.t_s = 0.0

    rocket_sim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=guidance)
    rocket = rocket_sim.initial_state()

    rocket_track = rocket_track_cfg or _default_track(target_id=scenario_cfg.target_id, refresh_s=max(dt, 1.0))
    target_track = target_track_cfg or _default_track(target_id=scenario_cfg.rocket_id, refresh_s=max(dt, 1.0))
    chaser_track = chaser_track_cfg or _default_track(target_id=scenario_cfg.target_id, refresh_s=max(dt, 1.0))

    rocket_kb = ObjectKnowledgeBase(
        observer_id=scenario_cfg.rocket_id, tracked_objects=[rocket_track], dt_s=dt, rng=np.random.default_rng(rng.integers(0, 2**31 - 1))
    )
    target_kb = ObjectKnowledgeBase(
        observer_id=scenario_cfg.target_id, tracked_objects=[target_track], dt_s=dt, rng=np.random.default_rng(rng.integers(0, 2**31 - 1))
    )
    chaser_kb = ObjectKnowledgeBase(
        observer_id=scenario_cfg.chaser_id, tracked_objects=[chaser_track], dt_s=dt, rng=np.random.default_rng(rng.integers(0, 2**31 - 1))
    )

    ids = [scenario_cfg.rocket_id, scenario_cfg.target_id, scenario_cfg.chaser_id]
    truth_hist = {oid: np.full((n, 14), np.nan) for oid in ids}
    accel_cmd_hist = {oid: np.full((n, 3), np.nan) for oid in ids}
    throttle_hist = {scenario_cfg.rocket_id: np.full(n, np.nan)}
    knowledge_hist = {
        scenario_cfg.rocket_id: {scenario_cfg.target_id: np.full((n, 6), np.nan)},
        scenario_cfg.target_id: {scenario_cfg.rocket_id: np.full((n, 6), np.nan)},
        scenario_cfg.chaser_id: {scenario_cfg.target_id: np.full((n, 6), np.nan)},
    }
    chaser_deployed = False
    deploy_index = -1
    chaser: StateTruth | None = None
    terminated_early = False
    termination_reason: str | None = None
    termination_time_s: float | None = None
    termination_object_id: str | None = None
    final_index = n - 1

    truth_hist[scenario_cfg.rocket_id][0, :] = _state_truth_to_vec(_rocket_to_truth(rocket))
    truth_hist[scenario_cfg.target_id][0, :] = _state_truth_to_vec(target)
    accel_cmd_hist[scenario_cfg.rocket_id][0, :] = 0.0
    accel_cmd_hist[scenario_cfg.target_id][0, :] = 0.0

    propagator = OrbitPropagator(integrator="rk4", plugins=[])

    for k in range(n - 1):
        t = t_s[k]
        world_truth: dict[str, StateTruth] = {
            scenario_cfg.rocket_id: _rocket_to_truth(rocket),
            scenario_cfg.target_id: target,
        }
        if chaser is not None:
            world_truth[scenario_cfg.chaser_id] = chaser

        if t >= gates.rocket_starts_tracking_target_at_s:
            rocket_kb.update(observer_truth=world_truth[scenario_cfg.rocket_id], world_truth=world_truth, t_s=t)
        if t >= gates.target_starts_tracking_rocket_at_s:
            target_kb.update(observer_truth=world_truth[scenario_cfg.target_id], world_truth=world_truth, t_s=t)
        if chaser is not None and t >= gates.chaser_starts_tracking_target_at_s:
            chaser_kb.update(observer_truth=world_truth[scenario_cfg.chaser_id], world_truth=world_truth, t_s=t)

        rb = rocket_kb.snapshot().get(scenario_cfg.target_id)
        if rb is not None:
            knowledge_hist[scenario_cfg.rocket_id][scenario_cfg.target_id][k, :] = rb.state[:6]
        tb = target_kb.snapshot().get(scenario_cfg.rocket_id)
        if tb is not None:
            knowledge_hist[scenario_cfg.target_id][scenario_cfg.rocket_id][k, :] = tb.state[:6]
        if chaser is not None:
            cb = chaser_kb.snapshot().get(scenario_cfg.target_id)
            if cb is not None:
                knowledge_hist[scenario_cfg.chaser_id][scenario_cfg.target_id][k, :] = cb.state[:6]

        rocket_cmd = guidance.command(rocket, sim_cfg, vehicle_cfg)
        rocket_throttle = float(np.clip(rocket_cmd.throttle, 0.0, 1.0))
        throttle_hist[scenario_cfg.rocket_id][k] = rocket_throttle
        # Optional strategy acceleration term for the rocket (knowledge-driven guidance extension).
        rocket_a_extra = _strategy_accel_eci(
            self_truth=world_truth[scenario_cfg.rocket_id], own_knowledge=rocket_kb.snapshot(), cfg=rocket_strategy
        )
        accel_cmd_hist[scenario_cfg.rocket_id][k, :] = rocket_a_extra
        rocket = rocket_sim.step(rocket, rocket_cmd, dt)

        target_accel = _strategy_accel_eci(
            self_truth=target,
            own_knowledge=target_kb.snapshot(),
            cfg=target_strategy,
        )
        accel_cmd_hist[scenario_cfg.target_id][k, :] = target_accel
        x_t = np.hstack((target.position_eci_km, target.velocity_eci_km_s))
        x_tn = propagator.propagate(
            x_eci=x_t,
            dt_s=dt,
            t_s=t,
            command_accel_eci_km_s2=target_accel,
            env={"atmosphere_model": scenario_cfg.atmosphere_model},
            ctx=OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=target.mass_kg),
        )
        target = StateTruth(
            position_eci_km=x_tn[:3],
            velocity_eci_km_s=x_tn[3:],
            attitude_quat_bn=target.attitude_quat_bn.copy(),
            angular_rate_body_rad_s=target.angular_rate_body_rad_s.copy(),
            mass_kg=target.mass_kg,
            t_s=t + dt,
        )

        if (not chaser_deployed) and ((t + dt) >= scenario_cfg.deploy_time_s):
            chaser_deployed = True
            deploy_index = k + 1
            c_bn = quaternion_to_dcm_bn(rocket.attitude_quat_bn)
            dv_eci_km_s = (c_bn.T @ np.array(scenario_cfg.chaser_deploy_dv_body_m_s, dtype=float).reshape(3)) / 1e3
            chaser = StateTruth(
                position_eci_km=np.array(rocket.position_eci_km, dtype=float),
                velocity_eci_km_s=np.array(rocket.velocity_eci_km_s, dtype=float) + dv_eci_km_s,
                attitude_quat_bn=np.array(rocket.attitude_quat_bn, dtype=float),
                angular_rate_body_rad_s=np.array(rocket.angular_rate_body_rad_s, dtype=float),
                mass_kg=float(scenario_cfg.chaser_mass_kg),
                t_s=t + dt,
            )

        if chaser is not None:
            chaser_accel = _strategy_accel_eci(
                self_truth=chaser,
                own_knowledge=chaser_kb.snapshot(),
                cfg=chaser_strategy,
            )
            accel_cmd_hist[scenario_cfg.chaser_id][k, :] = chaser_accel
            x_c = np.hstack((chaser.position_eci_km, chaser.velocity_eci_km_s))
            x_cn = propagator.propagate(
                x_eci=x_c,
                dt_s=dt,
                t_s=t,
                command_accel_eci_km_s2=chaser_accel,
                env={"atmosphere_model": scenario_cfg.atmosphere_model},
                ctx=OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=chaser.mass_kg),
            )
            chaser = StateTruth(
                position_eci_km=x_cn[:3],
                velocity_eci_km_s=x_cn[3:],
                attitude_quat_bn=chaser.attitude_quat_bn.copy(),
                angular_rate_body_rad_s=chaser.angular_rate_body_rad_s.copy(),
                mass_kg=chaser.mass_kg,
                t_s=t + dt,
            )

        truth_hist[scenario_cfg.rocket_id][k + 1, :] = _state_truth_to_vec(_rocket_to_truth(rocket))
        truth_hist[scenario_cfg.target_id][k + 1, :] = _state_truth_to_vec(target)
        if chaser is not None:
            truth_hist[scenario_cfg.chaser_id][k + 1, :] = _state_truth_to_vec(chaser)
        else:
            accel_cmd_hist[scenario_cfg.chaser_id][k + 1, :] = np.nan

        if scenario_cfg.terminate_on_earth_impact:
            impact_map = {
                scenario_cfg.rocket_id: float(np.linalg.norm(rocket.position_eci_km)) <= float(scenario_cfg.earth_impact_radius_km),
                scenario_cfg.target_id: float(np.linalg.norm(target.position_eci_km)) <= float(scenario_cfg.earth_impact_radius_km),
            }
            if chaser is not None:
                impact_map[scenario_cfg.chaser_id] = float(np.linalg.norm(chaser.position_eci_km)) <= float(
                    scenario_cfg.earth_impact_radius_km
                )
            impacted = [oid for oid, hit in impact_map.items() if hit]
            if impacted:
                terminated_early = True
                termination_reason = "earth_impact"
                termination_object_id = impacted[0]
                termination_time_s = float(t_s[k + 1])
                final_index = k + 1
                break

    rb = rocket_kb.snapshot().get(scenario_cfg.target_id)
    if rb is not None:
        knowledge_hist[scenario_cfg.rocket_id][scenario_cfg.target_id][-1, :] = rb.state[:6]
    tb = target_kb.snapshot().get(scenario_cfg.rocket_id)
    if tb is not None:
        knowledge_hist[scenario_cfg.target_id][scenario_cfg.rocket_id][-1, :] = tb.state[:6]
    cb = chaser_kb.snapshot().get(scenario_cfg.target_id)
    if cb is not None:
        knowledge_hist[scenario_cfg.chaser_id][scenario_cfg.target_id][-1, :] = cb.state[:6]

    n_used = final_index + 1
    t_out = t_s[:n_used].copy()
    truth_out = {oid: arr[:n_used, :].copy() for oid, arr in truth_hist.items()}
    accel_out = {oid: arr[:n_used, :].copy() for oid, arr in accel_cmd_hist.items()}
    throttle_out = throttle_hist[scenario_cfg.rocket_id][:n_used].copy()
    knowledge_out: dict[str, dict[str, np.ndarray]] = {}
    for obs, by_tgt in knowledge_hist.items():
        knowledge_out[obs] = {tgt: hist[:n_used, :].copy() for tgt, hist in by_tgt.items()}

    min_range_km = np.nan
    if chaser_deployed:
        rel = truth_out[scenario_cfg.chaser_id][:, :3] - truth_out[scenario_cfg.target_id][:, :3]
        rr = np.linalg.norm(rel, axis=1)
        good = np.isfinite(rr)
        if np.any(good):
            min_range_km = float(np.min(rr[good]))

    return {
        "time_s": t_out,
        "truth_by_object": truth_out,
        "knowledge_by_observer": knowledge_out,
        "accel_cmd_eci_km_s2_by_object": accel_out,
        "rocket_throttle_cmd": throttle_out,
        "chaser_deployed": chaser_deployed,
        "chaser_deploy_index": int(deploy_index),
        "chaser_deploy_time_s": float(t_s[deploy_index]) if deploy_index >= 0 else None,
        "min_chaser_target_range_km": min_range_km,
        "terminated_early": terminated_early,
        "termination_reason": termination_reason,
        "termination_time_s": termination_time_s,
        "termination_object_id": termination_object_id,
    }
