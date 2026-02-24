from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .atmosphere import EARTH_EQUATORIAL_RADIUS_KM
from .dynamics import two_body_deriv
from .frames import eci_to_rsw_dcm
from .integrators import rk4_step
from .knowledge import KnowledgeModel
from .launch import (
    InsertionCriteria,
    LaunchResult,
    LaunchSite,
    LaunchTimingMode,
    OrbitTarget,
    Rocket,
    simulate_launch_to_insertion,
)
from .satellite import Policy, Satellite
from .strategies import StrategyLike


@dataclass(frozen=True)
class MasterScenario:
    dt_s: float
    pre_sim_duration_s: float = 0.0
    sim_duration_s: float = 0.0
    sim_steps: Optional[int] = None
    terminate_below_radius_km: Optional[float] = EARTH_EQUATORIAL_RADIUS_KM

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("MasterScenario.dt_s must be positive.")
        if self.pre_sim_duration_s < 0.0:
            raise ValueError("MasterScenario.pre_sim_duration_s must be non-negative.")
        if self.sim_duration_s < 0.0:
            raise ValueError("MasterScenario.sim_duration_s must be non-negative.")
        if self.sim_steps is not None and self.sim_steps < 0:
            raise ValueError("MasterScenario.sim_steps must be non-negative when provided.")
        if self.sim_steps is None and self.sim_duration_s <= 0.0:
            raise ValueError("Provide either sim_steps or positive sim_duration_s.")
        if self.terminate_below_radius_km is not None and self.terminate_below_radius_km <= 0.0:
            raise ValueError("terminate_below_radius_km must be positive when provided.")

    @property
    def pre_steps(self) -> int:
        return int(np.ceil(self.pre_sim_duration_s / self.dt_s))

    @property
    def main_steps(self) -> int:
        if self.sim_steps is not None:
            return int(self.sim_steps)
        return int(np.ceil(self.sim_duration_s / self.dt_s))

    @property
    def total_steps(self) -> int:
        return self.pre_steps + self.main_steps


@dataclass
class SatelliteAgentConfig:
    name: str
    satellite: Satellite
    observe_target_name: Optional[str] = None
    pre_sim_strategy: Optional[StrategyLike] = None
    pre_sim_policy: Optional[Policy] = None
    allow_pre_sim_maneuvers: bool = False


@dataclass
class RocketAgentConfig:
    name: str
    site: LaunchSite
    rocket: Rocket
    target_orbit: OrbitTarget
    insertion_criteria: InsertionCriteria
    launch_time_s: float = 0.0
    timing_mode: LaunchTimingMode = LaunchTimingMode.GO_NOW
    inserted_satellite_name: Optional[str] = None
    inserted_strategy: Optional[StrategyLike] = None
    inserted_policy: Optional[Policy] = None
    observe_target_name: Optional[str] = None
    pre_sim_strategy: Optional[StrategyLike] = None
    pre_sim_policy: Optional[Policy] = None
    allow_pre_sim_maneuvers: bool = False


@dataclass(frozen=True)
class RocketLaunchEvent:
    rocket_name: str
    scheduled_launch_time_s: float
    insertion_time_s: float
    inserted_satellite_name: str
    inserted: bool
    insertion_reason: str


@dataclass
class MasterSimLog:
    t_s: np.ndarray
    phase: np.ndarray
    x_eci_by_agent: dict[str, np.ndarray]
    u_eci_by_agent: dict[str, np.ndarray]
    u_ric_by_agent: dict[str, np.ndarray]
    rocket_t_s_by_name: dict[str, np.ndarray] = field(default_factory=dict)
    rocket_x_eci_by_name: dict[str, np.ndarray] = field(default_factory=dict)
    rocket_thrust_newton_by_name: dict[str, np.ndarray] = field(default_factory=dict)
    terminated_early: bool = False
    termination_reason: str = "max_time_elapsed"
    launches: list[RocketLaunchEvent] = field(default_factory=list)

    @classmethod
    def allocate(cls, agent_names: list[str], steps: int, pre_steps: int, dt_s: float) -> "MasterSimLog":
        n = steps + 1
        phase = np.full(n, "sim", dtype="<U4")
        if pre_steps > 0:
            phase[:] = "pre"
        return cls(
            t_s=np.arange(n, dtype=float) * dt_s,
            phase=phase,
            x_eci_by_agent={name: np.full((n, 6), np.nan) for name in agent_names},
            u_eci_by_agent={name: np.full((n, 3), np.nan) for name in agent_names},
            u_ric_by_agent={name: np.full((n, 3), np.nan) for name in agent_names},
            rocket_t_s_by_name={},
            rocket_x_eci_by_name={},
            rocket_thrust_newton_by_name={},
            terminated_early=False,
            termination_reason="max_time_elapsed",
            launches=[],
        )

    def save_npz(self, path: str) -> None:
        agent_names = sorted(self.x_eci_by_agent.keys())
        payload: dict[str, np.ndarray] = {
            "t_s": self.t_s,
            "phase": self.phase,
            "agent_names": np.asarray(agent_names, dtype=str),
            "launches_json": np.asarray(json.dumps([event.__dict__ for event in self.launches]), dtype=str),
            "terminated_early": np.asarray(int(self.terminated_early), dtype=np.int64),
            "termination_reason": np.asarray(self.termination_reason, dtype=str),
        }
        for name in agent_names:
            payload[f"x_eci::{name}"] = self.x_eci_by_agent[name]
            payload[f"u_eci::{name}"] = self.u_eci_by_agent[name]
            payload[f"u_ric::{name}"] = self.u_ric_by_agent[name]
        rocket_names = sorted(self.rocket_t_s_by_name.keys())
        payload["rocket_names"] = np.asarray(rocket_names, dtype=str)
        for name in rocket_names:
            payload[f"rocket_t_s::{name}"] = self.rocket_t_s_by_name[name]
            payload[f"rocket_x_eci::{name}"] = self.rocket_x_eci_by_name[name]
            payload[f"rocket_thrust_newton::{name}"] = self.rocket_thrust_newton_by_name[name]

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **payload)

    @classmethod
    def load_npz(cls, path: str) -> "MasterSimLog":
        with np.load(path, allow_pickle=False) as data:
            t_s = data["t_s"].copy()
            phase = data["phase"].copy()
            agent_names = [str(name) for name in data["agent_names"]]
            x_eci_by_agent = {name: data[f"x_eci::{name}"].copy() for name in agent_names}
            u_eci_by_agent = {name: data[f"u_eci::{name}"].copy() for name in agent_names}
            u_ric_by_agent = {name: data[f"u_ric::{name}"].copy() for name in agent_names}
            rocket_t_s_by_name: dict[str, np.ndarray] = {}
            rocket_x_eci_by_name: dict[str, np.ndarray] = {}
            rocket_thrust_newton_by_name: dict[str, np.ndarray] = {}
            if "rocket_names" in data:
                rocket_names = [str(name) for name in data["rocket_names"]]
                for name in rocket_names:
                    rocket_t_s_by_name[name] = data[f"rocket_t_s::{name}"].copy()
                    rocket_x_eci_by_name[name] = data[f"rocket_x_eci::{name}"].copy()
                    rocket_thrust_newton_by_name[name] = data[f"rocket_thrust_newton::{name}"].copy()
            terminated_early = bool(int(data["terminated_early"])) if "terminated_early" in data else False
            termination_reason = (
                str(data["termination_reason"]) if "termination_reason" in data else "max_time_elapsed"
            )
            launches_raw = json.loads(str(data["launches_json"]))
            launches = [RocketLaunchEvent(**item) for item in launches_raw]

        return cls(
            t_s=t_s,
            phase=phase,
            x_eci_by_agent=x_eci_by_agent,
            u_eci_by_agent=u_eci_by_agent,
            u_ric_by_agent=u_ric_by_agent,
            rocket_t_s_by_name=rocket_t_s_by_name,
            rocket_x_eci_by_name=rocket_x_eci_by_name,
            rocket_thrust_newton_by_name=rocket_thrust_newton_by_name,
            terminated_early=terminated_early,
            termination_reason=termination_reason,
            launches=launches,
        )


@dataclass
class _PendingInsertion:
    activation_time_s: float
    agent_cfg: SatelliteAgentConfig


class MasterSimulator:
    def __init__(self, scenario: MasterScenario, knowledge_model: Optional[KnowledgeModel] = None):
        self.scenario = scenario
        self.knowledge_model = knowledge_model or KnowledgeModel()

    def run(
        self,
        satellites: list[SatelliteAgentConfig],
        rockets: Optional[list[RocketAgentConfig]] = None,
        log_path: Optional[str] = None,
    ) -> MasterSimLog:
        rockets = rockets or []

        base_names = [cfg.name for cfg in satellites]
        rocket_sat_names = [cfg.inserted_satellite_name or f"{cfg.name}_inserted" for cfg in rockets]
        all_names = base_names + rocket_sat_names
        if len(set(all_names)) != len(all_names):
            raise ValueError("Agent names must be unique across satellites and rocket insertion products.")

        active_agents: dict[str, SatelliteAgentConfig] = {cfg.name: cfg for cfg in satellites}
        pending_insertions: list[_PendingInsertion] = []
        launched_rockets: set[str] = set()

        max_total_steps = self.scenario.pre_steps + self.scenario.main_steps
        log = MasterSimLog.allocate(
            agent_names=all_names,
            steps=max_total_steps,
            pre_steps=self.scenario.pre_steps,
            dt_s=self.scenario.dt_s,
        )

        self._activate_insertions(
            pending_insertions=pending_insertions,
            active_agents=active_agents,
            t_now_s=0.0,
        )
        self._log_snapshot(log=log, idx=0, active_agents=active_agents)

        pre_phase_active = len(rockets) > 0
        sim_elapsed_steps = 0
        final_k = 0
        log.phase[0] = "pre" if pre_phase_active else "sim"
        for k in range(max_total_steps):
            t_now = k * self.scenario.dt_s
            launched_this_step = False

            for rocket_cfg in rockets:
                if rocket_cfg.name in launched_rockets:
                    continue
                if rocket_cfg.launch_time_s > t_now:
                    continue

                result = simulate_launch_to_insertion(
                    site=rocket_cfg.site,
                    rocket=rocket_cfg.rocket,
                    target_orbit=rocket_cfg.target_orbit,
                    insertion_criteria=rocket_cfg.insertion_criteria,
                    timing_mode=rocket_cfg.timing_mode,
                    satellite_name=rocket_cfg.inserted_satellite_name or f"{rocket_cfg.name}_inserted",
                )
                pending_insertions.append(self._build_pending_insertion(rocket_cfg, result, t_now))
                launched_rockets.add(rocket_cfg.name)
                launched_this_step = True
                t_abs = t_now + (result.t_s - result.launch_delay_s)
                log.rocket_t_s_by_name[rocket_cfg.name] = t_abs.copy()
                log.rocket_x_eci_by_name[rocket_cfg.name] = result.x_eci.copy()
                log.rocket_thrust_newton_by_name[rocket_cfg.name] = result.thrust_newton.copy()
                log.launches.append(
                    RocketLaunchEvent(
                        rocket_name=rocket_cfg.name,
                        scheduled_launch_time_s=rocket_cfg.launch_time_s,
                        insertion_time_s=pending_insertions[-1].activation_time_s,
                        inserted_satellite_name=pending_insertions[-1].agent_cfg.name,
                        inserted=result.inserted,
                        insertion_reason=result.insertion_reason,
                    )
                )

            if pre_phase_active and launched_this_step:
                pre_phase_active = False

            in_pre_sim = pre_phase_active
            self._activate_insertions(
                pending_insertions=pending_insertions,
                active_agents=active_agents,
                t_now_s=t_now,
            )

            state_snapshot = {name: cfg.satellite.state.x_eci.copy() for name, cfg in active_agents.items()}
            u_eci_map: dict[str, np.ndarray] = {}
            u_ric_map: dict[str, np.ndarray] = {}

            for name, cfg in active_agents.items():
                obs = None
                detected = False
                if cfg.observe_target_name is not None and cfg.observe_target_name in state_snapshot:
                    obs, detected = self.knowledge_model.observe(
                        observer_x_eci=state_snapshot[name],
                        target_x_eci=state_snapshot[cfg.observe_target_name],
                    )

                u_ric = np.zeros(3, dtype=float)
                if detected:
                    pre_strategy = cfg.pre_sim_strategy or cfg.pre_sim_policy
                    if in_pre_sim and not cfg.allow_pre_sim_maneuvers and pre_strategy is None:
                        u_ric = np.zeros(3, dtype=float)
                    elif in_pre_sim and pre_strategy is not None:
                        u_ric = self._command_accel_ric(cfg.satellite, t_now, obs, pre_strategy)
                    elif in_pre_sim and cfg.allow_pre_sim_maneuvers:
                        u_ric = self._command_accel_ric(cfg.satellite, t_now, obs, None)
                    elif not in_pre_sim:
                        u_ric = self._command_accel_ric(cfg.satellite, t_now, obs, None)

                rsw = eci_to_rsw_dcm(state_snapshot[name][0:3], state_snapshot[name][3:6])
                u_eci_map[name] = rsw @ u_ric
                u_ric_map[name] = u_ric

            for name, cfg in active_agents.items():
                sat = cfg.satellite
                sat.state.x_eci = rk4_step(
                    two_body_deriv,
                    sat.state.x_eci,
                    self.scenario.dt_s,
                    sat.params.mu,
                    u_eci_map[name],
                )
                sat.state.t = t_now + self.scenario.dt_s

            self._activate_insertions(
                pending_insertions=pending_insertions,
                active_agents=active_agents,
                t_now_s=t_now + self.scenario.dt_s,
            )

            self._log_snapshot(log=log, idx=k + 1, active_agents=active_agents)
            log.phase[k + 1] = "pre" if in_pre_sim else "sim"
            for name in u_eci_map:
                log.u_eci_by_agent[name][k + 1, :] = u_eci_map[name]
                log.u_ric_by_agent[name][k + 1, :] = u_ric_map[name]

            final_k = k + 1
            if pre_phase_active:
                if (k + 1) >= self.scenario.pre_steps:
                    raise RuntimeError(
                        "Pre-sim ended without rocket launch. Increase pre_sim_duration_s or adjust launch conditions."
                    )
            else:
                termination_reason = self._termination_reason(active_agents)
                if termination_reason is not None:
                    log.terminated_early = True
                    log.termination_reason = termination_reason
                    break
                sim_elapsed_steps += 1
                if sim_elapsed_steps >= self.scenario.main_steps:
                    log.terminated_early = False
                    log.termination_reason = "max_time_elapsed"
                    break

        self._trim_log_in_place(log, last_idx=final_k)
        if log_path is not None:
            log.save_npz(log_path)
        return log

    def _trim_log_in_place(self, log: MasterSimLog, last_idx: int) -> None:
        keep = slice(0, last_idx + 1)
        log.t_s = log.t_s[keep].copy()
        log.phase = log.phase[keep].copy()
        for name in list(log.x_eci_by_agent.keys()):
            log.x_eci_by_agent[name] = log.x_eci_by_agent[name][keep, :].copy()
            log.u_eci_by_agent[name] = log.u_eci_by_agent[name][keep, :].copy()
            log.u_ric_by_agent[name] = log.u_ric_by_agent[name][keep, :].copy()

    def _termination_reason(self, active_agents: dict[str, SatelliteAgentConfig]) -> Optional[str]:
        min_radius_km = self.scenario.terminate_below_radius_km
        if min_radius_km is None:
            return None

        for name, cfg in active_agents.items():
            radius_km = float(np.linalg.norm(cfg.satellite.state.x_eci[0:3]))
            if radius_km < min_radius_km:
                return f"agent_below_radius::{name}"
        return None

    def _build_pending_insertion(self, cfg: RocketAgentConfig, result: LaunchResult, launch_time_s: float) -> _PendingInsertion:
        insertion_name = cfg.inserted_satellite_name or f"{cfg.name}_inserted"
        inserted_sat = result.satellite
        inserted_sat.set_strategy(cfg.inserted_strategy or cfg.inserted_policy)
        inserted_sat.state.t = launch_time_s + (result.t_s[-1] - result.launch_delay_s)

        agent_cfg = SatelliteAgentConfig(
            name=insertion_name,
            satellite=inserted_sat,
            observe_target_name=cfg.observe_target_name,
            pre_sim_strategy=cfg.pre_sim_strategy,
            pre_sim_policy=cfg.pre_sim_policy,
            allow_pre_sim_maneuvers=cfg.allow_pre_sim_maneuvers,
        )
        return _PendingInsertion(activation_time_s=inserted_sat.state.t, agent_cfg=agent_cfg)

    def _activate_insertions(
        self,
        pending_insertions: list[_PendingInsertion],
        active_agents: dict[str, SatelliteAgentConfig],
        t_now_s: float,
    ) -> None:
        still_pending: list[_PendingInsertion] = []
        for pending in pending_insertions:
            if pending.activation_time_s > t_now_s:
                still_pending.append(pending)
                continue

            sat = pending.agent_cfg.satellite
            dt_backlog = t_now_s - pending.activation_time_s
            if dt_backlog > 0.0:
                sat.state.x_eci = rk4_step(
                    two_body_deriv,
                    sat.state.x_eci,
                    dt_backlog,
                    sat.params.mu,
                    np.zeros(3, dtype=float),
                )
            sat.state.t = t_now_s
            active_agents[pending.agent_cfg.name] = pending.agent_cfg

        pending_insertions[:] = still_pending

    def _log_snapshot(self, log: MasterSimLog, idx: int, active_agents: dict[str, SatelliteAgentConfig]) -> None:
        for name, cfg in active_agents.items():
            log.x_eci_by_agent[name][idx, :] = cfg.satellite.state.x_eci

    def _command_accel_ric(
        self,
        sat: Satellite,
        t_s: float,
        obs: Optional[np.ndarray],
        policy_override: Optional[StrategyLike],
    ) -> np.ndarray:
        if obs is None:
            return np.zeros(3, dtype=float)

        if policy_override is None:
            return sat.command_accel_ric(t_s, obs, self.scenario.dt_s)

        previous_strategy = sat.strategy
        sat.set_strategy(policy_override)
        try:
            return sat.command_accel_ric(t_s, obs, self.scenario.dt_s)
        finally:
            sat.strategy = previous_strategy
