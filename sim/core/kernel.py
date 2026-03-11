from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .interfaces import Actuator, Controller, DynamicsModel, Estimator, SensorModel
from .models import Command, ObjectConfig, SimConfig, SimLog, StateBelief, StateTruth
from .scheduler import evaluate_controller_runtime
from sim.dynamics.orbit.epoch import resolve_time_dependent_env
from sim.dynamics.orbit.eclipse import srp_shadow_factor


@dataclass
class SimObject:
    cfg: ObjectConfig
    truth: StateTruth
    belief: StateBelief
    dynamics: DynamicsModel
    sensor: SensorModel
    estimator: Estimator
    controller: Controller
    actuator: Actuator
    limits: dict
    knowledge_base: Any | None = None


class SimulationKernel:
    def __init__(self, config: SimConfig, objects: list[SimObject], env: dict | None = None):
        self.config = config
        self.objects = {obj.cfg.object_id: obj for obj in objects}
        self.object_ids = sorted(self.objects.keys())
        if len(self.object_ids) != len(objects):
            raise ValueError("object ids must be unique")
        self.env = env or {}
        if self.config.initial_jd_utc is not None and "jd_utc_start" not in self.env:
            self.env["jd_utc_start"] = float(self.config.initial_jd_utc)
        if "jd_utc_start" in self.env and "ephemeris_mode" not in self.env:
            self.env["ephemeris_mode"] = "analytic_simple"

    def run(self) -> SimLog:
        n = self.config.steps + 1
        t_s = np.arange(n, dtype=float) * self.config.dt_s

        truth_by_object = {oid: np.zeros((n, 14)) for oid in self.object_ids}
        belief_by_object = {oid: np.zeros((n, self.objects[oid].belief.state.size)) for oid in self.object_ids}
        knowledge_by_observer: dict[str, dict[str, np.ndarray]] = {}
        thrust_by_object = {oid: np.zeros((n, 3)) for oid in self.object_ids}
        torque_by_object = {oid: np.zeros((n, 3)) for oid in self.object_ids}
        runtime_by_object = {oid: np.zeros(n) for oid in self.object_ids}
        skipped_by_object = {oid: np.zeros(n, dtype=bool) for oid in self.object_ids}
        srp_shadow_by_object = {oid: np.ones(n) for oid in self.object_ids}

        pending_applied_command = {oid: Command.zero() for oid in self.object_ids}

        for oid in self.object_ids:
            obj = self.objects[oid]
            truth_by_object[oid][0, :] = _truth_to_array(obj.truth)
            belief_by_object[oid][0, :] = obj.belief.state
            if obj.knowledge_base is not None:
                knowledge_by_observer[oid] = {}
                for target_id in obj.knowledge_base.target_ids():
                    knowledge_by_observer[oid][target_id] = np.full((n, 6), np.nan)
                snapshot = obj.knowledge_base.snapshot()
                for target_id, kb in snapshot.items():
                    if target_id in knowledge_by_observer[oid]:
                        knowledge_by_observer[oid][target_id][0, :] = kb.state[:6]

        for k in range(self.config.steps):
            now_s = t_s[k]
            env_now = resolve_time_dependent_env(self.env, now_s)
            env_next = resolve_time_dependent_env(self.env, now_s + self.config.dt_s)

            propagated_truth: dict[str, StateTruth] = {}
            for oid in self.object_ids:
                obj = self.objects[oid]
                propagated_truth[oid] = obj.dynamics.step(
                    state=obj.truth,
                    command=pending_applied_command[oid],
                    env={"world_truth": {i: self.objects[i].truth for i in self.object_ids}, **env_now},
                    dt_s=self.config.dt_s,
                )

            measurements = {}
            for oid in self.object_ids:
                obj = self.objects[oid]
                obj.truth = propagated_truth[oid]
                measurements[oid] = obj.sensor.measure(
                    truth=obj.truth,
                    env={"world_truth": propagated_truth, **env_next},
                    t_s=now_s + self.config.dt_s,
                )

            for oid in self.object_ids:
                obj = self.objects[oid]
                obj.belief = obj.estimator.update(
                    belief=obj.belief,
                    measurement=measurements[oid],
                    t_s=now_s + self.config.dt_s,
                )

            for oid in self.object_ids:
                obj = self.objects[oid]
                if obj.knowledge_base is None:
                    continue
                obj.knowledge_base.update(
                    observer_truth=obj.truth,
                    world_truth=propagated_truth,
                    t_s=now_s + self.config.dt_s,
                )

            next_applied_command = {}
            for oid in self.object_ids:
                obj = self.objects[oid]
                budget_ms = obj.cfg.budget_ms(self.config.controller_budget_ms)
                start = perf_counter()
                cmd = obj.controller.act(obj.belief, now_s + self.config.dt_s, budget_ms)
                runtime_ms = (perf_counter() - start) * 1000.0
                decision = evaluate_controller_runtime(runtime_ms=runtime_ms, budget_ms=budget_ms)

                skipped = False
                if self.config.realtime_mode and decision.overrun:
                    cmd = Command.zero()
                    skipped = True

                applied = obj.actuator.apply(cmd, obj.limits, self.config.dt_s)
                next_applied_command[oid] = applied

                runtime_by_object[oid][k + 1] = runtime_ms
                skipped_by_object[oid][k + 1] = skipped
                thrust_by_object[oid][k + 1, :] = applied.thrust_eci_km_s2
                torque_by_object[oid][k + 1, :] = applied.torque_body_nm

            pending_applied_command = next_applied_command

            for oid in self.object_ids:
                obj = self.objects[oid]
                truth_by_object[oid][k + 1, :] = _truth_to_array(obj.truth)
                belief_by_object[oid][k + 1, :] = obj.belief.state
                if obj.knowledge_base is not None and oid in knowledge_by_observer:
                    snapshot = obj.knowledge_base.snapshot()
                    for target_id, arr in knowledge_by_observer[oid].items():
                        kb = snapshot.get(target_id)
                        if kb is not None:
                            arr[k + 1, :] = kb.state[:6]
                        else:
                            arr[k + 1, :] = arr[k, :]
                srp_shadow_by_object[oid][k + 1] = srp_shadow_factor(
                    r_sc_eci_km=obj.truth.position_eci_km,
                    t_s=now_s + self.config.dt_s,
                    env=env_next,
                )

        return SimLog(
            t_s=t_s,
            truth_by_object=truth_by_object,
            belief_by_object=belief_by_object,
            knowledge_by_observer=knowledge_by_observer,
            applied_thrust_by_object=thrust_by_object,
            applied_torque_by_object=torque_by_object,
            controller_runtime_ms_by_object=runtime_by_object,
            controller_skipped_by_object=skipped_by_object,
            srp_shadow_by_object=srp_shadow_by_object,
        )


def _truth_to_array(truth: StateTruth) -> np.ndarray:
    return np.hstack(
        (
            truth.position_eci_km,
            truth.velocity_eci_km_s,
            truth.attitude_quat_bn,
            truth.angular_rate_body_rad_s,
            np.array([truth.mass_kg]),
        )
    )
