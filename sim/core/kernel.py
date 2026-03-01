from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .interfaces import Actuator, Controller, DynamicsModel, Estimator, SensorModel
from .models import Command, ObjectConfig, SimConfig, SimLog, StateBelief, StateTruth
from .scheduler import evaluate_controller_runtime


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


class SimulationKernel:
    def __init__(self, config: SimConfig, objects: list[SimObject], env: dict | None = None):
        self.config = config
        self.objects = {obj.cfg.object_id: obj for obj in objects}
        self.object_ids = sorted(self.objects.keys())
        if len(self.object_ids) != len(objects):
            raise ValueError("object ids must be unique")
        self.env = env or {}

    def run(self) -> SimLog:
        n = self.config.steps + 1
        t_s = np.arange(n, dtype=float) * self.config.dt_s

        truth_by_object = {oid: np.zeros((n, 14)) for oid in self.object_ids}
        belief_by_object = {oid: np.zeros((n, self.objects[oid].belief.state.size)) for oid in self.object_ids}
        thrust_by_object = {oid: np.zeros((n, 3)) for oid in self.object_ids}
        torque_by_object = {oid: np.zeros((n, 3)) for oid in self.object_ids}
        runtime_by_object = {oid: np.zeros(n) for oid in self.object_ids}
        skipped_by_object = {oid: np.zeros(n, dtype=bool) for oid in self.object_ids}

        pending_applied_command = {oid: Command.zero() for oid in self.object_ids}

        for oid in self.object_ids:
            obj = self.objects[oid]
            truth_by_object[oid][0, :] = _truth_to_array(obj.truth)
            belief_by_object[oid][0, :] = obj.belief.state

        for k in range(self.config.steps):
            now_s = t_s[k]

            propagated_truth: dict[str, StateTruth] = {}
            for oid in self.object_ids:
                obj = self.objects[oid]
                propagated_truth[oid] = obj.dynamics.step(
                    state=obj.truth,
                    command=pending_applied_command[oid],
                    env={"world_truth": {i: self.objects[i].truth for i in self.object_ids}, **self.env},
                    dt_s=self.config.dt_s,
                )

            measurements = {}
            for oid in self.object_ids:
                obj = self.objects[oid]
                obj.truth = propagated_truth[oid]
                measurements[oid] = obj.sensor.measure(
                    truth=obj.truth,
                    env={"world_truth": propagated_truth, **self.env},
                    t_s=now_s + self.config.dt_s,
                )

            for oid in self.object_ids:
                obj = self.objects[oid]
                obj.belief = obj.estimator.update(
                    belief=obj.belief,
                    measurement=measurements[oid],
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

        return SimLog(
            t_s=t_s,
            truth_by_object=truth_by_object,
            belief_by_object=belief_by_object,
            applied_thrust_by_object=thrust_by_object,
            applied_torque_by_object=torque_by_object,
            controller_runtime_ms_by_object=runtime_by_object,
            controller_skipped_by_object=skipped_by_object,
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
