from __future__ import annotations

import unittest

import numpy as np

from sim.core.kernel import SimObject, SimulationKernel
from sim.core.models import Command, Measurement, ObjectConfig, SimConfig, StateBelief, StateTruth


class _PassThroughDynamics:
    def step(self, state: StateTruth, command: Command, env: dict, dt_s: float) -> StateTruth:
        pos = state.position_eci_km.copy()
        # Drive altitude down quickly to trigger impact termination.
        pos[0] -= 50.0
        return StateTruth(
            position_eci_km=pos,
            velocity_eci_km_s=state.velocity_eci_km_s.copy(),
            attitude_quat_bn=state.attitude_quat_bn.copy(),
            angular_rate_body_rad_s=state.angular_rate_body_rad_s.copy(),
            mass_kg=state.mass_kg,
            t_s=state.t_s + dt_s,
        )


class _NoSensor:
    def measure(self, truth: StateTruth, env: dict, t_s: float) -> Measurement | None:
        return None


class _HoldEstimator:
    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        return StateBelief(state=belief.state.copy(), covariance=belief.covariance.copy(), last_update_t_s=t_s)


class _ZeroController:
    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        return Command.zero()


class _PassActuator:
    def apply(self, command: Command, limits: dict, dt_s: float) -> Command:
        return command


class TestTerminationConditions(unittest.TestCase):
    def _make_object(self) -> SimObject:
        truth = StateTruth(
            position_eci_km=np.array([6385.0, 0.0, 0.0]),
            velocity_eci_km_s=np.zeros(3),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=1.0,
            t_s=0.0,
        )
        belief = StateBelief(state=np.zeros(6), covariance=np.eye(6), last_update_t_s=0.0)
        return SimObject(
            cfg=ObjectConfig(object_id="obj"),
            truth=truth,
            belief=belief,
            dynamics=_PassThroughDynamics(),
            sensor=_NoSensor(),
            estimator=_HoldEstimator(),
            controller=_ZeroController(),
            actuator=_PassActuator(),
            limits={},
        )

    def test_kernel_terminates_on_earth_impact(self):
        kernel = SimulationKernel(
            config=SimConfig(dt_s=1.0, steps=20, terminate_on_earth_impact=True, earth_impact_radius_km=6378.137),
            objects=[self._make_object()],
            env={},
        )
        log = kernel.run()
        self.assertTrue(log.terminated_early)
        self.assertEqual(log.termination_reason, "earth_impact")
        self.assertEqual(log.termination_object_id, "obj")
        self.assertLess(log.t_s.size, 21)


if __name__ == "__main__":
    unittest.main()
