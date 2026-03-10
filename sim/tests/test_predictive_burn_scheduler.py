import unittest

import numpy as np

from sim.control.orbit.predictive_burn import PredictiveBurnConfig, PredictiveBurnScheduler
from sim.core.models import Command, StateBelief, StateTruth


class _FakeLQR:
    ric_curv_state_slice = (0, 6)
    chief_eci_state_slice = (6, 12)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        return Command(thrust_eci_km_s2=np.array([0.0, 0.0, -1e-5]), torque_body_nm=np.zeros(3), mode_flags={})


class TestPredictiveBurnScheduler(unittest.TestCase):
    def test_plans_waits_then_burns_when_aligned(self):
        sched = PredictiveBurnScheduler(
            orbit_lqr=_FakeLQR(),
            thruster_direction_body=np.array([0.0, 0.0, 1.0]),
            config=PredictiveBurnConfig(horizon_steps=2, attitude_tolerance_rad=np.deg2rad(5.0)),
        )
        chaser = StateTruth(
            position_eci_km=np.array([7000.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.5, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=100.0,
            t_s=0.0,
        )
        chief = StateTruth(
            position_eci_km=np.array([7000.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.5, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=100.0,
            t_s=0.0,
        )
        b = StateBelief(state=np.hstack((chaser.position_eci_km, chaser.velocity_eci_km_s)), covariance=np.eye(6), last_update_t_s=0.0)

        d1 = sched.step(chaser, chief, b, b, dt_s=2.0)
        d2 = sched.step(chaser, chief, b, b, dt_s=2.0)
        d3 = sched.step(chaser, chief, b, b, dt_s=2.0)

        self.assertFalse(d1["fire"])
        self.assertFalse(d2["fire"])
        self.assertTrue(d3["fire"])
        self.assertGreater(np.linalg.norm(d3["thrust_eci_km_s2"]), 0.0)


if __name__ == "__main__":
    unittest.main()
