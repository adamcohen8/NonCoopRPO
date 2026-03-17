import unittest

import numpy as np

from sim.dynamics.attitude.rigid_body import (
    get_attitude_guardrail_stats,
    propagate_attitude_exponential_map,
    reset_attitude_guardrail_stats,
)


class TestAttitudeGuardrailStats(unittest.TestCase):
    def test_guardrail_stats_increment_on_nonfinite_inputs(self) -> None:
        reset_attitude_guardrail_stats()
        q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        w0 = np.array([np.inf, 0.0, 0.0], dtype=float)
        inertia = np.diag([120.0, 100.0, 80.0])
        torque = np.array([np.nan, 0.0, 0.0], dtype=float)

        q1, w1 = propagate_attitude_exponential_map(
            quat_bn=q0,
            omega_body_rad_s=w0,
            inertia_kg_m2=inertia,
            torque_body_nm=torque,
            dt_s=0.1,
        )

        stats = get_attitude_guardrail_stats()
        self.assertGreater(stats["non_finite_input_events"], 0)
        self.assertGreater(stats["rate_clamp_events"], 0)
        self.assertGreater(stats["torque_clamp_events"], 0)
        self.assertTrue(np.all(np.isfinite(q1)))
        self.assertTrue(np.all(np.isfinite(w1)))


if __name__ == "__main__":
    unittest.main()
