import unittest

import numpy as np

from sim.dynamics.attitude.rigid_body import propagate_attitude_exponential_map


class TestAttitudeExponentialMap(unittest.TestCase):
    def test_zero_torque_constant_rate_about_z(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        w0 = np.array([0.0, 0.0, 0.1])
        dt = 10.0

        q1, w1 = propagate_attitude_exponential_map(
            quat_bn=q0,
            omega_body_rad_s=w0,
            inertia_kg_m2=inertia,
            torque_body_nm=np.zeros(3),
            dt_s=dt,
        )

        theta = w0[2] * dt
        q_expected = np.array([np.cos(0.5 * theta), 0.0, 0.0, np.sin(0.5 * theta)])
        # Resolve sign ambiguity for quaternion comparison.
        if np.dot(q1, q_expected) < 0.0:
            q1 = -q1
        self.assertTrue(np.allclose(q1, q_expected, atol=1e-8))
        self.assertTrue(np.allclose(w1, w0, atol=1e-12))
        self.assertAlmostEqual(float(np.linalg.norm(q1)), 1.0, places=12)

    def test_large_dt_keeps_unit_norm(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        q = np.array([0.9, 0.2, -0.1, 0.35])
        q = q / np.linalg.norm(q)
        w = np.array([0.3, -0.2, 0.4])
        qn, _ = propagate_attitude_exponential_map(
            quat_bn=q,
            omega_body_rad_s=w,
            inertia_kg_m2=inertia,
            torque_body_nm=np.array([0.01, -0.02, 0.005]),
            dt_s=25.0,
        )
        self.assertAlmostEqual(float(np.linalg.norm(qn)), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
