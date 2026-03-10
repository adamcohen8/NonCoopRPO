import unittest

import numpy as np

from sim.control.attitude.baseline import SmallAngleLQRController
from sim.core.models import StateBelief


class TestSmallAngleLQRController(unittest.TestCase):
    def test_generalized_inertia_and_wheel_mounting(self):
        inertia = np.array(
            [
                [120.0, 4.0, 0.0],
                [4.0, 100.0, 2.0],
                [0.0, 2.0, 80.0],
            ]
        )
        wheel_axes = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        ctrl = SmallAngleLQRController(
            inertia_kg_m2=inertia,
            wheel_axes_body=wheel_axes,
            wheel_torque_limits_nm=np.array([0.05, 0.05, 0.03]),
            q_weights=np.array([10.0, 10.0, 8.0, 1.0, 1.0, 0.8]),
            r_weights=np.array([1.0, 1.0, 1.2]),
            design_dt_s=0.2,
        )

        state = np.zeros(13)
        state[6:10] = np.array([0.9987, 0.0300, -0.0200, 0.0100])
        state[10:13] = np.array([0.01, -0.02, 0.03])
        belief = StateBelief(state=state, covariance=np.eye(13), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)

        self.assertIn(cmd.mode_flags["mode"], {"lqr_track", "lqr_capture"})
        self.assertEqual(len(cmd.mode_flags["wheel_torque_cmd_nm"]), 3)
        self.assertEqual(cmd.torque_body_nm.shape, (3,))
        self.assertTrue(np.all(np.isfinite(cmd.torque_body_nm)))

    def test_zero_error_commands_zero_torque(self):
        ctrl = SmallAngleLQRController(
            inertia_kg_m2=np.diag([110.0, 90.0, 75.0]),
            wheel_axes_body=np.eye(3),
            wheel_torque_limits_nm=np.array([0.05, 0.05, 0.05]),
        )
        state = np.zeros(13)
        state[6:10] = np.array([1.0, 0.0, 0.0, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(13), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)

        self.assertTrue(np.allclose(cmd.torque_body_nm, np.zeros(3), atol=1e-12))
        wheel_cmd = np.array(cmd.mode_flags["wheel_torque_cmd_nm"], dtype=float)
        self.assertTrue(np.allclose(wheel_cmd, np.zeros(3), atol=1e-12))

    def test_large_error_uses_capture_mode(self):
        ctrl = SmallAngleLQRController(
            inertia_kg_m2=np.diag([110.0, 90.0, 75.0]),
            wheel_axes_body=np.eye(3),
            wheel_torque_limits_nm=np.array([0.05, 0.05, 0.05]),
            capture_enabled=True,
            capture_angle_deg=15.0,
        )
        state = np.zeros(13)
        # About 120 deg rotation about +Y.
        state[6:10] = np.array([0.5, 0.0, 0.8660254, 0.0])
        state[10:13] = np.array([0.02, -0.01, 0.015])
        belief = StateBelief(state=state, covariance=np.eye(13), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)
        self.assertEqual(cmd.mode_flags["mode"], "lqr_capture")
        self.assertGreater(cmd.mode_flags["attitude_error_deg"], 15.0)

    def test_robust_profile_factory(self):
        ctrl = SmallAngleLQRController.robust_profile(
            inertia_kg_m2=np.diag([110.0, 90.0, 75.0]),
            wheel_axes_body=np.eye(3),
            wheel_torque_limits_nm=np.array([0.05, 0.05, 0.05]),
            design_dt_s=1.0,
        )
        self.assertTrue(ctrl.capture_enabled)
        self.assertGreater(ctrl.capture_angle_deg, 0.0)


if __name__ == "__main__":
    unittest.main()
