import unittest

import numpy as np

from sim.control.orbit.lqr_no_radial import HCWNoRadialLQRController
from sim.core.models import StateBelief


class TestHCWNoRadialLQRController(unittest.TestCase):
    def test_zero_state_commands_zero_accel(self):
        ctrl = HCWNoRadialLQRController(mean_motion_rad_s=0.0011, max_accel_km_s2=1e-4, design_dt_s=10.0)
        state = np.zeros(12)
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)
        self.assertTrue(np.allclose(cmd.thrust_eci_km_s2, np.zeros(3), atol=1e-12))
        self.assertEqual(cmd.mode_flags["mode"], "hcw_lqr_no_radial")

    def test_radial_axis_is_uncontrolled(self):
        ctrl = HCWNoRadialLQRController(mean_motion_rad_s=0.0011, max_accel_km_s2=1e-4, design_dt_s=10.0)
        state = np.zeros(12)
        state[0:6] = np.array([1.0, -2.0, 0.75, 0.02, -0.03, 0.01])
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)
        accel_ric = np.array(cmd.mode_flags["accel_ric_km_s2"], dtype=float)
        self.assertEqual(cmd.mode_flags["control_axes"], ["I", "C"])
        self.assertAlmostEqual(float(accel_ric[0]), 0.0, places=12)
        self.assertGreater(float(np.linalg.norm(accel_ric[1:])), 0.0)

    def test_saturates_to_max_accel(self):
        ctrl = HCWNoRadialLQRController(mean_motion_rad_s=0.0011, max_accel_km_s2=2e-5, design_dt_s=10.0)
        state = np.zeros(12)
        state[0:6] = np.array([2.0, -1.0, 0.5, 0.01, -0.02, 0.03])
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)
        self.assertLessEqual(np.linalg.norm(cmd.thrust_eci_km_s2), 2e-5 + 1e-12)

    def test_state_slice_signs_and_frame_rotation_are_applied(self):
        ctrl = HCWNoRadialLQRController(
            mean_motion_rad_s=0.0011,
            max_accel_km_s2=1e-4,
            ric_curv_state_slice=(2, 8),
            chief_eci_state_slice=(8, 14),
            state_signs=np.array([1, -1, 1, 1, -1, 1], dtype=float),
            design_dt_s=10.0,
        )
        full_state = np.zeros(14)
        full_state[2:8] = np.array([0.3, 0.2, -0.1, 0.0, 0.01, -0.02])
        full_state[8:14] = np.array([0.0, 7000.0, 0.0, -7.5, 0.0, 0.0])
        belief = StateBelief(state=full_state, covariance=np.eye(14), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)
        self.assertEqual(cmd.mode_flags["ric_curv_state_slice"], [2, 8])
        self.assertEqual(cmd.mode_flags["chief_eci_state_slice"], [8, 14])
        self.assertEqual(len(cmd.mode_flags["state_signs"]), 6)
        self.assertEqual(cmd.thrust_eci_km_s2.shape, (3,))
        accel_ric = np.array(cmd.mode_flags["accel_ric_km_s2"], dtype=float)
        self.assertAlmostEqual(float(accel_ric[0]), 0.0, places=12)
        self.assertFalse(np.allclose(cmd.thrust_eci_km_s2, accel_ric))


if __name__ == "__main__":
    unittest.main()
