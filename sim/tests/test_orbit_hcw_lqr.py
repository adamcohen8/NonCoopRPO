import unittest

import numpy as np

from sim.control.orbit.lqr import HCWLQRController
from sim.core.models import StateBelief


class TestHCWLQRController(unittest.TestCase):
    def test_zero_state_commands_zero_accel(self):
        ctrl = HCWLQRController(mean_motion_rad_s=0.0011, max_accel_km_s2=1e-4, design_dt_s=10.0)
        state = np.zeros(12)
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)
        self.assertTrue(np.allclose(cmd.thrust_eci_km_s2, np.zeros(3), atol=1e-12))
        self.assertEqual(cmd.mode_flags["mode"], "hcw_lqr")

    def test_saturates_to_max_accel(self):
        ctrl = HCWLQRController(mean_motion_rad_s=0.0011, max_accel_km_s2=2e-5, design_dt_s=10.0)
        state = np.zeros(12)
        state[0:6] = np.array([2.0, -1.0, 0.5, 0.01, -0.02, 0.03])
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)
        self.assertLessEqual(np.linalg.norm(cmd.thrust_eci_km_s2), 2e-5 + 1e-12)

    def test_state_slice_signs_and_frame_rotation_are_applied(self):
        ctrl = HCWLQRController(
            mean_motion_rad_s=0.0011,
            max_accel_km_s2=1e-4,
            ric_curv_state_slice=(2, 8),
            chief_eci_state_slice=(8, 14),
            state_signs=np.array([1, -1, 1, 1, -1, 1], dtype=float),
            design_dt_s=10.0,
        )
        full_state = np.zeros(14)
        full_state[2:8] = np.array([0.3, 0.2, -0.1, 0.0, 0.01, -0.02])
        # Chief chosen so RIC basis differs from ECI; this validates rotation to ECI output.
        full_state[8:14] = np.array([0.0, 7000.0, 0.0, -7.5, 0.0, 0.0])
        belief = StateBelief(state=full_state, covariance=np.eye(14), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)
        self.assertEqual(cmd.mode_flags["ric_curv_state_slice"], [2, 8])
        self.assertEqual(cmd.mode_flags["chief_eci_state_slice"], [8, 14])
        self.assertEqual(len(cmd.mode_flags["state_signs"]), 6)
        self.assertEqual(cmd.thrust_eci_km_s2.shape, (3,))
        a_ric = np.array(cmd.mode_flags["accel_ric_km_s2"], dtype=float)
        self.assertFalse(np.allclose(cmd.thrust_eci_km_s2, a_ric))

    def test_linear_feedback_debug_terms_sum_to_axis_commands(self):
        ctrl = HCWLQRController(mean_motion_rad_s=0.0011, max_accel_km_s2=1e-4, design_dt_s=10.0)
        state = np.zeros(12)
        state[0:6] = np.array([0.7, -0.4, 0.2, 0.01, -0.015, 0.005])
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)
        debug = dict(cmd.mode_flags["linear_feedback_debug"])
        self.assertEqual(debug["law_label"], "-Kx")
        self.assertEqual(debug["control_axes"], ["R", "I", "C"])
        contrib = np.array(debug["term_contributions_post_limit"], dtype=float)
        summed = np.sum(contrib, axis=1)
        self.assertTrue(np.allclose(summed, np.array(debug["control_post_limit"], dtype=float)))

    def test_linear_system_summary_reports_closed_loop_poles_and_position_zeros(self):
        ctrl = HCWLQRController(mean_motion_rad_s=0.0011, max_accel_km_s2=1e-4, design_dt_s=10.0)
        summary = ctrl.linear_system_summary()
        self.assertEqual(summary["system_type"], "discrete_state_feedback")
        self.assertEqual(summary["control_axes"], ["R", "I", "C"])
        self.assertEqual(len(summary["closed_loop_poles"]), 6)
        self.assertEqual(len(summary["position_channel_zeros"]), 3)
        self.assertEqual(summary["position_channel_zeros"][0]["axis"], "R")


if __name__ == "__main__":
    unittest.main()
