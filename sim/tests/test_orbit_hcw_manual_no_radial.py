import unittest

import numpy as np

from sim.control.orbit.lqr_no_radial import HCWNoRadialManualController
from sim.core.models import StateBelief


class TestHCWNoRadialManualController(unittest.TestCase):
    def test_zero_gain_commands_zero_accel(self):
        ctrl = HCWNoRadialManualController(
            mean_motion_rad_s=0.0011,
            max_accel_km_s2=1e-4,
            design_dt_s=10.0,
            k_gain=np.zeros((2, 6), dtype=float),
        )
        state = np.zeros(12)
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)
        self.assertTrue(np.allclose(cmd.thrust_eci_km_s2, np.zeros(3), atol=1e-12))
        self.assertEqual(cmd.mode_flags["mode"], "hcw_manual_no_radial")
        self.assertEqual(cmd.mode_flags["control_axes"], ["I", "C"])

    def test_manual_gain_maps_to_in_track_and_cross_track_outputs(self):
        ctrl = HCWNoRadialManualController(
            mean_motion_rad_s=0.0011,
            max_accel_km_s2=20.0,
            design_dt_s=10.0,
            k_gain=np.array(
                [
                    [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
        )
        state = np.zeros(12)
        state[0:6] = np.array([0.5, -2.0, 4.0, 0.0, 0.0, 0.0])
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)
        accel_ric = np.array(cmd.mode_flags["accel_ric_km_s2"], dtype=float)
        self.assertAlmostEqual(float(accel_ric[0]), 0.0, places=12)
        self.assertGreater(float(accel_ric[1]), 0.0)
        self.assertLess(float(accel_ric[2]), 0.0)
        debug = dict(cmd.mode_flags["linear_feedback_debug"])
        self.assertEqual(debug["control_axes"], ["I", "C"])
        self.assertEqual(np.array(debug["gain_matrix"], dtype=float).shape, (2, 6))

    def test_rejects_wrong_gain_size(self):
        with self.assertRaises(ValueError):
            HCWNoRadialManualController(
                mean_motion_rad_s=0.0011,
                max_accel_km_s2=1e-4,
                k_gain=np.ones(6, dtype=float),
            )


if __name__ == "__main__":
    unittest.main()
