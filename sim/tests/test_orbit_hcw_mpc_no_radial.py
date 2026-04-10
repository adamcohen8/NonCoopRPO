import unittest

import numpy as np

from sim.control.orbit.hcw_mpc import HCWInTrackCrossTrackMPCController
from sim.core.models import StateBelief


class TestHCWInTrackCrossTrackMPCController(unittest.TestCase):
    def test_zero_state_commands_zero_accel(self):
        ctrl = HCWInTrackCrossTrackMPCController(
            max_accel_km_s2=1e-4,
            horizon_time_s=20.0,
            model_dt_s=10.0,
            max_iterations=1,
        )
        state = np.zeros(12)
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=5.0)
        self.assertTrue(np.allclose(cmd.thrust_eci_km_s2, np.zeros(3), atol=1e-12))
        self.assertEqual(cmd.mode_flags["mode"], "relative_orbit_hcw_mpc_no_radial")
        self.assertEqual(cmd.mode_flags["control_axes"], ["I", "C"])

    def test_radial_axis_is_uncontrolled(self):
        ctrl = HCWInTrackCrossTrackMPCController(
            max_accel_km_s2=2e-5,
            horizon_time_s=20.0,
            model_dt_s=10.0,
            max_iterations=1,
        )
        state = np.zeros(12)
        state[0:6] = np.array([1.0, -2.0, 0.75, 0.02, -0.03, 0.01])
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=5.0)
        accel_ric = np.array(cmd.mode_flags["accel_ric_km_s2"], dtype=float)
        self.assertAlmostEqual(float(accel_ric[0]), 0.0, places=12)
        self.assertGreater(float(np.linalg.norm(accel_ric[1:])), 0.0)

    def test_saturates_to_max_accel(self):
        ctrl = HCWInTrackCrossTrackMPCController(
            max_accel_km_s2=1e-6,
            horizon_time_s=20.0,
            model_dt_s=10.0,
            max_iterations=1,
        )
        state = np.zeros(12)
        state[0:6] = np.array([2.0, -1.0, 0.5, 0.01, -0.02, 0.03])
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=5.0)
        self.assertLessEqual(np.linalg.norm(cmd.thrust_eci_km_s2), 1e-6 + 1e-12)

    def test_accepts_two_axis_weight_vectors(self):
        ctrl = HCWInTrackCrossTrackMPCController(
            max_accel_km_s2=1e-4,
            horizon_time_s=20.0,
            model_dt_s=10.0,
            max_iterations=1,
            r_weights=np.array([1.0e10, 2.0e10]),
            rd_weights=np.array([3.0e10, 4.0e10]),
        )
        self.assertTrue(np.allclose(ctrl.r_weights, np.array([1.0e10, 2.0e10])))
        self.assertTrue(np.allclose(ctrl.rd_weights, np.array([3.0e10, 4.0e10])))

    def test_control_signs_can_flip_in_track_command(self):
        ctrl = HCWInTrackCrossTrackMPCController(
            max_accel_km_s2=1e-4,
            horizon_time_s=20.0,
            model_dt_s=10.0,
            max_iterations=1,
            control_signs=np.array([-1.0, 1.0]),
        )
        self.assertTrue(np.allclose(ctrl.control_signs, np.array([-1.0, 1.0])))
        self.assertTrue(np.allclose(ctrl._control_to_ric(np.array([2.0e-6, 3.0e-6])), np.array([0.0, -2.0e-6, 3.0e-6])))


if __name__ == "__main__":
    unittest.main()
