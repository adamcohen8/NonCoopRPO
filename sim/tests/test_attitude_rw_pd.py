import unittest

import numpy as np

from sim.control.attitude import ReactionWheelPDController
from sim.core.models import StateBelief


class TestReactionWheelPDController(unittest.TestCase):
    def test_zero_error_commands_zero_torque(self):
        ctrl = ReactionWheelPDController(
            wheel_axes_body=np.eye(3),
            wheel_torque_limits_nm=np.array([0.05, 0.05, 0.05]),
            kp=np.array([0.2, 0.2, 0.2]),
            kd=np.array([1.0, 1.0, 1.0]),
        )
        state = np.zeros(13)
        state[6:10] = np.array([1.0, 0.0, 0.0, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(13), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)
        self.assertTrue(np.allclose(cmd.torque_body_nm, np.zeros(3), atol=1e-12))
        self.assertTrue(np.allclose(np.array(cmd.mode_flags["wheel_torque_cmd_nm"], dtype=float), np.zeros(3), atol=1e-12))
        self.assertEqual(cmd.mode_flags["mode"], "rw_pd")

    def test_saturates_per_wheel_and_supports_general_mounting(self):
        wheel_axes = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        limits = np.array([0.01, 0.02, 0.015])
        ctrl = ReactionWheelPDController(
            wheel_axes_body=wheel_axes,
            wheel_torque_limits_nm=limits,
            kp=np.array([3.0, 3.0, 3.0]),
            kd=np.array([8.0, 8.0, 8.0]),
        )
        state = np.zeros(13)
        state[6:10] = np.array([0.9238795, 0.0, 0.3826834, 0.0])  # ~45 deg about +Y
        state[10:13] = np.array([0.4, -0.3, 0.2])
        belief = StateBelief(state=state, covariance=np.eye(13), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)
        wheel_cmd = np.array(cmd.mode_flags["wheel_torque_cmd_nm"], dtype=float)
        self.assertEqual(wheel_cmd.shape, (3,))
        self.assertTrue(np.all(np.abs(wheel_cmd) <= limits + 1e-12))
        self.assertEqual(cmd.torque_body_nm.shape, (3,))
        self.assertTrue(np.all(np.isfinite(cmd.torque_body_nm)))


if __name__ == "__main__":
    unittest.main()
