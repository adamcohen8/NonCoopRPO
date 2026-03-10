import unittest

import numpy as np

from sim.control.attitude import RICFramePIDController, ReactionWheelPIDController
from sim.core.models import StateBelief
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import dcm_to_quaternion_bn


class TestRICFramePIDController(unittest.TestCase):
    def test_zero_error_and_correct_frame_rate_produce_small_torque(self):
        pid = ReactionWheelPIDController(
            wheel_axes_body=np.eye(3),
            wheel_torque_limits_nm=np.array([0.1, 0.1, 0.1]),
            kp=np.array([0.2, 0.2, 0.2]),
            kd=np.array([1.0, 1.0, 1.0]),
            ki=np.array([0.03, 0.03, 0.03]),
        )
        ctrl = RICFramePIDController(pid=pid)

        r = np.array([7000.0, 0.0, 0.0])
        v = np.array([0.0, 7.5, 0.0])
        c_ir = ric_dcm_ir_from_rv(r, v)
        q_bn = dcm_to_quaternion_bn(c_ir.T)  # body aligned with RIC
        h = np.cross(r, v)
        w_eci = h / (np.linalg.norm(r) ** 2)
        w_body = c_ir.T @ w_eci

        x = np.hstack((r, v, q_bn, w_body))
        belief = StateBelief(state=x, covariance=np.eye(13), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)

        self.assertEqual(cmd.mode_flags["mode"], "pid_ric")
        self.assertTrue(np.linalg.norm(cmd.torque_body_nm) < 1e-6)


if __name__ == "__main__":
    unittest.main()
