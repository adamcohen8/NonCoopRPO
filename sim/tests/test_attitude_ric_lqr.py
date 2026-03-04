import unittest

import numpy as np

from sim.control.attitude import RICFrameLQRController, SmallAngleLQRController
from sim.core.models import StateBelief
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import dcm_to_quaternion_bn


class TestRICFrameLQRController(unittest.TestCase):
    def test_zero_error_and_correct_frame_rate_produce_small_torque(self):
        inertia = np.diag([110.0, 95.0, 80.0])
        lqr = SmallAngleLQRController(
            inertia_kg_m2=inertia,
            wheel_axes_body=np.eye(3),
            wheel_torque_limits_nm=np.array([0.1, 0.1, 0.1]),
            q_weights=np.array([30.0, 30.0, 30.0, 3.0, 3.0, 3.0]),
            r_weights=np.array([1.0, 1.0, 1.0]),
            design_dt_s=1.0,
        )
        ctrl = RICFrameLQRController(lqr=lqr)

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

        self.assertEqual(cmd.mode_flags["mode"], "lqr_ric")
        self.assertTrue(np.linalg.norm(cmd.torque_body_nm) < 1e-6)


if __name__ == "__main__":
    unittest.main()
