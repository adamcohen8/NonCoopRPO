import unittest

import numpy as np

from sim.control.orbit import HCWCurvInputRectOutputController, HCWLQRController
from sim.core.models import StateBelief


class TestHCWCurvVariant(unittest.TestCase):
    def test_variant_matches_base_accel(self):
        base = HCWLQRController(
            mean_motion_rad_s=0.0011,
            max_accel_km_s2=5e-5,
            design_dt_s=10.0,
            q_weights=np.array([8.66, 8.66, 8.66, 1.33, 1.33, 1.33]) * 1e3,
            r_weights=np.ones(3) * 1.94e13,
        )
        var = HCWCurvInputRectOutputController(base_lqr=base)

        state = np.zeros(12)
        state[0:6] = np.array([0.8, -0.3, 0.2, 5e-4, -2e-4, 1e-4])
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)

        c_base = base.act(belief, t_s=0.0, budget_ms=1.0)
        c_var = var.act(belief, t_s=0.0, budget_ms=1.0)
        self.assertTrue(np.allclose(c_base.thrust_eci_km_s2, c_var.thrust_eci_km_s2, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
