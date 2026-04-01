import unittest

import numpy as np

from sim.control.orbit.hcw_transfer import (
    optimize_hcw_evasion_burn_direction,
    solve_hcw_position_rendezvous,
)


class TestHCWTransfer(unittest.TestCase):
    def test_position_rendezvous_solution_zeros_final_position(self):
        x0 = np.array([1.5, -3.0, 0.7, 0.0003, -0.0005, 0.0001], dtype=float)
        solution = solve_hcw_position_rendezvous(
            initial_rel_state_ric=x0,
            target_delta_v_ric_km_s=np.array([0.0, 0.001, -0.0005], dtype=float),
            mean_motion_rad_s=0.001078007612872506,
            transfer_time_s=1700.0,
        )
        self.assertLess(np.linalg.norm(solution.rendezvous_state_ric[:3]), 1e-9)

    def test_optimization_returns_unit_direction_and_not_worse_than_baseline(self):
        x0 = np.array([1.2, -4.0, 0.8, 0.0002, -0.0008, 0.00015], dtype=float)
        baseline = solve_hcw_position_rendezvous(
            initial_rel_state_ric=x0,
            target_delta_v_ric_km_s=np.zeros(3, dtype=float),
            mean_motion_rad_s=0.001078007612872506,
            transfer_time_s=1800.0,
        )
        result = optimize_hcw_evasion_burn_direction(
            initial_rel_state_ric=x0,
            target_delta_v_mag_km_s=0.0025,
            mean_motion_rad_s=0.001078007612872506,
            transfer_time_s=1800.0,
            iterations=30,
            seed=4,
        )
        self.assertAlmostEqual(float(np.linalg.norm(result.best_direction_ric)), 1.0, places=10)
        self.assertGreaterEqual(result.best_solution.required_delta_v_mag_km_s, baseline.required_delta_v_mag_km_s)


if __name__ == "__main__":
    unittest.main()
