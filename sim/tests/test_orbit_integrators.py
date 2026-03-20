import unittest

import numpy as np

from sim.config import scenario_config_from_dict
from sim.dynamics.orbit.integrators import integrate_adaptive
from sim.master_simulator import _build_orbit_propagator


class TestOrbitIntegrators(unittest.TestCase):
    def test_rkf78_adaptive_step_matches_requested_outer_dt(self):
        eval_times: list[float] = []

        def deriv(t_s: float, x: np.ndarray) -> np.ndarray:
            eval_times.append(float(t_s))
            return x

        x0 = np.array([1.0], dtype=float)
        x1 = integrate_adaptive(
            deriv_fn=deriv,
            t_s=0.0,
            x=x0,
            dt_s=1.0,
            atol=1e-12,
            rtol=1e-10,
            method="rkf78",
        )

        self.assertTrue(np.all(x1 > x0))
        self.assertAlmostEqual(float(x1[0]), float(np.e), places=8)
        self.assertTrue(all(0.0 <= t <= 1.0 for t in eval_times))

    def test_master_simulator_orbit_propagator_can_select_rkf78(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "rkf78_builder",
                "simulator": {
                    "duration_s": 1.0,
                    "dt_s": 1.0,
                    "dynamics": {
                        "orbit": {
                            "integrator": "rkf78",
                            "adaptive_atol": 1e-12,
                            "adaptive_rtol": 1e-10,
                        }
                    },
                },
            }
        )

        prop = _build_orbit_propagator(cfg)
        self.assertEqual(prop.integrator, "rkf78")
        self.assertAlmostEqual(prop.adaptive_atol, 1e-12)
        self.assertAlmostEqual(prop.adaptive_rtol, 1e-10)


if __name__ == "__main__":
    unittest.main()
