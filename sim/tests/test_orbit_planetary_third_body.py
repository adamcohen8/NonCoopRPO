import unittest

import numpy as np

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import third_body_planets_plugin


class PlanetaryThirdBodyTests(unittest.TestCase):
    def test_planetary_plugin_with_explicit_positions(self):
        x = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=float)
        env = {
            "third_body_planets": ["venus", "jupiter"],
            "venus_pos_eci_km": np.array([1.0e8, 2.0e7, 0.0]),
            "jupiter_pos_eci_km": np.array([-6.0e8, 1.0e8, 5.0e7]),
        }
        ctx = OrbitContext(mu_km3_s2=398600.4418, mass_kg=100.0)
        a = third_body_planets_plugin(0.0, x, env, ctx)
        self.assertEqual(a.shape, (3,))
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertGreater(float(np.linalg.norm(a)), 0.0)

    def test_planetary_plugin_spice_mode_with_callable(self):
        def body_cb(name: str, jd_utc: float, env: dict):
            lut = {
                "mars": np.array([2.0e8, 0.0, 0.0]),
                "saturn": np.array([1.2e9, 2.0e8, 0.0]),
            }
            return lut[name]

        x = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=float)
        env = {
            "third_body_planets": ["mars", "saturn"],
            "ephemeris_mode": "spice",
            "jd_utc_start": 2451545.0,
            "spice_body_ephemeris_callable": body_cb,
        }
        ctx = OrbitContext(mu_km3_s2=398600.4418, mass_kg=100.0)
        a = third_body_planets_plugin(100.0, x, env, ctx)
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertGreater(float(np.linalg.norm(a)), 0.0)

    def test_planetary_plugin_all_keyword(self):
        x = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=float)
        env = {
            "third_body_planets": "all",
            "ephemeris_mode": "spice",
            "jd_utc_start": 2451545.0,
            "spice_body_ephemeris_callable": lambda name, jd, e: np.array([1.0e8, 0.0, 0.0]),
        }
        ctx = OrbitContext(mu_km3_s2=398600.4418, mass_kg=100.0)
        a = third_body_planets_plugin(0.0, x, env, ctx)
        self.assertGreater(float(np.linalg.norm(a)), 0.0)


if __name__ == "__main__":
    unittest.main()
