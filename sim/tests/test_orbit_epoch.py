from datetime import datetime, timezone
import unittest

import numpy as np

from sim.dynamics.orbit.epoch import (
    datetime_to_julian_date,
    julian_date_to_datetime,
    moon_position_eci_km_enhanced,
    resolve_time_dependent_env,
    resolve_sun_moon_positions,
    sun_position_eci_km_enhanced,
)


class OrbitEpochTests(unittest.TestCase):
    def test_julian_round_trip(self):
        dt = datetime(2026, 3, 11, 12, 0, 0, tzinfo=timezone.utc)
        jd = datetime_to_julian_date(dt)
        dt_back = julian_date_to_datetime(jd)
        self.assertLess(abs((dt_back - dt).total_seconds()), 1e-3)

    def test_time_dependent_env_adds_dynamic_fields(self):
        env = {"jd_utc_start": 2451545.0, "ephemeris_mode": "analytic_simple"}
        out = resolve_time_dependent_env(env, 120.0)
        self.assertIn("jd_utc", out)
        self.assertIn("sun_pos_eci_km", out)
        self.assertIn("moon_pos_eci_km", out)
        self.assertIn("sun_dir_eci", out)
        self.assertAlmostEqual(float(np.linalg.norm(out["sun_dir_eci"])), 1.0, places=8)

    def test_enhanced_ephemeris_vectors_finite(self):
        jd = 2461110.5
        s = sun_position_eci_km_enhanced(jd)
        m = moon_position_eci_km_enhanced(jd)
        self.assertEqual(s.shape, (3,))
        self.assertEqual(m.shape, (3,))
        self.assertTrue(np.all(np.isfinite(s)))
        self.assertTrue(np.all(np.isfinite(m)))
        self.assertGreater(float(np.linalg.norm(s)), 1.0e8)
        self.assertGreater(float(np.linalg.norm(m)), 1.0e5)

    def test_external_ephemeris_callable(self):
        def eph_cb(jd_utc: float, env: dict):
            return {
                "sun_pos_eci_km": np.array([1.0, 2.0, 3.0]),
                "moon_pos_eci_km": np.array([4.0, 5.0, 6.0]),
            }

        env = {
            "jd_utc_start": 2451545.0,
            "ephemeris_mode": "external",
            "ephemeris_callable": eph_cb,
        }
        sun, moon = resolve_sun_moon_positions(env, t_s=10.0)
        self.assertTrue(np.allclose(sun, np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.allclose(moon, np.array([4.0, 5.0, 6.0])))

    def test_spice_mode_with_callable(self):
        def spice_cb(jd_utc: float, env: dict):
            return {
                "sun_pos_eci_km": np.array([10.0, 20.0, 30.0]),
                "moon_pos_eci_km": np.array([40.0, 50.0, 60.0]),
            }

        env = {
            "jd_utc_start": 2451545.0,
            "ephemeris_mode": "spice",
            "spice_ephemeris_callable": spice_cb,
        }
        sun, moon = resolve_sun_moon_positions(env, t_s=1.0)
        self.assertTrue(np.allclose(sun, np.array([10.0, 20.0, 30.0])))
        self.assertTrue(np.allclose(moon, np.array([40.0, 50.0, 60.0])))

    def test_explicit_ephemeris_history_interpolates_moon(self):
        env = {
            "sun_pos_eci_km": np.array([1.0, 2.0, 3.0]),
            "moon_ephemeris_time_s": [0.0, 10.0],
            "moon_ephemeris_eci_km": [[4.0, 5.0, 6.0], [14.0, 15.0, 16.0]],
        }
        sun, moon = resolve_sun_moon_positions(env, t_s=5.0)
        self.assertTrue(np.allclose(sun, np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.allclose(moon, np.array([9.0, 10.0, 11.0])))


if __name__ == "__main__":
    unittest.main()
