import unittest
from datetime import datetime, timezone

import numpy as np

from sim.dynamics.orbit.accelerations import accel_drag
from sim.dynamics.orbit.atmosphere import density_exponential, density_from_model, density_ussa1976
from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S


class TestOrbitAtmosphereModels(unittest.TestCase):
    def test_ussa1976_density_reasonable_at_sea_level(self):
        r = np.array([6378.137, 0.0, 0.0], dtype=float)
        rho = density_ussa1976(r, t_s=0.0)
        self.assertGreater(rho, 1.0)
        self.assertLess(rho, 1.4)

    def test_density_models_selectable(self):
        r = np.array([6778.137, 0.0, 0.0], dtype=float)
        rho_exp = density_from_model("exponential", r, 0.0, env={})
        rho_ussa = density_from_model("ussa1976", r, 0.0, env={})
        self.assertGreaterEqual(rho_exp, 0.0)
        self.assertGreaterEqual(rho_ussa, 0.0)

    def test_density_nrlmsise00_callable_hook(self):
        calls = []

        def _fn(alt_km, lat_deg, lon_deg, dt_utc, env):
            calls.append((alt_km, lat_deg, lon_deg, dt_utc))
            return 1.23e-11

        env = {
            "nrlmsise00_density_callable": _fn,
            "atmo_epoch_utc": datetime(2024, 1, 1, tzinfo=timezone.utc),
        }
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        rho = density_from_model("nrlmsise00", r, t_s=60.0, env=env)
        self.assertAlmostEqual(rho, 1.23e-11)
        self.assertEqual(len(calls), 1)

    def test_density_jb2008_callable_hook(self):
        calls = []

        def _fn(alt_km, lat_deg, lon_deg, dt_utc, env):
            calls.append((alt_km, lat_deg, lon_deg, dt_utc))
            return 4.56e-12

        env = {
            "jb2008_density_callable": _fn,
            "atmo_epoch_utc": datetime(2024, 1, 1, tzinfo=timezone.utc),
        }
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        rho = density_from_model("jb2008", r, t_s=60.0, env=env)
        self.assertAlmostEqual(rho, 4.56e-12)
        self.assertEqual(len(calls), 1)

    def test_density_jb2008_without_backend_raises(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        with self.assertRaises(RuntimeError):
            _ = density_from_model("jb2008", r, t_s=60.0, env={})

    def test_drag_uses_rotating_atmosphere_relative_velocity(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        omega = np.array([0.0, 0.0, EARTH_ROT_RATE_RAD_S], dtype=float)
        v_atm = np.cross(omega, r)
        a = accel_drag(
            r_eci_km=r,
            v_eci_km_s=v_atm,  # matches corotating atmosphere speed at position
            t_s=0.0,
            mass_kg=100.0,
            area_m2=1.0,
            cd=2.2,
            env={"density_kg_m3": density_exponential(r, 0.0)},
        )
        self.assertTrue(np.linalg.norm(a) < 1e-14)


if __name__ == "__main__":
    unittest.main()
