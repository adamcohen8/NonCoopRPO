import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

from sim.dynamics.orbit.accelerations import accel_drag, accel_srp
from sim.dynamics.orbit.atmosphere import density_exponential, density_from_model, density_ussa1976
from sim.dynamics.orbit.eclipse import resolve_srp_geometry, srp_shadow_factor
from sim.dynamics.orbit.epoch import AU_KM
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

    def test_density_exponential_remains_positive_above_180_km(self):
        r_200km = np.array([6378.137 + 200.0, 0.0, 0.0], dtype=float)
        rho = density_exponential(r_200km, t_s=0.0)
        self.assertGreater(rho, 0.0)

    def test_density_exponential_skips_ecef_conversion(self):
        r = np.array([6378.137 + 200.0, 0.0, 0.0], dtype=float)
        with patch("sim.dynamics.orbit.atmosphere.eci_to_ecef", side_effect=AssertionError("should not be called")):
            rho = density_exponential(r, t_s=123.0)
            rho_from_model = density_from_model("exponential", r, 123.0, env={"geodetic_model": "wgs84"})
        self.assertGreater(rho, 0.0)
        self.assertAlmostEqual(rho_from_model, rho)

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

    def test_density_jb2008_builtin_backend_returns_finite_density(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        rho = density_from_model("jb2008", r, t_s=60.0, env={"jd_utc_start": 2460400.5, "geodetic_model": "wgs84"})
        self.assertTrue(np.isfinite(rho))
        self.assertGreaterEqual(rho, 0.0)

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

    def test_srp_scales_with_sun_spacecraft_distance(self):
        r = np.array([6878.137, 0.0, 0.0], dtype=float)
        a_1au = accel_srp(
            r_eci_km=r,
            mass_kg=100.0,
            area_m2=1.0,
            cr=1.0,
            t_s=0.0,
            env={"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0]), "srp_shadow_model": "none"},
        )
        a_half_au = accel_srp(
            r_eci_km=r,
            mass_kg=100.0,
            area_m2=1.0,
            cr=1.0,
            t_s=0.0,
            env={"sun_pos_eci_km": np.array([0.5 * AU_KM, 0.0, 0.0]), "srp_shadow_model": "none"},
        )
        ratio = float(np.linalg.norm(a_half_au) / max(np.linalg.norm(a_1au), 1e-18))
        self.assertGreater(ratio, 3.95)
        self.assertLess(ratio, 4.05)

    def test_srp_accel_accepts_precomputed_geometry_bundle(self):
        r = np.array([6878.137, 0.0, 0.0], dtype=float)
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0]), "srp_shadow_model": "conical"}
        geometry = resolve_srp_geometry(r, 0.0, env)
        shadow = srp_shadow_factor(r, 0.0, env, srp_geometry=geometry)

        a_direct = accel_srp(
            r_eci_km=r,
            mass_kg=100.0,
            area_m2=1.0,
            cr=1.0,
            t_s=0.0,
            env=env,
        )
        a_cached = accel_srp(
            r_eci_km=r,
            mass_kg=100.0,
            area_m2=1.0,
            cr=1.0,
            t_s=0.0,
            env={
                "srp_geometry": geometry,
                "srp_sun_dir_eci": geometry["sun_dir_sc_eci"],
                "srp_distance_scale": geometry["distance_scale"],
                "srp_shadow_factor": shadow,
                "srp_area_m2": 1.0,
                "srp_shadow_model": "conical",
            },
        )
        np.testing.assert_allclose(a_cached, a_direct, rtol=0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
