import unittest

import numpy as np

from sim.dynamics.orbit.eclipse import srp_shadow_factor
from sim.dynamics.orbit.epoch import AU_KM


class OrbitEclipseTests(unittest.TestCase):
    def test_day_side_full_light(self):
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0])}
        r = np.array([6878.137, 0.0, 0.0])  # day side
        f = srp_shadow_factor(r_sc_eci_km=r, t_s=0.0, env=env)
        self.assertGreater(f, 0.999)

    def test_night_side_umbra(self):
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0])}
        r = np.array([-6878.137, 0.0, 0.0])  # anti-sun side
        f = srp_shadow_factor(r_sc_eci_km=r, t_s=0.0, env=env)
        self.assertLess(f, 1e-6)

    def test_penumbra_partial(self):
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0])}
        # Around Earth apparent limb from LEO, expect transition zone.
        delta = np.deg2rad(68.0)
        rmag = 6878.137
        r = rmag * np.array([-np.cos(delta), np.sin(delta), 0.0])
        f = srp_shadow_factor(r_sc_eci_km=r, t_s=0.0, env=env)
        self.assertGreater(f, 0.0)
        self.assertLess(f, 1.0)

    def test_shadow_disabled(self):
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0]), "srp_shadow_model": "none"}
        r = np.array([-6878.137, 0.0, 0.0])
        f = srp_shadow_factor(r_sc_eci_km=r, t_s=0.0, env=env)
        self.assertAlmostEqual(f, 1.0)

    def test_cylindrical_shadow(self):
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0]), "srp_shadow_model": "cylindrical"}
        r1 = np.array([-6878.137, 0.0, 0.0])
        r2 = np.array([0.0, 6878.137, 0.0])
        self.assertAlmostEqual(srp_shadow_factor(r_sc_eci_km=r1, t_s=0.0, env=env), 0.0)
        self.assertAlmostEqual(srp_shadow_factor(r_sc_eci_km=r2, t_s=0.0, env=env), 1.0)


if __name__ == "__main__":
    unittest.main()
