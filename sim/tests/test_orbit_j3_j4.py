import unittest

import numpy as np

from sim.dynamics.orbit.accelerations import accel_j2, accel_j3, accel_j4
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.propagator import j3_plugin, j4_plugin


class TestOrbitJ3J4(unittest.TestCase):
    def test_zero_radius_returns_zero(self):
        r = np.zeros(3)
        self.assertTrue(np.allclose(accel_j3(r, EARTH_MU_KM3_S2), np.zeros(3)))
        self.assertTrue(np.allclose(accel_j4(r, EARTH_MU_KM3_S2), np.zeros(3)))

    def test_j3_j4_symmetry_wrt_z_sign(self):
        r = np.array([7000.0, -1200.0, 900.0], dtype=float)
        r_flip = np.array([r[0], r[1], -r[2]], dtype=float)

        a3 = accel_j3(r, EARTH_MU_KM3_S2)
        a3f = accel_j3(r_flip, EARTH_MU_KM3_S2)
        # J3: x/y odd in z, z even in z.
        self.assertAlmostEqual(a3[0], -a3f[0], places=14)
        self.assertAlmostEqual(a3[1], -a3f[1], places=14)
        self.assertAlmostEqual(a3[2], a3f[2], places=14)

        a4 = accel_j4(r, EARTH_MU_KM3_S2)
        a4f = accel_j4(r_flip, EARTH_MU_KM3_S2)
        # J4: x/y even in z, z odd in z.
        self.assertAlmostEqual(a4[0], a4f[0], places=14)
        self.assertAlmostEqual(a4[1], a4f[1], places=14)
        self.assertAlmostEqual(a4[2], -a4f[2], places=14)

    def test_plugins_match_accel_functions(self):
        x = np.array([7000.0, 0.0, 10.0, 0.0, 7.5, 0.0], dtype=float)

        class _Ctx:
            mu_km3_s2 = EARTH_MU_KM3_S2

        a3p = j3_plugin(0.0, x, env={}, ctx=_Ctx())
        a4p = j4_plugin(0.0, x, env={}, ctx=_Ctx())
        self.assertTrue(np.allclose(a3p, accel_j3(x[:3], EARTH_MU_KM3_S2)))
        self.assertTrue(np.allclose(a4p, accel_j4(x[:3], EARTH_MU_KM3_S2)))

    def test_j3_j4_magnitudes_below_j2_for_leo_scale(self):
        r = np.array([7000.0, 100.0, 50.0], dtype=float)
        a2 = np.linalg.norm(accel_j2(r, EARTH_MU_KM3_S2))
        a3 = np.linalg.norm(accel_j3(r, EARTH_MU_KM3_S2))
        a4 = np.linalg.norm(accel_j4(r, EARTH_MU_KM3_S2))
        self.assertLess(a3, a2)
        self.assertLess(a4, a2)


if __name__ == "__main__":
    unittest.main()
