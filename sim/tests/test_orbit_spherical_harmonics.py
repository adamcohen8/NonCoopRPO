import unittest
import tempfile
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.propagator import spherical_harmonics_plugin
from sim.dynamics.orbit.spherical_harmonics import (
    SphericalHarmonicTerm,
    accel_spherical_harmonics_terms,
    load_icgem_gfc_terms,
    parse_spherical_harmonic_terms,
)


class TestOrbitSphericalHarmonics(unittest.TestCase):
    def test_parse_terms(self):
        raw = [
            {"n": 3, "m": 3, "c_nm": 1e-6, "s_nm": -2e-6},
            {"n": 4, "m": 2, "c": 5e-7, "s": 3e-7},
        ]
        terms = parse_spherical_harmonic_terms(raw)
        self.assertEqual(len(terms), 2)
        self.assertEqual((terms[0].n, terms[0].m), (3, 3))
        self.assertEqual((terms[1].n, terms[1].m), (4, 2))

    def test_sectoral_and_tesseral_nonzero(self):
        r = np.array([7000.0, 200.0, 300.0], dtype=float)
        terms = [
            SphericalHarmonicTerm(n=3, m=3, c_nm=1e-6, s_nm=2e-6),  # sectoral
            SphericalHarmonicTerm(n=4, m=2, c_nm=-2e-6, s_nm=1e-6),  # tesseral
        ]
        a = accel_spherical_harmonics_terms(r_eci_km=r, t_s=0.0, terms=terms, mu_km3_s2=EARTH_MU_KM3_S2)
        self.assertGreater(float(np.linalg.norm(a)), 0.0)

    def test_plugin_reads_env_m_n_terms(self):
        x = np.array([7000.0, 10.0, 20.0, 0.0, 7.5, 0.0], dtype=float)
        env = {
            "spherical_harmonics_terms": [
                {"n": 3, "m": 3, "c_nm": 1e-6, "s_nm": 0.0},
                {"n": 5, "m": 2, "c_nm": -1e-6, "s_nm": 1e-6},
            ],
            "spherical_harmonics_fd_step_km": 1e-3,
        }

        class _Ctx:
            mu_km3_s2 = EARTH_MU_KM3_S2

        a = spherical_harmonics_plugin(0.0, x, env=env, ctx=_Ctx())
        self.assertGreater(float(np.linalg.norm(a)), 0.0)

    def test_plugin_respects_epoch_for_tesseral_terms(self):
        x = np.array([7000.0, 0.0, 100.0, 0.0, 7.5, 0.0], dtype=float)
        env0 = {
            "spherical_harmonics_terms": [{"n": 2, "m": 2, "c_nm": 1e-6, "s_nm": 0.0}],
            "jd_utc_start": 2451545.0,
        }
        env1 = {
            "spherical_harmonics_terms": [{"n": 2, "m": 2, "c_nm": 1e-6, "s_nm": 0.0}],
            "jd_utc_start": 2451545.25,
        }

        class _Ctx:
            mu_km3_s2 = EARTH_MU_KM3_S2

        a0 = spherical_harmonics_plugin(0.0, x, env=env0, ctx=_Ctx())
        a1 = spherical_harmonics_plugin(0.0, x, env=env1, ctx=_Ctx())
        self.assertFalse(np.allclose(a0, a1))

    def test_load_icgem_gfc_terms_normalized_flag(self):
        gfc_txt = "\n".join(
            [
                "modelname TEST",
                "norm fully_normalized",
                "gfc 2 0 -4.84165371736e-04 0.0 0.0 0.0",
                "gfc 2 2 2.43914352398e-06 -1.40016683654e-06 0.0 0.0",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.gfc"
            p.write_text(gfc_txt, encoding="utf-8")
            terms = load_icgem_gfc_terms(p, max_degree=8, max_order=8)
        self.assertEqual(len(terms), 2)
        self.assertTrue(all(t.normalized for t in terms))


if __name__ == "__main__":
    unittest.main()
