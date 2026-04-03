import unittest
import tempfile
from pathlib import Path

import numpy as np
from sim.dynamics.orbit.accelerations import accel_j2
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.propagator import spherical_harmonics_plugin
from sim.dynamics.orbit.spherical_harmonics import (
    SphericalHarmonicTerm,
    accel_spherical_harmonics_terms,
    configure_spherical_harmonics_env,
    default_hpop_ggm03_coeff_path,
    load_icgem_gfc_terms,
    parse_spherical_harmonic_terms,
)


class TestOrbitSphericalHarmonics(unittest.TestCase):
    @staticmethod
    def _write_minimal_eop(path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "VERSION 1.1",
                    "UPDATED 2026 Apr 03 00:00:00 UTC",
                    "NUM_OBSERVED_POINTS 2",
                    "2024 03 31 60400 0.0 0.0 0.0 0 0 0 0 0 37",
                    "2024 04 01 60401 0.0 0.0 0.0 0 0 0 0 0 37",
                    "",
                ]
            ),
            encoding="utf-8",
        )

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

    def test_normalized_c20_matches_equivalent_j2_perturbation(self):
        r = np.array([7000.0, 100.0, 200.0], dtype=float)
        terms = [SphericalHarmonicTerm(n=2, m=0, c_nm=-4.841693259705e-04, s_nm=0.0, normalized=True)]
        with tempfile.TemporaryDirectory() as td:
            eop_path = Path(td) / "EOP-All.txt"
            self._write_minimal_eop(eop_path)
            a = accel_spherical_harmonics_terms(
                r_eci_km=r,
                t_s=0.0,
                terms=terms,
                mu_km3_s2=EARTH_MU_KM3_S2,
                re_km=6378.1363,
                jd_utc_start=2460400.5,
                frame_model="hpop_like",
                eop_path=str(eop_path),
            )
        a_j2 = accel_j2(r, EARTH_MU_KM3_S2, j2=0.0010826355254902923, re_km=6378.1363)
        self.assertLess(float(np.linalg.norm(a - a_j2)), 1e-10)

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

    def test_configure_env_loads_hpop_terms_and_radius(self):
        orbit_cfg = {
            "spherical_harmonics": {
                "enabled": True,
                "degree": 2,
                "order": 0,
                "source": "hpop_ggm03",
            }
        }
        env = configure_spherical_harmonics_env({}, orbit_cfg)
        self.assertIn("spherical_harmonics_terms", env)
        self.assertEqual([(t.n, t.m) for t in env["spherical_harmonics_terms"]], [(2, 0)])
        term = env["spherical_harmonics_terms"][0]
        self.assertEqual((term.n, term.m), (2, 0))
        self.assertTrue(term.normalized)
        self.assertEqual(Path(env["spherical_harmonics_source"]), default_hpop_ggm03_coeff_path().resolve())
        self.assertAlmostEqual(float(env["spherical_harmonics_reference_radius_km"]), 6378.1363)


if __name__ == "__main__":
    unittest.main()
