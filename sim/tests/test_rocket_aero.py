import unittest

import numpy as np

from sim.rocket.aero import RocketAeroConfig, compute_aero_loads, compute_aero_state


class TestRocketAero(unittest.TestCase):
    def test_state_zero_relative_speed(self):
        s = compute_aero_state(
            rho_kg_m3=1.225,
            pressure_pa=101325.0,
            temperature_k=288.15,
            sound_speed_m_s=340.0,
            v_rel_body_m_s=np.zeros(3),
            alpha_limit_deg=20.0,
            beta_limit_deg=20.0,
        )
        self.assertAlmostEqual(s.dynamic_pressure_pa, 0.0, places=12)
        self.assertAlmostEqual(s.speed_m_s, 0.0, places=12)
        self.assertAlmostEqual(s.mach, 0.0, places=12)
        self.assertAlmostEqual(s.alpha_rad, 0.0, places=12)
        self.assertAlmostEqual(s.beta_rad, 0.0, places=12)

    def test_state_angle_limits_apply(self):
        s = compute_aero_state(
            rho_kg_m3=1.0,
            pressure_pa=90000.0,
            temperature_k=260.0,
            sound_speed_m_s=320.0,
            v_rel_body_m_s=np.array([10.0, 9.0, 20.0]),
            alpha_limit_deg=5.0,
            beta_limit_deg=3.0,
        )
        self.assertAlmostEqual(s.alpha_rad, np.deg2rad(5.0), places=12)
        self.assertAlmostEqual(s.beta_rad, np.deg2rad(3.0), places=12)
        self.assertGreater(s.dynamic_pressure_pa, 0.0)
        self.assertGreater(s.mach, 0.0)

    def test_loads_zero_when_disabled_or_no_q(self):
        cfg = RocketAeroConfig(enabled=False)
        s = compute_aero_state(
            rho_kg_m3=1.2,
            pressure_pa=101325.0,
            temperature_k=288.0,
            sound_speed_m_s=340.0,
            v_rel_body_m_s=np.array([100.0, 0.0, 0.0]),
            alpha_limit_deg=20.0,
            beta_limit_deg=20.0,
        )
        loads = compute_aero_loads(np.array([100.0, 0.0, 0.0]), s, cfg)
        self.assertTrue(np.allclose(loads.force_body_n, np.zeros(3)))
        self.assertTrue(np.allclose(loads.moment_body_nm, np.zeros(3)))

    def test_drag_and_cp_offset_moment(self):
        cfg = RocketAeroConfig(
            enabled=True,
            reference_area_m2=2.0,
            reference_length_m=1.0,
            cp_offset_body_m=np.array([0.0, 1.0, 0.0]),
            cd_base=0.5,
            cd_alpha2=0.0,
            cd_supersonic=0.5,
            transonic_peak_cd=0.0,
            cl_alpha_per_rad=0.0,
            cy_beta_per_rad=0.0,
            cm_alpha_per_rad=0.0,
            cn_beta_per_rad=0.0,
            cl_roll_per_rad=0.0,
        )
        s = compute_aero_state(
            rho_kg_m3=1.2,
            pressure_pa=101325.0,
            temperature_k=288.0,
            sound_speed_m_s=340.0,
            v_rel_body_m_s=np.array([100.0, 0.0, 0.0]),
            alpha_limit_deg=20.0,
            beta_limit_deg=20.0,
        )
        loads = compute_aero_loads(np.array([100.0, 0.0, 0.0]), s, cfg)

        q = 0.5 * 1.2 * 100.0 * 100.0
        fx_expected = -q * cfg.reference_area_m2 * cfg.cd_base
        self.assertAlmostEqual(loads.force_body_n[0], fx_expected, places=6)
        self.assertAlmostEqual(loads.force_body_n[1], 0.0, places=10)
        self.assertAlmostEqual(loads.force_body_n[2], 0.0, places=10)
        self.assertAlmostEqual(loads.moment_body_nm[0], 0.0, places=8)
        self.assertAlmostEqual(loads.moment_body_nm[1], 0.0, places=8)
        self.assertAlmostEqual(loads.moment_body_nm[2], -fx_expected, places=6)


if __name__ == "__main__":
    unittest.main()
