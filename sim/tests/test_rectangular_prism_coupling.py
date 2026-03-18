import unittest

import numpy as np

from presets import build_sim_object_from_presets
from sim.core.models import StateTruth
from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel
from sim.dynamics.spacecraft_geometry import RectangularPrismGeometry


class TestRectangularPrismCoupling(unittest.TestCase):
    def test_projected_area_matches_expected_faces(self):
        g = RectangularPrismGeometry(lx_m=2.0, ly_m=3.0, lz_m=4.0)
        self.assertAlmostEqual(g.projected_area_m2(np.array([1.0, 0.0, 0.0])), 12.0, places=12)
        self.assertAlmostEqual(g.projected_area_m2(np.array([0.0, 1.0, 0.0])), 8.0, places=12)
        self.assertAlmostEqual(g.projected_area_m2(np.array([0.0, 0.0, 1.0])), 6.0, places=12)

        u = np.array([1.0, 1.0, 0.0], dtype=float) / np.sqrt(2.0)
        expected = (12.0 + 8.0) / np.sqrt(2.0)
        self.assertAlmostEqual(g.projected_area_m2(u), expected, places=12)

    def test_face_torque_symmetric_axis_flow_is_zero(self):
        g = RectangularPrismGeometry(lx_m=1.2, ly_m=1.0, lz_m=0.8)
        tau = g.face_torque_sum_body_nm(np.array([1.0, 0.0, 0.0]), pressure_n_m2=2.0)
        self.assertTrue(np.linalg.norm(tau) < 1e-12)

    def test_face_force_follows_incoming_flux_direction(self):
        g = RectangularPrismGeometry(lx_m=2.0, ly_m=3.0, lz_m=4.0)
        incoming = np.array([-1.0, 0.0, 0.0], dtype=float)
        f_total = np.sum(g.face_forces_body_n(incoming, pressure_n_m2=2.0), axis=0)
        self.assertTrue(np.allclose(f_total, np.array([-24.0, 0.0, 0.0])))

    def test_prism_srp_torque_matches_incoming_sun_direction(self):
        cfg = DisturbanceTorqueConfig(
            use_gravity_gradient=False,
            use_magnetic=False,
            use_drag=False,
            use_srp=True,
            use_rectangular_prism_faces=True,
            rectangular_prism_dims_m=(1.0, 2.0, 3.0),
            srp_cr=1.0,
        )
        model = DisturbanceTorqueModel(mu_km3_s2=398600.4418, inertia_kg_m2=np.diag([1.0, 2.0, 3.0]), config=cfg)
        truth = StateTruth(
            position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
            velocity_eci_km_s=np.array([0.0, 7.5, 0.0], dtype=float),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=100.0,
            t_s=0.0,
        )
        sun_dir = np.array([1.0, 1.0, 0.0], dtype=float) / np.sqrt(2.0)
        p_srp = 4.56e-6
        expected = RectangularPrismGeometry(1.0, 2.0, 3.0).face_torque_sum_body_nm(-sun_dir, p_srp)
        tau = model.total_torque_body_nm(
            truth,
            env={"sun_dir_eci": sun_dir, "srp_shadow_model": "none"},
        )
        self.assertTrue(np.allclose(tau, expected))

    def test_prism_mode_requires_disturbance_enabled(self):
        with self.assertRaises(ValueError):
            build_sim_object_from_presets(
                object_id="sat_prism_invalid",
                dt_s=1.0,
                enable_disturbances=False,
                use_rectangular_prism_aero_srp=True,
                rectangular_prism_dims_m=(1.0, 1.0, 1.0),
            )

    def test_prism_mode_wires_dynamics_and_disturbance(self):
        sat = build_sim_object_from_presets(
            object_id="sat_prism_valid",
            dt_s=1.0,
            enable_disturbances=True,
            use_rectangular_prism_aero_srp=True,
            rectangular_prism_dims_m=(1.4, 1.1, 0.9),
        )
        self.assertTrue(sat.dynamics.use_rectangular_prism_for_aero_srp)
        self.assertEqual(tuple(float(v) for v in sat.dynamics.rectangular_prism_dims_m), (1.4, 1.1, 0.9))
        self.assertIsNotNone(sat.dynamics.disturbance_model)
        cfg = sat.dynamics.disturbance_model.config
        self.assertTrue(cfg.use_rectangular_prism_faces)
        self.assertEqual(tuple(float(v) for v in cfg.rectangular_prism_dims_m), (1.4, 1.1, 0.9))


if __name__ == "__main__":
    unittest.main()
