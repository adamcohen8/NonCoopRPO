import unittest

import numpy as np

from presets.rockets import RocketStackPreset, RocketStagePreset
from sim.rocket import HoldAttitudeGuidance, RocketAscentSimulator, RocketSimConfig, RocketVehicleConfig


class TestRocketAscentEngine(unittest.TestCase):
    def _tiny_stack(self) -> RocketStackPreset:
        s1 = RocketStagePreset(
            name="s1",
            dry_mass_kg=100.0,
            propellant_mass_kg=200.0,
            max_thrust_n=2.0e5,
            isp_s=280.0,
            burn_time_s=20.0,
            diameter_m=1.5,
            length_m=8.0,
        )
        s2 = RocketStagePreset(
            name="s2",
            dry_mass_kg=40.0,
            propellant_mass_kg=80.0,
            max_thrust_n=7.0e4,
            isp_s=310.0,
            burn_time_s=30.0,
            diameter_m=1.2,
            length_m=5.0,
        )
        return RocketStackPreset(name="tiny", stages=(s1, s2))

    def test_mass_decreases_and_stage_progresses(self):
        sim_cfg = RocketSimConfig(
            dt_s=0.5,
            max_time_s=200.0,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=20.0)
        sim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=1.0))
        out = sim.run()

        self.assertLess(out.mass_kg[-1], out.mass_kg[0])
        self.assertGreaterEqual(int(np.max(out.active_stage_index)), 1)

    def test_returns_result_arrays_consistent(self):
        sim_cfg = RocketSimConfig(
            dt_s=1.0,
            max_time_s=10.0,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=0.5))
        out = sim.run()
        n = out.time_s.size
        self.assertEqual(out.position_eci_km.shape, (n, 3))
        self.assertEqual(out.velocity_eci_km_s.shape, (n, 3))
        self.assertEqual(out.attitude_quat_bn.shape, (n, 4))
        self.assertEqual(out.angular_rate_body_rad_s.shape, (n, 3))
        self.assertEqual(out.mass_kg.shape, (n,))

    def test_stagewise_aero_geometry_updates_with_stage(self):
        sim_cfg = RocketSimConfig(
            dt_s=1.0,
            max_time_s=1.0,
            enable_drag=True,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            area_ref_m2=None,
            use_stagewise_aero_geometry=True,
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=0.0))

        cfg0 = sim._resolve_aero_config_for_stage(0)
        cfg1 = sim._resolve_aero_config_for_stage(1)
        a0_expected = np.pi * 0.25 * (1.5**2)
        a1_expected = np.pi * 0.25 * (1.2**2)
        self.assertAlmostEqual(cfg0.reference_area_m2, a0_expected, places=10)
        self.assertAlmostEqual(cfg1.reference_area_m2, a1_expected, places=10)
        self.assertAlmostEqual(cfg0.reference_length_m, 8.0, places=10)
        self.assertAlmostEqual(cfg1.reference_length_m, 5.0, places=10)

    def test_area_override_takes_priority_over_stage_geometry(self):
        sim_cfg = RocketSimConfig(
            dt_s=1.0,
            max_time_s=1.0,
            enable_drag=True,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            area_ref_m2=4.2,
            use_stagewise_aero_geometry=True,
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=0.0))

        cfg0 = sim._resolve_aero_config_for_stage(0)
        cfg1 = sim._resolve_aero_config_for_stage(1)
        self.assertAlmostEqual(cfg0.reference_area_m2, 4.2, places=12)
        self.assertAlmostEqual(cfg1.reference_area_m2, 4.2, places=12)


if __name__ == "__main__":
    unittest.main()
