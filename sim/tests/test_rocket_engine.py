import unittest
import warnings

import numpy as np

from sim.presets.rockets import RocketStackPreset, RocketStagePreset
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.rocket import GuidanceCommand, HoldAttitudeGuidance, RocketAscentSimulator, RocketSimConfig, RocketVehicleConfig


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
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
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
        self.assertEqual(out.latitude_deg.shape, (n,))
        self.assertEqual(out.longitude_deg.shape, (n,))
        self.assertEqual(out.wind_body_m_s.shape, (n, 3))
        self.assertEqual(out.tvc_gimbal_deg.shape, (n,))

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

    def test_wgs84_launch_outputs_geodetic_lat_lon(self):
        sim_cfg = RocketSimConfig(
            dt_s=1.0,
            max_time_s=0.0 + 1.0,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            launch_lat_deg=28.5,
            launch_lon_deg=-80.6,
            use_wgs84_geodesy=True,
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=0.0))
        out = sim.run()
        self.assertAlmostEqual(out.latitude_deg[0], 28.5, places=3)
        self.assertAlmostEqual(out.longitude_deg[0], -80.6, places=3)
        self.assertAlmostEqual(out.altitude_km[0], 0.0, places=3)

    def test_wind_is_reflected_in_logged_body_wind(self):
        sim_cfg = RocketSimConfig(
            dt_s=1.0,
            max_time_s=1.0,
            enable_drag=True,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            wind_enu_m_s=np.array([30.0, 0.0, 0.0]),
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=0.0))
        out = sim.run()
        self.assertGreater(np.linalg.norm(out.wind_body_m_s[0]), 1.0)

    def test_tvc_command_lags_and_generates_gimbal_angle(self):
        class _TvcGuidance:
            def command(self, state, sim_cfg, vehicle_cfg):
                return GuidanceCommand(
                    throttle=1.0,
                    torque_body_nm_cmd=np.zeros(3),
                    thrust_vector_body_cmd=np.array([1.0, 0.1, 0.0]),
                )

        sim_cfg = RocketSimConfig(
            dt_s=0.1,
            max_time_s=0.2,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            attitude_mode="dynamic",
            tvc_time_constant_s=0.5,
            tvc_rate_limit_deg_s=10.0,
            tvc_max_gimbal_deg=5.0,
            tvc_pivot_offset_body_m=np.array([0.0, 0.0, -1.0]),
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=_TvcGuidance())
        out = sim.run()
        self.assertGreater(out.tvc_gimbal_deg[0], 0.0)
        self.assertLessEqual(out.tvc_gimbal_deg[0], 1.1)

    def test_engine_performance_increases_toward_vacuum(self):
        stage = RocketStagePreset(
            name="perf",
            dry_mass_kg=10.0,
            propellant_mass_kg=50.0,
            max_thrust_n=1200.0,
            isp_s=300.0,
            burn_time_s=20.0,
            diameter_m=1.0,
            length_m=3.0,
            sea_level_thrust_n=1000.0,
            vacuum_thrust_n=1200.0,
            sea_level_isp_s=250.0,
            vacuum_isp_s=300.0,
        )
        stack = RocketStackPreset(name="perf_stack", stages=(stage,))
        sim_cfg = RocketSimConfig(
            dt_s=0.5,
            max_time_s=0.5,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
        )
        vehicle_cfg = RocketVehicleConfig(stack=stack, payload_mass_kg=0.0)
        sim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=1.0))
        sea_state = sim.initial_state()
        high_state = sea_state.copy()
        high_state.position_eci_km = np.array([EARTH_RADIUS_KM + 200.0, 0.0, 0.0], dtype=float)
        high_state.velocity_eci_km_s = np.zeros(3)
        sea_next = sim.step(sea_state, GuidanceCommand(throttle=1.0), dt_s=0.5)
        high_next = sim.step(high_state, GuidanceCommand(throttle=1.0), dt_s=0.5)
        self.assertGreater(getattr(high_next, "_last_step_thrust_n"), getattr(sea_next, "_last_step_thrust_n"))


if __name__ == "__main__":
    unittest.main()
