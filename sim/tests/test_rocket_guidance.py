import unittest

import numpy as np

from presets.rockets import BASIC_TWO_STAGE_STACK
from sim.rocket import (
    GuidanceCommand,
    HoldAttitudeGuidance,
    MaxQThrottleLimiterGuidance,
    RocketSimConfig,
    RocketState,
    TVCSteeringGuidance,
    RocketVehicleConfig,
)


class TestRocketGuidance(unittest.TestCase):
    def _vehicle(self) -> RocketVehicleConfig:
        return RocketVehicleConfig(stack=BASIC_TWO_STAGE_STACK, payload_mass_kg=0.0)

    def _state(self, r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> RocketState:
        return RocketState(
            t_s=0.0,
            position_eci_km=np.array(r_eci_km, dtype=float),
            velocity_eci_km_s=np.array(v_eci_km_s, dtype=float),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=1000.0,
            active_stage_index=0,
            stage_prop_remaining_kg=np.array([100.0, 0.0]),
            payload_attached=True,
        )

    def test_maxq_limiter_reduces_throttle_when_q_is_high(self):
        sim_cfg = RocketSimConfig(max_time_s=1.0, dt_s=1.0, atmosphere_model="ussa1976")
        base = HoldAttitudeGuidance(throttle=1.0)
        lim = MaxQThrottleLimiterGuidance(base_guidance=base, max_q_pa=45_000.0, min_throttle=0.0)
        r = np.array([6378.137, 0.0, 0.0], dtype=float)
        v = np.array([0.0, 7.8, 0.0], dtype=float)
        s = self._state(r, v)
        cmd = lim.command(s, sim_cfg, self._vehicle())
        self.assertLess(cmd.throttle, 1.0)
        self.assertGreaterEqual(cmd.throttle, 0.0)

    def test_maxq_limiter_leaves_throttle_when_q_is_low(self):
        sim_cfg = RocketSimConfig(max_time_s=1.0, dt_s=1.0, atmosphere_model="ussa1976")
        base = HoldAttitudeGuidance(throttle=0.7)
        lim = MaxQThrottleLimiterGuidance(base_guidance=base, max_q_pa=45_000.0, min_throttle=0.0)
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        omega = np.array([0.0, 0.0, 7.2921159e-5], dtype=float)
        v_atm = np.cross(omega, r)
        s = self._state(r, v_atm)
        cmd = lim.command(s, sim_cfg, self._vehicle())
        self.assertAlmostEqual(cmd.throttle, 0.7, places=8)

    def test_maxq_limiter_accounts_for_wind(self):
        base = HoldAttitudeGuidance(throttle=0.7)
        lim = MaxQThrottleLimiterGuidance(base_guidance=base, max_q_pa=100.0, min_throttle=0.0)
        sim_cfg = RocketSimConfig(
            max_time_s=1.0,
            dt_s=1.0,
            atmosphere_model="ussa1976",
            wind_enu_m_s=np.array([80.0, 0.0, 0.0]),
        )
        r = np.array([6378.137, 0.0, 0.0], dtype=float)
        omega = np.array([0.0, 0.0, 7.2921159e-5], dtype=float)
        v = np.cross(omega, r)
        s = self._state(r, v)
        cmd = lim.command(s, sim_cfg, self._vehicle())
        self.assertLess(cmd.throttle, 0.7)

    def test_tvc_wrapper_converts_attitude_target_to_body_vector_command(self):
        class _Base:
            def command(self, state, sim_cfg, vehicle_cfg):
                return GuidanceCommand(
                    throttle=0.6,
                    attitude_quat_bn_cmd=np.array([np.cos(np.pi / 8), 0.0, np.sin(np.pi / 8), 0.0]),
                    torque_body_nm_cmd=np.zeros(3),
                )

        sim_cfg = RocketSimConfig(max_time_s=1.0, dt_s=1.0)
        s = self._state(np.array([6378.137, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        cmd = TVCSteeringGuidance(base_guidance=_Base()).command(s, sim_cfg, self._vehicle())
        self.assertIsNone(cmd.attitude_quat_bn_cmd)
        self.assertIsNotNone(cmd.thrust_vector_body_cmd)
        self.assertGreater(np.linalg.norm(cmd.thrust_vector_body_cmd[1:]), 1e-6)


if __name__ == "__main__":
    unittest.main()
