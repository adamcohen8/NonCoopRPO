import unittest

import numpy as np

from sim.control.orbit.impulsive import (
    AttitudeAgnosticImpulsiveManeuverer,
    ThrustLimitedDeltaVManeuver,
)
from sim.core.models import StateTruth
from sim.utils.quaternion import quaternion_to_dcm_bn


class TestImpulsiveAttitudeTarget(unittest.TestCase):
    def setUp(self):
        self.truth = StateTruth(
            position_eci_km=np.array([7000.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.5, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=100.0,
            t_s=0.0,
        )
        self.maneuverer = AttitudeAgnosticImpulsiveManeuverer()
        self.thruster_dir_body = np.array([0.0, 0.0, 1.0])

    def test_required_attitude_aligns_thruster_with_negative_delta_v(self):
        dv = np.array([0.01, 0.0, 0.0])
        q_req = self.maneuverer.required_attitude_for_delta_v(
            truth=self.truth,
            delta_v_eci_km_s=dv,
            thruster_direction_body=self.thruster_dir_body,
        )
        c_bn = quaternion_to_dcm_bn(q_req)
        thrust_axis_eci = c_bn.T @ self.thruster_dir_body
        target_axis_eci = -dv / np.linalg.norm(dv)
        self.assertTrue(np.allclose(thrust_axis_eci, target_axis_eci, atol=1e-8))

    def test_thrust_limited_result_includes_required_attitude(self):
        dv = np.array([0.0, 0.0, -0.001])
        _, result = self.maneuverer.execute_delta_v_with_thrust_limit(
            truth=self.truth,
            maneuver=ThrustLimitedDeltaVManeuver(
                delta_v_eci_km_s=dv,
                max_thrust_n=1000.0,
                min_thrust_n=0.0,
                dt_s=1.0,
                require_attitude_alignment=False,
                thruster_position_body_m=np.array([0.0, 0.0, -0.5]),
                thruster_direction_body=self.thruster_dir_body,
                alignment_tolerance_rad=np.deg2rad(5.0),
            ),
            available_delta_v_km_s=1.0,
        )
        self.assertIsNotNone(result.required_attitude_quat_bn)
        c_bn = quaternion_to_dcm_bn(result.required_attitude_quat_bn)
        thrust_axis_eci = c_bn.T @ self.thruster_dir_body
        target_axis_eci = -dv / np.linalg.norm(dv)
        self.assertTrue(np.allclose(thrust_axis_eci, target_axis_eci, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
