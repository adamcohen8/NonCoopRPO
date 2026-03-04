import unittest

import numpy as np

from sim.control.orbit import IntegratedManeuverCommand, OrbitalAttitudeManeuverCoordinator
from sim.core.models import StateTruth


class TestIntegratedTransitionSequence(unittest.TestCase):
    def test_misaligned_then_below_min_then_fire(self):
        coord = OrbitalAttitudeManeuverCoordinator()
        truth = StateTruth(
            position_eci_km=np.array([7000.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.5, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=100.0,
            t_s=0.0,
        )
        thruster_pos = np.array([0.0, 0.0, -0.5])
        thruster_dir = np.array([0.0, 0.0, 1.0])

        cmd1 = IntegratedManeuverCommand(
            delta_v_eci_km_s=np.array([0.001, 0.0, 0.0]),
            available_delta_v_km_s=1.0,
            strategy="thrust_limited",
            max_thrust_n=1000.0,
            dt_s=1.0,
            min_thrust_n=0.0,
            require_attitude_alignment=True,
            thruster_position_body_m=thruster_pos,
            thruster_direction_body=thruster_dir,
        )
        _, d1 = coord.execute(truth, cmd1)
        self.assertEqual(d1.action, "slew")
        self.assertFalse(d1.executed)
        self.assertFalse(d1.alignment_ok)
        self.assertIsNotNone(d1.required_attitude_quat_bn)

        truth_aligned = truth.copy()
        truth_aligned.attitude_quat_bn = d1.required_attitude_quat_bn.copy()

        cmd2 = IntegratedManeuverCommand(
            delta_v_eci_km_s=np.array([1e-6, 0.0, 0.0]),
            available_delta_v_km_s=d1.remaining_delta_v_km_s,
            strategy="thrust_limited",
            max_thrust_n=1000.0,
            dt_s=1.0,
            min_thrust_n=0.2,
            require_attitude_alignment=True,
            thruster_position_body_m=thruster_pos,
            thruster_direction_body=thruster_dir,
        )
        _, d2 = coord.execute(truth_aligned, cmd2)
        self.assertEqual(d2.action, "slew")
        self.assertFalse(d2.executed)
        self.assertTrue(d2.alignment_ok)
        self.assertTrue(d2.below_min_thrust)

        cmd3 = IntegratedManeuverCommand(
            delta_v_eci_km_s=np.array([1e-3, 0.0, 0.0]),
            available_delta_v_km_s=d2.remaining_delta_v_km_s,
            strategy="thrust_limited",
            max_thrust_n=1000.0,
            dt_s=1.0,
            min_thrust_n=0.2,
            require_attitude_alignment=True,
            thruster_position_body_m=thruster_pos,
            thruster_direction_body=thruster_dir,
        )
        _, d3 = coord.execute(truth_aligned, cmd3)
        self.assertEqual(d3.action, "fire")
        self.assertTrue(d3.executed)
        self.assertGreater(d3.applied_delta_v_km_s, 0.0)


if __name__ == "__main__":
    unittest.main()
