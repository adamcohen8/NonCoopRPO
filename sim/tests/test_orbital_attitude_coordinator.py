import unittest

import numpy as np

from sim.control.orbit.integrated import IntegratedManeuverCommand, OrbitalAttitudeManeuverCoordinator
from sim.core.models import StateTruth


class TestOrbitalAttitudeManeuverCoordinator(unittest.TestCase):
    def setUp(self):
        self.truth = StateTruth(
            position_eci_km=np.array([7000.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.5, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=100.0,
            t_s=0.0,
        )
        self.coordinator = OrbitalAttitudeManeuverCoordinator()
        self.thruster_pos = np.array([0.0, 0.0, -0.5])
        self.thruster_dir = np.array([0.0, 0.0, 1.0])

    def test_fires_when_aligned_and_valid(self):
        cmd = IntegratedManeuverCommand(
            delta_v_eci_km_s=np.array([0.0, 0.0, -0.001]),
            available_delta_v_km_s=1.0,
            strategy="thrust_limited",
            max_thrust_n=1000.0,
            dt_s=1.0,
            min_thrust_n=0.0,
            require_attitude_alignment=True,
            thruster_position_body_m=self.thruster_pos,
            thruster_direction_body=self.thruster_dir,
        )
        _, decision = self.coordinator.execute(self.truth, cmd)
        self.assertTrue(decision.executed)
        self.assertEqual(decision.action, "fire")
        self.assertFalse(decision.should_slew)

    def test_slews_when_misaligned(self):
        cmd = IntegratedManeuverCommand(
            delta_v_eci_km_s=np.array([0.001, 0.0, 0.0]),
            available_delta_v_km_s=1.0,
            strategy="thrust_limited",
            max_thrust_n=1000.0,
            dt_s=1.0,
            min_thrust_n=0.0,
            require_attitude_alignment=True,
            thruster_position_body_m=self.thruster_pos,
            thruster_direction_body=self.thruster_dir,
        )
        _, decision = self.coordinator.execute(self.truth, cmd)
        self.assertFalse(decision.executed)
        self.assertEqual(decision.action, "slew")
        self.assertTrue(decision.should_slew)
        self.assertFalse(decision.alignment_ok)
        self.assertIsNotNone(decision.required_attitude_quat_bn)

    def test_slews_without_firing_when_below_min_thrust(self):
        cmd = IntegratedManeuverCommand(
            delta_v_eci_km_s=np.array([0.0, 0.0, -1e-6]),
            available_delta_v_km_s=1.0,
            strategy="thrust_limited",
            max_thrust_n=1000.0,
            dt_s=1.0,
            min_thrust_n=0.2,
            require_attitude_alignment=True,
            thruster_position_body_m=self.thruster_pos,
            thruster_direction_body=self.thruster_dir,
        )
        _, decision = self.coordinator.execute(self.truth, cmd)
        self.assertFalse(decision.executed)
        self.assertEqual(decision.action, "slew")
        self.assertTrue(decision.should_slew)
        self.assertTrue(decision.below_min_thrust)

    def test_holds_when_insufficient_delta_v(self):
        cmd = IntegratedManeuverCommand(
            delta_v_eci_km_s=np.array([0.0, 0.0, -0.01]),
            available_delta_v_km_s=1e-6,
            strategy="thrust_limited",
            max_thrust_n=10000.0,
            dt_s=1.0,
            min_thrust_n=0.0,
            require_attitude_alignment=True,
            thruster_position_body_m=self.thruster_pos,
            thruster_direction_body=self.thruster_dir,
        )
        _, decision = self.coordinator.execute(self.truth, cmd)
        self.assertFalse(decision.executed)
        self.assertEqual(decision.action, "hold")
        self.assertFalse(decision.should_slew)
        self.assertTrue(decision.insufficient_delta_v)


if __name__ == "__main__":
    unittest.main()
