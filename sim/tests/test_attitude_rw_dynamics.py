import unittest

import numpy as np

from sim.actuators.attitude import AttitudeActuator, ReactionWheelLimits
from sim.core.models import Command


class TestReactionWheelDynamics(unittest.TestCase):
    def test_reaction_wheel_torque_lag_updates_wheel_state(self):
        act = AttitudeActuator(
            reaction_wheels=ReactionWheelLimits(
                max_torque_nm=np.array([0.2, 0.2, 0.2]),
                max_momentum_nms=np.array([1.0, 1.0, 1.0]),
                wheel_axes_body=np.eye(3),
                wheel_inertia_kg_m2=np.array([0.1, 0.1, 0.1]),
                torque_time_constant_s=0.5,
            )
        )
        cmd = Command(torque_body_nm=np.array([0.2, 0.0, 0.0]))
        out = act.apply(cmd, limits={}, dt_s=0.1)

        # First-order lag: alpha=dt/tau=0.2, so applied torque is 0.04 N*m.
        self.assertAlmostEqual(float(out.torque_body_nm[0]), 0.04, places=6)
        # Wheel acceleration: wdot=tau/J=0.4 rad/s^2 over 0.1s => 0.04 rad/s.
        self.assertAlmostEqual(float(act.wheel_speed_rad_s[0]), 0.04, places=6)
        self.assertAlmostEqual(float(act.wheel_momentum_wheels_nms[0]), 0.004, places=6)

    def test_reaction_wheel_momentum_and_speed_saturation(self):
        act = AttitudeActuator(
            reaction_wheels=ReactionWheelLimits(
                max_torque_nm=np.array([0.5, 0.5, 0.5]),
                max_momentum_nms=np.array([0.01, 0.01, 0.01]),
                wheel_axes_body=np.eye(3),
                wheel_inertia_kg_m2=np.array([0.1, 0.1, 0.1]),
                max_speed_rad_s=np.array([0.2, 0.2, 0.2]),
                torque_time_constant_s=0.0,
            )
        )
        cmd = Command(torque_body_nm=np.array([0.5, 0.0, 0.0]))

        for _ in range(20):
            _ = act.apply(cmd, limits={}, dt_s=0.1)

        # max momentum=0.01 with J=0.1 => max speed from momentum is 0.1 rad/s.
        self.assertLessEqual(float(abs(act.wheel_speed_rad_s[0])), 0.1000001)
        self.assertLessEqual(float(abs(act.wheel_momentum_wheels_nms[0])), 0.0100001)

        # Once saturated, actuator should not continue to apply full commanded torque.
        out = act.apply(cmd, limits={}, dt_s=0.1)
        self.assertLess(float(abs(out.torque_body_nm[0])), 1e-6)


if __name__ == "__main__":
    unittest.main()
