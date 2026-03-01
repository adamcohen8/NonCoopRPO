import unittest

import numpy as np

from sim.core.models import Command, StateTruth
from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel
from sim.dynamics.model import OrbitalAttitudeDynamics


class TestAttitudeDisturbances(unittest.TestCase):
    def test_disturbance_torque_nonzero_for_representative_state(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6778.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.67, 0.1]),
            attitude_quat_bn=np.array([0.9, 0.2, -0.1, 0.35]),
            angular_rate_body_rad_s=np.array([0.01, -0.015, 0.02]),
            mass_kg=300.0,
            t_s=0.0,
        )
        model = DisturbanceTorqueModel(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            config=DisturbanceTorqueConfig(),
        )

        tau = model.total_torque_body_nm(state)
        self.assertGreater(np.linalg.norm(tau), 0.0)

    def test_dynamics_with_disturbances_changes_angular_rate(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6778.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.67, 0.1]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.array([0.01, -0.015, 0.02]),
            mass_kg=300.0,
            t_s=0.0,
        )

        no_dist = OrbitalAttitudeDynamics(mu_km3_s2=398600.4418, inertia_kg_m2=inertia)
        with_dist = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            disturbance_model=DisturbanceTorqueModel(
                mu_km3_s2=398600.4418,
                inertia_kg_m2=inertia,
                config=DisturbanceTorqueConfig(),
            ),
        )

        command = Command.zero()
        x_no = no_dist.step(state.copy(), command, env={}, dt_s=2.0)
        x_yes = with_dist.step(state.copy(), command, env={}, dt_s=2.0)
        diff_norm = np.linalg.norm(x_yes.angular_rate_body_rad_s - x_no.angular_rate_body_rad_s)
        self.assertGreater(diff_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
