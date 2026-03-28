from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from sim.core.models import Command, StateTruth
from sim.dynamics.attitude import disturbances as disturbance_module
from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel
from sim.dynamics import model as dynamics_model_module
from sim.dynamics.model import OrbitalAttitudeDynamics


class _AttitudeCoupledDisturbance:
    def total_torque_body_nm(self, state: StateTruth, env: dict | None = None) -> np.ndarray:
        q = np.array(state.attitude_quat_bn, dtype=float).reshape(4)
        return np.array([0.0, 0.0, 50.0 * q[2]], dtype=float)


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

    def test_disturbance_torque_recomputed_each_attitude_substep(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6778.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.67, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.array([0.0, 0.4, 0.0]),
            mass_kg=300.0,
            t_s=0.0,
        )
        dyn = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            disturbance_model=_AttitudeCoupledDisturbance(),
            orbit_substep_s=1.0,
            attitude_substep_s=0.1,
        )

        out = dyn.step(state.copy(), Command.zero(), env={}, dt_s=1.0)

        self.assertGreater(abs(float(out.angular_rate_body_rad_s[2])), 1e-8)

    def test_srp_shadow_factor_cached_across_attitude_substeps(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6778.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.67, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.array([0.0, 0.0, 0.0]),
            mass_kg=300.0,
            t_s=0.0,
        )
        dyn = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            disturbance_model=DisturbanceTorqueModel(
                mu_km3_s2=398600.4418,
                inertia_kg_m2=inertia,
                config=DisturbanceTorqueConfig(
                    use_gravity_gradient=False,
                    use_magnetic=False,
                    use_drag=False,
                    use_srp=True,
                ),
            ),
            orbit_substep_s=1.0,
            attitude_substep_s=0.1,
        )

        with patch("sim.dynamics.model.srp_shadow_factor", return_value=1.0) as shadow_mock:
            dyn.step(
                state.copy(),
                Command.zero(),
                env={"sun_dir_eci": np.array([1.0, 0.0, 0.0]), "srp_shadow_model": "none"},
                dt_s=1.0,
            )

        self.assertEqual(shadow_mock.call_count, 1)

    def test_drag_density_cached_across_attitude_substeps(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6778.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.67, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.array([0.0, 0.0, 0.0]),
            mass_kg=300.0,
            t_s=0.0,
        )
        dyn = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            disturbance_model=DisturbanceTorqueModel(
                mu_km3_s2=398600.4418,
                inertia_kg_m2=inertia,
                config=DisturbanceTorqueConfig(
                    use_gravity_gradient=False,
                    use_magnetic=False,
                    use_drag=True,
                    use_srp=False,
                ),
            ),
            orbit_substep_s=1.0,
            attitude_substep_s=0.1,
        )

        with patch.object(dynamics_model_module, "density_from_model", return_value=1.0e-12) as density_mock:
            dyn.step(
                state.copy(),
                Command.zero(),
                env={"atmosphere_model": "exponential"},
                dt_s=1.0,
            )

        self.assertEqual(density_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
