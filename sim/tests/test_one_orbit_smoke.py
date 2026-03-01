import unittest

import numpy as np

from sim.scenarios.free_tumble_one_orbit import MU_EARTH_KM3_S2
from sim.actuators.simple import ActuatorLimits, SimpleActuator
from sim.control.orbit.zero_controller import ZeroController
from sim.core.kernel import SimObject, SimulationKernel
from sim.core.models import ObjectConfig, SimConfig, StateBelief, StateTruth
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.sensors.noisy_own_state import NoisyOwnStateSensor


class TestOneOrbitSmoke(unittest.TestCase):
    def test_uncontrolled_satellite_completes_approximately_one_orbit(self):
        rng = np.random.default_rng(0)
        radius_km = 6778.0
        speed_km_s = np.sqrt(MU_EARTH_KM3_S2 / radius_km)
        period_s = 2.0 * np.pi * np.sqrt(radius_km**3 / MU_EARTH_KM3_S2)
        dt_s = 2.0
        steps = int(np.ceil(period_s / dt_s))

        init_truth = StateTruth(
            position_eci_km=np.array([radius_km, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, speed_km_s, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.array([0.01, -0.02, 0.015]),
            mass_kg=300.0,
            t_s=0.0,
        )

        init_belief = StateBelief(
            state=np.hstack((init_truth.position_eci_km, init_truth.velocity_eci_km_s)),
            covariance=np.eye(6) * 1e-6,
            last_update_t_s=0.0,
        )

        sat = SimObject(
            cfg=ObjectConfig(object_id="sat"),
            truth=init_truth,
            belief=init_belief,
            dynamics=OrbitalAttitudeDynamics(mu_km3_s2=MU_EARTH_KM3_S2, inertia_kg_m2=np.diag([120.0, 100.0, 80.0])),
            sensor=NoisyOwnStateSensor(pos_sigma_km=0.001, vel_sigma_km_s=1e-5, rng=rng),
            estimator=OrbitEKFEstimator(
                mu_km3_s2=MU_EARTH_KM3_S2,
                dt_s=dt_s,
                process_noise_diag=np.ones(6) * 1e-10,
                meas_noise_diag=np.ones(6) * 1e-8,
            ),
            controller=ZeroController(),
            actuator=SimpleActuator(),
            limits={"actuator": ActuatorLimits(max_accel_km_s2=1e-5, max_torque_nm=0.01)},
        )

        kernel = SimulationKernel(SimConfig(dt_s=dt_s, steps=steps, controller_budget_ms=1.0), [sat])
        log = kernel.run()

        r0 = np.linalg.norm(log.truth_by_object["sat"][0, :3])
        rf = np.linalg.norm(log.truth_by_object["sat"][-1, :3])
        self.assertLess(abs(rf - r0), 1.0)

        q_norms = np.linalg.norm(log.truth_by_object["sat"][:, 6:10], axis=1)
        self.assertTrue(np.allclose(q_norms, 1.0, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
