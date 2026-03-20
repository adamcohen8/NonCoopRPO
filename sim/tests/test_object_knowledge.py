import unittest

import numpy as np

from sim.actuators.simple import ActuatorLimits, SimpleActuator
from sim.control.orbit.zero_controller import ZeroController
from sim.core.kernel import SimObject, SimulationKernel
from sim.core.models import ObjectConfig, SimConfig, StateBelief, StateTruth
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.knowledge import (
    KnowledgeConditionConfig,
    KnowledgeEKFConfig,
    KnowledgeNoiseConfig,
    ObjectKnowledgeBase,
    TrackedObjectConfig,
)
from sim.sensors.noisy_own_state import NoisyOwnStateSensor


def _make_sat(object_id: str, phase_rad: float, dt_s: float) -> SimObject:
    radius_km = 6778.0
    speed_km_s = np.sqrt(EARTH_MU_KM3_S2 / radius_km)
    pos = np.array([radius_km * np.cos(phase_rad), radius_km * np.sin(phase_rad), 0.0])
    vel = np.array([-speed_km_s * np.sin(phase_rad), speed_km_s * np.cos(phase_rad), 0.0])
    truth = StateTruth(
        position_eci_km=pos,
        velocity_eci_km_s=vel,
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.zeros(3),
        mass_kg=100.0,
        t_s=0.0,
    )
    belief = StateBelief(
        state=np.hstack((pos, vel)),
        covariance=np.eye(6) * 1e-3,
        last_update_t_s=0.0,
    )
    return SimObject(
        cfg=ObjectConfig(object_id=object_id),
        truth=truth,
        belief=belief,
        dynamics=OrbitalAttitudeDynamics(mu_km3_s2=EARTH_MU_KM3_S2, inertia_kg_m2=np.diag([100.0, 95.0, 90.0])),
        sensor=NoisyOwnStateSensor(pos_sigma_km=0.0, vel_sigma_km_s=0.0, rng=np.random.default_rng(0)),
        estimator=OrbitEKFEstimator(
            mu_km3_s2=EARTH_MU_KM3_S2,
            dt_s=dt_s,
            process_noise_diag=np.ones(6) * 1e-12,
            meas_noise_diag=np.ones(6) * 1e-12,
        ),
        controller=ZeroController(),
        actuator=SimpleActuator(),
        limits={"actuator": ActuatorLimits(max_accel_km_s2=0.0, max_torque_nm=0.0)},
    )


class TestObjectKnowledge(unittest.TestCase):
    def test_observer_tracks_target_when_conditions_allow(self):
        dt_s = 1.0
        observer = _make_sat("obs", phase_rad=0.0, dt_s=dt_s)
        target = _make_sat("tgt", phase_rad=0.05, dt_s=dt_s)
        observer.knowledge_base = ObjectKnowledgeBase(
            observer_id="obs",
            dt_s=dt_s,
            rng=np.random.default_rng(1),
            tracked_objects=[
                TrackedObjectConfig(
                    target_id="tgt",
                    conditions=KnowledgeConditionConfig(refresh_rate_s=2.0, max_range_km=5000.0, require_line_of_sight=True),
                    sensor_noise=KnowledgeNoiseConfig(
                        pos_sigma_km=np.zeros(3),
                        vel_sigma_km_s=np.zeros(3),
                    ),
                    estimator="ekf",
                    ekf=KnowledgeEKFConfig(
                        process_noise_diag=np.ones(6) * 1e-12,
                        meas_noise_diag=np.ones(6) * 1e-12,
                        init_cov_diag=np.ones(6) * 1e-6,
                    ),
                )
            ],
        )

        log = SimulationKernel(SimConfig(dt_s=dt_s, steps=40, controller_budget_ms=1.0), [observer, target]).run()
        self.assertIn("obs", log.knowledge_by_observer)
        self.assertIn("tgt", log.knowledge_by_observer["obs"])
        hist = log.knowledge_by_observer["obs"]["tgt"]
        self.assertEqual(hist.shape, (41, 6))
        self.assertTrue(np.any(np.isfinite(hist)))
        summary = observer.knowledge_base.consistency_summary()
        self.assertIn("tgt", summary)
        self.assertGreater(float(summary["tgt"]["measurement_count"]), 0.0)
        self.assertGreater(float(summary["tgt"]["update_count"]), 0.0)
        self.assertIsNotNone(summary["tgt"]["nis_mean"])
        self.assertIsNotNone(summary["tgt"]["nees_mean"])
        detect = observer.knowledge_base.detection_summary()
        self.assertEqual(int(detect["tgt"]["status_counts"]["detected"]), int(detect["tgt"]["detected_count"]))
        self.assertGreater(float(detect["tgt"]["detection_rate"]), 0.0)

    def test_observer_has_no_knowledge_when_out_of_range(self):
        dt_s = 1.0
        observer = _make_sat("obs", phase_rad=0.0, dt_s=dt_s)
        target = _make_sat("tgt", phase_rad=np.pi, dt_s=dt_s)
        observer.knowledge_base = ObjectKnowledgeBase(
            observer_id="obs",
            dt_s=dt_s,
            rng=np.random.default_rng(2),
            tracked_objects=[
                TrackedObjectConfig(
                    target_id="tgt",
                    conditions=KnowledgeConditionConfig(refresh_rate_s=1.0, max_range_km=1.0),
                    sensor_noise=KnowledgeNoiseConfig(
                        pos_sigma_km=np.zeros(3),
                        vel_sigma_km_s=np.zeros(3),
                    ),
                )
            ],
        )

        log = SimulationKernel(SimConfig(dt_s=dt_s, steps=20, controller_budget_ms=1.0), [observer, target]).run()
        hist = log.knowledge_by_observer["obs"]["tgt"]
        self.assertTrue(np.isnan(hist).all())
        summary = observer.knowledge_base.consistency_summary()
        self.assertEqual(int(summary["tgt"]["measurement_count"]), 0)
        self.assertEqual(int(summary["tgt"]["update_count"]), 0)
        detect = observer.knowledge_base.detection_summary()
        self.assertEqual(int(detect["tgt"]["detected_count"]), 0)
        self.assertGreater(int(detect["tgt"]["status_counts"]["range"]), 0)

    def test_sensor_solid_angle_blocks_detection_when_boresight_misses(self):
        dt_s = 1.0
        observer = _make_sat("obs", phase_rad=0.0, dt_s=dt_s)
        target = _make_sat("tgt", phase_rad=0.0, dt_s=dt_s)
        observer.truth.position_eci_km = np.array([7000.0, 0.0, 0.0], dtype=float)
        target.truth.position_eci_km = np.array([7100.0, 0.0, 0.0], dtype=float)
        target.truth.velocity_eci_km_s = observer.truth.velocity_eci_km_s.copy()
        observer.knowledge_base = ObjectKnowledgeBase(
            observer_id="obs",
            dt_s=dt_s,
            rng=np.random.default_rng(3),
            tracked_objects=[
                TrackedObjectConfig(
                    target_id="tgt",
                    conditions=KnowledgeConditionConfig(
                        refresh_rate_s=1.0,
                        max_range_km=5000.0,
                        require_line_of_sight=False,
                        sensor_position_body_m=np.array([1.0, 0.0, 0.0], dtype=float),
                        solid_angle_sr=float(np.pi),
                    ),
                    sensor_noise=KnowledgeNoiseConfig(
                        pos_sigma_km=np.zeros(3),
                        vel_sigma_km_s=np.zeros(3),
                    ),
                )
            ],
        )
        log = SimulationKernel(SimConfig(dt_s=dt_s, steps=2, controller_budget_ms=1.0), [observer, target]).run()
        self.assertTrue(np.any(np.isfinite(log.knowledge_by_observer["obs"]["tgt"])))

        observer = _make_sat("obs", phase_rad=0.0, dt_s=dt_s)
        target = _make_sat("tgt", phase_rad=0.0, dt_s=dt_s)
        observer.truth.position_eci_km = np.array([7000.0, 0.0, 0.0], dtype=float)
        observer.truth.attitude_quat_bn = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        target.truth.position_eci_km = np.array([7100.0, 0.0, 0.0], dtype=float)
        target.truth.velocity_eci_km_s = observer.truth.velocity_eci_km_s.copy()
        observer.knowledge_base = ObjectKnowledgeBase(
            observer_id="obs",
            dt_s=dt_s,
            rng=np.random.default_rng(4),
            tracked_objects=[
                TrackedObjectConfig(
                    target_id="tgt",
                    conditions=KnowledgeConditionConfig(
                        refresh_rate_s=1.0,
                        max_range_km=5000.0,
                        require_line_of_sight=False,
                        sensor_position_body_m=np.array([1.0, 0.0, 0.0], dtype=float),
                        solid_angle_sr=float(np.pi),
                    ),
                    sensor_noise=KnowledgeNoiseConfig(
                        pos_sigma_km=np.zeros(3),
                        vel_sigma_km_s=np.zeros(3),
                    ),
                )
            ],
        )
        log = SimulationKernel(SimConfig(dt_s=dt_s, steps=2, controller_budget_ms=1.0), [observer, target]).run()
        self.assertTrue(np.isnan(log.knowledge_by_observer["obs"]["tgt"]).all())
        detect = observer.knowledge_base.detection_summary()
        self.assertGreater(int(detect["tgt"]["status_counts"]["solid_angle"]), 0)

    def test_solid_angle_4pi_disables_directional_gating(self):
        dt_s = 1.0
        observer = _make_sat("obs", phase_rad=0.0, dt_s=dt_s)
        target = _make_sat("tgt", phase_rad=0.0, dt_s=dt_s)
        observer.truth.position_eci_km = np.array([7000.0, 0.0, 0.0], dtype=float)
        observer.truth.attitude_quat_bn = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        target.truth.position_eci_km = np.array([7100.0, 0.0, 0.0], dtype=float)
        target.truth.velocity_eci_km_s = observer.truth.velocity_eci_km_s.copy()
        observer.knowledge_base = ObjectKnowledgeBase(
            observer_id="obs",
            dt_s=dt_s,
            rng=np.random.default_rng(5),
            tracked_objects=[
                TrackedObjectConfig(
                    target_id="tgt",
                    conditions=KnowledgeConditionConfig(
                        refresh_rate_s=1.0,
                        max_range_km=5000.0,
                        require_line_of_sight=False,
                        sensor_position_body_m=np.array([1.0, 0.0, 0.0], dtype=float),
                        solid_angle_sr=float(4.0 * np.pi),
                    ),
                    sensor_noise=KnowledgeNoiseConfig(
                        pos_sigma_km=np.zeros(3),
                        vel_sigma_km_s=np.zeros(3),
                    ),
                )
            ],
        )
        log = SimulationKernel(SimConfig(dt_s=dt_s, steps=2, controller_budget_ms=1.0), [observer, target]).run()
        self.assertTrue(np.any(np.isfinite(log.knowledge_by_observer["obs"]["tgt"])))


if __name__ == "__main__":
    unittest.main()
