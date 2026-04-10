from __future__ import annotations

import unittest

import numpy as np

from sim.actuators.attitude import AttitudeActuator, ReactionWheelLimits
from sim.actuators.combined import CombinedActuator
from sim.actuators.orbital import OrbitalActuator, OrbitalActuatorLimits
from sim.core.kernel import SimObject, SimulationKernel
from sim.core.models import Command, Measurement, ObjectConfig, SimConfig, StateBelief, StateTruth


class _MassDepletingDynamics:
    def step(self, state: StateTruth, command: Command, env: dict, dt_s: float) -> StateTruth:
        return StateTruth(
            position_eci_km=state.position_eci_km.copy(),
            velocity_eci_km_s=state.velocity_eci_km_s.copy(),
            attitude_quat_bn=state.attitude_quat_bn.copy(),
            angular_rate_body_rad_s=state.angular_rate_body_rad_s.copy(),
            mass_kg=float(state.mass_kg - float(command.mode_flags.get("delta_mass_kg", 0.0))),
            t_s=state.t_s + dt_s,
        )


class _NoSensor:
    def measure(self, truth: StateTruth, env: dict, t_s: float) -> Measurement | None:
        return None


class _HoldEstimator:
    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        return StateBelief(state=belief.state.copy(), covariance=belief.covariance.copy(), last_update_t_s=t_s)


class _ConstantThrustController:
    def __init__(self, thrust_eci_km_s2: np.ndarray):
        self.thrust_eci_km_s2 = np.array(thrust_eci_km_s2, dtype=float)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        return Command(thrust_eci_km_s2=self.thrust_eci_km_s2.copy(), torque_body_nm=np.zeros(3))


class TestOrbitalActuator(unittest.TestCase):
    def test_delta_mass_scales_with_current_vehicle_mass(self):
        actuator = OrbitalActuator()
        limits = {
            "orbital": OrbitalActuatorLimits(
                max_accel_km_s2=1.0,
                max_throttle_rate_km_s2_s=1.0,
                isp_s=250.0,
            )
        }
        command = Command(
            thrust_eci_km_s2=np.array([0.002, 0.0, 0.0], dtype=float),
            mode_flags={"current_mass_kg": 500.0},
        )

        out = actuator.apply(command, limits, dt_s=4.0)

        expected = 500.0 * 2.0 / (250.0 * 9.80665) * 4.0
        self.assertAlmostEqual(float(out.mode_flags["delta_mass_kg"]), expected, places=10)

    def test_max_thrust_n_limits_applied_accel_by_current_mass(self):
        actuator = OrbitalActuator()
        limits = {
            "orbital": OrbitalActuatorLimits(
                max_accel_km_s2=10.0,
                max_thrust_n=500.0,
                max_throttle_rate_km_s2_s=10.0,
                isp_s=250.0,
            )
        }
        command = Command(
            thrust_eci_km_s2=np.array([0.002, 0.0, 0.0], dtype=float),
            mode_flags={"current_mass_kg": 500.0},
        )

        out = actuator.apply(command, limits, dt_s=1.0)

        self.assertTrue(np.allclose(np.array(out.thrust_eci_km_s2, dtype=float), np.array([0.001, 0.0, 0.0], dtype=float)))
        self.assertAlmostEqual(float(out.mode_flags["effective_max_accel_km_s2"]), 0.001, places=12)

    def test_kernel_supplies_current_mass_to_orbital_actuator(self):
        truth0 = StateTruth(
            position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
            velocity_eci_km_s=np.zeros(3, dtype=float),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_rate_body_rad_s=np.zeros(3, dtype=float),
            mass_kg=500.0,
            t_s=0.0,
        )
        sat = SimObject(
            cfg=ObjectConfig(object_id="sat"),
            truth=truth0,
            belief=StateBelief(state=np.zeros(6), covariance=np.eye(6), last_update_t_s=0.0),
            dynamics=_MassDepletingDynamics(),
            sensor=_NoSensor(),
            estimator=_HoldEstimator(),
            controller=_ConstantThrustController(np.array([0.002, 0.0, 0.0], dtype=float)),
            actuator=OrbitalActuator(),
            limits={
                "orbital": OrbitalActuatorLimits(
                    max_accel_km_s2=1.0,
                    max_throttle_rate_km_s2_s=1.0,
                    isp_s=250.0,
                )
            },
        )

        log = SimulationKernel(
            config=SimConfig(dt_s=4.0, steps=2, controller_budget_ms=1.0, terminate_on_earth_impact=False),
            objects=[sat],
            env={},
        ).run()

        expected_delta_mass = 500.0 * 2.0 / (250.0 * 9.80665) * 4.0
        final_mass = float(log.truth_by_object["sat"][-1, 13])
        self.assertAlmostEqual(final_mass, 500.0 - expected_delta_mass, places=10)

    def test_kernel_max_thrust_limit_increases_accel_as_mass_drops(self):
        truth0 = StateTruth(
            position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
            velocity_eci_km_s=np.zeros(3, dtype=float),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_rate_body_rad_s=np.zeros(3, dtype=float),
            mass_kg=500.0,
            t_s=0.0,
        )
        sat = SimObject(
            cfg=ObjectConfig(object_id="sat"),
            truth=truth0,
            belief=StateBelief(state=np.zeros(6), covariance=np.eye(6), last_update_t_s=0.0),
            dynamics=_MassDepletingDynamics(),
            sensor=_NoSensor(),
            estimator=_HoldEstimator(),
            controller=_ConstantThrustController(np.array([0.01, 0.0, 0.0], dtype=float)),
            actuator=OrbitalActuator(),
            limits={
                "orbital": OrbitalActuatorLimits(
                    max_accel_km_s2=10.0,
                    max_thrust_n=500.0,
                    max_throttle_rate_km_s2_s=10.0,
                    isp_s=250.0,
                )
            },
        )

        log = SimulationKernel(
            config=SimConfig(dt_s=4.0, steps=2, controller_budget_ms=1.0, terminate_on_earth_impact=False),
            objects=[sat],
            env={},
        ).run()

        first_accel = float(log.applied_thrust_by_object["sat"][1, 0])
        second_accel = float(log.applied_thrust_by_object["sat"][2, 0])
        self.assertAlmostEqual(first_accel, 500.0 / 500.0 / 1e3, places=12)
        self.assertGreater(second_accel, first_accel)

    def test_orbital_actuator_couples_applied_thrust_to_current_attitude(self):
        actuator = OrbitalActuator()
        limits = {
            "orbital": OrbitalActuatorLimits(
                max_accel_km_s2=1.0,
                max_throttle_rate_km_s2_s=1.0,
                isp_s=250.0,
                thruster_direction_body=np.array([0.0, 0.0, 1.0], dtype=float),
            )
        }
        command = Command(
            thrust_eci_km_s2=np.array([0.002, 0.0, 0.0], dtype=float),
            mode_flags={
                "current_mass_kg": 500.0,
                "current_attitude_quat_bn": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            },
        )

        out = actuator.apply(command, limits, dt_s=1.0)

        self.assertTrue(np.allclose(np.array(out.thrust_eci_km_s2, dtype=float), np.array([0.0, 0.0, -0.002], dtype=float)))

    def test_orbital_actuator_adds_thruster_mount_torque(self):
        actuator = OrbitalActuator()
        limits = {
            "orbital": OrbitalActuatorLimits(
                max_accel_km_s2=1.0,
                max_throttle_rate_km_s2_s=1.0,
                isp_s=250.0,
                thruster_direction_body=np.array([0.0, 0.0, 1.0], dtype=float),
                thruster_position_body_m=np.array([0.2, 0.0, 0.0], dtype=float),
            )
        }
        command = Command(
            thrust_eci_km_s2=np.array([2.0e-6, 0.0, 0.0], dtype=float),
            torque_body_nm=np.array([0.1, 0.0, 0.0], dtype=float),
            mode_flags={
                "current_mass_kg": 500.0,
                "current_attitude_quat_bn": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            },
        )

        out = actuator.apply(command, limits, dt_s=1.0)

        self.assertTrue(np.allclose(np.array(out.torque_body_nm, dtype=float), np.array([0.1, 0.2, 0.0], dtype=float)))

    def test_combined_actuator_preserves_thruster_torque_after_attitude_limits(self):
        actuator = CombinedActuator(
            orbital=OrbitalActuator(),
            attitude=AttitudeActuator(
                reaction_wheels=ReactionWheelLimits(
                    max_torque_nm=np.array([0.1, 0.1, 0.1], dtype=float),
                    max_momentum_nms=np.array([1.0, 1.0, 1.0], dtype=float),
                )
            ),
        )
        limits = {
            "orbital": OrbitalActuatorLimits(
                max_accel_km_s2=1.0,
                max_throttle_rate_km_s2_s=1.0,
                isp_s=250.0,
                thruster_direction_body=np.array([0.0, 0.0, 1.0], dtype=float),
                thruster_position_body_m=np.array([0.2, 0.0, 0.0], dtype=float),
            )
        }
        command = Command(
            thrust_eci_km_s2=np.array([2.0e-6, 0.0, 0.0], dtype=float),
            torque_body_nm=np.array([0.2, 0.0, 0.0], dtype=float),
            mode_flags={
                "current_mass_kg": 500.0,
                "current_attitude_quat_bn": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            },
        )

        out = actuator.apply(command, limits, dt_s=1.0)

        self.assertTrue(np.allclose(np.array(out.torque_body_nm, dtype=float), np.array([0.1, 0.2, 0.0], dtype=float)))


if __name__ == "__main__":
    unittest.main()
