from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from presets.attitude_control import BASIC_REACTION_WHEEL_TRIAD, ReactionWheelAssemblyPreset
from presets.rockets import BASIC_SSTO_ROCKET, BASIC_TWO_STAGE_STACK, RocketStackPreset, RocketStagePreset
from presets.satellites import BASIC_SATELLITE, SatellitePreset
from presets.thrusters import BASIC_CHEMICAL_BOTTOM_Z, ChemicalPropulsionPreset
from sim.actuators import AttitudeActuator, CombinedActuator, OrbitalActuator, OrbitalActuatorLimits, ReactionWheelLimits
from sim.control.orbit.zero_controller import ZeroController
from sim.core.kernel import SimObject
from sim.core.models import ObjectConfig, StateBelief, StateTruth
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.estimation.joint_state import JointStateEstimator
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.sensors.joint_state import JointStateSensor
from sim.sensors.noisy_own_state import NoisyOwnStateSensor


@dataclass(frozen=True)
class RocketVehiclePreset:
    name: str
    stages: tuple[RocketStagePreset, ...]
    liftoff_mass_kg: float


def build_rocket_vehicle_from_presets(
    stack: RocketStackPreset | None = None,
    ssto: RocketStagePreset | None = None,
) -> RocketVehiclePreset:
    if stack is not None:
        return RocketVehiclePreset(name=stack.name, stages=stack.stages, liftoff_mass_kg=stack.liftoff_mass_kg)
    stage = BASIC_SSTO_ROCKET if ssto is None else ssto
    return RocketVehiclePreset(
        name=stage.name,
        stages=(stage,),
        liftoff_mass_kg=stage.dry_mass_kg + stage.propellant_mass_kg,
    )


def build_sim_object_from_presets(
    object_id: str,
    dt_s: float,
    satellite: SatellitePreset = BASIC_SATELLITE,
    thruster: ChemicalPropulsionPreset = BASIC_CHEMICAL_BOTTOM_Z,
    rw_assembly: ReactionWheelAssemblyPreset = BASIC_REACTION_WHEEL_TRIAD,
    orbit_radius_km: float = 6778.0,
    phase_rad: float = 0.0,
    attitude_quat_bn: np.ndarray | None = None,
    angular_rate_body_rad_s: np.ndarray | None = None,
    controller=None,
    rng: np.random.Generator | None = None,
    controller_budget_ms: float = 1.0,
    enable_disturbances: bool = False,
    enable_attitude_knowledge: bool = False,
    use_rectangular_prism_aero_srp: bool = False,
    rectangular_prism_dims_m: tuple[float, float, float] | None = None,
    orbit_substep_s: float | None = None,
    attitude_substep_s: float | None = None,
) -> SimObject:
    rng = np.random.default_rng(0) if rng is None else rng
    attitude_quat_bn = np.array([1.0, 0.0, 0.0, 0.0]) if attitude_quat_bn is None else np.array(attitude_quat_bn)
    angular_rate_body_rad_s = (
        np.zeros(3) if angular_rate_body_rad_s is None else np.array(angular_rate_body_rad_s, dtype=float)
    )

    speed_km_s = np.sqrt(EARTH_MU_KM3_S2 / orbit_radius_km)
    pos = np.array([orbit_radius_km * np.cos(phase_rad), orbit_radius_km * np.sin(phase_rad), 0.0])
    vel = np.array([-speed_km_s * np.sin(phase_rad), speed_km_s * np.cos(phase_rad), 0.0])

    truth = StateTruth(
        position_eci_km=pos,
        velocity_eci_km_s=vel,
        attitude_quat_bn=attitude_quat_bn,
        angular_rate_body_rad_s=angular_rate_body_rad_s,
        mass_kg=satellite.wet_mass_kg,
        t_s=0.0,
    )

    if enable_attitude_knowledge:
        belief = StateBelief(
            state=np.hstack((pos, vel, attitude_quat_bn, angular_rate_body_rad_s)),
            covariance=np.diag([1e-4, 1e-4, 1e-4, 1e-8, 1e-8, 1e-8, 1e-6, 1e-6, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7]),
            last_update_t_s=0.0,
        )
    else:
        belief = StateBelief(
            state=np.hstack((pos, vel)),
            covariance=np.diag([1e-4, 1e-4, 1e-4, 1e-8, 1e-8, 1e-8]),
            last_update_t_s=0.0,
        )

    if use_rectangular_prism_aero_srp and not enable_disturbances:
        raise ValueError(
            "Rectangular prism aero/SRP mode requires coupled orbit+attitude disturbance simulation (enable_disturbances=True)."
        )
    prism_dims = satellite.bus_size_m if rectangular_prism_dims_m is None else rectangular_prism_dims_m

    if enable_disturbances:
        from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel

        disturbance = DisturbanceTorqueModel(
            mu_km3_s2=EARTH_MU_KM3_S2,
            inertia_kg_m2=satellite.inertia_kg_m2,
            config=DisturbanceTorqueConfig(
                use_rectangular_prism_faces=bool(use_rectangular_prism_aero_srp),
                rectangular_prism_dims_m=tuple(float(v) for v in prism_dims) if use_rectangular_prism_aero_srp else None,
            ),
        )
    else:
        disturbance = None

    dynamics = OrbitalAttitudeDynamics(
        mu_km3_s2=EARTH_MU_KM3_S2,
        inertia_kg_m2=satellite.inertia_kg_m2,
        disturbance_model=disturbance,
        use_rectangular_prism_for_aero_srp=bool(use_rectangular_prism_aero_srp),
        rectangular_prism_dims_m=tuple(float(v) for v in prism_dims) if use_rectangular_prism_aero_srp else None,
        orbit_substep_s=orbit_substep_s,
        attitude_substep_s=attitude_substep_s,
    )

    orbit_estimator = OrbitEKFEstimator(
        mu_km3_s2=EARTH_MU_KM3_S2,
        dt_s=dt_s,
        process_noise_diag=np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10]),
        meas_noise_diag=np.array([1e-6, 1e-6, 1e-6, 1e-10, 1e-10, 1e-10]),
    )
    if enable_attitude_knowledge:
        sensor = JointStateSensor(
            pos_sigma_km=0.001,
            vel_sigma_km_s=1e-5,
            quat_sigma=2e-4,
            omega_sigma_rad_s=2e-5,
            rng=rng,
        )
        estimator = JointStateEstimator(orbit_estimator=orbit_estimator, dt_s=dt_s)
    else:
        sensor = NoisyOwnStateSensor(pos_sigma_km=0.001, vel_sigma_km_s=1e-5, rng=rng)
        estimator = orbit_estimator

    ctrl = ZeroController(simulated_runtime_ms=0.0) if controller is None else controller

    orbital_act = OrbitalActuator(lag_tau_s=0.0)
    rw_torque = np.zeros(3)
    rw_momentum = np.zeros(3)
    for wheel in rw_assembly.wheels:
        axis = np.abs(np.array(wheel.axis_body, dtype=float))
        rw_torque += axis * wheel.max_torque_nm
        rw_momentum += axis * wheel.max_momentum_nms
    attitude_act = AttitudeActuator(
        reaction_wheels=ReactionWheelLimits(max_torque_nm=rw_torque, max_momentum_nms=rw_momentum)
    )
    actuator = CombinedActuator(orbital=orbital_act, attitude=attitude_act)

    limits = {
        "orbital": OrbitalActuatorLimits(
            max_accel_km_s2=thruster.max_thrust_n / max(satellite.wet_mass_kg, 1e-9) / 1e3,
            min_impulse_bit_km_s=thruster.min_impulse_bit_n_s / max(satellite.wet_mass_kg, 1e-9) / 1e3,
            max_throttle_rate_km_s2_s=1e-4,
            isp_s=thruster.isp_s,
        )
    }

    return SimObject(
        cfg=ObjectConfig(object_id=object_id, controller_budget_ms=controller_budget_ms),
        truth=truth,
        belief=belief,
        dynamics=dynamics,
        sensor=sensor,
        estimator=estimator,
        controller=ctrl,
        actuator=actuator,
        limits=limits,
    )


# Convenience aliases for immediate use.
DEFAULT_ROCKET_VEHICLE = build_rocket_vehicle_from_presets(ssto=BASIC_SSTO_ROCKET)
DEFAULT_TWO_STAGE_VEHICLE = build_rocket_vehicle_from_presets(stack=BASIC_TWO_STAGE_STACK)
