from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from sim.actuators.attitude import AttitudeActuator, ReactionWheelLimits
from sim.actuators.orbital import OrbitalActuator, OrbitalActuatorLimits
from sim.control.orbit.baseline import RiskThresholdController, SafetyBarrierController, StationkeepingController
from sim.core.kernel import SimObject, SimulationKernel
from sim.core.models import Command, ObjectConfig, SimConfig, StateBelief, StateTruth
from sim.config import (
    build_default_ops_orbit_propagator,
    default_disturbance_config_for_profile,
    default_env_for_profile,
    get_simulation_profile,
    resolve_dt_s,
)
from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.estimation.aoi import AoITrackingEstimator
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.metrics.engagement import compute_engagement_metrics
from sim.sensors.access import AccessConfig, AccessModel
from sim.sensors.composite import CompositeSensorModel
from sim.sensors.models import OwnStateSensor, RelativeSensor, SensorNoiseConfig
from sim.utils.io import write_json
from sim.utils.plotting import plot_attitude_tumble, plot_orbit_eci


class CombinedActuator:
    def __init__(
        self,
        orbital: OrbitalActuator,
        attitude: AttitudeActuator,
        thrust_magnitude_scale: float = 1.0,
        pointing_error_rad: np.ndarray | None = None,
    ):
        self.orbital = orbital
        self.attitude = attitude
        self.thrust_magnitude_scale = float(thrust_magnitude_scale)
        self.pointing_error_rad = np.array(pointing_error_rad if pointing_error_rad is not None else np.zeros(3), dtype=float)

    def apply(self, command: Command, limits: dict, dt_s: float) -> Command:
        thrust = _apply_pointing_error(np.array(command.thrust_eci_km_s2, dtype=float), self.pointing_error_rad)
        thrust *= self.thrust_magnitude_scale
        command = Command(
            thrust_eci_km_s2=thrust,
            torque_body_nm=np.array(command.torque_body_nm, dtype=float),
            mode_flags=dict(command.mode_flags),
        )
        c1 = self.orbital.apply(command, limits, dt_s)
        return self.attitude.apply(c1, limits, dt_s)


class CompositeController:
    def __init__(self, orbit_ctrl, attitude_ctrl=None):
        self.orbit_ctrl = orbit_ctrl
        self.attitude_ctrl = attitude_ctrl

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        c_orb = self.orbit_ctrl.act(belief, t_s, budget_ms)
        if self.attitude_ctrl is None:
            return c_orb
        c_att = self.attitude_ctrl.act(belief, t_s, budget_ms)
        return Command(
            thrust_eci_km_s2=c_orb.thrust_eci_km_s2,
            torque_body_nm=c_att.torque_body_nm,
            mode_flags={**c_orb.mode_flags, **c_att.mode_flags},
        )


def _make_object(
    object_id: str,
    r_km: float,
    phase_rad: float,
    target_id: str,
    rng: np.random.Generator,
    profile: str,
    dt_s: float,
    initial_position_offset_km: np.ndarray | None = None,
    initial_velocity_offset_km_s: np.ndarray | None = None,
    deployment_timing_jitter_s: float = 0.0,
    thrust_magnitude_scale: float = 1.0,
    actuator_pointing_error_rad: np.ndarray | None = None,
    sensor_noise_scale_multiplier: float = 1.0,
    update_cadence_jitter_s: float = 0.0,
    dropout_probability: float | None = None,
) -> SimObject:
    phase = float(phase_rad + _phase_offset_from_timing_jitter(r_km=r_km, timing_jitter_s=deployment_timing_jitter_s))
    speed = np.sqrt(EARTH_MU_KM3_S2 / r_km)
    pos = np.array([r_km * np.cos(phase), r_km * np.sin(phase), 0.0])
    vel = np.array([-speed * np.sin(phase), speed * np.cos(phase), 0.0])
    if initial_position_offset_km is not None:
        pos = pos + np.array(initial_position_offset_km, dtype=float)
    if initial_velocity_offset_km_s is not None:
        vel = vel + np.array(initial_velocity_offset_km_s, dtype=float)

    truth = StateTruth(
        position_eci_km=pos,
        velocity_eci_km_s=vel,
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.array([0.005, -0.004, 0.006]),
        mass_kg=220.0,
        t_s=0.0,
    )
    belief_state = np.hstack((pos, vel))
    belief = StateBelief(
        state=belief_state,
        covariance=np.eye(6) * 1e-6,
        last_update_t_s=0.0,
    )

    propagator = build_default_ops_orbit_propagator(profile)
    p = get_simulation_profile(profile)
    dcfg = default_disturbance_config_for_profile(profile)
    dynamics = OrbitalAttitudeDynamics(
        mu_km3_s2=EARTH_MU_KM3_S2,
        inertia_kg_m2=np.diag([110.0, 95.0, 85.0]),
        disturbance_model=DisturbanceTorqueModel(
            EARTH_MU_KM3_S2,
            np.diag([110.0, 95.0, 85.0]),
            DisturbanceTorqueConfig(
                use_gravity_gradient=dcfg.use_gravity_gradient,
                use_magnetic=dcfg.use_magnetic,
                use_drag=dcfg.use_drag,
                use_srp=dcfg.use_srp,
            ),
        ),
        area_m2=1.3,
        orbit_substep_s=p.orbit_substep_s,
        attitude_substep_s=p.attitude_substep_s,
        orbit_propagator=propagator,
    )

    sensor = CompositeSensorModel(
        sensors=[
            OwnStateSensor(
                noise=SensorNoiseConfig(
                    sigma=np.array([0.005, 0.005, 0.005, 5e-5, 5e-5, 5e-5]) * float(sensor_noise_scale_multiplier),
                    dropout_prob=float(np.clip(0.05 if dropout_probability is None else dropout_probability, 0.0, 1.0)),
                    latency_s=0.2,
                ),
                rng=rng,
                access_model=AccessModel(
                    AccessConfig(update_cadence_s=max(1e-6, max(dt_s, 1.0) + float(update_cadence_jitter_s)))
                ),
            ),
            RelativeSensor(
                target_id=target_id,
                mode="range_rate",
                noise=SensorNoiseConfig(
                    sigma=np.array([0.01, 5e-4]) * float(sensor_noise_scale_multiplier),
                    dropout_prob=float(np.clip(0.1 if dropout_probability is None else dropout_probability, 0.0, 1.0)),
                ),
                rng=rng,
                access_model=AccessModel(
                    AccessConfig(
                        update_cadence_s=max(1e-6, max(2.0 * dt_s, 2.0) + float(update_cadence_jitter_s)),
                        max_range_km=200.0,
                    )
                ),
            ),
        ]
    )

    base_est = OrbitEKFEstimator(
        mu_km3_s2=EARTH_MU_KM3_S2,
        dt_s=dt_s,
        process_noise_diag=np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10]),
        meas_noise_diag=np.array([2.5e-5, 2.5e-5, 2.5e-5, 2.5e-9, 2.5e-9, 2.5e-9]),
    )
    estimator = AoITrackingEstimator(base_estimator=base_est)

    target = np.hstack((pos, vel))
    nominal = StationkeepingController(target_state=target)
    evasive = SafetyBarrierController(keep_out_radius_km=6600.0)
    controller = CompositeController(
        orbit_ctrl=RiskThresholdController(
            risk_fn=lambda b, t: 1.0 if np.linalg.norm(b.state[:3]) < 6650.0 else 0.0,
            nominal=nominal,
            evasive=evasive,
            threshold=0.5,
        )
    )

    actuator = CombinedActuator(
        orbital=OrbitalActuator(lag_tau_s=0.4),
        attitude=AttitudeActuator(reaction_wheels=ReactionWheelLimits(max_torque_nm=np.array([0.05, 0.05, 0.05]), max_momentum_nms=np.array([0.2, 0.2, 0.2]))),
        thrust_magnitude_scale=thrust_magnitude_scale,
        pointing_error_rad=actuator_pointing_error_rad,
    )

    return SimObject(
        cfg=ObjectConfig(object_id=object_id, controller_budget_ms=2.0),
        truth=truth,
        belief=belief,
        dynamics=dynamics,
        sensor=sensor,
        estimator=estimator,
        controller=controller,
        actuator=actuator,
        limits={
            "orbital": OrbitalActuatorLimits(max_accel_km_s2=8e-5, min_impulse_bit_km_s=2e-5, max_throttle_rate_km_s2_s=1.2e-4, isp_s=230.0),
        },
    )


def run_full_stack_demo(
    output_dir: str = "outputs/full_stack_demo",
    seed: int = 7,
    pos_sigma_km: float = 0.01,
    vel_sigma_km_s: float = 1e-4,
    mc_sample: Mapping[str, Any] | None = None,
    plot_mode: Literal["interactive", "save", "both"] = "interactive",
    profile: str = "ops",
):
    rng = np.random.default_rng(seed)
    p = get_simulation_profile(profile)
    dt_s = resolve_dt_s(profile)
    sample = _normalize_mc_sample(mc_sample)

    obj_a_sample = sample["by_object"].get("sat_a", {})
    obj_b_sample = sample["by_object"].get("sat_b", {})

    obj_a = _make_object(
        "sat_a",
        6778.0 + rng.normal(0.0, pos_sigma_km),
        0.0,
        "sat_b",
        rng,
        profile=profile,
        dt_s=dt_s,
        initial_position_offset_km=np.array(obj_a_sample.get("initial_position_offset_km", np.zeros(3)), dtype=float),
        initial_velocity_offset_km_s=np.array(obj_a_sample.get("initial_velocity_offset_km_s", rng.normal(0.0, vel_sigma_km_s, size=3)), dtype=float),
        deployment_timing_jitter_s=float(obj_a_sample.get("deployment_timing_jitter_s", 0.0)),
        thrust_magnitude_scale=float(obj_a_sample.get("thrust_magnitude_scale", 1.0)),
        actuator_pointing_error_rad=np.array(obj_a_sample.get("actuator_pointing_error_rad", np.zeros(3)), dtype=float),
        sensor_noise_scale_multiplier=float(obj_a_sample.get("sensor_noise_scale_multiplier", 1.0)),
        update_cadence_jitter_s=float(obj_a_sample.get("update_cadence_jitter_s", 0.0)),
        dropout_probability=obj_a_sample.get("dropout_probability"),
    )
    obj_b = _make_object(
        "sat_b",
        6778.0 + rng.normal(0.0, pos_sigma_km),
        0.03,
        "sat_a",
        rng,
        profile=profile,
        dt_s=dt_s,
        initial_position_offset_km=np.array(obj_b_sample.get("initial_position_offset_km", np.zeros(3)), dtype=float),
        initial_velocity_offset_km_s=np.array(obj_b_sample.get("initial_velocity_offset_km_s", rng.normal(0.0, vel_sigma_km_s, size=3)), dtype=float),
        deployment_timing_jitter_s=float(obj_b_sample.get("deployment_timing_jitter_s", 0.0)),
        thrust_magnitude_scale=float(obj_b_sample.get("thrust_magnitude_scale", 1.0)),
        actuator_pointing_error_rad=np.array(obj_b_sample.get("actuator_pointing_error_rad", np.zeros(3)), dtype=float),
        sensor_noise_scale_multiplier=float(obj_b_sample.get("sensor_noise_scale_multiplier", 1.0)),
        update_cadence_jitter_s=float(obj_b_sample.get("update_cadence_jitter_s", 0.0)),
        dropout_probability=obj_b_sample.get("dropout_probability"),
    )

    kernel = SimulationKernel(
        config=SimConfig(
            dt_s=dt_s,
            steps=int(np.ceil(1800.0 / dt_s)),
            integrator=p.kernel_integrator,
            realtime_mode=p.realtime_mode,
            controller_budget_ms=p.controller_budget_ms,
            rng_seed=seed,
        ),
        objects=[obj_a, obj_b],
        env={**default_env_for_profile(profile), "sun_dir_eci": np.array([1.0, 0.2, 0.1]), "density_kg_m3": max(0.0, rng.normal(1e-12, 1e-13))},
    )

    log = kernel.run()
    metrics = compute_engagement_metrics(log, keepout_radius_km=6650.0)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_orbit_eci(log.truth_by_object["sat_a"], mode=plot_mode, out_path=str(out / "sat_a_orbit.png"))
    plot_orbit_eci(log.truth_by_object["sat_b"], mode=plot_mode, out_path=str(out / "sat_b_orbit.png"))
    plot_attitude_tumble(log.t_s, log.truth_by_object["sat_a"], mode=plot_mode, out_path=str(out / "sat_a_attitude.png"))

    write_json(str(out / "sim_log.json"), log.to_jsonable())
    write_json(str(out / "engagement_metrics.json"), asdict(metrics))

    return {
        "log": log,
        "metrics": metrics,
        "output_dir": str(out),
        "keepout_radius_km": 6650.0,
        "sampled_parameters": sample,
    }


def _normalize_mc_sample(mc_sample: Mapping[str, Any] | None) -> dict[str, Any]:
    if mc_sample is None:
        return {"by_object": {}, "global": {}, "legacy_args": {}}
    by_object = mc_sample.get("by_object", {}) if isinstance(mc_sample, Mapping) else {}
    global_values = mc_sample.get("global", {}) if isinstance(mc_sample, Mapping) else {}
    legacy_args = mc_sample.get("legacy_args", {}) if isinstance(mc_sample, Mapping) else {}
    return {
        "by_object": dict(by_object) if isinstance(by_object, Mapping) else {},
        "global": dict(global_values) if isinstance(global_values, Mapping) else {},
        "legacy_args": dict(legacy_args) if isinstance(legacy_args, Mapping) else {},
    }


def _phase_offset_from_timing_jitter(r_km: float, timing_jitter_s: float) -> float:
    if timing_jitter_s == 0.0:
        return 0.0
    mean_motion_rad_s = float(np.sqrt(EARTH_MU_KM3_S2 / (r_km ** 3)))
    return mean_motion_rad_s * float(timing_jitter_s)


def _apply_pointing_error(vector: np.ndarray, error_rad: np.ndarray) -> np.ndarray:
    if np.linalg.norm(vector) <= 0.0:
        return vector
    rx, ry, rz = np.array(error_rad, dtype=float).reshape(3)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(rx), -np.sin(rx)], [0.0, np.sin(rx), np.cos(rx)]])
    rot_y = np.array([[np.cos(ry), 0.0, np.sin(ry)], [0.0, 1.0, 0.0], [-np.sin(ry), 0.0, np.cos(ry)]])
    rot_z = np.array([[np.cos(rz), -np.sin(rz), 0.0], [np.sin(rz), np.cos(rz), 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x @ vector
