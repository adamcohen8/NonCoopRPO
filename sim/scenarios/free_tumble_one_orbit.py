from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np

from sim.actuators.simple import ActuatorLimits, SimpleActuator
from sim.control.orbit.zero_controller import ZeroController
from sim.core.kernel import SimObject, SimulationKernel
from sim.core.models import ObjectConfig, SimConfig, StateBelief, StateTruth
from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.config import get_simulation_profile, resolve_dt_s
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.metrics.scoring import compute_scores
from sim.sensors.noisy_own_state import NoisyOwnStateSensor
from sim.utils.io import write_json

MU_EARTH_KM3_S2 = 398600.4418


def run_free_tumble_one_orbit(
    output_dir: str = "outputs/free_tumble_one_orbit",
    plot_mode: Literal["interactive", "save", "both"] = "interactive",
    profile: str = "ops",
) -> dict[str, str]:
    from sim.utils.plotting import plot_attitude_tumble, plot_orbit_eci

    rng = np.random.default_rng(42)

    radius_km = 6778.0
    speed_km_s = np.sqrt(MU_EARTH_KM3_S2 / radius_km)
    orbital_period_s = 2.0 * np.pi * np.sqrt((radius_km**3) / MU_EARTH_KM3_S2)

    p = get_simulation_profile(profile)
    dt_s = resolve_dt_s(profile)
    steps = int(np.ceil(orbital_period_s / dt_s))

    init_truth = StateTruth(
        position_eci_km=np.array([radius_km, 0.0, 0.0]),
        velocity_eci_km_s=np.array([0.0, speed_km_s, 0.0]),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.array([0.015, -0.01, 0.02]),
        mass_kg=300.0,
        t_s=0.0,
    )

    init_belief = StateBelief(
        state=np.hstack((init_truth.position_eci_km, init_truth.velocity_eci_km_s)),
        covariance=np.diag([1e-4, 1e-4, 1e-4, 1e-8, 1e-8, 1e-8]),
        last_update_t_s=0.0,
    )

    obj = SimObject(
        cfg=ObjectConfig(object_id="sat_01", controller_budget_ms=1.0),
        truth=init_truth,
        belief=init_belief,
        dynamics=OrbitalAttitudeDynamics(
            mu_km3_s2=MU_EARTH_KM3_S2,
            inertia_kg_m2=np.diag([120.0, 100.0, 80.0]),
            orbit_substep_s=p.orbit_substep_s,
            attitude_substep_s=p.attitude_substep_s,
            disturbance_model=DisturbanceTorqueModel(
                mu_km3_s2=MU_EARTH_KM3_S2,
                inertia_kg_m2=np.diag([120.0, 100.0, 80.0]),
                config=DisturbanceTorqueConfig(),
            ),
        ),
        sensor=NoisyOwnStateSensor(pos_sigma_km=0.001, vel_sigma_km_s=1e-5, rng=rng),
        estimator=OrbitEKFEstimator(
            mu_km3_s2=MU_EARTH_KM3_S2,
            dt_s=dt_s,
            process_noise_diag=np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10]),
            meas_noise_diag=np.array([1e-6, 1e-6, 1e-6, 1e-10, 1e-10, 1e-10]),
        ),
        controller=ZeroController(simulated_runtime_ms=0.0),
        actuator=SimpleActuator(lag_tau_s=0.0),
        limits={"actuator": ActuatorLimits(max_accel_km_s2=1e-5, max_torque_nm=0.01)},
    )

    kernel = SimulationKernel(
        config=SimConfig(
            dt_s=dt_s,
            steps=steps,
            integrator=p.kernel_integrator,
            realtime_mode=p.realtime_mode,
            controller_budget_ms=p.controller_budget_ms,
            rng_seed=42,
        ),
        objects=[obj],
        env={},
    )

    log = kernel.run()
    scores = compute_scores(log)

    out_dir = Path(output_dir)
    orbit_png = out_dir / "orbit_eci.png"
    attitude_png = out_dir / "attitude_tumble.png"
    log_json = out_dir / "sim_log.json"
    score_json = out_dir / "scores.json"

    truth_hist = log.truth_by_object["sat_01"]
    plot_orbit_eci(truth_hist=truth_hist, mode=plot_mode, out_path=str(orbit_png))
    plot_attitude_tumble(t_s=log.t_s, truth_hist=truth_hist, mode=plot_mode, out_path=str(attitude_png))

    write_json(str(log_json), log.to_jsonable())
    write_json(str(score_json), asdict(scores))

    return {
        "plot_mode": plot_mode,
        "orbit_plot": str(orbit_png) if plot_mode in ("save", "both") else "",
        "attitude_plot": str(attitude_png) if plot_mode in ("save", "both") else "",
        "log_json": str(log_json),
        "score_json": str(score_json),
    }
