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
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.metrics.scoring import compute_scores
from sim.sensors.noisy_own_state import NoisyOwnStateSensor
from sim.utils.io import write_json

MU_EARTH_KM3_S2 = 398600.4418


def run_free_tumble_one_orbit_ric(
    output_dir: str = "outputs/free_tumble_one_orbit_ric",
    plot_mode: Literal["interactive", "save", "both"] = "interactive",
) -> dict[str, str]:
    from sim.utils.plotting import plot_angular_rates, plot_attitude_ric, plot_orbit_eci

    rng = np.random.default_rng(24)

    radius_km = 6778.0
    speed_km_s = np.sqrt(MU_EARTH_KM3_S2 / radius_km)
    orbital_period_s = 2.0 * np.pi * np.sqrt((radius_km**3) / MU_EARTH_KM3_S2)

    dt_s = 2.0
    steps = int(np.ceil(orbital_period_s / dt_s))

    init_truth = StateTruth(
        position_eci_km=np.array([radius_km, 0.0, 0.0]),
        velocity_eci_km_s=np.array([0.0, speed_km_s, 0.0]),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.array([0.0, 0.0, 0.0]),
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
            disturbance_model=DisturbanceTorqueModel(
                mu_km3_s2=MU_EARTH_KM3_S2,
                inertia_kg_m2=np.diag([120.0, 100.0, 80.0]),
                config=DisturbanceTorqueConfig(
                    use_gravity_gradient=True,
                    use_magnetic=False,
                    use_drag=False,
                    use_srp=False,
                ),
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
        limits={"actuator": ActuatorLimits(max_accel_km_s2=0.0, max_torque_nm=0.0)},
    )

    kernel = SimulationKernel(
        config=SimConfig(
            dt_s=dt_s,
            steps=steps,
            integrator="rk4",
            realtime_mode=True,
            controller_budget_ms=1.0,
            rng_seed=24,
        ),
        objects=[obj],
        env={},
    )

    log = kernel.run()
    scores = compute_scores(log)

    out_dir = Path(output_dir)
    orbit_png = out_dir / "orbit_eci.png"
    attitude_ric_png = out_dir / "attitude_ric.png"
    angular_rate_png = out_dir / "angular_rates.png"
    log_json = out_dir / "sim_log.json"
    score_json = out_dir / "scores.json"

    truth_hist = log.truth_by_object["sat_01"]
    plot_orbit_eci(truth_hist=truth_hist, mode=plot_mode, out_path=str(orbit_png))
    plot_attitude_ric(t_s=log.t_s, truth_hist=truth_hist, mode=plot_mode, out_path=str(attitude_ric_png))
    plot_angular_rates(t_s=log.t_s, truth_hist=truth_hist, mode=plot_mode, out_path=str(angular_rate_png))

    write_json(str(log_json), log.to_jsonable())
    write_json(str(score_json), asdict(scores))

    return {
        "plot_mode": plot_mode,
        "orbit_plot": str(orbit_png) if plot_mode in ("save", "both") else "",
        "attitude_ric_plot": str(attitude_ric_png) if plot_mode in ("save", "both") else "",
        "angular_rate_plot": str(angular_rate_png) if plot_mode in ("save", "both") else "",
        "log_json": str(log_json),
        "score_json": str(score_json),
    }
