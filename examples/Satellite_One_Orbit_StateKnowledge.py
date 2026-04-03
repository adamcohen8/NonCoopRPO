import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.presets import build_sim_object_from_presets
from sim.core.kernel import SimulationKernel
from sim.core.models import SimConfig
from sim.config import get_simulation_profile, profile_choices, resolve_dt_s
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.sensors.access import AccessConfig, AccessModel
from sim.sensors.models import OwnStateSensor, SensorNoiseConfig


def run_one_orbit_state_knowledge(plot_mode: str = "interactive", profile: str = "ops") -> dict[str, str]:
    p = get_simulation_profile(profile)
    dt_s = resolve_dt_s(profile)
    update_cadence_s = 2.0
    sat = build_sim_object_from_presets(
        object_id="sat_knowledge",
        dt_s=dt_s,
        orbit_radius_km=6778.0,
        phase_rad=0.0,
        enable_disturbances=False,
        orbit_substep_s=p.orbit_substep_s,
        attitude_substep_s=p.attitude_substep_s,
        profile=profile,
    )
    sat.sensor = OwnStateSensor(
        noise=SensorNoiseConfig(sigma=np.array([1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4])),
        rng=np.random.default_rng(123),
        access_model=AccessModel(AccessConfig(update_cadence_s=update_cadence_s)),
    )
    sat.estimator = OrbitEKFEstimator(
        mu_km3_s2=EARTH_MU_KM3_S2,
        dt_s=dt_s,
        process_noise_diag=np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10]),
        meas_noise_diag=np.array([1e-6, 1e-6, 1e-6, 1e-10, 1e-10, 1e-10]),
    )

    r0_km = np.linalg.norm(sat.truth.position_eci_km)
    orbital_period_s = 2.0 * np.pi * np.sqrt((r0_km**3) / EARTH_MU_KM3_S2)
    steps = int(np.ceil(orbital_period_s / dt_s))

    kernel = SimulationKernel(
        config=SimConfig(
            dt_s=dt_s,
            steps=steps,
            integrator=p.kernel_integrator,
            realtime_mode=p.realtime_mode,
            controller_budget_ms=p.controller_budget_ms,
        ),
        objects=[sat],
        env={},
    )
    log = kernel.run()

    truth = log.truth_by_object["sat_knowledge"][:, :6]
    belief = log.belief_by_object["sat_knowledge"][:, :6]
    t = log.t_s

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    axes = axes.ravel()
    labels = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]

    for i in range(6):
        ax = axes[i]
        ax.plot(t, truth[:, i], label="truth", linewidth=1.6)
        ax.plot(t, belief[:, i], label="knowledge", linewidth=1.3, linestyle="--")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")

    axes[4].set_xlabel("Time (s)")
    axes[5].set_xlabel("Time (s)")
    fig.suptitle("One-Orbit ECI States: Truth vs EKF Knowledge (500 s updates)", y=0.98)
    fig.tight_layout()

    out_path = REPO_ROOT / "outputs" / "one_orbit_state_knowledge" / "eci_state_vs_knowledge.png"
    if plot_mode in ("save", "both"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    err = belief - truth
    fig_err, axes_err = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    axes_err = axes_err.ravel()
    err_labels = ["ex (km)", "ey (km)", "ez (km)", "evx (km/s)", "evy (km/s)", "evz (km/s)"]
    for i in range(6):
        ax = axes_err[i]
        ax.plot(t, err[:, i], linewidth=1.4)
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(err_labels[i])
        ax.grid(True, alpha=0.3)
    axes_err[4].set_xlabel("Time (s)")
    axes_err[5].set_xlabel("Time (s)")
    fig_err.suptitle("One-Orbit ECI State EKF Error (Estimate - Truth, 500 s updates)", y=0.98)
    fig_err.tight_layout()

    err_out_path = REPO_ROOT / "outputs" / "one_orbit_state_knowledge" / "eci_state_knowledge_error.png"
    if plot_mode in ("save", "both"):
        err_out_path.parent.mkdir(parents=True, exist_ok=True)
        fig_err.savefig(err_out_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_err)

    return {
        "plot_mode": plot_mode,
        "state_knowledge_plot": str(out_path) if plot_mode in ("save", "both") else "",
        "state_error_plot": str(err_out_path) if plot_mode in ("save", "both") else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one-orbit satellite state vs knowledge plot in ECI.")
    parser.add_argument(
        "--plot-mode",
        choices=["interactive", "save", "both"],
        default="interactive",
        help="Plot behavior; interactive is default.",
    )
    parser.add_argument(
        "--profile",
        choices=list(profile_choices()),
        default="ops",
        help="Fidelity profile: fast, ops, or high_fidelity.",
    )
    args = parser.parse_args()

    outputs = run_one_orbit_state_knowledge(plot_mode=args.plot_mode, profile=args.profile)
    print("Generated outputs:")
    for k, v in outputs.items():
        if v:
            print(f"  {k}: {v}")
