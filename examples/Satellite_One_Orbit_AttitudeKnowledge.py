import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from presets import build_sim_object_from_presets
from sim.core.kernel import SimulationKernel
from sim.core.models import SimConfig
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.estimation.attitude_ekf import AttitudeEKFEstimator
from sim.estimation.joint_ekf import JointStateEKFEstimator
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.sensors.joint_state import JointStateSensor


def run_one_orbit_attitude_knowledge(plot_mode: str = "interactive") -> dict[str, str]:
    dt_s = 2.0
    update_cadence_s = 500.0
    sat = build_sim_object_from_presets(
        object_id="sat_att_knowledge",
        dt_s=dt_s,
        orbit_radius_km=6778.0,
        phase_rad=0.0,
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.array([0.0, 0.0, 0.0]),
        enable_disturbances=True,
        enable_attitude_knowledge=True,
    )
    sat.sensor = JointStateSensor(
        pos_sigma_km=1e-3,
        vel_sigma_km_s=1e-5,
        quat_sigma=2e-4,
        omega_sigma_rad_s=2e-5,
        update_cadence_s=update_cadence_s,
        dropout_prob=0.0,
        rng=np.random.default_rng(234),
    )
    orbit_ekf = OrbitEKFEstimator(
        mu_km3_s2=EARTH_MU_KM3_S2,
        dt_s=dt_s,
        process_noise_diag=np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10]),
        meas_noise_diag=np.array([1e-6, 1e-6, 1e-6, 1e-10, 1e-10, 1e-10]),
    )
    attitude_ekf = AttitudeEKFEstimator(
        dt_s=dt_s,
        inertia_kg_m2=sat.dynamics.inertia_kg_m2,
        process_noise_diag=np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-10, 1e-10, 1e-10]),
        meas_noise_diag=np.array([4e-8, 4e-8, 4e-8, 4e-8, 4e-10, 4e-10, 4e-10]),
    )
    sat.estimator = JointStateEKFEstimator(orbit_estimator=orbit_ekf, attitude_estimator=attitude_ekf)

    r0_km = np.linalg.norm(sat.truth.position_eci_km)
    orbital_period_s = 2.0 * np.pi * np.sqrt((r0_km**3) / EARTH_MU_KM3_S2)
    steps = int(np.ceil(orbital_period_s / dt_s))

    kernel = SimulationKernel(
        config=SimConfig(dt_s=dt_s, steps=steps, controller_budget_ms=1.0),
        objects=[sat],
        env={},
    )
    log = kernel.run()

    truth = log.truth_by_object["sat_att_knowledge"][:, 6:13]   # q0..q3, wx..wz
    belief = log.belief_by_object["sat_att_knowledge"][:, 6:13]  # q0..q3, wx..wz
    t = log.t_s

    # Align quaternion sign to avoid artificial +/- flips in plotted error.
    belief_aligned = belief.copy()
    for k in range(belief_aligned.shape[0]):
        if np.dot(belief_aligned[k, :4], truth[k, :4]) < 0.0:
            belief_aligned[k, :4] *= -1.0

    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()
    labels = ["q0", "q1", "q2", "q3", "wx (rad/s)", "wy (rad/s)", "wz (rad/s)"]
    for i in range(7):
        ax = axes[i]
        ax.plot(t, truth[:, i], label="truth", linewidth=1.6)
        ax.plot(t, belief_aligned[:, i], label="estimate", linewidth=1.3, linestyle="--")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")
    axes[7].axis("off")
    axes[6].set_xlabel("Time (s)")
    fig.suptitle("One-Orbit Attitude States: Truth vs EKF Estimate (500 s updates)", y=0.98)
    fig.tight_layout()

    err = belief_aligned - truth
    fig_err, axes_err = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    axes_err = axes_err.ravel()
    err_labels = ["eq0", "eq1", "eq2", "eq3", "ewx (rad/s)", "ewy (rad/s)", "ewz (rad/s)"]
    for i in range(7):
        ax = axes_err[i]
        ax.plot(t, err[:, i], linewidth=1.4)
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(err_labels[i])
        ax.grid(True, alpha=0.3)
    axes_err[7].axis("off")
    axes_err[6].set_xlabel("Time (s)")
    fig_err.suptitle("One-Orbit Attitude EKF Error (Estimate - Truth, 500 s updates)", y=0.98)
    fig_err.tight_layout()

    out_dir = REPO_ROOT / "outputs" / "one_orbit_attitude_knowledge"
    truth_est_path = out_dir / "attitude_truth_vs_estimate.png"
    err_path = out_dir / "attitude_estimation_error.png"

    if plot_mode in ("save", "both"):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(truth_est_path, dpi=160)
        fig_err.savefig(err_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)
    plt.close(fig_err)

    return {
        "plot_mode": plot_mode,
        "attitude_truth_vs_estimate_plot": str(truth_est_path) if plot_mode in ("save", "both") else "",
        "attitude_error_plot": str(err_path) if plot_mode in ("save", "both") else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one-orbit attitude truth-vs-estimate and error plots.")
    parser.add_argument(
        "--plot-mode",
        choices=["interactive", "save", "both"],
        default="interactive",
        help="Plot behavior; interactive is default.",
    )
    args = parser.parse_args()

    outputs = run_one_orbit_attitude_knowledge(plot_mode=args.plot_mode)
    print("Generated outputs:")
    for k, v in outputs.items():
        if v:
            print(f"  {k}: {v}")
