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
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.knowledge import (
    KnowledgeConditionConfig,
    KnowledgeEKFConfig,
    KnowledgeNoiseConfig,
    ObjectKnowledgeBase,
    TrackedObjectConfig,
)


def run_demo(plot_mode: str = "interactive") -> dict[str, str]:
    dt_s = 2.0
    observer = build_sim_object_from_presets(
        object_id="observer",
        dt_s=dt_s,
        orbit_radius_km=6778.0,
        phase_rad=0.0,
        enable_disturbances=False,
    )
    target = build_sim_object_from_presets(
        object_id="target",
        dt_s=dt_s,
        orbit_radius_km=6778.0,
        phase_rad=0.12,
        enable_disturbances=False,
    )

    observer.knowledge_base = ObjectKnowledgeBase(
        observer_id="observer",
        dt_s=dt_s,
        mu_km3_s2=EARTH_MU_KM3_S2,
        rng=np.random.default_rng(42),
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(
                    refresh_rate_s=30.0,
                    max_range_km=2000.0,
                    require_line_of_sight=True,
                    dropout_prob=0.02,
                ),
                sensor_noise=KnowledgeNoiseConfig(
                    pos_sigma_km=np.array([5e-3, 5e-3, 5e-3]),
                    vel_sigma_km_s=np.array([5e-5, 5e-5, 5e-5]),
                ),
                estimator="ekf",
                ekf=KnowledgeEKFConfig(
                    process_noise_diag=np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10]),
                    meas_noise_diag=np.array([2.5e-5, 2.5e-5, 2.5e-5, 2.5e-9, 2.5e-9, 2.5e-9]),
                    init_cov_diag=np.array([1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-2]),
                ),
            )
        ],
    )

    r0_km = np.linalg.norm(observer.truth.position_eci_km)
    period_s = 2.0 * np.pi * np.sqrt((r0_km**3) / EARTH_MU_KM3_S2)
    steps = int(np.ceil(period_s / dt_s))

    kernel = SimulationKernel(
        config=SimConfig(dt_s=dt_s, steps=steps, controller_budget_ms=1.0),
        objects=[observer, target],
        env={},
    )
    log = kernel.run()

    target_truth = log.truth_by_object["target"][:, :6]
    target_knowledge = log.knowledge_by_observer["observer"]["target"]
    t = log.t_s

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    axes = axes.ravel()
    labels = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]
    for i in range(6):
        axes[i].plot(t, target_truth[:, i], label="target truth", linewidth=1.4)
        axes[i].plot(t, target_knowledge[:, i], "--", label="observer knowledge", linewidth=1.2)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[4].set_xlabel("Time (s)")
    axes[5].set_xlabel("Time (s)")
    fig.suptitle("Cross-Object Knowledge (Observer Tracking Target with EKF)", y=0.99)
    fig.tight_layout()

    out = REPO_ROOT / "outputs" / "object_knowledge_demo" / "target_truth_vs_observer_knowledge.png"
    if plot_mode in ("save", "both"):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    err = target_knowledge - target_truth
    fig_e, axes_e = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    axes_e = axes_e.ravel()
    err_labels = ["ex (km)", "ey (km)", "ez (km)", "evx (km/s)", "evy (km/s)", "evz (km/s)"]
    for i in range(6):
        axes_e[i].plot(t, err[:, i], linewidth=1.3)
        axes_e[i].axhline(0.0, color="k", linewidth=0.7, alpha=0.5)
        axes_e[i].set_ylabel(err_labels[i])
        axes_e[i].grid(True, alpha=0.3)
    axes_e[4].set_xlabel("Time (s)")
    axes_e[5].set_xlabel("Time (s)")
    fig_e.suptitle("Cross-Object Knowledge Error (Estimate - Truth)", y=0.99)
    fig_e.tight_layout()

    out_e = REPO_ROOT / "outputs" / "object_knowledge_demo" / "target_knowledge_error.png"
    if plot_mode in ("save", "both"):
        out_e.parent.mkdir(parents=True, exist_ok=True)
        fig_e.savefig(out_e, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_e)

    return {
        "plot_mode": plot_mode,
        "truth_vs_knowledge_plot": str(out) if plot_mode in ("save", "both") else "",
        "knowledge_error_plot": str(out_e) if plot_mode in ("save", "both") else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for cross-object knowledge using configurable EKF tracking.")
    parser.add_argument(
        "--plot-mode",
        choices=["interactive", "save", "both"],
        default="interactive",
        help="Plot behavior; interactive is default.",
    )
    args = parser.parse_args()
    outputs = run_demo(plot_mode=args.plot_mode)
    print("Generated outputs:")
    for k, v in outputs.items():
        if v:
            print(f"  {k}: {v}")
