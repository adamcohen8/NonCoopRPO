from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.control.orbit import HCWRelativeOrbitMPCController
from sim.core.models import StateBelief
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.utils.frames import eci_relative_to_ric_rect, ric_rect_to_curv


def _relative_rect_ric(x_chaser_eci: np.ndarray, x_target_eci: np.ndarray) -> np.ndarray:
    return eci_relative_to_ric_rect(x_chaser_eci, x_target_eci)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relative orbit HCW-MPC demo (curvilinear RIC input -> ECI thrust output, HCW prediction)."
    )
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=5000.0, help="Simulation duration in seconds.")
    parser.add_argument("--budget-ms", type=float, default=50.0, help="Controller compute budget per step in milliseconds.")
    parser.add_argument("--horizon-time", type=float, default=500.0, help="Prediction horizon in seconds.")
    parser.add_argument("--model-dt", type=float, default=None, help="Optional fixed model step for internal horizon discretization.")
    parser.add_argument("--max-horizon-steps", type=int, default=400, help="Cap on internal horizon steps.")
    parser.add_argument("--iterations", type=int, default=50, help="Max optimizer iterations per control call.")
    parser.add_argument(
        "--mean-motion",
        type=float,
        default=None,
        help="Optional fixed mean motion [rad/s] for HCW model. Default computes from chief radius.",
    )
    args = parser.parse_args()

    dt_s = float(args.dt)
    mu_km3_s2 = 398600.4418
    max_accel_km_s2 = 5e-5

    ctrl = HCWRelativeOrbitMPCController(
        max_accel_km_s2=max_accel_km_s2,
        horizon_time_s=float(args.horizon_time),
        default_model_dt_s=dt_s,
        model_dt_s=(None if args.model_dt is None else float(args.model_dt)),
        max_horizon_steps=int(args.max_horizon_steps),
        mu_km3_s2=mu_km3_s2,
        mean_motion_rad_s=(None if args.mean_motion is None else float(args.mean_motion)),
        max_iterations=int(args.iterations),
        ric_curv_state_slice=(0, 6),
        chief_eci_state_slice=(6, 12),
    )

    # Circular target orbit with an initial relative offset in RIC.
    r0_km = 7000.0
    v0_km_s = float(np.sqrt(mu_km3_s2 / r0_km))
    x_target = np.array([r0_km, 0.0, 0.0, 0.0, v0_km_s, 0.0], dtype=float)
    x_rel0_rect = np.array([8.0, -3.0, 2.0, 0.0010, -0.0006, 0.0004], dtype=float)
    c_ir0 = ric_dcm_ir_from_rv(x_target[:3], x_target[3:])
    x_chaser = np.hstack(
        (
            x_target[:3] + c_ir0 @ x_rel0_rect[:3],
            x_target[3:] + c_ir0 @ x_rel0_rect[3:],
        )
    )
    x_target_passive = np.array(x_target, dtype=float)
    x_chaser_passive = np.array(x_chaser, dtype=float)

    steps = int(np.ceil(float(args.duration) / dt_s))
    t = np.arange(steps + 1, dtype=float) * dt_s
    pos_norm = np.zeros(steps + 1)
    vel_norm = np.zeros(steps + 1)
    pos_norm_passive = np.zeros(steps + 1)
    vel_norm_passive = np.zeros(steps + 1)
    accel_norm = np.zeros(steps + 1)
    accel_ric_hist = np.zeros((steps + 1, 3))
    accel_eci_hist = np.zeros((steps + 1, 3))
    rel_hist = np.zeros((steps + 1, 6))
    rel_passive_hist = np.zeros((steps + 1, 6))
    iter_hist = np.zeros(steps + 1)
    cost_hist = np.zeros(steps + 1)
    eval_hist = np.zeros(steps + 1)
    align_hist = np.zeros(steps + 1)
    cumulative_dv = np.zeros(steps + 1)

    x_rel_rect = _relative_rect_ric(x_chaser_eci=x_chaser, x_target_eci=x_target)
    x_rel_passive_rect = _relative_rect_ric(x_chaser_eci=x_chaser_passive, x_target_eci=x_target_passive)
    rel_hist[0, :] = x_rel_rect
    rel_passive_hist[0, :] = x_rel_passive_rect
    pos_norm[0] = np.linalg.norm(x_rel_rect[:3])
    vel_norm[0] = np.linalg.norm(x_rel_rect[3:])
    pos_norm_passive[0] = np.linalg.norm(x_rel_passive_rect[:3])
    vel_norm_passive[0] = np.linalg.norm(x_rel_passive_rect[3:])

    for k in range(steps):
        x_rel_rect = _relative_rect_ric(x_chaser_eci=x_chaser, x_target_eci=x_target)
        x_rel_curv = ric_rect_to_curv(x_rel_rect, r0_km=float(np.linalg.norm(x_target[:3])))
        belief = StateBelief(
            state=np.hstack((x_rel_curv, x_target)),
            covariance=np.eye(12),
            last_update_t_s=t[k],
        )
        cmd = ctrl.act(belief, t_s=t[k], budget_ms=float(args.budget_ms))
        a_eci = np.array(cmd.thrust_eci_km_s2, dtype=float)
        a_ric = np.array(cmd.mode_flags.get("accel_ric_km_s2", np.zeros(3)), dtype=float)
        rel_pos_norm = float(np.linalg.norm(x_rel_rect[:3]))
        if rel_pos_norm > 1e-12 and float(np.linalg.norm(a_ric)) > 0.0:
            align_hist[k + 1] = float(np.dot(a_ric, -x_rel_rect[:3] / rel_pos_norm) / max(np.linalg.norm(a_ric), 1e-12))
        else:
            align_hist[k + 1] = 0.0

        x_target = propagate_two_body_rk4(
            x_eci=x_target,
            dt_s=dt_s,
            mu_km3_s2=mu_km3_s2,
            accel_cmd_eci_km_s2=np.zeros(3, dtype=float),
        )
        x_chaser = propagate_two_body_rk4(
            x_eci=x_chaser,
            dt_s=dt_s,
            mu_km3_s2=mu_km3_s2,
            accel_cmd_eci_km_s2=a_eci,
        )
        x_target_passive = propagate_two_body_rk4(
            x_eci=x_target_passive,
            dt_s=dt_s,
            mu_km3_s2=mu_km3_s2,
            accel_cmd_eci_km_s2=np.zeros(3, dtype=float),
        )
        x_chaser_passive = propagate_two_body_rk4(
            x_eci=x_chaser_passive,
            dt_s=dt_s,
            mu_km3_s2=mu_km3_s2,
            accel_cmd_eci_km_s2=np.zeros(3, dtype=float),
        )

        x_rel_next = _relative_rect_ric(x_chaser_eci=x_chaser, x_target_eci=x_target)
        x_rel_passive_next = _relative_rect_ric(x_chaser_eci=x_chaser_passive, x_target_eci=x_target_passive)
        rel_hist[k + 1, :] = x_rel_next
        rel_passive_hist[k + 1, :] = x_rel_passive_next
        pos_norm[k + 1] = np.linalg.norm(x_rel_next[:3])
        vel_norm[k + 1] = np.linalg.norm(x_rel_next[3:])
        pos_norm_passive[k + 1] = np.linalg.norm(x_rel_passive_next[:3])
        vel_norm_passive[k + 1] = np.linalg.norm(x_rel_passive_next[3:])
        accel_ric_hist[k + 1, :] = a_ric
        accel_eci_hist[k + 1, :] = a_eci
        accel_norm[k + 1] = np.linalg.norm(a_eci)
        cumulative_dv[k + 1] = cumulative_dv[k] + accel_norm[k + 1] * dt_s
        iter_hist[k + 1] = float(cmd.mode_flags.get("iterations", 0.0))
        cost_hist[k + 1] = float(cmd.mode_flags.get("cost", 0.0))
        eval_hist[k + 1] = float(cmd.mode_flags.get("cost_evals", 0.0))

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(t, pos_norm, label="controlled")
    axes[0].set_ylabel("||r_rel|| (km)")
    axes[0].set_title("Relative Orbit HCW-MPC: Relative Position Norm")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t, vel_norm, label="controlled")
    axes[1].set_ylabel("||v_rel|| (km/s)")
    axes[1].set_title("Relative Velocity Norm")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(t, accel_norm)
    axes[2].axhline(max_accel_km_s2, linestyle="--")
    axes[2].set_ylabel("||a_cmd|| (km/s^2)")
    axes[2].set_title("Commanded Acceleration Norm (ECI)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, iter_hist, label="iterations")
    axes[3].set_ylabel("Solve iters")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("MPC Solver Iterations")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")
    fig.tight_layout()

    fig_rel, axes_rel = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    pos_labels = ["R", "I", "C"]
    vel_labels = ["dR", "dI", "dC"]
    for i in range(3):
        axes_rel[i, 0].plot(t, rel_hist[:, i], label="controlled")
        axes_rel[i, 0].set_ylabel(f"{pos_labels[i]} (km)")
        axes_rel[i, 0].grid(True, alpha=0.3)
        axes_rel[i, 1].plot(t, rel_hist[:, i + 3], label="controlled")
        axes_rel[i, 1].set_ylabel(f"{vel_labels[i]} (km/s)")
        axes_rel[i, 1].grid(True, alpha=0.3)
    axes_rel[0, 0].set_title("RIC Position Components")
    axes_rel[0, 1].set_title("RIC Velocity Components")
    axes_rel[0, 0].legend(loc="best")
    axes_rel[0, 1].legend(loc="best")
    axes_rel[2, 0].set_xlabel("Time (s)")
    axes_rel[2, 1].set_xlabel("Time (s)")
    fig_rel.tight_layout()

    fig_cmd, axes_cmd = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    axes_cmd[0].plot(t, accel_ric_hist[:, 0], label="a_R")
    axes_cmd[0].plot(t, accel_ric_hist[:, 1], label="a_I")
    axes_cmd[0].plot(t, accel_ric_hist[:, 2], label="a_C")
    axes_cmd[0].set_ylabel("RIC accel")
    axes_cmd[0].set_title("Command Components in RIC")
    axes_cmd[0].grid(True, alpha=0.3)
    axes_cmd[0].legend(loc="best")

    axes_cmd[1].plot(t, cumulative_dv)
    axes_cmd[1].set_ylabel("cum dv (km/s)")
    axes_cmd[1].set_title("Integrated Command Magnitude")
    axes_cmd[1].grid(True, alpha=0.3)

    axes_cmd[2].plot(t, cost_hist, label="cost")
    axes_cmd[2].plot(t, eval_hist, label="cost evals")
    axes_cmd[2].set_ylabel("optimizer")
    axes_cmd[2].set_title("MPC Internal Diagnostics")
    axes_cmd[2].grid(True, alpha=0.3)
    axes_cmd[2].legend(loc="best")

    axes_cmd[3].plot(t, align_hist)
    axes_cmd[3].axhline(0.0, linestyle="--")
    axes_cmd[3].set_ylabel("cos(angle)")
    axes_cmd[3].set_xlabel("Time (s)")
    axes_cmd[3].set_title("Alignment of a_cmd with -r_rel in RIC")
    axes_cmd[3].grid(True, alpha=0.3)
    fig_cmd.tight_layout()

    fig_proj, axes_proj = plt.subplots(1, 3, figsize=(15, 4.8))
    axes_proj[0].plot(rel_hist[:, 1], rel_hist[:, 0], label="controlled")
    axes_proj[0].set_xlabel("In-track I (km)")
    axes_proj[0].set_ylabel("Radial R (km)")
    axes_proj[0].set_title("RIC: I vs R")
    axes_proj[0].grid(True, alpha=0.3)
    axes_proj[0].legend(loc="best")

    axes_proj[1].plot(rel_hist[:, 2], rel_hist[:, 0], label="controlled")
    axes_proj[1].set_xlabel("Cross-track C (km)")
    axes_proj[1].set_ylabel("Radial R (km)")
    axes_proj[1].set_title("RIC: C vs R")
    axes_proj[1].grid(True, alpha=0.3)
    axes_proj[1].legend(loc="best")

    axes_proj[2].plot(rel_hist[:, 1], rel_hist[:, 2], label="controlled")
    axes_proj[2].set_xlabel("In-track I (km)")
    axes_proj[2].set_ylabel("Cross-track C (km)")
    axes_proj[2].set_title("RIC: I vs C")
    axes_proj[2].grid(True, alpha=0.3)
    axes_proj[2].legend(loc="best")
    fig_proj.tight_layout()

    if args.plot_mode in ("save", "both"):
        outdir = REPO_ROOT / "outputs" / "orbit_relative_hcw_mpc_demo"
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "convergence.png", dpi=150)
        fig_rel.savefig(outdir / "relative_components.png", dpi=150)
        fig_cmd.savefig(outdir / "control_diagnostics.png", dpi=150)
        fig_proj.savefig(outdir / "ric_2d_projections.png", dpi=150)
    if args.plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)
    plt.close(fig_rel)
    plt.close(fig_cmd)
    plt.close(fig_proj)

    print("Relative Orbit HCW-MPC Demo complete")
    print("horizon_time_s:", float(args.horizon_time))
    print("model_dt_s:", "auto_from_call_dt" if args.model_dt is None else float(args.model_dt))
    print("max_horizon_steps:", int(args.max_horizon_steps))
    print("max_iterations:", int(args.iterations))
    print("budget_ms:", float(args.budget_ms))
    print("mean_motion_rad_s:", "auto" if args.mean_motion is None else float(args.mean_motion))
    print("initial ||r_rel|| km:", float(pos_norm[0]))
    print("initial ||v_rel|| km/s:", float(vel_norm[0]))
    print("final ||r_rel|| km:", float(pos_norm[-1]))
    print("final ||v_rel|| km/s:", float(vel_norm[-1]))
    print("final passive ||r_rel|| km:", float(pos_norm_passive[-1]))
    print("final passive ||v_rel|| km/s:", float(vel_norm_passive[-1]))
    print("peak ||a_cmd|| km/s^2:", float(np.max(accel_norm)))
    print("integrated command dv (km/s):", float(cumulative_dv[-1]))
