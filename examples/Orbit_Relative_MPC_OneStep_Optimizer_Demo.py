from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.control.orbit import RelativeOrbitMPCController
from sim.core.models import StateBelief


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-step optimizer diagnostic for RelativeOrbitMPCController."
    )
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--budget-ms", type=float, default=2000.0)
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--max-accel", type=float, default=5e-5)
    args = parser.parse_args()

    ctrl = RelativeOrbitMPCController(
        max_accel_km_s2=float(args.max_accel),
        horizon_steps=int(args.horizon_steps),
        step_dt_s=float(args.dt),
        max_iterations=int(args.iterations),
    )

    # Representative nonzero relative state: this should produce active control.
    state = np.zeros(12, dtype=float)
    state[0:6] = np.array([8.0, -3.0, 2.0, 0.0010, -0.0006, 0.0004], dtype=float)
    state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.546049108166282, 0.0], dtype=float)
    belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)

    cmd = ctrl.act(belief, t_s=0.0, budget_ms=float(args.budget_ms))
    mode = cmd.mode_flags
    grad_hist = np.array(mode.get("grad_norm_history", []), dtype=float)
    alpha_hist = np.array(mode.get("accepted_alpha_history", []), dtype=float)
    cost_hist = np.array(mode.get("cost_history", []), dtype=float)
    first_u_hist = np.array(mode.get("first_u_eci_history", []), dtype=float)
    k_grad = np.arange(1, grad_hist.size + 1, dtype=int)
    k_cost = np.arange(0, cost_hist.size, dtype=int)
    k_alpha = np.arange(1, alpha_hist.size + 1, dtype=int)
    k_u = np.arange(0, first_u_hist.shape[0], dtype=int)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    axes[0].plot(k_cost, cost_hist, marker="o")
    axes[0].set_title("Optimizer Cost History")
    axes[0].set_xlabel("Accepted Step Index")
    axes[0].set_ylabel("Cost")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_grad, grad_hist, marker="o")
    axes[1].set_title("Gradient Norm per Iteration")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("||grad||")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(k_alpha, alpha_hist, marker="o")
    axes[2].set_title("Accepted Line-Search Step Size")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("alpha")
    axes[2].grid(True, alpha=0.3)

    if first_u_hist.size > 0:
        axes[3].plot(k_u, first_u_hist[:, 0], marker="o", label="u0_x")
        axes[3].plot(k_u, first_u_hist[:, 1], marker="o", label="u0_y")
        axes[3].plot(k_u, first_u_hist[:, 2], marker="o", label="u0_z")
        axes[3].plot(k_u, np.linalg.norm(first_u_hist, axis=1), marker="o", label="||u0||")
    axes[3].set_title("First Thrust Command per Accepted Step (ECI)")
    axes[3].set_xlabel("Accepted Step Index")
    axes[3].set_ylabel("km/s^2")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")
    fig.tight_layout()

    if args.plot_mode in ("save", "both"):
        outdir = REPO_ROOT / "outputs" / "orbit_relative_mpc_one_step_optimizer_demo"
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "optimizer_trace.png", dpi=150)
    if args.plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    print("Relative Orbit MPC One-Step Optimizer Demo complete")
    print("budget_ms:", float(args.budget_ms))
    print("horizon_steps:", int(args.horizon_steps))
    print("max_iterations:", int(args.iterations))
    print("gradient_method:", str(mode.get("gradient_method", "unknown")))
    print("returned iterations:", int(mode.get("iterations", -1)))
    print("termination_reason:", str(mode.get("termination_reason", "unknown")))
    print("deadline_hit:", bool(mode.get("deadline_hit", False)))
    print("solve_time_ms:", float(mode.get("solve_time_ms", float("nan"))))
    print("final_trust_step_km_s2:", float(mode.get("final_trust_step_km_s2", float("nan"))))
    print("cost_evals:", int(mode.get("cost_evals", -1)))
    print("cost_start:", float(cost_hist[0]) if cost_hist.size > 0 else float("nan"))
    print("cost_end:", float(cost_hist[-1]) if cost_hist.size > 0 else float("nan"))
    print("thrust_eci_km_s2:", np.array(cmd.thrust_eci_km_s2, dtype=float).tolist())
    print("thrust_norm_km_s2:", float(np.linalg.norm(np.array(cmd.thrust_eci_km_s2, dtype=float))))
