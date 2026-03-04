from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.control.orbit import HCWLQRController
from sim.core.models import StateBelief
from sim.utils.frames import ric_rect_to_curv


def _hcw_derivative(x_rect: np.ndarray, n_rad_s: float, accel_ric_km_s2: np.ndarray) -> np.ndarray:
    x_r, x_i, x_c, x_rdot, x_idot, x_cdot = x_rect
    ux, uy, uz = accel_ric_km_s2
    return np.array(
        [
            x_rdot,
            x_idot,
            x_cdot,
            3.0 * n_rad_s * n_rad_s * x_r + 2.0 * n_rad_s * x_idot + ux,
            -2.0 * n_rad_s * x_rdot + uy,
            -(n_rad_s * n_rad_s) * x_c + uz,
        ],
        dtype=float,
    )


def _rk4_step(x_rect: np.ndarray, dt_s: float, n_rad_s: float, accel_ric_km_s2: np.ndarray) -> np.ndarray:
    k1 = _hcw_derivative(x_rect, n_rad_s, accel_ric_km_s2)
    k2 = _hcw_derivative(x_rect + 0.5 * dt_s * k1, n_rad_s, accel_ric_km_s2)
    k3 = _hcw_derivative(x_rect + 0.5 * dt_s * k2, n_rad_s, accel_ric_km_s2)
    k4 = _hcw_derivative(x_rect + dt_s * k3, n_rad_s, accel_ric_km_s2)
    return x_rect + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HCW LQR convergence demo (curvilinear RIC input -> ECI thrust output).")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=3600.0, help="Simulation duration in seconds.")
    args = parser.parse_args()

    dt_s = float(args.dt)
    n_rad_s = 0.0011
    max_accel_km_s2 = 5e-5

    # Tuned for reliable convergence in the demo envelope (starts within 10 km).
    q_weights = np.array([8.66, 8.66, 8.66, 1.33, 1.33, 1.33]) * 1e3
    r_weights = np.ones(3) * 1.94e13

    ctrl = HCWLQRController(
        mean_motion_rad_s=n_rad_s,
        max_accel_km_s2=max_accel_km_s2,
        design_dt_s=dt_s,
        ric_curv_state_slice=(0, 6),
        chief_eci_state_slice=(6, 12),
        q_weights=q_weights,
        r_weights=r_weights,
    )

    # Demo initial condition inside requested 10 km envelope.
    x_rel_rect = np.array([8.0, -3.0, 2.0, 0.0010, -0.0006, 0.0004], dtype=float)
    chief_eci = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=float)
    r0 = float(np.linalg.norm(chief_eci[:3]))

    steps = int(np.ceil(float(args.duration) / dt_s))
    t = np.arange(steps + 1, dtype=float) * dt_s
    pos_norm = np.zeros(steps + 1)
    vel_norm = np.zeros(steps + 1)
    accel_norm = np.zeros(steps + 1)
    accel_ric_hist = np.zeros((steps + 1, 3))
    accel_eci_hist = np.zeros((steps + 1, 3))
    pos_norm[0] = np.linalg.norm(x_rel_rect[:3])
    vel_norm[0] = np.linalg.norm(x_rel_rect[3:])

    for k in range(steps):
        x_rel_curv = ric_rect_to_curv(x_rel_rect, r0_km=r0)
        belief = StateBelief(state=np.hstack((x_rel_curv, chief_eci)), covariance=np.eye(12), last_update_t_s=t[k])
        cmd = ctrl.act(belief, t_s=t[k], budget_ms=1.0)
        a_ric = np.array(cmd.mode_flags["accel_ric_km_s2"], dtype=float)
        a_eci = np.array(cmd.thrust_eci_km_s2, dtype=float)

        x_rel_rect = _rk4_step(x_rel_rect, dt_s, n_rad_s, a_ric)

        pos_norm[k + 1] = np.linalg.norm(x_rel_rect[:3])
        vel_norm[k + 1] = np.linalg.norm(x_rel_rect[3:])
        accel_ric_hist[k + 1, :] = a_ric
        accel_eci_hist[k + 1, :] = a_eci
        accel_norm[k + 1] = np.linalg.norm(a_eci)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, pos_norm)
    axes[0].set_ylabel("||r_rel|| (km)")
    axes[0].set_title("HCW LQR Relative Position Norm")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, vel_norm)
    axes[1].set_ylabel("||v_rel|| (km/s)")
    axes[1].set_title("HCW LQR Relative Velocity Norm")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, accel_norm)
    axes[2].axhline(max_accel_km_s2, linestyle="--")
    axes[2].set_ylabel("||a_cmd|| (km/s^2)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Commanded Acceleration Norm (ECI)")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()

    if args.plot_mode in ("save", "both"):
        outdir = REPO_ROOT / "outputs" / "orbit_hcw_lqr_demo"
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "convergence.png", dpi=150)
    if args.plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    print("HCW LQR Demo complete")
    print("Q diag:", q_weights.tolist())
    print("R diag:", r_weights.tolist())
    print("initial ||r_rel|| km:", float(pos_norm[0]))
    print("initial ||v_rel|| km/s:", float(vel_norm[0]))
    print("final ||r_rel|| km:", float(pos_norm[-1]))
    print("final ||v_rel|| km/s:", float(vel_norm[-1]))
