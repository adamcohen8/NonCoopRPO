from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from presets import BASIC_REACTION_WHEEL_TRIAD, BASIC_SATELLITE, build_sim_object_from_presets
from sim.control.attitude import RICFrameLQRController, SmallAngleLQRController
from sim.core.models import Command, StateBelief
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.utils.plotting import plot_angular_rates, plot_attitude_ric
from sim.utils.quaternion import normalize_quaternion


def _quat_error_angle_deg(q_des_bn: np.ndarray, q_bn: np.ndarray) -> float:
    qd = normalize_quaternion(np.array(q_des_bn, dtype=float))
    q = normalize_quaternion(np.array(q_bn, dtype=float))
    qd_conj = np.array([qd[0], -qd[1], -qd[2], -qd[3]])
    qe = np.array(
        [
            qd_conj[0] * q[0] - qd_conj[1] * q[1] - qd_conj[2] * q[2] - qd_conj[3] * q[3],
            qd_conj[0] * q[1] + qd_conj[1] * q[0] + qd_conj[2] * q[3] - qd_conj[3] * q[2],
            qd_conj[0] * q[2] - qd_conj[1] * q[3] + qd_conj[2] * q[0] + qd_conj[3] * q[1],
            qd_conj[0] * q[3] + qd_conj[1] * q[2] - qd_conj[2] * q[1] + qd_conj[3] * q[0],
        ]
    )
    if qe[0] < 0.0:
        qe *= -1.0
    qe0 = float(np.clip(qe[0], -1.0, 1.0))
    return np.rad2deg(2.0 * np.arccos(qe0))


def main() -> None:
    parser = argparse.ArgumentParser(description="RIC-frame attitude LQR hold with all disturbance torques enabled.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--yaw-r-deg", type=float, default=0.0)
    parser.add_argument("--roll-i-deg", type=float, default=0.0)
    parser.add_argument("--pitch-c-deg", type=float, default=0.0)
    args = parser.parse_args()

    dt_s = float(args.dt)
    sat = build_sim_object_from_presets(
        object_id="sat_lqr_ricframe_hold",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=True,
        enable_attitude_knowledge=True,
        # Intentionally offset initial attitude.
        attitude_quat_bn=np.array([0.96592583, 0.0, 0.25881905, 0.0]),
        # Start with non-zero body rates for a tumble-like initial condition.
        angular_rate_body_rad_s=np.array([0.0002, -0.00015, 0.00012]),
    )

    wheel_axes = np.vstack([w.axis_body for w in BASIC_REACTION_WHEEL_TRIAD.wheels])
    wheel_limits = np.array([w.max_torque_nm for w in BASIC_REACTION_WHEEL_TRIAD.wheels], dtype=float)
    lqr = SmallAngleLQRController.robust_profile(
        inertia_kg_m2=BASIC_SATELLITE.inertia_kg_m2,
        wheel_axes_body=wheel_axes,
        wheel_torque_limits_nm=wheel_limits,
        design_dt_s=dt_s,
    )
    ric_lqr = RICFrameLQRController(
        lqr=lqr,
        desired_ric_euler_rad=np.deg2rad(np.array([args.yaw_r_deg, args.roll_i_deg, args.pitch_c_deg], dtype=float)),
        desired_ric_rate_rad_s=np.zeros(3),
    )

    r0 = float(np.linalg.norm(sat.truth.position_eci_km))
    orbit_period_s = 2.0 * np.pi * np.sqrt(r0**3 / EARTH_MU_KM3_S2)
    steps = int(np.ceil(orbit_period_s / dt_s)/4)

    truth_hist = np.zeros((steps + 1, 14))
    torque_hist = np.zeros((steps + 1, 3))
    err_hist_deg = np.zeros(steps + 1)
    t_hist = np.zeros(steps + 1)

    truth_hist[0, :] = np.hstack(
        (
            sat.truth.position_eci_km,
            sat.truth.velocity_eci_km_s,
            sat.truth.attitude_quat_bn,
            sat.truth.angular_rate_body_rad_s,
            np.array([sat.truth.mass_kg]),
        )
    )
    belief0 = StateBelief(
        state=np.hstack(
            (
                sat.truth.position_eci_km,
                sat.truth.velocity_eci_km_s,
                sat.truth.attitude_quat_bn,
                sat.truth.angular_rate_body_rad_s,
            )
        ),
        covariance=np.eye(13),
        last_update_t_s=sat.truth.t_s,
    )
    _ = ric_lqr.act(belief0, t_s=sat.truth.t_s, budget_ms=2.0)
    q_des0 = np.array(lqr.desired_attitude_quat_bn, dtype=float)
    err_hist_deg[0] = _quat_error_angle_deg(q_des0, sat.truth.attitude_quat_bn)

    for k in range(steps):
        truth = sat.truth
        belief = StateBelief(
            state=np.hstack(
                (
                    truth.position_eci_km,
                    truth.velocity_eci_km_s,
                    truth.attitude_quat_bn,
                    truth.angular_rate_body_rad_s,
                )
            ),
            covariance=np.eye(13),
            last_update_t_s=truth.t_s,
        )
        c_att = ric_lqr.act(belief, t_s=truth.t_s, budget_ms=2.0)
        q_des_bn = np.array(lqr.desired_attitude_quat_bn, dtype=float)
        cmd = Command(thrust_eci_km_s2=np.zeros(3), torque_body_nm=c_att.torque_body_nm, mode_flags={"mode": "lqr_ric"})
        applied = sat.actuator.apply(cmd, sat.limits, dt_s)
        sat.truth = sat.dynamics.step(truth, applied, env={}, dt_s=dt_s)

        t_hist[k + 1] = sat.truth.t_s
        torque_hist[k + 1, :] = applied.torque_body_nm
        err_hist_deg[k + 1] = _quat_error_angle_deg(q_des_bn, sat.truth.attitude_quat_bn)
        truth_hist[k + 1, :] = np.hstack(
            (
                sat.truth.position_eci_km,
                sat.truth.velocity_eci_km_s,
                sat.truth.attitude_quat_bn,
                sat.truth.angular_rate_body_rad_s,
                np.array([sat.truth.mass_kg]),
            )
        )

    outdir = REPO_ROOT / "outputs" / "attitude_lqr_ricframe_hold"
    if args.plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    plot_attitude_ric(
        t_hist,
        truth_hist,
        mode=args.plot_mode,
        out_path=str(outdir / "attitude_ric.png") if args.plot_mode in ("save", "both") else None,
    )
    plot_angular_rates(
        t_hist,
        truth_hist,
        mode=args.plot_mode,
        out_path=str(outdir / "angular_rates.png") if args.plot_mode in ("save", "both") else None,
    )

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t_hist, err_hist_deg)
    axes[0].set_ylabel("deg")
    axes[0].set_title("Quaternion Tracking Error (RIC-Referenced LQR)")
    axes[0].grid(True, alpha=0.3)
    for i in range(3):
        axes[i + 1].plot(t_hist, torque_hist[:, i])
        axes[i + 1].set_ylabel(f"Tau{i+1} (Nm)")
        axes[i + 1].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[1].set_title("Applied Body Torque")
    fig.tight_layout()

    if args.plot_mode in ("save", "both"):
        fig.savefig(outdir / "torque_and_error.png", dpi=150)
    if args.plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    print("Completed one-orbit RIC-frame LQR hold with disturbances enabled.")
    print("Initial body rates (rad/s):", [0.02, -0.015, 0.012])
    print("Initial attitude error (deg):", float(err_hist_deg[0]))
    print("Final attitude error (deg):", float(err_hist_deg[-1]))
    print("RMS attitude error (deg):", float(np.sqrt(np.mean(err_hist_deg**2))))


if __name__ == "__main__":
    main()
