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
from sim.actuators.attitude import ReactionWheelLimits
from sim.control.attitude import RICDetumblePDController, ReactionWheelPDController
from sim.core.models import StateBelief
from sim.utils.quaternion import dcm_to_quaternion_bn


def _rot_x(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, ca, sa], [0.0, -sa, ca]])


def _rot_y(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])


def _rot_z(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])


def run_demo(
    plot_mode: str = "interactive",
    dt_s: float = 1.0,
    duration_s: float = 1800.0,
    attitude_dt_s: float = 0.01,
    wheel_scale: float = 80.0,
    kp: np.ndarray | None = None,
    kd: np.ndarray | None = None,
    w0_body_rad_s: np.ndarray | None = None,
) -> dict[str, str]:
    steps = int(np.ceil(duration_s / dt_s))

    c_bn0 = _rot_z(np.deg2rad(-10.0)) @ _rot_y(np.deg2rad(15.0)) @ _rot_x(np.deg2rad(-20.0))
    q0 = dcm_to_quaternion_bn(c_bn0)
    w0 = np.array([0.07, 0.05, -0.06], dtype=float) if w0_body_rad_s is None else np.array(w0_body_rad_s, dtype=float).reshape(3)

    kp_vec = np.array([0.0, 0.0, 0.0], dtype=float) if kp is None else np.array(kp, dtype=float).reshape(3)
    kd_vec = np.array([8.0, 8.0, 8.0], dtype=float) if kd is None else np.array(kd, dtype=float).reshape(3)

    sat = build_sim_object_from_presets(
        object_id="sat_pd_ric_detumble_demo",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=False,
        enable_attitude_knowledge=True,
        attitude_quat_bn=q0,
        angular_rate_body_rad_s=w0,
        attitude_substep_s=attitude_dt_s,
    )

    wheel_axes = np.vstack([w.axis_body for w in BASIC_REACTION_WHEEL_TRIAD.wheels])
    wheel_limits = np.array([w.max_torque_nm for w in BASIC_REACTION_WHEEL_TRIAD.wheels], dtype=float) * float(wheel_scale)
    pd = ReactionWheelPDController(
        wheel_axes_body=wheel_axes,
        wheel_torque_limits_nm=wheel_limits,
        desired_attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        desired_rate_body_rad_s=np.zeros(3),
        kp=kp_vec,
        kd=kd_vec,
    )
    ctrl = RICDetumblePDController(pd=pd, rate_only=True, lock_reference_on_first_call=True)
    sat.controller = ctrl

    if hasattr(sat.actuator, "attitude") and hasattr(sat.actuator.attitude, "reaction_wheels"):
        rw = sat.actuator.attitude.reaction_wheels
        if rw is not None:
            sat.actuator.attitude.reaction_wheels = ReactionWheelLimits(
                max_torque_nm=rw.max_torque_nm * float(wheel_scale),
                max_momentum_nms=rw.max_momentum_nms * float(wheel_scale),
            )

    t = np.zeros(steps + 1)
    q_hist = np.zeros((steps + 1, 4))
    w_hist = np.zeros((steps + 1, 3))
    tau_hist = np.zeros((steps + 1, 3))
    w_norm = np.zeros(steps + 1)
    h_rw = np.zeros((steps + 1, 3))
    h_rw_norm = np.zeros(steps + 1)
    sat_rw = np.zeros(steps + 1)
    q_hist[0, :] = sat.truth.attitude_quat_bn
    w_hist[0, :] = sat.truth.angular_rate_body_rad_s
    w_norm[0] = float(np.linalg.norm(w_hist[0, :]))
    if hasattr(sat.actuator, "wheel_momentum_nms"):
        h_rw[0, :] = np.array(sat.actuator.wheel_momentum_nms, dtype=float)
        h_rw_norm[0] = float(np.linalg.norm(h_rw[0, :]))

    for k in range(steps):
        t_now = sat.truth.t_s
        meas = sat.sensor.measure(sat.truth, env={}, t_s=t_now + dt_s)
        sat.belief = sat.estimator.update(sat.belief, meas, t_s=t_now + dt_s)

        belief_att = StateBelief(
            state=np.hstack(
                (
                    sat.truth.position_eci_km,
                    sat.truth.velocity_eci_km_s,
                    sat.truth.attitude_quat_bn,
                    sat.truth.angular_rate_body_rad_s,
                )
            ),
            covariance=np.eye(13),
            last_update_t_s=t_now,
        )
        cmd = ctrl.act(belief_att, t_s=t_now, budget_ms=1.0)
        applied = sat.actuator.apply(cmd, sat.limits, dt_s)
        sat.truth = sat.dynamics.step(sat.truth, applied, env={}, dt_s=dt_s)

        t[k + 1] = sat.truth.t_s
        q_hist[k + 1, :] = sat.truth.attitude_quat_bn
        w_hist[k + 1, :] = sat.truth.angular_rate_body_rad_s
        tau_hist[k + 1, :] = applied.torque_body_nm
        w_norm[k + 1] = float(np.linalg.norm(w_hist[k + 1, :]))
        if hasattr(sat.actuator, "wheel_momentum_nms"):
            h_rw[k + 1, :] = np.array(sat.actuator.wheel_momentum_nms, dtype=float)
            h_rw_norm[k + 1] = float(np.linalg.norm(h_rw[k + 1, :]))
            rw_lim = sat.actuator.attitude.reaction_wheels if hasattr(sat.actuator, "attitude") else None
            if rw_lim is not None:
                sat_rw[k + 1] = float(np.any(np.abs(h_rw[k + 1, :]) >= np.array(rw_lim.max_momentum_nms, dtype=float) - 1e-12))

    outdir = REPO_ROOT / "outputs" / "attitude_pd_ric_detumble_demo"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    fig_q, axes_q = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    labels_q = ["q0", "q1", "q2", "q3"]
    for i in range(4):
        axes_q[i].plot(t, q_hist[:, i])
        axes_q[i].grid(True, alpha=0.3)
        axes_q[i].set_ylabel(labels_q[i])
    axes_q[0].set_title("RIC Detumble PD: Quaternion")
    axes_q[-1].set_xlabel("Time (s)")
    fig_q.tight_layout()
    q_path = outdir / "ric_detumble_quaternion.png"
    if plot_mode in ("save", "both"):
        fig_q.savefig(q_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_q)

    fig_w, axes_w = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    labels_w = ["wx", "wy", "wz"]
    for i in range(3):
        axes_w[i].plot(t, w_hist[:, i])
        axes_w[i].grid(True, alpha=0.3)
        axes_w[i].set_ylabel("rad/s")
        axes_w[i].set_title(labels_w[i])
    axes_w[3].plot(t, w_norm)
    axes_w[3].grid(True, alpha=0.3)
    axes_w[3].set_ylabel("|w|")
    axes_w[3].set_xlabel("Time (s)")
    axes_w[3].set_title("Body-Rate Norm")
    fig_w.tight_layout()
    w_path = outdir / "ric_detumble_rates.png"
    if plot_mode in ("save", "both"):
        fig_w.savefig(w_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_w)

    fig_t, axes_t = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    labels_t = ["tau_x", "tau_y", "tau_z"]
    for i in range(3):
        axes_t[i].plot(t, tau_hist[:, i])
        axes_t[i].set_ylabel("N m")
        axes_t[i].grid(True, alpha=0.3)
        axes_t[i].set_title(labels_t[i])
    axes_t[3].plot(t, h_rw_norm)
    axes_t[3].set_ylabel("N m s")
    axes_t[3].set_title("Wheel Momentum Norm")
    axes_t[3].grid(True, alpha=0.3)
    axes_t[4].plot(t, sat_rw)
    axes_t[4].set_ylabel("flag")
    axes_t[4].set_title("Wheel Saturation (any axis)")
    axes_t[4].set_xlabel("Time (s)")
    axes_t[4].grid(True, alpha=0.3)
    fig_t.tight_layout()
    tau_path = outdir / "ric_detumble_torque_and_error.png"
    if plot_mode in ("save", "both"):
        fig_t.savefig(tau_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_t)

    return {
        "plot_mode": plot_mode,
        "quat_plot": str(q_path) if plot_mode in ("save", "both") else "",
        "rates_plot": str(w_path) if plot_mode in ("save", "both") else "",
        "torque_error_plot": str(tau_path) if plot_mode in ("save", "both") else "",
        "final_rate_norm_rad_s": str(float(w_norm[-1])),
        "final_wheel_momentum_norm_nms": str(float(h_rw_norm[-1])),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone test demo for RIC detumble PD attitude controller with reaction wheels.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=1800.0)
    parser.add_argument("--attitude-dt", type=float, default=0.01)
    parser.add_argument("--wheel-scale", type=float, default=80.0)
    parser.add_argument("--kp", type=float, default=0.0, help="Scalar Kp for all 3 PD axes.")
    parser.add_argument("--kd", type=float, default=8.0, help="Scalar Kd for all 3 PD axes.")
    parser.add_argument("--w0-x", type=float, default=0.07, help="Initial body rate wx (rad/s).")
    parser.add_argument("--w0-y", type=float, default=0.05, help="Initial body rate wy (rad/s).")
    parser.add_argument("--w0-z", type=float, default=-0.06, help="Initial body rate wz (rad/s).")
    args = parser.parse_args()

    outputs = run_demo(
        plot_mode=args.plot_mode,
        dt_s=float(args.dt),
        duration_s=float(args.duration),
        attitude_dt_s=float(args.attitude_dt),
        wheel_scale=float(args.wheel_scale),
        kp=np.array([float(args.kp)] * 3),
        kd=np.array([float(args.kd)] * 3),
        w0_body_rad_s=np.array([float(args.w0_x), float(args.w0_y), float(args.w0_z)]),
    )
    print("RIC detumble PD demo outputs:")
    for k, v in outputs.items():
        if v:
            print(f"  {k}: {v}")
