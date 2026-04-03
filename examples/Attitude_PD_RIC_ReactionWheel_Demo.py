from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.presets import BASIC_REACTION_WHEEL_TRIAD, BASIC_SATELLITE, build_sim_object_from_presets
from sim.actuators.attitude import ReactionWheelLimits
from sim.control.attitude import RICFramePDController, ReactionWheelPDController
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
    wheel_scale: float = 1.0,
    kp: np.ndarray | None = None,
    kd: np.ndarray | None = None,
    yaw_r_rad: float = 0.0,
    roll_i_rad: float = 0.0,
    pitch_c_rad: float = 0.0,
) -> dict[str, str]:
    steps = int(np.ceil(duration_s / dt_s))

    c_bn0 = _rot_z(np.deg2rad(20.0)) @ _rot_y(np.deg2rad(-15.0)) @ _rot_x(np.deg2rad(25.0))
    q0 = dcm_to_quaternion_bn(c_bn0)
    w0 = np.array([0.0, -0.0, 0.0], dtype=float)
    kp_vec = np.array([0.2, 0.2, 0.2], dtype=float) if kp is None else np.array(kp, dtype=float).reshape(3)
    kd_vec = np.array([1.0, 1.0, 1.0], dtype=float) if kd is None else np.array(kd, dtype=float).reshape(3)

    sat = build_sim_object_from_presets(
        object_id="sat_pd_ric_demo",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=True,
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
    ctrl = RICFramePDController(pd=pd)
    ctrl.set_desired_ric_state(yaw_r_rad=yaw_r_rad, roll_i_rad=roll_i_rad, pitch_c_rad=pitch_c_rad, w_ric_rad_s=np.zeros(3))
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
    qerr_deg = np.zeros(steps + 1)
    q_hist[0, :] = sat.truth.attitude_quat_bn
    w_hist[0, :] = sat.truth.angular_rate_body_rad_s

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
        qerr_deg[k + 1] = float(cmd.mode_flags.get("attitude_error_deg", np.nan))

    outdir = REPO_ROOT / "outputs" / "attitude_pd_ric_demo"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    fig_q, axes_q = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    labels_q = ["q0", "q1", "q2", "q3"]
    for i in range(4):
        axes_q[i].plot(t, q_hist[:, i], label=labels_q[i])
        axes_q[i].grid(True, alpha=0.3)
        axes_q[i].set_ylabel(labels_q[i])
    axes_q[0].set_title("RIC-Frame Reaction-Wheel PD: Quaternion")
    axes_q[-1].set_xlabel("Time (s)")
    fig_q.tight_layout()
    q_path = outdir / "pd_ric_quaternion.png"
    if plot_mode in ("save", "both"):
        fig_q.savefig(q_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_q)

    fig_w, axes_w = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels_w = ["wx", "wy", "wz"]
    for i in range(3):
        axes_w[i].plot(t, w_hist[:, i], label=labels_w[i])
        axes_w[i].grid(True, alpha=0.3)
        axes_w[i].set_ylabel("rad/s")
    axes_w[0].set_title("RIC-Frame Reaction-Wheel PD: Body Rates")
    axes_w[-1].set_xlabel("Time (s)")
    fig_w.tight_layout()
    w_path = outdir / "pd_ric_rates.png"
    if plot_mode in ("save", "both"):
        fig_w.savefig(w_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_w)

    fig_t, axes_t = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    labels_t = ["tau_x", "tau_y", "tau_z"]
    for i in range(3):
        axes_t[i].plot(t, tau_hist[:, i])
        axes_t[i].set_ylabel("N m")
        axes_t[i].grid(True, alpha=0.3)
        axes_t[i].set_title(labels_t[i])
    axes_t[3].plot(t, qerr_deg)
    axes_t[3].set_ylabel("deg")
    axes_t[3].set_title("RIC Attitude Error Angle")
    axes_t[3].set_xlabel("Time (s)")
    axes_t[3].grid(True, alpha=0.3)
    fig_t.tight_layout()
    tau_path = outdir / "pd_ric_torque_and_error.png"
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
        "final_attitude_error_deg": str(float(qerr_deg[-1])),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone test demo for 3-axis RIC-frame reaction-wheel PD attitude controller.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=1800.0)
    parser.add_argument("--attitude-dt", type=float, default=0.01)
    parser.add_argument("--wheel-scale", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=0.2, help="Scalar Kp for all 3 PD axes.")
    parser.add_argument("--kd", type=float, default=1.0, help="Scalar Kd for all 3 PD axes.")
    parser.add_argument("--yaw-rad", type=float, default=0.0, help="Desired RIC yaw (rad, about R-axis).")
    parser.add_argument("--roll-rad", type=float, default=0.0, help="Desired RIC roll (rad, about I-axis).")
    parser.add_argument("--pitch-rad", type=float, default=0.0, help="Desired RIC pitch (rad, about C-axis).")
    args = parser.parse_args()

    outputs = run_demo(
        plot_mode=args.plot_mode,
        dt_s=float(args.dt),
        duration_s=float(args.duration),
        attitude_dt_s=float(args.attitude_dt),
        wheel_scale=float(args.wheel_scale),
        kp=np.array([float(args.kp)] * 3),
        kd=np.array([float(args.kd)] * 3),
        yaw_r_rad=float(args.yaw_rad),
        roll_i_rad=float(args.roll_rad),
        pitch_c_rad=float(args.pitch_rad),
    )
    print("RIC-PD demo outputs:")
    for k, v in outputs.items():
        if v:
            print(f"  {k}: {v}")
