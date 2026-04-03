from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.presets import BASIC_REACTION_WHEEL_TRIAD, BASIC_SATELLITE, build_sim_object_from_presets
from sim.control.attitude import ReactionWheelPDController
from sim.core.models import StateBelief
from sim.utils.quaternion import dcm_to_quaternion_bn, quaternion_to_dcm_bn


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


def _dcm_bn_from_euler_zyx_deg(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    y = np.deg2rad(float(yaw_deg))
    p = np.deg2rad(float(pitch_deg))
    r = np.deg2rad(float(roll_deg))
    return _rot_z(y) @ _rot_y(p) @ _rot_x(r)


def _box_vertices_body(lx_m: float, ly_m: float, lz_m: float) -> np.ndarray:
    hx = 0.5 * lx_m
    hy = 0.5 * ly_m
    hz = 0.5 * lz_m
    return np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=float,
    )


def _box_faces(vertices: np.ndarray) -> list[np.ndarray]:
    return [
        vertices[[0, 1, 2, 3]],
        vertices[[4, 5, 6, 7]],
        vertices[[0, 1, 5, 4]],
        vertices[[2, 3, 7, 6]],
        vertices[[1, 2, 6, 5]],
        vertices[[0, 3, 7, 4]],
    ]


def _quat_error_deg(q_target: np.ndarray, q_current: np.ndarray) -> float:
    qt = np.array(q_target, dtype=float)
    qc = np.array(q_current, dtype=float)
    qt /= max(np.linalg.norm(qt), 1e-12)
    qc /= max(np.linalg.norm(qc), 1e-12)
    qt_conj = np.array([qt[0], -qt[1], -qt[2], -qt[3]], dtype=float)
    qe = np.array(
        [
            qt_conj[0] * qc[0] - qt_conj[1] * qc[1] - qt_conj[2] * qc[2] - qt_conj[3] * qc[3],
            qt_conj[0] * qc[1] + qt_conj[1] * qc[0] + qt_conj[2] * qc[3] - qt_conj[3] * qc[2],
            qt_conj[0] * qc[2] - qt_conj[1] * qc[3] + qt_conj[2] * qc[0] + qt_conj[3] * qc[1],
            qt_conj[0] * qc[3] + qt_conj[1] * qc[2] - qt_conj[2] * qc[1] + qt_conj[3] * qc[0],
        ],
        dtype=float,
    )
    if qe[0] < 0.0:
        qe *= -1.0
    return float(np.rad2deg(2.0 * np.arccos(np.clip(qe[0], -1.0, 1.0))))


def _propagate_pd_slew(
    duration_s: float,
    dt_s: float,
    attitude_dt_s: float,
    q_init_bn: np.ndarray,
    q_target_bn: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    wheel_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sat = build_sim_object_from_presets(
        object_id="sat_rect_anim_pd_eci",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=False,
        enable_attitude_knowledge=True,
        attitude_quat_bn=np.array(q_init_bn, dtype=float),
        angular_rate_body_rad_s=np.zeros(3),
        attitude_substep_s=attitude_dt_s,
    )

    wheel_axes = np.vstack([w.axis_body for w in BASIC_REACTION_WHEEL_TRIAD.wheels])
    wheel_limits = np.array([w.max_torque_nm for w in BASIC_REACTION_WHEEL_TRIAD.wheels], dtype=float) * float(wheel_scale)
    ctrl = ReactionWheelPDController(
        wheel_axes_body=wheel_axes,
        wheel_torque_limits_nm=wheel_limits,
        desired_attitude_quat_bn=np.array(q_target_bn, dtype=float),
        desired_rate_body_rad_s=np.zeros(3),
        kp=np.array(kp, dtype=float).reshape(3),
        kd=np.array(kd, dtype=float).reshape(3),
    )

    n = int(np.ceil(duration_s / dt_s))
    t_hist = np.zeros(n + 1)
    q_hist = np.zeros((n + 1, 4))
    w_hist = np.zeros((n + 1, 3))
    err_hist = np.zeros(n + 1)
    t_hist[0] = sat.truth.t_s
    q_hist[0, :] = sat.truth.attitude_quat_bn
    w_hist[0, :] = sat.truth.angular_rate_body_rad_s
    err_hist[0] = _quat_error_deg(q_target_bn, sat.truth.attitude_quat_bn)

    for k in range(n):
        t_now = sat.truth.t_s
        belief = StateBelief(
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
        cmd = ctrl.act(belief=belief, t_s=t_now, budget_ms=1.0)
        applied = sat.actuator.apply(cmd, sat.limits, dt_s)
        sat.truth = sat.dynamics.step(state=sat.truth, command=applied, env={}, dt_s=dt_s)

        t_hist[k + 1] = sat.truth.t_s
        q_hist[k + 1, :] = sat.truth.attitude_quat_bn
        w_hist[k + 1, :] = sat.truth.angular_rate_body_rad_s
        err_hist[k + 1] = _quat_error_deg(q_target_bn, sat.truth.attitude_quat_bn)

    return t_hist, q_hist, w_hist, err_hist


def run_demo(
    plot_mode: str = "interactive",
    duration_s: float = 240.0,
    dt_s: float = 0.05,
    attitude_dt_s: float = 0.01,
    init_yaw_deg: float = 0.0,
    init_pitch_deg: float = 0.0,
    init_roll_deg: float = 0.0,
    target_yaw_deg: float = 30.0,
    target_pitch_deg: float = -15.0,
    target_roll_deg: float = 20.0,
    kp: np.ndarray | None = None,
    kd: np.ndarray | None = None,
    wheel_scale: float = 1.0,
    frame_skip: int = 1,
    render_fps: float = 60.0,
    sim_seconds_per_real_second: float = 1.0,
) -> dict[str, str]:
    kp_vec = np.array([0.25, 0.25, 0.25], dtype=float) if kp is None else np.array(kp, dtype=float).reshape(3)
    kd_vec = np.array([1.0, 1.0, 1.0], dtype=float) if kd is None else np.array(kd, dtype=float).reshape(3)

    q_init_bn = dcm_to_quaternion_bn(_dcm_bn_from_euler_zyx_deg(init_yaw_deg, init_pitch_deg, init_roll_deg))
    q_target_bn = dcm_to_quaternion_bn(_dcm_bn_from_euler_zyx_deg(target_yaw_deg, target_pitch_deg, target_roll_deg))

    t_hist, q_hist, w_hist, err_hist = _propagate_pd_slew(
        duration_s=float(duration_s),
        dt_s=float(dt_s),
        attitude_dt_s=float(attitude_dt_s),
        q_init_bn=q_init_bn,
        q_target_bn=q_target_bn,
        kp=kp_vec,
        kd=kd_vec,
        wheel_scale=float(wheel_scale),
    )

    lx, ly, lz = (float(BASIC_SATELLITE.bus_size_m[0]), float(BASIC_SATELLITE.bus_size_m[1]), float(BASIC_SATELLITE.bus_size_m[2]))
    verts_body = _box_vertices_body(lx, ly, lz)
    max_len = max(lx, ly, lz)
    lim = 0.9 * max_len

    sim_to_real = max(float(sim_seconds_per_real_second), 1e-9)
    effective_frame_skip = max(1, int(frame_skip))
    if render_fps > 0.0:
        # Ensure requested speed does not require drawing faster than render_fps.
        effective_frame_skip = max(1, int(np.ceil(sim_to_real / (float(render_fps) * float(dt_s)))))
    frame_sim_s = float(dt_s) * float(effective_frame_skip)

    frame_idx = np.arange(0, q_hist.shape[0], effective_frame_skip)
    if frame_idx[-1] != q_hist.shape[0] - 1:
        frame_idx = np.append(frame_idx, q_hist.shape[0] - 1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.set_xlabel("ECI X (m)")
    ax.set_ylabel("ECI Y (m)")
    ax.set_zlabel("ECI Z (m)")
    ax.set_title("ECI PD Slew: Rectangular Satellite Animation")
    ax.grid(True, alpha=0.3)

    axis_scale = 0.75 * lim
    ax.plot([0, axis_scale], [0, 0], [0, 0], color="r", linewidth=2, label="ECI X")
    ax.plot([0, 0], [0, axis_scale], [0, 0], color="g", linewidth=2, label="ECI Y")
    ax.plot([0, 0], [0, 0], [0, axis_scale], color="b", linewidth=2, label="ECI Z")
    ax.legend(loc="upper right")

    poly = Poly3DCollection(_box_faces(verts_body), alpha=0.45, facecolor="tab:orange", edgecolor="k")
    ax.add_collection3d(poly)
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def _update(i: int):
        k = int(frame_idx[i])
        c_bn = quaternion_to_dcm_bn(q_hist[k, :])
        c_nb = c_bn.T
        verts_eci = (c_nb @ verts_body.T).T
        poly.set_verts(_box_faces(verts_eci))
        txt.set_text(f"t = {t_hist[k]:.1f} s | q_err = {err_hist[k]:.2f} deg")
        return (poly, txt)

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=frame_idx.size,
        interval=1000.0 * frame_sim_s / sim_to_real,
        blit=False,
        repeat=True,
    )

    outdir = REPO_ROOT / "outputs" / "attitude_rectangle_animation_eci_pd"
    outpath = outdir / "attitude_rectangle_eci_pd.gif"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            ani.save(
                str(outpath),
                writer="pillow",
                fps=max(1, int(round(sim_to_real / frame_sim_s))),
            )
        except Exception as exc:
            print(f"Warning: failed to save animation ({exc}).")

    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    return {
        "plot_mode": plot_mode,
        "duration_s": f"{float(duration_s):.2f}",
        "dt_s": f"{float(dt_s):.3f}",
        "attitude_dt_s": f"{float(attitude_dt_s):.3f}",
        "render_fps": f"{float(render_fps):.1f}",
        "sim_seconds_per_real_second": f"{float(sim_to_real):.3f}",
        "effective_frame_skip": str(int(effective_frame_skip)),
        "init_quat_bn": str(np.array(q_init_bn, dtype=float).tolist()),
        "target_quat_bn": str(np.array(q_target_bn, dtype=float).tolist()),
        "final_rate_body_rad_s": str(np.array(w_hist[-1, :], dtype=float).tolist()),
        "final_attitude_error_deg": f"{float(err_hist[-1]):.6f}",
        "animation_path": str(outpath) if plot_mode in ("save", "both") and outpath.exists() else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animate rectangular satellite ECI slew under reaction-wheel PD (zero initial body rate)."
    )
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--duration", type=float, default=240.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--attitude-dt", type=float, default=0.01)
    parser.add_argument("--init-yaw-deg", type=float, default=0.0)
    parser.add_argument("--init-pitch-deg", type=float, default=0.0)
    parser.add_argument("--init-roll-deg", type=float, default=0.0)
    parser.add_argument("--target-yaw-deg", type=float, default=30.0)
    parser.add_argument("--target-pitch-deg", type=float, default=-15.0)
    parser.add_argument("--target-roll-deg", type=float, default=20.0)
    parser.add_argument("--kp", type=float, default=0.25, help="Scalar PD Kp used for all three axes.")
    parser.add_argument("--kd", type=float, default=1.0, help="Scalar PD Kd used for all three axes.")
    parser.add_argument("--wheel-scale", type=float, default=1.0)
    parser.add_argument("--frame-skip", type=int, default=1, help="Fallback frame skip when render-fps <= 0")
    parser.add_argument("--render-fps", type=float, default=60.0)
    parser.add_argument(
        "--sim-seconds-per-real-second",
        type=float,
        default=10.0,
        help="Playback speed multiplier: X means X seconds of simulation per 1 second of real time.",
    )
    args = parser.parse_args()

    out = run_demo(
        plot_mode=args.plot_mode,
        duration_s=float(args.duration),
        dt_s=float(args.dt),
        attitude_dt_s=float(args.attitude_dt),
        init_yaw_deg=float(args.init_yaw_deg),
        init_pitch_deg=float(args.init_pitch_deg),
        init_roll_deg=float(args.init_roll_deg),
        target_yaw_deg=float(args.target_yaw_deg),
        target_pitch_deg=float(args.target_pitch_deg),
        target_roll_deg=float(args.target_roll_deg),
        kp=np.array([float(args.kp)] * 3, dtype=float),
        kd=np.array([float(args.kd)] * 3, dtype=float),
        wheel_scale=float(args.wheel_scale),
        frame_skip=int(args.frame_skip),
        render_fps=float(args.render_fps),
        sim_seconds_per_real_second=float(args.sim_seconds_per_real_second),
    )
    print("ECI PD rectangle animation demo outputs:")
    for k, v in out.items():
        if v:
            print(f"  {k}: {v}")
