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

from presets import BASIC_SATELLITE, build_sim_object_from_presets
from sim.core.models import Command
from sim.utils.quaternion import quaternion_to_dcm_bn


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
        vertices[[0, 1, 2, 3]],  # -z
        vertices[[4, 5, 6, 7]],  # +z
        vertices[[0, 1, 5, 4]],  # -y
        vertices[[2, 3, 7, 6]],  # +y
        vertices[[1, 2, 6, 5]],  # +x
        vertices[[0, 3, 7, 4]],  # -x
    ]


def _propagate_free_tumble(
    duration_s: float,
    dt_s: float,
    attitude_dt_s: float,
    angular_rate_body_rad_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sat = build_sim_object_from_presets(
        object_id="sat_rect_anim_eci",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=False,  # user requested no disturbance torques
        enable_attitude_knowledge=True,
        angular_rate_body_rad_s=np.array(angular_rate_body_rad_s, dtype=float),
        attitude_substep_s=attitude_dt_s,
    )

    n = int(np.ceil(duration_s / dt_s))
    t_hist = np.zeros(n + 1)
    q_hist = np.zeros((n + 1, 4))
    t_hist[0] = sat.truth.t_s
    q_hist[0, :] = sat.truth.attitude_quat_bn

    for k in range(n):
        sat.truth = sat.dynamics.step(
            state=sat.truth,
            command=Command.zero(),  # no control torque
            env={},
            dt_s=dt_s,
        )
        t_hist[k + 1] = sat.truth.t_s
        q_hist[k + 1, :] = sat.truth.attitude_quat_bn
    return t_hist, q_hist


def run_demo(
    plot_mode: str = "interactive",
    duration_s: float = 180.0,
    dt_s: float = 0.01,
    attitude_dt_s: float = 0.01,
    omega0_body_rad_s: np.ndarray | None = None,
    frame_skip: int = 2,
    render_fps: float = 60.0,
    sim_seconds_per_real_second: float = 1.0,
) -> dict[str, str]:
    omega0 = np.array([0.02, -0.015, 0.01], dtype=float) if omega0_body_rad_s is None else np.array(omega0_body_rad_s, dtype=float).reshape(3)
    t_hist, q_hist = _propagate_free_tumble(
        duration_s=float(duration_s),
        dt_s=float(dt_s),
        attitude_dt_s=float(attitude_dt_s),
        angular_rate_body_rad_s=omega0,
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
    ax.set_title("Rectangular Satellite Attitude Animation in ECI")
    ax.grid(True, alpha=0.3)

    # Inertial frame axes for reference.
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
        q = q_hist[k, :]
        c_bn = quaternion_to_dcm_bn(q)
        c_nb = c_bn.T
        verts_eci = (c_nb @ verts_body.T).T
        poly.set_verts(_box_faces(verts_eci))
        txt.set_text(f"t = {t_hist[k]:.1f} s")
        return (poly, txt)

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=frame_idx.size,
        interval=1000.0 * frame_sim_s / sim_to_real,
        blit=False,
        repeat=True,
    )

    outdir = REPO_ROOT / "outputs" / "attitude_rectangle_animation_eci"
    outpath = outdir / "attitude_rectangle_eci.gif"
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
        "render_fps": f"{float(render_fps):.1f}",
        "sim_seconds_per_real_second": f"{float(sim_to_real):.3f}",
        "effective_frame_skip": str(int(effective_frame_skip)),
        "omega0_body_rad_s": f"{omega0.tolist()}",
        "animation_path": str(outpath) if plot_mode in ("save", "both") and outpath.exists() else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animate a rectangular satellite attitude in ECI (free tumble, no disturbances, no control)."
    )
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--attitude-dt", type=float, default=0.01)
    parser.add_argument("--omega-x", type=float, default=0.02, help="Initial body rate wx (rad/s)")
    parser.add_argument("--omega-y", type=float, default=-0.015, help="Initial body rate wy (rad/s)")
    parser.add_argument("--omega-z", type=float, default=0.01, help="Initial body rate wz (rad/s)")
    parser.add_argument("--frame-skip", type=int, default=2, help="Fallback frame skip when render-fps <= 0")
    parser.add_argument("--render-fps", type=float, default=60.0, help="Target animation FPS for smoother playback")
    parser.add_argument(
        "--sim-seconds-per-real-second",
        type=float,
        default=1.0,
        help="Playback speed multiplier: X means X seconds of simulation per 1 second of real time.",
    )
    args = parser.parse_args()

    out = run_demo(
        plot_mode=args.plot_mode,
        duration_s=float(args.duration),
        dt_s=float(args.dt),
        attitude_dt_s=float(args.attitude_dt),
        omega0_body_rad_s=np.array([float(args.omega_x), float(args.omega_y), float(args.omega_z)], dtype=float),
        frame_skip=int(args.frame_skip),
        render_fps=float(args.render_fps),
        sim_seconds_per_real_second=float(args.sim_seconds_per_real_second),
    )
    print("ECI rectangle animation demo outputs:")
    for k, v in out.items():
        if v:
            print(f"  {k}: {v}")
