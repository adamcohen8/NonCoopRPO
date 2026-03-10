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
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import dcm_to_quaternion_bn, quaternion_to_dcm_bn


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _ric_rect_to_eci(x_host_eci: np.ndarray, x_rel_rect: np.ndarray) -> np.ndarray:
    r_host = x_host_eci[:3]
    v_host = x_host_eci[3:]
    xr = x_rel_rect[:3]
    xv = x_rel_rect[3:]

    h = np.cross(r_host, v_host)
    in_vec = np.cross(h, r_host)
    rsw = np.column_stack((_unit(r_host), _unit(in_vec), _unit(h)))
    dr = np.linalg.inv(rsw.T) @ xr

    rtemp = np.cross(h, v_host)
    vtemp = np.cross(h, r_host)
    drsw = np.column_stack((v_host / max(np.linalg.norm(r_host), 1e-12), rtemp / max(np.linalg.norm(vtemp), 1e-12), np.zeros(3)))
    frame_mv = np.array(
        [
            xr[0] * (r_host @ v_host) / (max(np.linalg.norm(r_host), 1e-12) ** 2),
            xr[1] * (vtemp @ rtemp) / (max(np.linalg.norm(vtemp), 1e-12) ** 2),
            0.0,
        ]
    )
    dv = np.linalg.inv(rsw.T) @ (xv + frame_mv - (drsw.T @ dr))

    return np.hstack((r_host + dr, v_host + dv))


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


def _ric_to_plot_coords(ric_vec: np.ndarray) -> np.ndarray:
    # Plot axes are arranged as [I, C, R] so that R is vertical ("up").
    v = np.array(ric_vec, dtype=float).reshape(3)
    return np.array([v[1], v[2], v[0]], dtype=float)


def _nmc_inplane_initial_state(
    chief_r_eci_km: np.ndarray,
    radius_m: float,
    phase_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Bounded HCW/NMC in-plane relative motion:
    # x(t) = rho cos(nt + phi), y(t) = -2 rho sin(nt + phi), z(t)=0
    # where x=R, y=I in rectangular RIC.
    r_mag = float(np.linalg.norm(chief_r_eci_km))
    n = float(np.sqrt(EARTH_MU_KM3_S2 / max(r_mag**3, 1e-12)))
    rho_km = max(float(radius_m), 0.0) / 1e3
    phi = np.deg2rad(float(phase_deg))
    rel_pos = np.array(
        [
            rho_km * np.cos(phi),
            -2.0 * rho_km * np.sin(phi),
            0.0,
        ],
        dtype=float,
    )
    rel_vel = np.array(
        [
            -n * rho_km * np.sin(phi),
            -2.0 * n * rho_km * np.cos(phi),
            0.0,
        ],
        dtype=float,
    )
    return rel_pos, rel_vel


def run_demo(
    plot_mode: str = "interactive",
    duration_s: float = 2400.0,
    dt_s: float = 1.0,
    attitude_dt_s: float = 0.05,
    rel_pos0_ric_km: np.ndarray | None = None,
    rel_vel0_ric_km_s: np.ndarray | None = None,
    use_nmc_init: bool = True,
    nmc_radius_m: float = 75.0,
    nmc_phase_deg: float = 30.0,
    chaser_omega0_body_rad_s: np.ndarray | None = None,
    target_omega0_body_rad_s: np.ndarray | None = None,
    render_fps: float = 60.0,
    frame_skip: int = 1,
    sim_seconds_per_real_second: float = 20.0,
    body_scale: float = 25.0,
    target_body_scale: float = 25.0,
    trail_points: int = 300,
    zoom: float = 3.0,
) -> dict[str, str]:
    rel_pos0_user = (
        np.array([0.10, -0.15, 0.05], dtype=float)
        if rel_pos0_ric_km is None
        else np.array(rel_pos0_ric_km, dtype=float).reshape(3)
    )
    rel_vel0_user = (
        np.array([0.0, 0.0, 0.0], dtype=float)
        if rel_vel0_ric_km_s is None
        else np.array(rel_vel0_ric_km_s, dtype=float).reshape(3)
    )
    w0 = (
        np.array([0.02, -0.015, 0.012], dtype=float)
        if chaser_omega0_body_rad_s is None
        else np.array(chaser_omega0_body_rad_s, dtype=float).reshape(3)
    )
    w0_target = (
        np.array([0.01, 0.008, -0.006], dtype=float)
        if target_omega0_body_rad_s is None
        else np.array(target_omega0_body_rad_s, dtype=float).reshape(3)
    )

    target = build_sim_object_from_presets(
        object_id="target_obs",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=True,
        enable_attitude_knowledge=False,
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=w0_target,
        attitude_substep_s=attitude_dt_s,
    )
    chaser = build_sim_object_from_presets(
        object_id="chaser_passive",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=True,  # user requested disturbance torques on
        enable_attitude_knowledge=False,
        attitude_quat_bn=dcm_to_quaternion_bn(np.eye(3)),
        angular_rate_body_rad_s=w0,
        attitude_substep_s=attitude_dt_s,
    )

    # Initialize chaser from target-centered rectangular RIC offset.
    x_target0 = np.hstack((target.truth.position_eci_km, target.truth.velocity_eci_km_s))
    if bool(use_nmc_init):
        rel_pos0, rel_vel0 = _nmc_inplane_initial_state(
            chief_r_eci_km=target.truth.position_eci_km,
            radius_m=float(nmc_radius_m),
            phase_deg=float(nmc_phase_deg),
        )
    else:
        rel_pos0, rel_vel0 = rel_pos0_user, rel_vel0_user
    x_rel0_rect = np.hstack((rel_pos0, rel_vel0))
    x_chaser0 = _ric_rect_to_eci(x_target0, x_rel0_rect)
    chaser.truth.position_eci_km = x_chaser0[:3]
    chaser.truth.velocity_eci_km_s = x_chaser0[3:]

    n = int(np.ceil(duration_s / dt_s))
    t_hist = np.zeros(n + 1)
    rel_pos_ric_hist_km = np.zeros((n + 1, 3))
    rel_range_m = np.zeros(n + 1)
    c_rb_hist = np.zeros((n + 1, 3, 3))
    c_rb_target_hist = np.zeros((n + 1, 3, 3))
    w_chaser_hist = np.zeros((n + 1, 3))
    w_target_hist = np.zeros((n + 1, 3))

    # Initial target-frame geometry.
    c_ir = ric_dcm_ir_from_rv(target.truth.position_eci_km, target.truth.velocity_eci_km_s)
    rel_eci0 = chaser.truth.position_eci_km - target.truth.position_eci_km
    rel_pos_ric_hist_km[0, :] = c_ir.T @ rel_eci0
    rel_range_m[0] = float(np.linalg.norm(rel_pos_ric_hist_km[0, :]) * 1e3)
    c_bn0 = quaternion_to_dcm_bn(chaser.truth.attitude_quat_bn)
    c_rb_hist[0, :, :] = c_ir.T @ c_bn0.T
    c_bn0_target = quaternion_to_dcm_bn(target.truth.attitude_quat_bn)
    c_rb_target_hist[0, :, :] = c_ir.T @ c_bn0_target.T
    w_chaser_hist[0, :] = chaser.truth.angular_rate_body_rad_s
    w_target_hist[0, :] = target.truth.angular_rate_body_rad_s
    t_hist[0] = 0.0

    for k in range(n):
        target.truth = target.dynamics.step(state=target.truth, command=Command.zero(), env={}, dt_s=dt_s)
        chaser.truth = chaser.dynamics.step(state=chaser.truth, command=Command.zero(), env={}, dt_s=dt_s)

        c_ir = ric_dcm_ir_from_rv(target.truth.position_eci_km, target.truth.velocity_eci_km_s)
        rel_eci = chaser.truth.position_eci_km - target.truth.position_eci_km
        rel_pos_ric = c_ir.T @ rel_eci

        c_bn = quaternion_to_dcm_bn(chaser.truth.attitude_quat_bn)
        c_rb = c_ir.T @ c_bn.T  # body -> target RIC
        c_bn_target = quaternion_to_dcm_bn(target.truth.attitude_quat_bn)
        c_rb_target = c_ir.T @ c_bn_target.T  # target body -> target RIC

        t_hist[k + 1] = target.truth.t_s
        rel_pos_ric_hist_km[k + 1, :] = rel_pos_ric
        rel_range_m[k + 1] = float(np.linalg.norm(rel_pos_ric) * 1e3)
        c_rb_hist[k + 1, :, :] = c_rb
        c_rb_target_hist[k + 1, :, :] = c_rb_target
        w_chaser_hist[k + 1, :] = chaser.truth.angular_rate_body_rad_s
        w_target_hist[k + 1, :] = target.truth.angular_rate_body_rad_s

    # Box geometry in km (scaled for visibility).
    lx_m = float(BASIC_SATELLITE.bus_size_m[0]) * float(body_scale)
    ly_m = float(BASIC_SATELLITE.bus_size_m[1]) * float(body_scale)
    lz_m = float(BASIC_SATELLITE.bus_size_m[2]) * float(body_scale)
    verts_body_km = _box_vertices_body(lx_m, ly_m, lz_m) / 1e3
    lx_t_m = float(BASIC_SATELLITE.bus_size_m[0]) * float(target_body_scale)
    ly_t_m = float(BASIC_SATELLITE.bus_size_m[1]) * float(target_body_scale)
    lz_t_m = float(BASIC_SATELLITE.bus_size_m[2]) * float(target_body_scale)
    verts_target_body_km = _box_vertices_body(lx_t_m, ly_t_m, lz_t_m) / 1e3

    # Axis limits from trajectory extent.
    max_abs = float(np.max(np.abs(rel_pos_ric_hist_km)))
    zoom_eff = max(float(zoom), 1e-6)
    lim_km = max(0.03, (1.2 * max_abs) / zoom_eff)

    sim_to_real = max(float(sim_seconds_per_real_second), 1e-9)
    effective_frame_skip = max(1, int(frame_skip))
    if render_fps > 0.0:
        effective_frame_skip = max(1, int(np.ceil(sim_to_real / (float(render_fps) * float(dt_s)))))
    frame_sim_s = float(dt_s) * float(effective_frame_skip)

    frame_idx = np.arange(0, t_hist.size, effective_frame_skip)
    if frame_idx[-1] != t_hist.size - 1:
        frame_idx = np.append(frame_idx, t_hist.size - 1)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-lim_km, lim_km)
    ax.set_ylim(-lim_km, lim_km)
    ax.set_zlim(-lim_km, lim_km)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.set_xlabel("I (km)")
    ax.set_ylabel("C (km)")
    ax.set_zlabel("R (km)")
    ax.set_title("Target-Centered View: Passive Chaser Drift + Tumble (RIC Frame)")
    ax.grid(True, alpha=0.3)

    # Target marker at frame origin.
    ax.scatter([0.0], [0.0], [0.0], s=60, c="tab:blue", marker="o", label="Target COM")
    ax.legend(loc="upper right")

    poly_chaser = Poly3DCollection(_box_faces(verts_body_km), alpha=0.45, facecolor="tab:orange", edgecolor="k")
    poly_target = Poly3DCollection(_box_faces(verts_target_body_km), alpha=0.50, facecolor="tab:cyan", edgecolor="k")
    ax.add_collection3d(poly_chaser)
    ax.add_collection3d(poly_target)
    trail, = ax.plot([], [], [], color="tab:red", linewidth=1.4, alpha=0.8, label="Chaser trail")
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def _update(i: int):
        k = int(frame_idx[i])
        c_rb = c_rb_hist[k, :, :]
        center_ric = rel_pos_ric_hist_km[k, :]
        verts_ric = (c_rb @ verts_body_km.T).T + center_ric.reshape(1, 3)
        verts_plot = np.array([_ric_to_plot_coords(v) for v in verts_ric], dtype=float)
        poly_chaser.set_verts(_box_faces(verts_plot))

        c_rb_target = c_rb_target_hist[k, :, :]
        verts_target_ric = (c_rb_target @ verts_target_body_km.T).T
        verts_target_plot = np.array([_ric_to_plot_coords(v) for v in verts_target_ric], dtype=float)
        poly_target.set_verts(_box_faces(verts_target_plot))

        k0 = max(0, k - max(1, int(trail_points)))
        trail_x = rel_pos_ric_hist_km[k0 : k + 1, 1]
        trail_y = rel_pos_ric_hist_km[k0 : k + 1, 2]
        trail_z = rel_pos_ric_hist_km[k0 : k + 1, 0]
        trail.set_data(trail_x, trail_y)
        trail.set_3d_properties(trail_z)

        w_ch = float(np.linalg.norm(w_chaser_hist[k, :]))
        w_tg = float(np.linalg.norm(w_target_hist[k, :]))
        txt.set_text(
            f"t = {t_hist[k]:.1f} s | range = {rel_range_m[k]:.1f} m | "
            f"|w_tgt|={w_tg:.3f} rad/s | |w_chs|={w_ch:.3f} rad/s"
        )
        return (poly_chaser, poly_target, trail, txt)

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=frame_idx.size,
        interval=1000.0 * frame_sim_s / sim_to_real,
        blit=False,
        repeat=True,
    )

    outdir = REPO_ROOT / "outputs" / "two_sat_target_frame_drift_tumble"
    outpath = outdir / "target_frame_drift_tumble.gif"
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
        "sim_seconds_per_real_second": f"{float(sim_to_real):.3f}",
        "render_fps": f"{float(render_fps):.1f}",
        "effective_frame_skip": str(int(effective_frame_skip)),
        "initial_rel_pos_ric_km": str(rel_pos0.tolist()),
        "initial_rel_vel_ric_km_s": str(rel_vel0.tolist()),
        "use_nmc_init": str(bool(use_nmc_init)),
        "nmc_radius_m": f"{float(nmc_radius_m):.3f}",
        "nmc_phase_deg": f"{float(nmc_phase_deg):.3f}",
        "initial_target_omega_body_rad_s": str(w0_target.tolist()),
        "initial_chaser_omega_body_rad_s": str(w0.tolist()),
        "max_range_m": f"{float(np.max(rel_range_m)):.3f}",
        "min_range_m": f"{float(np.min(rel_range_m)):.3f}",
        "zoom": f"{float(zoom_eff):.3f}",
        "animation_path": str(outpath) if plot_mode in ("save", "both") and outpath.exists() else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Two-satellite demo in target-centered RIC frame. "
            "Target is passive observer, chaser is passive with disturbance-torque-driven tumble."
        )
    )
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--duration", type=float, default=2400.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--attitude-dt", type=float, default=0.05)
    parser.add_argument("--rel-r-km", type=float, default=0.10, help="Initial relative radial position (km).")
    parser.add_argument("--rel-i-km", type=float, default=-0.15, help="Initial relative in-track position (km).")
    parser.add_argument("--rel-c-km", type=float, default=0.05, help="Initial relative cross-track position (km).")
    parser.add_argument("--rel-vr-km-s", type=float, default=0.0, help="Initial relative radial velocity (km/s).")
    parser.add_argument("--rel-vi-km-s", type=float, default=0.0, help="Initial relative in-track velocity (km/s).")
    parser.add_argument("--rel-vc-km-s", type=float, default=0.0, help="Initial relative cross-track velocity (km/s).")
    parser.add_argument(
        "--no-nmc-init",
        action="store_true",
        help="Disable default NMC initialization and use explicit rel position/velocity inputs instead.",
    )
    parser.add_argument("--nmc-radius-m", type=float, default=75.0, help="In-plane NMC radial amplitude rho (meters).")
    parser.add_argument("--nmc-phase-deg", type=float, default=30.0, help="Initial NMC phase angle (degrees).")
    parser.add_argument("--omega-x", type=float, default=0.02, help="Initial chaser body rate wx (rad/s).")
    parser.add_argument("--omega-y", type=float, default=-0.015, help="Initial chaser body rate wy (rad/s).")
    parser.add_argument("--omega-z", type=float, default=0.012, help="Initial chaser body rate wz (rad/s).")
    parser.add_argument("--target-omega-x", type=float, default=0.01, help="Initial target body rate wx (rad/s).")
    parser.add_argument("--target-omega-y", type=float, default=0.008, help="Initial target body rate wy (rad/s).")
    parser.add_argument("--target-omega-z", type=float, default=-0.006, help="Initial target body rate wz (rad/s).")
    parser.add_argument("--body-scale", type=float, default=25.0, help="Visualization scale factor for chaser body size.")
    parser.add_argument("--target-body-scale", type=float, default=25.0, help="Visualization scale factor for target body size.")
    parser.add_argument("--trail-points", type=int, default=300, help="Number of recent points in trail.")
    parser.add_argument("--zoom", type=float, default=3.0, help="Zoom factor (>1 zooms in, <1 zooms out).")
    parser.add_argument("--frame-skip", type=int, default=1, help="Fallback frame skip when render-fps <= 0")
    parser.add_argument("--render-fps", type=float, default=60.0)
    parser.add_argument(
        "--sim-seconds-per-real-second",
        type=float,
        default=20.0,
        help="Playback speed multiplier: X means X seconds of simulation per 1 second of real time.",
    )
    args = parser.parse_args()

    out = run_demo(
        plot_mode=args.plot_mode,
        duration_s=float(args.duration),
        dt_s=float(args.dt),
        attitude_dt_s=float(args.attitude_dt),
        rel_pos0_ric_km=np.array([float(args.rel_r_km), float(args.rel_i_km), float(args.rel_c_km)], dtype=float),
        rel_vel0_ric_km_s=np.array(
            [float(args.rel_vr_km_s), float(args.rel_vi_km_s), float(args.rel_vc_km_s)],
            dtype=float,
        ),
        use_nmc_init=not bool(args.no_nmc_init),
        nmc_radius_m=float(args.nmc_radius_m),
        nmc_phase_deg=float(args.nmc_phase_deg),
        target_omega0_body_rad_s=np.array(
            [float(args.target_omega_x), float(args.target_omega_y), float(args.target_omega_z)],
            dtype=float,
        ),
        chaser_omega0_body_rad_s=np.array([float(args.omega_x), float(args.omega_y), float(args.omega_z)], dtype=float),
        render_fps=float(args.render_fps),
        frame_skip=int(args.frame_skip),
        sim_seconds_per_real_second=float(args.sim_seconds_per_real_second),
        body_scale=float(args.body_scale),
        target_body_scale=float(args.target_body_scale),
        trail_points=int(args.trail_points),
        zoom=float(args.zoom),
    )

    print("Two-satellite target-frame drift+tumble demo outputs:")
    for k, v in out.items():
        if v:
            print(f"  {k}: {v}")
