from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sim.dynamics.orbit.frames import eci_to_ecef
from sim.dynamics.orbit.epoch import julian_date_to_datetime
from sim.utils.frames import ric_curv_to_rect, ric_dcm_ir_from_rv, ric_rect_to_curv
from sim.utils.ground_track import ground_track_from_eci_history, split_ground_track_dateline
from sim.utils.quaternion import dcm_to_quaternion_bn, quaternion_to_dcm_bn
from sim.utils.plotting import (
    plot_angular_rates as plot_angular_rates_legacy,
    plot_attitude_ric as plot_attitude_ric_legacy,
    plot_attitude_tumble as plot_attitude_tumble_legacy,
    plot_ground_track as plot_ground_track_legacy,
    plot_orbit_eci as plot_orbit_eci_legacy,
)

PlotMode = Literal["interactive", "save", "both"]
FrameName = Literal["eci", "ecef", "ric_rect", "ric_curv"]
AttitudeFrame = Literal["eci", "ric"]
Layout = Literal["single", "subplots"]

try:
    import cartopy.crs as ccrs  # type: ignore
    import cartopy.feature as cfeature  # type: ignore

    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False


def _show_save_close(fig: plt.Figure, *, mode: PlotMode, out_path: str | None, dpi: int = 150) -> None:
    if mode in ("save", "both"):
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'.")
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=dpi)
    if mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)


def _draw_stylized_earth_map(ax: plt.Axes) -> None:
    ocean = Rectangle((-180.0, -90.0), 360.0, 180.0, facecolor="#cfe8ff", edgecolor="none", zorder=0)
    ax.add_patch(ocean)
    continents = [
        [(-168, 72), (-145, 68), (-130, 55), (-123, 50), (-118, 34), (-105, 24), (-97, 17), (-83, 20), (-80, 27), (-66, 45), (-82, 55), (-110, 72)],
        [(-81, 12), (-72, 8), (-66, -5), (-62, -18), (-58, -33), (-54, -54), (-69, -56), (-76, -40), (-78, -20), (-81, 0)],
        [(-18, 35), (2, 37), (20, 33), (33, 23), (40, 8), (47, -12), (40, -28), (28, -35), (13, -35), (3, -24), (-4, -6), (-9, 14), (-16, 28)],
        [(-10, 36), (8, 46), (30, 56), (55, 64), (90, 72), (120, 66), (145, 58), (170, 50), (155, 40), (120, 24), (102, 12), (80, 8), (55, 16), (30, 26), (18, 32), (5, 38)],
        [(72, 23), (85, 22), (95, 15), (103, 8), (106, 2), (102, -4), (90, 2), (82, 8), (75, 16)],
        [(113, -12), (132, -11), (150, -20), (154, -32), (145, -42), (129, -42), (116, -33), (111, -22)],
        [(-56, 82), (-42, 82), (-28, 74), (-34, 62), (-49, 60), (-60, 68)],
        [(-180, -62), (-120, -64), (-60, -66), (0, -68), (60, -66), (120, -64), (180, -62), (180, -90), (-180, -90)],
    ]
    for poly in continents:
        ax.add_patch(Polygon(poly, closed=True, facecolor="#dbe7c9", edgecolor="#8aa27a", linewidth=0.6, zorder=1))


def _setup_ground_track_axes(
    *,
    title: str,
    draw_earth_map: bool,
) -> tuple[plt.Figure, Any, bool]:
    if draw_earth_map and _HAS_CARTOPY:
        fig = plt.figure(figsize=(11, 5))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        ax.set_global()
        ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#cfe8ff", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#dbe7c9", edgecolor="#8aa27a", linewidth=0.4, zorder=1)
        ax.coastlines(resolution="110m", linewidth=0.5, color="#5e6f57", zorder=2)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="-")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        ax.set_title(title)
        return fig, ax, True

    fig, ax = plt.subplots(figsize=(11, 5))
    if draw_earth_map:
        _draw_stylized_earth_map(ax)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 15))
    for xv in np.arange(-180, 181, 30):
        ax.axvline(xv, color="gray", linewidth=0.35, alpha=0.35, zorder=0)
    for yv in np.arange(-90, 91, 15):
        ax.axhline(yv, color="gray", linewidth=0.35, alpha=0.35, zorder=0)
    return fig, ax, False


def _truth_quaternion_in_frame(truth_hist: np.ndarray, frame: AttitudeFrame) -> np.ndarray:
    q_bn = np.array(truth_hist[:, 6:10], dtype=float)
    if frame == "eci":
        return q_bn
    out = np.zeros_like(q_bn)
    for k in range(truth_hist.shape[0]):
        r = truth_hist[k, 0:3]
        v = truth_hist[k, 3:6]
        c_bn = quaternion_to_dcm_bn(q_bn[k, :])
        c_ir = ric_dcm_ir_from_rv(r, v)
        c_br = c_bn @ c_ir
        out[k, :] = dcm_to_quaternion_bn(c_br)
    return out


def _rates_in_frame(truth_hist: np.ndarray, frame: AttitudeFrame) -> np.ndarray:
    w_body = np.array(truth_hist[:, 10:13], dtype=float)
    if frame == "eci":
        return w_body
    out = np.zeros_like(w_body)
    q_bn = np.array(truth_hist[:, 6:10], dtype=float)
    for k in range(truth_hist.shape[0]):
        r = truth_hist[k, 0:3]
        v = truth_hist[k, 3:6]
        c_bn = quaternion_to_dcm_bn(q_bn[k, :])
        c_ir = ric_dcm_ir_from_rv(r, v)
        c_br = c_bn @ c_ir
        out[k, :] = c_br.T @ w_body[k, :]
    return out


def plot_quaternion_components(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    frame: AttitudeFrame = "eci",
    layout: Layout = "single",
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    q = _truth_quaternion_in_frame(truth_hist, frame)
    labels = ["q0", "q1", "q2", "q3"]
    if layout == "single":
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(4):
            ax.plot(t_s, q[:, i], label=labels[i])
        ax.set_title(f"Quaternion Components ({frame.upper()} frame)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Quaternion")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    else:
        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(t_s, q[:, i], linewidth=1.3)
            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.3)
        axes[0].set_title(f"Quaternion Components ({frame.upper()} frame)")
        axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)


def plot_body_rates(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    frame: AttitudeFrame = "eci",
    layout: Layout = "subplots",
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    w = _rates_in_frame(truth_hist, frame)
    labels = ["wx", "wy", "wz"]
    if layout == "single":
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(3):
            ax.plot(t_s, w[:, i], label=labels[i])
        ax.set_title(f"Body Angular Rates ({frame.upper()} frame)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("rad/s")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    else:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(t_s, w[:, i], linewidth=1.3)
            ax.set_ylabel(f"{labels[i]} (rad/s)")
            ax.grid(True, alpha=0.3)
        axes[0].set_title(f"Body Angular Rates ({frame.upper()} frame)")
        axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)


def _trajectory_in_frame(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    frame: FrameName,
    jd_utc_start: float | None = None,
    reference_truth_hist: np.ndarray | None = None,
) -> np.ndarray:
    r_eci = np.array(truth_hist[:, 0:3], dtype=float)
    if frame == "eci":
        return r_eci
    if frame == "ecef":
        out = np.zeros_like(r_eci)
        for k in range(r_eci.shape[0]):
            out[k, :] = eci_to_ecef(r_eci[k, :], float(t_s[k]), jd_utc_start=jd_utc_start)
        return out
    if reference_truth_hist is None:
        raise ValueError("reference_truth_hist is required for RIC frame plots.")
    r_ref = np.array(reference_truth_hist[:, 0:3], dtype=float)
    v_ref = np.array(reference_truth_hist[:, 3:6], dtype=float)
    rel_rect = np.zeros_like(r_eci)
    for k in range(r_eci.shape[0]):
        c_ir = ric_dcm_ir_from_rv(r_ref[k, :], v_ref[k, :])
        rel_rect[k, :] = c_ir.T @ (r_eci[k, :] - r_ref[k, :])
    if frame == "ric_rect":
        return rel_rect
    out = np.zeros_like(rel_rect)
    for k in range(rel_rect.shape[0]):
        x_rect = np.hstack((rel_rect[k, :], np.zeros(3)))
        x_curv = ric_rect_to_curv(x_rect, r0_km=float(np.linalg.norm(r_ref[k, :])))
        out[k, :] = x_curv[:3]
    return out


def plot_trajectory_frame(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    frame: FrameName = "eci",
    jd_utc_start: float | None = None,
    reference_truth_hist: np.ndarray | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    r = _trajectory_in_frame(
        t_s=t_s,
        truth_hist=truth_hist,
        frame=frame,
        jd_utc_start=jd_utc_start,
        reference_truth_hist=reference_truth_hist,
    )
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(r[:, 0], r[:, 1], r[:, 2], linewidth=1.4)
    ax.set_title(f"Trajectory ({frame.upper()})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)


def plot_multi_trajectory_frame(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    frame: FrameName = "eci",
    jd_utc_start: float | None = None,
    reference_truth_hist: np.ndarray | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for oid, hist in truth_hist_by_object.items():
        if hist.size == 0 or not np.any(np.isfinite(hist[:, 0])):
            continue
        r = _trajectory_in_frame(
            t_s=t_s,
            truth_hist=hist,
            frame=frame,
            jd_utc_start=jd_utc_start,
            reference_truth_hist=reference_truth_hist,
        )
        ax.plot(r[:, 0], r[:, 1], r[:, 2], linewidth=1.4, label=oid)
    ax.set_title(f"Trajectories ({frame.upper()})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    ax.legend(loc="best")
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)


def plot_control_commands(
    t_s: np.ndarray,
    u_hist: np.ndarray,
    *,
    layout: Layout = "subplots",
    input_labels: list[str] | None = None,
    title: str = "Control Commands",
    y_label: str = "",
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    u = np.array(u_hist, dtype=float)
    if u.ndim != 2:
        raise ValueError("u_hist must be shape (N, M).")
    m = u.shape[1]
    labels = input_labels if input_labels is not None else [f"u{i}" for i in range(m)]
    if len(labels) != m:
        raise ValueError("input_labels length must match u_hist second dimension.")
    if layout == "single":
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(m):
            ax.plot(t_s, u[:, i], label=labels[i])
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(y_label if y_label else "Command")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    else:
        fig, axes = plt.subplots(m, 1, figsize=(10, max(3.0, 2.4 * m)), sharex=True)
        if m == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(t_s, u[:, i], linewidth=1.3)
            ax.set_ylabel(labels[i] if not y_label else f"{labels[i]} ({y_label})")
            ax.grid(True, alpha=0.3)
        axes[0].set_title(title)
        axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)


def plot_multi_control_commands(
    t_s: np.ndarray,
    u_hist_by_object: dict[str, np.ndarray],
    *,
    component_index: int = 0,
    title: str = "Control Command Overlay",
    y_label: str = "",
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for oid, u in u_hist_by_object.items():
        arr = np.array(u, dtype=float)
        if arr.ndim != 2 or arr.shape[1] <= component_index:
            continue
        ax.plot(t_s, arr[:, component_index], label=oid)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label if y_label else f"u[{component_index}]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)


def animate_rectangular_prism_attitude(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    lx_m: float,
    ly_m: float,
    lz_m: float,
    frame: AttitudeFrame = "eci",
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
) -> None:
    verts_body = np.array(
        [
            [-0.5 * lx_m, -0.5 * ly_m, -0.5 * lz_m],
            [-0.5 * lx_m, -0.5 * ly_m, +0.5 * lz_m],
            [-0.5 * lx_m, +0.5 * ly_m, -0.5 * lz_m],
            [-0.5 * lx_m, +0.5 * ly_m, +0.5 * lz_m],
            [+0.5 * lx_m, -0.5 * ly_m, -0.5 * lz_m],
            [+0.5 * lx_m, -0.5 * ly_m, +0.5 * lz_m],
            [+0.5 * lx_m, +0.5 * ly_m, -0.5 * lz_m],
            [+0.5 * lx_m, +0.5 * ly_m, +0.5 * lz_m],
        ],
        dtype=float,
    )
    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]

    q_bn = np.array(truth_hist[:, 6:10], dtype=float)
    c_anim = np.zeros((truth_hist.shape[0], 3, 3))
    for k in range(truth_hist.shape[0]):
        c_bn = quaternion_to_dcm_bn(q_bn[k, :])
        if frame == "eci":
            c_anim[k, :, :] = c_bn.T  # body -> ECI
        else:
            r = truth_hist[k, 0:3]
            v = truth_hist[k, 3:6]
            c_ir = ric_dcm_ir_from_rv(r, v)
            c_anim[k, :, :] = c_ir.T @ c_bn.T  # body -> RIC

    max_dim = 0.7 * max(lx_m, ly_m, lz_m)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-max_dim, max_dim)
    ax.set_ylim(-max_dim, max_dim)
    ax.set_zlim(-max_dim, max_dim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"Rectangular Prism Attitude Animation ({frame.upper()})")
    poly = Poly3DCollection([], alpha=0.35, facecolor="#4C9F70", edgecolor="k", linewidth=0.7)
    ax.add_collection3d(poly)

    def _frame_verts(i: int) -> list[np.ndarray]:
        v = (c_anim[i, :, :] @ verts_body.T).T
        return [v[idx, :] for idx in faces]

    def update(i: int):
        poly.set_verts(_frame_verts(i))
        ax.set_xlabel(f"t={t_s[i]:.1f}s")
        return [poly]

    dt = float(np.median(np.diff(t_s))) if t_s.size > 1 else 1.0
    interval_ms = 1000.0 * dt / max(speed_multiple, 1e-6)
    ani = animation.FuncAnimation(fig, update, frames=t_s.size, interval=interval_ms, blit=False)

    if mode in ("save", "both"):
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'.")
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            ani.save(str(p), fps=max(float(fps), 1.0))
        except Exception as exc:
            print(f"Warning: failed to save animation ({exc}).")
    if mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)


def animate_trajectory_frame(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    frame: FrameName = "eci",
    jd_utc_start: float | None = None,
    reference_truth_hist: np.ndarray | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
) -> None:
    r = _trajectory_in_frame(
        t_s=t_s,
        truth_hist=truth_hist,
        frame=frame,
        jd_utc_start=jd_utc_start,
        reference_truth_hist=reference_truth_hist,
    )
    lim = np.nanmax(np.abs(r))
    lim = float(max(lim, 1.0))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"Trajectory Animation ({frame.upper()})")
    line, = ax.plot([], [], [], linewidth=1.4)
    dot, = ax.plot([], [], [], marker="o", markersize=4)

    def update(i: int):
        line.set_data(r[: i + 1, 0], r[: i + 1, 1])
        line.set_3d_properties(r[: i + 1, 2])
        dot.set_data([r[i, 0]], [r[i, 1]])
        dot.set_3d_properties([r[i, 2]])
        ax.set_xlabel(f"t={t_s[i]:.1f}s")
        return [line, dot]

    dt = float(np.median(np.diff(t_s))) if t_s.size > 1 else 1.0
    interval_ms = 1000.0 * dt / max(speed_multiple, 1e-6)
    ani = animation.FuncAnimation(fig, update, frames=t_s.size, interval=interval_ms, blit=False)

    if mode in ("save", "both"):
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'.")
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            ani.save(str(p), fps=max(float(fps), 1.0))
        except Exception as exc:
            print(f"Warning: failed to save animation ({exc}).")
    if mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)


def animate_ground_track(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    *,
    t_s: np.ndarray | None = None,
    jd_utc_start: float | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
    draw_earth_map: bool = True,
    frame_stride: int = 1,
) -> None:
    lon_p, lat_p = split_ground_track_dateline(lon_deg=lon_deg, lat_deg=lat_deg, jump_threshold_deg=180.0)
    t_arr = np.array(t_s, dtype=float).reshape(-1) if t_s is not None else np.arange(len(lon_deg), dtype=float)
    if t_arr.size < len(lon_deg):
        t_arr = np.pad(t_arr, (0, len(lon_deg) - t_arr.size), mode="edge")
    fig, ax, is_cartopy = _setup_ground_track_axes(title="Ground Track Animation", draw_earth_map=draw_earth_map)
    if is_cartopy:
        line, = ax.plot([], [], linewidth=1.4, transform=ccrs.PlateCarree(), zorder=3)
        dot, = ax.plot([], [], marker="o", markersize=4, transform=ccrs.PlateCarree(), zorder=4)
    else:
        line, = ax.plot([], [], linewidth=1.4, zorder=3)
        dot, = ax.plot([], [], marker="o", markersize=4, zorder=4)
    time_text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        zorder=10,
    )

    stride = int(max(frame_stride, 1))
    frame_ids = np.arange(0, len(lon_p), stride, dtype=int)
    if frame_ids.size == 0 or frame_ids[-1] != (len(lon_p) - 1):
        frame_ids = np.append(frame_ids, len(lon_p) - 1)

    def update(i: int):
        idx = int(frame_ids[i])
        line.set_data(lon_p[: idx + 1], lat_p[: idx + 1])
        dot.set_data([lon_p[idx]], [lat_p[idx]])
        t_now = float(t_arr[min(idx, t_arr.size - 1)])
        if jd_utc_start is not None:
            dt_utc = julian_date_to_datetime(float(jd_utc_start) + t_now / 86400.0)
            time_text.set_text(f"UTC: {dt_utc.strftime('%Y-%m-%d %H:%M:%S')}\nSim t: {t_now:.1f} s")
        else:
            time_text.set_text(f"Sim t: {t_now:.1f} s")
        return [line, dot, time_text]

    interval_ms = 1000.0 / max(float(fps) * max(speed_multiple, 1e-6), 1e-3)
    ani = None
    if mode in ("save", "both"):
        ani = animation.FuncAnimation(fig, update, frames=int(frame_ids.size), interval=interval_ms, blit=False)
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'.")
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            ani.save(str(p), fps=max(float(fps), 1.0))
        except Exception as exc:
            print(f"Warning: failed to save animation ({exc}).")
    if mode in ("interactive", "both"):
        # Explicit interactive loop is more reliable than backend animation playback in IDE windows.
        plt.ion()
        fig.show()
        for i in range(int(frame_ids.size)):
            update(i)
            fig.canvas.draw_idle()
            plt.pause(interval_ms / 1000.0)
        plt.ioff()
        plt.show()
    if ani is not None:
        del ani
    plt.close(fig)


def animate_multi_ground_track(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    jd_utc_start: float | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
    draw_earth_map: bool = True,
    frame_stride: int = 1,
) -> None:
    tracks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    tracks_t: dict[str, np.ndarray] = {}
    n_frames = 0
    for oid, hist in truth_hist_by_object.items():
        arr = np.array(hist, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
            continue
        mask = np.isfinite(arr[:, 0])
        if not np.any(mask):
            continue
        lat, lon, _ = ground_track_from_eci_history(arr[:, :3], t_s=t_s, jd_utc_start=jd_utc_start)
        lon_p, lat_p = split_ground_track_dateline(lon_deg=lon, lat_deg=lat, jump_threshold_deg=180.0)
        tracks[oid] = (lon_p, lat_p)
        t_local = np.array(t_s, dtype=float).reshape(-1)
        if t_local.size < arr.shape[0]:
            t_local = np.pad(t_local, (0, arr.shape[0] - t_local.size), mode="edge")
        # For inserted NaNs at dateline splits, approximate expanded time vector linearly.
        if lon_p.size == t_local.size:
            tracks_t[oid] = t_local
        else:
            tracks_t[oid] = np.linspace(float(t_local[0]), float(t_local[-1]), num=lon_p.size, endpoint=True)
        n_frames = max(n_frames, int(lon_p.size))

    if not tracks:
        return

    fig, ax, is_cartopy = _setup_ground_track_axes(
        title="Ground Track Animation (Multi-Object)",
        draw_earth_map=draw_earth_map,
    )

    line_by_obj: dict[str, Any] = {}
    dot_by_obj: dict[str, Any] = {}
    for oid in sorted(tracks.keys()):
        if is_cartopy:
            line, = ax.plot([], [], linewidth=1.4, label=oid, transform=ccrs.PlateCarree(), zorder=3)
            dot, = ax.plot([], [], marker="o", markersize=4, transform=ccrs.PlateCarree(), zorder=4)
        else:
            line, = ax.plot([], [], linewidth=1.4, label=oid, zorder=3)
            dot, = ax.plot([], [], marker="o", markersize=4, zorder=4)
        line_by_obj[oid] = line
        dot_by_obj[oid] = dot
    ax.legend(loc="best")
    time_text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        zorder=10,
    )

    stride = int(max(frame_stride, 1))
    frame_ids = np.arange(0, max(n_frames, 1), stride, dtype=int)
    if frame_ids.size == 0 or frame_ids[-1] != (max(n_frames, 1) - 1):
        frame_ids = np.append(frame_ids, max(n_frames, 1) - 1)

    def update(i: int):
        artists = []
        frame_i = int(frame_ids[i])
        t_now = 0.0
        for oid, (lon_p, lat_p) in tracks.items():
            idx = min(frame_i, lon_p.size - 1)
            line_by_obj[oid].set_data(lon_p[: idx + 1], lat_p[: idx + 1])
            dot_by_obj[oid].set_data([lon_p[idx]], [lat_p[idx]])
            t_track = tracks_t.get(oid)
            if t_track is not None and t_track.size > 0:
                t_now = max(t_now, float(t_track[min(idx, t_track.size - 1)]))
            artists.extend([line_by_obj[oid], dot_by_obj[oid]])
        if jd_utc_start is not None:
            dt_utc = julian_date_to_datetime(float(jd_utc_start) + t_now / 86400.0)
            time_text.set_text(f"UTC: {dt_utc.strftime('%Y-%m-%d %H:%M:%S')}\nSim t: {t_now:.1f} s")
        else:
            time_text.set_text(f"Sim t: {t_now:.1f} s")
        artists.append(time_text)
        return artists

    interval_ms = 1000.0 / max(float(fps) * max(speed_multiple, 1e-6), 1e-3)
    ani = None
    if mode in ("save", "both"):
        ani = animation.FuncAnimation(fig, update, frames=int(frame_ids.size), interval=interval_ms, blit=False)
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'.")
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            ani.save(str(p), fps=max(float(fps), 1.0))
        except Exception as exc:
            print(f"Warning: failed to save animation ({exc}).")
    if mode in ("interactive", "both"):
        # Explicit interactive loop is more reliable than backend animation playback in IDE windows.
        plt.ion()
        fig.show()
        for i in range(int(frame_ids.size)):
            update(i)
            fig.canvas.draw_idle()
            plt.pause(interval_ms / 1000.0)
        plt.ioff()
        plt.show()
    if ani is not None:
        del ani
    plt.close(fig)


# Legacy plotting API re-export wrappers to keep one plotting surface.
def plot_orbit_eci(*args, **kwargs):
    return plot_orbit_eci_legacy(*args, **kwargs)


def plot_attitude_tumble(*args, **kwargs):
    return plot_attitude_tumble_legacy(*args, **kwargs)


def plot_attitude_ric(*args, **kwargs):
    return plot_attitude_ric_legacy(*args, **kwargs)


def plot_angular_rates(*args, **kwargs):
    return plot_angular_rates_legacy(*args, **kwargs)


def plot_ground_track(*args, **kwargs):
    return plot_ground_track_legacy(*args, **kwargs)
