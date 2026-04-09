from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import eci_to_ecef
from sim.dynamics.orbit.epoch import julian_date_to_datetime
from sim.utils.frames import ric_curv_to_rect, ric_dcm_ir_from_rv, ric_rect_to_curv
from sim.utils.ground_track import ground_track_from_eci_history, split_ground_track_dateline
from sim.utils.figure_size import cap_figsize
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
        # Non-blocking show lets master simulation queue all figures first.
        plt.show(block=False)
    else:
        plt.close(fig)


def _play_interactive_animation(
    fig: plt.Figure,
    *,
    update: Any,
    frame_count: int,
    interval_ms: float,
) -> None:
    if frame_count <= 0:
        return
    dt_s = max(float(interval_ms) / 1000.0, 1e-4)
    plt.ion()
    fig.show()
    t0 = perf_counter()
    i = 0
    while i < frame_count:
        if not plt.fignum_exists(fig.number):
            break
        elapsed_s = perf_counter() - t0
        target_i = min(int(elapsed_s / dt_s), frame_count - 1)
        if target_i < i:
            target_i = i
        update(target_i)
        fig.canvas.draw_idle()
        plt.pause(0.001)
        i = target_i + 1
    plt.ioff()
    plt.show()


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
        fig = plt.figure(figsize=cap_figsize(11, 5))
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

    fig, ax = plt.subplots(figsize=cap_figsize(11, 5))
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
        fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
        for i in range(4):
            ax.plot(t_s, q[:, i], label=labels[i])
        ax.set_title(f"Quaternion Components ({frame.upper()} frame)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Quaternion")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    else:
        fig, axes = plt.subplots(4, 1, figsize=cap_figsize(10, 9), sharex=True)
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
        fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
        for i in range(3):
            ax.plot(t_s, w[:, i], label=labels[i])
        ax.set_title(f"Body Angular Rates ({frame.upper()} frame)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("rad/s")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    else:
        fig, axes = plt.subplots(3, 1, figsize=cap_figsize(10, 8), sharex=True)
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


def _first_last_finite_indices(r: np.ndarray) -> tuple[int | None, int | None]:
    arr = np.array(r, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None, None
    mask = np.all(np.isfinite(arr), axis=1)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None, None
    return int(idx[0]), int(idx[-1])


def _draw_earth_sphere_3d(ax: Any, radius_km: float = EARTH_RADIUS_KM) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 24)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color="#6EA8D9", alpha=0.18, linewidth=0.0, zorder=0)
    ax.plot_wireframe(x, y, z, rstride=6, cstride=6, color="#5D86AA", alpha=0.15, linewidth=0.4, zorder=0)


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
    fig = plt.figure(figsize=cap_figsize(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if frame in ("ric_rect", "ric_curv"):
        # Display RIC with radial on y-axis: x=I, y=R, z=C.
        ix, iy, iz = 1, 0, 2
        xlbl, ylbl, zlbl = "I", "R", "C"
    else:
        ix, iy, iz = 0, 1, 2
        xlbl, ylbl, zlbl = "x", "y", "z"
        if frame in ("eci", "ecef"):
            _draw_earth_sphere_3d(ax)
    ax.plot(r[:, ix], r[:, iy], r[:, iz], linewidth=1.4)
    i0, i1 = _first_last_finite_indices(r)
    if i0 is not None:
        ax.scatter([r[i0, ix]], [r[i0, iy]], [r[i0, iz]], color="green", s=28, zorder=5)
    if i1 is not None:
        ax.scatter([r[i1, ix]], [r[i1, iy]], [r[i1, iz]], color="red", s=28, zorder=5)
    ax.set_title(f"Trajectory ({frame.upper()})")
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_zlabel(zlbl)
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
    fig = plt.figure(figsize=cap_figsize(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if frame in ("ric_rect", "ric_curv"):
        # Display RIC with radial on y-axis: x=I, y=R, z=C.
        ix, iy, iz = 1, 0, 2
        xlbl, ylbl, zlbl = "I", "R", "C"
    else:
        ix, iy, iz = 0, 1, 2
        xlbl, ylbl, zlbl = "x", "y", "z"
        if frame in ("eci", "ecef"):
            _draw_earth_sphere_3d(ax)
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
        ax.plot(r[:, ix], r[:, iy], r[:, iz], linewidth=1.4, label=oid)
        i0, i1 = _first_last_finite_indices(r)
        if i0 is not None:
            ax.scatter([r[i0, ix]], [r[i0, iy]], [r[i0, iz]], color="green", s=24, zorder=5)
        if i1 is not None:
            ax.scatter([r[i1, ix]], [r[i1, iy]], [r[i1, iz]], color="red", s=24, zorder=5)
    ax.set_title(f"Trajectories ({frame.upper()})")
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_zlabel(zlbl)
    ax.legend(loc="best")
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)


def _ric_2d_plane_axes(plane: str) -> tuple[int, int, str, str]:
    p = str(plane).strip().lower()
    if p == "ri":
        return 1, 0, "I", "R"
    if p == "ic":
        return 1, 2, "I", "C"
    if p == "rc":
        return 2, 0, "C", "R"
    raise ValueError("plane must be one of: 'ri', 'ic', 'rc'.")


def plot_ric_2d_projections(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    frame: Literal["ric_rect", "ric_curv"] = "ric_rect",
    reference_truth_hist: np.ndarray,
    planes: list[str] | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    if frame not in ("ric_rect", "ric_curv"):
        raise ValueError("frame must be 'ric_rect' or 'ric_curv'.")
    r = _trajectory_in_frame(
        t_s=t_s,
        truth_hist=truth_hist,
        frame=frame,
        reference_truth_hist=reference_truth_hist,
    )
    p_list = planes if planes is not None and len(planes) > 0 else ["ri", "ic", "rc"]
    fig, axes = plt.subplots(1, len(p_list), figsize=cap_figsize(5.0 * len(p_list), 4.5))
    if len(p_list) == 1:
        axes = [axes]
    for ax, p in zip(axes, p_list):
        ix, iy, xlbl, ylbl = _ric_2d_plane_axes(p)
        ax.plot(r[:, ix], r[:, iy], linewidth=1.4)
        i0, i1 = _first_last_finite_indices(r[:, [ix, iy]])
        if i0 is not None:
            ax.scatter([r[i0, ix]], [r[i0, iy]], color="green", s=24, zorder=5)
        if i1 is not None:
            ax.scatter([r[i1, ix]], [r[i1, iy]], color="red", s=24, zorder=5)
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.set_title(f"{xlbl}-{ylbl}")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"RIC 2D Projections ({'Rect' if frame == 'ric_rect' else 'Curvilinear'})")
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)


def plot_multi_ric_2d_projections(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    frame: Literal["ric_rect", "ric_curv"] = "ric_rect",
    reference_truth_hist: np.ndarray,
    planes: list[str] | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    if frame not in ("ric_rect", "ric_curv"):
        raise ValueError("frame must be 'ric_rect' or 'ric_curv'.")
    p_list = planes if planes is not None and len(planes) > 0 else ["ri", "ic", "rc"]
    fig, axes = plt.subplots(1, len(p_list), figsize=cap_figsize(5.0 * len(p_list), 4.5))
    if len(p_list) == 1:
        axes = [axes]
    for oid, hist in truth_hist_by_object.items():
        if hist.size == 0 or not np.any(np.isfinite(hist[:, 0])):
            continue
        r = _trajectory_in_frame(
            t_s=t_s,
            truth_hist=hist,
            frame=frame,
            reference_truth_hist=reference_truth_hist,
        )
        for ax, p in zip(axes, p_list):
            ix, iy, _, _ = _ric_2d_plane_axes(p)
            ax.plot(r[:, ix], r[:, iy], linewidth=1.2, label=oid)
            i0, i1 = _first_last_finite_indices(r[:, [ix, iy]])
            if i0 is not None:
                ax.scatter([r[i0, ix]], [r[i0, iy]], color="green", s=18, zorder=5)
            if i1 is not None:
                ax.scatter([r[i1, ix]], [r[i1, iy]], color="red", s=18, zorder=5)
    for ax, p in zip(axes, p_list):
        _, _, xlbl, ylbl = _ric_2d_plane_axes(p)
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.set_title(f"{xlbl}-{ylbl}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.suptitle(f"RIC 2D Projections Multi ({'Rect' if frame == 'ric_rect' else 'Curvilinear'})")
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)


def animate_multi_ric_2d_projections(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    frame: Literal["ric_rect", "ric_curv"] = "ric_curv",
    reference_truth_hist: np.ndarray,
    planes: list[str] | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
    frame_stride: int = 1,
    show_trajectory: bool = True,
) -> None:
    if frame not in ("ric_rect", "ric_curv"):
        raise ValueError("frame must be 'ric_rect' or 'ric_curv'.")
    trajectories: dict[str, np.ndarray] = {}
    for oid, hist in truth_hist_by_object.items():
        if hist.size == 0 or not np.any(np.isfinite(hist[:, 0])):
            continue
        trajectories[oid] = _trajectory_in_frame(
            t_s=t_s,
            truth_hist=hist,
            frame=frame,
            reference_truth_hist=reference_truth_hist,
        )
    if not trajectories:
        return

    p_list = planes if planes is not None and len(planes) > 0 else ["ri", "ic", "rc"]
    fig, axes = plt.subplots(1, len(p_list), figsize=cap_figsize(5.0 * len(p_list), 4.5))
    if len(p_list) == 1:
        axes = [axes]

    line_by_plane_obj: dict[tuple[str, str], Any] = {}
    dot_by_plane_obj: dict[tuple[str, str], Any] = {}
    ax_by_plane: dict[str, Any] = {}
    for ax, p in zip(axes, p_list):
        _, _, xlbl, ylbl = _ric_2d_plane_axes(p)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.set_title(f"{xlbl}-{ylbl}")
        ax.grid(True, alpha=0.3)
        ax_by_plane[p] = ax
        for oid in sorted(trajectories.keys()):
            line, = ax.plot([], [], linewidth=1.2, label=oid)
            dot, = ax.plot([], [], marker="o", markersize=4)
            line_by_plane_obj[(p, oid)] = line
            dot_by_plane_obj[(p, oid)] = dot
    axes[0].legend(loc="best")

    stride = int(max(frame_stride, 1))
    max_frames = max(arr.shape[0] for arr in trajectories.values())
    frame_ids = np.arange(0, max_frames, stride, dtype=int)
    if frame_ids.size == 0 or frame_ids[-1] != (max_frames - 1):
        frame_ids = np.append(frame_ids, max_frames - 1)

    def update(i: int):
        artists = []
        frame_i = int(frame_ids[i])
        for p in p_list:
            ix, iy, _, _ = _ric_2d_plane_axes(p)
            plane_arrays: list[np.ndarray] = []
            for oid, arr in trajectories.items():
                idx = min(frame_i, arr.shape[0] - 1)
                start = 0 if show_trajectory else idx
                seg = arr[start : idx + 1, :]
                line_by_plane_obj[(p, oid)].set_data(seg[:, ix], seg[:, iy])
                dot_by_plane_obj[(p, oid)].set_data([arr[idx, ix]], [arr[idx, iy]])
                plane_arrays.extend([arr[: idx + 1, ix], arr[: idx + 1, iy]])
                artists.extend([line_by_plane_obj[(p, oid)], dot_by_plane_obj[(p, oid)]])
            lim = _symmetric_limit_from_arrays(plane_arrays, min_lim=1.0, margin=1.15)
            ax_by_plane[p].set_xlim(-lim, lim)
            ax_by_plane[p].set_ylim(-lim, lim)
        t_now = float(t_s[min(frame_i, t_s.size - 1)]) if t_s.size else 0.0
        fig.suptitle(
            f"RIC 2D Projections Animation ({'Curvilinear' if frame == 'ric_curv' else 'Rect'})  t={t_now:.1f}s"
        )
        return artists

    dt = float(np.median(np.diff(t_s))) if t_s.size > 1 else 1.0
    interval_ms = 1000.0 * dt * float(stride) / max(speed_multiple, 1e-6)
    if mode in ("interactive", "both"):
        _play_interactive_animation(fig, update=update, frame_count=int(frame_ids.size), interval_ms=interval_ms)
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
        del ani
    plt.close(fig)


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
        fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
        for i in range(m):
            ax.plot(t_s, u[:, i], label=labels[i])
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(y_label if y_label else "Command")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    else:
        fig, axes = plt.subplots(m, 1, figsize=cap_figsize(10, max(3.0, 2.4 * m)), sharex=True)
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
    fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
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


def _rectangular_prism_vertices_body(lx_m: float, ly_m: float, lz_m: float) -> np.ndarray:
    return np.array(
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


def _rectangular_prism_faces() -> list[list[int]]:
    return [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]


def _attitude_rotation_history(truth_hist: np.ndarray, frame: AttitudeFrame) -> np.ndarray:
    q_bn = np.array(truth_hist[:, 6:10], dtype=float)
    c_anim = np.zeros((truth_hist.shape[0], 3, 3), dtype=float)
    for k in range(truth_hist.shape[0]):
        c_bn = quaternion_to_dcm_bn(q_bn[k, :])
        if frame == "eci":
            c_anim[k, :, :] = c_bn.T
        else:
            r = truth_hist[k, 0:3]
            v = truth_hist[k, 3:6]
            c_ir = ric_dcm_ir_from_rv(r, v)
            c_anim[k, :, :] = c_ir.T @ c_bn.T
    return c_anim


def _rectangular_prism_frame_vertices(
    body_vertices: np.ndarray,
    rotation_history: np.ndarray,
    faces: list[list[int]],
    frame_idx: int,
) -> list[np.ndarray]:
    verts = (rotation_history[frame_idx, :, :] @ body_vertices.T).T
    return [verts[idx, :] for idx in faces]


def _default_thruster_mount_position_body(
    *,
    lx_m: float,
    ly_m: float,
    lz_m: float,
    outward_normal_body: np.ndarray,
) -> np.ndarray:
    dims = np.array([lx_m, ly_m, lz_m], dtype=float)
    axis = np.array(outward_normal_body, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, -0.5 * lz_m], dtype=float)
    axis /= norm
    idx = int(np.argmax(np.abs(axis)))
    pos = np.zeros(3, dtype=float)
    pos[idx] = 0.5 * dims[idx] * float(np.sign(axis[idx]) if abs(axis[idx]) > 1e-12 else 1.0)
    return pos


def _orthonormal_basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.array(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if n <= 1e-12:
        a = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        a /= n
    basis = np.eye(3)
    ref = basis[:, int(np.argmin(np.abs(a)))]
    s = ref - np.dot(ref, a) * a
    s_norm = float(np.linalg.norm(s))
    if s_norm <= 1e-12:
        ref = basis[:, (int(np.argmin(np.abs(a))) + 1) % 3]
        s = ref - np.dot(ref, a) * a
        s_norm = float(np.linalg.norm(s))
    s /= max(s_norm, 1e-12)
    u = np.cross(a, s)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    return a, s, u


def _thruster_face_mount_body(
    *,
    lx_m: float,
    ly_m: float,
    lz_m: float,
    thruster_position_body_m: np.ndarray | None,
    outward_normal_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    half_dims = 0.5 * np.array([lx_m, ly_m, lz_m], dtype=float)
    axis = np.array(outward_normal_body, dtype=float).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        axis = np.array([0.0, 0.0, -1.0], dtype=float)
    else:
        axis /= axis_norm

    if thruster_position_body_m is None:
        mount = _default_thruster_mount_position_body(
            lx_m=lx_m,
            ly_m=ly_m,
            lz_m=lz_m,
            outward_normal_body=axis,
        )
        return mount, axis

    raw_mount = np.array(thruster_position_body_m, dtype=float).reshape(3)
    raw_mount = np.clip(raw_mount, -half_dims, half_dims)
    if float(np.linalg.norm(raw_mount)) > 1e-12:
        scaled = np.abs(raw_mount) / np.maximum(half_dims, 1e-12)
        idx = int(np.argmax(scaled))
        sign = float(np.sign(raw_mount[idx])) if abs(raw_mount[idx]) > 1e-12 else float(np.sign(axis[idx]))
    else:
        idx = int(np.argmax(np.abs(axis)))
        sign = float(np.sign(axis[idx]))
    if abs(sign) <= 1e-12:
        sign = 1.0
    face_axis = np.zeros(3, dtype=float)
    face_axis[idx] = sign
    mount = raw_mount.copy()
    mount[idx] = sign * half_dims[idx]
    return mount, face_axis


def _thruster_marker_geometry_body(
    *,
    lx_m: float,
    ly_m: float,
    lz_m: float,
    thruster_position_body_m: np.ndarray | None = None,
    thruster_direction_body: np.ndarray | None = None,
) -> tuple[np.ndarray, list[list[int]]]:
    dims = np.array([lx_m, ly_m, lz_m], dtype=float)
    marker_scale = float(max(np.min(dims), 1e-6))
    axis_raw = (
        np.array(thruster_direction_body, dtype=float).reshape(3)
        if thruster_direction_body is not None
        else np.array([0.0, 0.0, 1.0], dtype=float)
    )
    # Render the nozzle on the exterior face indicated by the mount location.
    outward_axis_hint = -axis_raw
    mount, outward_axis = _thruster_face_mount_body(
        lx_m=lx_m,
        ly_m=ly_m,
        lz_m=lz_m,
        thruster_position_body_m=thruster_position_body_m,
        outward_normal_body=outward_axis_hint,
    )
    axis, side, up = _orthonormal_basis_from_axis(outward_axis)

    inner_radius = 0.065 * marker_scale
    outer_radius = 0.13 * marker_scale
    collar_radius = 0.17 * marker_scale
    nozzle_length = 0.26 * marker_scale
    base_offset = 0.02 * marker_scale
    exit_offset = base_offset + nozzle_length
    collar_offset = 0.008 * marker_scale
    segments = 24

    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=float)
    unit_ring = np.vstack([np.cos(theta), np.sin(theta)]).T
    base_center = mount + axis * base_offset
    exit_center = mount + axis * exit_offset
    collar_center = mount + axis * collar_offset
    base_ring = np.vstack([base_center + inner_radius * (c * side + s * up) for c, s in unit_ring])
    exit_ring = np.vstack([exit_center + outer_radius * (c * side + s * up) for c, s in unit_ring])
    collar_ring = np.vstack([collar_center + collar_radius * (c * side + s * up) for c, s in unit_ring])
    points = np.vstack([base_ring, exit_ring, collar_ring])

    faces: list[list[int]] = []
    for idx in range(segments):
        nxt = (idx + 1) % segments
        faces.append([idx, nxt, segments + nxt, segments + idx])
    for idx in range(segments):
        nxt = (idx + 1) % segments
        faces.append([2 * segments + idx, 2 * segments + nxt, nxt, idx])
    faces.append(list(range(segments)))
    faces.append(list(range(2 * segments - 1, segments - 1, -1)))
    faces.append(list(range(3 * segments - 1, 2 * segments - 1, -1)))
    return points, faces


def _marker_frame_faces(
    marker_points_body: np.ndarray,
    rotation_history: np.ndarray,
    faces: list[list[int]],
    frame_idx: int,
) -> list[np.ndarray]:
    pts = (rotation_history[frame_idx, :, :] @ marker_points_body.T).T
    return [pts[idx, :] for idx in faces]


def _symmetric_limit_from_arrays(
    arrays: list[np.ndarray],
    *,
    min_lim: float = 1.0,
    margin: float = 1.15,
) -> float:
    lim = 0.0
    for arr in arrays:
        a = np.array(arr, dtype=float)
        finite = a[np.isfinite(a)]
        if finite.size > 0:
            lim = max(lim, float(np.max(np.abs(finite))))
    return float(max(min_lim, margin * lim))


def animate_rectangular_prism_attitude(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    lx_m: float,
    ly_m: float,
    lz_m: float,
    frame: AttitudeFrame = "eci",
    thruster_active_mask: np.ndarray | None = None,
    thruster_position_body_m: np.ndarray | None = None,
    thruster_direction_body: np.ndarray | None = None,
    body_facecolor: str = "#1F77B4",
    thruster_inactive_facecolor: str = "#808080",
    thruster_active_facecolor: str = "#D95F02",
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
) -> None:
    verts_body = _rectangular_prism_vertices_body(lx_m=lx_m, ly_m=ly_m, lz_m=lz_m)
    faces = _rectangular_prism_faces()
    c_anim = _attitude_rotation_history(truth_hist=truth_hist, frame=frame)
    marker_points_body, marker_faces = _thruster_marker_geometry_body(
        lx_m=lx_m,
        ly_m=ly_m,
        lz_m=lz_m,
        thruster_position_body_m=thruster_position_body_m,
        thruster_direction_body=thruster_direction_body,
    )

    active_mask = np.zeros(truth_hist.shape[0], dtype=bool)
    if thruster_active_mask is not None:
        mask_arr = np.array(thruster_active_mask, dtype=bool).reshape(-1)
        n_copy = min(mask_arr.size, active_mask.size)
        active_mask[:n_copy] = mask_arr[:n_copy]

    max_dim = 0.7 * max(lx_m, ly_m, lz_m)
    fig = plt.figure(figsize=cap_figsize(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-max_dim, max_dim)
    ax.set_ylim(-max_dim, max_dim)
    ax.set_zlim(-max_dim, max_dim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"Rectangular Prism Attitude Animation ({frame.upper()})")
    poly = Poly3DCollection([], alpha=0.35, facecolor=body_facecolor, edgecolor="k", linewidth=0.7)
    ax.add_collection3d(poly)
    thruster_poly = Poly3DCollection(
        [],
        alpha=1.0,
        facecolor=thruster_inactive_facecolor,
        edgecolor="#2F241E",
        linewidth=0.85,
    )
    ax.add_collection3d(thruster_poly)

    def _frame_verts(i: int) -> list[np.ndarray]:
        return _rectangular_prism_frame_vertices(
            body_vertices=verts_body,
            rotation_history=c_anim,
            faces=faces,
            frame_idx=i,
        )

    def update(i: int):
        poly.set_verts(_frame_verts(i))
        poly.set_facecolor(body_facecolor)
        thruster_poly.set_verts(
            _marker_frame_faces(
                marker_points_body=marker_points_body,
                rotation_history=c_anim,
                faces=marker_faces,
                frame_idx=i,
            )
        )
        thruster_poly.set_facecolor(thruster_active_facecolor if bool(active_mask[i]) else thruster_inactive_facecolor)
        ax.set_xlabel(f"t={t_s[i]:.1f}s")
        return [poly, thruster_poly]

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


def animate_battlespace_dashboard(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    reference_truth_hist: np.ndarray,
    target_object_id: str = "target",
    chaser_object_id: str = "chaser",
    thrust_hist_by_object: dict[str, np.ndarray] | None = None,
    delta_v_remaining_m_s_by_object: dict[str, np.ndarray] | None = None,
    prism_dims_m_by_object: dict[str, list[float] | np.ndarray] | None = None,
    thruster_mounts_by_object: dict[str, dict[str, np.ndarray] | None] | None = None,
    thruster_active_threshold_km_s2: float = 1e-15,
    show_trajectory: bool = True,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
    frame_stride: int = 1,
) -> None:
    target_hist_raw = np.array(truth_hist_by_object.get(target_object_id, np.array([])), dtype=float)
    chaser_hist_raw = np.array(truth_hist_by_object.get(chaser_object_id, np.array([])), dtype=float)
    ref_hist_raw = np.array(reference_truth_hist, dtype=float)
    if target_hist_raw.ndim != 2 or chaser_hist_raw.ndim != 2 or ref_hist_raw.ndim != 2:
        return
    if target_hist_raw.shape[0] == 0 or chaser_hist_raw.shape[0] == 0 or ref_hist_raw.shape[0] == 0:
        return

    n_frames = min(t_s.size, target_hist_raw.shape[0], chaser_hist_raw.shape[0], ref_hist_raw.shape[0])
    if n_frames <= 0:
        return

    t_plot = np.array(t_s[:n_frames], dtype=float)
    target_hist = target_hist_raw[:n_frames, :]
    chaser_hist = chaser_hist_raw[:n_frames, :]
    ref_hist = ref_hist_raw[:n_frames, :]

    rel_truth_by_object = {
        target_object_id: target_hist,
        chaser_object_id: chaser_hist,
    }
    curv_traj_by_object = {
        oid: _trajectory_in_frame(
            t_s=t_plot,
            truth_hist=hist,
            frame="ric_curv",
            reference_truth_hist=ref_hist,
        )
        for oid, hist in rel_truth_by_object.items()
    }

    default_dims = np.array([4.0, 2.0, 2.0], dtype=float)
    dims_map = prism_dims_m_by_object or {}
    mount_map = thruster_mounts_by_object or {}
    body_vertices_by_object: dict[str, np.ndarray] = {}
    marker_points_by_object: dict[str, np.ndarray] = {}
    marker_faces_by_object: dict[str, list[list[int]]] = {}
    rotations_by_object: dict[str, np.ndarray] = {}
    active_by_object: dict[str, np.ndarray] = {}
    faces = _rectangular_prism_faces()
    for oid, hist in rel_truth_by_object.items():
        dims = np.array(dims_map.get(oid, default_dims), dtype=float).reshape(-1)
        if dims.size != 3 or not np.all(np.isfinite(dims)) or np.any(dims <= 0.0):
            dims = default_dims.copy()
        body_vertices_by_object[oid] = _rectangular_prism_vertices_body(
            lx_m=float(dims[0]),
            ly_m=float(dims[1]),
            lz_m=float(dims[2]),
        )
        mount = mount_map.get(oid) if isinstance(mount_map.get(oid), dict) else {}
        marker_points_by_object[oid], marker_faces_by_object[oid] = _thruster_marker_geometry_body(
            lx_m=float(dims[0]),
            ly_m=float(dims[1]),
            lz_m=float(dims[2]),
            thruster_position_body_m=None if not isinstance(mount, dict) else mount.get("position_body_m"),
            thruster_direction_body=None if not isinstance(mount, dict) else mount.get("direction_body"),
        )
        rotations_by_object[oid] = _attitude_rotation_history(truth_hist=hist, frame="ric")
        thrust_hist = np.array((thrust_hist_by_object or {}).get(oid, np.zeros((n_frames, 3))), dtype=float)
        thrust_local = thrust_hist[:n_frames, :] if thrust_hist.ndim == 2 else np.zeros((n_frames, 3), dtype=float)
        active_by_object[oid] = np.linalg.norm(np.nan_to_num(thrust_local, nan=0.0), axis=1) > float(
            thruster_active_threshold_km_s2
        )

    dv_remaining_by_object: dict[str, np.ndarray] = {}
    for oid in (target_object_id, chaser_object_id):
        arr = np.array((delta_v_remaining_m_s_by_object or {}).get(oid, np.full(n_frames, np.nan)), dtype=float).reshape(-1)
        dv_remaining_by_object[oid] = arr[:n_frames] if arr.size >= n_frames else np.pad(arr, (0, n_frames - arr.size), constant_values=np.nan)

    rel_r_km = chaser_hist[:, 0:3] - target_hist[:, 0:3]
    rel_v_km_s = chaser_hist[:, 3:6] - target_hist[:, 3:6]
    rel_range_km = np.linalg.norm(rel_r_km, axis=1)
    rel_speed_km_s = np.linalg.norm(rel_v_km_s, axis=1)

    fig = plt.figure(figsize=cap_figsize(12, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 0.85])
    ax_ri = fig.add_subplot(gs[0, 0])
    ax_chaser = fig.add_subplot(gs[0, 1], projection="3d")
    ax_rc = fig.add_subplot(gs[1, 0])
    ax_target = fig.add_subplot(gs[1, 1], projection="3d")

    color_by_object = {
        target_object_id: "#1F77B4",
        chaser_object_id: "#D62728",
    }
    thruster_inactive_facecolor = "#808080"
    thruster_active_facecolor = "#D95F02"

    for ax, plane, lim, title in (
        (ax_ri, "ri", 1.0, "RI Relative Motion"),
        (ax_rc, "rc", 1.0, "RC Relative Motion"),
    ):
        _, _, xlbl, ylbl = _ric_2d_plane_axes(plane)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"{xlbl} (km)")
        ax.set_ylabel(f"{ylbl} (km)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    prism_poly_by_object: dict[str, Poly3DCollection] = {}
    thruster_poly_by_object: dict[str, Poly3DCollection] = {}
    for oid, ax, title in (
        (chaser_object_id, ax_chaser, "Chaser Attitude + Thrust (RIC)"),
        (target_object_id, ax_target, "Target Attitude + Thrust (RIC)"),
    ):
        body_vertices = body_vertices_by_object[oid]
        body_span = np.ptp(body_vertices, axis=0)
        lim = 0.7 * float(max(np.max(body_span), 1.0))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22.0, azim=35.0)
        ax.set_xlabel("R (m)")
        ax.set_ylabel("I (m)")
        ax.set_zlabel("C (m)")
        ax.set_title(title)
        poly = Poly3DCollection([], alpha=0.35, facecolor=color_by_object[oid], edgecolor="k", linewidth=0.7)
        ax.add_collection3d(poly)
        prism_poly_by_object[oid] = poly
        thruster_poly = Poly3DCollection(
            [],
            alpha=1.0,
            facecolor=thruster_inactive_facecolor,
            edgecolor="#2F241E",
            linewidth=0.85,
        )
        ax.add_collection3d(thruster_poly)
        thruster_poly_by_object[oid] = thruster_poly

    ri_line_by_object: dict[str, Any] = {}
    ri_dot_by_object: dict[str, Any] = {}
    rc_line_by_object: dict[str, Any] = {}
    rc_dot_by_object: dict[str, Any] = {}
    ri_ix, ri_iy, _, _ = _ric_2d_plane_axes("ri")
    rc_ix, rc_iy, _, _ = _ric_2d_plane_axes("rc")
    for oid in (target_object_id, chaser_object_id):
        color = color_by_object[oid]
        ri_line, = ax_ri.plot([], [], linewidth=1.5, color=color, label=oid)
        ri_dot, = ax_ri.plot([], [], marker="o", markersize=5, color=color)
        rc_line, = ax_rc.plot([], [], linewidth=1.5, color=color, label=oid)
        rc_dot, = ax_rc.plot([], [], marker="o", markersize=5, color=color)
        ri_line_by_object[oid] = ri_line
        ri_dot_by_object[oid] = ri_dot
        rc_line_by_object[oid] = rc_line
        rc_dot_by_object[oid] = rc_dot
    ax_ri.legend(loc="best")
    ax_rc.legend(loc="best")

    fig.suptitle("Battlespace Visualization Dashboard", fontsize=14)
    status_text = fig.text(
        0.5,
        0.015,
        "",
        ha="center",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.95])

    stride = int(max(frame_stride, 1))
    frame_ids = np.arange(0, n_frames, stride, dtype=int)
    if frame_ids.size == 0 or frame_ids[-1] != (n_frames - 1):
        frame_ids = np.append(frame_ids, n_frames - 1)

    def _dv_text(oid: str, idx: int) -> str:
        dv_val = float(dv_remaining_by_object[oid][idx])
        if np.isfinite(dv_val):
            return f"{dv_val:7.2f} m/s"
        return "   n/a  "

    def update(i: int):
        artists: list[Any] = []
        frame_i = int(frame_ids[i])
        for oid, traj in curv_traj_by_object.items():
            start = 0 if show_trajectory else frame_i
            seg = traj[start : frame_i + 1, :]
            ri_line_by_object[oid].set_data(seg[:, ri_ix], seg[:, ri_iy])
            ri_dot_by_object[oid].set_data([traj[frame_i, ri_ix]], [traj[frame_i, ri_iy]])
            rc_line_by_object[oid].set_data(seg[:, rc_ix], seg[:, rc_iy])
            rc_dot_by_object[oid].set_data([traj[frame_i, rc_ix]], [traj[frame_i, rc_iy]])
            artists.extend(
                [
                    ri_line_by_object[oid],
                    ri_dot_by_object[oid],
                    rc_line_by_object[oid],
                    rc_dot_by_object[oid],
                ]
            )

        ri_lim = _symmetric_limit_from_arrays(
            [
                traj[: frame_i + 1, ri_ix]
                for traj in curv_traj_by_object.values()
            ]
            + [
                traj[: frame_i + 1, ri_iy]
                for traj in curv_traj_by_object.values()
            ],
            min_lim=1.0,
            margin=1.15,
        )
        rc_lim = _symmetric_limit_from_arrays(
            [
                traj[: frame_i + 1, rc_ix]
                for traj in curv_traj_by_object.values()
            ]
            + [
                traj[: frame_i + 1, rc_iy]
                for traj in curv_traj_by_object.values()
            ],
            min_lim=1.0,
            margin=1.15,
        )
        ax_ri.set_xlim(-ri_lim, ri_lim)
        ax_ri.set_ylim(-ri_lim, ri_lim)
        ax_rc.set_xlim(-rc_lim, rc_lim)
        ax_rc.set_ylim(-rc_lim, rc_lim)

        for oid in (chaser_object_id, target_object_id):
            prism_poly_by_object[oid].set_verts(
                _rectangular_prism_frame_vertices(
                    body_vertices=body_vertices_by_object[oid],
                    rotation_history=rotations_by_object[oid],
                    faces=faces,
                    frame_idx=frame_i,
                )
            )
            prism_poly_by_object[oid].set_facecolor(color_by_object[oid])
            thruster_poly_by_object[oid].set_verts(
                _marker_frame_faces(
                    marker_points_body=marker_points_by_object[oid],
                    rotation_history=rotations_by_object[oid],
                    faces=marker_faces_by_object[oid],
                    frame_idx=frame_i,
                )
            )
            thruster_poly_by_object[oid].set_facecolor(
                thruster_active_facecolor if bool(active_by_object[oid][frame_i]) else thruster_inactive_facecolor
            )
            artists.append(prism_poly_by_object[oid])
            artists.append(thruster_poly_by_object[oid])

        status_text.set_text(
            "\n".join(
                [
                    f"t = {t_plot[frame_i]:7.1f} s   Relative Range = {rel_range_km[frame_i]:8.3f} km   Relative Speed = {rel_speed_km_s[frame_i]:8.5f} km/s",
                    f"Chaser dV Remaining = {_dv_text(chaser_object_id, frame_i)}   Target dV Remaining = {_dv_text(target_object_id, frame_i)}",
                ]
            )
        )
        artists.append(status_text)
        return artists

    dt = float(np.median(np.diff(t_plot))) if t_plot.size > 1 else 1.0
    interval_ms = 1000.0 * dt * float(stride) / max(speed_multiple, 1e-6)
    if mode in ("interactive", "both"):
        _play_interactive_animation(fig, update=update, frame_count=int(frame_ids.size), interval_ms=interval_ms)
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
        del ani
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
    lim = _symmetric_limit_from_arrays([r[:, 0], r[:, 1], r[:, 2]], min_lim=1.0, margin=1.0)
    fig = plt.figure(figsize=cap_figsize(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    init_lim = 1.0 if frame in ("ric_rect", "ric_curv") else lim
    ax.set_xlim(-init_lim, init_lim)
    ax.set_ylim(-init_lim, init_lim)
    ax.set_zlim(-init_lim, init_lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"Trajectory Animation ({frame.upper()})")
    line, = ax.plot([], [], [], linewidth=1.4)
    dot, = ax.plot([], [], [], marker="o", markersize=4)

    def update(i: int):
        line.set_data(r[: i + 1, 0], r[: i + 1, 1])
        line.set_3d_properties(r[: i + 1, 2])
        dot.set_data([r[i, 0]], [r[i, 1]])
        dot.set_3d_properties([r[i, 2]])
        if frame in ("ric_rect", "ric_curv"):
            lim = _symmetric_limit_from_arrays(
                [r[: i + 1, 0], r[: i + 1, 1], r[: i + 1, 2]],
                min_lim=1.0,
                margin=1.15,
            )
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
        ax.set_xlabel(f"t={t_s[i]:.1f}s")
        return [line, dot]

    dt = float(np.median(np.diff(t_s))) if t_s.size > 1 else 1.0
    interval_ms = 1000.0 * dt / max(speed_multiple, 1e-6)
    if mode in ("interactive", "both"):
        _play_interactive_animation(fig, update=update, frame_count=int(t_s.size), interval_ms=interval_ms)
    if mode in ("save", "both"):
        ani = animation.FuncAnimation(fig, update, frames=t_s.size, interval=interval_ms, blit=False)
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'.")
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            ani.save(str(p), fps=max(float(fps), 1.0))
        except Exception as exc:
            print(f"Warning: failed to save animation ({exc}).")
        del ani
    plt.close(fig)


def animate_multi_trajectory_frame(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    frame: FrameName = "eci",
    jd_utc_start: float | None = None,
    reference_truth_hist: np.ndarray | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
    frame_stride: int = 1,
    show_trajectory: bool = True,
) -> None:
    trajectories: dict[str, np.ndarray] = {}
    for oid, hist in truth_hist_by_object.items():
        arr = np.array(hist, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or not np.any(np.isfinite(arr[:, 0])):
            continue
        trajectories[oid] = _trajectory_in_frame(
            t_s=t_s,
            truth_hist=arr,
            frame=frame,
            jd_utc_start=jd_utc_start,
            reference_truth_hist=reference_truth_hist,
        )
    if not trajectories:
        return

    lim = 0.0
    for arr in trajectories.values():
        lim = max(lim, _symmetric_limit_from_arrays([arr[:, 0], arr[:, 1], arr[:, 2]], min_lim=1.0, margin=1.0))

    fig = plt.figure(figsize=cap_figsize(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if frame in ("ric_rect", "ric_curv"):
        ix, iy, iz = 1, 0, 2
        xlbl, ylbl, zlbl = "I", "R", "C"
    else:
        ix, iy, iz = 0, 1, 2
        xlbl, ylbl, zlbl = "x", "y", "z"
        if frame in ("eci", "ecef"):
            _draw_earth_sphere_3d(ax)
    init_lim = 1.0 if frame in ("ric_rect", "ric_curv") else lim
    ax.set_xlim(-init_lim, init_lim)
    ax.set_ylim(-init_lim, init_lim)
    ax.set_zlim(-init_lim, init_lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"Trajectories Animation ({frame.upper()})")
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_zlabel(zlbl)

    line_by_obj: dict[str, Any] = {}
    dot_by_obj: dict[str, Any] = {}
    for oid in sorted(trajectories.keys()):
        line, = ax.plot([], [], [], linewidth=1.4, label=oid)
        dot, = ax.plot([], [], [], marker="o", markersize=4)
        line_by_obj[oid] = line
        dot_by_obj[oid] = dot
    ax.legend(loc="best")

    stride = int(max(frame_stride, 1))
    max_frames = max(arr.shape[0] for arr in trajectories.values())
    frame_ids = np.arange(0, max_frames, stride, dtype=int)
    if frame_ids.size == 0 or frame_ids[-1] != (max_frames - 1):
        frame_ids = np.append(frame_ids, max_frames - 1)

    def update(i: int):
        artists = []
        frame_i = int(frame_ids[i])
        scale_arrays: list[np.ndarray] = []
        for oid, arr in trajectories.items():
            idx = min(frame_i, arr.shape[0] - 1)
            start = 0 if show_trajectory else idx
            seg = arr[start : idx + 1, :]
            line_by_obj[oid].set_data(seg[:, ix], seg[:, iy])
            line_by_obj[oid].set_3d_properties(seg[:, iz])
            dot_by_obj[oid].set_data([arr[idx, ix]], [arr[idx, iy]])
            dot_by_obj[oid].set_3d_properties([arr[idx, iz]])
            if frame in ("ric_rect", "ric_curv"):
                scale_arrays.extend([arr[: idx + 1, ix], arr[: idx + 1, iy], arr[: idx + 1, iz]])
            artists.extend([line_by_obj[oid], dot_by_obj[oid]])
        if frame in ("ric_rect", "ric_curv"):
            lim = _symmetric_limit_from_arrays(scale_arrays, min_lim=1.0, margin=1.15)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
        t_now = float(t_s[min(frame_i, t_s.size - 1)]) if t_s.size else 0.0
        ax.set_title(f"Trajectories Animation ({frame.upper()})  t={t_now:.1f}s")
        return artists

    dt = float(np.median(np.diff(t_s))) if t_s.size > 1 else 1.0
    interval_ms = 1000.0 * dt * float(stride) / max(speed_multiple, 1e-6)
    if mode in ("interactive", "both"):
        _play_interactive_animation(fig, update=update, frame_count=int(frame_ids.size), interval_ms=interval_ms)
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
        del ani
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
        del ani
    plt.close(fig)


def animate_multi_rectangular_prism_ric_curv(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    target_object_id: str = "target",
    object_ids: list[str] | None = None,
    prism_dims_m_by_object: dict[str, list[float] | tuple[float, float, float]] | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
    frame_stride: int = 1,
) -> None:
    """Animate multiple spacecraft as rectangular prisms in target-centered curvilinear RIC.

    Display axes are arranged as (I, C, R), so radial is the vertical axis.
    """
    target_hist = truth_hist_by_object.get(target_object_id)
    if target_hist is None:
        return
    tgt = np.array(target_hist, dtype=float)
    if tgt.ndim != 2 or tgt.shape[0] == 0 or tgt.shape[1] < 10:
        return

    all_ids = sorted(list(truth_hist_by_object.keys()))
    if object_ids is None:
        if "target" in all_ids and "chaser" in all_ids:
            obj_ids = ["target", "chaser"]
        else:
            obj_ids = all_ids
    else:
        obj_ids = [oid for oid in object_ids if oid in truth_hist_by_object]
    if not obj_ids:
        return
    if target_object_id not in obj_ids:
        obj_ids = [target_object_id, *obj_ids]

    dims_map = dict(prism_dims_m_by_object or {})
    default_dims_m = np.array([4.0, 2.0, 2.0], dtype=float)
    perm_icr_from_ric = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    n_frames = int(min([t_s.size] + [np.array(truth_hist_by_object[oid], dtype=float).shape[0] for oid in obj_ids] + [tgt.shape[0]]))
    if n_frames <= 0:
        return
    t_loc = np.array(t_s[:n_frames], dtype=float)

    pos_by_obj: dict[str, np.ndarray] = {}
    c_by_obj: dict[str, np.ndarray] = {}
    verts_body_km_by_obj: dict[str, np.ndarray] = {}

    for oid in obj_ids:
        hist = np.array(truth_hist_by_object[oid], dtype=float)
        arr = hist[:n_frames, :]
        pos_by_obj[oid] = np.full((n_frames, 3), np.nan, dtype=float)
        c_by_obj[oid] = np.full((n_frames, 3, 3), np.nan, dtype=float)
        dims = np.array(dims_map.get(oid, default_dims_m), dtype=float).reshape(-1)
        if dims.size != 3:
            dims = default_dims_m.copy()
        lx_km, ly_km, lz_km = (dims * 1e-3).tolist()
        verts_body_km_by_obj[oid] = np.array(
            [
                [-0.5 * lx_km, -0.5 * ly_km, -0.5 * lz_km],
                [-0.5 * lx_km, -0.5 * ly_km, +0.5 * lz_km],
                [-0.5 * lx_km, +0.5 * ly_km, -0.5 * lz_km],
                [-0.5 * lx_km, +0.5 * ly_km, +0.5 * lz_km],
                [+0.5 * lx_km, -0.5 * ly_km, -0.5 * lz_km],
                [+0.5 * lx_km, -0.5 * ly_km, +0.5 * lz_km],
                [+0.5 * lx_km, +0.5 * ly_km, -0.5 * lz_km],
                [+0.5 * lx_km, +0.5 * ly_km, +0.5 * lz_km],
            ],
            dtype=float,
        )

        for k in range(n_frames):
            r_t = tgt[k, 0:3]
            v_t = tgt[k, 3:6]
            if not (np.all(np.isfinite(r_t)) and np.all(np.isfinite(v_t))):
                continue
            c_ir = ric_dcm_ir_from_rv(r_t, v_t)

            r = arr[k, 0:3]
            q_bn = arr[k, 6:10]
            if not (np.all(np.isfinite(r)) and np.all(np.isfinite(q_bn))):
                continue

            dr_rect = c_ir.T @ (r - r_t)
            x_curv = ric_rect_to_curv(
                np.hstack((dr_rect, np.zeros(3, dtype=float))),
                r0_km=float(np.linalg.norm(r_t)),
            )
            ric_curv_pos = x_curv[:3]  # [R, I, C] in km-equivalent curvilinear coordinates
            pos_by_obj[oid][k, :] = np.array([ric_curv_pos[1], ric_curv_pos[2], ric_curv_pos[0]], dtype=float)

            c_bn = quaternion_to_dcm_bn(q_bn)
            c_rb = c_ir.T @ c_bn.T  # body -> RIC
            c_by_obj[oid][k, :, :] = perm_icr_from_ric @ c_rb  # body -> display (I,C,R)

    # Global limits over all objects and all frames.
    all_pos = np.vstack([v for v in pos_by_obj.values() if v.size > 0])
    finite = np.isfinite(all_pos)
    if not np.any(finite):
        return
    lim = float(max(np.nanmax(np.abs(all_pos)), 1.0))

    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]

    fig = plt.figure(figsize=cap_figsize(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("I (km)")
    ax.set_ylabel("C (km)")
    ax.set_zlabel("R (km)")
    ax.set_title("Target-Centered Curvilinear RIC Prism Animation")

    cmap = plt.get_cmap("tab10")
    poly_by_obj: dict[str, Poly3DCollection] = {}
    trail_by_obj: dict[str, Any] = {}
    dot_by_obj: dict[str, Any] = {}
    for i, oid in enumerate(obj_ids):
        color = cmap(i % 10)
        poly = Poly3DCollection([], alpha=0.35, facecolor=color, edgecolor="k", linewidth=0.7)
        ax.add_collection3d(poly)
        poly_by_obj[oid] = poly
        trail, = ax.plot([], [], [], linewidth=1.2, color=color, label=oid)
        dot, = ax.plot([], [], [], marker="o", markersize=4, color=color)
        trail_by_obj[oid] = trail
        dot_by_obj[oid] = dot
    ax.legend(loc="best")

    frame_ids = np.arange(0, n_frames, max(int(frame_stride), 1), dtype=int)
    if frame_ids.size == 0 or frame_ids[-1] != (n_frames - 1):
        frame_ids = np.append(frame_ids, n_frames - 1)

    def _frame_verts(oid: str, i_frame: int) -> list[np.ndarray] | None:
        p = pos_by_obj[oid][i_frame, :]
        c_bd = c_by_obj[oid][i_frame, :, :]
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(c_bd))):
            return None
        verts = (c_bd @ verts_body_km_by_obj[oid].T).T + p
        return [verts[idx, :] for idx in faces]

    def update(i: int):
        k = int(frame_ids[i])
        artists: list[Any] = []
        for oid in obj_ids:
            poly = poly_by_obj[oid]
            fv = _frame_verts(oid, k)
            if fv is None:
                poly.set_verts([])
            else:
                poly.set_verts(fv)
            artists.append(poly)

            tr = trail_by_obj[oid]
            dd = dot_by_obj[oid]
            p = pos_by_obj[oid][: k + 1, :]
            mask = np.all(np.isfinite(p), axis=1)
            if np.any(mask):
                p_ok = p[mask, :]
                tr.set_data(p_ok[:, 0], p_ok[:, 1])
                tr.set_3d_properties(p_ok[:, 2])
                dd.set_data([p_ok[-1, 0]], [p_ok[-1, 1]])
                dd.set_3d_properties([p_ok[-1, 2]])
            else:
                tr.set_data([], [])
                tr.set_3d_properties([])
                dd.set_data([], [])
                dd.set_3d_properties([])
            artists.extend([tr, dd])
        ax.set_title(f"Target-Centered Curvilinear RIC Prism Animation (t={t_loc[k]:.1f}s)")
        return artists

    dt = float(np.median(np.diff(t_loc))) if t_loc.size > 1 else 1.0
    interval_ms = 1000.0 * dt / max(speed_multiple, 1e-6)
    ani = animation.FuncAnimation(fig, update, frames=int(frame_ids.size), interval=interval_ms, blit=False)

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


def animate_side_by_side_rectangular_prism_ric_attitude(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    left_object_id: str = "target",
    right_object_id: str = "chaser",
    prism_dims_m_by_object: dict[str, list[float] | tuple[float, float, float]] | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
    frame_stride: int = 1,
) -> None:
    left_hist = np.array(truth_hist_by_object.get(left_object_id, np.empty((0, 14))), dtype=float)
    right_hist = np.array(truth_hist_by_object.get(right_object_id, np.empty((0, 14))), dtype=float)
    if left_hist.ndim != 2 or right_hist.ndim != 2:
        return
    n_frames = int(min(t_s.size, left_hist.shape[0], right_hist.shape[0]))
    if n_frames <= 0:
        return
    t_loc = np.array(t_s[:n_frames], dtype=float)
    left_hist = left_hist[:n_frames, :]
    right_hist = right_hist[:n_frames, :]

    dims_map = dict(prism_dims_m_by_object or {})
    default_dims_m = np.array([4.0, 2.0, 2.0], dtype=float)
    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]

    def _dims_km(oid: str) -> np.ndarray:
        dims = np.array(dims_map.get(oid, default_dims_m), dtype=float).reshape(-1)
        if dims.size != 3:
            dims = default_dims_m.copy()
        return dims * 1e-3

    def _verts_body_km(oid: str) -> np.ndarray:
        lx_km, ly_km, lz_km = _dims_km(oid).tolist()
        return np.array(
            [
                [-0.5 * lx_km, -0.5 * ly_km, -0.5 * lz_km],
                [-0.5 * lx_km, -0.5 * ly_km, +0.5 * lz_km],
                [-0.5 * lx_km, +0.5 * ly_km, -0.5 * lz_km],
                [-0.5 * lx_km, +0.5 * ly_km, +0.5 * lz_km],
                [+0.5 * lx_km, -0.5 * ly_km, -0.5 * lz_km],
                [+0.5 * lx_km, -0.5 * ly_km, +0.5 * lz_km],
                [+0.5 * lx_km, +0.5 * ly_km, -0.5 * lz_km],
                [+0.5 * lx_km, +0.5 * ly_km, +0.5 * lz_km],
            ],
            dtype=float,
        )

    left_verts_body = _verts_body_km(left_object_id)
    right_verts_body = _verts_body_km(right_object_id)
    lim_km = float(max(np.max(np.abs(left_verts_body)), np.max(np.abs(right_verts_body)), 1e-3)) * 2.2

    def _body_to_ric_dcm(hist: np.ndarray) -> np.ndarray:
        c_arr = np.full((n_frames, 3, 3), np.nan, dtype=float)
        for k in range(n_frames):
            r = hist[k, 0:3]
            v = hist[k, 3:6]
            q_bn = hist[k, 6:10]
            if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v)) and np.all(np.isfinite(q_bn))):
                continue
            c_bn = quaternion_to_dcm_bn(q_bn)
            c_ir = ric_dcm_ir_from_rv(r, v)
            c_arr[k, :, :] = c_ir.T @ c_bn.T  # body -> local RIC
        return c_arr

    c_left = _body_to_ric_dcm(left_hist)
    c_right = _body_to_ric_dcm(right_hist)

    fig = plt.figure(figsize=cap_figsize(12, 6))
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")
    for ax, title in ((ax_left, left_object_id), (ax_right, right_object_id)):
        ax.set_xlim(-lim_km, lim_km)
        ax.set_ylim(-lim_km, lim_km)
        ax.set_zlim(-lim_km, lim_km)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("I (km)")
        ax.set_ylabel("R (km)")
        ax.set_zlabel("C (km)")
        ax.set_title(f"{title} Body in Local RIC")

    poly_left = Poly3DCollection([], alpha=0.4, facecolor="#4C9F70", edgecolor="k", linewidth=0.7)
    poly_right = Poly3DCollection([], alpha=0.4, facecolor="#2E86C1", edgecolor="k", linewidth=0.7)
    ax_left.add_collection3d(poly_left)
    ax_right.add_collection3d(poly_right)

    frame_ids = np.arange(0, n_frames, max(int(frame_stride), 1), dtype=int)
    if frame_ids.size == 0 or frame_ids[-1] != (n_frames - 1):
        frame_ids = np.append(frame_ids, n_frames - 1)

    def _frame_verts(c_arr: np.ndarray, verts_body: np.ndarray, i_frame: int) -> list[np.ndarray] | None:
        c_rb = c_arr[i_frame, :, :]
        if not np.all(np.isfinite(c_rb)):
            return None
        verts = (c_rb @ verts_body.T).T
        return [verts[idx, :] for idx in faces]

    def update(i: int):
        k = int(frame_ids[i])
        lv = _frame_verts(c_left, left_verts_body, k)
        rv = _frame_verts(c_right, right_verts_body, k)
        poly_left.set_verts([] if lv is None else lv)
        poly_right.set_verts([] if rv is None else rv)
        fig.suptitle(f"Side-by-Side Local RIC Attitude Animation (t={t_loc[k]:.1f}s)")
        return [poly_left, poly_right]

    dt = float(np.median(np.diff(t_loc))) if t_loc.size > 1 else 1.0
    interval_ms = 1000.0 * dt / max(speed_multiple, 1e-6)
    ani = animation.FuncAnimation(fig, update, frames=int(frame_ids.size), interval=interval_ms, blit=False)

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
