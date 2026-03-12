from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sim.dynamics.orbit.frames import eci_to_ecef
from sim.utils.frames import ric_curv_to_rect, ric_dcm_ir_from_rv, ric_rect_to_curv
from sim.utils.ground_track import split_ground_track_dateline
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
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
) -> None:
    lon_p, lat_p = split_ground_track_dateline(lon_deg=lon_deg, lat_deg=lat_deg, jump_threshold_deg=180.0)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Ground Track Animation")
    ax.grid(True, alpha=0.3)
    line, = ax.plot([], [], linewidth=1.4)
    dot, = ax.plot([], [], marker="o", markersize=4)

    def update(i: int):
        line.set_data(lon_p[: i + 1], lat_p[: i + 1])
        dot.set_data([lon_p[i]], [lat_p[i]])
        return [line, dot]

    interval_ms = 1000.0 / max(float(fps) * max(speed_multiple, 1e-6), 1e-3)
    ani = animation.FuncAnimation(fig, update, frames=len(lon_p), interval=interval_ms, blit=False)
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
