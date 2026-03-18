from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.utils.figure_size import cap_figsize
from sim.utils.frames import dcm_to_euler_321, ric_dcm_ir_from_rv
from sim.utils.ground_track import split_ground_track_dateline
from sim.utils.quaternion import quaternion_to_dcm_bn

try:
    import cartopy.crs as ccrs  # type: ignore
    import cartopy.feature as cfeature  # type: ignore

    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False


PlotMode = Literal["interactive", "save", "both"]


def _draw_earth_sphere_3d(ax: plt.Axes, radius_km: float = EARTH_RADIUS_KM) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 24)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color="#6EA8D9", alpha=0.18, linewidth=0.0, zorder=0)
    ax.plot_wireframe(x, y, z, rstride=6, cstride=6, color="#5D86AA", alpha=0.15, linewidth=0.4, zorder=0)


def plot_orbit_eci(truth_hist: np.ndarray, mode: PlotMode = "interactive", out_path: str | None = None) -> None:
    r = truth_hist[:, :3]
    fig = plt.figure(figsize=cap_figsize(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    _draw_earth_sphere_3d(ax)
    ax.plot(r[:, 0], r[:, 1], r[:, 2], linewidth=1.5)
    mask = np.all(np.isfinite(r), axis=1)
    idx = np.where(mask)[0]
    if idx.size > 0:
        i0 = int(idx[0])
        i1 = int(idx[-1])
        ax.scatter([r[i0, 0]], [r[i0, 1]], [r[i0, 2]], color="green", s=30, zorder=5)
        ax.scatter([r[i1, 0]], [r[i1, 1]], [r[i1, 2]], color="red", s=30, zorder=5)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title("One-Orbit ECI Trajectory")
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    if mode in ("save", "both"):
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    if mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)


def plot_attitude_tumble(
    t_s: np.ndarray, truth_hist: np.ndarray, mode: PlotMode = "interactive", out_path: str | None = None
) -> None:
    q = truth_hist[:, 6:10]
    w = truth_hist[:, 10:13]

    fig, axes = plt.subplots(2, 1, figsize=cap_figsize(10, 7), sharex=True)
    axes[0].plot(t_s, q[:, 0], label="q0")
    axes[0].plot(t_s, q[:, 1], label="q1")
    axes[0].plot(t_s, q[:, 2], label="q2")
    axes[0].plot(t_s, q[:, 3], label="q3")
    axes[0].set_ylabel("Quaternion")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_s, w[:, 0], label="wx")
    axes[1].plot(t_s, w[:, 1], label="wy")
    axes[1].plot(t_s, w[:, 2], label="wz")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angular rate (rad/s)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if mode in ("save", "both"):
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    if mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)


def plot_attitude_ric(
    t_s: np.ndarray, truth_hist: np.ndarray, mode: PlotMode = "interactive", out_path: str | None = None
) -> None:
    # Internal 3-2-1 extraction on RIC axes [R, I, C]:
    #   roll_321 -> about R, pitch_321 -> about I, yaw_321 -> about C.
    # User convention for this project:
    #   yaw -> about R, roll -> about I, pitch -> about C.
    euler_321_deg = np.zeros((truth_hist.shape[0], 3))
    for k in range(truth_hist.shape[0]):
        r = truth_hist[k, :3]
        v = truth_hist[k, 3:6]
        q_bn = truth_hist[k, 6:10]
        c_bn = quaternion_to_dcm_bn(q_bn)
        c_ir = ric_dcm_ir_from_rv(r, v)
        c_br = c_bn @ c_ir
        euler_321_deg[k, :] = np.rad2deg(dcm_to_euler_321(c_br))

    yaw_about_r_deg = euler_321_deg[:, 0]
    roll_about_i_deg = euler_321_deg[:, 1]
    pitch_about_c_deg = euler_321_deg[:, 2]

    fig, axes = plt.subplots(3, 1, figsize=cap_figsize(10, 8), sharex=True)
    axes[0].plot(t_s, yaw_about_r_deg, linewidth=1.4)
    axes[0].set_ylabel("yaw about R (deg)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_s, roll_about_i_deg, linewidth=1.4)
    axes[1].set_ylabel("roll about I (deg)")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t_s, pitch_about_c_deg, linewidth=1.4)
    axes[2].set_ylabel("pitch about C (deg)")
    axes[2].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Body Attitude Relative to RIC Frame (Project Axis Convention)")

    fig.tight_layout()
    if mode in ("save", "both"):
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    if mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)


def plot_angular_rates(
    t_s: np.ndarray, truth_hist: np.ndarray, mode: PlotMode = "interactive", out_path: str | None = None
) -> None:
    w = truth_hist[:, 10:13]
    fig, axes = plt.subplots(3, 1, figsize=cap_figsize(10, 8), sharex=True)
    labels = ["wx (rad/s)", "wy (rad/s)", "wz (rad/s)"]
    for i, ax in enumerate(axes):
        ax.plot(t_s, w[:, i], linewidth=1.4)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Body Angular Rates Over Time")

    fig.tight_layout()
    if mode in ("save", "both"):
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    if mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)


def plot_ground_track(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    title: str = "Ground Track (Lat/Lon)",
    draw_earth_map: bool = True,
) -> None:
    lon_p, lat_p = split_ground_track_dateline(lon_deg=lon_deg, lat_deg=lat_deg, jump_threshold_deg=180.0)
    cartopy_attempted = False
    cartopy_ok = False
    if draw_earth_map and _HAS_CARTOPY:
        cartopy_attempted = True
        fig = plt.figure(figsize=cap_figsize(11, 5))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        ax.set_global()
        ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#cfe8ff", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#dbe7c9", edgecolor="#8aa27a", linewidth=0.4, zorder=1)
        ax.coastlines(resolution="110m", linewidth=0.5, color="#5e6f57", zorder=2)
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.4,
            color="gray",
            alpha=0.4,
            linestyle="-",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        ax.plot(lon_p, lat_p, linewidth=1.4, transform=ccrs.PlateCarree(), zorder=3)
        if lon_deg.size > 0 and lat_deg.size > 0:
            if np.isfinite(lon_deg[0]) and np.isfinite(lat_deg[0]):
                ax.scatter([lon_deg[0]], [lat_deg[0]], color="green", s=28, transform=ccrs.PlateCarree(), zorder=4)
            if np.isfinite(lon_deg[-1]) and np.isfinite(lat_deg[-1]):
                ax.scatter([lon_deg[-1]], [lat_deg[-1]], color="red", s=28, transform=ccrs.PlateCarree(), zorder=4)
        ax.set_title(title)
        fig.tight_layout()
        try:
            if mode in ("save", "both"):
                if out_path is None:
                    raise ValueError("out_path is required when mode is 'save' or 'both'")
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_path, dpi=150)
            if mode in ("interactive", "both"):
                plt.show()
            cartopy_ok = True
        except Exception:
            cartopy_ok = False
        finally:
            plt.close(fig)

    if cartopy_attempted and cartopy_ok:
        return

    fig, ax = plt.subplots(figsize=cap_figsize(11, 5))
    if draw_earth_map:
        _draw_stylized_earth_map(ax)
    ax.plot(lon_p, lat_p, linewidth=1.4)
    if lon_deg.size > 0 and lat_deg.size > 0:
        if np.isfinite(lon_deg[0]) and np.isfinite(lat_deg[0]):
            ax.scatter([lon_deg[0]], [lat_deg[0]], color="green", s=28, zorder=4)
        if np.isfinite(lon_deg[-1]) and np.isfinite(lat_deg[-1]):
            ax.scatter([lon_deg[-1]], [lat_deg[-1]], color="red", s=28, zorder=4)
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
    fig.tight_layout()
    if mode in ("save", "both"):
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    if mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)


def _draw_stylized_earth_map(ax: plt.Axes) -> None:
    ocean = Rectangle((-180.0, -90.0), 360.0, 180.0, facecolor="#cfe8ff", edgecolor="none", zorder=0)
    ax.add_patch(ocean)

    continents = [
        # North America (very coarse)
        [(-168, 72), (-145, 68), (-130, 55), (-123, 50), (-118, 34), (-105, 24), (-97, 17), (-83, 20), (-80, 27), (-66, 45), (-82, 55), (-110, 72)],
        # South America
        [(-81, 12), (-72, 8), (-66, -5), (-62, -18), (-58, -33), (-54, -54), (-69, -56), (-76, -40), (-78, -20), (-81, 0)],
        # Africa
        [(-18, 35), (2, 37), (20, 33), (33, 23), (40, 8), (47, -12), (40, -28), (28, -35), (13, -35), (3, -24), (-4, -6), (-9, 14), (-16, 28)],
        # Eurasia
        [(-10, 36), (8, 46), (30, 56), (55, 64), (90, 72), (120, 66), (145, 58), (170, 50), (155, 40), (120, 24), (102, 12), (80, 8), (55, 16), (30, 26), (18, 32), (5, 38)],
        # India/SE Asia extension
        [(72, 23), (85, 22), (95, 15), (103, 8), (106, 2), (102, -4), (90, 2), (82, 8), (75, 16)],
        # Australia
        [(113, -12), (132, -11), (150, -20), (154, -32), (145, -42), (129, -42), (116, -33), (111, -22)],
        # Greenland
        [(-56, 82), (-42, 82), (-28, 74), (-34, 62), (-49, 60), (-60, 68)],
        # Antarctica band
        [(-180, -62), (-120, -64), (-60, -66), (0, -68), (60, -66), (120, -64), (180, -62), (180, -90), (-180, -90)],
    ]
    for poly in continents:
        ax.add_patch(Polygon(poly, closed=True, facecolor="#dbe7c9", edgecolor="#8aa27a", linewidth=0.6, zorder=1))
