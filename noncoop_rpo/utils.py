from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np

from .atmosphere import EARTH_EQUATORIAL_RADIUS_KM, EARTH_ROT_RATE_RAD_S
from .frames import eci2hcw, eci2hcw_curv
from .master_simulator import MasterSimLog

PlotOutputMode = Literal["show", "png"]


def _load_log(log_or_path: MasterSimLog | str | Path) -> MasterSimLog:
    if isinstance(log_or_path, MasterSimLog):
        return log_or_path
    return MasterSimLog.load_npz(str(log_or_path))


def _finalize_plot(fig, output_mode: PlotOutputMode, png_path: Optional[str | Path]) -> None:
    import matplotlib.pyplot as plt

    if output_mode == "show":
        plt.show()
        return
    if output_mode == "png":
        if png_path is None:
            raise ValueError("png_path is required when output_mode='png'.")
        out = Path(png_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return
    raise ValueError("output_mode must be one of {'show', 'png'}")


def eci_to_ecef(r_eci_km: np.ndarray, t_s: float) -> np.ndarray:
    theta = EARTH_ROT_RATE_RAD_S * float(t_s)
    c = np.cos(theta)
    s = np.sin(theta)
    rot = np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rot @ np.asarray(r_eci_km, dtype=float)


def eci_history_to_ecef(r_eci_hist_km: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    r_eci_hist_km = np.asarray(r_eci_hist_km, dtype=float)
    t_s = np.asarray(t_s, dtype=float)
    out = np.full_like(r_eci_hist_km, np.nan, dtype=float)
    valid = np.isfinite(r_eci_hist_km[:, 0])
    for idx in np.where(valid)[0]:
        out[idx, :] = eci_to_ecef(r_eci_hist_km[idx, :], t_s[idx])
    return out


def get_relative_ric_curv(log_or_path: MasterSimLog | str | Path, host_name: str, other_name: str) -> np.ndarray:
    log = _load_log(log_or_path)
    x_host = log.x_eci_by_agent[host_name]
    x_other = log.x_eci_by_agent[other_name]
    rel = np.full((len(log.t_s), 6), np.nan, dtype=float)
    valid = np.isfinite(x_host[:, 0]) & np.isfinite(x_other[:, 0])
    for idx in np.where(valid)[0]:
        rel[idx, :] = eci2hcw_curv(x_host[idx, :], x_other[idx, :])
    return rel


def get_relative_ric_rect(log_or_path: MasterSimLog | str | Path, host_name: str, other_name: str) -> np.ndarray:
    log = _load_log(log_or_path)
    x_host = log.x_eci_by_agent[host_name]
    x_other = log.x_eci_by_agent[other_name]
    rel = np.full((len(log.t_s), 6), np.nan, dtype=float)
    valid = np.isfinite(x_host[:, 0]) & np.isfinite(x_other[:, 0])
    for idx in np.where(valid)[0]:
        rel[idx, :] = eci2hcw(x_host[idx, :], x_other[idx, :])
    return rel


def _plot_3d(vec_hist: np.ndarray, valid: np.ndarray, label: str, title: str, x_label: str, y_label: str, z_label: str):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(vec_hist[valid, 0], vec_hist[valid, 1], vec_hist[valid, 2], label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.legend()
    return fig, ax


def _plot_2d(
    vec_hist: np.ndarray,
    valid: np.ndarray,
    idx_a: int,
    idx_b: int,
    label: str,
    title: str,
    x_label: str,
    y_label: str,
):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(vec_hist[valid, idx_a], vec_hist[valid, idx_b], label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    return fig, ax


def plot_eci_3d(
    log_or_path: MasterSimLog | str | Path,
    agent_name: str,
    output_mode: PlotOutputMode = "show",
    png_path: Optional[str | Path] = None,
):
    log = _load_log(log_or_path)
    x = log.x_eci_by_agent[agent_name]
    valid = np.isfinite(x[:, 0])
    fig, ax = _plot_3d(
        x[:, :3],
        valid,
        label=agent_name,
        title=f"ECI Trajectory: {agent_name}",
        x_label="X (km)",
        y_label="Y (km)",
        z_label="Z (km)",
    )
    _finalize_plot(fig, output_mode=output_mode, png_path=png_path)
    return fig, ax


def plot_eci_2d(
    log_or_path: MasterSimLog | str | Path,
    agent_name: str,
    plane: str = "XY",
    output_mode: PlotOutputMode = "show",
    png_path: Optional[str | Path] = None,
):
    plane = plane.upper()
    map_idx = {"X": 0, "Y": 1, "Z": 2}
    if len(plane) != 2 or plane[0] not in map_idx or plane[1] not in map_idx:
        raise ValueError("plane must be two letters from X, Y, Z (e.g., 'XY', 'XZ').")
    log = _load_log(log_or_path)
    x = log.x_eci_by_agent[agent_name]
    valid = np.isfinite(x[:, 0])
    ia, ib = map_idx[plane[0]], map_idx[plane[1]]
    fig, ax = _plot_2d(
        x[:, :3],
        valid,
        ia,
        ib,
        label=agent_name,
        title=f"ECI {plane}: {agent_name}",
        x_label=f"{plane[0]} (km)",
        y_label=f"{plane[1]} (km)",
    )
    _finalize_plot(fig, output_mode=output_mode, png_path=png_path)
    return fig, ax


def plot_ecef_3d(
    log_or_path: MasterSimLog | str | Path,
    agent_name: str,
    output_mode: PlotOutputMode = "show",
    png_path: Optional[str | Path] = None,
):
    log = _load_log(log_or_path)
    x = log.x_eci_by_agent[agent_name]
    r_ecef = eci_history_to_ecef(x[:, :3], log.t_s)
    valid = np.isfinite(r_ecef[:, 0])
    fig, ax = _plot_3d(
        r_ecef,
        valid,
        label=agent_name,
        title=f"ECEF Trajectory: {agent_name}",
        x_label="X (km)",
        y_label="Y (km)",
        z_label="Z (km)",
    )
    _finalize_plot(fig, output_mode=output_mode, png_path=png_path)
    return fig, ax


def plot_ecef_2d(
    log_or_path: MasterSimLog | str | Path,
    agent_name: str,
    plane: str = "XY",
    output_mode: PlotOutputMode = "show",
    png_path: Optional[str | Path] = None,
):
    plane = plane.upper()
    map_idx = {"X": 0, "Y": 1, "Z": 2}
    if len(plane) != 2 or plane[0] not in map_idx or plane[1] not in map_idx:
        raise ValueError("plane must be two letters from X, Y, Z (e.g., 'XY', 'XZ').")
    log = _load_log(log_or_path)
    x = log.x_eci_by_agent[agent_name]
    r_ecef = eci_history_to_ecef(x[:, :3], log.t_s)
    valid = np.isfinite(r_ecef[:, 0])
    ia, ib = map_idx[plane[0]], map_idx[plane[1]]
    fig, ax = _plot_2d(
        r_ecef,
        valid,
        ia,
        ib,
        label=agent_name,
        title=f"ECEF {plane}: {agent_name}",
        x_label=f"{plane[0]} (km)",
        y_label=f"{plane[1]} (km)",
    )
    _finalize_plot(fig, output_mode=output_mode, png_path=png_path)
    return fig, ax


def plot_ric_curv_3d(
    log_or_path: MasterSimLog | str | Path,
    host_name: str,
    other_name: str,
    output_mode: PlotOutputMode = "show",
    png_path: Optional[str | Path] = None,
):
    rel = get_relative_ric_curv(log_or_path, host_name=host_name, other_name=other_name)
    valid = np.isfinite(rel[:, 0])
    fig, ax = _plot_3d(
        rel[:, :3],
        valid,
        label=f"{other_name} wrt {host_name}",
        title=f"RIC Curv 3D: {other_name} wrt {host_name}",
        x_label="R (km)",
        y_label="I (km)",
        z_label="C (km)",
    )
    _finalize_plot(fig, output_mode=output_mode, png_path=png_path)
    return fig, ax


def plot_ric_curv_2d(
    log_or_path: MasterSimLog | str | Path,
    host_name: str,
    other_name: str,
    plane: str = "RI",
    output_mode: PlotOutputMode = "show",
    png_path: Optional[str | Path] = None,
):
    plane = plane.upper()
    map_idx = {"R": 0, "I": 1, "C": 2}
    if len(plane) != 2 or plane[0] not in map_idx or plane[1] not in map_idx:
        raise ValueError("plane must be two letters from R, I, C (e.g., 'RI', 'RC').")
    rel = get_relative_ric_curv(log_or_path, host_name=host_name, other_name=other_name)
    valid = np.isfinite(rel[:, 0])
    ia, ib = map_idx[plane[0]], map_idx[plane[1]]
    fig, ax = _plot_2d(
        rel[:, :3],
        valid,
        ia,
        ib,
        label=f"{other_name} wrt {host_name}",
        title=f"RIC Curv {plane}: {other_name} wrt {host_name}",
        x_label=f"{plane[0]} (km)",
        y_label=f"{plane[1]} (km)",
    )
    _finalize_plot(fig, output_mode=output_mode, png_path=png_path)
    return fig, ax


def plot_ric_rect_3d(
    log_or_path: MasterSimLog | str | Path,
    host_name: str,
    other_name: str,
    output_mode: PlotOutputMode = "show",
    png_path: Optional[str | Path] = None,
):
    rel = get_relative_ric_rect(log_or_path, host_name=host_name, other_name=other_name)
    valid = np.isfinite(rel[:, 0])
    fig, ax = _plot_3d(
        rel[:, :3],
        valid,
        label=f"{other_name} wrt {host_name}",
        title=f"RIC Rect 3D: {other_name} wrt {host_name}",
        x_label="R (km)",
        y_label="I (km)",
        z_label="C (km)",
    )
    _finalize_plot(fig, output_mode=output_mode, png_path=png_path)
    return fig, ax


def plot_ric_rect_2d(
    log_or_path: MasterSimLog | str | Path,
    host_name: str,
    other_name: str,
    plane: str = "RI",
    output_mode: PlotOutputMode = "show",
    png_path: Optional[str | Path] = None,
):
    plane = plane.upper()
    map_idx = {"R": 0, "I": 1, "C": 2}
    if len(plane) != 2 or plane[0] not in map_idx or plane[1] not in map_idx:
        raise ValueError("plane must be two letters from R, I, C (e.g., 'RI', 'RC').")
    rel = get_relative_ric_rect(log_or_path, host_name=host_name, other_name=other_name)
    valid = np.isfinite(rel[:, 0])
    ia, ib = map_idx[plane[0]], map_idx[plane[1]]
    fig, ax = _plot_2d(
        rel[:, :3],
        valid,
        ia,
        ib,
        label=f"{other_name} wrt {host_name}",
        title=f"RIC Rect {plane}: {other_name} wrt {host_name}",
        x_label=f"{plane[0]} (km)",
        y_label=f"{plane[1]} (km)",
    )
    _finalize_plot(fig, output_mode=output_mode, png_path=png_path)
    return fig, ax


def _last_valid_row(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr[:, 0])
    if not np.any(valid):
        return np.full(arr.shape[1], np.nan)
    return arr[np.where(valid)[0][-1], :]


def _integrated_norm(t_s: np.ndarray, vec_hist: np.ndarray) -> float:
    valid = np.isfinite(vec_hist[:, 0])
    if np.count_nonzero(valid) < 2:
        return 0.0
    mag = np.linalg.norm(vec_hist[valid, :], axis=1)
    return float(np.trapz(mag, t_s[valid]))


def master_log_summary_text(log_or_path: MasterSimLog | str | Path) -> str:
    log = _load_log(log_or_path)
    lines: list[str] = []
    lines.append("=== Master Simulation Analysis ===")
    lines.append(f"Samples: {len(log.t_s)}")
    if len(log.t_s) >= 2:
        lines.append(f"Duration: {log.t_s[-1]:.1f} s")
        lines.append(f"Nominal dt: {log.t_s[1] - log.t_s[0]:.3f} s")
    lines.append(f"Pre-sim samples: {int(np.count_nonzero(log.phase == 'pre'))}")
    lines.append(f"Sim samples: {int(np.count_nonzero(log.phase == 'sim'))}")
    lines.append(f"Terminated early: {log.terminated_early}")
    lines.append(f"Termination reason: {log.termination_reason}")

    lines.append("")
    lines.append("Launch events:")
    if not log.launches:
        lines.append("  none")
    for evt in log.launches:
        lines.append(
            f"  {evt.rocket_name}: launch@{evt.scheduled_launch_time_s:.1f}s "
            f"inserted@{evt.insertion_time_s:.1f}s reason={evt.insertion_reason}"
        )

    lines.append("")
    lines.append("Agent summary:")
    for name in sorted(log.x_eci_by_agent.keys()):
        x_last = _last_valid_row(log.x_eci_by_agent[name])
        if not np.isfinite(x_last[0]):
            lines.append(f"  {name}: never active")
            continue
        final_alt_km = float(np.linalg.norm(x_last[:3]) - EARTH_EQUATORIAL_RADIUS_KM)
        dv_like = 1000.0 * _integrated_norm(log.t_s, log.u_ric_by_agent[name])
        lines.append(f"  {name}: final_alt={final_alt_km:.2f} km, integrated_|u|*dt={dv_like:.2f} m/s")

    return "\n".join(lines)


def print_master_log_summary(log_or_path: MasterSimLog | str | Path) -> None:
    print(master_log_summary_text(log_or_path))


def plot_control_input_over_time(
    log_or_path: MasterSimLog | str | Path,
    agent_name: str,
    frame: str = "ric",
    output_mode: PlotOutputMode = "show",
    png_path: Optional[str | Path] = None,
):
    import matplotlib.pyplot as plt

    log = _load_log(log_or_path)
    frame_l = frame.lower()
    if frame_l not in {"ric", "eci"}:
        raise ValueError("frame must be one of {'ric', 'eci'}")
    u = log.u_ric_by_agent[agent_name] if frame_l == "ric" else log.u_eci_by_agent[agent_name]

    valid = np.isfinite(u[:, 0])
    t_min = log.t_s[valid] / 60.0
    u_valid = u[valid, :]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(t_min, u_valid[:, 0], label="u1")
    ax.plot(t_min, u_valid[:, 1], label="u2")
    ax.plot(t_min, u_valid[:, 2], label="u3")
    ax.plot(t_min, np.linalg.norm(u_valid, axis=1), label="|u|")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Acceleration (km/s^2)")
    ax.set_title(f"Control Input Over Time ({frame_l.upper()}): {agent_name}")
    ax.grid(True)
    ax.legend()
    _finalize_plot(fig, output_mode=output_mode, png_path=png_path)
    return fig, ax


def plot_state_over_time(
    log_or_path: MasterSimLog | str | Path,
    agent_name: str,
    frame: str = "eci",
    host_name: Optional[str] = None,
    ric_mode: str = "curv",
    output_mode: PlotOutputMode = "show",
    png_path: Optional[str | Path] = None,
):
    import matplotlib.pyplot as plt

    log = _load_log(log_or_path)
    frame_l = frame.lower()
    if frame_l == "eci":
        x = log.x_eci_by_agent[agent_name]
        labels = ["X", "Y", "Z", "Vx", "Vy", "Vz"]
        units = ["km", "km", "km", "km/s", "km/s", "km/s"]
    elif frame_l == "ric":
        if host_name is None:
            raise ValueError("host_name is required when frame='ric'.")
        mode = ric_mode.lower()
        if mode not in {"curv", "rect"}:
            raise ValueError("ric_mode must be one of {'curv', 'rect'}")
        if mode == "curv":
            x = get_relative_ric_curv(log, host_name=host_name, other_name=agent_name)
        else:
            x = get_relative_ric_rect(log, host_name=host_name, other_name=agent_name)
        labels = ["R", "I", "C", "dR", "dI", "dC"]
        units = ["km", "km", "km", "km/s", "km/s", "km/s"]
    else:
        raise ValueError("frame must be one of {'eci', 'ric'}")

    valid = np.isfinite(x[:, 0])
    t_min = log.t_s[valid] / 60.0
    x_valid = x[valid, :]

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()
    for i in range(6):
        axes[i].plot(t_min, x_valid[:, i])
        axes[i].set_ylabel(f"{labels[i]} ({units[i]})")
        axes[i].grid(True)
    axes[-2].set_xlabel("Time (min)")
    axes[-1].set_xlabel("Time (min)")
    if frame_l == "eci":
        fig.suptitle(f"State Over Time (ECI): {agent_name}")
    else:
        fig.suptitle(f"State Over Time (RIC-{ric_mode.lower()}): {agent_name} wrt {host_name}")
    fig.tight_layout()
    _finalize_plot(fig, output_mode=output_mode, png_path=png_path)
    return fig, axes
