import argparse
import sys
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np

# Allow running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noncoop_rpo import (
    EARTH_EQUATORIAL_RADIUS_KM,
    MasterSimLog,
    get_relative_ric_curv,
    print_master_log_summary,
)

PlotOutputMode = Literal["show", "png"]


def _finalize_plot(fig, output_mode: PlotOutputMode, png_path: Optional[Path]) -> None:
    if output_mode == "show":
        return
    if output_mode == "png":
        if png_path is None:
            raise ValueError("png_path is required when output_mode='png'.")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        return
    raise ValueError("output_mode must be one of {'show', 'png'}")


def _pick_primary_rocket(log: MasterSimLog) -> Optional[str]:
    if not log.rocket_t_s_by_name:
        return None
    return sorted(log.rocket_t_s_by_name.keys())[0]


def _pick_inserted_satellite(log: MasterSimLog) -> Optional[str]:
    if log.launches:
        return log.launches[0].inserted_satellite_name
    for name in sorted(log.x_eci_by_agent.keys()):
        if "insert" in name.lower() or "chaser" in name.lower():
            return name
    return None


def _pick_target(log: MasterSimLog, inserted_name: Optional[str]) -> Optional[str]:
    if "target" in log.x_eci_by_agent:
        return "target"
    for name in sorted(log.x_eci_by_agent.keys()):
        if name != inserted_name:
            return name
    return None


def _plot_trajectory_3d(log: MasterSimLog, output_mode: PlotOutputMode, outdir: Path) -> Optional[Path]:
    re_km = EARTH_EQUATORIAL_RADIUS_KM
    rocket_name = _pick_primary_rocket(log)
    inserted_name = _pick_inserted_satellite(log)
    if rocket_name is None or inserted_name is None:
        return None

    rocket_traj = log.rocket_x_eci_by_name[rocket_name][:, 0:3]
    x_inserted = log.x_eci_by_agent[inserted_name]
    valid_inserted = np.isfinite(x_inserted[:, 0])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(rocket_traj[:, 0], rocket_traj[:, 1], rocket_traj[:, 2], label="Rocket Trajectory", color="tab:blue")
    if np.any(valid_inserted):
        ax.plot(
            x_inserted[valid_inserted, 0],
            x_inserted[valid_inserted, 1],
            x_inserted[valid_inserted, 2],
            label="Post-Switch Satellite Drift",
            color="tab:orange",
            linestyle="--",
        )

    ax.scatter(rocket_traj[0, 0], rocket_traj[0, 1], rocket_traj[0, 2], color="green", s=50, label="Start")
    ax.scatter(rocket_traj[-1, 0], rocket_traj[-1, 1], rocket_traj[-1, 2], color="red", s=50, label="End")

    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 30)
    x_sphere = re_km * np.outer(np.cos(u), np.sin(v))
    y_sphere = re_km * np.outer(np.sin(u), np.sin(v))
    z_sphere = re_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.25, linewidth=0.0, color="tab:gray")

    ax.set_xlabel("X ECI (km)")
    ax.set_ylabel("Y ECI (km)")
    ax.set_zlabel("Z ECI (km)")
    ax.set_title("Rocket Ascent Trajectory in ECI")
    ax.legend()
    ax.set_box_aspect((1.0, 1.0, 1.0))

    out = outdir / "trajectory_eci_3d.png" if output_mode == "png" else None
    _finalize_plot(fig, output_mode=output_mode, png_path=out)
    return out


def _plot_rel_ri(log: MasterSimLog, target: str, inserted: str, output_mode: PlotOutputMode, outdir: Path) -> Optional[Path]:
    rel_curv = get_relative_ric_curv(log, host_name=target, other_name=inserted)
    valid = np.isfinite(rel_curv[:, 0])
    if not np.any(valid):
        return None

    fig = plt.figure(figsize=(8, 6))
    plt.plot(rel_curv[valid, 1], rel_curv[valid, 0], color="tab:orange")
    valid_idx = np.where(valid)[0]
    plt.scatter(rel_curv[valid_idx[0], 1], rel_curv[valid_idx[0], 0], color="green", s=40, label="Start")
    plt.scatter(rel_curv[valid_idx[-1], 1], rel_curv[valid_idx[-1], 0], color="red", s=40, label="End")
    plt.xlabel("In-track (km)")
    plt.ylabel("Radial (km)")
    plt.title("Post-Insertion Natural Drift (Curvilinear RIC)\\nOne Target Orbital Period")
    plt.grid(True)
    plt.legend()

    out = outdir / "relative_ri.png" if output_mode == "png" else None
    _finalize_plot(fig, output_mode=output_mode, png_path=out)
    return out


def _plot_rel_rc(log: MasterSimLog, target: str, inserted: str, output_mode: PlotOutputMode, outdir: Path) -> Optional[Path]:
    rel_curv = get_relative_ric_curv(log, host_name=target, other_name=inserted)
    valid = np.isfinite(rel_curv[:, 0])
    if not np.any(valid):
        return None

    fig = plt.figure(figsize=(8, 6))
    plt.plot(rel_curv[valid, 2], rel_curv[valid, 0], color="tab:purple")
    valid_idx = np.where(valid)[0]
    plt.scatter(rel_curv[valid_idx[0], 2], rel_curv[valid_idx[0], 0], color="green", s=40, label="Start")
    plt.scatter(rel_curv[valid_idx[-1], 2], rel_curv[valid_idx[-1], 0], color="red", s=40, label="End")
    plt.xlabel("Cross-track (km)")
    plt.ylabel("Radial (km)")
    plt.title("Post-Insertion Natural Drift (Radial vs Cross-track)")
    plt.grid(True)
    plt.legend()

    out = outdir / "relative_rc.png" if output_mode == "png" else None
    _finalize_plot(fig, output_mode=output_mode, png_path=out)
    return out


def _plot_altitude_handoff(log: MasterSimLog, output_mode: PlotOutputMode, outdir: Path) -> Optional[Path]:
    re_km = EARTH_EQUATORIAL_RADIUS_KM
    rocket_name = _pick_primary_rocket(log)
    inserted_name = _pick_inserted_satellite(log)
    if rocket_name is None or inserted_name is None:
        return None

    t_rocket = log.rocket_t_s_by_name[rocket_name]
    x_rocket = log.rocket_x_eci_by_name[rocket_name]
    rocket_alt_km = np.linalg.norm(x_rocket[:, 0:3], axis=1) - re_km

    x_inserted = log.x_eci_by_agent[inserted_name]
    valid_inserted = np.isfinite(x_inserted[:, 0])
    t_inserted = log.t_s[valid_inserted]
    sat_alt_km = np.linalg.norm(x_inserted[valid_inserted, 0:3], axis=1) - re_km

    fig = plt.figure(figsize=(9, 6))
    handoff_t_min = t_rocket[-1] / 60.0
    plt.plot(
        t_rocket / 60.0,
        rocket_alt_km,
        color="tab:blue",
        linewidth=2.8,
        linestyle="--",
        label="Rocket Altitude (Ascent)",
        zorder=3,
    )
    if np.any(valid_inserted):
        plt.plot(
            t_inserted / 60.0,
            sat_alt_km,
            color="tab:orange",
            linewidth=2.2,
            linestyle="-",
            label="Satellite Altitude (Post-Switch)",
            zorder=2,
        )
    plt.axvline(
        handoff_t_min,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Rocket-to-Satellite Handoff",
        zorder=1,
    )
    plt.scatter(t_rocket[0] / 60.0, rocket_alt_km[0], color="green", s=35, label="Start", zorder=4)
    if np.any(valid_inserted):
        plt.scatter(t_inserted[-1] / 60.0, sat_alt_km[-1], color="red", s=35, label="End", zorder=4)
    plt.xlabel("Time Since Liftoff (min)")
    plt.ylabel("Altitude (km)")
    plt.title("Rocket and Satellite Altitude vs Time")
    plt.grid(True)
    plt.legend()

    out = outdir / "altitude_handoff.png" if output_mode == "png" else None
    _finalize_plot(fig, output_mode=output_mode, png_path=out)
    return out


def _plot_thrust(log: MasterSimLog, output_mode: PlotOutputMode, outdir: Path) -> Optional[Path]:
    rocket_name = _pick_primary_rocket(log)
    if rocket_name is None:
        return None

    t_rocket = log.rocket_t_s_by_name[rocket_name]
    thrust = log.rocket_thrust_newton_by_name[rocket_name]

    fig = plt.figure(figsize=(9, 5))
    plt.plot(t_rocket / 60.0, thrust / 1e6, color="tab:red", linewidth=2.0)
    plt.xlabel("Time Since Liftoff (min)")
    plt.ylabel("Thrust (MN)")
    plt.title("Rocket Thrust vs Time")
    plt.grid(True)

    out = outdir / "thrust_vs_time.png" if output_mode == "png" else None
    _finalize_plot(fig, output_mode=output_mode, png_path=out)
    return out


def _plot_chaser_ric_velocity_states(
    log: MasterSimLog,
    target: str,
    chaser: str,
    output_mode: PlotOutputMode,
    outdir: Path,
) -> Optional[Path]:
    rel_curv = get_relative_ric_curv(log, host_name=target, other_name=chaser)
    valid = np.isfinite(rel_curv[:, 3])
    if not np.any(valid):
        return None

    t_min = log.t_s[valid] / 60.0
    fig = plt.figure(figsize=(9, 6))
    plt.plot(t_min, rel_curv[valid, 3], label="dR (km/s)")
    plt.plot(t_min, rel_curv[valid, 4], label="dI (km/s)")
    plt.plot(t_min, rel_curv[valid, 5], label="dC (km/s)")
    plt.xlabel("Time (min)")
    plt.ylabel("RIC Velocity (km/s)")
    plt.title(f"Chaser RIC Velocity States vs Time ({chaser} wrt {target})")
    plt.grid(True)
    plt.legend()

    out = outdir / "chaser_ric_velocity_states.png" if output_mode == "png" else None
    _finalize_plot(fig, output_mode=output_mode, png_path=out)
    return out


def _plot_chaser_control_input(
    log: MasterSimLog,
    chaser: str,
    output_mode: PlotOutputMode,
    outdir: Path,
) -> Optional[Path]:
    if chaser not in log.u_ric_by_agent:
        return None
    u = log.u_ric_by_agent[chaser]
    valid = np.isfinite(u[:, 0])
    if not np.any(valid):
        return None

    t_min = log.t_s[valid] / 60.0
    u_valid = u[valid, :]
    fig = plt.figure(figsize=(9, 6))
    plt.plot(t_min, u_valid[:, 0], label="u_R")
    plt.plot(t_min, u_valid[:, 1], label="u_I")
    plt.plot(t_min, u_valid[:, 2], label="u_C")
    plt.plot(t_min, np.linalg.norm(u_valid, axis=1), label="|u|")
    plt.xlabel("Time (min)")
    plt.ylabel("Control Input (km/s^2)")
    plt.title(f"Chaser Control Input vs Time ({chaser})")
    plt.grid(True)
    plt.legend()

    out = outdir / "chaser_control_input.png" if output_mode == "png" else None
    _finalize_plot(fig, output_mode=output_mode, png_path=out)
    return out


def _print_analysis(log: MasterSimLog) -> None:
    print_master_log_summary(log)


def _show_all_if_requested(output_mode: PlotOutputMode) -> None:
    if output_mode == "show":
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and plot a saved master simulation log.")
    parser.add_argument(
        "--log",
        type=str,
        default=str(REPO_ROOT / "outputs" / "master_sim_log.npz"),
        help="Path to .npz log generated by MasterSimulator.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "master_sim_plots"),
        help="Directory where analysis plots are written.",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        choices=["show", "png"],
        default="show",
        help="Show figures immediately ('show') or save them as PNG files ('png').",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    outdir = Path(args.outdir)
    output_mode: PlotOutputMode = args.output_mode  # type: ignore[assignment]
    if output_mode == "png":
        outdir.mkdir(parents=True, exist_ok=True)

    log = MasterSimLog.load_npz(str(log_path))
    _print_analysis(log)

    inserted = _pick_inserted_satellite(log)
    target = _pick_target(log, inserted)

    saved: list[Path] = []
    for maybe in [
        _plot_trajectory_3d(log, output_mode=output_mode, outdir=outdir),
        _plot_altitude_handoff(log, output_mode=output_mode, outdir=outdir),
        _plot_thrust(log, output_mode=output_mode, outdir=outdir),
    ]:
        if maybe is not None:
            saved.append(maybe)

    if inserted is not None and target is not None:
        for maybe in [
            _plot_rel_ri(log, target=target, inserted=inserted, output_mode=output_mode, outdir=outdir),
            _plot_rel_rc(log, target=target, inserted=inserted, output_mode=output_mode, outdir=outdir),
            _plot_chaser_ric_velocity_states(log, target=target, chaser=inserted, output_mode=output_mode, outdir=outdir),
            _plot_chaser_control_input(log, chaser=inserted, output_mode=output_mode, outdir=outdir),
        ]:
            if maybe is not None:
                saved.append(maybe)

    if output_mode == "png":
        print("\nSaved plots:")
        for p in saved:
            print(f"  {p}")
    else:
        _show_all_if_requested(output_mode)


if __name__ == "__main__":
    main()
