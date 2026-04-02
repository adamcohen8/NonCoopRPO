from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.dynamics.orbit import (
    EARTH_MU_KM3_S2,
    EARTH_RADIUS_KM,
    OrbitContext,
    OrbitPropagator,
    datetime_to_julian_date,
    drag_plugin,
    resolve_time_dependent_env,
    third_body_moon_plugin,
)


@dataclass(frozen=True)
class SearchWindow:
    angle_center_deg: float = 6.0
    angle_span_deg: float = 1.0
    speed_offset_center_km_s: float = 0.010
    speed_offset_span_km_s: float = 0.004
    points_per_axis: int = 5
    rounds: int = 1
    shrink: float = 0.35


@dataclass(frozen=True)
class SearchCandidate:
    score: float
    phase_angle_deg: float
    speed_km_s: float
    state_eci_km_s: np.ndarray
    min_moon_distance_km: float
    time_to_moon_days: float
    max_earth_range_km: float
    post_flyby_perigee_radius_km: float
    post_flyby_perigee_altitude_km: float
    earth_impact: bool
    return_time_days: float | None
    duration_days: float


@dataclass(frozen=True)
class TrajectoryResult:
    t_s: np.ndarray
    x_eci_km_s: np.ndarray
    moon_eci_km: np.ndarray
    earth_radius_km: np.ndarray
    moon_distance_km: np.ndarray
    min_moon_distance_km: float
    time_to_moon_days: float
    max_earth_range_km: float
    post_flyby_perigee_radius_km: float
    post_flyby_perigee_altitude_km: float
    earth_impact: bool
    return_time_days: float | None
    duration_days: float


def _draw_body_sphere(ax: plt.Axes, radius_km: float, color: str, alpha: float, center_km: np.ndarray | None = None) -> None:
    center = np.zeros(3, dtype=float) if center_km is None else np.asarray(center_km, dtype=float).reshape(3)
    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 24)
    x = radius_km * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius_km * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius_km * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color=color, alpha=alpha, linewidth=0.0, zorder=0)


def _moon_plane_basis(env_base: dict, epoch_offset_s: float = 3600.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moon_0 = np.asarray(resolve_time_dependent_env(env_base, 0.0)["moon_pos_eci_km"], dtype=float)
    moon_1 = np.asarray(resolve_time_dependent_env(env_base, float(epoch_offset_s))["moon_pos_eci_km"], dtype=float)
    moon_hat = moon_0 / float(np.linalg.norm(moon_0))
    moon_vel_like = moon_1 - moon_0
    h_hat = np.cross(moon_0, moon_vel_like)
    h_hat = h_hat / float(np.linalg.norm(h_hat))
    t_hat = np.cross(h_hat, moon_hat)
    t_hat = t_hat / float(np.linalg.norm(t_hat))
    return moon_hat, t_hat, h_hat


def _parameterized_state_vector_eci_km_s(
    env_base: dict,
    leo_altitude_km: float,
    target_apogee_radius_km: float,
    phase_angle_deg: float,
    speed_km_s: float,
) -> np.ndarray:
    moon_hat_0, _, h_hat = _moon_plane_basis(env_base)
    rp = EARTH_RADIUS_KM + float(leo_altitude_km)
    ra = float(target_apogee_radius_km)
    a = 0.5 * (rp + ra)
    tof_apogee_s = np.pi * np.sqrt((a**3) / EARTH_MU_KM3_S2)
    moon_hat_apogee = np.asarray(resolve_time_dependent_env(env_base, tof_apogee_s)["moon_pos_eci_km"], dtype=float)
    moon_hat_apogee = moon_hat_apogee / float(np.linalg.norm(moon_hat_apogee))

    # Start roughly opposite the Moon's predicted apogee direction, then bias forward
    # within the Moon's orbital plane to tune the free-return geometry.
    base_r_hat = -moon_hat_apogee
    base_t_hat = np.cross(h_hat, base_r_hat)
    base_t_hat = base_t_hat / float(np.linalg.norm(base_t_hat))
    phase_rad = np.deg2rad(float(phase_angle_deg))
    r_hat = np.cos(phase_rad) * base_r_hat + np.sin(phase_rad) * base_t_hat
    r_hat = r_hat / float(np.linalg.norm(r_hat))
    v_hat = np.cross(h_hat, r_hat)
    v_hat = v_hat / float(np.linalg.norm(v_hat))

    x0 = np.zeros(6, dtype=float)
    x0[:3] = rp * r_hat
    x0[3:] = float(speed_km_s) * v_hat
    return x0


def _propagate_trajectory(
    x0_eci_km_s: np.ndarray,
    env_base: dict,
    ctx: OrbitContext,
    dt_s: float,
    duration_days: float,
    integrator: str = "adaptive",
    stop_on_earth_impact: bool = True,
) -> TrajectoryResult:
    prop = OrbitPropagator(
        integrator=integrator,
        plugins=[drag_plugin, third_body_moon_plugin],
        adaptive_atol=1e-9,
        adaptive_rtol=1e-8,
    )
    steps = int(np.ceil(float(duration_days) * 86400.0 / float(dt_s)))
    t = np.arange(steps + 1, dtype=float) * float(dt_s)
    x = np.zeros((steps + 1, 6), dtype=float)
    moon = np.zeros((steps + 1, 3), dtype=float)
    x[0, :] = np.asarray(x0_eci_km_s, dtype=float).reshape(6)

    zero_cmd = np.zeros(3, dtype=float)
    earth_impact = False
    end_idx = steps
    for k in range(steps):
        env_k = resolve_time_dependent_env(env_base, float(t[k]))
        moon[k, :] = np.asarray(env_k["moon_pos_eci_km"], dtype=float)
        x[k + 1, :] = prop.propagate(
            x_eci=x[k, :],
            dt_s=float(dt_s),
            t_s=float(t[k]),
            command_accel_eci_km_s2=zero_cmd,
            env=env_k,
            ctx=ctx,
        )
        if stop_on_earth_impact and float(np.linalg.norm(x[k + 1, :3])) <= EARTH_RADIUS_KM:
            earth_impact = True
            end_idx = k + 1
            moon[k + 1, :] = np.asarray(resolve_time_dependent_env(env_base, float(t[k + 1]))["moon_pos_eci_km"], dtype=float)
            break
    else:
        moon[-1, :] = np.asarray(resolve_time_dependent_env(env_base, float(t[-1]))["moon_pos_eci_km"], dtype=float)

    t = t[: end_idx + 1]
    x = x[: end_idx + 1, :]
    moon = moon[: end_idx + 1, :]
    earth_radius = np.linalg.norm(x[:, :3], axis=1)
    moon_distance = np.linalg.norm(x[:, :3] - moon, axis=1)
    i_moon = int(np.argmin(moon_distance))
    min_moon_distance_km = float(moon_distance[i_moon])
    time_to_moon_days = float(t[i_moon] / 86400.0)
    max_earth_range_km = float(np.max(earth_radius))
    if i_moon < earth_radius.size - 1:
        post_perigee_radius_km = float(np.min(earth_radius[i_moon + 1 :]))
    else:
        post_perigee_radius_km = float(earth_radius[-1])
    post_perigee_altitude_km = float(post_perigee_radius_km - EARTH_RADIUS_KM)
    return_time_days = None
    if earth_impact:
        return_time_days = float(t[-1] / 86400.0)

    return TrajectoryResult(
        t_s=t,
        x_eci_km_s=x,
        moon_eci_km=moon,
        earth_radius_km=earth_radius,
        moon_distance_km=moon_distance,
        min_moon_distance_km=min_moon_distance_km,
        time_to_moon_days=time_to_moon_days,
        max_earth_range_km=max_earth_range_km,
        post_flyby_perigee_radius_km=post_perigee_radius_km,
        post_flyby_perigee_altitude_km=post_perigee_altitude_km,
        earth_impact=earth_impact,
        return_time_days=return_time_days,
        duration_days=float(t[-1] / 86400.0),
    )


def _base_free_return_speed_km_s(leo_altitude_km: float, target_apogee_radius_km: float) -> float:
    rp = EARTH_RADIUS_KM + float(leo_altitude_km)
    ra = float(target_apogee_radius_km)
    a = 0.5 * (rp + ra)
    return float(np.sqrt(EARTH_MU_KM3_S2 * (2.0 / rp - 1.0 / a)))


def _score_candidate(
    traj: TrajectoryResult,
    target_duration_days: float,
    moon_distance_goal_km: float,
    min_lunar_encounter_km: float,
    outbound_floor_km: float,
) -> float:
    if traj.max_earth_range_km < float(outbound_floor_km):
        return 1.0e12 + float(outbound_floor_km) - traj.max_earth_range_km
    if traj.min_moon_distance_km > float(min_lunar_encounter_km):
        return 5.0e11 + traj.min_moon_distance_km

    moon_term = abs(traj.min_moon_distance_km - float(moon_distance_goal_km)) / 1000.0
    perigee_term = abs(traj.post_flyby_perigee_radius_km - EARTH_RADIUS_KM) / 10.0
    duration_term = 0.0
    if traj.return_time_days is None:
        duration_term += 1000.0
        duration_term += max(0.0, traj.post_flyby_perigee_altitude_km) / 5.0
    else:
        duration_term += 50.0 * abs(traj.return_time_days - float(target_duration_days))
    return moon_term + perigee_term + duration_term


def _search_initial_state(
    env_base: dict,
    ctx: OrbitContext,
    leo_altitude_km: float,
    target_apogee_radius_km: float,
    search_dt_s: float,
    search_duration_days: float,
    target_duration_days: float,
    moon_distance_goal_km: float,
    min_lunar_encounter_km: float,
    outbound_floor_km: float,
    window: SearchWindow,
) -> SearchCandidate:
    base_speed = _base_free_return_speed_km_s(
        leo_altitude_km=float(leo_altitude_km),
        target_apogee_radius_km=float(target_apogee_radius_km),
    )
    best: SearchCandidate | None = None
    angle_center = float(window.angle_center_deg)
    angle_span = float(window.angle_span_deg)
    speed_offset_center = float(window.speed_offset_center_km_s)
    speed_offset_span = float(window.speed_offset_span_km_s)

    for _ in range(int(window.rounds)):
        angle_grid = np.linspace(angle_center - angle_span, angle_center + angle_span, int(window.points_per_axis))
        speed_grid = np.linspace(
            base_speed + speed_offset_center - speed_offset_span,
            base_speed + speed_offset_center + speed_offset_span,
            int(window.points_per_axis),
        )
        round_best: SearchCandidate | None = None
        for phase_angle_deg in angle_grid:
            for speed_km_s in speed_grid:
                x0 = _parameterized_state_vector_eci_km_s(
                    env_base=env_base,
                    leo_altitude_km=float(leo_altitude_km),
                    target_apogee_radius_km=float(target_apogee_radius_km),
                    phase_angle_deg=float(phase_angle_deg),
                    speed_km_s=float(speed_km_s),
                )
                traj = _propagate_trajectory(
                    x0_eci_km_s=x0,
                    env_base=env_base,
                    ctx=ctx,
                    dt_s=float(search_dt_s),
                    duration_days=float(search_duration_days),
                    stop_on_earth_impact=True,
                )
                score = _score_candidate(
                    traj=traj,
                    target_duration_days=float(target_duration_days),
                    moon_distance_goal_km=float(moon_distance_goal_km),
                    min_lunar_encounter_km=float(min_lunar_encounter_km),
                    outbound_floor_km=float(outbound_floor_km),
                )
                candidate = SearchCandidate(
                    score=float(score),
                    phase_angle_deg=float(phase_angle_deg),
                    speed_km_s=float(speed_km_s),
                    state_eci_km_s=x0.copy(),
                    min_moon_distance_km=float(traj.min_moon_distance_km),
                    time_to_moon_days=float(traj.time_to_moon_days),
                    max_earth_range_km=float(traj.max_earth_range_km),
                    post_flyby_perigee_radius_km=float(traj.post_flyby_perigee_radius_km),
                    post_flyby_perigee_altitude_km=float(traj.post_flyby_perigee_altitude_km),
                    earth_impact=bool(traj.earth_impact),
                    return_time_days=None if traj.return_time_days is None else float(traj.return_time_days),
                    duration_days=float(traj.duration_days),
                )
                if round_best is None or candidate.score < round_best.score:
                    round_best = candidate
        if round_best is None:
            raise RuntimeError("Search failed to evaluate any candidate state vectors.")
        best = round_best
        angle_center = float(round_best.phase_angle_deg)
        speed_offset_center = float(round_best.speed_km_s - base_speed)
        angle_span *= float(window.shrink)
        speed_offset_span *= float(window.shrink)

    if best is None:
        raise RuntimeError("Search did not produce a best-fit state vector.")
    return best


def _plot_static_summary(traj: TrajectoryResult, outdir: Path, plot_mode: str) -> tuple[Path, Path]:
    if plot_mode == "none":
        return outdir / "artemis2_style_orbit_3d.png", outdir / "artemis2_style_metrics.png"
    import matplotlib.pyplot as plt

    orbit_path = outdir / "artemis2_style_orbit_3d.png"
    metrics_path = outdir / "artemis2_style_metrics.png"

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_body_sphere(ax, radius_km=EARTH_RADIUS_KM, color="#6EA8D9", alpha=0.18)
    ax.plot(
        traj.x_eci_km_s[:, 0],
        traj.x_eci_km_s[:, 1],
        traj.x_eci_km_s[:, 2],
        color="tab:orange",
        linewidth=1.6,
        label="Spacecraft",
    )
    ax.plot(
        traj.moon_eci_km[:, 0],
        traj.moon_eci_km[:, 1],
        traj.moon_eci_km[:, 2],
        color="0.45",
        linestyle="--",
        linewidth=1.1,
        label="Moon",
    )
    ax.scatter(
        [traj.x_eci_km_s[0, 0]],
        [traj.x_eci_km_s[0, 1]],
        [traj.x_eci_km_s[0, 2]],
        c="tab:green",
        s=35,
        label="Start",
    )
    ax.scatter(
        [traj.x_eci_km_s[-1, 0]],
        [traj.x_eci_km_s[-1, 1]],
        [traj.x_eci_km_s[-1, 2]],
        c="tab:red",
        s=35,
        label="End",
    )
    extent = max(
        float(np.max(np.abs(traj.x_eci_km_s[:, :3]))),
        float(np.max(np.abs(traj.moon_eci_km))),
        EARTH_RADIUS_KM,
    )
    extent *= 1.08
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("X ECI (km)")
    ax.set_ylabel("Y ECI (km)")
    ax.set_zlabel("Z ECI (km)")
    ax.set_title("Artemis II-Style Free-Return Approximation")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if plot_mode in ("save", "both"):
        fig.savefig(orbit_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    fig2, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    time_days = traj.t_s / 86400.0
    axes[0].plot(time_days, traj.earth_radius_km - EARTH_RADIUS_KM, color="tab:orange")
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title("Earth Range / Lunar Encounter Metrics")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time_days, traj.moon_distance_km, color="0.35")
    axes[1].set_ylabel("Moon Range (km)")
    axes[1].set_xlabel("Time (days)")
    axes[1].grid(True, alpha=0.3)
    fig2.tight_layout()
    if plot_mode in ("save", "both"):
        fig2.savefig(metrics_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig2)
    return orbit_path, metrics_path


def _animate_trajectory(
    traj: TrajectoryResult,
    outdir: Path,
    plot_mode: str,
    moon_radius_km: float = 1737.4,
    max_frames: int = 240,
    fps: int = 24,
) -> Path:
    if plot_mode == "none":
        return outdir / "artemis2_style_free_return.mp4"
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle

    anim_path = outdir / "artemis2_style_free_return.mp4"
    frame_ids = np.arange(traj.t_s.size, dtype=int)
    if frame_ids.size > int(max_frames):
        frame_ids = np.linspace(0, frame_ids.size - 1, int(max_frames), dtype=int)
    if frame_ids[-1] != traj.t_s.size - 1:
        frame_ids = np.append(frame_ids, traj.t_s.size - 1)

    fig, ax = plt.subplots(figsize=(9, 9))

    extent = max(
        float(np.max(np.abs(traj.x_eci_km_s[:, :3]))),
        float(np.max(np.abs(traj.moon_eci_km))),
        EARTH_RADIUS_KM,
    )
    extent *= 1.08
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("X ECI (km)")
    ax.set_ylabel("Y ECI (km)")
    ax.set_title("Artemis II-Style Free-Return Animation (XY Plane)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    earth_patch = Circle(
        (0.0, 0.0),
        radius=EARTH_RADIUS_KM,
        facecolor="#6EA8D9",
        edgecolor="#5D86AA",
        linewidth=1.1,
        alpha=0.22,
        label="Earth body",
    )
    moon_patch = Circle(
        (traj.moon_eci_km[0, 0], traj.moon_eci_km[0, 1]),
        radius=moon_radius_km,
        facecolor="0.7",
        edgecolor="0.35",
        linewidth=1.0,
        alpha=0.9,
        label="Moon body",
    )
    ax.add_patch(earth_patch)
    ax.add_patch(moon_patch)

    moon_path, = ax.plot([], [], "--", color="0.55", linewidth=1.0, label="Moon path")
    sc_path, = ax.plot([], [], color="tab:orange", linewidth=1.4, label="Spacecraft path")
    moon_dot, = ax.plot([], [], "o", color="0.25", markersize=2.5, alpha=0.7, label="Moon center")
    sc_dot, = ax.plot([], [], "o", color="tab:orange", markersize=4, label="Spacecraft")
    moon_radius_text = ax.text(0.02, 0.93, "", transform=ax.transAxes)
    status_text = ax.text(0.02, 0.97, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def _update(frame_idx: int):
        k = int(frame_ids[frame_idx])
        sc_xyz = traj.x_eci_km_s[k, :3]
        moon_xyz = traj.moon_eci_km[k, :]
        sc_path.set_data(traj.x_eci_km_s[: k + 1, 0], traj.x_eci_km_s[: k + 1, 1])
        moon_path.set_data(traj.moon_eci_km[: k + 1, 0], traj.moon_eci_km[: k + 1, 1])
        sc_dot.set_data([float(sc_xyz[0])], [float(sc_xyz[1])])
        moon_dot.set_data([float(moon_xyz[0])], [float(moon_xyz[1])])
        moon_patch.center = (float(moon_xyz[0]), float(moon_xyz[1]))
        status_text.set_text(f"t = {traj.t_s[k] / 86400.0:.2f} days")
        moon_radius_text.set_text(
            f"|r_sc-r_moon| = {traj.moon_distance_km[k]:,.0f} km | alt = {traj.earth_radius_km[k] - EARTH_RADIUS_KM:,.0f} km | z = {sc_xyz[2]:,.0f} km"
        )
        return sc_path, moon_path, sc_dot, moon_dot, moon_patch, status_text, moon_radius_text

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=int(frame_ids.size),
        interval=int(round(1000.0 / max(int(fps), 1))),
        blit=False,
        repeat=True,
    )

    if plot_mode in ("save", "both"):
        try:
            ani.save(str(anim_path), writer="ffmpeg", fps=max(int(fps), 1), bitrate=1800)
        except Exception as exc:
            print(f"Warning: failed to save MP4 animation ({exc}). Ensure ffmpeg is installed and available on PATH.")
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)
    return anim_path


def run_demo(
    plot_mode: str = "interactive",
    epoch_utc: datetime | None = None,
    duration_days: float = 14.0,
    final_dt_s: float = 1800.0,
    search_dt_s: float = 1800.0,
    leo_altitude_km: float = 185.0,
    target_apogee_radius_km: float = 380000.0,
    target_duration_days: float = 10.0,
    moon_distance_goal_km: float = 28000.0,
    min_lunar_encounter_km: float = 80000.0,
    outbound_floor_km: float = 250000.0,
    phase_angle_deg: float | None = None,
    speed_km_s: float | None = None,
    state_vector_km_kmps: np.ndarray | None = None,
    search_window: SearchWindow | None = None,
) -> dict[str, str]:
    epoch = datetime(2026, 3, 11, 0, 0, 0, tzinfo=timezone.utc) if epoch_utc is None else epoch_utc.astimezone(timezone.utc)
    env_base = {
        "jd_utc_start": datetime_to_julian_date(epoch),
        "ephemeris_mode": "analytic_enhanced",
        "atmosphere_model": "ussa1976",
    }
    ctx = OrbitContext(
        mu_km3_s2=EARTH_MU_KM3_S2,
        mass_kg=26000.0,
        area_m2=80.0,
        cd=2.0,
        cr=1.2,
    )
    window = SearchWindow() if search_window is None else search_window
    base_speed = _base_free_return_speed_km_s(
        leo_altitude_km=float(leo_altitude_km),
        target_apogee_radius_km=float(target_apogee_radius_km),
    )

    selected_phase_angle_deg = phase_angle_deg
    selected_speed_km_s = speed_km_s
    selected_x0 = None if state_vector_km_kmps is None else np.asarray(state_vector_km_kmps, dtype=float).reshape(6)
    best = None

    if selected_x0 is None:
        if selected_phase_angle_deg is None or selected_speed_km_s is None:
            best = _search_initial_state(
                env_base=env_base,
                ctx=ctx,
                leo_altitude_km=float(leo_altitude_km),
                target_apogee_radius_km=float(target_apogee_radius_km),
                search_dt_s=float(search_dt_s),
                search_duration_days=float(duration_days),
                target_duration_days=float(target_duration_days),
                moon_distance_goal_km=float(moon_distance_goal_km),
                min_lunar_encounter_km=float(min_lunar_encounter_km),
                outbound_floor_km=float(outbound_floor_km),
                window=window,
            )
            selected_phase_angle_deg = float(best.phase_angle_deg)
            selected_speed_km_s = float(best.speed_km_s)
            selected_x0 = best.state_eci_km_s.copy()
        else:
            selected_x0 = _parameterized_state_vector_eci_km_s(
                env_base=env_base,
                leo_altitude_km=float(leo_altitude_km),
                target_apogee_radius_km=float(target_apogee_radius_km),
                phase_angle_deg=float(selected_phase_angle_deg),
                speed_km_s=float(selected_speed_km_s),
            )

    traj = _propagate_trajectory(
        x0_eci_km_s=selected_x0,
        env_base=env_base,
        ctx=ctx,
        dt_s=float(final_dt_s),
        duration_days=float(duration_days),
        stop_on_earth_impact=True,
    )

    if best is None and selected_phase_angle_deg is not None and selected_speed_km_s is not None:
        score = _score_candidate(
            traj=traj,
            target_duration_days=float(target_duration_days),
            moon_distance_goal_km=float(moon_distance_goal_km),
            min_lunar_encounter_km=float(min_lunar_encounter_km),
            outbound_floor_km=float(outbound_floor_km),
        )
        best = SearchCandidate(
            score=float(score),
            phase_angle_deg=float(selected_phase_angle_deg),
            speed_km_s=float(selected_speed_km_s),
            state_eci_km_s=selected_x0.copy(),
            min_moon_distance_km=float(traj.min_moon_distance_km),
            time_to_moon_days=float(traj.time_to_moon_days),
            max_earth_range_km=float(traj.max_earth_range_km),
            post_flyby_perigee_radius_km=float(traj.post_flyby_perigee_radius_km),
            post_flyby_perigee_altitude_km=float(traj.post_flyby_perigee_altitude_km),
            earth_impact=bool(traj.earth_impact),
            return_time_days=None if traj.return_time_days is None else float(traj.return_time_days),
            duration_days=float(traj.duration_days),
        )

    outdir = REPO_ROOT / "outputs" / "orbit_artemis2_style_free_return"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)
    orbit_path, metrics_path = _plot_static_summary(traj=traj, outdir=outdir, plot_mode=plot_mode)
    anim_path = _animate_trajectory(traj=traj, outdir=outdir, plot_mode=plot_mode)

    return {
        "epoch_utc": epoch.isoformat(),
        "plot_mode": plot_mode,
        "final_dt_s": f"{float(final_dt_s):.1f}",
        "search_dt_s": f"{float(search_dt_s):.1f}",
        "phase_angle_deg": "" if selected_phase_angle_deg is None else f"{float(selected_phase_angle_deg):.6f}",
        "speed_km_s": "" if selected_speed_km_s is None else f"{float(selected_speed_km_s):.9f}",
        "base_free_return_speed_km_s": f"{float(base_speed):.9f}",
        "initial_state_eci_km_kmps": np.array2string(selected_x0, precision=9, separator=", "),
        "score": "" if best is None else f"{float(best.score):.6f}",
        "min_moon_distance_km": f"{float(traj.min_moon_distance_km):.3f}",
        "time_to_moon_days": f"{float(traj.time_to_moon_days):.3f}",
        "max_earth_range_km": f"{float(traj.max_earth_range_km):.3f}",
        "post_flyby_perigee_altitude_km": f"{float(traj.post_flyby_perigee_altitude_km):.3f}",
        "earth_impact": str(bool(traj.earth_impact)),
        "return_time_days": "" if traj.return_time_days is None else f"{float(traj.return_time_days):.3f}",
        "orbit_plot": str(orbit_path) if plot_mode in ("save", "both") else "",
        "metrics_plot": str(metrics_path) if plot_mode in ("save", "both") else "",
        "animation_path": str(anim_path) if plot_mode in ("save", "both") and anim_path.exists() else "",
    }


def _parse_epoch_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Search and animate an Artemis II-style cislunar free-return approximation using "
            "Earth two-body gravity plus atmospheric drag and lunar third-body perturbations only."
        )
    )
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both", "none"], default="interactive")
    parser.add_argument("--epoch-utc", type=_parse_epoch_utc, default=datetime(2026, 3, 11, 0, 0, 0, tzinfo=timezone.utc))
    parser.add_argument("--duration-days", type=float, default=14.0, help="Propagation horizon in days.")
    parser.add_argument("--final-dt", type=float, default=1800.0, help="Final propagation step in seconds.")
    parser.add_argument("--search-dt", type=float, default=1800.0, help="Search propagation step in seconds.")
    parser.add_argument("--leo-altitude-km", type=float, default=185.0, help="Initial post-burn parking altitude in km.")
    parser.add_argument("--target-apogee-radius-km", type=float, default=380000.0, help="Analytic apogee target used to seed the search.")
    parser.add_argument("--target-duration-days", type=float, default=10.0, help="Search preference for total mission duration.")
    parser.add_argument("--moon-distance-goal-km", type=float, default=28000.0, help="Preferred closest-approach distance to the Moon.")
    parser.add_argument("--min-lunar-encounter-km", type=float, default=80000.0, help="Hard lunar-encounter gate for search acceptance.")
    parser.add_argument("--outbound-floor-km", type=float, default=250000.0, help="Minimum Earth range that counts as a real translunar outbound leg.")
    parser.add_argument("--phase-angle-deg", type=float, default=None, help="Skip search and use this parameterized phase angle.")
    parser.add_argument("--speed-km-s", type=float, default=None, help="Skip search and use this parameterized inertial speed.")
    parser.add_argument(
        "--state-vector",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "VX", "VY", "VZ"),
        help="Skip the parameterization entirely and propagate this exact ECI state vector [km, km/s].",
    )
    parser.add_argument("--search-angle-center-deg", type=float, default=6.0)
    parser.add_argument("--search-angle-span-deg", type=float, default=1.0)
    parser.add_argument("--search-speed-offset-center-km-s", type=float, default=0.010)
    parser.add_argument("--search-speed-offset-span-km-s", type=float, default=0.004)
    parser.add_argument("--search-points-per-axis", type=int, default=5)
    parser.add_argument("--search-rounds", type=int, default=1)
    parser.add_argument("--search-shrink", type=float, default=0.35)
    args = parser.parse_args()

    result = run_demo(
        plot_mode=args.plot_mode,
        epoch_utc=args.epoch_utc,
        duration_days=float(args.duration_days),
        final_dt_s=float(args.final_dt),
        search_dt_s=float(args.search_dt),
        leo_altitude_km=float(args.leo_altitude_km),
        target_apogee_radius_km=float(args.target_apogee_radius_km),
        target_duration_days=float(args.target_duration_days),
        moon_distance_goal_km=float(args.moon_distance_goal_km),
        min_lunar_encounter_km=float(args.min_lunar_encounter_km),
        outbound_floor_km=float(args.outbound_floor_km),
        phase_angle_deg=None if args.phase_angle_deg is None else float(args.phase_angle_deg),
        speed_km_s=None if args.speed_km_s is None else float(args.speed_km_s),
        state_vector_km_kmps=None if args.state_vector is None else np.array(args.state_vector, dtype=float),
        search_window=SearchWindow(
            angle_center_deg=float(args.search_angle_center_deg),
            angle_span_deg=float(args.search_angle_span_deg),
            speed_offset_center_km_s=float(args.search_speed_offset_center_km_s),
            speed_offset_span_km_s=float(args.search_speed_offset_span_km_s),
            points_per_axis=int(args.search_points_per_axis),
            rounds=int(args.search_rounds),
            shrink=float(args.search_shrink),
        ),
    )
    print("Artemis II-style free-return outputs:")
    for key, value in result.items():
        if value:
            print(f"  {key}: {value}")
